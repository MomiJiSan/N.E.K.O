from __future__ import annotations

from dataclasses import replace

import pytest

from main_logic.asr_client._provider_events import ProviderUtteranceKey
from main_logic.asr_client.admission.contracts import (
    AdmissionDisposition,
    AdmissionState,
    BoundaryUnknown,
    BoundaryExact,
    CandidateBindingState,
    CaptureState,
    CountDiagnostic,
    EvidenceState,
    PendingProviderFinal,
    ProviderBindingState,
    ProviderFinalReceived,
    ResolveReserved,
    RejectionCapability,
    RejectionCapabilityKind,
    RouteReplaced,
    SpeakerCaptureLeaseRecord,
    SpeakerCaptureLeaseToken,
    SpeakerCheckpointKind,
    SpeakerLeaseAbandoned,
    SpeakerLeaseCaptureClosed,
    SpeakerLeaseHigh,
    SpeakerLeaseLow,
    SpeakerLeaseState,
    SpeakerLeaseUnavailable,
    SpeakerAuthorityPending,
)
from main_logic.asr_client.admission.coordinator import (
    AdmissionIdentityError,
    SpeakerLeaseCapacityError,
    VoiceTurnAdmissionCoordinator,
)
from main_logic.asr_client.admission.ingress import (
    AdmissionIngressLane,
)
from main_logic.asr_client.admission.speaker_leases import (
    SpeakerLeaseChildCapacityError,
    SpeakerLeaseTerminalError,
    reduce_speaker_lease,
)
from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
)
from main_logic.voice_turn.contracts import VoiceIngressToken, VoiceTurnToken


pytestmark = pytest.mark.asyncio


def _lease(nonce: int = 1) -> SpeakerCaptureLeaseToken:
    return SpeakerCaptureLeaseToken(1, 2, 3, 4, nonce)


def _candidate(generation: int = 1) -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(4, generation, "provider_candidate")


def _key(utterance_id: int) -> ProviderUtteranceKey:
    return ProviderUtteranceKey(3, 0, utterance_id)


def _turn(turn_id: int) -> VoiceTurnToken:
    return VoiceTurnToken(VoiceIngressToken(1, "socket", 2, 3, 4), turn_id)


def _final(key: ProviderUtteranceKey, text: str) -> PendingProviderFinal:
    return PendingProviderFinal(key, "qwen", text, 10.0, 10.2)


async def test_pure_lease_reducer_requires_ordered_first_then_second_low():
    record = SpeakerCaptureLeaseRecord(_lease(), 1, _candidate())

    record, _ = reduce_speaker_lease(
        record,
        SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    assert record.state is SpeakerLeaseState.FIRST_LOW

    record, _ = reduce_speaker_lease(
        record,
        SpeakerLeaseLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    assert record.state is SpeakerLeaseState.DENY_LATCHED

    for late in (
        SpeakerLeaseUnavailable(_candidate(), 3),
        SpeakerLeaseCaptureClosed(_candidate(), 2),
        SpeakerLeaseHigh(_candidate(), 3),
        SpeakerLeaseAbandoned(),
    ):
        unchanged, _ = reduce_speaker_lease(record, late)
        assert unchanged is record
        assert unchanged.state is SpeakerLeaseState.DENY_LATCHED


async def test_second_low_without_first_fails_open_and_capture_close_is_terminal():
    record = SpeakerCaptureLeaseRecord(_lease(), 1, _candidate())
    unavailable, _ = reduce_speaker_lease(
        record,
        SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.SECOND),
    )
    assert unavailable.state is SpeakerLeaseState.UNAVAILABLE

    closed, _ = reduce_speaker_lease(
        record,
        SpeakerLeaseCaptureClosed(_candidate(), 0),
    )
    assert closed.state is SpeakerLeaseState.UNAVAILABLE
    assert closed.capture_through_sequence_no == 0


async def test_two_provider_children_share_one_sticky_deny_and_fan_out_in_order():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    # VoiceTurn ids intentionally oppose Provider order; fan-out follows keys.
    first, second = _turn(2), _turn(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(first, lease, _key(1))
    await coordinator.attach_turn_to_speaker_lease(second, lease, _key(2))
    await coordinator.post(first, ProviderFinalReceived(_final(_key(1), "a")))
    await coordinator.post(second, ProviderFinalReceived(_final(_key(2), "b")))

    assert (
        await coordinator.post_speaker_lease(
            lease,
            SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
        )
        == ()
    )
    assert (
        await coordinator.get_record(first)
    ).admission_state is AdmissionState.PENDING
    assert (
        await coordinator.get_record(second)
    ).admission_state is AdmissionState.PENDING

    results = await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    assert tuple(result.turn_token for result in results) == (first, second)
    assert all(
        any(
            isinstance(effect, ResolveReserved)
            and effect.disposition is AdmissionDisposition.DROP
            for effect in result.effects
        )
        for result in results
    )
    assert (
        await coordinator.get_record(first)
    ).admission_state is AdmissionState.DROPPED
    assert (
        await coordinator.get_record(second)
    ).admission_state is AdmissionState.DROPPED

    await coordinator.post(first, BoundaryUnknown(_key(1)))
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseUnavailable(_candidate(), 3),
    )
    lease_record = await coordinator.get_speaker_lease(lease)
    assert lease_record is not None
    assert lease_record.state is SpeakerLeaseState.DENY_LATCHED


@pytest.mark.parametrize(
    ("fact", "state", "disposition"),
    (
        (
            SpeakerLeaseHigh(_candidate(), 1),
            SpeakerLeaseState.ALLOW,
            AdmissionDisposition.FORWARD,
        ),
        (
            SpeakerLeaseUnavailable(_candidate(), 1),
            SpeakerLeaseState.UNAVAILABLE,
            AdmissionDisposition.FORWARD,
        ),
        (
            SpeakerLeaseAbandoned(),
            SpeakerLeaseState.ABANDONED,
            AdmissionDisposition.ABANDON,
        ),
    ),
)
async def test_terminal_parent_verdict_resolves_pending_child(
    fact,
    state: SpeakerLeaseState,
    disposition: AdmissionDisposition,
):
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    child = await coordinator.attach_turn_to_speaker_lease(turn, lease, _key(1))
    assert child.speaker_lease_token == lease
    assert child.provider_key == _key(1)
    await coordinator.post(turn, ProviderFinalReceived(_final(_key(1), "hello")))

    results = await coordinator.post_speaker_lease(lease, fact)
    assert len(results) == 1
    resolution = next(
        effect for effect in results[0].effects if isinstance(effect, ResolveReserved)
    )
    assert resolution.disposition is disposition
    assert (await coordinator.get_speaker_lease(lease)).state is state


async def test_attach_is_atomic_idempotent_ordered_and_rejects_abandoned_lease():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    turn = _turn(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    first = await coordinator.attach_turn_to_speaker_lease(turn, lease, _key(1))
    duplicate = await coordinator.attach_turn_to_speaker_lease(turn, lease, _key(1))
    assert duplicate is first

    with pytest.raises(AdmissionIdentityError, match="ALIAS_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(turn, lease, _key(2))

    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseAbandoned(),
    )
    with pytest.raises(SpeakerLeaseTerminalError, match="LEASE_TERMINAL"):
        await coordinator.attach_turn_to_speaker_lease(_turn(2), lease, _key(2))


@pytest.mark.parametrize(
    ("parent_events", "evidence", "capture", "disposition"),
    (
        (
            (SpeakerLeaseHigh(_candidate(), 1),),
            EvidenceState.ALLOW,
            CaptureState.COLLECTING,
            AdmissionDisposition.FORWARD,
        ),
        (
            (SpeakerLeaseUnavailable(_candidate(), 1),),
            EvidenceState.UNAVAILABLE,
            CaptureState.UNAVAILABLE,
            AdmissionDisposition.FORWARD,
        ),
        (
            (
                SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
                SpeakerLeaseLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
            ),
            EvidenceState.DENY_LATCHED,
            CaptureState.COLLECTING,
            AdmissionDisposition.DROP,
        ),
    ),
)
async def test_late_child_inherits_terminal_parent_and_resolves_final(
    parent_events,
    evidence: EvidenceState,
    capture: CaptureState,
    disposition: AdmissionDisposition,
):
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    first_turn = _turn(1)
    late_turn = _turn(2)
    first_key = _key(1)
    late_key = _key(2)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(first_turn, lease, first_key)
    for event in parent_events:
        await coordinator.post_speaker_lease(lease, event)

    late = await coordinator.attach_turn_to_speaker_lease(
        late_turn,
        lease,
        late_key,
    )

    assert late.evidence_state is evidence
    assert late.capture_state is capture
    assert late.provider_binding_state is ProviderBindingState.BOUND
    assert late.candidate_binding_state is CandidateBindingState.BOUND
    assert late.provider_key == late_key
    assert late.speaker_lease_token == lease
    assert late.speaker_candidate == _candidate()
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert tuple(binding.provider_key for binding in parent.child_bindings) == (
        first_key,
        late_key,
    )
    assert (
        await coordinator.attach_turn_to_speaker_lease(late_turn, lease, late_key)
        is late
    )

    effects = await coordinator.post(
        late_turn,
        ProviderFinalReceived(_final(late_key, "late")),
    )
    resolution = next(
        effect for effect in effects if isinstance(effect, ResolveReserved)
    )
    assert resolution.disposition is disposition
    assert (
        sum(
            isinstance(effect, CountDiagnostic)
            and effect.name == "speaker_deny_latched_count"
            for effect in effects
        )
        == 0
    )


async def test_terminal_parent_late_child_preserves_provider_order():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(_turn(1), lease, _key(2))
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseHigh(_candidate(), 1),
    )

    with pytest.raises(AdmissionIdentityError, match="PROVIDER_ORDER_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(_turn(2), lease, _key(1))

    assert await coordinator.get_record(_turn(2)) is None
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert tuple(binding.provider_key for binding in parent.child_bindings) == (
        _key(2),
    )


async def test_terminal_parent_late_child_preserves_capacity():
    coordinator = VoiceTurnAdmissionCoordinator(speaker_lease_child_capacity=1)
    lease = _lease()
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(_turn(1), lease, _key(1))
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseUnavailable(_candidate(), 1),
    )

    with pytest.raises(SpeakerLeaseChildCapacityError, match="CHILD_CAPACITY"):
        await coordinator.attach_turn_to_speaker_lease(_turn(2), lease, _key(2))

    assert await coordinator.get_record(_turn(2)) is None


async def test_terminal_parent_late_child_preserves_provider_key_uniqueness():
    coordinator = VoiceTurnAdmissionCoordinator()
    first_lease = _lease(1)
    terminal_lease = _lease(2)
    provider_key = _key(2)
    await coordinator.open_speaker_lease(first_lease, _candidate(1))
    await coordinator.open_speaker_lease(terminal_lease, _candidate(2))
    await coordinator.attach_turn_to_speaker_lease(
        _turn(1),
        first_lease,
        provider_key,
    )
    await coordinator.post_speaker_lease(
        terminal_lease,
        SpeakerLeaseHigh(_candidate(2), 1),
    )

    with pytest.raises(AdmissionIdentityError, match="KEY_ALREADY_BOUND"):
        await coordinator.attach_turn_to_speaker_lease(
            _turn(2),
            terminal_lease,
            provider_key,
        )

    assert await coordinator.get_record(_turn(2)) is None


async def test_terminal_parent_upgrades_exact_empty_placeholder():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseHigh(_candidate(), 1),
    )
    opened = await coordinator.open_turn(turn)
    await coordinator.post(turn, SpeakerAuthorityPending("provider-arming"))

    upgraded = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )

    assert upgraded.record_generation == opened.record_generation
    assert upgraded.provider_binding_state is ProviderBindingState.BOUND
    assert upgraded.candidate_binding_state is CandidateBindingState.BOUND
    assert upgraded.evidence_state is EvidenceState.ALLOW
    assert upgraded.speaker_authority_generation == "provider-arming"
    effects = await coordinator.post(
        turn,
        ProviderFinalReceived(_final(provider_key, "late")),
    )
    resolution = next(
        effect for effect in effects if isinstance(effect, ResolveReserved)
    )
    assert resolution.disposition is AdmissionDisposition.FORWARD


async def test_terminal_parent_rejects_placeholder_with_early_final():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseHigh(_candidate(), 1),
    )
    await coordinator.open_turn(turn)
    await coordinator.post(
        turn,
        ProviderFinalReceived(_final(provider_key, "early")),
    )
    pending = await coordinator.get_record(turn)
    assert pending is not None

    with pytest.raises(AdmissionIdentityError, match="TERMINAL_BINDING_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(
            turn,
            lease,
            provider_key,
        )

    assert await coordinator.get_record(turn) is pending
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == ()
    assert provider_key not in coordinator._provider_speaker_lease_bindings


async def test_attach_upgrades_exact_placeholder_and_preserves_pending_state():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    pending_final = _final(provider_key, "held")
    await coordinator.open_speaker_lease(lease, _candidate())
    opened = await coordinator.open_turn(turn)
    await coordinator.post(turn, SpeakerAuthorityPending("provider-arming"))
    await coordinator.post(turn, ProviderFinalReceived(pending_final))
    placeholder = await coordinator.get_record(turn)
    assert placeholder is not None
    assert placeholder.record_generation == opened.record_generation
    assert placeholder.admission_state is AdmissionState.PENDING
    assert placeholder.pending_final is pending_final
    assert placeholder.candidate_binding_state is CandidateBindingState.ARMING

    upgraded = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )

    assert upgraded.record_generation == placeholder.record_generation
    assert upgraded.logical_revision == placeholder.logical_revision + 1
    assert upgraded.admission_state is AdmissionState.PENDING
    assert upgraded.pending_final is pending_final
    assert upgraded.speaker_authority_generation == "provider-arming"
    assert upgraded.provider_binding_state is ProviderBindingState.BOUND
    assert upgraded.candidate_binding_state is CandidateBindingState.BOUND
    assert upgraded.capture_state is CaptureState.COLLECTING
    assert upgraded.provider_key == provider_key
    assert upgraded.speaker_lease_token == lease
    assert upgraded.speaker_candidate == _candidate()
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert tuple(
        (binding.provider_key, binding.turn_token) for binding in parent.child_bindings
    ) == ((provider_key, turn),)

    duplicate = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    assert duplicate is upgraded


async def test_attach_rejects_terminal_child_without_partial_parent_binding():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.open_turn(turn, provider_key=provider_key)
    await coordinator.post(turn, ProviderFinalReceived(_final(provider_key, "done")))
    terminal = await coordinator.get_record(turn)
    assert terminal is not None
    assert terminal.admission_state is AdmissionState.FORWARDED

    with pytest.raises(AdmissionIdentityError, match="TERMINAL_BINDING_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(turn, lease, provider_key)

    assert await coordinator.get_record(turn) is terminal
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == ()
    assert provider_key not in coordinator._provider_speaker_lease_bindings


async def test_attach_candidate_conflict_does_not_partially_commit_binding():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate(1))
    placeholder = await coordinator.open_turn(
        turn,
        provider_key=provider_key,
        speaker_candidate=_candidate(2),
    )

    with pytest.raises(AdmissionIdentityError, match="ALIAS_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(turn, lease, provider_key)

    assert await coordinator.get_record(turn) is placeholder
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == ()
    assert provider_key not in coordinator._provider_speaker_lease_bindings


@pytest.mark.parametrize(
    "malformed_changes",
    (
        {"provider_binding_state": ProviderBindingState.BOUND},
        {
            "provider_binding_state": ProviderBindingState.UNBOUND,
            "provider_key": _key(1),
        },
        {"candidate_binding_state": CandidateBindingState.BOUND},
        {
            "candidate_binding_state": CandidateBindingState.BOUND,
            "capture_state": CaptureState.COLLECTING,
            "speaker_candidate": _candidate(),
        },
        {"candidate_binding_state": CandidateBindingState.ARMING},
        {
            "candidate_binding_state": CandidateBindingState.UNBOUND,
            "speaker_lease_token": _lease(),
        },
        {
            "provider_binding_state": ProviderBindingState.BOUND,
            "provider_key": _key(1),
            "candidate_binding_state": CandidateBindingState.BOUND,
            "capture_state": CaptureState.COLLECTING,
            "speaker_candidate": _candidate(),
            "speaker_lease_token": _lease(),
        },
    ),
)
async def test_attach_rejects_malformed_state_field_combinations(
    malformed_changes,
):
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    placeholder = await coordinator.open_turn(turn)
    malformed = replace(placeholder, **malformed_changes)
    coordinator._records[turn] = malformed

    with pytest.raises(AdmissionIdentityError, match="ALIAS_CONFLICT"):
        await coordinator.attach_turn_to_speaker_lease(turn, lease, provider_key)

    assert await coordinator.get_record(turn) is malformed
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == ()
    assert provider_key not in coordinator._provider_speaker_lease_bindings


async def test_placeholder_upgrade_is_atomic_when_child_capacity_is_exhausted():
    coordinator = VoiceTurnAdmissionCoordinator(
        capacity=3,
        speaker_lease_child_capacity=1,
    )
    lease = _lease(1)
    first_turn = _turn(1)
    placeholder_turn = _turn(2)
    await coordinator.open_speaker_lease(lease, _candidate(1))
    await coordinator.attach_turn_to_speaker_lease(first_turn, lease, _key(1))
    placeholder = await coordinator.open_turn(placeholder_turn)

    with pytest.raises(SpeakerLeaseChildCapacityError, match="CHILD_CAPACITY"):
        await coordinator.attach_turn_to_speaker_lease(
            placeholder_turn,
            lease,
            _key(2),
        )

    assert await coordinator.get_record(placeholder_turn) is placeholder
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert tuple(binding.turn_token for binding in parent.child_bindings) == (
        first_turn,
    )
    assert _key(2) not in coordinator._provider_speaker_lease_bindings


async def test_detach_exact_child_atomically_releases_capacity():
    coordinator = VoiceTurnAdmissionCoordinator(
        capacity=2,
        speaker_lease_child_capacity=1,
    )
    lease = _lease()
    first_turn = _turn(1)
    second_turn = _turn(2)
    first_key = _key(1)
    second_key = _key(2)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(first_turn, lease, first_key)

    assert await coordinator.detach_turn_from_speaker_lease(
        first_turn,
        lease,
        first_key,
    )
    assert await coordinator.get_record(first_turn) is None
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == ()
    assert coordinator._speaker_candidate_bindings[_candidate()] == lease
    assert first_key not in coordinator._provider_speaker_lease_bindings

    replacement = await coordinator.attach_turn_to_speaker_lease(
        second_turn,
        lease,
        second_key,
    )
    assert replacement.turn_token == second_turn


async def test_detach_rejects_identity_conflict_without_touching_replacement():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease(1)
    replacement_lease = _lease(2)
    turn = _turn(1)
    replacement_turn = _turn(2)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate(1))
    await coordinator.open_speaker_lease(replacement_lease, _candidate(2))
    child = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    replacement_binding = (replacement_lease, replacement_turn)
    coordinator._provider_speaker_lease_bindings[provider_key] = replacement_binding

    with pytest.raises(AdmissionIdentityError, match="DETACH_IDENTITY_CONFLICT"):
        await coordinator.detach_turn_from_speaker_lease(
            turn,
            lease,
            provider_key,
        )

    assert await coordinator.get_record(turn) is child
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings == (runtime_binding := parent.child_bindings[0],)
    assert runtime_binding.provider_key == provider_key
    assert runtime_binding.turn_token == turn
    assert coordinator._provider_speaker_lease_bindings[provider_key] == (
        replacement_binding
    )


@pytest.mark.parametrize("terminal_parent", (False, True))
async def test_detach_rejects_final_or_terminal_binding(terminal_parent: bool):
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    child = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    if terminal_parent:
        await coordinator.post_speaker_lease(
            lease,
            SpeakerLeaseHigh(_candidate(), 1),
        )
    else:
        await coordinator.post(
            turn,
            ProviderFinalReceived(_final(provider_key, "held")),
        )

    with pytest.raises(AdmissionIdentityError, match="DETACH_ALREADY_COMMITTED"):
        await coordinator.detach_turn_from_speaker_lease(
            turn,
            lease,
            provider_key,
        )

    current = await coordinator.get_record(turn)
    assert current is not None
    assert current is not child
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.child_bindings[0].turn_token == turn
    assert coordinator._provider_speaker_lease_bindings[provider_key] == (
        lease,
        turn,
    )


async def test_detach_missing_and_duplicate_are_idempotent_false():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)

    assert not await coordinator.detach_turn_from_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(turn, lease, provider_key)
    assert await coordinator.detach_turn_from_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    assert not await coordinator.detach_turn_from_speaker_lease(
        turn,
        lease,
        provider_key,
    )


@pytest.mark.parametrize(
    ("parent_events", "parent_state"),
    (
        (
            (SpeakerLeaseHigh(_candidate(), 1),),
            SpeakerLeaseState.ALLOW,
        ),
        (
            (SpeakerLeaseUnavailable(_candidate(), 1),),
            SpeakerLeaseState.UNAVAILABLE,
        ),
        (
            (
                SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
                SpeakerLeaseLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
            ),
            SpeakerLeaseState.DENY_LATCHED,
        ),
    ),
)
async def test_detach_exact_terminal_late_child_preserves_parent_and_siblings(
    parent_events,
    parent_state: SpeakerLeaseState,
):
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease()
    first_turn = _turn(1)
    late_turn = _turn(2)
    first_key = _key(1)
    late_key = _key(2)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(first_turn, lease, first_key)
    for event in parent_events:
        await coordinator.post_speaker_lease(lease, event)
    sibling = await coordinator.get_record(first_turn)
    late = await coordinator.attach_turn_to_speaker_lease(
        late_turn,
        lease,
        late_key,
    )
    parent_before = await coordinator.get_speaker_lease(lease)
    assert sibling is not None
    assert parent_before is not None
    assert late.terminal_disposition is None

    assert await coordinator.detach_turn_from_speaker_lease(
        late_turn,
        lease,
        late_key,
    )

    assert await coordinator.get_record(late_turn) is None
    assert await coordinator.get_record(first_turn) is sibling
    parent_after = await coordinator.get_speaker_lease(lease)
    assert parent_after is not None
    assert parent_after.state is parent_state
    assert parent_after.terminal_sequence_no == parent_before.terminal_sequence_no
    assert tuple(parent_after.child_bindings) == (parent_before.child_bindings[0],)
    assert parent_after.child_bindings[0].turn_token == first_turn
    assert coordinator._speaker_candidate_bindings[_candidate()] == lease
    assert late_key not in coordinator._provider_speaker_lease_bindings


@pytest.mark.parametrize("advance_child", ("final", "boundary"))
async def test_detach_terminal_late_child_rejects_committed_child_state(
    advance_child: str,
):
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseHigh(_candidate(), 1),
    )
    await coordinator.attach_turn_to_speaker_lease(turn, lease, provider_key)
    if advance_child == "final":
        await coordinator.post(
            turn,
            ProviderFinalReceived(_final(provider_key, "committed")),
        )
    else:
        await coordinator.post(turn, BoundaryUnknown(provider_key))
    committed = await coordinator.get_record(turn)

    with pytest.raises(AdmissionIdentityError, match="DETACH_ALREADY_COMMITTED"):
        await coordinator.detach_turn_from_speaker_lease(
            turn,
            lease,
            provider_key,
        )

    assert await coordinator.get_record(turn) is committed
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.state is SpeakerLeaseState.ALLOW
    assert parent.child_bindings[0].turn_token == turn
    assert coordinator._provider_speaker_lease_bindings[provider_key] == (
        lease,
        turn,
    )


async def test_detach_terminal_late_child_does_not_touch_replacement_mapping():
    coordinator = VoiceTurnAdmissionCoordinator()
    lease = _lease(1)
    replacement_lease = _lease(2)
    turn = _turn(1)
    replacement_turn = _turn(2)
    provider_key = _key(1)
    await coordinator.open_speaker_lease(lease, _candidate(1))
    await coordinator.open_speaker_lease(replacement_lease, _candidate(2))
    await coordinator.post_speaker_lease(
        lease,
        SpeakerLeaseUnavailable(_candidate(1), 1),
    )
    child = await coordinator.attach_turn_to_speaker_lease(
        turn,
        lease,
        provider_key,
    )
    replacement = (replacement_lease, replacement_turn)
    coordinator._provider_speaker_lease_bindings[provider_key] = replacement

    with pytest.raises(AdmissionIdentityError, match="DETACH_IDENTITY_CONFLICT"):
        await coordinator.detach_turn_from_speaker_lease(
            turn,
            lease,
            provider_key,
        )

    assert await coordinator.get_record(turn) is child
    parent = await coordinator.get_speaker_lease(lease)
    assert parent is not None
    assert parent.state is SpeakerLeaseState.UNAVAILABLE
    assert parent.child_bindings[0].turn_token == turn
    assert coordinator._provider_speaker_lease_bindings[provider_key] == replacement


async def test_provider_key_cannot_be_attached_to_two_live_leases():
    coordinator = VoiceTurnAdmissionCoordinator()
    first_lease, second_lease = _lease(1), _lease(2)
    await coordinator.open_speaker_lease(first_lease, _candidate(1))
    await coordinator.open_speaker_lease(second_lease, _candidate(2))
    await coordinator.attach_turn_to_speaker_lease(
        _turn(1),
        first_lease,
        _key(1),
    )

    with pytest.raises(AdmissionIdentityError, match="KEY_ALREADY_BOUND"):
        await coordinator.attach_turn_to_speaker_lease(
            _turn(2),
            second_lease,
            _key(1),
        )


async def test_speaker_lease_and_child_capacities_are_strictly_bounded():
    coordinator = VoiceTurnAdmissionCoordinator(
        capacity=16,
        speaker_lease_capacity=8,
        speaker_lease_child_capacity=8,
    )
    for nonce in range(1, 9):
        await coordinator.open_speaker_lease(_lease(nonce), _candidate(nonce))
    with pytest.raises(SpeakerLeaseCapacityError, match="CAPACITY_EXHAUSTED"):
        await coordinator.open_speaker_lease(_lease(9), _candidate(9))

    lease = _lease(1)
    for child in range(1, 9):
        await coordinator.attach_turn_to_speaker_lease(
            _turn(child),
            lease,
            _key(child),
        )
    with pytest.raises(SpeakerLeaseChildCapacityError, match="CHILD_CAPACITY"):
        await coordinator.attach_turn_to_speaker_lease(_turn(9), lease, _key(9))
    assert len((await coordinator.get_speaker_lease(lease)).child_bindings) == 8


async def test_bulk_route_invalidation_abandons_parent_and_child_atomically():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lease = _lease()
    turn = _turn(1)
    await coordinator.open_speaker_lease(lease, _candidate())
    await coordinator.attach_turn_to_speaker_lease(turn, lease, _key(1))
    await coordinator.post(turn, ProviderFinalReceived(_final(_key(1), "hello")))

    results = await coordinator.invalidate_all(RouteReplaced())

    assert len(results) == 1
    assert (
        await coordinator.get_speaker_lease(lease)
    ).state is SpeakerLeaseState.ABANDONED
    assert (
        await coordinator.get_record(turn)
    ).admission_state is AdmissionState.ABANDONED


async def test_lease_facts_share_single_ingress_worker_and_reserved_capacity():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lane = AdmissionIngressLane(
        coordinator,
        data_capacity=1,
        control_capacity=4,
        speaker_control_capacity=2,
    )
    await lane.start()
    worker = lane._worker
    lease = _lease()
    turn = _turn(1)
    await lane.open_speaker_lease(lease, _candidate())
    await lane.attach_turn_to_speaker_lease(turn, lease, _key(1))

    await coordinator._lock.acquire()
    try:
        boundary = lane.post_nowait(
            turn,
            BoundaryExact(
                RejectionCapability(
                    capability_id=1,
                    owner_generation=1,
                    kind=RejectionCapabilityKind.SEALED,
                    turn_token=turn,
                    candidate=_candidate(),
                    provider_key=_key(1),
                )
            ),
        )
        first = lane.post_speaker_lease_nowait(
            lease,
            SpeakerLeaseLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
        )
        second = lane.post_speaker_lease_nowait(
            lease,
            SpeakerLeaseLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
        )
        assert lane.pending_data_count == 1
        assert lane.pending_speaker_control_count == 2
        assert lane._worker is worker
    finally:
        coordinator._lock.release()

    await boundary
    assert await first == ()
    results = await second
    assert tuple(result.turn_token for result in results) == (turn,)
    assert (
        await coordinator.get_record(turn)
    ).admission_state is AdmissionState.DROPPED
    assert lane._worker is worker
    await lane.close()


async def test_terminal_empty_lease_retires_through_same_ingress():
    coordinator = VoiceTurnAdmissionCoordinator()
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    lease = _lease()
    await lane.open_speaker_lease(lease, _candidate())
    await lane.post_speaker_lease(
        lease,
        SpeakerLeaseHigh(_candidate(), 1),
    )
    assert await lane.retire_speaker_lease(lease) is True
    assert await coordinator.get_speaker_lease(lease) is None
    await lane.close()
