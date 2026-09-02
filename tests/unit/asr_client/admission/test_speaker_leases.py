from __future__ import annotations

import pytest

from main_logic.asr_client._provider_events import ProviderUtteranceKey
from main_logic.asr_client.admission.contracts import (
    AdmissionDisposition,
    AdmissionState,
    BoundaryUnknown,
    BoundaryExact,
    PendingProviderFinal,
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


async def test_attach_is_atomic_idempotent_ordered_and_rejects_terminal_lease():
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
        SpeakerLeaseHigh(_candidate(), 1),
    )
    with pytest.raises(SpeakerLeaseTerminalError, match="LEASE_TERMINAL"):
        await coordinator.attach_turn_to_speaker_lease(_turn(2), lease, _key(2))


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
