from __future__ import annotations

from main_logic.asr_client._provider_events import ProviderUtteranceKey
from main_logic.asr_client.admission.contracts import (
    AbortProviderTransport,
    AdmissionDisposition,
    AdmissionResolutionTicket,
    AdmissionState,
    ApplyRejection,
    BoundaryExact,
    BoundaryState,
    BoundaryUnknown,
    CandidateBindingState,
    CapabilityRevokeFailed,
    CapabilityRevoked,
    CaptureClosed,
    CaptureState,
    CoreSettled,
    ConstrainRejectionDeadline,
    EvidenceState,
    FinalDeadlineExpired,
    LifecycleSettled,
    MicroEventPending,
    MicroEventSuppressed,
    MicroEventUnavailable,
    PendingProviderFinal,
    PoisonSpeakerAuthorityNamespace,
    ProviderBindingState,
    ProviderFinalReceived,
    RejectionApplied,
    RejectionApplyState,
    RejectionCapability,
    RejectionCapabilityKind,
    Reset,
    ResolveReserved,
    RevokeRejectionCapability,
    ScheduleFinalDeadline,
    SettlementState,
    SpeakerCheckpointKind,
    SpeakerLow,
    SpeakerUnavailable,
    SpeakerAuthorityNamespacePoisoned,
    TransportSettled,
    VoiceTurnAdmissionRecord,
)
from main_logic.asr_client.admission.reducer import reduce
from main_logic.asr_client.speaker_shadow.contracts import SpeakerShadowCandidateKey
from main_logic.voice_turn.contracts import VoiceIngressToken, VoiceTurnToken


def _token(turn_id: int = 1) -> VoiceTurnToken:
    return VoiceTurnToken(
        ingress=VoiceIngressToken(1, "socket", 2, 3, 4),
        turn_id=turn_id,
    )


def _provider_key(utterance_id: int = 1) -> ProviderUtteranceKey:
    return ProviderUtteranceKey(1, 0, utterance_id)


def _candidate(generation: int = 1) -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(5, generation, "provider_candidate")


def _record(*, capability_kind: RejectionCapabilityKind = RejectionCapabilityKind.SEALED):
    token = _token()
    key = _provider_key()
    candidate = _candidate()
    capability = RejectionCapability(
        capability_id=7,
        owner_generation=5,
        kind=capability_kind,
        turn_token=token,
        candidate=candidate,
        provider_key=key,
    )
    record = VoiceTurnAdmissionRecord(
        turn_token=token,
        record_generation=1,
        provider_binding_state=ProviderBindingState.BOUND,
        candidate_binding_state=CandidateBindingState.BOUND,
        capture_state=CaptureState.COLLECTING,
        provider_key=key,
        speaker_candidate=candidate,
    )
    return record, capability


def _final(*, text: str = "hello", deadline: float = 10.2) -> PendingProviderFinal:
    return PendingProviderFinal(
        provider_key=_provider_key(),
        provider="qwen",
        text=text,
        received_at=10.0,
        admission_deadline=deadline,
    )


def _step(record, event, now: float = 10.0):
    return reduce(record, event, now)


def _resolve_effects(effects):
    return [effect for effect in effects if isinstance(effect, ResolveReserved)]


def _apply_effect(effects) -> ApplyRejection:
    return next(effect for effect in effects if isinstance(effect, ApplyRejection))


def _deadline_effect(effects) -> ScheduleFinalDeadline:
    return next(effect for effect in effects if isinstance(effect, ScheduleFinalDeadline))


def test_final_then_second_low_and_exact_before_deadline_drops():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.PENDING
    assert isinstance(_deadline_effect(effects), ScheduleFinalDeadline)

    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
        now=10.05,
    )
    record, effects = _step(record, BoundaryExact(capability), now=10.06)
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
        now=10.07,
    )

    assert record.admission_state is AdmissionState.DROPPED
    assert [effect.disposition for effect in _resolve_effects(effects)] == [
        AdmissionDisposition.DROP
    ]


def test_exact_final_then_second_low_before_deadline_drops():
    record, capability = _record()
    record, _ = _step(record, BoundaryExact(capability))
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(record, ProviderFinalReceived(_final()))
    record, effects = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
        now=10.1,
    )
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
        now=10.11,
    )
    assert record.admission_state is AdmissionState.DROPPED
    assert len(_resolve_effects(effects)) == 1


def test_final_without_low_forwards_and_ignores_late_second_low():
    record, _ = _record()
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD

    record, later = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
        now=10.05,
    )
    record, latest = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
        now=10.06,
    )
    assert record.admission_state is AdmissionState.FORWARDED
    assert not _resolve_effects((*later, *latest))


def test_first_low_closed_without_second_low_forwards():
    record, _ = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(record, CaptureClosed(_candidate(), 1))
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_capture_complete_cannot_clear_reject_requested():
    record, _ = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, _ = _step(record, CaptureClosed(_candidate(), 2))
    assert record.capture_state is CaptureState.CLOSED
    assert record.evidence_state is EvidenceState.REJECT_REQUESTED


def test_deadline_forward_ignores_late_applied():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(record, ProviderFinalReceived(_final()))
    deadline = _deadline_effect(effects)
    record, effects = _step(
        record,
        FinalDeadlineExpired(deadline.ticket, deadline.absolute_deadline),
        now=deadline.absolute_deadline,
    )
    assert record.admission_state is AdmissionState.FORWARDED
    assert len(_resolve_effects(effects)) == 1

    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
        now=10.3,
    )
    assert record.admission_state is AdmissionState.FORWARDED
    assert not _resolve_effects(effects)


def test_reset_abandons_and_ignores_late_operation():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(record, Reset())
    assert record.admission_state is AdmissionState.ABANDONED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.ABANDON

    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    assert record.admission_state is AdmissionState.ABANDONED
    assert not _resolve_effects(effects)


def test_micro_event_and_speaker_veto_combine_once():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    assert record.admission_state is AdmissionState.RESERVED
    assert not _resolve_effects(effects)
    record, effects = _step(record, ProviderFinalReceived(_final()))
    first_resolutions = _resolve_effects(effects)
    record, effects = _step(record, MicroEventSuppressed())
    assert len(first_resolutions) == 1
    assert not _resolve_effects(effects)


def test_unrelated_revision_does_not_stale_operation_ticket():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    revision = record.logical_revision
    record, _ = _step(record, MicroEventPending())
    assert record.logical_revision > revision
    record, _ = _step(record, ProviderFinalReceived(_final()))
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    assert record.admission_state is AdmissionState.DROPPED
    assert len(_resolve_effects(effects)) == 1


def test_terminal_forward_ignores_late_boundary_speaker_and_micro_facts():
    record, capability = _record()
    record, _ = _step(record, ProviderFinalReceived(_final()))
    terminal = record

    record, boundary_effects = _step(record, BoundaryExact(capability), now=10.1)
    assert record.admission_state is terminal.admission_state
    assert record.pending_final == terminal.pending_final
    assert record.rejection_capability is None
    assert any(
        isinstance(effect, RevokeRejectionCapability)
        and effect.capability == capability
        for effect in boundary_effects
    )
    post_boundary = record

    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
        now=10.11,
    )
    record, _ = _step(record, MicroEventSuppressed(), now=10.12)
    assert record is post_boundary
    assert record.evidence_state is EvidenceState.NONE
    assert record.micro_event_state.value == "not_applicable"


def test_sealed_applied_then_same_key_unknown_before_final_fails_open():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    assert record.admission_state is AdmissionState.RESERVED
    assert record.rejection_apply_state is RejectionApplyState.APPLIED_SEALED
    assert not _resolve_effects(effects)

    record, effects = _step(record, BoundaryUnknown(_provider_key()))
    assert record.boundary_state is BoundaryState.UNKNOWN
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert record.rejection_capability is None
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)

    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_active_applied_remains_dropped_after_provider_unknown():
    record, capability = _record(capability_kind=RejectionCapabilityKind.ACTIVE)
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, _ = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.ACTIVE),
    )
    terminal = record
    record, effects = _step(record, BoundaryUnknown(_provider_key()))
    assert record is terminal
    assert record.admission_state is AdmissionState.DROPPED
    assert record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE
    assert not any(isinstance(effect, RevokeRejectionCapability) for effect in effects)


def test_pending_rejection_apply_waits_until_absolute_deadline():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    assert isinstance(_apply_effect(effects), ApplyRejection)
    record, effects = _step(record, ProviderFinalReceived(_final()), now=10.1)
    assert record.admission_state is AdmissionState.PENDING
    assert not _resolve_effects(effects)
    assert _deadline_effect(effects).absolute_deadline == 10.2


def test_empty_final_is_not_dropped_by_micro_event_suppress():
    record, _ = _record()
    record, _ = _step(record, MicroEventSuppressed())
    record, effects = _step(record, ProviderFinalReceived(_final(text="")))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_active_applied_drops_before_provider_final_and_late_final_only_settles():
    record, capability = _record(capability_kind=RejectionCapabilityKind.ACTIVE)
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.ACTIVE),
    )
    assert record.admission_state is AdmissionState.DROPPED
    assert any(isinstance(effect, AbortProviderTransport) for effect in effects)
    assert len(_resolve_effects(effects)) == 1

    record, effects = _step(record, ProviderFinalReceived(_final()), now=10.1)
    assert record.provider_final_state.value == "received"
    assert record.admission_state is AdmissionState.DROPPED
    assert not _resolve_effects(effects)


def test_inflight_rejection_is_bound_to_one_capability_and_kind():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)

    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.ACTIVE),
    )
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert record.rejection_capability is None
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)
    assert not _resolve_effects(effects)

    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    conflicting = RejectionCapability(
        capability_id=capability.capability_id + 1,
        owner_generation=capability.owner_generation,
        kind=capability.kind,
        turn_token=capability.turn_token,
        candidate=capability.candidate,
        provider_key=capability.provider_key,
    )
    record, effects = _step(record, BoundaryExact(conflicting))
    assert record.boundary_state is BoundaryState.UNKNOWN
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert record.rejection_capability is None
    assert sum(isinstance(effect, RevokeRejectionCapability) for effect in effects) == 2

    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert not _resolve_effects(effects)


def test_foreign_capability_does_not_destroy_current_exact_authority():
    record, capability = _record()
    record, _ = _step(record, BoundaryExact(capability))
    foreign = RejectionCapability(
        capability_id=capability.capability_id + 1,
        owner_generation=capability.owner_generation,
        kind=capability.kind,
        turn_token=capability.turn_token,
        candidate=_candidate(2),
        provider_key=capability.provider_key,
    )
    record, effects = _step(record, BoundaryExact(foreign))
    assert record.boundary_state is BoundaryState.EXACT
    assert record.rejection_capability == capability
    assert [
        effect.capability
        for effect in effects
        if isinstance(effect, RevokeRejectionCapability)
    ] == [foreign]


def test_unknown_boundary_absorbs_late_exact_capability():
    record, capability = _record()
    record, _ = _step(record, BoundaryUnknown(_provider_key()))
    record, effects = _step(record, BoundaryExact(capability))
    assert record.boundary_state is BoundaryState.UNKNOWN
    assert record.rejection_capability is None
    assert any(
        isinstance(effect, RevokeRejectionCapability)
        and effect.capability == capability
        for effect in effects
    )


def test_capture_close_sequence_gap_fails_open_and_late_low_is_ignored():
    record, _ = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(record, CaptureClosed(_candidate(), 2))
    assert record.capture_state is CaptureState.UNAVAILABLE
    assert record.evidence_state is EvidenceState.UNAVAILABLE

    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_speaker_delivery_failure_revokes_sticky_reject_request():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    assert isinstance(_apply_effect(effects), ApplyRejection)

    record, effects = _step(record, SpeakerUnavailable(_candidate(), 3))
    assert record.evidence_state is EvidenceState.UNAVAILABLE
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)

    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_absolute_deadline_wins_over_late_rejection_and_micro_suppression():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, _ = _step(record, ProviderFinalReceived(_final()))
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
        now=10.21,
    )
    assert record.admission_state is AdmissionState.FORWARDED
    assert record.rejection_apply_state is RejectionApplyState.STALE
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)

    record, _ = _record()
    record, _ = _step(record, MicroEventPending())
    record, _ = _step(record, ProviderFinalReceived(_final()))
    record, effects = _step(record, MicroEventSuppressed(), now=10.21)
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_empty_final_does_not_wait_for_reject_request():
    record, _ = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, ProviderFinalReceived(_final(text="")))
    assert record.admission_state is AdmissionState.FORWARDED
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD


def test_terminal_reset_revokes_active_capability_without_changing_disposition():
    record, capability = _record(capability_kind=RejectionCapabilityKind.ACTIVE)
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, _ = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.ACTIVE),
    )
    generation = record.record_generation

    record, effects = _step(record, Reset())
    assert record.admission_state is AdmissionState.DROPPED
    assert record.record_generation == generation + 1
    assert record.rejection_capability is None
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)


def test_settlement_requires_matching_resolution_nonce_and_disposition():
    record, _ = _record()
    record, effects = _step(record, ProviderFinalReceived(_final()))
    resolution = _resolve_effects(effects)[0]
    wrong = AdmissionResolutionTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        resolution_nonce=resolution.ticket.resolution_nonce + 1,
        disposition=AdmissionDisposition.FORWARD,
    )
    record, _ = _step(record, CoreSettled(wrong))
    assert record.core_settlement_state is SettlementState.NOT_STARTED

    wrong_disposition = AdmissionResolutionTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        resolution_nonce=resolution.ticket.resolution_nonce,
        disposition=AdmissionDisposition.DROP,
    )
    record, _ = _step(record, CoreSettled(wrong_disposition))
    assert record.core_settlement_state is SettlementState.NOT_STARTED

    record, _ = _step(record, CoreSettled(resolution.ticket))
    assert record.core_settlement_state is SettlementState.SETTLED


def test_revoked_inflight_apply_is_revoked_again_if_it_succeeds_late():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert any(
        isinstance(effect, ConstrainRejectionDeadline)
        and effect.ticket == apply.ticket
        and effect.absolute_deadline == 10.2
        for effect in effects
    )

    deadline = _deadline_effect(effects)
    record, effects = _step(
        record,
        FinalDeadlineExpired(deadline.ticket, deadline.absolute_deadline),
        now=10.2,
    )
    assert record.admission_state is AdmissionState.FORWARDED
    assert record.revoked_rejection_ticket == apply.ticket
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)

    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
        now=10.21,
    )
    assert record.revoked_rejection_ticket is None
    assert any(
        isinstance(effect, RevokeRejectionCapability)
        and effect.capability == capability
        for effect in effects
    )


def test_missing_provider_key_cannot_consume_exact_sealed_authority():
    record, capability = _record()
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, _ = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.SEALED),
    )
    missing_key_final = PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
    record, effects = _step(record, ProviderFinalReceived(missing_key_final))
    assert record.admission_state is AdmissionState.FORWARDED
    assert record.boundary_state is BoundaryState.UNKNOWN
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD
    assert any(isinstance(effect, RevokeRejectionCapability) for effect in effects)


def test_active_drop_releases_authority_after_transport_and_lifecycle_settle():
    record, capability = _record(capability_kind=RejectionCapabilityKind.ACTIVE)
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    record, _ = _step(
        record,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    record, effects = _step(record, BoundaryExact(capability))
    apply = _apply_effect(effects)
    record, effects = _step(
        record,
        RejectionApplied(apply.ticket, RejectionCapabilityKind.ACTIVE),
    )
    resolution = _resolve_effects(effects)[0]
    assert record.rejection_capability == capability

    record, effects = _step(record, TransportSettled(resolution.ticket))
    assert record.rejection_capability == capability
    assert not any(isinstance(effect, RevokeRejectionCapability) for effect in effects)
    record, effects = _step(record, LifecycleSettled(resolution.ticket))
    assert record.rejection_capability is None
    assert any(
        isinstance(effect, RevokeRejectionCapability)
        and effect.capability == capability
        for effect in effects
    )


def test_capability_revoke_requires_ack_and_failure_keeps_cleanup_handle():
    record, capability = _record()
    record, effects = _step(record, BoundaryExact(capability))
    record, effects = _step(record, BoundaryUnknown(_provider_key()))
    revoke = next(
        effect for effect in effects if isinstance(effect, RevokeRejectionCapability)
    )
    assert revoke.ticket is not None
    assert len(record.pending_revocations) == 1

    record, effects = _step(record, CapabilityRevokeFailed(revoke.ticket))
    assert len(record.pending_revocations) == 1
    assert record.pending_revocations[0].degraded is True
    assert record.revocation_degraded is True
    poison = next(
        effect
        for effect in effects
        if isinstance(effect, PoisonSpeakerAuthorityNamespace)
    )
    assert poison.ticket is not None

    record, _ = _step(record, CapabilityRevoked(revoke.ticket))
    assert record.pending_revocations == ()
    assert record.revocation_degraded is True
    record, _ = _step(record, SpeakerAuthorityNamespacePoisoned(poison.ticket))
    assert record.pending_revocations == ()
    assert record.revocation_degraded is False
    assert record.namespace_poison_ticket is None


def test_micro_terminal_results_are_monotonic_and_conflicts_fail_open():
    record, _ = _record()
    record, _ = _step(record, MicroEventPending())
    record, _ = _step(record, MicroEventUnavailable())
    record, _ = _step(record, MicroEventSuppressed())
    assert record.micro_event_state.value == "unavailable"
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD

    record, _ = _record()
    record, _ = _step(record, MicroEventPending())
    record, _ = _step(record, MicroEventSuppressed())
    record, _ = _step(record, MicroEventUnavailable())
    assert record.micro_event_state.value == "unavailable"
    record, effects = _step(record, ProviderFinalReceived(_final()))
    assert _resolve_effects(effects)[0].disposition is AdmissionDisposition.FORWARD
