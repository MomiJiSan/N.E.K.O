"""Pure state reduction for one logical voice turn."""

from __future__ import annotations

from dataclasses import replace

from .contracts import (
    AbortProviderTransport,
    AdmissionDisposition,
    AdmissionEffect,
    AdmissionEvent,
    AdmissionOperationKind,
    AdmissionOperationTicket,
    AdmissionResolutionTicket,
    AdmissionState,
    ApplyRejection,
    BoundaryExact,
    BoundaryState,
    BoundaryUnknown,
    CandidateBindingState,
    CandidateBound,
    CapabilityRevokeFailed,
    CapabilityRevoked,
    CaptureClosed,
    CaptureState,
    Close,
    ConstrainRejectionDeadline,
    CoreSettled,
    CountDiagnostic,
    EvidenceState,
    FinalDeadlineExpired,
    LifecycleSettled,
    MicroEventAllowed,
    MicroEventPending,
    MicroEventState,
    MicroEventSuppressed,
    MicroEventUnavailable,
    PendingProviderFinal,
    PendingCapabilityRevocation,
    PoisonSpeakerAuthorityNamespace,
    ProviderBindingState,
    ProviderBound,
    ProviderFinalReceived,
    ProviderFinalState,
    RejectionApplied,
    RejectionApplyState,
    RejectionCapabilityKind,
    RejectionFailed,
    RejectionStale,
    Reset,
    ResolveReserved,
    RevokeRejectionCapability,
    RouteReplaced,
    ScheduleFinalDeadline,
    SettlementState,
    SpeakerCheckpointKind,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
    SpeakerAuthorityNamespacePoisoned,
    SpeakerAuthorityNamespacePoisonFailed,
    TransportSettled,
    TurnOpened,
    TurnSealed,
    VoiceTurnAdmissionRecord,
)


_APPLIED_STATES = {
    RejectionApplyState.APPLIED_ACTIVE,
    RejectionApplyState.APPLIED_SEALED,
}
_TERMINAL_ADMISSION_STATES = {
    AdmissionState.FORWARDED,
    AdmissionState.DROPPED,
    AdmissionState.ABANDONED,
}
_MAX_PENDING_REVOCATIONS = 8


def _changed(
    record: VoiceTurnAdmissionRecord,
    **changes: object,
) -> VoiceTurnAdmissionRecord:
    if all(getattr(record, name) == value for name, value in changes.items()):
        return record
    return replace(
        record,
        logical_revision=record.logical_revision + 1,
        **changes,
    )


def _ticket_matches(
    record: VoiceTurnAdmissionRecord,
    ticket: AdmissionOperationTicket,
    *,
    kind: AdmissionOperationKind,
    nonce: int | None,
) -> bool:
    return bool(
        ticket.turn_token == record.turn_token
        and ticket.record_generation == record.record_generation
        and ticket.operation_kind is kind
        and nonce is not None
        and ticket.operation_nonce == nonce
    )


def _speaker_fact_is_current(
    record: VoiceTurnAdmissionRecord,
    candidate: object,
    sequence_no: object,
) -> bool:
    return bool(
        record.candidate_binding_state is CandidateBindingState.BOUND
        and record.capture_state is CaptureState.COLLECTING
        and candidate == record.speaker_candidate
        and type(sequence_no) is int
        and sequence_no == record.last_speaker_sequence_no + 1
    )


def _capability_matches_record(
    record: VoiceTurnAdmissionRecord,
    capability: object,
) -> bool:
    return bool(
        getattr(capability, "turn_token", None) == record.turn_token
        and (
            record.speaker_candidate is None
            or getattr(capability, "candidate", None) == record.speaker_candidate
        )
        and (
            record.provider_key is None
            or getattr(capability, "provider_key", None) in {None, record.provider_key}
        )
    )


def _current_rejection_ticket(
    record: VoiceTurnAdmissionRecord,
) -> AdmissionOperationTicket | None:
    if (
        record.rejection_apply_state is not RejectionApplyState.IN_FLIGHT
        or record.rejection_operation_nonce is None
        or record.rejection_operation_capability_id is None
        or record.rejection_operation_owner_generation is None
        or record.rejection_operation_kind is None
    ):
        return None
    return AdmissionOperationTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        operation_kind=AdmissionOperationKind.APPLY_REJECTION,
        operation_nonce=record.rejection_operation_nonce,
        capability_id=record.rejection_operation_capability_id,
        capability_owner_generation=record.rejection_operation_owner_generation,
        capability_kind=record.rejection_operation_kind,
    )


def _revoked_inflight_changes(
    record: VoiceTurnAdmissionRecord,
) -> dict[str, object]:
    ticket = _current_rejection_ticket(record)
    capability = record.rejection_capability
    if ticket is None or capability is None:
        return {}
    return {
        "revoked_rejection_ticket": ticket,
        "revoked_rejection_capability": capability,
    }


def _start_rejection_if_ready(
    record: VoiceTurnAdmissionRecord,
    *,
    now: float,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    capability = record.rejection_capability
    if (
        record.admission_state in _TERMINAL_ADMISSION_STATES
        or record.evidence_state is not EvidenceState.REJECT_REQUESTED
        or record.rejection_apply_state is not RejectionApplyState.NOT_STARTED
        or capability is None
        or record.boundary_state is not BoundaryState.EXACT
    ):
        return record, ()
    final = record.pending_final
    if final is not None and now >= final.admission_deadline:
        return record, ()
    nonce = record.operation_nonce_sequence + 1
    ticket = AdmissionOperationTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        operation_kind=AdmissionOperationKind.APPLY_REJECTION,
        operation_nonce=nonce,
        capability_id=capability.capability_id,
        capability_owner_generation=capability.owner_generation,
        capability_kind=capability.kind,
    )
    record = _changed(
        record,
        operation_nonce_sequence=nonce,
        rejection_operation_nonce=nonce,
        rejection_operation_capability_id=capability.capability_id,
        rejection_operation_owner_generation=capability.owner_generation,
        rejection_operation_kind=capability.kind,
        rejection_apply_state=RejectionApplyState.IN_FLIGHT,
    )
    return record, (
        ApplyRejection(
            ticket=ticket,
            capability=capability,
            absolute_deadline=(final.admission_deadline if final is not None else None),
        ),
    )


def _resolve(
    record: VoiceTurnAdmissionRecord,
    disposition: AdmissionDisposition,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    if record.admission_state in _TERMINAL_ADMISSION_STATES:
        return record, ()
    next_state = {
        AdmissionDisposition.FORWARD: AdmissionState.FORWARDED,
        AdmissionDisposition.DROP: AdmissionState.DROPPED,
        AdmissionDisposition.ABANDON: AdmissionState.ABANDONED,
    }[disposition]
    nonce = record.operation_nonce_sequence + 1
    resolution_ticket = AdmissionResolutionTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        resolution_nonce=nonce,
        disposition=disposition,
    )
    effects: list[AdmissionEffect] = [
        CountDiagnostic(f"admission_terminal_{disposition.value}"),
        ResolveReserved(
            ticket=resolution_ticket,
            final=record.pending_final,
        )
    ]
    applied = record.rejection_apply_state in _APPLIED_STATES
    applied_drop = disposition is AdmissionDisposition.DROP and applied
    keep_applied_authority = (
        disposition is AdmissionDisposition.DROP
        and record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE
    )
    capability = record.rejection_capability
    if capability is not None and not keep_applied_authority:
        effects.append(RevokeRejectionCapability(capability))
    if (
        disposition is AdmissionDisposition.DROP
        and record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE
    ):
        effects.append(AbortProviderTransport(record.turn_token))
    apply_state = record.rejection_apply_state
    revoked_inflight = _revoked_inflight_changes(record)
    if not applied_drop and apply_state in {
        RejectionApplyState.NOT_STARTED,
        RejectionApplyState.IN_FLIGHT,
        RejectionApplyState.APPLIED_ACTIVE,
        RejectionApplyState.APPLIED_SEALED,
    }:
        apply_state = RejectionApplyState.STALE
    final_state = record.provider_final_state
    if (
        disposition is AdmissionDisposition.DROP
        and record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE
        and final_state is ProviderFinalState.NOT_RECEIVED
    ):
        final_state = ProviderFinalState.ABORTED
    record = _changed(
        record,
        operation_nonce_sequence=nonce,
        admission_state=next_state,
        rejection_apply_state=apply_state,
        rejection_operation_nonce=None,
        rejection_operation_capability_id=None,
        rejection_operation_owner_generation=None,
        rejection_operation_kind=None,
        deadline_operation_nonce=None,
        rejection_capability=(capability if keep_applied_authority else None),
        provider_final_state=final_state,
        resolution_ticket=resolution_ticket,
        **revoked_inflight,
    )
    return record, tuple(effects)


def _release_active_authority_if_settled(
    record: VoiceTurnAdmissionRecord,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    capability = record.rejection_capability
    settled = {SettlementState.SETTLED, SettlementState.DEGRADED}
    if (
        record.admission_state is not AdmissionState.DROPPED
        or record.rejection_apply_state is not RejectionApplyState.APPLIED_ACTIVE
        or capability is None
        or record.transport_settlement_state not in settled
        or record.lifecycle_settlement_state not in settled
    ):
        return record, ()
    return _changed(record, rejection_capability=None), (
        RevokeRejectionCapability(capability),
    )


def _count_terminal_micro_event_if_settled(
    record: VoiceTurnAdmissionRecord,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    """Count only a terminal micro-event whose turn settlement stayed valid."""

    if record.micro_event_terminal_counted or record.terminal_disposition is None:
        return record, ()
    settlements = (
        record.transport_settlement_state,
        record.lifecycle_settlement_state,
    )
    terminal = {SettlementState.SETTLED, SettlementState.DEGRADED}
    if any(state not in terminal for state in settlements):
        return record, ()
    record = _changed(record, micro_event_terminal_counted=True)
    if any(state is SettlementState.DEGRADED for state in settlements):
        return record, ()
    if (
        record.terminal_disposition is AdmissionDisposition.DROP
        and record.micro_event_state is MicroEventState.SUPPRESS
    ):
        return record, (CountDiagnostic("micro_event_suppressed_count"),)
    if (
        record.terminal_disposition is AdmissionDisposition.FORWARD
        and record.micro_event_shadow_would_suppress
    ):
        return record, (CountDiagnostic("micro_event_shadow_forward_count"),)
    return record, ()


def _rejection_can_still_be_confirmed(record: VoiceTurnAdmissionRecord) -> bool:
    if record.boundary_state not in {BoundaryState.OPEN, BoundaryState.EXACT}:
        return False
    if record.rejection_apply_state in {
        RejectionApplyState.STALE,
        RejectionApplyState.FAILED,
    }:
        return False
    if record.evidence_state is EvidenceState.REJECT_REQUESTED:
        return True
    return bool(
        record.evidence_state is EvidenceState.FIRST_LOW
        and record.capture_state is CaptureState.COLLECTING
    )


def maybe_resolve(
    record: VoiceTurnAdmissionRecord,
    now: float,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    """Resolve one reservation if the accumulated facts make it terminal."""

    if record.admission_state in _TERMINAL_ADMISSION_STATES:
        return record, ()
    final = record.pending_final
    if final is not None and now >= final.admission_deadline:
        resolved, effects = _resolve(record, AdmissionDisposition.FORWARD)
        return resolved, (CountDiagnostic("admission_deadline_forward"), *effects)
    if record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE:
        return _resolve(record, AdmissionDisposition.DROP)
    if final is not None and not final.text.strip():
        return _resolve(record, AdmissionDisposition.FORWARD)
    if (
        final is not None
        and record.rejection_apply_state is RejectionApplyState.APPLIED_SEALED
    ):
        return _resolve(record, AdmissionDisposition.DROP)
    if (
        final is not None
        and final.text.strip()
        and record.micro_event_state is MicroEventState.SUPPRESS
    ):
        return _resolve(record, AdmissionDisposition.DROP)
    if final is None:
        return record, ()
    if _rejection_can_still_be_confirmed(record):
        return record, ()
    if record.micro_event_state is MicroEventState.PENDING:
        return record, ()
    return _resolve(record, AdmissionDisposition.FORWARD)


def _schedule_deadline_if_needed(
    record: VoiceTurnAdmissionRecord,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    final = record.pending_final
    if (
        record.admission_state is not AdmissionState.PENDING
        or final is None
        or record.deadline_operation_nonce is not None
    ):
        return record, ()
    nonce = record.operation_nonce_sequence + 1
    ticket = AdmissionOperationTicket(
        turn_token=record.turn_token,
        record_generation=record.record_generation,
        operation_kind=AdmissionOperationKind.FINAL_DEADLINE,
        operation_nonce=nonce,
    )
    record = _changed(
        record,
        operation_nonce_sequence=nonce,
        deadline_operation_nonce=nonce,
    )
    return record, (
        ScheduleFinalDeadline(
            ticket=ticket,
            absolute_deadline=final.admission_deadline,
        ),
    )


def _invalidate(
    record: VoiceTurnAdmissionRecord,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    capability = record.rejection_capability
    revoked_inflight = _revoked_inflight_changes(record)
    record = _changed(
        record,
        record_generation=record.record_generation + 1,
        provider_binding_state=ProviderBindingState.RETIRED,
        candidate_binding_state=CandidateBindingState.RETIRED,
        boundary_state=BoundaryState.RETIRED,
        rejection_apply_state=(
            record.rejection_apply_state
            if record.rejection_apply_state in _APPLIED_STATES
            else RejectionApplyState.STALE
        ),
        rejection_capability=None,
        rejection_operation_nonce=None,
        rejection_operation_capability_id=None,
        rejection_operation_owner_generation=None,
        rejection_operation_kind=None,
        deadline_operation_nonce=None,
        **revoked_inflight,
    )
    record, resolve_effects = _resolve(record, AdmissionDisposition.ABANDON)
    if capability is None or any(
        isinstance(effect, RevokeRejectionCapability) for effect in resolve_effects
    ):
        return record, resolve_effects
    return record, (RevokeRejectionCapability(capability), *resolve_effects)


def _reduce_untracked(
    record: VoiceTurnAdmissionRecord,
    event: AdmissionEvent,
    now: float,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    """Apply one immutable event and return effects for execution outside locks."""

    effects: list[AdmissionEffect] = []

    if isinstance(event, (Reset, Close, RouteReplaced)):
        return _invalidate(record)

    if (
        record.admission_state in _TERMINAL_ADMISSION_STATES
        and not isinstance(
            event,
            (
                ProviderFinalReceived,
                RejectionApplied,
                RejectionStale,
                RejectionFailed,
                CapabilityRevoked,
                CapabilityRevokeFailed,
                SpeakerAuthorityNamespacePoisoned,
                SpeakerAuthorityNamespacePoisonFailed,
                CoreSettled,
                TransportSettled,
                LifecycleSettled,
            ),
        )
    ):
        if isinstance(event, BoundaryExact):
            return record, (
                RevokeRejectionCapability(event.capability),
                CountDiagnostic("admission_late_boundary_ignored"),
            )
        return record, (CountDiagnostic("admission_late_fact_ignored"),)

    if isinstance(event, TurnOpened):
        if event.turn_token != record.turn_token:
            return record, (CountDiagnostic("admission_stale_turn_opened"),)
    elif isinstance(event, ProviderBound):
        if record.provider_key is None:
            record = _changed(
                record,
                provider_binding_state=ProviderBindingState.BOUND,
                provider_key=event.provider_key,
            )
        elif record.provider_key != event.provider_key:
            return record, (CountDiagnostic("admission_provider_alias_conflict"),)
    elif isinstance(event, CandidateBound):
        if record.speaker_candidate is None:
            record = _changed(
                record,
                candidate_binding_state=CandidateBindingState.BOUND,
                capture_state=CaptureState.COLLECTING,
                speaker_candidate=event.candidate,
            )
        elif record.speaker_candidate != event.candidate:
            return record, (CountDiagnostic("admission_candidate_alias_conflict"),)
    elif isinstance(event, SpeakerLow):
        if not _speaker_fact_is_current(record, event.candidate, event.sequence_no):
            return record, (CountDiagnostic("admission_stale_speaker_fact"),)
        evidence = record.evidence_state
        if evidence is not EvidenceState.REJECT_REQUESTED:
            if (
                event.checkpoint_kind is SpeakerCheckpointKind.FIRST
                and evidence is EvidenceState.NONE
            ):
                evidence = EvidenceState.FIRST_LOW
            elif (
                event.checkpoint_kind
                in {
                    SpeakerCheckpointKind.SECOND,
                    SpeakerCheckpointKind.COMPLETION_CONFIRMATION,
                }
                and evidence is EvidenceState.FIRST_LOW
            ):
                evidence = EvidenceState.REJECT_REQUESTED
            else:
                evidence = EvidenceState.UNAVAILABLE
        record = _changed(
            record,
            last_speaker_sequence_no=event.sequence_no,
            evidence_state=evidence,
        )
    elif isinstance(event, SpeakerHigh):
        if not _speaker_fact_is_current(record, event.candidate, event.sequence_no):
            return record, (CountDiagnostic("admission_stale_speaker_fact"),)
        record = _changed(
            record,
            last_speaker_sequence_no=event.sequence_no,
            evidence_state=(
                EvidenceState.REJECT_REQUESTED
                if record.evidence_state is EvidenceState.REJECT_REQUESTED
                else EvidenceState.ALLOW
            ),
        )
    elif isinstance(event, SpeakerUnavailable):
        if not _speaker_fact_is_current(record, event.candidate, event.sequence_no):
            return record, (CountDiagnostic("admission_stale_speaker_fact"),)
        capability = record.rejection_capability
        if capability is not None:
            effects.append(RevokeRejectionCapability(capability))
        revoked_inflight = _revoked_inflight_changes(record)
        record = _changed(
            record,
            last_speaker_sequence_no=event.sequence_no,
            capture_state=CaptureState.UNAVAILABLE,
            evidence_state=EvidenceState.UNAVAILABLE,
            rejection_apply_state=RejectionApplyState.STALE,
            rejection_capability=None,
            rejection_operation_nonce=None,
            rejection_operation_capability_id=None,
            rejection_operation_owner_generation=None,
            rejection_operation_kind=None,
            **revoked_inflight,
        )
    elif isinstance(event, CaptureClosed):
        if (
            event.candidate != record.speaker_candidate
            or type(event.through_sequence_no) is not int
            or event.through_sequence_no < record.last_speaker_sequence_no
        ):
            return record, (CountDiagnostic("admission_stale_capture_close"),)
        if event.through_sequence_no > record.last_speaker_sequence_no:
            capability = record.rejection_capability
            if capability is not None:
                effects.append(RevokeRejectionCapability(capability))
            revoked_inflight = _revoked_inflight_changes(record)
            record = _changed(
                record,
                capture_state=CaptureState.UNAVAILABLE,
                capture_through_sequence_no=event.through_sequence_no,
                evidence_state=EvidenceState.UNAVAILABLE,
                rejection_apply_state=RejectionApplyState.STALE,
                rejection_capability=None,
                rejection_operation_nonce=None,
                rejection_operation_capability_id=None,
                rejection_operation_owner_generation=None,
                rejection_operation_kind=None,
                **revoked_inflight,
            )
            effects.append(CountDiagnostic("admission_speaker_sequence_gap"))
        else:
            record = _changed(
                record,
                capture_state=CaptureState.CLOSED,
                capture_through_sequence_no=event.through_sequence_no,
            )
    elif isinstance(event, BoundaryExact):
        capability = event.capability
        if record.boundary_state in {BoundaryState.UNKNOWN, BoundaryState.RETIRED}:
            effects.extend(
                (
                    RevokeRejectionCapability(capability),
                    CountDiagnostic("admission_late_exact_after_unknown"),
                )
            )
        elif not _capability_matches_record(record, capability):
            effects.extend(
                (
                    RevokeRejectionCapability(capability),
                    CountDiagnostic("admission_foreign_capability_ignored"),
                )
            )
        elif (
            record.rejection_capability is not None
            and record.rejection_capability != capability
        ):
            revoked_inflight = _revoked_inflight_changes(record)
            effects.extend(
                (
                    RevokeRejectionCapability(record.rejection_capability),
                    RevokeRejectionCapability(capability),
                    CountDiagnostic("admission_capability_conflict"),
                )
            )
            record = _changed(
                record,
                boundary_state=BoundaryState.UNKNOWN,
                rejection_apply_state=RejectionApplyState.STALE,
                rejection_capability=None,
                rejection_operation_nonce=None,
                rejection_operation_capability_id=None,
                rejection_operation_owner_generation=None,
                rejection_operation_kind=None,
                **revoked_inflight,
            )
        else:
            changes: dict[str, object] = {
                "boundary_state": BoundaryState.EXACT,
                "rejection_capability": capability,
            }
            if record.speaker_candidate is None:
                changes.update(
                    candidate_binding_state=CandidateBindingState.BOUND,
                    capture_state=CaptureState.COLLECTING,
                    speaker_candidate=capability.candidate,
                )
            if record.provider_key is None and capability.provider_key is not None:
                changes.update(
                    provider_binding_state=ProviderBindingState.BOUND,
                    provider_key=capability.provider_key,
                )
            record = _changed(record, **changes)
    elif isinstance(event, BoundaryUnknown):
        if event.provider_key is not None and record.provider_key not in {
            None,
            event.provider_key,
        }:
            return record, (CountDiagnostic("admission_stale_boundary_unknown"),)
        capability = record.rejection_capability
        provider_authority_applied = (
            record.rejection_apply_state is RejectionApplyState.APPLIED_SEALED
        )
        if capability is not None and (
            provider_authority_applied
            or record.rejection_apply_state
            in {
                RejectionApplyState.NOT_STARTED,
                RejectionApplyState.IN_FLIGHT,
            }
        ):
            effects.append(RevokeRejectionCapability(capability))
        revoked_inflight = _revoked_inflight_changes(record)
        record = _changed(
            record,
            boundary_state=BoundaryState.UNKNOWN,
            rejection_capability=(
                capability
                if record.rejection_apply_state is RejectionApplyState.APPLIED_ACTIVE
                else None
            ),
            rejection_apply_state=(
                RejectionApplyState.STALE
                if record.rejection_apply_state
                in {
                    RejectionApplyState.NOT_STARTED,
                    RejectionApplyState.IN_FLIGHT,
                    RejectionApplyState.APPLIED_SEALED,
                }
                else record.rejection_apply_state
            ),
            rejection_operation_nonce=None,
            rejection_operation_capability_id=None,
            rejection_operation_owner_generation=None,
            rejection_operation_kind=None,
            **revoked_inflight,
        )
    elif isinstance(event, TurnSealed):
        if event.capability is not None:
            return reduce(record, BoundaryExact(event.capability), now)
    elif isinstance(event, RejectionApplied):
        if event.ticket == record.revoked_rejection_ticket:
            capability = record.revoked_rejection_capability
            record = _changed(
                record,
                revoked_rejection_ticket=None,
                revoked_rejection_capability=None,
            )
            return record, (
                *((RevokeRejectionCapability(capability),) if capability else ()),
                CountDiagnostic("admission_revoked_operation_applied_late"),
            )
        if not _ticket_matches(
            record,
            event.ticket,
            kind=AdmissionOperationKind.APPLY_REJECTION,
            nonce=record.rejection_operation_nonce,
        ) or (
            event.ticket.capability_id != record.rejection_operation_capability_id
            or event.ticket.capability_owner_generation
            != record.rejection_operation_owner_generation
            or event.ticket.capability_kind is not record.rejection_operation_kind
        ):
            return record, (CountDiagnostic("admission_late_operation_ignored"),)
        if event.kind is not record.rejection_operation_kind:
            capability = record.rejection_capability
            record = _changed(
                record,
                rejection_apply_state=RejectionApplyState.STALE,
                rejection_capability=None,
                rejection_operation_nonce=None,
                rejection_operation_capability_id=None,
                rejection_operation_owner_generation=None,
                rejection_operation_kind=None,
            )
            if capability is not None:
                effects.append(RevokeRejectionCapability(capability))
            effects.append(CountDiagnostic("admission_rejection_kind_mismatch"))
        else:
            record = _changed(
                record,
                rejection_apply_state=(
                    RejectionApplyState.APPLIED_ACTIVE
                    if event.kind is RejectionCapabilityKind.ACTIVE
                    else RejectionApplyState.APPLIED_SEALED
                ),
                rejection_operation_nonce=None,
                rejection_operation_capability_id=None,
                rejection_operation_owner_generation=None,
                rejection_operation_kind=None,
            )
            effects.append(
                CountDiagnostic(
                    "admission_rejection_applied_active"
                    if event.kind is RejectionCapabilityKind.ACTIVE
                    else "admission_rejection_applied_sealed"
                )
            )
    elif isinstance(event, (RejectionStale, RejectionFailed)):
        if event.ticket == record.revoked_rejection_ticket:
            record = _changed(
                record,
                revoked_rejection_ticket=None,
                revoked_rejection_capability=None,
            )
            return record, (CountDiagnostic("admission_revoked_operation_settled"),)
        if not _ticket_matches(
            record,
            event.ticket,
            kind=AdmissionOperationKind.APPLY_REJECTION,
            nonce=record.rejection_operation_nonce,
        ) or (
            event.ticket.capability_id != record.rejection_operation_capability_id
            or event.ticket.capability_owner_generation
            != record.rejection_operation_owner_generation
            or event.ticket.capability_kind is not record.rejection_operation_kind
        ):
            return record, (CountDiagnostic("admission_late_operation_ignored"),)
        capability = record.rejection_capability
        if capability is not None:
            effects.append(RevokeRejectionCapability(capability))
        record = _changed(
            record,
            rejection_apply_state=(
                RejectionApplyState.STALE
                if isinstance(event, RejectionStale)
                else RejectionApplyState.FAILED
            ),
            rejection_operation_nonce=None,
            rejection_operation_capability_id=None,
            rejection_operation_owner_generation=None,
            rejection_operation_kind=None,
            rejection_capability=None,
        )
    elif isinstance(event, CapabilityRevoked):
        remaining = tuple(
            operation
            for operation in record.pending_revocations
            if operation.ticket != event.ticket
        )
        if len(remaining) == len(record.pending_revocations):
            return record, (CountDiagnostic("admission_late_revoke_ack_ignored"),)
        record = _changed(record, pending_revocations=remaining)
    elif isinstance(event, CapabilityRevokeFailed):
        matched = False
        pending: list[PendingCapabilityRevocation] = []
        for operation in record.pending_revocations:
            if operation.ticket == event.ticket:
                matched = True
                pending.append(replace(operation, degraded=True))
            else:
                pending.append(operation)
        if not matched:
            return record, (CountDiagnostic("admission_late_revoke_failure_ignored"),)
        record = _changed(
            record,
            pending_revocations=tuple(pending),
            revocation_degraded=True,
        )
        effects.append(PoisonSpeakerAuthorityNamespace(record.turn_token))
    elif isinstance(event, SpeakerAuthorityNamespacePoisoned):
        if event.ticket != record.namespace_poison_ticket:
            return record, (CountDiagnostic("admission_late_namespace_poison_ack"),)
        record = _changed(
            record,
            pending_revocations=(),
            revocation_degraded=False,
            namespace_poison_ticket=None,
        )
    elif isinstance(event, SpeakerAuthorityNamespacePoisonFailed):
        if event.ticket != record.namespace_poison_ticket:
            return record, (CountDiagnostic("admission_late_namespace_poison_failure"),)
        record = _changed(record, revocation_degraded=True)
    elif isinstance(event, ProviderFinalReceived):
        if (
            record.admission_state is AdmissionState.ABANDONED
            or record.provider_binding_state is ProviderBindingState.RETIRED
        ):
            return record, (CountDiagnostic("admission_final_after_retirement_ignored"),)
        final = event.final
        if (
            record.admission_state not in _TERMINAL_ADMISSION_STATES
            and record.provider_key is not None
            and final.provider_key is None
            and record.boundary_state is not BoundaryState.UNKNOWN
        ):
            downgraded, downgrade_effects = reduce(
                record,
                BoundaryUnknown(record.provider_key),
                now,
            )
            accepted, final_effects = reduce(downgraded, event, now)
            return accepted, (*downgrade_effects, *final_effects)
        if final.provider_key is not None and record.provider_key not in {
            None,
            final.provider_key,
        }:
            return record, (CountDiagnostic("admission_stale_provider_final"),)
        if record.pending_final is not None:
            if record.pending_final == final:
                return record, ()
            return record, (CountDiagnostic("admission_conflicting_provider_final"),)
        changes = {
            "provider_final_state": ProviderFinalState.RECEIVED,
            "pending_final": final,
        }
        if record.admission_state is AdmissionState.RESERVED:
            changes["admission_state"] = AdmissionState.PENDING
        if record.provider_key is None and final.provider_key is not None:
            changes.update(
                provider_binding_state=ProviderBindingState.BOUND,
                provider_key=final.provider_key,
            )
        record = _changed(record, **changes)
        rejection_ticket = _current_rejection_ticket(record)
        if rejection_ticket is not None:
            effects.append(
                ConstrainRejectionDeadline(
                    ticket=rejection_ticket,
                    absolute_deadline=final.admission_deadline,
                )
            )
    elif isinstance(event, FinalDeadlineExpired):
        final = record.pending_final
        if (
            final is None
            or event.deadline != final.admission_deadline
            or not _ticket_matches(
                record,
                event.ticket,
                kind=AdmissionOperationKind.FINAL_DEADLINE,
                nonce=record.deadline_operation_nonce,
            )
        ):
            return record, (CountDiagnostic("admission_late_deadline_ignored"),)
        record = _changed(record, deadline_operation_nonce=None)
    elif isinstance(event, MicroEventPending):
        if record.micro_event_state is MicroEventState.NOT_APPLICABLE:
            record = _changed(record, micro_event_state=MicroEventState.PENDING)
    elif isinstance(event, MicroEventAllowed):
        accepted = record.micro_event_state in {
            MicroEventState.NOT_APPLICABLE,
            MicroEventState.PENDING,
        }
        record = _changed(
            record,
            micro_event_shadow_would_suppress=(
                event.shadow_would_suppress
                if accepted
                else (
                    (
                        record.micro_event_shadow_would_suppress
                        or event.shadow_would_suppress
                    )
                    if record.micro_event_state is MicroEventState.ALLOW
                    else False
                )
            ),
            micro_event_state=(
                MicroEventState.ALLOW
                if record.micro_event_state
                in {MicroEventState.NOT_APPLICABLE, MicroEventState.PENDING}
                else (
                    MicroEventState.ALLOW
                    if record.micro_event_state is MicroEventState.ALLOW
                    else MicroEventState.UNAVAILABLE
                )
            ),
        )
    elif isinstance(event, MicroEventSuppressed):
        record = _changed(
            record,
            micro_event_shadow_would_suppress=False,
            micro_event_state=(
                MicroEventState.SUPPRESS
                if record.micro_event_state
                in {MicroEventState.NOT_APPLICABLE, MicroEventState.PENDING}
                else (
                    MicroEventState.SUPPRESS
                    if record.micro_event_state is MicroEventState.SUPPRESS
                    else MicroEventState.UNAVAILABLE
                )
            ),
        )
    elif isinstance(event, MicroEventUnavailable):
        record = _changed(
            record,
            micro_event_state=MicroEventState.UNAVAILABLE,
            micro_event_shadow_would_suppress=False,
        )
    elif isinstance(event, CoreSettled):
        if event.ticket != record.resolution_ticket:
            return record, (CountDiagnostic("admission_stale_core_settlement"),)
        record = _changed(
            record,
            core_settlement_state=(
                SettlementState.DEGRADED if event.degraded else SettlementState.SETTLED
            ),
        )
        if event.degraded:
            effects.append(CountDiagnostic("admission_core_settlement_degraded"))
    elif isinstance(event, TransportSettled):
        if event.ticket != record.resolution_ticket:
            return record, (CountDiagnostic("admission_stale_transport_settlement"),)
        record = _changed(
            record,
            transport_settlement_state=(
                SettlementState.DEGRADED if event.degraded else SettlementState.SETTLED
            ),
        )
        if event.degraded:
            effects.append(
                CountDiagnostic("admission_transport_settlement_degraded")
            )
    elif isinstance(event, LifecycleSettled):
        if event.ticket != record.resolution_ticket:
            return record, (CountDiagnostic("admission_stale_lifecycle_settlement"),)
        record = _changed(
            record,
            lifecycle_settlement_state=(
                SettlementState.DEGRADED if event.degraded else SettlementState.SETTLED
            ),
        )
        if event.degraded:
            effects.append(
                CountDiagnostic("admission_lifecycle_settlement_degraded")
            )

    record, authority_effects = _release_active_authority_if_settled(record)
    effects.extend(authority_effects)
    record, micro_event_effects = _count_terminal_micro_event_if_settled(record)
    effects.extend(micro_event_effects)
    record, apply_effects = _start_rejection_if_ready(record, now=now)
    effects.extend(apply_effects)
    record, resolution_effects = maybe_resolve(record, now)
    effects.extend(resolution_effects)
    record, deadline_effects = _schedule_deadline_if_needed(record)
    effects.extend(deadline_effects)
    return record, tuple(effects)


def _track_revocation_effects(
    record: VoiceTurnAdmissionRecord,
    effects: tuple[AdmissionEffect, ...],
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    pending = list(record.pending_revocations)
    nonce = record.operation_nonce_sequence
    rewritten: list[AdmissionEffect] = []
    changed = False
    for effect in effects:
        if isinstance(effect, PoisonSpeakerAuthorityNamespace):
            ticket = record.namespace_poison_ticket
            if ticket is None:
                nonce += 1
                ticket = AdmissionOperationTicket(
                    turn_token=record.turn_token,
                    record_generation=record.record_generation,
                    operation_kind=AdmissionOperationKind.POISON_SPEAKER_NAMESPACE,
                    operation_nonce=nonce,
                )
                record = _changed(
                    record,
                    operation_nonce_sequence=nonce,
                    namespace_poison_ticket=ticket,
                    revocation_degraded=True,
                )
            rewritten.append(replace(effect, ticket=ticket))
            continue
        if not isinstance(effect, RevokeRejectionCapability):
            rewritten.append(effect)
            continue
        existing = next(
            (
                operation
                for operation in pending
                if operation.capability == effect.capability
            ),
            None,
        )
        if existing is not None:
            rewritten.append(replace(effect, ticket=existing.ticket))
            continue
        if len(pending) >= _MAX_PENDING_REVOCATIONS:
            poison_ticket = record.namespace_poison_ticket
            if poison_ticket is None:
                nonce += 1
                poison_ticket = AdmissionOperationTicket(
                    turn_token=record.turn_token,
                    record_generation=record.record_generation,
                    operation_kind=AdmissionOperationKind.POISON_SPEAKER_NAMESPACE,
                    operation_nonce=nonce,
                )
                record = _changed(
                    record,
                    operation_nonce_sequence=nonce,
                    namespace_poison_ticket=poison_ticket,
                    revocation_degraded=True,
                )
            rewritten.append(
                PoisonSpeakerAuthorityNamespace(record.turn_token, poison_ticket)
            )
            continue
        nonce += 1
        ticket = AdmissionOperationTicket(
            turn_token=record.turn_token,
            record_generation=record.record_generation,
            operation_kind=AdmissionOperationKind.REVOKE_CAPABILITY,
            operation_nonce=nonce,
            capability_id=effect.capability.capability_id,
            capability_owner_generation=effect.capability.owner_generation,
            capability_kind=effect.capability.kind,
        )
        pending.append(PendingCapabilityRevocation(ticket, effect.capability))
        rewritten.append(replace(effect, ticket=ticket))
        changed = True
    if changed:
        record = _changed(
            record,
            operation_nonce_sequence=nonce,
            pending_revocations=tuple(pending),
        )
    return record, tuple(rewritten)


def reduce(
    record: VoiceTurnAdmissionRecord,
    event: AdmissionEvent,
    now: float,
) -> tuple[VoiceTurnAdmissionRecord, tuple[AdmissionEffect, ...]]:
    reduced, effects = _reduce_untracked(record, event, now)
    return _track_revocation_effects(reduced, effects)


__all__ = ["maybe_resolve", "reduce"]
