"""Single-writer storage around the pure admission reducer."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace

from main_logic.voice_turn.contracts import VoiceTurnToken

from .._provider_events import ProviderUtteranceKey
from ..speaker_shadow.contracts import SpeakerShadowCandidateKey
from .contracts import (
    AdmissionState,
    AdmissionBulkResult,
    AdmissionEffect,
    AdmissionEvent,
    BoundaryState,
    CandidateBindingState,
    CaptureState,
    Close,
    EvidenceState,
    MicroEventState,
    ProviderBindingState,
    ProviderFinalState,
    RejectionApplyState,
    Reset,
    RouteReplaced,
    SettlementState,
    SpeakerCaptureLeaseRecord,
    SpeakerCaptureLeaseToken,
    SpeakerLeaseAbandoned,
    SpeakerLeaseCaptureClosed,
    SpeakerLeaseChildBinding,
    SpeakerLeaseEvent,
    SpeakerLeaseHigh,
    SpeakerLeaseLow,
    SpeakerLeaseState,
    SpeakerLeaseTransitionOutcome,
    SpeakerLeaseTransitionReceipt,
    SpeakerLeaseUnavailable,
    SpeakerCheckpointKind,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
    VoiceTurnAdmissionRecord,
)
from .reducer import reduce
from .speaker_leases import (
    MAX_SPEAKER_LEASE_CHILDREN,
    MAX_SPEAKER_LEASES,
    SpeakerLeaseIdentityError,
    SpeakerLeaseTerminalError,
    bind_speaker_lease_child,
    reduce_speaker_lease,
)


class AdmissionCapacityError(RuntimeError):
    """A core admission record could not be reserved without data loss."""


class AdmissionIdentityError(RuntimeError):
    """A logical turn token or one of its aliases was reused inconsistently."""


class SpeakerLeaseCapacityError(RuntimeError):
    """A live speaker lease could not be reserved without eviction."""


class VoiceTurnAdmissionCoordinator:
    """Own admission records while leaving every asynchronous effect outside."""

    def __init__(
        self,
        *,
        capacity: int = 8,
        speaker_lease_capacity: int = MAX_SPEAKER_LEASES,
        speaker_lease_child_capacity: int = MAX_SPEAKER_LEASE_CHILDREN,
        retired_speaker_lease_capacity: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        for name, value in (
            ("speaker_lease_capacity", speaker_lease_capacity),
            ("speaker_lease_child_capacity", speaker_lease_child_capacity),
            ("retired_speaker_lease_capacity", retired_speaker_lease_capacity),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if speaker_lease_capacity > MAX_SPEAKER_LEASES:
            raise ValueError(
                f"speaker_lease_capacity cannot exceed {MAX_SPEAKER_LEASES}"
            )
        if speaker_lease_child_capacity > MAX_SPEAKER_LEASE_CHILDREN:
            raise ValueError(
                "speaker_lease_child_capacity cannot exceed "
                f"{MAX_SPEAKER_LEASE_CHILDREN}"
            )
        self._capacity = capacity
        self._speaker_lease_capacity = speaker_lease_capacity
        self._speaker_lease_child_capacity = speaker_lease_child_capacity
        self._retired_speaker_lease_capacity = retired_speaker_lease_capacity
        self._clock = clock
        self._records: dict[VoiceTurnToken, VoiceTurnAdmissionRecord] = {}
        self._speaker_leases: dict[
            SpeakerCaptureLeaseToken,
            SpeakerCaptureLeaseRecord,
        ] = {}
        self._speaker_candidate_bindings: dict[
            SpeakerShadowCandidateKey,
            SpeakerCaptureLeaseToken,
        ] = {}
        self._provider_speaker_lease_bindings: dict[
            ProviderUtteranceKey,
            tuple[SpeakerCaptureLeaseToken, VoiceTurnToken],
        ] = {}
        self._retired_speaker_leases: OrderedDict[
            SpeakerCaptureLeaseToken,
            None,
        ] = OrderedDict()
        self._retired_turn_high_water: dict[object, int] = {}
        self._record_generation = 0
        self._speaker_lease_record_generation = 0
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def speaker_lease_capacity(self) -> int:
        return self._speaker_lease_capacity

    @property
    def speaker_lease_child_capacity(self) -> int:
        return self._speaker_lease_child_capacity

    async def open_speaker_lease(
        self,
        lease_token: SpeakerCaptureLeaseToken,
        candidate: SpeakerShadowCandidateKey,
    ) -> SpeakerCaptureLeaseRecord:
        """Reserve one stable parent verdict identity without evicting another."""

        if type(lease_token) is not SpeakerCaptureLeaseToken:
            raise TypeError("lease_token must be SpeakerCaptureLeaseToken")
        if type(candidate) is not SpeakerShadowCandidateKey:
            raise TypeError("candidate must be SpeakerShadowCandidateKey")
        async with self._lock:
            existing = self._speaker_leases.get(lease_token)
            if existing is not None:
                if existing.candidate != candidate:
                    raise AdmissionIdentityError("ASR_SPEAKER_LEASE_CANDIDATE_CONFLICT")
                return existing
            if lease_token in self._retired_speaker_leases:
                raise AdmissionIdentityError("ASR_SPEAKER_LEASE_ALREADY_RETIRED")
            existing_token = self._speaker_candidate_bindings.get(candidate)
            if existing_token is not None and existing_token != lease_token:
                raise AdmissionIdentityError(
                    "ASR_SPEAKER_LEASE_CANDIDATE_ALREADY_BOUND"
                )
            if len(self._speaker_leases) >= self._speaker_lease_capacity:
                raise SpeakerLeaseCapacityError("ASR_SPEAKER_LEASE_CAPACITY_EXHAUSTED")
            self._speaker_lease_record_generation += 1
            record = SpeakerCaptureLeaseRecord(
                lease_token=lease_token,
                record_generation=self._speaker_lease_record_generation,
                candidate=candidate,
            )
            self._speaker_leases[lease_token] = record
            self._speaker_candidate_bindings[candidate] = lease_token
            return record

    async def attach_turn_to_speaker_lease(
        self,
        turn_token: VoiceTurnToken,
        lease_token: SpeakerCaptureLeaseToken,
        provider_key: ProviderUtteranceKey,
    ) -> VoiceTurnAdmissionRecord:
        """Open and bind one Provider child atomically under the same writer."""

        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if type(lease_token) is not SpeakerCaptureLeaseToken:
            raise TypeError("lease_token must be SpeakerCaptureLeaseToken")
        if type(provider_key) is not ProviderUtteranceKey:
            raise TypeError("provider_key must be ProviderUtteranceKey")
        async with self._lock:
            lease = self._speaker_leases.get(lease_token)
            if lease is None:
                raise KeyError(lease_token)
            if self._speaker_candidate_bindings.get(lease.candidate) != lease_token:
                raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
            binding = SpeakerLeaseChildBinding(provider_key, turn_token)
            provider_binding = self._provider_speaker_lease_bindings.get(provider_key)
            expected_binding = (lease_token, turn_token)
            terminal_parent = lease.state in {
                SpeakerLeaseState.ALLOW,
                SpeakerLeaseState.UNAVAILABLE,
                SpeakerLeaseState.DENY_LATCHED,
                SpeakerLeaseState.ABANDONED,
            }
            if provider_binding not in {None, expected_binding}:
                raise AdmissionIdentityError(
                    "ASR_SPEAKER_LEASE_PROVIDER_KEY_ALREADY_BOUND"
                )
            existing = self._records.get(turn_token)
            if existing is not None:
                provider_placeholder = (
                    existing.provider_binding_state is ProviderBindingState.UNBOUND
                    and existing.provider_key is None
                )
                provider_exact = (
                    existing.provider_binding_state is ProviderBindingState.BOUND
                    and existing.provider_key == provider_key
                )
                candidate_unbound = (
                    existing.candidate_binding_state is CandidateBindingState.UNBOUND
                    and existing.speaker_candidate is None
                    and existing.speaker_lease_token is None
                    and existing.speaker_authority_generation is None
                    and existing.capture_state is CaptureState.NONE
                )
                candidate_arming = (
                    existing.candidate_binding_state is CandidateBindingState.ARMING
                    and existing.speaker_candidate is None
                    and existing.speaker_lease_token is None
                    and existing.speaker_authority_generation is not None
                    and existing.capture_state is CaptureState.NONE
                )
                candidate_exact = (
                    existing.candidate_binding_state is CandidateBindingState.BOUND
                    and existing.speaker_candidate == lease.candidate
                    and existing.speaker_lease_token == lease_token
                    and (
                        existing.capture_state is CaptureState.COLLECTING
                        or (
                            lease.state is SpeakerLeaseState.UNAVAILABLE
                            and existing.capture_state is CaptureState.UNAVAILABLE
                        )
                    )
                )
                if (
                    existing.turn_token != turn_token
                    or not (provider_placeholder or provider_exact)
                    or not (candidate_unbound or candidate_arming or candidate_exact)
                ):
                    raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
                if candidate_exact:
                    if (
                        not provider_exact
                        or provider_binding != expected_binding
                        or binding not in lease.child_bindings
                        or (
                            terminal_parent
                            and not self._terminal_parent_child_is_exact(
                                existing,
                                lease,
                            )
                        )
                    ):
                        raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
                    return existing
                if existing.terminal_disposition is not None:
                    raise AdmissionIdentityError(
                        "ASR_ADMISSION_TERMINAL_BINDING_CONFLICT"
                    )
                if provider_binding is not None or binding in lease.child_bindings:
                    raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
            else:
                if provider_binding is not None or binding in lease.child_bindings:
                    raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
                if turn_token.turn_id <= self._retired_turn_high_water.get(
                    turn_token.ingress,
                    0,
                ):
                    raise AdmissionIdentityError("ASR_ADMISSION_TURN_ALREADY_RETIRED")
                if len(self._records) >= self._capacity:
                    raise AdmissionCapacityError("ASR_ADMISSION_CAPACITY_EXHAUSTED")

            if terminal_parent:
                raise SpeakerLeaseTerminalError("ASR_SPEAKER_LEASE_TERMINAL")

            try:
                updated_lease = bind_speaker_lease_child(
                    lease,
                    binding,
                    capacity=self._speaker_lease_child_capacity,
                )
            except SpeakerLeaseIdentityError as exc:
                raise AdmissionIdentityError(str(exc)) from exc

            if existing is None:
                self._record_generation += 1
                record = VoiceTurnAdmissionRecord(
                    turn_token=turn_token,
                    record_generation=self._record_generation,
                    provider_binding_state=ProviderBindingState.BOUND,
                    candidate_binding_state=CandidateBindingState.BOUND,
                    capture_state=CaptureState.COLLECTING,
                    provider_key=provider_key,
                    speaker_lease_token=lease_token,
                    speaker_candidate=lease.candidate,
                )
            else:
                record = replace(
                    existing,
                    logical_revision=existing.logical_revision + 1,
                    provider_binding_state=ProviderBindingState.BOUND,
                    candidate_binding_state=CandidateBindingState.BOUND,
                    capture_state=CaptureState.COLLECTING,
                    provider_key=provider_key,
                    speaker_lease_token=lease_token,
                    speaker_candidate=lease.candidate,
                )
            self._speaker_leases[lease_token] = updated_lease
            self._records[turn_token] = record
            self._provider_speaker_lease_bindings[provider_key] = expected_binding
            return record

    @staticmethod
    def _terminal_parent_child_is_exact(
        record: VoiceTurnAdmissionRecord,
        lease: SpeakerCaptureLeaseRecord,
    ) -> bool:
        if lease.state is SpeakerLeaseState.ALLOW:
            return bool(
                record.capture_state is CaptureState.COLLECTING
                and record.evidence_state is EvidenceState.ALLOW
                and record.last_speaker_sequence_no == 1
            )
        if lease.state is SpeakerLeaseState.UNAVAILABLE:
            return bool(
                record.capture_state is CaptureState.UNAVAILABLE
                and record.evidence_state is EvidenceState.UNAVAILABLE
                and record.rejection_apply_state is RejectionApplyState.STALE
                and record.last_speaker_sequence_no == 1
            )
        if lease.state is SpeakerLeaseState.DENY_LATCHED:
            return bool(
                record.capture_state is CaptureState.COLLECTING
                and record.evidence_state is EvidenceState.DENY_LATCHED
                and record.last_speaker_sequence_no == 2
            )
        if lease.state is SpeakerLeaseState.ABANDONED:
            return record.admission_state is AdmissionState.ABANDONED
        return False

    async def detach_turn_from_speaker_lease(
        self,
        turn_token: VoiceTurnToken,
        lease_token: SpeakerCaptureLeaseToken,
        provider_key: ProviderUtteranceKey,
    ) -> bool:
        """Compensate one exact child attach before any final or side effect."""

        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if type(lease_token) is not SpeakerCaptureLeaseToken:
            raise TypeError("lease_token must be SpeakerCaptureLeaseToken")
        if type(provider_key) is not ProviderUtteranceKey:
            raise TypeError("provider_key must be ProviderUtteranceKey")
        async with self._lock:
            lease = self._speaker_leases.get(lease_token)
            record = self._records.get(turn_token)
            binding = SpeakerLeaseChildBinding(provider_key, turn_token)
            provider_binding = self._provider_speaker_lease_bindings.get(provider_key)
            expected_provider_binding = (lease_token, turn_token)
            binding_count = (
                lease.child_bindings.count(binding) if lease is not None else 0
            )

            if record is None and provider_binding is None and binding_count == 0:
                return False
            if lease is None:
                raise AdmissionIdentityError("ASR_ADMISSION_DETACH_IDENTITY_CONFLICT")
            pending_projection_exact = bool(
                lease.state
                in {SpeakerLeaseState.COLLECTING, SpeakerLeaseState.FIRST_LOW}
                and record is not None
                and record.capture_state is CaptureState.COLLECTING
                and record.evidence_state is EvidenceState.NONE
                and record.rejection_apply_state is RejectionApplyState.NOT_STARTED
                and record.last_speaker_sequence_no == 0
            )
            terminal_projection_exact = bool(
                record is not None
                and self._terminal_parent_child_is_exact(record, lease)
            )
            if (
                record is None
                or record.turn_token != turn_token
                or record.provider_binding_state is not ProviderBindingState.BOUND
                or record.candidate_binding_state is not CandidateBindingState.BOUND
                or record.provider_key != provider_key
                or record.speaker_lease_token != lease_token
                or record.speaker_candidate != lease.candidate
                or self._speaker_candidate_bindings.get(lease.candidate) != lease_token
                or provider_binding != expected_provider_binding
                or binding_count != 1
                or not (pending_projection_exact or terminal_projection_exact)
            ):
                raise AdmissionIdentityError("ASR_ADMISSION_DETACH_IDENTITY_CONFLICT")

            side_effect_free = bool(
                record.terminal_disposition is None
                and record.boundary_state is BoundaryState.OPEN
                and record.micro_event_state is MicroEventState.NOT_APPLICABLE
                and record.provider_final_state is ProviderFinalState.NOT_RECEIVED
                and record.admission_state is AdmissionState.RESERVED
                and record.operation_nonce_sequence == 0
                and record.core_settlement_state is SettlementState.NOT_STARTED
                and record.transport_settlement_state is SettlementState.NOT_STARTED
                and record.lifecycle_settlement_state is SettlementState.NOT_STARTED
                and record.rejection_capability is None
                and record.pending_final is None
                and record.resolution_ticket is None
                and record.capture_through_sequence_no is None
                and not record.micro_event_shadow_would_suppress
                and not record.micro_event_terminal_counted
                and record.rejection_operation_nonce is None
                and record.rejection_operation_capability_id is None
                and record.rejection_operation_owner_generation is None
                and record.rejection_operation_kind is None
                and record.revoked_rejection_ticket is None
                and record.revoked_rejection_capability is None
                and not record.pending_revocations
                and not record.revocation_degraded
                and record.namespace_poison_ticket is None
                and record.deadline_operation_nonce is None
                and not record.provider_boundary_deadline_expired
                and record.partial_settlement_disposition is None
                and not record.speaker_deny_cleanup_failed_counted
            )
            if not side_effect_free:
                raise AdmissionIdentityError("ASR_ADMISSION_DETACH_ALREADY_COMMITTED")

            self._speaker_leases[lease_token] = replace(
                lease,
                logical_revision=lease.logical_revision + 1,
                child_bindings=tuple(
                    child for child in lease.child_bindings if child != binding
                ),
            )
            self._records.pop(turn_token)
            self._provider_speaker_lease_bindings.pop(provider_key)
            return True

    async def post_speaker_lease(
        self,
        lease_token: SpeakerCaptureLeaseToken,
        event: SpeakerLeaseEvent,
        *,
        now: float | None = None,
    ) -> SpeakerLeaseTransitionReceipt:
        """Reduce one parent fact and return its typed linearization receipt."""

        if type(lease_token) is not SpeakerCaptureLeaseToken:
            raise TypeError("lease_token must be SpeakerCaptureLeaseToken")
        effective_now = self._clock() if now is None else now
        async with self._lock:
            record = self._speaker_leases.get(lease_token)
            if record is None:
                raise KeyError(lease_token)
            reduced, diagnostics = reduce_speaker_lease(record, event)
            if reduced.terminal_disposition is None:
                self._speaker_leases[lease_token] = reduced
                return SpeakerLeaseTransitionReceipt(
                    lease_token=lease_token,
                    before_state=record.state,
                    after_state=reduced.state,
                    outcome=(
                        SpeakerLeaseTransitionOutcome.NON_TERMINAL
                        if reduced is not record
                        else SpeakerLeaseTransitionOutcome.STALE
                    ),
                    terminal_sequence_no=None,
                    capture_through_sequence_no=reduced.capture_through_sequence_no,
                    frozen_children=(),
                    child_results=(),
                    diagnostics=diagnostics,
                )
            if record.terminal_disposition is not None:
                return SpeakerLeaseTransitionReceipt(
                    lease_token=lease_token,
                    before_state=record.state,
                    after_state=record.state,
                    outcome=self._terminal_speaker_event_outcome(record, event),
                    terminal_sequence_no=record.terminal_sequence_no,
                    capture_through_sequence_no=record.capture_through_sequence_no,
                    frozen_children=record.child_bindings,
                    child_results=(),
                    diagnostics=diagnostics,
                )
            results, child_updates = self._prepare_speaker_lease_terminal_fanout(
                reduced,
                now=effective_now,
            )
            self._speaker_leases[lease_token] = reduced
            for turn_token, child in child_updates:
                self._records[turn_token] = child
            return SpeakerLeaseTransitionReceipt(
                lease_token=lease_token,
                before_state=record.state,
                after_state=reduced.state,
                outcome=SpeakerLeaseTransitionOutcome.APPLIED,
                terminal_sequence_no=reduced.terminal_sequence_no,
                capture_through_sequence_no=reduced.capture_through_sequence_no,
                frozen_children=reduced.child_bindings,
                child_results=results,
                diagnostics=diagnostics,
            )

    @staticmethod
    def _terminal_speaker_event_outcome(
        record: SpeakerCaptureLeaseRecord,
        event: SpeakerLeaseEvent,
    ) -> SpeakerLeaseTransitionOutcome:
        if record.state is SpeakerLeaseState.ABANDONED:
            return (
                SpeakerLeaseTransitionOutcome.IDEMPOTENT
                if isinstance(event, SpeakerLeaseAbandoned)
                else SpeakerLeaseTransitionOutcome.STALE
            )
        candidate = getattr(event, "candidate", None)
        if candidate != record.candidate:
            return SpeakerLeaseTransitionOutcome.STALE
        terminal_sequence_no = record.terminal_sequence_no
        event_sequence_no = getattr(
            event,
            "through_sequence_no",
            getattr(event, "sequence_no", None),
        )
        if event_sequence_no != terminal_sequence_no:
            return SpeakerLeaseTransitionOutcome.STALE
        exact = bool(
            (
                record.state is SpeakerLeaseState.DENY_LATCHED
                and isinstance(event, SpeakerLeaseLow)
                and event.checkpoint_kind
                in {
                    SpeakerCheckpointKind.SECOND,
                    SpeakerCheckpointKind.COMPLETION_CONFIRMATION,
                }
            )
            or (
                record.state is SpeakerLeaseState.ALLOW
                and isinstance(event, SpeakerLeaseHigh)
            )
            or (
                record.state is SpeakerLeaseState.UNAVAILABLE
                and isinstance(event, SpeakerLeaseUnavailable)
            )
            or (
                record.state is SpeakerLeaseState.UNAVAILABLE
                and isinstance(event, SpeakerLeaseCaptureClosed)
                and record.capture_through_sequence_no == event.through_sequence_no
            )
        )
        return (
            SpeakerLeaseTransitionOutcome.IDEMPOTENT
            if exact
            else SpeakerLeaseTransitionOutcome.CONFLICT
        )

    def _prepare_speaker_lease_terminal_fanout(
        self,
        lease: SpeakerCaptureLeaseRecord,
        *,
        now: float,
    ) -> tuple[
        tuple[AdmissionBulkResult, ...],
        tuple[tuple[VoiceTurnToken, VoiceTurnAdmissionRecord], ...],
    ]:
        results: list[AdmissionBulkResult] = []
        updates: list[tuple[VoiceTurnToken, VoiceTurnAdmissionRecord]] = []
        for binding in lease.child_bindings:
            child = self._records.get(binding.turn_token)
            if child is None:
                continue
            events: tuple[AdmissionEvent, ...]
            if lease.state is SpeakerLeaseState.DENY_LATCHED:
                next_sequence = child.last_speaker_sequence_no + 1
                if child.evidence_state is EvidenceState.FIRST_LOW:
                    events = (
                        SpeakerLow(
                            lease.candidate,
                            next_sequence,
                            SpeakerCheckpointKind.SECOND,
                        ),
                    )
                else:
                    events = (
                        SpeakerLow(
                            lease.candidate,
                            next_sequence,
                            SpeakerCheckpointKind.FIRST,
                        ),
                        SpeakerLow(
                            lease.candidate,
                            next_sequence + 1,
                            SpeakerCheckpointKind.SECOND,
                        ),
                    )
            elif lease.state is SpeakerLeaseState.ALLOW:
                events = (
                    SpeakerHigh(
                        lease.candidate,
                        child.last_speaker_sequence_no + 1,
                    ),
                )
            elif lease.state is SpeakerLeaseState.UNAVAILABLE:
                events = (
                    SpeakerUnavailable(
                        lease.candidate,
                        child.last_speaker_sequence_no + 1,
                    ),
                )
            elif lease.state is SpeakerLeaseState.ABANDONED:
                events = (RouteReplaced(),)
            else:
                continue

            effects: list[AdmissionEffect] = []
            for event in events:
                child, emitted = reduce(child, event, now)
                effects.extend(emitted)
            updates.append((binding.turn_token, child))
            results.append(
                AdmissionBulkResult(
                    binding.turn_token,
                    tuple(effects),
                    lease.lease_token,
                    lease.state,
                )
            )
        return tuple(results), tuple(updates)

    async def get_speaker_lease(
        self,
        lease_token: SpeakerCaptureLeaseToken,
    ) -> SpeakerCaptureLeaseRecord | None:
        async with self._lock:
            return self._speaker_leases.get(lease_token)

    async def live_speaker_lease_tokens(
        self,
    ) -> tuple[SpeakerCaptureLeaseToken, ...]:
        async with self._lock:
            return tuple(self._speaker_leases)

    async def retire_speaker_lease(
        self,
        lease_token: SpeakerCaptureLeaseToken,
    ) -> bool:
        """Retire only a terminal parent whose child records are all gone."""

        if type(lease_token) is not SpeakerCaptureLeaseToken:
            raise TypeError("lease_token must be SpeakerCaptureLeaseToken")
        async with self._lock:
            record = self._speaker_leases.get(lease_token)
            if record is None:
                return False
            if record.terminal_disposition is None or any(
                binding.turn_token in self._records for binding in record.child_bindings
            ):
                return False
            self._speaker_leases.pop(lease_token, None)
            if self._speaker_candidate_bindings.get(record.candidate) == lease_token:
                self._speaker_candidate_bindings.pop(record.candidate, None)
            for binding in record.child_bindings:
                expected = (lease_token, binding.turn_token)
                if (
                    self._provider_speaker_lease_bindings.get(binding.provider_key)
                    == expected
                ):
                    self._provider_speaker_lease_bindings.pop(
                        binding.provider_key,
                        None,
                    )
            self._retired_speaker_leases[lease_token] = None
            while (
                len(self._retired_speaker_leases) > self._retired_speaker_lease_capacity
            ):
                self._retired_speaker_leases.popitem(last=False)
            return True

    async def open_turn(
        self,
        turn_token: VoiceTurnToken,
        *,
        provider_key: ProviderUtteranceKey | None = None,
        speaker_candidate: SpeakerShadowCandidateKey | None = None,
    ) -> VoiceTurnAdmissionRecord:
        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        async with self._lock:
            existing = self._records.get(turn_token)
            if existing is not None:
                if (
                    provider_key is not None and existing.provider_key != provider_key
                ) or (
                    speaker_candidate is not None
                    and existing.speaker_candidate != speaker_candidate
                ):
                    raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
                return existing
            if turn_token.turn_id <= self._retired_turn_high_water.get(
                turn_token.ingress,
                0,
            ):
                raise AdmissionIdentityError("ASR_ADMISSION_TURN_ALREADY_RETIRED")
            if len(self._records) >= self._capacity:
                raise AdmissionCapacityError("ASR_ADMISSION_CAPACITY_EXHAUSTED")
            self._record_generation += 1
            record = VoiceTurnAdmissionRecord(
                turn_token=turn_token,
                record_generation=self._record_generation,
                provider_binding_state=(
                    ProviderBindingState.BOUND
                    if provider_key is not None
                    else ProviderBindingState.UNBOUND
                ),
                candidate_binding_state=(
                    CandidateBindingState.BOUND
                    if speaker_candidate is not None
                    else CandidateBindingState.UNBOUND
                ),
                capture_state=(
                    CaptureState.COLLECTING
                    if speaker_candidate is not None
                    else CaptureState.NONE
                ),
                provider_key=provider_key,
                speaker_candidate=speaker_candidate,
            )
            self._records[turn_token] = record
            return record

    async def post(
        self,
        turn_token: VoiceTurnToken,
        event: AdmissionEvent,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionEffect, ...]:
        """Reduce under the short lock and return effects without executing them."""

        async with self._lock:
            record = self._records.get(turn_token)
            if record is None:
                raise KeyError(turn_token)
            reduced, effects = reduce(
                record,
                event,
                self._clock() if now is None else now,
            )
            self._records[turn_token] = reduced
            return effects

    async def get_record(
        self,
        turn_token: VoiceTurnToken,
    ) -> VoiceTurnAdmissionRecord | None:
        async with self._lock:
            return self._records.get(turn_token)

    async def live_turn_tokens(self) -> tuple[VoiceTurnToken, ...]:
        """Return one insertion-ordered snapshot without exposing the record table."""

        async with self._lock:
            return tuple(self._records)

    async def invalidate_all(
        self,
        event: Reset | Close | RouteReplaced,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionBulkResult, ...]:
        """Reduce one route invalidation against the complete live snapshot.

        The reducer is run for every record while this coordinator remains the
        single writer.  Effects are only returned after the lock is released;
        callers execute them and post their acknowledgements through ingress.
        """

        if type(event) not in {Reset, Close, RouteReplaced}:
            raise TypeError("event must be Reset, Close, or RouteReplaced")
        effective_now = self._clock() if now is None else now
        async with self._lock:
            lease_updates: list[
                tuple[SpeakerCaptureLeaseToken, SpeakerCaptureLeaseRecord]
            ] = []
            for lease_token, lease in self._speaker_leases.items():
                reduced, _ = reduce_speaker_lease(lease, SpeakerLeaseAbandoned())
                lease_updates.append((lease_token, reduced))
            results: list[AdmissionBulkResult] = []
            record_updates: list[tuple[VoiceTurnToken, VoiceTurnAdmissionRecord]] = []
            for turn_token, record in self._records.items():
                reduced, effects = reduce(record, event, effective_now)
                record_updates.append((turn_token, reduced))
                results.append(AdmissionBulkResult(turn_token, effects))
            for lease_token, lease in lease_updates:
                self._speaker_leases[lease_token] = lease
            for turn_token, record in record_updates:
                self._records[turn_token] = record
            return tuple(results)

    async def retire(self, turn_token: VoiceTurnToken) -> bool:
        """Remove only an already-settled record; never evict live admission."""

        async with self._lock:
            record = self._records.get(turn_token)
            if record is None:
                return False
            if record.terminal_disposition is None or any(
                state not in {SettlementState.SETTLED, SettlementState.DEGRADED}
                for state in (
                    record.core_settlement_state,
                    record.transport_settlement_state,
                    record.lifecycle_settlement_state,
                )
            ):
                return False
            if record.revoked_rejection_ticket is not None:
                return False
            if record.pending_revocations:
                return False
            if record.revocation_degraded:
                return False
            if record.namespace_poison_ticket is not None:
                return False
            if record.rejection_capability is not None:
                return False
            self._records.pop(turn_token, None)
            self._retired_turn_high_water[turn_token.ingress] = max(
                self._retired_turn_high_water.get(turn_token.ingress, 0),
                turn_token.turn_id,
            )
            return True


__all__ = [
    "AdmissionBulkResult",
    "AdmissionCapacityError",
    "AdmissionIdentityError",
    "SpeakerLeaseCapacityError",
    "SpeakerLeaseTransitionOutcome",
    "SpeakerLeaseTransitionReceipt",
    "VoiceTurnAdmissionCoordinator",
]
