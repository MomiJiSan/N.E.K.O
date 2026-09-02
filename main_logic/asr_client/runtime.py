"""Provider-neutral independent-ASR runtime with explicit Core callbacks."""

from __future__ import annotations

import asyncio
import time
import weakref
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from main_logic.asr_client import (
    _attach_partial_callback,
    _create_asr_session_from_selection,
    _resolve_asr_selection,
)
from main_logic.voice_turn.contracts import (
    AsrFailureEvent,
    AsrLifecycleNotification,
    AsrStatusEvent,
    AsrSubmitResult,
    AsrSubmitStatus,
    SpeechActivityEvent,
    VoicePartialEvent,
    VoiceTranscriptEvent,
    VoiceTurnToken,
)
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame

from ._infra import logger, _READY_TIMEOUT_SECONDS
from ._provider_events import (
    ProviderEndpointNotification,
    ProviderFinalNotification,
    ProviderUtteranceKey,
)
from .audio import AsrAudioDispatcher
from .admission.contracts import (
    AbortProviderTransport,
    AdmissionDisposition,
    AdmissionEffect,
    AdmissionOperationTicket,
    AdmissionResolutionTicket,
    ApplyRejection,
    BoundaryExact,
    BoundaryProof,
    BoundaryUnknown,
    CandidateBound,
    CapabilityRevokeFailed,
    CapabilityRevoked,
    CaptureClosed,
    Close,
    ConstrainRejectionDeadline,
    CoreSettled,
    CountDiagnostic,
    FinalDeadlineExpired,
    LifecycleSettled,
    MicroEventAllowed,
    MicroEventPending,
    MicroEventSuppressed,
    MicroEventUnavailable,
    PendingProviderFinal,
    PoisonSpeakerAuthorityNamespace,
    ProviderBound,
    ProviderFinalReceived,
    RejectionApplied,
    RejectionCapability,
    RejectionCapabilityKind,
    RejectionFailed,
    RejectionStale,
    Reset,
    ResolveReserved,
    RevokeRejectionCapability,
    RouteReplaced,
    ScheduleFinalDeadline,
    SettlePartial,
    SpeakerAuthorityPending,
    SpeakerAuthorityUnarmed,
    SpeakerAuthorityUnavailable,
    SpeakerCheckpointKind,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
    SpeakerAuthorityNamespacePoisonFailed,
    SpeakerAuthorityNamespacePoisoned,
    TransportSettled,
    TurnSealed,
)
from .admission.coordinator import VoiceTurnAdmissionCoordinator
from .admission.ingress import (
    AdmissionIngressCapacityError,
    AdmissionIngressClosedError,
    AdmissionIngressLane,
)
from .admission.provider_turns import (
    ProviderAliasConflictError,
    ProviderBoundaryResult,
    ProviderTurnCorrelator,
)
from ._registry_meta import AsrProviderAvailability
from .endpointing.detector import (
    AsrDetectorDispatcher,
    CoreDetectorEventEnvelope,
    DetectorActivityEvent,
    DetectorCandidateKey,
    DetectorIngressIdentity,
    DetectorPrewarmEvent,
    DetectorRuntimeEvent,
    DetectorTransportPrewarmEvent,
    DetectorSubmitStatus,
    DetectorTurnEvent,
    ProviderCandidateFence,
    ProviderSpeakerBoundarySnapshot,
)
from .endpointing.detector_runtime import (
    DetectorCandidateRejectionCommitResult,
    DetectorCandidateRejectionLease,
    DetectorRuntime,
    SmartTurnLease,
)
from .endpointing.micro_event_policy import (
    ProviderMicroEventConfig,
    ProviderMicroEventDecision,
)
from .endpointing.throttle_policy import ThrottleAction
from .lifecycle import (
    AudioDisposition,
    FinalKey,
    VoiceIngressToken,
    VoiceInputLifecycleController,
    VoiceLifecycleEvent,
    VoiceLifecycleState,
    VoiceRouteMode,
    VoiceTransportToken,
)
from .provider_policy import resolve_provider_policy
from .speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
    SpeakerShadowObserver,
)
from .transcript import (
    TranscriptDispatcher,
    TranscriptEnvelope,
    TranscriptTerminalSettlement,
)


# The frontend gives a voice start this long before it cancels and fires
# end_session (app-buttons.js, and the automatic-restart path in
# app-websocket.js use the same value). Mirrored here because
# _start_session_activate awaits the ASR connect loop BEFORE sending
# session_started: any retry budget that outlives this deadline cannot produce
# a verdict the client will still be listening for.
_FRONTEND_START_DEADLINE_SECONDS = 15.0

# Aggregate ceiling for the whole connect-and-retry phase. Deliberately under
# the deadline above, leaving room for the rest of the start (the ack send and
# the pending-input flush that follow it) so the fail-closed verdict lands
# BEFORE the client gives up rather than a second after.
_CONNECT_TOTAL_BUDGET_SECONDS = 12.0

# Public alias. The dedupe reroute in core/lifecycle.py runs a whole extra
# connect phase AFTER already spending part of the frontend deadline waiting,
# so it has to know this ceiling to tell whether its verdict can still land
# before the client gives up.
ASR_CONNECT_TOTAL_BUDGET_SECONDS = _CONNECT_TOTAL_BUDGET_SECONDS
_CANDIDATE_REJECTION_WATCHDOG_SECONDS = 10.0
_CANDIDATE_REJECTION_RECOVERY_STEP_TIMEOUT_SECONDS = 1.0
_CANDIDATE_REJECTION_REINSTALL_ATTEMPTS = 2
_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS = 0.2
_PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS = 0.2
_ASR_TERMINAL_CLOSE_TIMEOUT_SECONDS = 1.0
_ASR_TERMINAL_HARD_CLOSE_RESERVE_SECONDS = 0.6
_ASR_TERMINAL_CLOSE_JOIN_SLICE_SECONDS = 0.1
_MAX_BUFFERED_PROVIDER_SPEAKER_SPANS = 8
_MAX_PROVIDER_BOUNDARY_SNAPSHOTS = 8
_MAX_SPEAKER_EVIDENCE_BRIDGE_RECORDS = 256
_PROVIDER_MICRO_EVENT_SHADOW_CONFIG = ProviderMicroEventConfig(
    mode="shadow",
    calibration_revision=None,
    maximum_silero_span_ms=384,
    maximum_post_start_onset_windows=4,
    maximum_rnnoise_active_run_upper_bound_ms=160,
)
_SPEAKER_REJECTION_METRIC_NAMES = (
    "speaker_deny_latched_count",
    "speaker_deny_final_dropped_count",
    "speaker_deny_cleanup_failed_count",
    "speaker_late_fact_stale_count",
    "speaker_partial_quarantined_count",
    "rejection_request_failed_count",
    "rejection_task_scheduled_count",
    "rejection_task_applied_count",
    "rejection_task_stale_count",
    "rejection_stale_initial_count",
    "rejection_stale_prepare_count",
    "rejection_stale_runtime_fence_count",
    "rejection_stale_candidate_fence_count",
    "rejection_stale_smart_turn_count",
    "rejection_stale_commit_count",
    "rejection_task_cleanup_degraded_count",
    "rejection_task_failure_count",
    "rejection_task_cancelled_count",
    "admission_terminal_forward_count",
    "admission_terminal_drop_count",
    "admission_terminal_abandon_count",
    "admission_deadline_forward_count",
    "admission_rejection_applied_active_count",
    "admission_rejection_applied_sealed_count",
    "admission_core_settlement_degraded_count",
    "admission_transport_settlement_degraded_count",
    "admission_lifecycle_settlement_degraded_count",
    "admission_boundary_proof_retired_count",
    "admission_boundary_proof_overflow_count",
    "admission_late_boundary_ignored_count",
    "admission_late_fact_ignored_count",
    "admission_stale_turn_opened_count",
    "admission_provider_alias_conflict_count",
    "admission_candidate_alias_conflict_count",
    "admission_stale_speaker_fact_count",
    "admission_stale_capture_close_count",
    "admission_speaker_sequence_gap_count",
    "admission_late_exact_after_unknown_count",
    "admission_foreign_capability_ignored_count",
    "admission_capability_conflict_count",
    "admission_stale_boundary_unknown_count",
    "admission_revoked_operation_applied_late_count",
    "admission_late_operation_ignored_count",
    "admission_rejection_kind_mismatch_count",
    "admission_revoked_operation_settled_count",
    "admission_late_revoke_ack_ignored_count",
    "admission_late_revoke_failure_ignored_count",
    "admission_late_namespace_poison_ack_count",
    "admission_late_namespace_poison_failure_count",
    "admission_final_after_retirement_ignored_count",
    "admission_stale_provider_final_count",
    "admission_conflicting_provider_final_count",
    "admission_late_deadline_ignored_count",
    "admission_stale_core_settlement_count",
    "admission_stale_transport_settlement_count",
    "admission_stale_lifecycle_settlement_count",
    "provider_candidate_bind_missing_identity_count",
    "provider_candidate_bind_missing_candidate_count",
    "provider_candidate_bind_identity_rejected_count",
    "provider_candidate_bind_deferred_count",
    "provider_candidate_bind_state_skipped_count",
    "provider_candidate_bind_attempt_count",
    "provider_candidate_bind_success_count",
    "provider_candidate_bind_empty_count",
    "provider_candidate_bind_failed_count",
    "provider_boundary_preseal_started_count",
    "provider_boundary_exact_ready_count",
    "provider_boundary_unknown_ready_count",
    "provider_boundary_conflict_count",
    "provider_boundary_overflow_count",
    "provider_boundary_stale_count",
    "provider_boundary_ordered_jit_unknown_count",
    "provider_preseal_rejection_consumed_count",
    "provider_preseal_rejection_stale_count",
    "micro_event_suppressed_count",
    "micro_event_shadow_forward_count",
)


def _new_speaker_rejection_metrics() -> dict[str, int]:
    return {name: 0 for name in _SPEAKER_REJECTION_METRIC_NAMES}


def _speaker_factory_enforces_admission(
    factory: SpeakerShadowFactory | None,
) -> bool:
    """Accept only an explicit internal admission-enforcement declaration."""

    return getattr(factory, "enforces_admission", False) is True


def _uses_smart_turn_endpointing(provider_policy: Any) -> bool:
    """Honor the endpoint authority independently of transport shape."""

    return bool(provider_policy.endpoint_authority == "smart_turn")


class AsrStartStatus(Enum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AsrStartResult:
    status: AsrStartStatus
    provider: str | None = None
    failure_code: str | None = None
    session_epoch: int = -1


@dataclass(frozen=True, slots=True)
class AsrRuntimeCallbacks:
    display_name: Callable[[], str]
    on_prepare_turn: Callable[[VoiceTurnToken], Awaitable[bool]]
    on_partial: Callable[[VoicePartialEvent], Awaitable[None]]
    on_final: Callable[[VoiceTranscriptEvent], Awaitable[None]]
    on_turn_abandoned: Callable[[VoiceTurnToken], Awaitable[None]]
    on_failure: Callable[[AsrFailureEvent], Awaitable[None]]
    on_status: Callable[[AsrStatusEvent], Awaitable[None]]
    on_lifecycle: Callable[[AsrLifecycleNotification], Awaitable[None]]


SpeakerShadowFactory = Callable[[], SpeakerShadowObserver | None]


@dataclass(frozen=True, slots=True)
class _AsrRuntimeIdentity:
    start_generation: int
    session_epoch: int
    audio_generation: int
    lifecycle: VoiceInputLifecycleController | None
    transport_generation: int | None
    detector: DetectorRuntime | None
    session: Any
    provider: str | None
    session_factory: Any
    transport_selection: Any
    transport_task: asyncio.Task[None] | None
    ingress_token: VoiceIngressToken | None = None
    turn_token: VoiceTurnToken | None = None


@dataclass(slots=True)
class _BufferedProviderSpeakerSpan:
    """One PCM-free ordered span inside lifecycle-owned buffered audio."""

    start_byte: int
    end_byte: int
    first_identity: DetectorIngressIdentity | None
    last_identity: DetectorIngressIdentity | None
    split_before_audio: bool
    evidence_complete: bool


@dataclass(slots=True)
class _BufferedProviderSpeakerObservation:
    """Bounded span metadata for PCM retained only by the lifecycle."""

    total_bytes: int
    spans: list[_BufferedProviderSpeakerSpan]
    overflowed: bool = False


@dataclass(slots=True)
class _AdmissionCapabilityOwner:
    capability: RejectionCapability
    lease: DetectorCandidateRejectionLease
    detector: DetectorRuntime
    runtime_identity: _AsrRuntimeIdentity
    revoked: bool = False


@dataclass(slots=True)
class _AdmissionFinalContext:
    turn_token: VoiceTurnToken
    final_key: FinalKey
    epoch: int
    provider: str
    provider_key: ProviderUtteranceKey | None
    lifecycle: VoiceInputLifecycleController
    detector: DetectorRuntime | None
    correlator: ProviderTurnCorrelator | None
    sealed_token: VoiceTransportToken
    provider_fence: ProviderCandidateFence | None
    runtime_identity: _AsrRuntimeIdentity
    has_pending_turn: bool
    settled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass(slots=True)
class _AdmissionResolutionExecution:
    ticket: AdmissionResolutionTicket
    core_settled: bool = False
    transport_settled: bool = False
    lifecycle_settled: bool = False
    core_resolution_succeeded: bool | None = None
    late_context: _AdmissionFinalContext | None = None
    owner_done: bool = False
    settled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


@dataclass(slots=True)
class _AdmissionRejectionExecution:
    ticket: AdmissionOperationTicket
    absolute_deadline: float | None
    deadline_changed: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
    )


@dataclass(frozen=True, slots=True)
class _ProviderTurnSealTransaction:
    lifecycle: VoiceInputLifecycleController
    turn_token: VoiceTurnToken
    sealed_token: VoiceTransportToken
    final_key: FinalKey
    identity: _AsrRuntimeIdentity


class IndependentAsrRuntime:
    """Own one independent ASR session without reading Core manager state."""

    def __init__(self, callbacks: AsrRuntimeCallbacks) -> None:
        self._callbacks = callbacks
        self._init_asr_runtime_state()

    @property
    def display_name(self) -> str:
        return self._callbacks.display_name()

    async def close(self) -> None:
        """Permanently dispose this runtime and its admission ingress."""

        self._ensure_asr_runtime_state()
        close_task = self._asr_terminal_close_task
        if close_task is None:
            self._asr_terminal_close_requested = True
            self._begin_asr_start_operation()
            close_task = asyncio.create_task(
                self._finish_terminal_asr_close(),
                name="independent-asr-terminal-close",
            )
            # stop_session() snapshots the ordinary owned-cleanup registry, so
            # the terminal owner must stay outside it to avoid waiting itself.
            close_task.add_done_callback(self._log_asr_background_task_failure)
            self._asr_terminal_close_task = close_task
        await asyncio.shield(close_task)

    async def stop_session(self) -> None:
        """Stop one ASR session while keeping the admission lane reusable."""

        self._ensure_asr_runtime_state()
        close_task = self._asr_runtime_close_task
        if close_task is None:
            # A session stop owns a different operation from start's detached
            # predecessor cleanup. Invalidate the in-flight start before
            # awaiting either cleanup, then wait for both under one explicit
            # latch so cancellation/retry retains the same owner.
            operation_generation = self._begin_asr_start_operation()
            predecessor_cleanups = tuple(self._asr_owned_cleanup_tasks)
            cleanup = self._detach_independent_asr(
                operation_generation=operation_generation,
            )
            cleanup_task = (
                self._schedule_owned_cleanup(
                    cleanup,
                    name="independent-asr-stop-session-detached",
                )
                if cleanup is not None
                else None
            )
            close_task = self._schedule_owned_cleanup(
                self._finish_explicit_asr_close(
                    predecessor_cleanups,
                    cleanup_task,
                ),
                name="independent-asr-stop-session",
            )
            self._asr_runtime_close_task = close_task
        await asyncio.shield(close_task)

    async def _finish_terminal_asr_close(self) -> None:
        """Bounded terminal drain followed by permanent hard-close fences."""

        deadline = time.monotonic() + _ASR_TERMINAL_CLOSE_TIMEOUT_SECONDS
        drain_deadline = deadline - _ASR_TERMINAL_HARD_CLOSE_RESERVE_SECONDS
        stop_waiter = asyncio.create_task(
            self.stop_session(),
            name="independent-asr-terminal-stop-session",
        )
        stop_pending = await self._bounded_terminal_task_join(
            {stop_waiter},
            deadline=drain_deadline,
            label="session stop",
            cancel_first=False,
        )
        if (
            stop_pending
            or stop_waiter.cancelled()
            or any(not task.done() for task in self._asr_owned_cleanup_tasks)
        ):
            owned_cleanups = set(self._asr_owned_cleanup_tasks)
            await self._bounded_terminal_task_join(
                owned_cleanups,
                deadline=drain_deadline,
                label="owned cleanup",
                cancel_first=True,
            )

        await self._quiesce_terminal_admission_tasks(drain_deadline)

        transcript_dispatcher = self._asr_transcript_dispatcher
        detector_dispatcher = self._asr_detector_dispatcher
        audio_dispatcher = self._asr_audio_dispatcher
        transcript_dispatcher.invalidate_all()
        detector_dispatcher.invalidate_all()
        audio_dispatcher.abort()
        dispatcher_closes = {
            asyncio.create_task(
                detector_dispatcher.close(),
                name="independent-asr-terminal-detector-dispatcher-close",
            ),
            asyncio.create_task(
                audio_dispatcher.close(),
                name="independent-asr-terminal-audio-dispatcher-close",
            ),
        }
        self._track_terminal_close_tasks(dispatcher_closes)
        await self._bounded_terminal_task_join(
            dispatcher_closes,
            deadline=deadline,
            label="idle dispatcher close",
            cancel_first=False,
            cancel_on_timeout=False,
        )

        async def close_speaker_factory() -> None:
            async with self._speaker_verifier_lock:
                factory = self._speaker_verifier_factory
                self._speaker_verifier_factory = None
                self._speaker_verifier_activation_generation = None
                self._speaker_verifier_degraded = False
                if factory is not None:
                    self._close_speaker_verifier_factory(factory)

        speaker_close = asyncio.create_task(
            close_speaker_factory(),
            name="independent-asr-terminal-speaker-factory-close",
        )
        self._track_terminal_close_tasks({speaker_close})
        await self._bounded_terminal_task_join(
            {speaker_close},
            deadline=deadline,
            label="speaker factory close",
            cancel_first=False,
            cancel_on_timeout=False,
        )

        # A settlement may have scheduled one final producer before observing
        # the terminal generation. Re-snapshot without consuming hard-close
        # reserve, then publish the lane's permanent closing fence.
        await self._quiesce_terminal_admission_tasks(drain_deadline)
        if self._asr_admission_ingress_started:
            ingress_close = asyncio.create_task(
                self._asr_admission_ingress.close(),
                name="independent-asr-terminal-admission-ingress-close",
            )
            self._track_terminal_close_tasks({ingress_close})

            def finish_ingress_close(done: asyncio.Task[Any]) -> None:
                if not done.cancelled() and done.exception() is None:
                    self._asr_admission_ingress_started = False

            ingress_close.add_done_callback(finish_ingress_close)
            ingress_pending = await self._bounded_terminal_task_join(
                {ingress_close},
                deadline=deadline,
                label="admission ingress close",
                cancel_first=False,
                cancel_on_timeout=False,
            )
            if (
                not ingress_pending
                and not ingress_close.cancelled()
                and ingress_close.exception() is None
            ):
                self._asr_admission_ingress_started = False
            else:
                logger.warning(
                    "[%s] admission ingress terminal owner remains active",
                    self.display_name,
                )

    def _track_terminal_close_tasks(
        self,
        tasks: set[asyncio.Task[Any]],
    ) -> None:
        """Retain timed-out hard-close owners until their actual completion."""

        for task in tasks:
            self._asr_close_tasks.add(task)

            def reap(done: asyncio.Task[Any]) -> None:
                self._asr_close_tasks.discard(done)
                self._log_asr_background_task_failure(done)

            task.add_done_callback(reap)

    async def _bounded_terminal_task_join(
        self,
        tasks: set[asyncio.Task[Any]],
        *,
        deadline: float,
        label: str,
        cancel_first: bool,
        cancel_on_timeout: bool = True,
    ) -> set[asyncio.Task[Any]]:
        """Join one owned task set within the absolute terminal deadline."""

        current = asyncio.current_task()
        pending = {
            task for task in tasks if task is not current and not task.done()
        }
        if not pending:
            return set()
        owned = set(pending)
        if cancel_first:
            for task in pending:
                if task not in self._asr_terminal_cancel_requested_tasks:
                    self._asr_terminal_cancel_requested_tasks.add(task)
                    task.cancel()

        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            _, pending = await asyncio.wait(
                pending,
                timeout=min(
                    remaining / 2,
                    _ASR_TERMINAL_CLOSE_JOIN_SLICE_SECONDS,
                ),
            )
        else:
            # This schedules fresh hard-close tasks once so their synchronous
            # ownership fence can publish before timeout cancellation.
            _, pending = await asyncio.wait(pending, timeout=0)
        if pending and not cancel_first and cancel_on_timeout:
            for task in pending:
                if task not in self._asr_terminal_cancel_requested_tasks:
                    self._asr_terminal_cancel_requested_tasks.add(task)
                    task.cancel()
        remaining = max(0.0, deadline - time.monotonic())
        if pending and remaining > 0:
            _, pending = await asyncio.wait(
                pending,
                timeout=min(
                    remaining,
                    _ASR_TERMINAL_CLOSE_JOIN_SLICE_SECONDS,
                ),
            )
        elif pending:
            _, pending = await asyncio.wait(pending, timeout=0)
        if pending:
            logger.warning(
                "[%s] independent ASR terminal %s exceeded %.1fs deadline",
                self.display_name,
                label,
                _ASR_TERMINAL_CLOSE_TIMEOUT_SECONDS,
            )
        for task in owned - pending:
            self._log_asr_background_task_failure(task)
        return pending

    async def _quiesce_terminal_admission_tasks(self, deadline: float) -> None:
        """Cancel producers, then allow settlement owners to finish."""

        current = asyncio.current_task()
        producers = {
            task
            for task in (
                *tuple(self._asr_admission_candidate_owned_tasks),
                *tuple(self._asr_admission_deadline_tasks.values()),
            )
            if task is not current and not task.done()
        }
        pending_producers = await self._bounded_terminal_task_join(
            producers,
            deadline=deadline,
            label="admission producer drain",
            cancel_first=True,
        )
        completed_producers = producers - pending_producers
        self._asr_admission_candidate_owned_tasks.difference_update(
            completed_producers
        )
        for candidate, task in tuple(self._asr_admission_candidate_tasks.items()):
            if task in completed_producers:
                self._asr_admission_candidate_tasks.pop(candidate, None)
        for ticket, task in tuple(self._asr_admission_deadline_tasks.items()):
            if task in completed_producers:
                self._asr_admission_deadline_tasks.pop(ticket, None)

        settlements = {
            task
            for task in tuple(self._asr_admission_effect_tasks)
            if task is not current and not task.done()
        }
        pending_settlements = await self._bounded_terminal_task_join(
            settlements,
            deadline=deadline,
            label="admission settlement drain",
            cancel_first=False,
        )
        completed_settlements = settlements - pending_settlements
        self._asr_admission_effect_tasks.difference_update(completed_settlements)
        for task in completed_settlements:
            self._asr_admission_effect_task_turns.pop(task, None)

        late_producers = {
            task
            for task in (
                *tuple(self._asr_admission_candidate_owned_tasks),
                *tuple(self._asr_admission_deadline_tasks.values()),
            )
            if task is not current and not task.done()
        }
        pending_late_producers = await self._bounded_terminal_task_join(
            late_producers,
            deadline=deadline,
            label="late admission producer drain",
            cancel_first=True,
        )
        completed_late_producers = late_producers - pending_late_producers
        self._asr_admission_candidate_owned_tasks.difference_update(
            completed_late_producers
        )
        for candidate, task in tuple(self._asr_admission_candidate_tasks.items()):
            if task in completed_late_producers:
                self._asr_admission_candidate_tasks.pop(candidate, None)
        for ticket, task in tuple(self._asr_admission_deadline_tasks.items()):
            if task in completed_late_producers:
                self._asr_admission_deadline_tasks.pop(ticket, None)

    @staticmethod
    async def _finish_explicit_asr_close(
        predecessor_cleanups: tuple[asyncio.Task[Any], ...],
        cleanup_task: "asyncio.Task[Any] | None",
    ) -> None:
        """Join both teardowns; ``cleanup_task`` is already running."""

        if predecessor_cleanups:
            await asyncio.gather(
                *predecessor_cleanups,
                return_exceptions=True,
            )
        if cleanup_task is not None:
            # Awaited last but NOT started last, and awaited bare so its
            # failure still reaches the owned-cleanup logger.
            await cleanup_task

    async def set_speaker_verifier_factory(
        self,
        factory: SpeakerShadowFactory | None,
        *,
        activation_generation: str,
    ) -> bool:
        """Hot-replace Owner verification without restarting independent ASR."""

        if factory is not None and not callable(factory):
            raise TypeError("factory must be callable or None")
        if type(activation_generation) is not str or not activation_generation.strip():
            raise ValueError("activation_generation must be a non-empty string")
        self._ensure_asr_runtime_state()
        async with self._speaker_verifier_lock:
            if self._asr_terminal_close_requested:
                if factory is not None:
                    return False
                old_factory = self._speaker_verifier_factory
                self._speaker_verifier_factory = None
                self._speaker_verifier_activation_generation = activation_generation
                self._speaker_verifier_enforces_admission = False
                self._speaker_verifier_degraded = False
                if old_factory is not None:
                    self._close_speaker_verifier_factory(old_factory)
                return True
            return await self._set_speaker_verifier_factory_locked(
                factory,
                activation_generation=activation_generation,
            )

    async def _set_speaker_verifier_factory_locked(
        self,
        factory: SpeakerShadowFactory | None,
        *,
        activation_generation: str,
    ) -> bool:
        old_factory = self._speaker_verifier_factory
        if (
            factory is old_factory
            and activation_generation == self._speaker_verifier_activation_generation
            and not self._speaker_verifier_degraded
        ):
            return True

        # Revocation is a logical authority barrier, not a cleanup result.
        # Publish it before yielding so every callback from the old observer
        # becomes stale even if physical detector replacement later fails.
        revoking = factory is None
        if revoking:
            self._speaker_verifier_factory = None
            self._speaker_verifier_activation_generation = activation_generation
            self._speaker_verifier_enforces_admission = False
            if old_factory is not None:
                self._close_speaker_verifier_factory(old_factory)

        async def fence_failed_replacement() -> None:
            self._speaker_verifier_factory = None
            self._speaker_verifier_enforces_admission = False
            self._speaker_verifier_degraded = True
            if old_factory is not None and old_factory is not factory:
                self._close_speaker_verifier_factory(old_factory)
            await self._revoke_runtime_speaker_authority_for_verifier_change()

        detector = self._asr_detector
        if detector is not None:
            new_shadow: SpeakerShadowObserver | None = None
            if factory is not None:
                try:
                    new_shadow = factory()
                except Exception:
                    return False
                if new_shadow is None:
                    return False
                # Publish the new callback identity before Detector can install
                # the observer. replace_speaker_verifier transfers ownership at
                # call entry and may yield while closing the detached observer.
                self._speaker_verifier_activation_generation = (
                    activation_generation
                )
                self._speaker_verifier_enforces_admission = (
                    _speaker_factory_enforces_admission(factory)
                )
            await self._revoke_runtime_speaker_authority_for_verifier_change()
            try:
                await detector.replace_speaker_verifier(
                    new_shadow,
                    owner_generation=activation_generation,
                )
            except asyncio.CancelledError:
                if not revoking:
                    cleanup = asyncio.create_task(
                        fence_failed_replacement(),
                        name="speaker-verifier-replacement-cancel-fence",
                    )
                    await asyncio.shield(cleanup)
                raise
            except Exception:
                if not revoking:
                    await fence_failed_replacement()
                return False
            if self._asr_detector is not detector:
                # The detached detector owns and closes ``new_shadow``. Apply
                # the same activation to the replacement, if one appeared.
                replacement = self._asr_detector
                if replacement is not None:
                    replacement_shadow: SpeakerShadowObserver | None = None
                    if factory is not None:
                        try:
                            replacement_shadow = factory()
                        except Exception:
                            if not revoking:
                                await fence_failed_replacement()
                            return False
                        if replacement_shadow is None:
                            if not revoking:
                                await fence_failed_replacement()
                            return False
                    try:
                        await replacement.replace_speaker_verifier(
                            replacement_shadow,
                            owner_generation=activation_generation,
                        )
                    except asyncio.CancelledError:
                        if not revoking:
                            cleanup = asyncio.create_task(
                                fence_failed_replacement(),
                                name=(
                                    "speaker-verifier-replacement-"
                                    "cancel-fence"
                                ),
                            )
                            await asyncio.shield(cleanup)
                        raise
                    except Exception:
                        if not revoking:
                            await fence_failed_replacement()
                        return False
                    if self._asr_detector is not replacement:
                        if not revoking:
                            await fence_failed_replacement()
                        return False
        else:
            if not revoking:
                self._speaker_verifier_activation_generation = (
                    activation_generation
                )
                self._speaker_verifier_enforces_admission = (
                    _speaker_factory_enforces_admission(factory)
                )
            await self._revoke_runtime_speaker_authority_for_verifier_change()

        if self._asr_terminal_close_requested:
            self._speaker_verifier_factory = None
            self._speaker_verifier_activation_generation = None
            self._speaker_verifier_enforces_admission = False
            self._speaker_verifier_degraded = False
            if old_factory is not None and not revoking:
                self._close_speaker_verifier_factory(old_factory)
            return revoking
        if not revoking:
            self._speaker_verifier_factory = factory
            self._speaker_verifier_activation_generation = activation_generation
            self._speaker_verifier_enforces_admission = (
                _speaker_factory_enforces_admission(factory)
            )
            if old_factory is not None and old_factory is not factory:
                self._close_speaker_verifier_factory(old_factory)
        self._speaker_verifier_degraded = False
        return True

    async def _revoke_runtime_speaker_authority_for_verifier_change(self) -> None:
        """Fence every optional speaker capability without revising text."""

        if not self._speaker_verifier_degraded:
            self._speaker_verifier_health_generation += 1
        self._speaker_verifier_degraded = True
        candidate_turns = tuple(self._asr_admission_candidate_turns.items())
        pending_turns = tuple(
            self._asr_speaker_authority_pending_turns.items()
        )
        if self._asr_admission_ingress_started:
            for candidate, turn_token in candidate_turns:
                try:
                    future = self._asr_admission_ingress.post_nowait(
                        turn_token,
                        SpeakerAuthorityUnavailable(candidate),
                    )
                except (AdmissionIngressClosedError, AdmissionIngressCapacityError):
                    continue
                self._consume_admission_future(turn_token, future)
            for turn_token, owner_generation in pending_turns:
                try:
                    future = self._asr_admission_ingress.post_nowait(
                        turn_token,
                        SpeakerAuthorityUnarmed(owner_generation),
                    )
                except (AdmissionIngressClosedError, AdmissionIngressCapacityError):
                    continue
                self._consume_admission_future(turn_token, future)
        self._asr_admission_capability_generation += 1
        for owner in self._asr_admission_capabilities.values():
            owner.revoked = True
        self._asr_admission_capabilities.clear()
        self._asr_admission_candidate_turns.clear()
        self._asr_speaker_authority_pending_turns.clear()
        for turn_token in await self._asr_admission.live_turn_tokens():
            try:
                await self._post_admission_event(
                    turn_token,
                    BoundaryUnknown(),
                )
            except (AdmissionIngressClosedError, KeyError):
                continue

    def _accept_speaker_candidate_binding(
        self,
        candidate: SpeakerShadowCandidateKey,
        turn_token: VoiceTurnToken,
        *,
        detector: DetectorRuntime,
        activation_generation: str,
    ) -> bool:
        """Publish one stable candidate lease before any score can arrive."""

        lifecycle = self._asr_lifecycle
        sealed_token = self._asr_sealed_turn_token
        turn_is_current = bool(
            lifecycle is not None
            and lifecycle.snapshot.state
            in {VoiceLifecycleState.ACTIVE, VoiceLifecycleState.DRAINING}
            and (
                lifecycle.current_turn_token == turn_token
                or (
                    sealed_token is not None
                    and sealed_token.turn == turn_token
                )
            )
        )
        if (
            self._asr_terminal_close_requested
            or not self._speaker_verifier_enforces_admission
            or detector is not self._asr_detector
            or not self._asr_admission_ingress_started
            or activation_generation
            != self._speaker_verifier_activation_generation
            or candidate.detector_epoch != detector.detector_epoch
            or not self._ingress_token_matches(turn_token.ingress)
            or not turn_is_current
        ):
            return False
        existing = self._asr_admission_candidate_turns.get(candidate)
        if existing is not None:
            if existing != turn_token:
                self._speaker_rejection_metrics[
                    "admission_candidate_alias_conflict_count"
                ] += 1
                return False
            return True
        self._asr_admission_candidate_turns[candidate] = turn_token
        self._asr_speaker_authoritative_turns.add(turn_token)
        self._asr_speaker_authority_pending_turns[turn_token] = (
            activation_generation
        )
        try:
            pending = self._asr_admission_ingress.post_nowait(
                turn_token,
                SpeakerAuthorityPending(activation_generation),
            )
        except (AdmissionIngressClosedError, AdmissionIngressCapacityError):
            if self._asr_admission_candidate_turns.get(candidate) == turn_token:
                self._asr_admission_candidate_turns.pop(candidate, None)
            if (
                self._asr_speaker_authority_pending_turns.get(turn_token)
                == activation_generation
            ):
                self._asr_speaker_authority_pending_turns.pop(turn_token, None)
            return False
        self._consume_admission_future(turn_token, pending)
        try:
            future = self._asr_admission_ingress.post_nowait(
                turn_token,
                CandidateBound(candidate, activation_generation),
            )
        except (AdmissionIngressClosedError, AdmissionIngressCapacityError):
            if self._asr_admission_candidate_turns.get(candidate) == turn_token:
                self._asr_admission_candidate_turns.pop(candidate, None)
            if (
                self._asr_speaker_authority_pending_turns.get(turn_token)
                == activation_generation
            ):
                self._asr_speaker_authority_pending_turns.pop(turn_token, None)
            try:
                unarmed = self._asr_admission_ingress.post_nowait(
                    turn_token,
                    SpeakerAuthorityUnarmed(activation_generation),
                )
            except (AdmissionIngressClosedError, AdmissionIngressCapacityError):
                return False
            self._consume_admission_future(turn_token, unarmed)
            return False
        self._consume_admission_future(turn_token, future)
        if (
            self._asr_speaker_authority_pending_turns.get(turn_token)
            == activation_generation
        ):
            self._asr_speaker_authority_pending_turns.pop(turn_token, None)
        self._schedule_speaker_admission_item(
            candidate,
            self._ensure_speaker_admission_capability(
                candidate,
                turn_token,
                activation_generation,
            ),
        )
        return True

    def _accept_speaker_evidence_fact(
        self,
        fact: SpeakerLow | SpeakerHigh | SpeakerUnavailable,
        *,
        activation_generation: str,
        enforce: bool,
    ) -> bool:
        """Queue one ordered speaker fact; only the coordinator may resolve final."""

        self._ensure_asr_runtime_state()
        if self._asr_terminal_close_requested:
            return False
        if (
            not isinstance(fact, (SpeakerLow, SpeakerHigh, SpeakerUnavailable))
            or activation_generation
            != self._speaker_verifier_activation_generation
            or not self._speaker_verifier_enforces_admission
            or type(enforce) is not bool
        ):
            return False
        if not enforce:
            return True
        detector = self._asr_detector
        if detector is None or not self._asr_admission_ingress_started:
            return False
        candidate = fact.candidate
        turn_token = self._asr_admission_candidate_turns.get(candidate)
        if turn_token is None:
            turn_token = detector._bound_turn_token_for_speaker_candidate(
                candidate
            )
            if turn_token is None or not self._accept_speaker_candidate_binding(
                candidate,
                turn_token,
                detector=detector,
                activation_generation=activation_generation,
            ):
                return False
        try:
            future = self._asr_admission_ingress.post_nowait(turn_token, fact)
        except AdmissionIngressClosedError:
            return False
        self._consume_admission_future(turn_token, future)
        return True

    def _close_speaker_evidence(
        self,
        closed: CaptureClosed,
        *,
        activation_generation: str,
        enforce: bool,
        evidence_complete: bool,
    ) -> bool:
        """Queue capture close behind every observation for this candidate."""

        self._ensure_asr_runtime_state()
        if (
            type(closed) is not CaptureClosed
            or activation_generation
            != self._speaker_verifier_activation_generation
            or not self._speaker_verifier_enforces_admission
            or type(enforce) is not bool
            or type(evidence_complete) is not bool
        ):
            return False
        if not enforce:
            return True
        turn_token = self._asr_admission_candidate_turns.get(closed.candidate)
        if turn_token is None or not self._asr_admission_ingress_started:
            return False
        try:
            future = self._asr_admission_ingress.post_nowait(turn_token, closed)
        except AdmissionIngressClosedError:
            return False
        self._consume_admission_future(turn_token, future)
        detector = self._asr_detector
        if detector is not None:
            detector.release_speaker_candidate_binding(
                closed.candidate,
                turn_token,
            )
        if self._asr_admission_candidate_turns.get(closed.candidate) == turn_token:
            self._asr_admission_candidate_turns.pop(closed.candidate, None)
        return True

    def _consume_admission_future(
        self,
        turn_token: VoiceTurnToken,
        future: asyncio.Future[tuple[AdmissionEffect, ...]],
        *,
        suppress_terminal_errors: bool = True,
    ) -> asyncio.Task[tuple[AdmissionEffect, ...]]:
        """Execute effects owned by one synchronously queued ingress item."""

        async def consume() -> tuple[AdmissionEffect, ...]:
            try:
                effects = await asyncio.shield(future)
            except (AdmissionIngressClosedError, KeyError):
                if suppress_terminal_errors:
                    return ()
                raise
            async def execute_effects() -> None:
                for effect in effects:
                    await self._execute_admission_effect(effect)

            if effects:
                task = asyncio.create_task(
                    execute_effects(),
                    name="voice-turn-admission-effects",
                )
                self._track_admission_effect_task(task, turn_token)
                task.add_done_callback(self._admission_effect_done)
            try:
                await self._asr_admission_ingress.retire_turn(turn_token)
            except AdmissionIngressClosedError:
                if not suppress_terminal_errors:
                    raise
            return effects

        task = asyncio.create_task(
            consume(),
            name=f"voice-turn-admission-ingress-{turn_token.turn_id}",
        )
        self._track_admission_effect_task(task, turn_token)
        task.add_done_callback(self._admission_effect_done)
        return task

    async def _ensure_speaker_admission_capability(
        self,
        candidate: SpeakerShadowCandidateKey,
        turn_token: VoiceTurnToken,
        activation_generation: str,
    ) -> None:
        if activation_generation != self._speaker_verifier_activation_generation:
            return
        detector = self._asr_detector
        if detector is None:
            return
        try:
            lease = await detector.prepare_candidate_rejection(candidate)
        except asyncio.CancelledError:
            raise
        except Exception:
            lease = None
        if lease is None or lease.turn_token != turn_token:
            return
        if candidate.scope == "smart_turn_turn":
            capability = self._register_admission_capability(
                lease,
                kind=RejectionCapabilityKind.ACTIVE,
            )
            if capability is not None:
                await self._post_admission_event(
                    turn_token,
                    BoundaryExact(capability),
                )

    def _mark_speaker_evidence_backend_degraded(
        self,
        *,
        activation_generation: str,
    ) -> None:
        if activation_generation != self._speaker_verifier_activation_generation:
            return
        self._mark_speaker_verifier_degraded()

    def _mark_speaker_evidence_backend_healthy(
        self,
        *,
        activation_generation: str,
    ) -> None:
        if activation_generation != self._speaker_verifier_activation_generation:
            return
        self._mark_speaker_verifier_healthy()

    def _schedule_speaker_admission_item(
        self,
        candidate: SpeakerShadowCandidateKey,
        awaitable: Awaitable[None],
    ) -> bool:
        if self._asr_terminal_close_requested:
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if hasattr(awaitable, "close"):
                awaitable.close()  # type: ignore[attr-defined]
            return False
        predecessor = self._asr_admission_candidate_tasks.get(candidate)

        async def run() -> None:
            if predecessor is not None:
                try:
                    await asyncio.shield(predecessor)
                except (asyncio.CancelledError, Exception):
                    pass
            await awaitable

        task = loop.create_task(run(), name="voice-turn-speaker-admission")
        self._asr_admission_candidate_tasks[candidate] = task
        self._asr_admission_candidate_owned_tasks.add(task)

        def reap(done: asyncio.Task[None]) -> None:
            self._asr_admission_candidate_owned_tasks.discard(done)
            if self._asr_admission_candidate_tasks.get(candidate) is done:
                self._asr_admission_candidate_tasks.pop(candidate, None)
            self._log_asr_background_task_failure(done)

        task.add_done_callback(reap)
        return True

    def _speaker_verifier_diagnostics(self) -> dict[str, int]:
        """Return aggregate-only verifier diagnostics for local debugging."""

        metrics = dict(getattr(self, "_speaker_rejection_metrics", {}))
        factory = getattr(self, "_speaker_verifier_factory", None)
        snapshot = getattr(factory, "diagnostics_snapshot", None)
        if callable(snapshot):
            try:
                factory_metrics = snapshot()
            except Exception:
                factory_metrics = {}
            if not isinstance(factory_metrics, dict):
                factory_metrics = {}
            for name, value in factory_metrics.items():
                if type(name) is str and type(value) is int and value >= 0:
                    metrics[name] = value
        detector = getattr(self, "_asr_detector", None)
        detector_snapshot = getattr(
            detector,
            "speaker_rejection_diagnostics_snapshot",
            None,
        )
        if callable(detector_snapshot):
            try:
                detector_metrics = detector_snapshot()
            except Exception:
                detector_metrics = {}
            if isinstance(detector_metrics, dict):
                for name, value in detector_metrics.items():
                    if type(name) is str and type(value) is int and value >= 0:
                        metrics[name] = value
        metrics["verifier_installed_count"] = int(factory is not None)
        metrics["verifier_degraded_count"] = int(
            bool(getattr(self, "_speaker_verifier_degraded", False))
        )
        metrics["rejection_task_pending_count"] = len(
            getattr(self, "_asr_admission_rejection_executions", ())
        )
        metrics["rejection_in_progress_count"] = int(
            bool(getattr(self, "_asr_admission_rejection_executions", ()))
        )
        return metrics

    def _mark_speaker_verifier_degraded(
        self,
        *,
        preserve_reject_requested: bool = False,
    ) -> None:
        """Record backend health; ordered UNAVAILABLE facts revoke authority."""

        del preserve_reject_requested
        self._ensure_asr_runtime_state()
        if not self._speaker_verifier_degraded:
            self._speaker_verifier_health_generation += 1
        self._speaker_verifier_degraded = True

    def _mark_speaker_verifier_healthy(self) -> None:
        """Clear transient Owner verifier health degradation after recovery."""

        self._ensure_asr_runtime_state()
        if self._speaker_verifier_degraded:
            self._speaker_verifier_health_generation += 1
        self._speaker_verifier_degraded = False

    @staticmethod
    def _close_speaker_verifier_factory(factory: SpeakerShadowFactory) -> None:
        close = getattr(factory, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            return

    def _begin_asr_start_operation(self) -> int:
        self._asr_start_generation += 1
        return self._asr_start_generation

    def _asr_start_operation_matches(self, operation_generation: int) -> bool:
        return operation_generation == self._asr_start_generation

    def _invalidate_asr_start(self) -> None:
        self._begin_asr_start_operation()

    def capture_ingress_token(
        self,
        *,
        connection_id: str,
        lease_generation: int,
        route_generation: int,
    ) -> VoiceIngressToken:
        return VoiceIngressToken(
            session_epoch=self._asr_session_epoch,
            connection_id=connection_id,
            lease_generation=lease_generation,
            route_generation=route_generation,
            audio_generation=self._asr_audio_generation,
        )

    async def suspend(self, reason: str) -> None:
        lifecycle = self._asr_lifecycle
        if lifecycle is not None and lifecycle.snapshot.state not in {
            VoiceLifecycleState.OFF,
            VoiceLifecycleState.BLOCKED,
            VoiceLifecycleState.SUSPENDED,
        }:
            lifecycle.transition(VoiceLifecycleEvent.GAME_TAKEOVER)
        await self.abort(reason)

    async def resume(self, reason: str) -> None:
        del reason
        epoch = self._asr_session_epoch
        provider = self._asr_provider or "unknown"
        lifecycle = self._asr_lifecycle
        if lifecycle is not None and (
            lifecycle.snapshot.state is VoiceLifecycleState.SUSPENDED
        ):
            lifecycle.transition(VoiceLifecycleEvent.GAME_RELEASED)
            identity = self._capture_runtime_identity()
            await self._send_asr_lifecycle_state(
                lifecycle.snapshot.state,
                provider=provider,
                session_epoch=epoch,
                expected_identity=identity,
            )

    def _asr_runtime_refs_match(
        self,
        epoch: int,
        lifecycle: VoiceInputLifecycleController | None,
        detector: DetectorRuntime | None,
    ) -> bool:
        return bool(
            epoch == self._asr_session_epoch
            and self._asr_lifecycle is lifecycle
            and self._asr_detector is detector
        )

    def _capture_runtime_identity(
        self,
        *,
        ingress_token: VoiceIngressToken | None = None,
        turn_token: VoiceTurnToken | None = None,
    ) -> _AsrRuntimeIdentity:
        lifecycle = self._asr_lifecycle
        return _AsrRuntimeIdentity(
            start_generation=self._asr_start_generation,
            session_epoch=self._asr_session_epoch,
            audio_generation=self._asr_audio_generation,
            lifecycle=lifecycle,
            transport_generation=(
                lifecycle.snapshot.transport_generation
                if lifecycle is not None
                else None
            ),
            detector=self._asr_detector,
            session=self._asr_session,
            provider=self._asr_provider,
            session_factory=self._asr_session_factory,
            transport_selection=self._asr_transport_selection,
            transport_task=self._asr_transport_task,
            ingress_token=ingress_token,
            turn_token=turn_token,
        )

    def _runtime_identity_matches(
        self,
        identity: _AsrRuntimeIdentity,
    ) -> bool:
        lifecycle = self._asr_lifecycle
        if (
            identity.start_generation != self._asr_start_generation
            or identity.session_epoch != self._asr_session_epoch
            or identity.audio_generation != self._asr_audio_generation
            or lifecycle is not identity.lifecycle
            or self._asr_detector is not identity.detector
            or self._asr_session is not identity.session
            or self._asr_provider != identity.provider
            or self._asr_session_factory is not identity.session_factory
            or self._asr_transport_selection is not identity.transport_selection
            or self._asr_transport_task is not identity.transport_task
        ):
            return False
        transport_generation = (
            lifecycle.snapshot.transport_generation if lifecycle is not None else None
        )
        if transport_generation != identity.transport_generation:
            return False
        if identity.ingress_token is not None and (
            self._asr_current_ingress_token != identity.ingress_token
            or not self._ingress_token_matches(identity.ingress_token)
        ):
            return False
        if identity.turn_token is not None and (
            lifecycle is None
            or identity.turn_token.ingress != identity.ingress_token
            or lifecycle.snapshot.turn_id != identity.turn_token.turn_id
        ):
            return False
        return True

    async def abort(self, reason: str) -> None:
        if reason == "ingress_backpressure":
            token = self._asr_current_ingress_token
            if token is not None and self._ingress_token_matches(token):
                await self._handle_audio_ingress_backpressure(token)
                return
        epoch = self._asr_session_epoch
        lifecycle = self._asr_lifecycle
        detector = self._asr_detector
        provider = self._asr_provider or "unknown"
        if lifecycle is not None:
            lifecycle.invalidate_audio()
        post_detach = await self._abort_transport(reason)
        if not self._runtime_identity_matches(
            post_detach
        ) or not self._asr_runtime_refs_match(epoch, lifecycle, detector):
            return
        if reason == "ingress_backpressure":
            await self._send_asr_status(
                "ASR_INGRESS_BACKPRESSURE",
                provider,
                session_epoch=epoch,
                expected_identity=post_detach,
            )
        if detector is not None:
            try:
                await detector.reset()
            except Exception:
                logger.warning(
                    "[%s] detector reset failed during voice abort",
                    self.display_name,
                )
            if not self._runtime_identity_matches(
                post_detach
            ) or not self._asr_runtime_refs_match(epoch, lifecycle, detector):
                return
        if lifecycle is not None:
            await self._send_asr_lifecycle_state(
                lifecycle.snapshot.state,
                provider=provider,
                session_epoch=epoch,
                expected_identity=post_detach,
            )

    async def wait_transcript_idle(self) -> None:
        await self._asr_transcript_dispatcher.wait_idle()

    def has_pending_transcript_delivery(self) -> bool:
        """Return whether an accepted final has not finished Core dispatch."""

        return self._asr_transcript_dispatcher.has_pending_delivery

    def _init_asr_runtime_state(self) -> None:
        self._asr_session = None
        self._asr_session_epoch = 0
        self._asr_start_generation = 0
        self._asr_provider = None
        self._asr_turn_prepared = False
        self._asr_final_lock = asyncio.Lock()
        self._asr_admission = VoiceTurnAdmissionCoordinator(capacity=8)
        self._asr_admission_ingress = AdmissionIngressLane(
            self._asr_admission,
            data_capacity=64,
        )
        self._asr_admission_ingress_started = False
        self._asr_admission_capability_sequence = 0
        self._asr_admission_capability_generation = 0
        self._asr_admission_capabilities: dict[
            int,
            _AdmissionCapabilityOwner,
        ] = {}
        self._asr_admission_candidate_turns: dict[
            SpeakerShadowCandidateKey,
            VoiceTurnToken,
        ] = {}
        self._asr_admission_candidate_tasks: dict[
            SpeakerShadowCandidateKey,
            asyncio.Task[None],
        ] = {}
        self._asr_admission_candidate_owned_tasks: set[
            asyncio.Task[None]
        ] = set()
        self._asr_admission_deadline_tasks: dict[
            AdmissionOperationTicket,
            asyncio.Task[None],
        ] = {}
        self._asr_admission_effect_tasks: set[asyncio.Task[Any]] = set()
        self._asr_admission_effect_task_turns: dict[
            asyncio.Task[Any],
            VoiceTurnToken | None,
        ] = {}
        self._asr_admission_rejection_executions: dict[
            AdmissionOperationTicket,
            _AdmissionRejectionExecution,
        ] = {}
        self._asr_admission_rejection_deadlines: dict[
            AdmissionOperationTicket,
            float,
        ] = {}
        self._asr_admission_turn_sealed_events: dict[
            VoiceTurnToken,
            asyncio.Event,
        ] = {}
        self._asr_admission_final_contexts: dict[
            VoiceTurnToken,
            _AdmissionFinalContext,
        ] = {}
        self._asr_admission_resolutions: dict[
            FinalKey,
            _AdmissionResolutionExecution,
        ] = {}
        self._asr_admission_reservation_dispatchers: dict[
            FinalKey,
            TranscriptDispatcher,
        ] = {}
        self._asr_quarantined_partials: dict[VoiceTurnToken, str] = {}
        self._asr_partial_settlements: dict[
            VoiceTurnToken,
            tuple[int, AdmissionDisposition],
        ] = {}
        self._asr_speaker_authority_pending_turns: dict[
            VoiceTurnToken,
            str,
        ] = {}
        self._asr_speaker_authoritative_turns: set[VoiceTurnToken] = set()
        self._asr_speaker_authority_unarming_tasks: dict[
            tuple[VoiceTurnToken, str],
            asyncio.Task[None],
        ] = {}
        self._asr_provider_correlator: ProviderTurnCorrelator | None = None
        self._asr_provider_correlator_namespace: tuple[int, int] | None = None
        self._asr_provider_boundary_proof_sequence = 0
        self._asr_provider_boundary_proofs: dict[
            int,
            ProviderSpeakerBoundarySnapshot,
        ] = {}
        self._asr_audio_bytes = 0
        self._asr_received_audio = False
        self._asr_close_tasks: set[asyncio.Task[None]] = set()
        self._asr_owned_cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._asr_runtime_close_task: asyncio.Task[None] | None = None
        self._asr_terminal_close_requested = False
        self._asr_terminal_close_task: asyncio.Task[None] | None = None
        self._asr_terminal_cancel_requested_tasks: weakref.WeakSet[
            asyncio.Task[Any]
        ] = weakref.WeakSet()
        self._asr_lifecycle: VoiceInputLifecycleController | None = None
        self._asr_detector: DetectorRuntime | None = None
        self._asr_smart_turn_lease: SmartTurnLease | None = None
        self._asr_smart_turn_prepare_lock = asyncio.Lock()
        self._asr_smart_turn_prepare_scope: tuple[int, int, int] | None = None
        self._asr_session_factory = None
        self._asr_transport_selection = None
        self._asr_transport_task: asyncio.Task[None] | None = None
        self._asr_transport_lock = asyncio.Lock()
        self._asr_warm_expiry_task: asyncio.Task[None] | None = None
        self._asr_final_watchdog_task: asyncio.Task[None] | None = None
        self._asr_pending_speech_confirmed = False
        self._asr_pending_speech_onset_at = None
        self._asr_pending_detector_candidate = None
        self._asr_overlap_onset_token: VoiceIngressToken | None = None
        # 重叠发声的真实开口时刻。重放发生在「上一轮延迟 final 到达」之后，比用户
        # 实际开口晚得多；不把这一刻带过去，重放时取到的 onset 会把中间那段全算成
        # 「开口之后」，后继发声在重放前拍的帧就全被排除了。
        self._asr_overlap_onset_at: float | None = None
        self._asr_overlap_completed_token: VoiceIngressToken | None = None
        # 每张 credit 一个开口时刻：多个 onset+pause 周期可以在同一条延迟 final
        # 后面排队，用单个槽位会让所有重放共用最后那个时刻。
        self._asr_overlap_completed_onsets: deque[float] = deque()
        self._asr_overlap_completed_turns = 0
        self._asr_sealed_turn_token: VoiceTransportToken | None = None
        self._asr_provider_candidate_fence: ProviderCandidateFence | None = None
        self._asr_sealed_provider_key: ProviderUtteranceKey | None = None
        # Exact Provider text remains admissible when the advisory Detector
        # seal cannot acquire its lock inside the ordered/final 200 ms budget.
        # This key never grants speaker authority; it only lets the matching
        # transcript complete the already reserved Core turn fail-open.
        self._asr_provider_authority_reset_task: (
            asyncio.Task[bool] | None
        ) = None
        self._asr_provider_exact_session: Any | None = None
        self._asr_audio_sequence = 0
        self._asr_provider_speaker_sequence = 0
        self._asr_buffered_provider_speaker_observation: (
            _BufferedProviderSpeakerObservation | None
        ) = None
        self._asr_audio_generation = 0
        self._asr_current_ingress_token: VoiceIngressToken | None = None
        self._asr_partial_turn_token: VoiceTurnToken | None = None
        self._speaker_verifier_factory: SpeakerShadowFactory | None = None
        self._speaker_verifier_activation_generation: str | None = None
        self._speaker_verifier_enforces_admission = False
        self._speaker_verifier_degraded = False
        self._speaker_verifier_health_generation = 0
        self._speaker_verifier_lock = asyncio.Lock()
        self._speaker_rejection_metrics = _new_speaker_rejection_metrics()
        self._asr_transcript_dispatcher = self._new_asr_transcript_dispatcher()
        self._asr_detector_dispatcher = AsrDetectorDispatcher(
            self._dispatch_asr_detector_event,
            on_failure=self._handle_asr_detector_dispatcher_failure,
        )
        self._asr_audio_dispatcher = AsrAudioDispatcher(
            validator=self._asr_audio_command_is_valid,
            on_wire_audio=self._record_asr_dispatcher_wire_audio,
            on_failure=self._handle_asr_audio_dispatcher_failure,
        )
        self._asr_last_provider_wire_audio_ms = 0
        self._asr_turn_audio_started_at: float | None = None
        # 语义上的「用户开口时刻」。与上面那个的区别是**打点位置**：这个钉在
        # SPEECH_CONFIRMED 转换那一行，不跨 _send_asr_lifecycle_state() 的投递
        # await；上面那个在两条路径上是投递完成之后才打的，喂延迟指标够用，但拿
        # 来当视觉所有权的起点会把投递窗口里拍的帧判成"不属于这段发声"。
        self._asr_turn_onset_at: float | None = None
        # 语音已经检测到、但 ASR session 还没就绪（要等重连）时先把这一刻记下来。
        # 真正 SPEECH_CONFIRMED 要等 connect() 成功之后才发得出去，用那时的时钟
        # 当"用户开口时刻"会把整段重连等待算进去，重连期间拍的帧全被判成不属于
        # 这段发声。
        self._asr_pending_speech_onset_at: float | None = None
        # 上一回合还在排空（DRAINING）时用户就接着说了：pending turn 的真实开口时刻
        # 是 mark_pending_turn_speech() 那一刻，不是后面 begin_pending_turn() 激活的
        # 时刻。lifecycle 硬要求 DRAINING 才能标记，所以这个值必然晚于上一轮封口。
        self._asr_pending_turn_onset_at: float | None = None
        self._asr_turn_endpointed_at: float | None = None
        # 与上面那个一样在封口时刻打点，但**不在 PROVIDER_FINAL 时清掉**。Core 要
        # 到 transcript 派发之后才冻结多模态回合，那时上面那个已经是 None 了；
        # 消费方靠"这个时刻是否晚于本回合起点"排除上一轮的残值。
        self._asr_last_turn_endpointed_at: float | None = None
        # 上面那个保留副本**属于哪一轮**。时间戳分不清"上一轮的封口"和"本轮的封
        # 口"：monotonic 在 Windows 上是 ~15ms 粒度，两者都可能与后继 record 的注
        # 册时刻相等，往任一个方向猜都会错（猜"归上一轮"会丢掉本轮自己的截止点，
        # 猜"归本轮"会把上一轮的封口盖到后继头上）。带上身份就不用猜。
        self._asr_last_turn_endpointed_key: str | None = None
        self._asr_first_partial_recorded = False
        self._voice_input_resource_optimization_enabled = True

    def _schedule_owned_cleanup(
        self,
        awaitable: Awaitable[Any],
        *,
        name: str,
    ) -> asyncio.Task[Any]:
        """Keep teardown running when its caller is cancelled."""

        task = asyncio.create_task(awaitable, name=name)
        self._asr_owned_cleanup_tasks.add(task)
        task.add_done_callback(self._owned_cleanup_done)
        return task

    def _owned_cleanup_done(self, task: asyncio.Task[Any]) -> None:
        self._asr_owned_cleanup_tasks.discard(task)
        self._log_asr_background_task_failure(task)

    def _ensure_asr_runtime_state(self) -> None:
        # A number of focused unit tests intentionally construct the manager via
        # __new__. Keep those narrow lifecycle doubles compatible.
        if not hasattr(self, "_asr_session_epoch"):
            self._init_asr_runtime_state()
        elif not hasattr(self, "_asr_transcript_dispatcher"):
            self._asr_transcript_dispatcher = (
                self._new_asr_transcript_dispatcher()
            )
        if not hasattr(self, "_asr_detector_dispatcher"):
            self._asr_detector_dispatcher = AsrDetectorDispatcher(
                self._dispatch_asr_detector_event,
                on_failure=self._handle_asr_detector_dispatcher_failure,
            )
        if not hasattr(self, "_asr_audio_dispatcher"):
            self._asr_audio_dispatcher = AsrAudioDispatcher(
                validator=self._asr_audio_command_is_valid,
                on_wire_audio=self._record_asr_dispatcher_wire_audio,
                on_failure=self._handle_asr_audio_dispatcher_failure,
            )
            self._asr_audio_sequence = 0
            self._asr_pending_detector_candidate = None
        if not hasattr(self, "_asr_admission"):
            self._asr_admission = VoiceTurnAdmissionCoordinator(capacity=8)
            self._asr_admission_ingress = AdmissionIngressLane(
                self._asr_admission,
                data_capacity=64,
            )
            self._asr_admission_ingress_started = False
            self._asr_admission_capability_sequence = 0
            self._asr_admission_capability_generation = 0
            self._asr_admission_capabilities = {}
            self._asr_admission_candidate_turns = {}
            self._asr_admission_candidate_tasks = {}
            self._asr_admission_candidate_owned_tasks = set()
            self._asr_admission_deadline_tasks = {}
            self._asr_admission_effect_tasks = set()
            self._asr_admission_effect_task_turns = {}
            self._asr_admission_rejection_executions = {}
            self._asr_admission_rejection_deadlines = {}
            self._asr_admission_turn_sealed_events = {}
            self._asr_admission_final_contexts = {}
            self._asr_admission_resolutions = {}
            self._asr_admission_reservation_dispatchers = {}
            self._asr_quarantined_partials = {}
            self._asr_partial_settlements = {}
            self._asr_speaker_authority_pending_turns = {}
            self._asr_speaker_authoritative_turns = set()
            self._asr_speaker_authority_unarming_tasks = {}
            self._asr_provider_correlator = None
            self._asr_provider_correlator_namespace = None
            self._asr_provider_boundary_proof_sequence = 0
            self._asr_provider_boundary_proofs = {}
        elif not hasattr(self, "_asr_admission_ingress"):
            self._asr_admission_ingress = AdmissionIngressLane(
                self._asr_admission,
                data_capacity=64,
            )
            self._asr_admission_ingress_started = False
            self._asr_admission_rejection_executions = {}
            self._asr_admission_rejection_deadlines = {}
            self._asr_admission_turn_sealed_events = {}
            self._asr_admission_reservation_dispatchers = {}
        if not hasattr(self, "_asr_admission_effect_task_turns"):
            self._asr_admission_effect_task_turns = {}
        if not hasattr(self, "_asr_quarantined_partials"):
            self._asr_quarantined_partials = {}
        if not hasattr(self, "_asr_partial_settlements"):
            self._asr_partial_settlements = {}
        if not hasattr(self, "_asr_speaker_authority_pending_turns"):
            self._asr_speaker_authority_pending_turns = {}
        if not hasattr(self, "_asr_speaker_authoritative_turns"):
            self._asr_speaker_authoritative_turns = set()
        if not hasattr(self, "_asr_speaker_authority_unarming_tasks"):
            self._asr_speaker_authority_unarming_tasks = {}
        if not hasattr(self, "_asr_admission_candidate_owned_tasks"):
            self._asr_admission_candidate_owned_tasks = set(
                self._asr_admission_candidate_tasks.values()
            )
        if not hasattr(self, "_asr_provider_speaker_sequence"):
            self._asr_provider_speaker_sequence = 0
        if not hasattr(self, "_asr_buffered_provider_speaker_observation"):
            self._asr_buffered_provider_speaker_observation = None
        if not hasattr(self, "_asr_overlap_onset_token"):
            self._asr_overlap_onset_token = None
        if not hasattr(self, "_asr_overlap_onset_at"):
            self._asr_overlap_onset_at = None
        if not hasattr(self, "_asr_overlap_completed_onsets"):
            self._asr_overlap_completed_onsets = deque()
        if not hasattr(self, "_asr_partial_turn_token"):
            self._asr_partial_turn_token = None
        if not hasattr(self, "_asr_overlap_completed_token"):
            self._asr_overlap_completed_token = None
            self._asr_overlap_completed_turns = 0
        if not hasattr(self, "_asr_start_generation"):
            self._asr_start_generation = 0
        if not hasattr(self, "_asr_provider_candidate_fence"):
            self._asr_provider_candidate_fence = None
        if not hasattr(self, "_asr_sealed_provider_key"):
            self._asr_sealed_provider_key = None
        if not hasattr(self, "_asr_provider_authority_reset_task"):
            self._asr_provider_authority_reset_task = None
        if not hasattr(self, "_asr_provider_exact_session"):
            self._asr_provider_exact_session = None
        if not hasattr(self, "_asr_owned_cleanup_tasks"):
            self._asr_owned_cleanup_tasks = set()
        if not hasattr(self, "_asr_runtime_close_task"):
            self._asr_runtime_close_task = None
        if not hasattr(self, "_asr_terminal_close_requested"):
            self._asr_terminal_close_requested = False
        if not hasattr(self, "_asr_terminal_close_task"):
            self._asr_terminal_close_task = None
        if not hasattr(self, "_asr_terminal_cancel_requested_tasks"):
            self._asr_terminal_cancel_requested_tasks = weakref.WeakSet()
        if not hasattr(self, "_asr_smart_turn_prepare_lock"):
            self._asr_smart_turn_prepare_lock = asyncio.Lock()
            self._asr_smart_turn_prepare_scope = None
        if not hasattr(self, "_speaker_verifier_factory"):
            self._speaker_verifier_factory = None
            self._speaker_verifier_activation_generation = None
            self._speaker_verifier_enforces_admission = False
            self._speaker_verifier_degraded = False
            self._speaker_verifier_health_generation = 0
        elif not hasattr(self, "_speaker_verifier_degraded"):
            self._speaker_verifier_degraded = False
        if not hasattr(self, "_speaker_verifier_enforces_admission"):
            self._speaker_verifier_enforces_admission = (
                _speaker_factory_enforces_admission(
                    self._speaker_verifier_factory
                )
            )
        if not hasattr(self, "_speaker_verifier_health_generation"):
            self._speaker_verifier_health_generation = 0
        if not hasattr(self, "_speaker_verifier_lock"):
            self._speaker_verifier_lock = asyncio.Lock()

    def _capture_turn_token(
        self,
        lifecycle: VoiceInputLifecycleController,
    ) -> VoiceTurnToken:
        current_turn_token = lifecycle.current_turn_token
        if current_turn_token is not None:
            return current_turn_token
        ingress_token = self._asr_current_ingress_token
        if ingress_token is None or not self._ingress_token_matches(ingress_token):
            raise RuntimeError("ASR_INGRESS_TOKEN_REQUIRED")
        return lifecycle.bind_current_turn_token(ingress_token)

    async def _post_admission_event(
        self,
        turn_token: VoiceTurnToken,
        event: object,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionEffect, ...]:
        """Reduce one fact, then execute every effect outside coordinator lock."""

        if not self._asr_admission_ingress_started:
            await self._asr_admission_ingress.start()
            self._asr_admission_ingress_started = True
        try:
            future = self._asr_admission_ingress.post_nowait(
                turn_token,
                event,  # type: ignore[arg-type]
                now=now,
            )
        except AdmissionIngressCapacityError:
            if isinstance(event, BoundaryExact):
                owner = self._asr_admission_capabilities.pop(
                    event.capability.capability_id,
                    None,
                )
                if owner is not None:
                    owner.revoked = True
                future = self._asr_admission_ingress.post_nowait(
                    turn_token,
                    BoundaryUnknown(event.capability.provider_key),
                    now=now,
                )
            else:
                raise
        consumer = self._consume_admission_future(
            turn_token,
            future,
            suppress_terminal_errors=False,
        )
        return await asyncio.shield(consumer)

    def _track_admission_effect_task(
        self,
        task: asyncio.Task[Any],
        turn_token: VoiceTurnToken | None,
    ) -> None:
        self._asr_admission_effect_tasks.add(task)
        self._asr_admission_effect_task_turns[task] = turn_token

    def _admission_effect_done(self, task: asyncio.Task[Any]) -> None:
        self._asr_admission_effect_tasks.discard(task)
        self._asr_admission_effect_task_turns.pop(task, None)
        self._log_asr_background_task_failure(task)

    async def _finish_admission_invalidation(
        self,
        future: asyncio.Future[tuple[Any, ...]],
        transcript_dispatcher: TranscriptDispatcher,
        correlator: ProviderTurnCorrelator | None,
        namespace: tuple[int, int] | None,
        detector: DetectorRuntime | None,
    ) -> None:
        """Resolve every admission reservation before dispatcher teardown."""

        async def finish_owned() -> None:
            try:
                bulk_results = await asyncio.shield(future)
                retired_turns = {result.turn_token for result in bulk_results}
                for result in bulk_results:
                    for effect in result.effects:
                        await self._execute_admission_effect(effect)
                while True:
                    pending_effects = tuple(
                        task
                        for task, turn_token in tuple(
                            self._asr_admission_effect_task_turns.items()
                        )
                        if turn_token in retired_turns
                        and task is not asyncio.current_task()
                        and not task.done()
                    )
                    if not pending_effects:
                        break
                    await asyncio.gather(
                        *pending_effects,
                        return_exceptions=True,
                    )
                if correlator is not None and namespace is not None:
                    retired = correlator.retire_namespace(namespace)
                    await self._retire_admission_boundary_proofs(
                        retired.retired_proofs,
                        detector,
                    )
            finally:
                transcript_dispatcher.invalidate_all()

        cleanup = asyncio.create_task(
            finish_owned(),
            name="voice-turn-admission-invalidation-owner",
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
            raise

    def _register_admission_capability(
        self,
        lease: DetectorCandidateRejectionLease,
        *,
        kind: RejectionCapabilityKind,
        provider_key: ProviderUtteranceKey | None = None,
    ) -> RejectionCapability | None:
        detector = self._asr_detector
        if detector is None or not lease.belongs_to(detector):
            return None
        existing = next(
            (
                owner.capability
                for owner in self._asr_admission_capabilities.values()
                if not owner.revoked
                and owner.lease == lease
                and owner.capability.kind is kind
                and owner.capability.provider_key == provider_key
            ),
            None,
        )
        if existing is not None:
            return existing
        self._asr_admission_capability_sequence += 1
        capability = RejectionCapability(
            capability_id=self._asr_admission_capability_sequence,
            owner_generation=self._asr_admission_capability_generation,
            kind=kind,
            turn_token=lease.turn_token,
            candidate=lease.shadow_candidate,
            provider_key=provider_key,
        )
        self._asr_admission_capabilities[capability.capability_id] = (
            _AdmissionCapabilityOwner(
                capability=capability,
                lease=lease,
                detector=detector,
                runtime_identity=self._capture_runtime_identity(
                    ingress_token=lease.turn_token.ingress,
                    turn_token=lease.turn_token,
                ),
            )
        )
        return capability

    async def _execute_admission_effect(self, effect: AdmissionEffect) -> None:
        if isinstance(effect, CountDiagnostic):
            metric_name = (
                effect.name
                if effect.name.endswith("_count")
                else f"{effect.name}_count"
            )
            self._speaker_rejection_metrics[metric_name] = (
                self._speaker_rejection_metrics.get(metric_name, 0) + 1
            )
            return
        if isinstance(effect, ScheduleFinalDeadline):
            old = self._asr_admission_deadline_tasks.get(effect.ticket)
            if old is not None:
                return

            async def expire() -> None:
                try:
                    remaining = effect.absolute_deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await self._post_admission_event(
                        effect.ticket.turn_token,
                        FinalDeadlineExpired(
                            ticket=effect.ticket,
                            deadline=effect.absolute_deadline,
                        ),
                    )
                except (asyncio.CancelledError, KeyError):
                    return
                finally:
                    self._asr_admission_deadline_tasks.pop(effect.ticket, None)

            self._asr_admission_deadline_tasks[effect.ticket] = asyncio.create_task(
                expire(),
                name="voice-turn-admission-deadline",
            )
            return
        if isinstance(effect, ConstrainRejectionDeadline):
            current = self._asr_admission_rejection_deadlines.get(effect.ticket)
            if current is None or effect.absolute_deadline < current:
                self._asr_admission_rejection_deadlines[effect.ticket] = (
                    effect.absolute_deadline
                )
            execution = self._asr_admission_rejection_executions.get(
                effect.ticket
            )
            if execution is not None and (
                execution.absolute_deadline is None
                or effect.absolute_deadline < execution.absolute_deadline
            ):
                execution.absolute_deadline = effect.absolute_deadline
                execution.deadline_changed.set()
            return
        if isinstance(effect, ApplyRejection):
            owner = self._asr_admission_capabilities.get(
                effect.capability.capability_id
            )
            if (
                owner is None
                or owner.revoked
                or owner.capability != effect.capability
                or not self._runtime_identity_matches(owner.runtime_identity)
            ):
                await self._post_admission_event(
                    effect.ticket.turn_token,
                    RejectionStale(effect.ticket),
                )
                return
            constrained = self._asr_admission_rejection_deadlines.get(
                effect.ticket
            )
            deadline = effect.absolute_deadline
            if constrained is not None:
                deadline = (
                    constrained
                    if deadline is None
                    else min(deadline, constrained)
                )
            execution = _AdmissionRejectionExecution(
                ticket=effect.ticket,
                absolute_deadline=deadline,
            )
            existing = self._asr_admission_rejection_executions.setdefault(
                effect.ticket,
                execution,
            )
            if existing is not execution:
                return
            sealed_wait_event: asyncio.Event | None = None
            try:
                while True:
                    execution.deadline_changed.clear()
                    deadline = execution.absolute_deadline
                    if deadline is not None and time.monotonic() >= deadline:
                        result = DetectorCandidateRejectionCommitResult.STALE
                        break
                    commit_task = asyncio.create_task(
                        owner.lease.commit_async(deadline=deadline)
                    )
                    constraint_task = asyncio.create_task(
                        execution.deadline_changed.wait()
                    )
                    remaining = (
                        None
                        if deadline is None
                        else max(0.0, deadline - time.monotonic())
                    )
                    done, pending = await asyncio.wait(
                        {commit_task, constraint_task},
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for waiter in pending:
                        waiter.cancel()
                    if commit_task in done:
                        result = await commit_task
                    elif constraint_task in done:
                        await asyncio.gather(commit_task, return_exceptions=True)
                        continue
                    else:
                        await asyncio.gather(
                            commit_task,
                            constraint_task,
                            return_exceptions=True,
                        )
                        result = DetectorCandidateRejectionCommitResult.STALE
                        break
                    if (
                        result
                        is not DetectorCandidateRejectionCommitResult.PRESEAL_READY
                    ):
                        break
                    sealed_wait_event = (
                        self._asr_admission_turn_sealed_events.setdefault(
                            effect.ticket.turn_token,
                            asyncio.Event(),
                        )
                    )
                    waiters = {
                        asyncio.create_task(sealed_wait_event.wait()),
                        asyncio.create_task(execution.deadline_changed.wait()),
                    }
                    remaining = (
                        None
                        if execution.absolute_deadline is None
                        else max(
                            0.0,
                            execution.absolute_deadline - time.monotonic(),
                        )
                    )
                    done, pending = await asyncio.wait(
                        waiters,
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for waiter in pending:
                        waiter.cancel()
                    if not done:
                        result = DetectorCandidateRejectionCommitResult.STALE
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._post_admission_event(
                    effect.ticket.turn_token,
                    RejectionFailed(effect.ticket),
                )
                return
            finally:
                self._asr_admission_rejection_executions.pop(
                    effect.ticket,
                    None,
                )
                self._asr_admission_rejection_deadlines.pop(
                    effect.ticket,
                    None,
                )
                if (
                    sealed_wait_event is not None
                    and self._asr_admission_turn_sealed_events.get(
                        effect.ticket.turn_token
                    )
                    is sealed_wait_event
                ):
                    self._asr_admission_turn_sealed_events.pop(
                        effect.ticket.turn_token,
                        None,
                    )
            applied_kind = {
                DetectorCandidateRejectionCommitResult.ACTIVE_APPLIED: (
                    RejectionCapabilityKind.ACTIVE
                ),
                DetectorCandidateRejectionCommitResult.SEALED_APPLIED: (
                    RejectionCapabilityKind.SEALED
                ),
            }.get(result)
            if applied_kind is None or applied_kind is not effect.capability.kind:
                await self._post_admission_event(
                    effect.ticket.turn_token,
                    RejectionStale(effect.ticket),
                )
                return
            await self._post_admission_event(
                effect.ticket.turn_token,
                RejectionApplied(effect.ticket, applied_kind),
            )
            return
        if isinstance(effect, ResolveReserved):
            await self._resolve_admission_reservation(effect)
            return
        if isinstance(effect, SettlePartial):
            await self._settle_admission_partial(effect)
            return
        if isinstance(effect, RevokeRejectionCapability):
            owner = self._asr_admission_capabilities.pop(
                effect.capability.capability_id,
                None,
            )
            if owner is not None:
                owner.revoked = True
            if effect.ticket is not None:
                await self._post_admission_event(
                    effect.ticket.turn_token,
                    CapabilityRevoked(effect.ticket),
                )
            return
        if isinstance(effect, PoisonSpeakerAuthorityNamespace):
            self._asr_admission_capability_generation += 1
            for owner in self._asr_admission_capabilities.values():
                owner.revoked = True
            self._asr_admission_capabilities.clear()
            if effect.ticket is not None:
                await self._post_admission_event(
                    effect.ticket.turn_token,
                    SpeakerAuthorityNamespacePoisoned(effect.ticket),
                )
            return
        if isinstance(effect, AbortProviderTransport):
            await self._abort_admission_transport(effect.turn_token)

    async def _resolve_admission_reservation(
        self,
        effect: ResolveReserved,
    ) -> None:
        final_key = FinalKey.from_turn(effect.turn_token)
        envelope = None
        if effect.disposition is AdmissionDisposition.FORWARD:
            final = effect.final
            if final is None:
                return
            envelope = TranscriptEnvelope(
                turn_token=effect.turn_token,
                provider=final.provider,
                text=final.text.strip(),
            )
        execution = _AdmissionResolutionExecution(effect.ticket)
        existing = self._asr_admission_resolutions.setdefault(final_key, execution)
        if existing is not execution:
            late_context = self._asr_admission_final_contexts.pop(
                effect.turn_token,
                None,
            )
            if existing.ticket != effect.ticket:
                if late_context is not None:
                    late_context.settled.set()
                return
            if late_context is not None:
                if existing.late_context is None:
                    existing.late_context = late_context
                else:
                    late_context.settled.set()
            if existing.owner_done:
                await self._settle_late_admission_context(existing)
            return
        context = self._asr_admission_final_contexts.pop(effect.turn_token, None)
        existing.late_context = context
        dispatcher = self._asr_admission_reservation_dispatchers.pop(
            final_key,
            None,
        )
        try:
            resolved = bool(
                dispatcher is not None
                and dispatcher.resolve_reserved(
                    final_key,
                    effect.disposition,
                    envelope=envelope,
                )
            )
        except Exception:
            resolved = False
        existing.core_resolution_succeeded = resolved
        if not resolved:
            try:
                await self._post_admission_event(
                    effect.turn_token,
                    CoreSettled(effect.ticket, degraded=True),
                )
                await self._post_admission_event(
                    effect.turn_token,
                    TransportSettled(effect.ticket, degraded=True),
                )
                await self._post_admission_event(
                    effect.turn_token,
                    LifecycleSettled(effect.ticket, degraded=True),
                )
                existing.core_settled = True
                existing.transport_settled = True
                existing.lifecycle_settled = True
            finally:
                existing.settled.set()
                existing.owner_done = True
                context = existing.late_context
                existing.late_context = None
                if context is not None:
                    context.settled.set()
            self._asr_admission_resolutions.pop(final_key, None)
            return
        context = existing.late_context
        existing.late_context = None
        if context is not None:
            try:
                await self._settle_admission_final(effect.ticket, context)
            except Exception:
                for settlement in (
                    TransportSettled(effect.ticket, degraded=True),
                    LifecycleSettled(effect.ticket, degraded=True),
                ):
                    try:
                        await self._post_admission_event(
                            effect.turn_token,
                            settlement,
                        )
                    except (AdmissionIngressClosedError, KeyError):
                        pass
            finally:
                context.settled.set()
            existing.transport_settled = True
            existing.lifecycle_settled = True
        elif effect.disposition is AdmissionDisposition.ABANDON:
            await self._post_admission_event(
                effect.turn_token,
                TransportSettled(effect.ticket),
            )
            await self._post_admission_event(
                effect.turn_token,
                LifecycleSettled(effect.ticket),
            )
            existing.transport_settled = True
            existing.lifecycle_settled = True
        elif effect.final is not None:
            await self._post_admission_event(
                effect.turn_token,
                TransportSettled(effect.ticket, degraded=True),
            )
            await self._post_admission_event(
                effect.turn_token,
                LifecycleSettled(effect.ticket, degraded=True),
            )
            existing.transport_settled = True
            existing.lifecycle_settled = True
        elif effect.disposition is not AdmissionDisposition.DROP:
            await self._post_admission_event(
                effect.turn_token,
                TransportSettled(effect.ticket, degraded=True),
            )
            await self._post_admission_event(
                effect.turn_token,
                LifecycleSettled(effect.ticket, degraded=True),
            )
            existing.transport_settled = True
            existing.lifecycle_settled = True
        if existing.transport_settled and existing.lifecycle_settled:
            existing.settled.set()
        existing.owner_done = True
        if existing.late_context is not None:
            await self._settle_late_admission_context(existing)
        if existing.core_settled:
            self._asr_admission_resolutions.pop(final_key, None)

    async def _settle_late_admission_context(
        self,
        execution: _AdmissionResolutionExecution,
    ) -> None:
        """Attach one late final context to its exact completed ticket."""

        context = execution.late_context
        if context is None:
            return
        execution.late_context = None
        if execution.core_resolution_succeeded is not True:
            context.settled.set()
            return
        try:
            await self._settle_admission_final(execution.ticket, context)
        except Exception:
            for settlement in (
                TransportSettled(execution.ticket, degraded=True),
                LifecycleSettled(execution.ticket, degraded=True),
            ):
                try:
                    await self._post_admission_event(
                        execution.ticket.turn_token,
                        settlement,
                    )
                except (AdmissionIngressClosedError, KeyError):
                    pass
        finally:
            context.settled.set()
        execution.transport_settled = True
        execution.lifecycle_settled = True
        if execution.core_settled:
            execution.settled.set()
            final_key = FinalKey.from_turn(execution.ticket.turn_token)
            if self._asr_admission_resolutions.get(final_key) is execution:
                self._asr_admission_resolutions.pop(final_key, None)

    def _partial_turn_is_current(self, turn_token: VoiceTurnToken) -> bool:
        lifecycle = self._asr_lifecycle
        return bool(
            lifecycle is not None
            and self._asr_partial_turn_token == turn_token
            and self._asr_turn_prepared
            and lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
            and self._ingress_token_matches(turn_token.ingress)
            and lifecycle.snapshot.turn_id == turn_token.turn_id
            and self._asr_audio_dispatcher.active_turn == turn_token
        )

    async def _deliver_independent_asr_preview(
        self,
        turn_token: VoiceTurnToken,
        text: str,
    ) -> None:
        """Deliver one already-admitted display-only partial."""

        if not text or not self._partial_turn_is_current(turn_token):
            return
        lifecycle = self._asr_lifecycle
        assert lifecycle is not None
        if (
            not self._asr_first_partial_recorded
            and self._asr_turn_audio_started_at is not None
        ):
            lifecycle.metrics.first_partial_latency_ms = int(
                (time.monotonic() - self._asr_turn_audio_started_at) * 1_000
            )
            self._asr_first_partial_recorded = True
        try:
            await self._callbacks.on_partial(
                VoicePartialEvent(turn_token=turn_token, text=text)
            )
        except Exception:
            logger.debug(
                "[%s] independent ASR preview delivery failed",
                self.display_name,
            )

    async def _settle_admission_partial(self, effect: SettlePartial) -> None:
        """Apply one reducer-owned terminal verdict to the latest partial."""

        existing = self._asr_partial_settlements.get(effect.turn_token)
        if existing is not None and existing[0] >= effect.record_generation:
            return
        cached = self._asr_quarantined_partials.pop(effect.turn_token, None)
        if not self._partial_turn_is_current(effect.turn_token):
            self._asr_partial_settlements.pop(effect.turn_token, None)
            return
        self._asr_partial_settlements[effect.turn_token] = (
            effect.record_generation,
            effect.disposition,
        )
        if (
            effect.disposition is not AdmissionDisposition.FORWARD
            or effect.turn_token in self._asr_admission_final_contexts
            or cached is None
        ):
            return
        await self._deliver_independent_asr_preview(effect.turn_token, cached)

    async def _abort_admission_transport(self, turn_token: VoiceTurnToken) -> None:
        record = await self._asr_admission.get_record(turn_token)
        ticket = record.resolution_ticket if record is not None else None
        if ticket is None:
            return
        lifecycle = self._asr_lifecycle
        if lifecycle is not None and lifecycle.current_turn_token == turn_token:
            lifecycle.invalidate_transport()
            self._asr_audio_dispatcher.abort(turn_token)
            self._asr_turn_prepared = False
            self._asr_partial_turn_token = None
            self._asr_quarantined_partials.pop(turn_token, None)
            self._asr_partial_settlements.pop(turn_token, None)
            self._asr_speaker_authority_pending_turns.pop(turn_token, None)
            self._asr_speaker_authoritative_turns.discard(turn_token)
            self._asr_sealed_turn_token = None
            self._asr_provider_candidate_fence = None
        correlator = self._asr_provider_correlator
        detector = self._asr_detector
        if correlator is not None:
            retired = correlator.abandon_turn(turn_token)
            await self._retire_admission_boundary_proofs(
                retired.retired_proofs,
                detector,
            )
        await self._post_admission_event(turn_token, TransportSettled(ticket))
        await self._post_admission_event(turn_token, LifecycleSettled(ticket))
        final_key = FinalKey.from_turn(turn_token)
        execution = self._asr_admission_resolutions.get(final_key)
        if execution is not None and execution.ticket == ticket:
            execution.transport_settled = True
            execution.lifecycle_settled = True
            execution.settled.set()
            if execution.core_settled:
                self._asr_admission_resolutions.pop(final_key, None)

    async def _retire_admission_boundary_proofs(
        self,
        proofs: tuple[BoundaryProof, ...],
        detector: DetectorRuntime | None,
    ) -> None:
        if detector is None:
            for proof in proofs:
                snapshot = self._asr_provider_boundary_proofs.pop(
                    proof.proof_id,
                    None,
                )
                if snapshot is not None:
                    self._speaker_rejection_metrics[
                        "admission_boundary_proof_retired_count"
                    ] += 1
            return
        identity = self._capture_runtime_identity()
        for proof in proofs:
            snapshot = self._asr_provider_boundary_proofs.pop(
                proof.proof_id,
                None,
            )
            if snapshot is not None:
                self._speaker_rejection_metrics[
                    "admission_boundary_proof_retired_count"
                ] += 1
                await self._retire_provider_speaker_boundary_unknown(
                    detector,
                    identity,
                    snapshot,
                )

    async def _settle_admission_final(
        self,
        ticket: AdmissionResolutionTicket,
        context: _AdmissionFinalContext,
    ) -> None:
        """Settle Provider and lifecycle after disposition is tombstoned."""

        degraded = False
        successor_present = False
        detector = context.detector
        fence = context.provider_fence
        if detector is not None and fence is not None:
            try:
                completed = await detector.complete_provider_candidate(fence)
            except Exception:
                completed = None
            if completed is None:
                degraded = True
            else:
                successor_present = completed
        lifecycle = context.lifecycle
        owns_current_turn = (
            self._runtime_identity_matches(context.runtime_identity)
            and self._asr_lifecycle is lifecycle
            and lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
            and self._asr_sealed_turn_token == context.sealed_token
            and (
                context.provider_key is None
                or self._asr_sealed_provider_key == context.provider_key
            )
        )
        if owns_current_turn:
            lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
            self._asr_turn_prepared = False
            self._asr_received_audio = False
            self._asr_sealed_turn_token = None
            self._asr_provider_candidate_fence = None
            if context.provider_key is not None:
                self._asr_sealed_provider_key = None
            self._asr_turn_endpointed_at = None
            if self._asr_partial_turn_token == context.turn_token:
                self._asr_partial_turn_token = None
            self._asr_quarantined_partials.pop(context.turn_token, None)
            self._asr_partial_settlements.pop(context.turn_token, None)
            self._asr_speaker_authority_pending_turns.pop(
                context.turn_token,
                None,
            )
            self._asr_speaker_authoritative_turns.discard(context.turn_token)
            if successor_present and not context.has_pending_turn:
                lifecycle.preserve_unconfirmed_pending_audio()
            if not context.has_pending_turn:
                self._schedule_transport_warm_expiry(
                    context.epoch,
                    expected_state=VoiceLifecycleState.WARM_IDLE,
                )
        else:
            degraded = True
        lease = self._asr_smart_turn_lease
        if (
            owns_current_turn
            and lease is not None
            and lease.token == context.turn_token
        ):
            self._asr_smart_turn_lease = None
            try:
                await lease.release()
            except Exception:
                degraded = True
        correlator = context.correlator
        if context.provider_key is not None:
            if correlator is None:
                degraded = True
            else:
                try:
                    completion = correlator.complete(context.provider_key, ticket)
                    await self._retire_admission_boundary_proofs(
                        completion.retired_proofs,
                        detector,
                    )
                    if not completion.completed:
                        degraded = True
                except Exception:
                    degraded = True
        delivered = bool(
            owns_current_turn
            and await self._send_asr_lifecycle_state(
                VoiceLifecycleState.WARM_IDLE,
                provider=context.provider,
                session_epoch=context.epoch,
                expected_identity=context.runtime_identity,
            )
        )
        if not delivered:
            degraded = True
        elif context.has_pending_turn:
            await self._activate_pending_independent_turn(context.epoch)
        if detector is not None and fence is not None:
            try:
                await detector.release_deferred_turn()
            except Exception:
                degraded = True
        await self._post_admission_event(
            context.turn_token,
            TransportSettled(ticket, degraded=degraded),
        )
        await self._post_admission_event(
            context.turn_token,
            LifecycleSettled(ticket, degraded=degraded),
        )

    def _capture_transport_token(
        self,
        lifecycle: VoiceInputLifecycleController,
    ) -> VoiceTransportToken:
        return VoiceTransportToken(
            turn=self._capture_turn_token(lifecycle),
            transport_generation=lifecycle.snapshot.transport_generation,
        )

    def _ingress_token_matches(self, token: VoiceIngressToken) -> bool:
        return bool(
            token.session_epoch == self._asr_session_epoch
            and token.audio_generation == self._asr_audio_generation
        )

    def _transport_token_matches(
        self,
        token: VoiceTransportToken,
        lifecycle: VoiceInputLifecycleController,
    ) -> bool:
        snapshot = lifecycle.snapshot
        return bool(
            self._asr_lifecycle is lifecycle
            and self._ingress_token_matches(token.turn.ingress)
            and token.turn.turn_id == snapshot.turn_id
            and token.transport_generation == snapshot.transport_generation
        )

    def _asr_audio_command_is_valid(
        self,
        turn_token: VoiceTurnToken,
        session_ref: Any,
    ) -> bool:
        lifecycle = self._asr_lifecycle
        detector = self._asr_detector
        return bool(
            lifecycle is not None
            and detector is not None
            and self._asr_session is session_ref
            and self._ingress_token_matches(turn_token.ingress)
            and lifecycle.snapshot.turn_id == turn_token.turn_id
            and self._asr_endpointing_ready(lifecycle, detector, turn_token)
        )

    def _asr_endpointing_ready(
        self,
        lifecycle: VoiceInputLifecycleController,
        detector: DetectorRuntime | None,
        turn_token: VoiceTurnToken,
    ) -> bool:
        """Accept provider authority without manufacturing a SmartTurn lease."""

        if detector is None:
            return False
        if not _uses_smart_turn_endpointing(lifecycle.provider_policy):
            return True
        return detector.endpointing_ready(turn_token)

    async def _record_asr_dispatcher_wire_audio(
        self,
        turn_token: VoiceTurnToken,
        session_ref: Any,
        byte_count: int,
    ) -> None:
        if byte_count <= 0:
            return
        self._sync_provider_wire_metrics(
            session_ref,
            fallback_audio_bytes=byte_count,
        )
        if self._asr_session is session_ref:
            self._asr_received_audio = True
            self._asr_audio_bytes += byte_count
            lifecycle = self._asr_lifecycle
            if lifecycle is not None:
                lifecycle.metrics.provider_wire_sequence = (
                    self._asr_audio_dispatcher.provider_wire_sequence
                )
                lifecycle.metrics.asr_audio_command_queue_ms = (
                    self._asr_audio_dispatcher.asr_audio_command_queue_ms
                )

    async def _handle_asr_audio_dispatcher_failure(
        self,
        turn_token: VoiceTurnToken,
        error: BaseException,
    ) -> None:
        if not self._ingress_token_matches(turn_token.ingress):
            return
        identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        status_code = (
            "ASR_STREAM_BACKPRESSURE"
            if "BACKPRESSURE" in str(error)
            else "ASR_INDEPENDENT_STREAM_FAILED"
        )
        await self._handle_independent_asr_error(
            identity.session_epoch,
            identity.provider or "unknown",
            status_code=status_code,
            expected_identity=identity,
        )

    async def _handle_asr_detector_dispatcher_failure(
        self,
        envelope: CoreDetectorEventEnvelope,
        error: BaseException,
    ) -> None:
        detector = self._asr_detector
        lifecycle = self._asr_lifecycle
        event = envelope.event
        if (
            envelope.session_epoch != self._asr_session_epoch
            or detector is not envelope.detector_ref
            or lifecycle is not envelope.lifecycle_ref
            or detector is None
            or lifecycle is None
            or event.ingress.detector_epoch != detector.detector_epoch
            or not self._ingress_token_matches(event.ingress.ingress_token)
        ):
            return
        logger.error(
            "[%s] detector event dispatcher failed epoch=%s",
            self.display_name,
            envelope.session_epoch,
            exc_info=(type(error), error, error.__traceback__),
        )
        identity = self._capture_runtime_identity(
            ingress_token=event.ingress.ingress_token,
        )
        await self._handle_independent_asr_error(
            identity.session_epoch,
            identity.provider or "unknown",
            status_code="ASR_ENDPOINTING_FAILED",
            expected_identity=identity,
        )

    def _detector_envelope_is_current(
        self,
        envelope: CoreDetectorEventEnvelope,
    ) -> bool:
        detector = self._asr_detector
        lifecycle = self._asr_lifecycle
        event = envelope.event
        return bool(
            envelope.session_epoch == self._asr_session_epoch
            and detector is envelope.detector_ref
            and lifecycle is envelope.lifecycle_ref
            and detector is not None
            and lifecycle is not None
            and event.ingress.detector_epoch == detector.detector_epoch
            and self._ingress_token_matches(event.ingress.ingress_token)
        )

    async def _dispatch_asr_detector_event(
        self,
        envelope: CoreDetectorEventEnvelope,
    ) -> None:
        event = envelope.event
        detector = self._asr_detector
        lifecycle = self._asr_lifecycle
        if not self._detector_envelope_is_current(envelope):
            stale_metrics = getattr(envelope.lifecycle_ref, "metrics", None)
            if stale_metrics is not None:
                stale_metrics.detector_stale_event_count += 1
            return
        assert detector is not None
        assert lifecycle is not None
        lifecycle.metrics.smart_turn_inference_ms = detector.smart_turn_evaluation_ms
        lifecycle.metrics.smart_turn_stale_result_count = (
            detector.smart_turn_stale_result_count
        )
        lifecycle.metrics.smart_turn_coalesced_evaluation_count = (
            detector.smart_turn_coalesced_evaluation_count
        )
        if isinstance(event, DetectorRuntimeEvent):
            identity = self._capture_runtime_identity(
                ingress_token=event.ingress.ingress_token,
            )
            await self._handle_independent_asr_error(
                envelope.session_epoch,
                identity.provider or "unknown",
                status_code=(
                    "ASR_INGRESS_BACKPRESSURE"
                    if event.kind == "audio_backpressure"
                    else "ASR_ENDPOINTING_FAILED"
                ),
                expected_identity=identity,
            )
            return
        if isinstance(event, DetectorTransportPrewarmEvent):
            await self._handle_transport_prewarm_event(
                event,
                detector,
                lifecycle,
                envelope.session_epoch,
            )
            return
        if isinstance(event, DetectorPrewarmEvent):
            await self._handle_detector_prewarm_event(
                event,
                detector,
                lifecycle,
                envelope.session_epoch,
            )
            return
        if isinstance(event, DetectorActivityEvent):
            await self._handle_independent_asr_activity(
                event.activity,
                envelope.session_epoch,
            )
            if not self._detector_envelope_is_current(envelope):
                return
            lifecycle = self._asr_lifecycle
            assert lifecycle is envelope.lifecycle_ref
            if event.activity not in {
                SpeechActivityEvent.SPEECH_STARTED,
                SpeechActivityEvent.SPEECH_RESUMED,
            }:
                return
            if lifecycle.snapshot.state is VoiceLifecycleState.DRAINING:
                self._asr_pending_detector_candidate = event.candidate
                return
            if lifecycle.snapshot.state not in {
                VoiceLifecycleState.PREWARMING,
                VoiceLifecycleState.ACTIVE,
            }:
                return
            turn_token = self._capture_turn_token(lifecycle)
            bound = await detector.bind_candidate(event.candidate, turn_token)
            if bound is None:
                return
            if not self._detector_envelope_is_current(envelope):
                return
            if lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE:
                await self._activate_asr_audio_dispatcher(lifecycle, turn_token)
            return
        if not isinstance(event, DetectorTurnEvent):
            return
        turn_token = event.bound_turn.turn_token
        if (
            not self._ingress_token_matches(turn_token.ingress)
            or lifecycle.snapshot.turn_id != turn_token.turn_id
            or not detector.endpointing_ready(turn_token)
        ):
            return
        await self._handle_independent_asr_endpoint(envelope.session_epoch)
        if not self._detector_envelope_is_current(envelope):
            return
        session_ref = self._asr_session
        if session_ref is None:
            return
        if not self._asr_audio_dispatcher.seal(
            turn_token,
            session_ref,
            after_sequence=self._asr_audio_sequence,
        ):
            identity = self._capture_runtime_identity(
                ingress_token=turn_token.ingress,
                turn_token=turn_token,
            )
            await self._handle_independent_asr_error(
                envelope.session_epoch,
                identity.provider or "unknown",
                status_code="ASR_AUDIO_ORDERING_FAILED",
                expected_identity=identity,
            )

    async def _handle_detector_prewarm_event(
        self,
        event: DetectorPrewarmEvent,
        detector: DetectorRuntime,
        lifecycle: VoiceInputLifecycleController,
        epoch: int,
    ) -> None:
        """Prepare segmented endpointing and transport without final authority."""

        # 用户开口的时刻是**进这个处理函数**的时刻，不是底下 prewarm / transport
        # gather 跑完的时刻。视觉所有权拿 onset 当下界，晚打点会把整段 prewarm+
        # 重连等待算成「用户开口之后」，期间拍的帧全被判成不属于这段发声。
        detected_at = time.monotonic()

        def event_is_current() -> bool:
            return bool(
                epoch == self._asr_session_epoch
                and detector is self._asr_detector
                and lifecycle is self._asr_lifecycle
                and event.ingress.detector_epoch == detector.detector_epoch
                and self._ingress_token_matches(event.ingress.ingress_token)
            )

        if not event_is_current():
            return
        state = lifecycle.snapshot.state
        if state is VoiceLifecycleState.DRAINING:
            if event.kind == "continuous":
                lifecycle.mark_pending_turn_speech(
                    event.ingress.ingress_token
                )
                if self._asr_pending_turn_onset_at is None:
                    self._asr_pending_turn_onset_at = detected_at
                self._asr_pending_detector_candidate = event.candidate
            return
        if state in {
            VoiceLifecycleState.LOCAL_LISTEN,
            VoiceLifecycleState.WARM_IDLE,
            VoiceLifecycleState.DEEP_SLEEP,
        }:
            warm_task = self._asr_warm_expiry_task
            if warm_task is not None:
                warm_task.cancel()
                self._asr_warm_expiry_task = None
            if state is VoiceLifecycleState.WARM_IDLE:
                lifecycle.metrics.warm_hit_count += 1
            lifecycle.open_turn(event.ingress.ingress_token)
            prewarm_identity = self._capture_runtime_identity(
                ingress_token=event.ingress.ingress_token,
            )
            await self._send_asr_lifecycle_state(
                VoiceLifecycleState.PREWARMING,
                provider=prewarm_identity.provider or "unknown",
                session_epoch=prewarm_identity.session_epoch,
                expected_identity=prewarm_identity,
            )
            if not event_is_current():
                return
        if lifecycle.snapshot.state not in {
            VoiceLifecycleState.PREWARMING,
            VoiceLifecycleState.ACTIVE,
        }:
            return

        turn_token = self._capture_turn_token(lifecycle)
        bound = await detector.bind_candidate(event.candidate, turn_token)
        if bound is None or not event_is_current():
            return
        if lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE:
            await self._activate_asr_audio_dispatcher(lifecycle, turn_token)
            if event.kind == "continuous":
                await self._prepare_independent_asr_turn(epoch)
            return

        smart_turn_task = asyncio.create_task(
            self._ensure_smart_turn_ready(lifecycle, epoch),
            name="independent-asr-prewarm-smart-turn",
        )
        transport_task = asyncio.create_task(
            self._restart_transport(),
            name="independent-asr-prewarm-transport",
        )
        smart_turn_ready, _transport_result = await asyncio.gather(
            smart_turn_task,
            transport_task,
            return_exceptions=True,
        )
        if (
            smart_turn_ready is not True
            or not event_is_current()
            or lifecycle.snapshot.state is not VoiceLifecycleState.PREWARMING
        ):
            return
        if event.kind != "continuous":
            self._schedule_transport_warm_expiry(
                epoch,
                expected_state=VoiceLifecycleState.PREWARMING,
            )
            return
        session_ref = self._asr_session
        if session_ref is None or not getattr(session_ref, "is_ready", True):
            self._asr_pending_speech_confirmed = True
            if self._asr_pending_speech_onset_at is None:
                self._asr_pending_speech_onset_at = detected_at
            return
        lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
        # session 先未就绪、随后又 ready 时，真实开口时刻是当初记下的那个
        # pending onset —— 直接用当前时钟会把整段重连等待算成「开口之后」，
        # 期间拍的帧全被排除在本回合外（CodeRabbit Major）。
        self._asr_turn_onset_at = (
            self._asr_pending_speech_onset_at
            if self._asr_pending_speech_onset_at is not None
            else detected_at
        )
        # 直接确认这一路同样要把待确认状态清干净：session 在标记 pending 之后
        # 才 ready 时，直接路径可能先完成确认，旧 flag / 旧 onset 会留到下一轮
        # 被复用（CodeRabbit Major）。
        self._asr_pending_speech_confirmed = False
        self._asr_pending_speech_onset_at = None
        active_identity = self._capture_runtime_identity(
            ingress_token=event.ingress.ingress_token,
        )
        await self._send_asr_lifecycle_state(
            VoiceLifecycleState.ACTIVE,
            provider=active_identity.provider or "unknown",
            session_epoch=active_identity.session_epoch,
            expected_identity=active_identity,
        )
        if not event_is_current():
            return
        self._asr_turn_audio_started_at = time.monotonic()
        self._asr_first_partial_recorded = False
        await self._activate_asr_audio_dispatcher(lifecycle, turn_token)
        await self._prepare_independent_asr_turn(epoch)

    async def _handle_transport_prewarm_event(
        self,
        event: DetectorTransportPrewarmEvent,
        detector: DetectorRuntime,
        lifecycle: VoiceInputLifecycleController,
        epoch: int,
    ) -> None:
        """Preconnect a streaming transport without opening a logical turn."""

        def event_is_current() -> bool:
            return bool(
                epoch == self._asr_session_epoch
                and detector is self._asr_detector
                and lifecycle is self._asr_lifecycle
                and event.ingress.detector_epoch == detector.detector_epoch
                and self._ingress_token_matches(event.ingress.ingress_token)
            )

        if not event_is_current():
            return
        state = lifecycle.snapshot.state
        if state is VoiceLifecycleState.DRAINING:
            return
        if state in {
            VoiceLifecycleState.LOCAL_LISTEN,
            VoiceLifecycleState.WARM_IDLE,
            VoiceLifecycleState.DEEP_SLEEP,
        }:
            warm_task = self._asr_warm_expiry_task
            if warm_task is not None:
                warm_task.cancel()
                self._asr_warm_expiry_task = None
            if state is VoiceLifecycleState.WARM_IDLE:
                lifecycle.metrics.warm_hit_count += 1
            lifecycle.open_turn(event.ingress.ingress_token)
            prewarm_identity = self._capture_runtime_identity(
                ingress_token=event.ingress.ingress_token,
            )
            await self._send_asr_lifecycle_state(
                VoiceLifecycleState.PREWARMING,
                provider=prewarm_identity.provider or "unknown",
                session_epoch=prewarm_identity.session_epoch,
                expected_identity=prewarm_identity,
            )
            if not event_is_current():
                return
        if lifecycle.snapshot.state is not VoiceLifecycleState.PREWARMING:
            return
        session_ref = self._asr_session
        if session_ref is None or not getattr(session_ref, "is_ready", True):
            await self._restart_transport()
        if (
            not event_is_current()
            or lifecycle.snapshot.state is not VoiceLifecycleState.PREWARMING
        ):
            return
        self._schedule_transport_warm_expiry(
            epoch,
            expected_state=VoiceLifecycleState.PREWARMING,
        )

    async def _bind_provider_detector_candidate(
        self,
        lifecycle: VoiceInputLifecycleController,
        detector: DetectorRuntime,
        *,
        detector_identity: DetectorIngressIdentity | None,
        candidate: DetectorCandidateKey | None,
        expected_identity: _AsrRuntimeIdentity,
        pending_speech_confirmed: bool = False,
    ) -> bool:
        """Bind advisory Provider identity and report whether the runtime is current."""

        if detector_identity is None:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_missing_identity_count"
            ] += 1
            return self._runtime_identity_matches(expected_identity)
        if candidate is None:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_missing_candidate_count"
            ] += 1
            return self._runtime_identity_matches(expected_identity)
        if (
            not self._runtime_identity_matches(expected_identity)
            or expected_identity.lifecycle is not lifecycle
            or expected_identity.detector is not detector
            or expected_identity.ingress_token is None
            or detector_identity.ingress_token != expected_identity.ingress_token
            or detector_identity.detector_epoch != detector.detector_epoch
            or candidate.detector_epoch != detector_identity.detector_epoch
        ):
            self._speaker_rejection_metrics[
                "provider_candidate_bind_identity_rejected_count"
            ] += 1
            # Speaker identity is a soft filter. Ambiguous authority never
            # blocks the independent-ASR hard route.
            return self._runtime_identity_matches(expected_identity)

        state = lifecycle.snapshot.state
        if state is VoiceLifecycleState.DRAINING:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_deferred_count"
            ] += 1
            if pending_speech_confirmed or lifecycle.has_pending_turn:
                self._asr_pending_detector_candidate = candidate
            return True
        if state not in {
            VoiceLifecycleState.PREWARMING,
            VoiceLifecycleState.ACTIVE,
        }:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_state_skipped_count"
            ] += 1
            return True

        turn_token = self._capture_turn_token(lifecycle)
        bind_identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        self._speaker_rejection_metrics[
            "provider_candidate_bind_attempt_count"
        ] += 1
        try:
            bound = await detector.bind_candidate(candidate, turn_token)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_failed_count"
            ] += 1
            # Binding is advisory for Provider endpoint authority. The later
            # speaker verdict fails open when no exact detector turn exists.
            return self._runtime_identity_matches(bind_identity)
        if bound is None:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_empty_count"
            ] += 1
        else:
            self._speaker_rejection_metrics[
                "provider_candidate_bind_success_count"
            ] += 1
        return self._runtime_identity_matches(bind_identity)

    async def _ensure_continuous_provider_wake(
        self,
        lifecycle: VoiceInputLifecycleController,
        epoch: int,
        *,
        detector_identity: DetectorIngressIdentity | None = None,
        candidate: DetectorCandidateKey | None = None,
        expected_identity: _AsrRuntimeIdentity | None = None,
    ) -> bool:
        """Open a provider-owned streaming turn without fabricating VAD activity."""

        # 同 _handle_detector_prewarm_event：onset 取进函数的时刻，不取底下各段
        # await 跑完的时刻。
        detected_at = time.monotonic()
        detector = self._asr_detector
        ingress_token = self._asr_current_ingress_token

        def wake_is_current() -> bool:
            return bool(
                epoch == self._asr_session_epoch
                and lifecycle is self._asr_lifecycle
                and detector is self._asr_detector
                and ingress_token is not None
                and self._ingress_token_matches(ingress_token)
            )

        if not wake_is_current():
            return False
        if expected_identity is None:
            expected_identity = self._capture_runtime_identity(
                ingress_token=ingress_token,
            )
        state = lifecycle.snapshot.state
        if state is VoiceLifecycleState.DRAINING:
            lifecycle.mark_pending_turn_speech(ingress_token)
            if self._asr_pending_turn_onset_at is None:
                self._asr_pending_turn_onset_at = detected_at
            return wake_is_current() and await self._bind_provider_detector_candidate(
                lifecycle,
                detector,
                detector_identity=detector_identity,
                candidate=candidate,
                expected_identity=expected_identity,
                pending_speech_confirmed=True,
            )
        if state in {
            VoiceLifecycleState.LOCAL_LISTEN,
            VoiceLifecycleState.WARM_IDLE,
            VoiceLifecycleState.DEEP_SLEEP,
        }:
            warm_task = self._asr_warm_expiry_task
            if warm_task is not None:
                warm_task.cancel()
                self._asr_warm_expiry_task = None
            if state is VoiceLifecycleState.WARM_IDLE:
                lifecycle.metrics.warm_hit_count += 1
            lifecycle.open_turn(ingress_token)
            prewarm_identity = self._capture_runtime_identity(
                ingress_token=ingress_token,
            )
            delivered = await self._send_asr_lifecycle_state(
                VoiceLifecycleState.PREWARMING,
                provider=prewarm_identity.provider or "unknown",
                session_epoch=prewarm_identity.session_epoch,
                expected_identity=prewarm_identity,
            )
            if not delivered or not wake_is_current():
                return False
        if not await self._bind_provider_detector_candidate(
            lifecycle,
            detector,
            detector_identity=detector_identity,
            candidate=candidate,
            expected_identity=expected_identity,
        ):
            return False
        if lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE:
            return True
        if lifecycle.snapshot.state is not VoiceLifecycleState.PREWARMING:
            return False
        session_ref = self._asr_session
        if session_ref is None or not getattr(session_ref, "is_ready", True):
            self._asr_pending_speech_confirmed = True
            if self._asr_pending_speech_onset_at is None:
                self._asr_pending_speech_onset_at = detected_at
            self._ensure_transport_restart_task()
            return wake_is_current()
        turn_token = self._capture_turn_token(lifecycle)
        if not self._asr_endpointing_ready(lifecycle, detector, turn_token):
            identity = self._capture_runtime_identity(
                ingress_token=ingress_token,
                turn_token=turn_token,
            )
            await self._handle_independent_asr_error(
                epoch,
                identity.provider or "unknown",
                status_code="ASR_BLOCKED_ENDPOINTING",
                expected_identity=identity,
            )
            return False
        lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
        # session 先未就绪、随后又 ready 时，真实开口时刻是当初记下的那个
        # pending onset —— 直接用当前时钟会把整段重连等待算成「开口之后」，
        # 期间拍的帧全被排除在本回合外（CodeRabbit Major）。
        self._asr_turn_onset_at = (
            self._asr_pending_speech_onset_at
            if self._asr_pending_speech_onset_at is not None
            else detected_at
        )
        self._asr_pending_speech_confirmed = False
        self._asr_pending_speech_onset_at = None
        active_identity = self._capture_runtime_identity(
            ingress_token=ingress_token,
            turn_token=turn_token,
        )
        delivered = await self._send_asr_lifecycle_state(
            VoiceLifecycleState.ACTIVE,
            provider=active_identity.provider or "unknown",
            session_epoch=active_identity.session_epoch,
            expected_identity=active_identity,
        )
        if not delivered or not wake_is_current():
            return False
        self._asr_turn_audio_started_at = time.monotonic()
        self._asr_first_partial_recorded = False
        await self._prepare_independent_asr_turn(epoch)
        if not wake_is_current():
            return False
        return await self._activate_asr_audio_dispatcher(
            lifecycle,
            turn_token,
        )

    async def _activate_asr_audio_dispatcher(
        self,
        lifecycle: VoiceInputLifecycleController,
        turn_token: VoiceTurnToken,
        *,
        buffered_pcm16: bytes | None = None,
    ) -> bool:
        detector = self._asr_detector
        session_ref = self._asr_session
        if (
            session_ref is None
            or detector is None
            or not getattr(session_ref, "is_ready", True)
            or not self._asr_endpointing_ready(lifecycle, detector, turn_token)
        ):
            return False
        if self._asr_audio_dispatcher.active_turn == turn_token:
            return True
        self._asr_audio_sequence = 0
        buffered_observation = self._asr_buffered_provider_speaker_observation
        self._asr_buffered_provider_speaker_observation = None
        payload = (
            lifecycle.drain_active_start_audio()
            if buffered_pcm16 is None
            else buffered_pcm16
        )
        if payload:
            armed_generation = (
                await self._arm_speaker_authority_for_provider_audio(turn_token)
            )
            if (
                self._asr_session is not session_ref
                or self._asr_detector is not detector
                or self._asr_lifecycle is not lifecycle
                or lifecycle.current_turn_token != turn_token
                or not self._ingress_token_matches(turn_token.ingress)
            ):
                return False
            if (
                self._speaker_verifier_enforces_admission
                and armed_generation
                != self._speaker_verifier_activation_generation
            ):
                return False
        activated = self._asr_audio_dispatcher.activate(
            turn_token,
            session_ref,
            payload,
            sample_rate_hz=16_000,
        )
        if activated:
            spans = (
                buffered_observation.spans
                if buffered_observation is not None
                and buffered_observation.total_bytes == len(payload)
                and buffered_observation.spans
                else None
            )
            if spans is not None:
                expected_start = 0
                for span in spans:
                    if (
                        span.start_byte != expected_start
                        or span.end_byte <= span.start_byte
                        or span.end_byte > len(payload)
                    ):
                        spans = None
                        break
                    expected_start = span.end_byte
                if expected_start != len(payload):
                    spans = None
            if spans is None:
                if not await self._observe_admitted_provider_audio(
                    lifecycle,
                    detector,
                    payload,
                    sample_rate_hz=16_000,
                    identity=None,
                    split_before_audio=False,
                    evidence_complete=False,
                    turn_token=turn_token,
                ):
                    return False
            elif spans is not None:
                for span in spans:
                    span_payload = payload[span.start_byte : span.end_byte]
                    if not await self._observe_admitted_provider_audio(
                        lifecycle,
                        detector,
                        span_payload,
                        sample_rate_hz=16_000,
                        identity=span.last_identity,
                        split_before_audio=span.split_before_audio,
                        evidence_complete=bool(
                            not buffered_observation.overflowed
                            and span.evidence_complete
                        ),
                        turn_token=turn_token,
                    ):
                        return False
        return activated

    def _turn_has_speaker_candidate(self, turn_token: VoiceTurnToken) -> bool:
        return any(
            candidate_turn == turn_token
            for candidate_turn in self._asr_admission_candidate_turns.values()
        )

    async def _arm_speaker_authority_for_provider_audio(
        self,
        turn_token: VoiceTurnToken,
    ) -> str | None:
        """Publish HOLD authority before the first Provider PCM can escape."""

        owner_generation = self._speaker_verifier_activation_generation
        if (
            not self._speaker_verifier_enforces_admission
            or owner_generation is None
            or self._turn_has_speaker_candidate(turn_token)
        ):
            return owner_generation
        lifecycle = self._asr_lifecycle
        if (
            lifecycle is None
            or lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
            or lifecycle.current_turn_token != turn_token
            or not self._ingress_token_matches(turn_token.ingress)
        ):
            return None
        identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        if (
            self._asr_speaker_authority_pending_turns.get(turn_token)
            == owner_generation
        ):
            return owner_generation
        self._asr_speaker_authority_pending_turns[turn_token] = owner_generation
        self._asr_speaker_authoritative_turns.add(turn_token)
        try:
            await self._post_admission_event(
                turn_token,
                SpeakerAuthorityPending(owner_generation),
            )
        except (AdmissionIngressClosedError, AdmissionIngressCapacityError, KeyError):
            if (
                self._asr_speaker_authority_pending_turns.get(turn_token)
                == owner_generation
            ):
                self._asr_speaker_authority_pending_turns.pop(turn_token, None)
            if not self._turn_has_speaker_candidate(turn_token):
                self._asr_speaker_authoritative_turns.discard(turn_token)
            return None
        if (
            not self._speaker_verifier_enforces_admission
            or self._speaker_verifier_activation_generation != owner_generation
            or not self._runtime_identity_matches(identity)
            or self._asr_lifecycle is not lifecycle
            or lifecycle.current_turn_token != turn_token
        ):
            await self._unarm_speaker_authority_after_observation(
                turn_token,
                owner_generation,
            )
            return None
        return owner_generation

    async def _unarm_speaker_authority_after_observation(
        self,
        turn_token: VoiceTurnToken,
        owner_generation: str | None,
    ) -> None:
        """Fail open one ARMING turn when observation produced no candidate."""

        if (
            owner_generation is None
            or self._asr_speaker_authority_pending_turns.get(turn_token)
            != owner_generation
            or self._turn_has_speaker_candidate(turn_token)
        ):
            return
        operation = (turn_token, owner_generation)
        task = self._asr_speaker_authority_unarming_tasks.get(operation)
        if task is None:
            async def settle_unarmed() -> None:
                try:
                    await self._post_admission_event(
                        turn_token,
                        SpeakerAuthorityUnarmed(owner_generation),
                    )
                except (
                    AdmissionIngressClosedError,
                    AdmissionIngressCapacityError,
                    KeyError,
                ):
                    return
                if (
                    self._asr_speaker_authority_pending_turns.get(turn_token)
                    == owner_generation
                ):
                    self._asr_speaker_authority_pending_turns.pop(
                        turn_token,
                        None,
                    )

            task = asyncio.create_task(
                settle_unarmed(),
                name="speaker-authority-unarmed-settlement",
            )
            self._asr_speaker_authority_unarming_tasks[operation] = task
            self._track_admission_effect_task(task, turn_token)

            def done(done_task: asyncio.Task[None]) -> None:
                if self._asr_speaker_authority_unarming_tasks.get(operation) is done_task:
                    self._asr_speaker_authority_unarming_tasks.pop(operation, None)
                self._admission_effect_done(done_task)

            task.add_done_callback(done)
        await asyncio.shield(task)

    def _record_buffered_provider_speaker_observation(
        self,
        *,
        identity: DetectorIngressIdentity | None,
        byte_count: int,
        split_before_audio: bool,
        evidence_complete: bool,
    ) -> None:
        if byte_count <= 0:
            return
        buffered = self._asr_buffered_provider_speaker_observation
        if buffered is None:
            self._asr_buffered_provider_speaker_observation = _BufferedProviderSpeakerObservation(
                total_bytes=byte_count,
                spans=[
                    _BufferedProviderSpeakerSpan(
                        start_byte=0,
                        end_byte=byte_count,
                        first_identity=identity,
                        last_identity=identity,
                        split_before_audio=bool(split_before_audio),
                        evidence_complete=bool(
                            evidence_complete and identity is not None
                        ),
                    )
                ],
            )
            return
        start_byte = buffered.total_bytes
        end_byte = start_byte + byte_count
        buffered.total_bytes = end_byte
        if buffered.overflowed:
            collapsed = buffered.spans[0]
            collapsed.end_byte = end_byte
            collapsed.last_identity = identity
            collapsed.evidence_complete = False
            return

        previous = buffered.spans[-1]
        previous_identity = previous.last_identity
        compatible = bool(
            previous_identity is not None
            and identity is not None
            and previous_identity.ingress_token == identity.ingress_token
            and previous_identity.detector_epoch == identity.detector_epoch
            and previous_identity.sequence_no < identity.sequence_no
        )
        if split_before_audio:
            if len(buffered.spans) >= _MAX_BUFFERED_PROVIDER_SPEAKER_SPANS:
                first = buffered.spans[0]
                buffered.spans[:] = [
                    _BufferedProviderSpeakerSpan(
                        start_byte=0,
                        end_byte=end_byte,
                        first_identity=first.first_identity,
                        last_identity=identity,
                        split_before_audio=False,
                        evidence_complete=False,
                    )
                ]
                buffered.overflowed = True
                return
            buffered.spans.append(
                _BufferedProviderSpeakerSpan(
                    start_byte=start_byte,
                    end_byte=end_byte,
                    first_identity=identity,
                    last_identity=identity,
                    split_before_audio=True,
                    evidence_complete=bool(
                        evidence_complete and identity is not None and compatible
                    ),
                )
            )
            return
        previous.end_byte = end_byte
        previous.last_identity = identity
        previous.evidence_complete = bool(
            previous.evidence_complete
            and evidence_complete
            and identity is not None
            and compatible
        )

    async def _observe_admitted_provider_audio(
        self,
        lifecycle: VoiceInputLifecycleController,
        detector: DetectorRuntime,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
        identity: DetectorIngressIdentity | None,
        split_before_audio: bool,
        evidence_complete: bool,
        turn_token: VoiceTurnToken,
    ) -> bool:
        if not pcm16:
            return True
        owner_generation = self._asr_speaker_authority_pending_turns.get(
            turn_token
        )
        observation_identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        try:
            if _uses_smart_turn_endpointing(lifecycle.provider_policy):
                self._observe_provider_speaker_shadow(
                    detector,
                    pcm16,
                    sample_rate_hz=sample_rate_hz,
                )
                return True
            observe_ordered = getattr(
                detector,
                "observe_provider_audio_ordered",
                None,
            )
            if identity is not None and callable(observe_ordered):
                # Number ordered-observer dispatch attempts only. Explicit
                # fallback revokes incomplete evidence directly.
                self._asr_provider_speaker_sequence += 1
                sequence_no = self._asr_provider_speaker_sequence
                await observe_ordered(
                    pcm16,
                    sample_rate_hz=sample_rate_hz,
                    identity=identity,
                    sequence_no=sequence_no,
                    split_before_audio=split_before_audio,
                    evidence_complete=evidence_complete,
                )
            else:
                self._observe_provider_speaker_shadow(
                    detector,
                    pcm16,
                    sample_rate_hz=sample_rate_hz,
                )
            return self._runtime_identity_matches(observation_identity)
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._runtime_identity_matches(observation_identity)
        finally:
            await self._unarm_speaker_authority_after_observation(
                turn_token,
                owner_generation,
            )

    @staticmethod
    def _observe_provider_speaker_shadow(
        detector: DetectorRuntime,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
    ) -> None:
        try:
            detector.observe_provider_audio(
                pcm16,
                sample_rate_hz=sample_rate_hz,
            )
        except Exception:
            # Observation never participates in ASR acceptance or failure.
            return

    async def _ensure_smart_turn_ready(
        self,
        lifecycle: VoiceInputLifecycleController,
        epoch: int,
    ) -> bool:
        if epoch != self._asr_session_epoch or self._asr_lifecycle is not lifecycle:
            return False
        if not _uses_smart_turn_endpointing(lifecycle.provider_policy):
            return True
        turn_token = self._capture_turn_token(lifecycle)
        detector = self._asr_detector
        identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        if detector is None:
            await self._handle_independent_asr_error(
                epoch,
                identity.provider or "unknown",
                status_code="ASR_BLOCKED_ENDPOINTING",
                expected_identity=identity,
            )
            return False
        prepare_scope = (epoch, id(lifecycle), id(detector))
        if self._asr_smart_turn_prepare_scope != prepare_scope:
            self._asr_smart_turn_prepare_scope = prepare_scope
            self._asr_smart_turn_prepare_lock = asyncio.Lock()
        prepare_lock = self._asr_smart_turn_prepare_lock
        async with prepare_lock:
            if not self._runtime_identity_matches(identity):
                return False
            return await self._ensure_smart_turn_ready_for_identity(
                detector,
                turn_token,
                identity,
                epoch=epoch,
            )

    async def _ensure_smart_turn_ready_for_identity(
        self,
        detector: DetectorRuntime,
        turn_token: VoiceTurnToken,
        identity: _AsrRuntimeIdentity,
        *,
        epoch: int,
    ) -> bool:
        lease = self._asr_smart_turn_lease
        if (
            lease is not None
            and lease.token == turn_token
            and detector.endpointing_ready(turn_token)
        ):
            return True
        if lease is not None:
            await lease.release()
            if self._asr_smart_turn_lease is not lease:
                return False
            self._asr_smart_turn_lease = None
            if not self._runtime_identity_matches(identity):
                return False
        lease = await detector.prepare_endpointing(turn_token)
        if (
            not self._runtime_identity_matches(identity)
            or self._asr_smart_turn_lease is not None
        ):
            if lease is not None:
                await lease.release()
            return False
        if lease is None or not detector.endpointing_ready(turn_token):
            if lease is not None:
                await lease.release()
                if not self._runtime_identity_matches(identity):
                    return False
            await self._handle_independent_asr_error(
                epoch,
                identity.provider or "unknown",
                status_code="ASR_BLOCKED_ENDPOINTING",
                expected_identity=identity,
            )
            return False
        self._asr_smart_turn_lease = lease
        return True

    async def _handle_audio_ingress_backpressure(
        self,
        token: VoiceIngressToken,
        *,
        observed_state: VoiceLifecycleState | None = None,
    ) -> None:
        """Invalidate a whole candidate/turn instead of dropping middle PCM."""

        lifecycle = self._asr_lifecycle
        if lifecycle is None or not self._ingress_token_matches(token):
            return
        epoch = self._asr_session_epoch
        detector = self._asr_detector
        provider = self._asr_provider or "unknown"
        state = observed_state or lifecycle.snapshot.state
        if state is VoiceLifecycleState.DRAINING and not _uses_smart_turn_endpointing(
            lifecycle.provider_policy
        ):
            discard_failed = False
            discard_handled = False
            final_completed_before_discard = False
            async with self._asr_final_lock:
                if (
                    self._asr_lifecycle is not lifecycle
                    or self._asr_detector is not detector
                    or epoch != self._asr_session_epoch
                    or not self._ingress_token_matches(token)
                ):
                    return
                state = lifecycle.snapshot.state
                lifecycle.discard_pending_turn()
                self._asr_pending_turn_onset_at = None
                self._asr_pending_speech_confirmed = False
                self._asr_pending_speech_onset_at = None
                self._asr_pending_detector_candidate = None
                if state is VoiceLifecycleState.DRAINING:
                    sealed_token = self._asr_sealed_turn_token
                    provider_fence = self._asr_provider_candidate_fence
                    if (
                        detector is None
                        or sealed_token is None
                        or provider_fence is None
                        or not self._transport_token_matches(
                            sealed_token,
                            lifecycle,
                        )
                    ):
                        discard_failed = True
                    else:
                        try:
                            discard_handled = await detector.discard_provider_successor(
                                provider_fence
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.warning(
                                "[%s] provider successor discard failed",
                                self.display_name,
                            )
                        discard_failed = not discard_handled
                elif state is VoiceLifecycleState.WARM_IDLE:
                    final_completed_before_discard = True
            if discard_failed:
                identity = self._capture_runtime_identity(ingress_token=token)
                await self._handle_independent_asr_error(
                    epoch,
                    provider,
                    status_code="ASR_ENDPOINTING_FAILED",
                    expected_identity=identity,
                )
                return
            if discard_handled:
                identity = self._capture_runtime_identity(ingress_token=token)
                await self._send_asr_status(
                    "ASR_INGRESS_BACKPRESSURE",
                    provider,
                    session_epoch=epoch,
                    expected_identity=identity,
                )
                return
            if final_completed_before_discard:
                if detector is not None and detector is self._asr_detector:
                    try:
                        await detector.reset()
                    except Exception:
                        logger.warning(
                            "[%s] detector reset failed after pending overflow",
                            self.display_name,
                        )
                identity = self._capture_runtime_identity(ingress_token=token)
                await self._send_asr_status(
                    "ASR_INGRESS_BACKPRESSURE",
                    provider,
                    session_epoch=epoch,
                    expected_identity=identity,
                )
                return
            if state is VoiceLifecycleState.ACTIVE:
                await self._asr_transcript_dispatcher.wait_idle()
        if state is VoiceLifecycleState.DRAINING:
            lifecycle.discard_pending_turn()
            self._asr_pending_turn_onset_at = None
            self._asr_pending_speech_confirmed = False
            self._asr_pending_speech_onset_at = None
            self._asr_pending_detector_candidate = None
            if detector is not None:
                identity = self._capture_runtime_identity(ingress_token=token)
                await detector.reset()
                if not self._runtime_identity_matches(
                    identity
                ) or not self._asr_runtime_refs_match(
                    epoch,
                    lifecycle,
                    detector,
                ):
                    return
            identity = self._capture_runtime_identity(ingress_token=token)
            await self._send_asr_status(
                "ASR_INGRESS_BACKPRESSURE",
                provider,
                session_epoch=epoch,
                expected_identity=identity,
            )
            return
        if state in {
            VoiceLifecycleState.LOCAL_LISTEN,
            VoiceLifecycleState.WARM_IDLE,
            VoiceLifecycleState.DEEP_SLEEP,
        }:
            self._asr_audio_generation += 1
            lifecycle.invalidate_audio()
            if detector is not None:
                identity = self._capture_runtime_identity()
                try:
                    await detector.reset()
                except Exception:
                    logger.warning(
                        "[%s] detector reset failed after ingress backpressure",
                        self.display_name,
                    )
                if not self._runtime_identity_matches(
                    identity
                ) or not self._asr_runtime_refs_match(
                    epoch,
                    lifecycle,
                    detector,
                ):
                    return
            identity = self._capture_runtime_identity()
            await self._send_asr_status(
                "ASR_INGRESS_BACKPRESSURE",
                provider,
                session_epoch=epoch,
                expected_identity=identity,
            )
            return
        if state in {
            VoiceLifecycleState.PREWARMING,
            VoiceLifecycleState.BACKOFF,
            VoiceLifecycleState.ACTIVE,
        }:
            abandoned_turn = (
                self._capture_turn_token(lifecycle)
                if state is VoiceLifecycleState.ACTIVE and self._asr_turn_prepared
                else None
            )
            try:
                lifecycle.invalidate_audio()
                post_detach = await self._abort_transport("detector_audio_backpressure")
                if not self._runtime_identity_matches(
                    post_detach
                ) or not self._asr_runtime_refs_match(
                    epoch,
                    lifecycle,
                    detector,
                ):
                    return
                if detector is not None:
                    await detector.reset()
                    if not self._runtime_identity_matches(
                        post_detach
                    ) or not self._asr_runtime_refs_match(
                        epoch,
                        lifecycle,
                        detector,
                    ):
                        return
                await self._send_asr_status(
                    "ASR_INGRESS_BACKPRESSURE",
                    provider,
                    session_epoch=epoch,
                    expected_identity=post_detach,
                )
                if not self._runtime_identity_matches(post_detach):
                    return
                await self._send_asr_lifecycle_state(
                    VoiceLifecycleState.LOCAL_LISTEN,
                    provider=provider,
                    session_epoch=epoch,
                    expected_identity=post_detach,
                )
                return
            finally:
                if abandoned_turn is not None:
                    await self._notify_asr_turn_abandoned(abandoned_turn)
        identity = self._capture_runtime_identity()
        await self._send_asr_status(
            "ASR_INGRESS_BACKPRESSURE",
            provider,
            session_epoch=epoch,
            expected_identity=identity,
        )

    async def start(
        self,
        *,
        route_key: str,
        resource_optimization_enabled: bool,
        user_language: str | None = None,
        speaker_shadow_factory: SpeakerShadowFactory | None = None,
    ) -> AsrStartResult:
        """Resolve and start one independent-ASR route.

        ``user_language`` is the caller's normalized language preference; the
        session factory maps it onto each provider's accepted hints and falls
        back to automatic detection when it is unknown or unsupported.
        """

        self._ensure_asr_runtime_state()
        if self._asr_terminal_close_requested:
            return AsrStartResult(
                AsrStartStatus.FAILED,
                failure_code="ASR_START_STALE",
                session_epoch=self._asr_session_epoch,
            )
        self._asr_runtime_close_task = None
        operation_generation = self._begin_asr_start_operation()
        if not self._asr_admission_ingress_started:
            await self._asr_admission_ingress.start()
            self._asr_admission_ingress_started = True
        if (
            self._asr_terminal_close_requested
            or not self._asr_start_operation_matches(operation_generation)
        ):
            return AsrStartResult(
                AsrStartStatus.FAILED,
                failure_code="ASR_START_STALE",
                session_epoch=self._asr_session_epoch,
            )
        cleanup = self._detach_independent_asr(
            operation_generation=operation_generation,
        )
        if cleanup is not None:
            cleanup_task = self._schedule_owned_cleanup(
                cleanup,
                name="independent-asr-start-predecessor-close",
            )
            await asyncio.shield(cleanup_task)
        if not self._asr_start_operation_matches(operation_generation):
            return AsrStartResult(
                AsrStartStatus.FAILED,
                failure_code="ASR_START_STALE",
            )
        epoch = self._asr_session_epoch
        audio_generation = self._asr_audio_generation

        def operation_is_current() -> bool:
            return bool(
                not self._asr_terminal_close_requested
                and self._asr_start_operation_matches(operation_generation)
                and epoch == self._asr_session_epoch
                and audio_generation == self._asr_audio_generation
            )

        def stale_result(provider: str | None = None) -> AsrStartResult:
            return AsrStartResult(
                AsrStartStatus.FAILED,
                provider=provider,
                failure_code="ASR_START_STALE",
                session_epoch=epoch,
            )

        self._asr_audio_bytes = 0
        self._voice_input_resource_optimization_enabled = bool(
            resource_optimization_enabled
        )
        core_type = str(route_key or "").strip().lower()

        try:
            # The resolver reads core config synchronously from disk; keep
            # that blocking read off the event loop.
            selection = await asyncio.to_thread(_resolve_asr_selection, core_type)
            selected_provider = getattr(selection, "provider_key", None)
            if not isinstance(selected_provider, str) or not selected_provider.strip():
                raise ValueError("invalid ASR provider selection")
            provider = selected_provider.strip().lower()
            endpointing_mode = getattr(selection, "endpointing_mode", None)
            if endpointing_mode not in {"manual", "provider"}:
                raise ValueError("invalid ASR endpointing selection")
            availability = getattr(
                selection,
                "availability",
                AsrProviderAvailability.IMPLEMENTED,
            )
            if availability is not AsrProviderAvailability.IMPLEMENTED:
                if not operation_is_current():
                    return stale_result(provider)
                failure_code = "ASR_INDEPENDENT_UNAVAILABLE"
                status_identity = self._capture_runtime_identity()
                delivered = await self._send_asr_status(
                    failure_code,
                    provider,
                    session_epoch=epoch,
                    expected_identity=status_identity,
                )
                if not delivered or not operation_is_current():
                    return stale_result(provider)
                return AsrStartResult(
                    AsrStartStatus.UNAVAILABLE,
                    provider=provider,
                    failure_code=failure_code,
                    session_epoch=epoch,
                )
            policy = resolve_provider_policy(provider, endpointing_mode)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Configuration errors must not abort the already-started Core
            # session. Keep the microphone fail-closed and report only the
            # fixed status code/provider category.
            if not operation_is_current():
                return stale_result()
            self._asr_session = None
            self._asr_provider = None
            failure_code = "ASR_INDEPENDENT_FAILED"
            status_identity = self._capture_runtime_identity()
            delivered = await self._send_asr_status(
                failure_code,
                core_type or "unknown",
                session_epoch=epoch,
                expected_identity=status_identity,
            )
            if not delivered or not operation_is_current():
                return stale_result()
            return AsrStartResult(
                AsrStartStatus.FAILED,
                failure_code=failure_code,
                session_epoch=epoch,
            )

        # Provider selection is immutable for this session epoch. Expose the
        # selected provider during connect retries, then clear it only if the
        # startup attempt ultimately fails.
        if not operation_is_current():
            return stale_result(provider)
        self._asr_provider = provider

        def create_candidate(candidate_selection: Any) -> Any:
            """Create one startup candidate with callbacks bound to its identity."""

            candidate_provider = candidate_selection.provider_key
            candidate_endpointing = candidate_selection.endpointing_mode
            candidate_policy = resolve_provider_policy(
                candidate_provider,
                candidate_endpointing,
            )
            candidate_session = None

            def is_adopted_candidate() -> bool:
                return (
                    candidate_session is not None
                    and self._asr_session is candidate_session
                    and epoch == self._asr_session_epoch
                )

            async def on_final(text: str) -> None:
                if not is_adopted_candidate():
                    return
                await self._handle_independent_asr_final(
                    text, epoch, candidate_provider
                )

            async def on_provider_final_ready(
                notification: ProviderFinalNotification,
            ) -> None:
                if not is_adopted_candidate():
                    return
                if (
                    candidate_policy.endpoint_authority == "provider"
                    and notification.key is not None
                ):
                    await self._handle_provider_final(
                        notification.key,
                        notification.text,
                        epoch,
                        candidate_provider,
                        received_at=notification.received_at,
                        admission_deadline=notification.admission_deadline,
                    )
                else:
                    await self._handle_independent_asr_final(
                        notification.text,
                        epoch,
                        candidate_provider,
                        received_at=notification.received_at,
                        deadline=notification.admission_deadline,
                    )

            async def on_error(_message: str) -> None:
                if not is_adopted_candidate():
                    return
                await self._handle_independent_asr_error(epoch, candidate_provider)

            async def on_status(_message: str) -> None:
                # Provider status strings are intentionally not forwarded verbatim.
                return None

            async def on_activity(event: SpeechActivityEvent) -> None:
                if not is_adopted_candidate():
                    return
                await self._handle_independent_asr_activity(event, epoch)

            async def on_endpoint() -> None:
                if not is_adopted_candidate():
                    return
                await self._handle_independent_asr_endpoint(epoch)

            async def on_provider_endpoint(
                notification: ProviderEndpointNotification,
            ) -> None:
                if not is_adopted_candidate():
                    return
                await self._handle_provider_endpoint_notification(
                    notification,
                    epoch,
                )

            async def on_partial(text: str) -> None:
                if not is_adopted_candidate():
                    return
                await self._send_independent_asr_preview(text, epoch)

            candidate_session = _create_asr_session_from_selection(
                core_type,
                selection=candidate_selection,
                on_input_transcript=on_final,
                on_connection_error=on_error,
                on_status_message=on_status,
                on_speech_activity=on_activity,
                on_turn_endpointed=on_endpoint,
                on_provider_endpoint=(
                    on_provider_endpoint
                    if candidate_policy.endpoint_authority == "provider"
                    else None
                ),
                on_provider_final_ready=on_provider_final_ready,
                external_endpointing_runtime=(
                    _uses_smart_turn_endpointing(candidate_policy)
                ),
                user_language=user_language,
            )
            _attach_partial_callback(candidate_session, on_partial)
            return candidate_session

        asr_session = None
        detector_ref: DetectorRuntime | None = None
        connect_started_at = time.monotonic()
        try:
            max_attempts = policy.connect_max_attempts
            for attempt in range(max_attempts):
                if not operation_is_current():
                    return stale_result(provider)
                asr_session = create_candidate(selection)
                try:
                    await asr_session.connect()
                    if not operation_is_current():
                        await self._close_asr_session(asr_session)
                        asr_session = None
                        return stale_result(provider)
                    break
                except asyncio.CancelledError:
                    try:
                        await asr_session.close()
                    except Exception:
                        pass
                    asr_session = None
                    raise
                except Exception:
                    try:
                        await asr_session.close()
                    except Exception:
                        pass
                    asr_session = None
                    if not operation_is_current():
                        return stale_result(provider)
                    if attempt + 1 >= max_attempts:
                        raise
                    backoff = min(
                        policy.connect_retry_cap_seconds,
                        policy.connect_retry_base_seconds * (2**attempt),
                    )
                    # Aggregate retry budget (Codex P1). Each attempt can burn
                    # _READY_TIMEOUT_SECONDS before ASR_CONNECT_TIMEOUT, and
                    # _start_session_activate awaits this whole loop before it
                    # sends session_started -- while the frontend cancels the
                    # start and fires end_session at
                    # _FRONTEND_START_DEADLINE_SECONDS. So on a sustained
                    # provider outage a second attempt could not finish in time
                    # no matter what: the frontend always tore the session down
                    # mid-retry, and the user saw a generic start timeout
                    # instead of the fail-closed ASR verdict this code exists to
                    # produce. Only start another attempt when its worst case
                    # still fits.
                    elapsed = time.monotonic() - connect_started_at
                    if (
                        elapsed + backoff + _READY_TIMEOUT_SECONDS
                        > _CONNECT_TOTAL_BUDGET_SECONDS
                    ):
                        logger.warning(
                            "[asr] connect retry budget exhausted after %.1fs "
                            "(provider=%s attempt=%d/%d); failing closed so the "
                            "verdict reaches the client before its start deadline",
                            elapsed,
                            provider,
                            attempt + 1,
                            max_attempts,
                        )
                        raise
                    await asyncio.sleep(backoff)
                    if not operation_is_current():
                        return stale_result(provider)
            if asr_session is None:
                raise RuntimeError("ASR_CONNECT_FAILED")
            if not operation_is_current():
                await self._close_asr_session(asr_session)
                return stale_result(provider)
            self._asr_session = asr_session
            self._asr_last_provider_wire_audio_ms = 0
            self._asr_provider = provider
            self._asr_lifecycle = VoiceInputLifecycleController(
                provider_policy=policy,
                shadow_mode=False,
                resource_optimization_enabled=(
                    self._voice_input_resource_optimization_enabled
                ),
            )
            self._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
            self._asr_lifecycle.metrics.connect_latency_ms = int(
                (time.monotonic() - connect_started_at) * 1_000
            )
            lifecycle_ref = self._asr_lifecycle

            async def on_detector_endpointing_failure() -> None:
                if not self._asr_runtime_refs_match(
                    epoch,
                    lifecycle_ref,
                    detector_ref,
                ):
                    return
                identity = self._capture_runtime_identity(
                    ingress_token=self._asr_current_ingress_token,
                )
                await self._handle_independent_asr_error(
                    epoch,
                    provider,
                    status_code="ASR_ENDPOINTING_FAILED",
                    expected_identity=identity,
                )

            async def on_detector_event(event) -> None:
                current_lifecycle_ref = self._asr_lifecycle
                if (
                    detector_ref is None
                    or current_lifecycle_ref is None
                    or epoch != self._asr_session_epoch
                ):
                    return
                accepted = self._asr_detector_dispatcher.submit_nowait(
                    CoreDetectorEventEnvelope(
                        event=event,
                        detector_ref=detector_ref,
                        lifecycle_ref=current_lifecycle_ref,
                        session_epoch=epoch,
                    )
                )
                if not accepted:
                    raise RuntimeError("ASR_DETECTOR_CONTROL_BACKPRESSURE")

            def on_speaker_candidate_bound(
                candidate: SpeakerShadowCandidateKey,
                turn_token: VoiceTurnToken,
                speaker_owner_generation: str | None,
            ) -> None:
                if (
                    detector_ref is None
                    or detector_ref is not self._asr_detector
                    or epoch != self._asr_session_epoch
                    or speaker_owner_generation is None
                    or speaker_owner_generation
                    != self._speaker_verifier_activation_generation
                ):
                    return
                self._accept_speaker_candidate_binding(
                    candidate,
                    turn_token,
                    detector=detector_ref,
                    activation_generation=speaker_owner_generation,
                )

            async with self._speaker_verifier_lock:
                if not operation_is_current():
                    return stale_result(provider)
                current_factory = (
                    speaker_shadow_factory
                    if self._speaker_verifier_activation_generation is None
                    else self._speaker_verifier_factory
                )
                factory_activation = getattr(
                    current_factory,
                    "activation_generation",
                    None,
                )
                if (
                    self._speaker_verifier_activation_generation is None
                    and type(factory_activation) is str
                    and factory_activation
                ):
                    self._speaker_verifier_activation_generation = (
                        factory_activation
                    )
                self._speaker_verifier_enforces_admission = (
                    _speaker_factory_enforces_admission(current_factory)
                )
                speaker_shadow = self._create_speaker_shadow(current_factory)
                if speaker_shadow is None:
                    # A declared policy without an installed observer has no
                    # authority. Preserve the existing fail-open contract for
                    # partials and finals until an explicit hot replacement.
                    self._speaker_verifier_enforces_admission = False
                try:
                    detector_ref = DetectorRuntime(
                        resource_optimization_enabled=(
                            self._voice_input_resource_optimization_enabled
                        ),
                        provider_policy=policy,
                        on_endpointing_failure=(
                            on_detector_endpointing_failure
                            if _uses_smart_turn_endpointing(policy)
                            else None
                        ),
                        on_event=on_detector_event,
                        speaker_shadow=speaker_shadow,
                        speaker_owner_generation=(
                            self._speaker_verifier_activation_generation
                            if speaker_shadow is not None
                            else None
                        ),
                        on_speaker_candidate_bound=on_speaker_candidate_bound,
                        provider_micro_event_config=(
                            None
                            if _uses_smart_turn_endpointing(policy)
                            else _PROVIDER_MICRO_EVENT_SHADOW_CONFIG
                        ),
                    )
                except Exception:
                    await self._close_created_speaker_shadow(speaker_shadow)
                    raise
                self._asr_detector = detector_ref
                # The startup detector and Provider session share a fresh
                # physical audio timeline. Reconnects earn this capability
                # only after reset_provider_audio_timeline() succeeds.
                self._asr_provider_exact_session = (
                    asr_session
                    if policy.endpoint_authority == "provider"
                    else None
                )
            self._asr_session_factory = create_candidate
            self._asr_transport_selection = selection
            self._schedule_transport_warm_expiry(
                epoch,
                expected_state=VoiceLifecycleState.LOCAL_LISTEN,
            )
            start_identity = self._capture_runtime_identity()
            delivered = await self._send_asr_lifecycle_state(
                VoiceLifecycleState.LOCAL_LISTEN,
                provider=provider,
                session_epoch=epoch,
                expected_identity=start_identity,
            )
            if (
                not delivered
                or not operation_is_current()
                or not self._runtime_identity_matches(start_identity)
            ):
                return stale_result(provider)
            delivered = await self._send_asr_status(
                "ASR_INDEPENDENT_READY",
                provider,
                session_epoch=epoch,
                expected_identity=start_identity,
            )
            if (
                not delivered
                or not operation_is_current()
                or not self._runtime_identity_matches(start_identity)
            ):
                return stale_result(provider)
            return AsrStartResult(
                AsrStartStatus.READY,
                provider=provider,
                session_epoch=epoch,
            )
        except asyncio.CancelledError:
            if detector_ref is not None and self._asr_detector is detector_ref:
                self._asr_detector = None
                try:
                    await detector_ref.close()
                except Exception:
                    pass
            if asr_session is not None:
                await self._close_asr_session(asr_session)
            raise
        except Exception:
            if detector_ref is not None and self._asr_detector is detector_ref:
                self._asr_detector = None
                try:
                    await detector_ref.close()
                except Exception:
                    pass
            if asr_session is not None:
                await self._close_asr_session(asr_session)
            if operation_is_current():
                self._asr_session = None
                self._asr_provider = None
                failure_code = (
                    "ASR_INDEPENDENT_PROVIDER_UNAVAILABLE"
                    if policy.connect_max_attempts > 1
                    else "ASR_INDEPENDENT_FAILED"
                )
                failure_identity = self._capture_runtime_identity()
                delivered = await self._send_asr_status(
                    failure_code,
                    provider,
                    session_epoch=epoch,
                    expected_identity=failure_identity,
                )
                if not delivered or not operation_is_current():
                    return stale_result(provider)
                return AsrStartResult(
                    AsrStartStatus.UNAVAILABLE
                    if policy.connect_max_attempts > 1
                    else AsrStartStatus.FAILED,
                    provider=provider,
                    failure_code=failure_code,
                    session_epoch=epoch,
                )
            return stale_result(provider)

    def _create_speaker_shadow(
        self,
        factory: SpeakerShadowFactory | None,
    ) -> SpeakerShadowObserver | None:
        """Construct one lightweight observer without risking ASR startup."""

        if factory is None:
            return None
        try:
            # Model/process creation remains lazy inside the observer's first
            # accepted submission.
            shadow = factory()
        except Exception:
            self._speaker_verifier_degraded = True
            logger.warning(
                "[%s] speaker shadow factory failed; continuing without observer",
                self.display_name,
            )
            return None
        if shadow is None:
            self._speaker_verifier_degraded = True
            return None
        return shadow

    @staticmethod
    async def _close_created_speaker_shadow(
        shadow: SpeakerShadowObserver | None,
    ) -> None:
        if shadow is None:
            return
        try:
            await shadow.close()
        except Exception:
            return

    def _reset_asr_provider_transport_namespace(
        self,
        *,
        retire_owned_proofs: bool = False,
    ) -> None:
        """Detach private state keyed to one physical Provider session.

        Boundary snapshots stay in the proof registry until their owning
        correlator returns the corresponding proof for retirement.  Clearing
        the registry here would race the asynchronous Detector cleanup and
        leave the old speaker authority live.
        """

        correlator = self._asr_provider_correlator
        namespace = self._asr_provider_correlator_namespace
        if (
            retire_owned_proofs
            and correlator is not None
            and namespace is not None
        ):
            try:
                retired = correlator.retire_namespace(namespace)
            except ProviderAliasConflictError:
                retired = None
            if retired is not None and retired.retired_proofs:
                task = asyncio.create_task(
                    self._retire_admission_boundary_proofs(
                        retired.retired_proofs,
                        self._asr_detector,
                    ),
                    name="provider-boundary-namespace-reset",
                )
                self._track_admission_effect_task(task, None)
                task.add_done_callback(self._admission_effect_done)
        self._asr_provider_correlator = None
        self._asr_provider_correlator_namespace = None
        self._asr_sealed_provider_key = None
        self._asr_provider_exact_session = None

    def _reset_asr_turn_state(self) -> None:
        """Reset per-turn bookkeeping shared by close/abort/error teardown."""

        self._asr_turn_prepared = False
        self._asr_received_audio = False
        self._asr_pending_speech_confirmed = False
        self._asr_pending_speech_onset_at = None
        self._asr_pending_detector_candidate = None
        self._asr_overlap_onset_token = None
        self._asr_overlap_onset_at = None
        self._asr_overlap_completed_token = None
        self._asr_overlap_completed_onsets.clear()
        self._asr_overlap_completed_turns = 0
        self._asr_audio_sequence = 0
        self._asr_current_ingress_token = None
        self._asr_partial_turn_token = None
        self._asr_admission_candidate_turns.clear()
        self._asr_speaker_authority_pending_turns.clear()
        self._asr_speaker_authoritative_turns.clear()
        self._asr_quarantined_partials.clear()
        self._asr_partial_settlements.clear()
        self._asr_sealed_turn_token = None
        self._asr_provider_candidate_fence = None
        self._reset_asr_provider_transport_namespace()
        self._asr_turn_endpointed_at = None
        self._asr_turn_audio_started_at = None
        self._asr_turn_onset_at = None
        self._asr_pending_turn_onset_at = None
        self._asr_first_partial_recorded = False

    async def _notify_asr_turn_abandoned(
        self,
        turn_token: VoiceTurnToken,
    ) -> None:
        """Release the Core-side pause keyed to an abandoned prepared turn."""

        try:
            await self._callbacks.on_turn_abandoned(turn_token)
        except Exception:
            logger.debug(
                "[%s] independent ASR turn abandonment callback failed",
                self.display_name,
            )

    async def _settle_asr_transcript_terminal(
        self,
        settlement: TranscriptTerminalSettlement,
    ) -> None:
        """Settle Core ownership without revising the admission tombstone."""

        if type(settlement.admission_disposition) is not AdmissionDisposition:
            raise TypeError("ASR_TRANSCRIPT_DISPOSITION_INVALID")
        degraded = False
        try:
            await self._notify_asr_turn_abandoned(
                settlement.final_key.turn_token
            )
        except Exception:
            degraded = True
        execution = self._asr_admission_resolutions.get(settlement.final_key)
        if execution is not None and not execution.core_settled:
            execution.core_settled = True
            try:
                await self._post_admission_event(
                    settlement.final_key.turn_token,
                    CoreSettled(execution.ticket, degraded=degraded),
                )
            except KeyError:
                pass
            if execution.settled.is_set():
                self._asr_admission_resolutions.pop(
                    settlement.final_key,
                    None,
                )

    def _new_asr_transcript_dispatcher(self) -> TranscriptDispatcher:
        """Construct the production dispatcher with mandatory settlement."""

        return TranscriptDispatcher(
            self._dispatch_asr_transcript_envelope,
            settle_terminal=self._settle_asr_transcript_terminal,
            require_terminal_settlement=True,
        )

    async def _close_independent_asr(
        self,
        *,
        operation_generation: int | None = None,
    ) -> None:
        """Invalidate callbacks first, then release the detached provider session."""

        cleanup = self._detach_independent_asr(
            operation_generation=operation_generation,
        )
        if cleanup is not None:
            await cleanup

    def _schedule_provider_authority_reset(
        self,
        detector: DetectorRuntime,
        identity: _AsrRuntimeIdentity,
        *,
        activation_ready: asyncio.Future[bool],
    ) -> asyncio.Task[bool]:
        """Retire a timed-out Provider candidate before any successor feed."""

        existing = self._asr_provider_authority_reset_task
        if existing is not None and not existing.done():
            return existing

        async def reset_authority() -> bool:
            try:
                await asyncio.wait_for(
                    detector.reset(),
                    timeout=_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._runtime_identity_matches(identity):
                    await self._handle_independent_asr_error(
                        identity.session_epoch,
                        identity.provider or "unknown",
                        status_code="ASR_ENDPOINTING_FAILED",
                        expected_identity=identity,
                    )
                return False
            if not self._runtime_identity_matches(identity):
                return False
            should_activate = await activation_ready
            if not self._runtime_identity_matches(identity):
                return False
            if should_activate:
                await self._activate_pending_independent_turn(
                    identity.session_epoch,
                )
            return self._asr_runtime_refs_match(
                identity.session_epoch,
                identity.lifecycle,
                detector,
            )

        task = asyncio.create_task(
            reset_authority(),
            name="provider-authority-fail-open-reset",
        )
        self._asr_provider_authority_reset_task = task
        return task

    def _detach_independent_asr(
        self,
        *,
        operation_generation: int | None = None,
    ) -> Awaitable[None] | None:
        """Synchronously seize one generation and return its owned cleanup."""

        self._ensure_asr_runtime_state()
        if operation_generation is None:
            operation_generation = self._begin_asr_start_operation()
        elif not self._asr_start_operation_matches(operation_generation):
            return None
        self._asr_session_epoch += 1
        self._asr_audio_generation += 1
        transcript_dispatcher = self._asr_transcript_dispatcher
        detector_dispatcher = self._asr_detector_dispatcher
        audio_dispatcher = self._asr_audio_dispatcher
        admission_cleanup_task: asyncio.Task[None] | None = None
        if self._asr_admission_ingress_started:
            admission_future = self._asr_admission_ingress.invalidate_all_nowait(
                RouteReplaced()
            )
            admission_cleanup_task = asyncio.create_task(
                self._finish_admission_invalidation(
                    admission_future,
                    transcript_dispatcher,
                    self._asr_provider_correlator,
                    self._asr_provider_correlator_namespace,
                    self._asr_detector,
                ),
                name="voice-turn-admission-route-replaced",
            )
        else:
            transcript_dispatcher.invalidate_all()
        detector_dispatcher.invalidate_all()
        audio_dispatcher.abort()
        self._asr_transcript_dispatcher = self._new_asr_transcript_dispatcher()
        self._asr_detector_dispatcher = AsrDetectorDispatcher(
            self._dispatch_asr_detector_event,
            on_failure=self._handle_asr_detector_dispatcher_failure,
        )
        self._asr_audio_dispatcher = AsrAudioDispatcher(
            validator=self._asr_audio_command_is_valid,
            on_wire_audio=self._record_asr_dispatcher_wire_audio,
            on_failure=self._handle_asr_audio_dispatcher_failure,
        )
        asr_session, self._asr_session = self._asr_session, None
        lifecycle, self._asr_lifecycle = self._asr_lifecycle, None
        detector, self._asr_detector = self._asr_detector, None
        lease, self._asr_smart_turn_lease = self._asr_smart_turn_lease, None
        detached_tasks: list[asyncio.Task[Any]] = []
        authority_reset_task = self._asr_provider_authority_reset_task
        self._asr_provider_authority_reset_task = None
        if (
            authority_reset_task is not None
            and authority_reset_task is not asyncio.current_task()
        ):
            authority_reset_task.cancel()
            detached_tasks.append(authority_reset_task)
        for task_name in (
            "_asr_transport_task",
            "_asr_warm_expiry_task",
            "_asr_final_watchdog_task",
        ):
            task = getattr(self, task_name, None)
            setattr(self, task_name, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
                detached_tasks.append(task)
        close_tasks = tuple(self._asr_close_tasks)
        self._asr_close_tasks = set()
        self._asr_provider = None
        if lifecycle is not None:
            lifecycle.stop()
        self._asr_provider_speaker_sequence = 0
        self._asr_buffered_provider_speaker_observation = None
        self._reset_asr_turn_state()
        self._asr_session_factory = None
        self._asr_transport_selection = None

        async def finish_detached_cleanup() -> None:
            if admission_cleanup_task is not None:
                await asyncio.gather(
                    admission_cleanup_task,
                    return_exceptions=True,
                )
            if detector is not None:
                try:
                    await detector.close()
                except Exception:
                    logger.warning(
                        "[%s] detector close failed during ASR close",
                        self.display_name,
                    )
            if lease is not None:
                try:
                    await lease.release()
                except Exception:
                    logger.warning(
                        "[%s] SmartTurn lease release failed during ASR close",
                        self.display_name,
                    )
            if asr_session is not None:
                try:
                    await asr_session.close()
                except Exception:
                    logger.warning(
                        "[%s] independent ASR close failed",
                        self.display_name,
                    )
            wait_tasks = (
                *detached_tasks,
                *close_tasks,
            )
            if wait_tasks:
                await asyncio.gather(*wait_tasks, return_exceptions=True)
            await detector_dispatcher.close()
            await audio_dispatcher.close()

        return finish_detached_cleanup()

    async def submit(
        self,
        frame: ProcessedVoiceFrame,
        *,
        ingress_token: VoiceIngressToken,
    ) -> AsrSubmitResult:
        """Submit one normalized frame to the independent-ASR hard route."""

        self._ensure_asr_runtime_state()
        if self._asr_lifecycle is None:
            return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
        if not self._ingress_token_matches(ingress_token):
            return AsrSubmitResult(AsrSubmitStatus.STALE)
        self._asr_current_ingress_token = ingress_token
        authority_reset_task = self._asr_provider_authority_reset_task
        if authority_reset_task is not None:
            reset_succeeded = False
            try:
                reset_succeeded = bool(
                    await asyncio.wait_for(
                        asyncio.shield(authority_reset_task),
                        timeout=_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                reset_succeeded = False
            if self._asr_provider_authority_reset_task is authority_reset_task:
                if authority_reset_task.done():
                    self._asr_provider_authority_reset_task = None
                elif not reset_succeeded:
                    authority_reset_task.cancel()
            if not reset_succeeded:
                identity = self._capture_runtime_identity(
                    ingress_token=ingress_token,
                )
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    status_code="ASR_ENDPOINTING_FAILED",
                    expected_identity=identity,
                )
                return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
        identity = self._capture_runtime_identity(ingress_token=ingress_token)

        pcm16 = frame.pcm16
        sample_rate_hz = frame.sample_rate_hz
        speech_probability = frame.speech_probability
        rnnoise_available = frame.rnnoise_available
        rnnoise_evidence = frame.rnnoise_evidence
        provider_detector_identity: DetectorIngressIdentity | None = None
        split_before_provider_audio = False
        uses_smart_turn = False

        try:
            lifecycle = identity.lifecycle
            detector = identity.detector

            def ingress_is_current() -> bool:
                return self._runtime_identity_matches(identity)

            if lifecycle is not None and detector is not None:
                submit_audio = getattr(detector, "submit_audio", None)
                uses_smart_turn = _uses_smart_turn_endpointing(
                    lifecycle.provider_policy
                )
                if uses_smart_turn and callable(submit_audio):
                    detector_submit_started_at = time.perf_counter()
                    submitted = await submit_audio(
                        pcm16,
                        ingress_token=ingress_token,
                        sample_rate_hz=sample_rate_hz,
                        speech_probability=speech_probability,
                        rnnoise_available=bool(rnnoise_available),
                        rnnoise_evidence=rnnoise_evidence,
                        allow_baseline_update=(
                            lifecycle.snapshot.state
                            in {
                                VoiceLifecycleState.LOCAL_LISTEN,
                                VoiceLifecycleState.WARM_IDLE,
                            }
                        ),
                    )
                    if not ingress_is_current():
                        return AsrSubmitResult(AsrSubmitStatus.STALE)
                    lifecycle.metrics.detector_submit_latency_ms = int(
                        (time.perf_counter() - detector_submit_started_at) * 1_000
                    )
                    lifecycle.metrics.detector_queue_audio_ms = detector.queued_audio_ms
                    lifecycle.metrics.detector_queue_high_water_ms = max(
                        lifecycle.metrics.detector_queue_high_water_ms,
                        detector.queued_audio_ms,
                    )
                    lifecycle.metrics.smart_turn_inference_ms = (
                        detector.smart_turn_evaluation_ms
                    )
                    lifecycle.metrics.smart_turn_stale_result_count = (
                        detector.smart_turn_stale_result_count
                    )
                    lifecycle.metrics.smart_turn_coalesced_evaluation_count = (
                        detector.smart_turn_coalesced_evaluation_count
                    )
                    if submitted.status is DetectorSubmitStatus.SKIPPED_QUIET:
                        return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
                    if submitted.status is DetectorSubmitStatus.BACKPRESSURE:
                        lifecycle.metrics.detector_overflow_count += 1
                        await self._handle_audio_ingress_backpressure(
                            ingress_token,
                            observed_state=lifecycle.snapshot.state,
                        )
                        return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
                    if (
                        submitted.status
                        in {DetectorSubmitStatus.CLOSED, DetectorSubmitStatus.FAILED}
                        or not submitted.endpointing_available
                    ):
                        await self._handle_independent_asr_error(
                            identity.session_epoch,
                            identity.provider or "unknown",
                            status_code="ASR_ENDPOINTING_FAILED",
                            expected_identity=identity,
                        )
                        return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
                    if not submitted.throttle_available:
                        lifecycle.enable_independent_asr_fail_open()
                        if (
                            not submitted.control_event_emitted
                            and submitted.identity is not None
                            and submitted.candidate is not None
                        ):
                            accepted = self._asr_detector_dispatcher.submit_nowait(
                                CoreDetectorEventEnvelope(
                                    event=DetectorPrewarmEvent(
                                        ingress=submitted.identity,
                                        candidate=submitted.candidate,
                                        kind="continuous",
                                    ),
                                    detector_ref=detector,
                                    lifecycle_ref=lifecycle,
                                    session_epoch=identity.session_epoch,
                                )
                            )
                            if not accepted:
                                await self._handle_independent_asr_error(
                                    identity.session_epoch,
                                    identity.provider or "unknown",
                                    status_code="ASR_ENDPOINTING_FAILED",
                                    expected_identity=identity,
                                )
                                return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
                else:
                    detector_result = await detector.feed(
                        pcm16,
                        speech_probability=speech_probability,
                        rnnoise_available=rnnoise_available,
                        rnnoise_evidence=rnnoise_evidence,
                        ingress_token=ingress_token,
                        allow_baseline_update=(
                            lifecycle.snapshot.state
                            in {
                                VoiceLifecycleState.LOCAL_LISTEN,
                                VoiceLifecycleState.WARM_IDLE,
                            }
                        ),
                    )
                    if not ingress_is_current():
                        return AsrSubmitResult(AsrSubmitStatus.STALE)
                    if not detector_result.endpointing_available:
                        await self._handle_independent_asr_error(
                            identity.session_epoch,
                            identity.provider or "unknown",
                            status_code="ASR_ENDPOINTING_FAILED",
                            expected_identity=identity,
                        )
                        return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
                    if detector_result.throttle_action is ThrottleAction.SKIP_IDLE_PCM:
                        return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
                    if not detector_result.throttle_available:
                        lifecycle.enable_independent_asr_fail_open()
                    else:
                        provider_detector_identity = detector_result.identity
                        for event in detector_result.events:
                            split_before_provider_audio = bool(
                                await self._handle_independent_asr_activity(
                                    event,
                                    identity.session_epoch,
                                )
                                or split_before_provider_audio
                            )
                            if not ingress_is_current():
                                return AsrSubmitResult(AsrSubmitStatus.STALE)
                    pending_speech_confirmed = bool(
                        lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
                        and any(
                            event
                            in {
                                SpeechActivityEvent.SPEECH_STARTED,
                                SpeechActivityEvent.SPEECH_RESUMED,
                            }
                            for event in detector_result.events
                        )
                    )
                    continuous_provider_wake = bool(
                        not detector_result.throttle_available
                        or not self._voice_input_resource_optimization_enabled
                    )
                    if continuous_provider_wake:
                        if not await self._ensure_continuous_provider_wake(
                            lifecycle,
                            identity.session_epoch,
                            detector_identity=detector_result.identity,
                            candidate=detector_result.candidate,
                            expected_identity=identity,
                        ):
                            if not ingress_is_current():
                                return AsrSubmitResult(AsrSubmitStatus.STALE)
                            return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
                    elif not await self._bind_provider_detector_candidate(
                        lifecycle,
                        detector,
                        detector_identity=detector_result.identity,
                        candidate=detector_result.candidate,
                        expected_identity=identity,
                        pending_speech_confirmed=pending_speech_confirmed,
                    ):
                        return AsrSubmitResult(AsrSubmitStatus.STALE)
            if lifecycle is not None and not ingress_is_current():
                return AsrSubmitResult(AsrSubmitStatus.STALE)
            decision = (
                lifecycle.accept_audio(pcm16, sample_rate_hz=sample_rate_hz)
                if lifecycle is not None
                else None
            )
            if decision is not None and decision.disposition is AudioDisposition.BLOCK:
                if decision.backpressure:
                    await self._handle_audio_ingress_backpressure(
                        ingress_token,
                        observed_state=lifecycle.snapshot.state,
                    )
                return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
            if decision is not None and decision.disposition in {
                AudioDisposition.BUFFER,
                AudioDisposition.SUPPRESS,
            }:
                if (
                    decision.disposition is AudioDisposition.BUFFER
                    and not uses_smart_turn
                ):
                    self._record_buffered_provider_speaker_observation(
                        identity=provider_detector_identity,
                        byte_count=len(pcm16),
                        split_before_audio=split_before_provider_audio,
                        evidence_complete=(provider_detector_identity is not None),
                    )
                if (
                    lifecycle is not None
                    and lifecycle.snapshot.state
                    in {
                        VoiceLifecycleState.PREWARMING,
                        VoiceLifecycleState.BACKOFF,
                    }
                    and (
                        self._asr_session is None
                        or not getattr(self._asr_session, "is_ready", True)
                    )
                ):
                    self._ensure_transport_restart_task()
                return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
            if lifecycle is None or detector is None:
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    status_code="ASR_BLOCKED_ENDPOINTING",
                    expected_identity=identity,
                )
                return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
            turn_token = self._capture_turn_token(lifecycle)
            if (
                lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
                or not self._asr_endpointing_ready(lifecycle, detector, turn_token)
            ):
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    status_code="ASR_BLOCKED_ENDPOINTING",
                    expected_identity=identity,
                )
                return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
            asr_session = self._asr_session
            if asr_session is None or not getattr(asr_session, "is_ready", True):
                self._ensure_transport_restart_task()
                return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
            payload = (
                decision.pre_roll
                if decision is not None
                and decision.disposition is AudioDisposition.FORWARD_WITH_PRE_ROLL
                else pcm16
            )
            if not payload:
                return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
            if not ingress_is_current():
                return AsrSubmitResult(AsrSubmitStatus.STALE)
            if self._asr_audio_dispatcher.active_turn != turn_token:
                if not await self._activate_asr_audio_dispatcher(
                    lifecycle,
                    turn_token,
                ):
                    await self._handle_independent_asr_error(
                        identity.session_epoch,
                        identity.provider or "unknown",
                        status_code="ASR_AUDIO_ORDERING_FAILED",
                        expected_identity=identity,
                    )
                    return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
            armed_generation = (
                await self._arm_speaker_authority_for_provider_audio(turn_token)
            )
            if (
                not ingress_is_current()
                or self._asr_session is not asr_session
                or self._asr_detector is not detector
                or self._asr_lifecycle is not lifecycle
                or lifecycle.current_turn_token != turn_token
            ):
                return AsrSubmitResult(AsrSubmitStatus.STALE)
            if (
                self._speaker_verifier_enforces_admission
                and armed_generation
                != self._speaker_verifier_activation_generation
            ):
                return AsrSubmitResult(
                    AsrSubmitStatus.STALE
                    if not ingress_is_current()
                    else AsrSubmitStatus.UNAVAILABLE
                )
            self._asr_audio_sequence += 1
            if not self._asr_audio_dispatcher.enqueue_audio(
                turn_token,
                asr_session,
                payload,
                sample_rate_hz=sample_rate_hz,
                sequence_no=self._asr_audio_sequence,
            ):
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    status_code="ASR_AUDIO_ORDERING_FAILED",
                    expected_identity=identity,
                )
                return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
            split_payload_is_ambiguous = bool(
                split_before_provider_audio
                and decision is not None
                and decision.disposition is AudioDisposition.FORWARD_WITH_PRE_ROLL
            )
            if not await self._observe_admitted_provider_audio(
                lifecycle,
                detector,
                payload,
                sample_rate_hz=sample_rate_hz,
                identity=provider_detector_identity,
                split_before_audio=bool(
                    split_before_provider_audio and not split_payload_is_ambiguous
                ),
                evidence_complete=not split_payload_is_ambiguous,
                turn_token=turn_token,
            ):
                return AsrSubmitResult(AsrSubmitStatus.STALE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._runtime_identity_matches(identity):
                return AsrSubmitResult(AsrSubmitStatus.STALE)
            self._asr_received_audio = True
            status_code = (
                "ASR_STREAM_BACKPRESSURE"
                if str(exc).startswith("ASR_STREAM_BACKPRESSURE:")
                else "ASR_INDEPENDENT_STREAM_FAILED"
            )
            if (
                status_code == "ASR_STREAM_BACKPRESSURE"
                and identity.lifecycle is not None
            ):
                identity.lifecycle.metrics.queue_backpressure_count += 1
            await self._handle_independent_asr_error(
                identity.session_epoch,
                identity.provider or "unknown",
                status_code=status_code,
                expected_identity=identity,
            )
            return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)

        return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)

    def _ensure_transport_restart_task(self) -> None:
        task = self._asr_transport_task
        if task is not None and not task.done():
            return
        task = asyncio.create_task(
            self._restart_transport(),
            name="independent-asr-transport-restart",
        )
        task.add_done_callback(self._log_asr_background_task_failure)
        self._asr_transport_task = task

    def _log_asr_background_task_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "[%s] independent ASR background task %s failed",
                self.display_name,
                task.get_name(),
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _restart_transport(self, *, max_attempts: int | None = None) -> None:
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        async with self._asr_transport_lock:
            lifecycle = self._asr_lifecycle
            if lifecycle is None:
                return
            existing = self._asr_session
            if existing is not None and getattr(existing, "is_ready", True):
                return
            if existing is not None:
                self._asr_session = None
                self._asr_provider_exact_session = None
                detached_identity = self._capture_runtime_identity()
                await self._close_asr_session(existing)
                if not self._runtime_identity_matches(detached_identity):
                    return
            lifecycle = self._asr_lifecycle
            factory = self._asr_session_factory
            selection = self._asr_transport_selection
            identity = self._capture_runtime_identity()
            if factory is None or selection is None or lifecycle is None:
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    expected_identity=identity,
                )
                return
            # Mirror initial startup: the active provider policy decides the
            # attempt budget and backoff ladder unless the caller overrides it.
            policy = lifecycle.provider_policy
            if max_attempts is None:
                max_attempts = policy.connect_max_attempts

            for attempt in range(max_attempts):
                if not self._runtime_identity_matches(identity):
                    return
                if lifecycle.snapshot.state is VoiceLifecycleState.BACKOFF:
                    lifecycle.transition(VoiceLifecycleEvent.RETRY)
                    lifecycle.metrics.reconnect_count += 1
                    identity = self._capture_runtime_identity()
                    await self._send_asr_lifecycle_state(
                        VoiceLifecycleState.PREWARMING,
                        provider=identity.provider or "unknown",
                        session_epoch=identity.session_epoch,
                        expected_identity=identity,
                    )
                    if not self._runtime_identity_matches(identity):
                        return
                candidate = None
                try:
                    connect_started_at = time.monotonic()
                    candidate = factory(selection)
                    await candidate.connect()
                    if not self._runtime_identity_matches(identity):
                        try:
                            await candidate.close()
                        except Exception:
                            pass
                        return
                    detector = self._asr_detector
                    reset_provider_timeline = getattr(
                        detector,
                        "reset_provider_audio_timeline",
                        None,
                    )
                    exact_timeline_ready = False
                    if (
                        policy.endpoint_authority == "provider"
                        and callable(reset_provider_timeline)
                    ):
                        try:
                            exact_timeline_ready = bool(
                                await reset_provider_timeline()
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            # A speaker-only reset failure must not turn a
                            # connected replacement into an ASR outage.
                            exact_timeline_ready = False
                    if not self._runtime_identity_matches(identity):
                        try:
                            await candidate.close()
                        except Exception:
                            pass
                        return
                    # No await is allowed between namespace retirement and
                    # adoption. Provider callbacks are accepted only after
                    # _asr_session points at this candidate.
                    self._reset_asr_provider_transport_namespace(
                        retire_owned_proofs=True
                    )
                    self._asr_provider_exact_session = (
                        candidate
                        if policy.endpoint_authority == "provider"
                        and exact_timeline_ready
                        else None
                    )
                    self._asr_session = candidate
                    self._asr_last_provider_wire_audio_ms = 0
                    lifecycle.invalidate_transport()
                    connected_identity = self._capture_runtime_identity()
                    lifecycle.metrics.connect_latency_ms = int(
                        (time.monotonic() - connect_started_at) * 1_000
                    )
                    if (
                        self._asr_pending_speech_confirmed
                        and lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
                    ):
                        detector = self._asr_detector
                        turn_token = self._capture_turn_token(lifecycle)
                        if detector is None or not self._asr_endpointing_ready(
                            lifecycle,
                            detector,
                            turn_token,
                        ):
                            await self._handle_independent_asr_error(
                                connected_identity.session_epoch,
                                connected_identity.provider or "unknown",
                                status_code="ASR_BLOCKED_ENDPOINTING",
                                expected_identity=connected_identity,
                            )
                            return
                        lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
                        self._asr_turn_onset_at = (
                            self._asr_pending_speech_onset_at
                            if self._asr_pending_speech_onset_at is not None
                            else time.monotonic()
                        )
                        self._asr_pending_speech_confirmed = False
                        self._asr_pending_speech_onset_at = None
                        self._asr_turn_audio_started_at = time.monotonic()
                        self._asr_first_partial_recorded = False
                        await self._send_asr_lifecycle_state(
                            VoiceLifecycleState.ACTIVE,
                            provider=connected_identity.provider or "unknown",
                            session_epoch=connected_identity.session_epoch,
                            expected_identity=connected_identity,
                        )
                        if not self._runtime_identity_matches(connected_identity):
                            return
                        payload = lifecycle.drain_active_start_audio()
                        await self._prepare_independent_asr_turn(
                            connected_identity.session_epoch
                        )
                        if not self._runtime_identity_matches(connected_identity):
                            return
                        if not await self._activate_asr_audio_dispatcher(
                            lifecycle,
                            turn_token,
                            buffered_pcm16=payload,
                        ):
                            await self._handle_independent_asr_error(
                                connected_identity.session_epoch,
                                connected_identity.provider or "unknown",
                                status_code="ASR_AUDIO_ORDERING_FAILED",
                                expected_identity=connected_identity,
                            )
                            return
                    return
                except asyncio.CancelledError:
                    if candidate is not None and self._asr_session is candidate:
                        adopted_identity = self._capture_runtime_identity()
                        await self._handle_independent_asr_error(
                            adopted_identity.session_epoch,
                            adopted_identity.provider or "unknown",
                            status_code="ASR_INDEPENDENT_FAILED",
                            expected_identity=adopted_identity,
                        )
                    elif candidate is not None:
                        try:
                            await candidate.close()
                        except Exception:
                            pass
                    raise
                except Exception:
                    if candidate is not None and self._asr_session is candidate:
                        adopted_identity = self._capture_runtime_identity()
                        await self._handle_independent_asr_error(
                            adopted_identity.session_epoch,
                            adopted_identity.provider or "unknown",
                            status_code="ASR_INDEPENDENT_FAILED",
                            expected_identity=adopted_identity,
                        )
                        return
                    if candidate is not None:
                        try:
                            await candidate.close()
                        except Exception:
                            pass
                    if not self._runtime_identity_matches(identity):
                        return
                    if lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING:
                        lifecycle.transition(VoiceLifecycleEvent.CONNECT_FAILED)
                        identity = self._capture_runtime_identity()
                        await self._send_asr_lifecycle_state(
                            VoiceLifecycleState.BACKOFF,
                            provider=identity.provider or "unknown",
                            session_epoch=identity.session_epoch,
                            expected_identity=identity,
                        )
                        if not self._runtime_identity_matches(identity):
                            return
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(
                            min(
                                policy.connect_retry_cap_seconds,
                                policy.connect_retry_base_seconds * (2**attempt),
                            )
                        )
                        if not self._runtime_identity_matches(identity):
                            return
                        continue
            if not self._runtime_identity_matches(identity):
                return
            if lifecycle.snapshot.state is VoiceLifecycleState.BACKOFF:
                lifecycle.transition(VoiceLifecycleEvent.RETRIES_EXHAUSTED)
            await self._handle_independent_asr_error(
                identity.session_epoch,
                identity.provider or "unknown",
                status_code="ASR_INDEPENDENT_FAILED",
                expected_identity=identity,
            )

    async def _abort_transport(
        self,
        reason: str,
    ) -> _AsrRuntimeIdentity:
        """Invalidate provider I/O before closing a live transport."""

        self._begin_asr_start_operation()
        self._asr_audio_generation += 1
        transcript_dispatcher = self._asr_transcript_dispatcher
        if self._asr_admission_ingress_started:
            await self._finish_admission_invalidation(
                self._asr_admission_ingress.invalidate_all_nowait(
                    RouteReplaced()
                ),
                transcript_dispatcher,
                self._asr_provider_correlator,
                self._asr_provider_correlator_namespace,
                self._asr_detector,
            )
        else:
            transcript_dispatcher.invalidate_all()
        self._asr_detector_dispatcher.invalidate_all()
        self._asr_audio_dispatcher.abort()
        self._asr_provider_speaker_sequence = 0
        self._asr_buffered_provider_speaker_observation = None
        self._reset_asr_turn_state()
        lease, self._asr_smart_turn_lease = self._asr_smart_turn_lease, None
        for task_name in (
            "_asr_transport_task",
            "_asr_warm_expiry_task",
            "_asr_final_watchdog_task",
        ):
            task = getattr(self, task_name, None)
            setattr(self, task_name, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        asr_session, self._asr_session = self._asr_session, None
        lifecycle = self._asr_lifecycle
        if lifecycle is not None:
            lifecycle.metrics.asr_abort_discarded_command_count = (
                self._asr_audio_dispatcher.asr_abort_discarded_command_count
            )
            lifecycle.invalidate_transport()
        post_detach = self._capture_runtime_identity()

        async def finish_abort() -> None:
            try:
                if lease is not None:
                    await lease.release()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "[%s] SmartTurn lease release failed during ASR abort",
                    self.display_name,
                )
            finally:
                if asr_session is not None:
                    try:
                        await asr_session.close()
                    except Exception:
                        logger.warning(
                            "[%s] independent ASR abort failed reason=%s",
                            self.display_name,
                            reason,
                        )

        cleanup_task = self._schedule_owned_cleanup(
            finish_abort(),
            name="independent-asr-abort-transport",
        )
        await asyncio.shield(cleanup_task)
        return post_detach

    async def _close_transport_only(self) -> None:
        """Enter deep sleep while preserving microphone detection."""

        epoch = self._asr_session_epoch
        provider = self._asr_provider or "unknown"
        warm_task = self._asr_warm_expiry_task
        if warm_task is not None and warm_task is not asyncio.current_task():
            warm_task.cancel()
        self._asr_warm_expiry_task = None
        asr_session, self._asr_session = self._asr_session, None
        self._asr_provider_exact_session = None
        session_close_task = None
        if asr_session is not None:
            async def close_transport() -> None:
                try:
                    await asr_session.close()
                except Exception:
                    logger.warning(
                        "[%s] independent ASR transport-only close failed",
                        self.display_name,
                    )

            session_close_task = self._schedule_owned_cleanup(
                close_transport(),
                name="independent-asr-transport-close",
            )
        lifecycle = self._asr_lifecycle
        if lifecycle is not None:
            lifecycle.invalidate_transport()
            if lifecycle.snapshot.state in {
                VoiceLifecycleState.LOCAL_LISTEN,
                VoiceLifecycleState.WARM_IDLE,
            }:
                lifecycle.transition(VoiceLifecycleEvent.WARM_EXPIRED)
                identity = self._capture_runtime_identity()
                await self._send_asr_lifecycle_state(
                    VoiceLifecycleState.DEEP_SLEEP,
                    provider=provider,
                    session_epoch=epoch,
                    expected_identity=identity,
                )
        if session_close_task is not None:
            await asyncio.shield(session_close_task)

    def _schedule_transport_warm_expiry(
        self,
        epoch: int,
        *,
        expected_state: VoiceLifecycleState,
    ) -> None:
        task = self._asr_warm_expiry_task
        if task is not None:
            task.cancel()
        lifecycle = self._asr_lifecycle
        if lifecycle is None or not self._voice_input_resource_optimization_enabled:
            return
        if expected_state is VoiceLifecycleState.WARM_IDLE:
            ttl_ms = lifecycle.provider_policy.warm_transport_ms
        elif expected_state in {
            VoiceLifecycleState.LOCAL_LISTEN,
            VoiceLifecycleState.PREWARMING,
        }:
            ttl_ms = lifecycle.config.default_warm_transport_ms
        else:
            raise ValueError(
                "transport expiry requires local-listen, prewarming, or warm-idle"
            )
        session_ref = self._asr_session
        detector_ref = self._asr_detector
        transport_generation = lifecycle.snapshot.transport_generation

        def timer_is_current() -> bool:
            return bool(
                epoch == self._asr_session_epoch
                and self._asr_lifecycle is lifecycle
                and self._asr_session is session_ref
                and self._asr_detector is detector_ref
                and lifecycle.snapshot.transport_generation == transport_generation
            )

        async def expire() -> None:
            try:
                await asyncio.sleep(ttl_ms / 1_000)
                if (
                    not timer_is_current()
                    or lifecycle.snapshot.state is not expected_state
                ):
                    return
                if expected_state is VoiceLifecycleState.PREWARMING:
                    lease, self._asr_smart_turn_lease = (
                        self._asr_smart_turn_lease,
                        None,
                    )
                    if lease is not None:
                        await lease.release()
                    if not timer_is_current():
                        return
                    if detector_ref is not None:
                        await detector_ref.reset()
                    if (
                        not timer_is_current()
                        or lifecycle.snapshot.state
                        is not VoiceLifecycleState.PREWARMING
                    ):
                        return
                    lifecycle.transition(VoiceLifecycleEvent.PREWARM_EXPIRED)
                    self._asr_pending_speech_confirmed = False
                    self._asr_pending_speech_onset_at = None
                    self._asr_pending_detector_candidate = None
                    identity = self._capture_runtime_identity()
                    delivered = await self._send_asr_lifecycle_state(
                        VoiceLifecycleState.LOCAL_LISTEN,
                        provider=identity.provider or "unknown",
                        session_epoch=identity.session_epoch,
                        expected_identity=identity,
                    )
                    if (
                        not delivered
                        or not timer_is_current()
                        or lifecycle.snapshot.state
                        is not VoiceLifecycleState.LOCAL_LISTEN
                    ):
                        return
                await self._close_transport_only()
            except asyncio.CancelledError:
                return
            finally:
                if self._asr_warm_expiry_task is asyncio.current_task():
                    self._asr_warm_expiry_task = None

        warm_task = asyncio.create_task(
            expire(),
            name="independent-asr-warm-expiry",
        )
        warm_task.add_done_callback(self._log_asr_background_task_failure)
        self._asr_warm_expiry_task = warm_task

    def _schedule_provider_final_watchdog(
        self,
        epoch: int,
        lifecycle: VoiceInputLifecycleController,
        sealed_token: VoiceTransportToken,
    ) -> None:
        task = self._asr_final_watchdog_task
        if task is not None:
            task.cancel()
        timeout_ms = lifecycle.provider_policy.provider_final_timeout_ms

        async def expire() -> None:
            try:
                await asyncio.sleep(timeout_ms / 1_000)
                if (
                    epoch != self._asr_session_epoch
                    or self._asr_lifecycle is not lifecycle
                    or self._asr_sealed_turn_token != sealed_token
                    or lifecycle.snapshot.state is not VoiceLifecycleState.DRAINING
                ):
                    return
                await self._handle_independent_asr_error(
                    epoch,
                    self._asr_provider or "unknown",
                    status_code="ASR_PROVIDER_FINAL_TIMEOUT",
                )
            except asyncio.CancelledError:
                return

        watchdog_task = asyncio.create_task(
            expire(),
            name="independent-asr-provider-final-watchdog",
        )
        watchdog_task.add_done_callback(self._log_asr_background_task_failure)
        self._asr_final_watchdog_task = watchdog_task

    def _sync_provider_wire_metrics(
        self,
        asr_session: Any,
        *,
        fallback_audio_bytes: int = 0,
    ) -> None:
        lifecycle = self._asr_lifecycle
        if lifecycle is None:
            return
        cumulative_ms = getattr(asr_session, "provider_wire_audio_ms", None)
        if isinstance(cumulative_ms, int) and not isinstance(cumulative_ms, bool):
            delta_ms = max(0, cumulative_ms - self._asr_last_provider_wire_audio_ms)
            self._asr_last_provider_wire_audio_ms = max(
                self._asr_last_provider_wire_audio_ms,
                cumulative_ms,
            )
            if delta_ms:
                lifecycle.record_provider_wire_audio(delta_ms)
            return
        if (
            lifecycle.provider_policy.transport == "streaming"
            and fallback_audio_bytes > 0
        ):
            lifecycle.record_provider_wire_audio(
                fallback_audio_bytes * 1_000 // (16_000 * 2)
            )

    async def _handle_independent_asr_activity(
        self,
        event: SpeechActivityEvent,
        epoch: int,
    ) -> bool:
        # 同上：onset 是收到这个语音活动事件的时刻。
        detected_at = time.monotonic()
        if epoch != self._asr_session_epoch:
            return False
        provider = self._asr_provider or "unknown"
        lifecycle = self._asr_lifecycle
        if (
            lifecycle is not None
            and lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
            and event
            in {
                SpeechActivityEvent.SPEECH_STARTED,
                SpeechActivityEvent.SPEECH_RESUMED,
            }
        ):
            ingress_token = self._asr_current_ingress_token
            if ingress_token is None or not self._ingress_token_matches(
                ingress_token
            ):
                return False
            lifecycle.mark_pending_turn_speech(ingress_token)
            if self._asr_pending_turn_onset_at is None:
                self._asr_pending_turn_onset_at = detected_at
            return False
        if (
            lifecycle is not None
            and lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
            and lifecycle.has_pending_turn
            and event
            in {
                SpeechActivityEvent.SPEECH_STARTED,
                SpeechActivityEvent.SPEECH_RESUMED,
            }
        ):
            # The DRAINING path already confirmed this pending turn. Re-marking
            # it after PROVIDER_FINAL reaches WARM_IDLE violates the lifecycle
            # guard and can fail the replacement turn during activation.
            return False
        if lifecycle is not None and event in {
            SpeechActivityEvent.SPEECH_STARTED,
            SpeechActivityEvent.SPEECH_RESUMED,
        }:
            ingress_token = self._asr_current_ingress_token
            if ingress_token is None or not self._ingress_token_matches(ingress_token):
                # An idle ingress-backpressure bump keeps the provider session
                # adopted, so a trailing session-side speech event can still
                # reach this handler with a stale audio generation. The wake
                # path below cannot mint a turn token without a current
                # ingress token, so drop the stale event cleanly instead of
                # raising into the provider adapter. Genuinely new speech
                # re-arms the current token through submit() first.
                return False
            previous_state = lifecycle.snapshot.state
            state = previous_state
            if state in {
                VoiceLifecycleState.LOCAL_LISTEN,
                VoiceLifecycleState.DEEP_SLEEP,
                VoiceLifecycleState.WARM_IDLE,
            }:
                warm_task = self._asr_warm_expiry_task
                if warm_task is not None:
                    warm_task.cancel()
                    self._asr_warm_expiry_task = None
                if state is VoiceLifecycleState.WARM_IDLE:
                    lifecycle.metrics.warm_hit_count += 1
                lifecycle.open_turn(ingress_token)
                state = lifecycle.snapshot.state
            if state is VoiceLifecycleState.PREWARMING:
                if not await self._ensure_smart_turn_ready(lifecycle, epoch):
                    return False
                asr_session = self._asr_session
                if asr_session is not None and getattr(asr_session, "is_ready", True):
                    lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
                    # session 先未就绪、随后又 ready 时，真实开口时刻是当初记下的那个
                    # pending onset —— 直接用当前时钟会把整段重连等待算成「开口之后」，
                    # 期间拍的帧全被排除在本回合外（CodeRabbit Major）。
                    self._asr_turn_onset_at = (
                        self._asr_pending_speech_onset_at
                        if self._asr_pending_speech_onset_at is not None
                        else detected_at
                    )
                    self._asr_pending_speech_confirmed = False
                    self._asr_pending_speech_onset_at = None
                else:
                    self._asr_pending_speech_confirmed = True
                    if self._asr_pending_speech_onset_at is None:
                        self._asr_pending_speech_onset_at = detected_at
            if lifecycle.snapshot.state is not previous_state:
                identity = self._capture_runtime_identity(
                    ingress_token=self._asr_current_ingress_token,
                )
                delivered = await self._send_asr_lifecycle_state(
                    lifecycle.snapshot.state,
                    provider=provider,
                    session_epoch=epoch,
                    expected_identity=identity,
                )
                if not delivered:
                    return False
            if (
                lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
                and previous_state is not VoiceLifecycleState.ACTIVE
            ):
                self._asr_turn_audio_started_at = time.monotonic()
                self._asr_first_partial_recorded = False
        if event not in {
            SpeechActivityEvent.SPEECH_STARTED,
            SpeechActivityEvent.SPEECH_RESUMED,
        }:
            if event is SpeechActivityEvent.CANDIDATE_PAUSE:
                # Once local VAD observes a pause, a later provider final may
                # simply be the current utterance ending, so replaying the
                # remembered onset at that final would wake a ghost turn. The
                # onset must not be dropped outright either: when the pause
                # closes a genuine overlapping utterance, its provider endpoint
                # and final are still queued in the ordered FIFO behind the
                # previous turn's final. Convert the onset into a
                # completed-overlap credit; only a provider endpoint arriving
                # in WARM_IDLE proves a queued turn exists and redeems it.
                onset_token = self._asr_overlap_onset_token
                onset_at = self._asr_overlap_onset_at
                self._asr_overlap_onset_token = None
                self._asr_overlap_onset_at = None
                if onset_token is not None:
                    # 一张 credit 配一个时刻，按兑付顺序排队。
                    self._asr_overlap_completed_onsets.append(
                        onset_at if onset_at is not None else detected_at
                    )
                    if onset_token == self._asr_overlap_completed_token:
                        # Each additional onset+pause cycle observed while the
                        # first turn stays ACTIVE queues one more provider
                        # endpoint/final pair, so count credits per cycle.
                        self._asr_overlap_completed_turns += 1
                    else:
                        # 换了 ingress 身份：旧队列作废，只留这一张。
                        last = self._asr_overlap_completed_onsets.pop()
                        self._asr_overlap_completed_onsets.clear()
                        self._asr_overlap_completed_onsets.append(last)
                        self._asr_overlap_completed_token = onset_token
                        self._asr_overlap_completed_turns = 1
            return False
        if self._asr_turn_prepared:
            if (
                lifecycle is not None
                and lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
                and lifecycle.provider_policy.endpoint_authority == "provider"
            ):
                # Provider-VAD endpoints ride the ordered callback FIFO right
                # before their own final, so a genuine next-turn onset can
                # reach Core while the previous turn is still ACTIVE and
                # prepared. Remember the onset (ingress-fenced) so the delayed
                # final can replay it instead of dropping the next turn.
                self._asr_overlap_onset_token = self._asr_current_ingress_token
                self._asr_overlap_onset_at = detected_at
                return event is SpeechActivityEvent.SPEECH_RESUMED
            return False
        if (
            lifecycle is not None
            and lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
        ):
            return False

        await self._prepare_independent_asr_turn(epoch)
        return False

    async def _prepare_independent_asr_turn(self, epoch: int) -> None:
        """Reserve Core and admission ownership before observations can arrive."""

        if epoch != self._asr_session_epoch or self._asr_turn_prepared:
            return
        lifecycle = self._asr_lifecycle
        if (
            lifecycle is None
            or lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
        ):
            return
        turn_token = self._capture_turn_token(lifecycle)
        identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        operation_generation = self._asr_start_generation
        transcript_dispatcher = self._asr_transcript_dispatcher

        def preparation_is_current() -> bool:
            return bool(
                not self._asr_terminal_close_requested
                and self._asr_start_operation_matches(operation_generation)
                and self._runtime_identity_matches(identity)
                and self._asr_transcript_dispatcher is transcript_dispatcher
            )

        try:
            if not self._asr_admission_ingress_started:
                await self._asr_admission_ingress.start()
                self._asr_admission_ingress_started = True
                if not preparation_is_current():
                    return
            await self._asr_admission_ingress.open_turn(turn_token)
        except Exception:
            await self._handle_independent_asr_error(
                epoch,
                self._asr_provider or "unknown",
                status_code="ASR_CORE_TRANSCRIPT_BACKPRESSURE",
                expected_identity=identity,
            )
            return
        if not preparation_is_current():
            # Detach already queued RouteReplaced on this lane; it owns stale
            # record retirement. Never reserve into the replacement dispatcher.
            return
        final_key = FinalKey.from_turn(turn_token)
        if not transcript_dispatcher.try_reserve(final_key):
            await self._post_admission_event(turn_token, Reset())
            await self._handle_independent_asr_error(
                epoch,
                self._asr_provider or "unknown",
                status_code="ASR_CORE_TRANSCRIPT_BACKPRESSURE",
                expected_identity=identity,
            )
            return
        self._asr_admission_reservation_dispatchers[
            final_key
        ] = transcript_dispatcher
        self._asr_turn_prepared = True

        async def abandon_preparation() -> None:
            if (
                self._runtime_identity_matches(identity)
                and self._asr_transcript_dispatcher is transcript_dispatcher
            ):
                self._asr_turn_prepared = False
                if self._asr_partial_turn_token == turn_token:
                    self._asr_partial_turn_token = None
            try:
                await self._post_admission_event(turn_token, Reset())
            except (AdmissionIngressClosedError, KeyError):
                pass

        try:
            accepted = await self._callbacks.on_prepare_turn(turn_token)
        except asyncio.CancelledError:
            await abandon_preparation()
            raise
        except Exception:
            accepted = False
            if self._runtime_identity_matches(identity):
                logger.warning(
                    "[%s] independent ASR turn preparation failed",
                    self.display_name,
                )
        if accepted and preparation_is_current():
            self._asr_partial_turn_token = turn_token
            return
        await abandon_preparation()

    def _consume_overlap_completed_credit(self) -> None:
        """Retire one redeemed completed-overlap credit and its onset."""

        self._asr_overlap_completed_turns -= 1
        if self._asr_overlap_completed_onsets:
            self._asr_overlap_completed_onsets.popleft()
        if self._asr_overlap_completed_turns == 0:
            self._asr_overlap_completed_token = None
            self._asr_overlap_completed_onsets.clear()

    @staticmethod
    def _provider_key_namespace(
        key: ProviderUtteranceKey,
    ) -> tuple[int, int]:
        return key.generation, key.buffer_epoch

    def _accept_provider_timeline(self, key: ProviderUtteranceKey) -> bool:
        """Accept one current/new Provider namespace and reject stale epochs."""

        namespace = self._provider_key_namespace(key)
        current = self._asr_provider_correlator_namespace
        if current is not None and namespace < current:
            return False
        if current != namespace:
            previous = self._asr_provider_correlator
            if previous is not None and current is not None:
                try:
                    retired = previous.retire_namespace(current)
                except ProviderAliasConflictError:
                    retired = None
                if retired is not None and retired.retired_proofs:
                    task = asyncio.create_task(
                        self._retire_admission_boundary_proofs(
                            retired.retired_proofs,
                            self._asr_detector,
                        ),
                        name="provider-boundary-namespace-retirement",
                    )
                    self._track_admission_effect_task(task, None)
                    task.add_done_callback(self._admission_effect_done)
            self._asr_provider_correlator_namespace = namespace
            self._asr_provider_correlator = ProviderTurnCorrelator(
                namespace=namespace,
                proof_capacity=_MAX_PROVIDER_BOUNDARY_SNAPSHOTS,
            )
        return True

    def _provider_key_timeline_is_current(
        self,
        key: ProviderUtteranceKey,
    ) -> bool:
        return (
            self._asr_provider_correlator_namespace
            == self._provider_key_namespace(key)
        )

    async def _retire_provider_speaker_boundary_unknown(
        self,
        detector: DetectorRuntime,
        identity: _AsrRuntimeIdentity,
        verdict: ProviderSpeakerBoundarySnapshot | None = None,
    ) -> tuple[bool, ProviderSpeakerBoundarySnapshot | None]:
        retire = getattr(
            detector,
            "retire_provider_speaker_boundary_unknown",
            None,
        )
        retired: ProviderSpeakerBoundarySnapshot | None = None
        if callable(retire):
            try:
                result = await retire(verdict)
                if type(result) is ProviderSpeakerBoundarySnapshot:
                    retired = result
            except asyncio.CancelledError:
                raise
            except Exception:
                # Speaker ownership is advisory. A cleanup failure must not
                # turn an unknown boundary into an ASR transport failure.
                pass
        return self._runtime_identity_matches(identity), retired

    async def _handle_provider_endpoint_notification(
        self,
        notification: ProviderEndpointNotification,
        epoch: int,
    ) -> None:
        """Reconcile raw boundaries early and seal keyed text turns in order."""

        if (
            type(notification) is not ProviderEndpointNotification
            or epoch != self._asr_session_epoch
        ):
            return
        if notification.phase == "boundary":
            await self._handle_provider_boundary_notification(
                notification,
                epoch,
            )
            return
        await self._handle_ordered_provider_endpoint(
            notification,
            epoch,
        )

    async def _handle_provider_boundary_notification(
        self,
        notification: ProviderEndpointNotification,
        epoch: int,
    ) -> None:
        key = notification.key
        if not self._accept_provider_timeline(key):
            return
        correlator = self._asr_provider_correlator
        detector = self._asr_detector
        if correlator is None or detector is None:
            return
        identity = self._capture_runtime_identity()
        deadline = time.monotonic() + _PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS
        snapshot: ProviderSpeakerBoundarySnapshot | None = None
        exact = False
        try:
            if (
                notification.boundary_quality == "exact"
                and notification.audio_range is not None
                and self._asr_session is self._asr_provider_exact_session
            ):
                remaining = deadline - time.monotonic()
                observed = bool(
                    remaining > 0
                    and await asyncio.wait_for(
                        detector.wait_provider_audio_observed_through(
                            notification.audio_range.end_sample_16k
                        ),
                        timeout=remaining,
                    )
                )
                remaining = deadline - time.monotonic()
                if observed and remaining > 0:
                    snapshot = await asyncio.wait_for(
                        detector.reconcile_provider_endpoint(
                            notification.audio_range
                        ),
                        timeout=remaining,
                    )
                remaining = deadline - time.monotonic()
                if type(snapshot) is ProviderSpeakerBoundarySnapshot and remaining > 0:
                    exact = bool(
                        await asyncio.wait_for(
                            detector.wait_provider_speaker_preseal(
                                snapshot,
                                deadline=deadline,
                            ),
                            timeout=remaining,
                        )
                        and time.monotonic() < deadline
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            exact = False
        if not self._runtime_identity_matches(identity):
            return
        result = ProviderBoundaryResult.unknown()
        if exact and snapshot is not None and notification.audio_range is not None:
            self._asr_provider_boundary_proof_sequence += 1
            proof = BoundaryProof(
                proof_id=self._asr_provider_boundary_proof_sequence,
                owner_generation=self._asr_admission_capability_generation,
                provider_key=key,
            )
            self._asr_provider_boundary_proofs[proof.proof_id] = snapshot
            result = ProviderBoundaryResult(
                quality="exact",
                audio_range=notification.audio_range,
                proof=proof,
            )
        existing_boundary = correlator.record_for(key)
        recorded = correlator.record_boundary_result(key, result)
        if (
            result.quality == "exact"
            and recorded.quality == "unknown"
            and existing_boundary is None
        ):
            self._speaker_rejection_metrics[
                "admission_boundary_proof_overflow_count"
            ] += 1
        await self._retire_admission_boundary_proofs(
            recorded.retired_proofs,
            detector,
        )
        self._speaker_rejection_metrics[
            "provider_boundary_exact_ready_count"
            if recorded.quality == "exact"
            else "provider_boundary_unknown_ready_count"
        ] += 1

    async def _handle_ordered_provider_endpoint(
        self,
        notification: ProviderEndpointNotification,
        epoch: int,
        *,
        deadline: float | None = None,
    ) -> None:
        if deadline is None:
            deadline = (
                time.monotonic()
                + _PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS
            )
        key = notification.key
        if not self._accept_provider_timeline(key):
            return
        correlator = self._asr_provider_correlator
        if correlator is None or correlator.is_completed(key):
            return
        try:
            alias = correlator.mark_ordered(key)
        except ProviderAliasConflictError:
            return
        boundary = alias.boundary_result or ProviderBoundaryResult.unknown()
        if alias.boundary_result is None:
            self._speaker_rejection_metrics[
                "provider_boundary_ordered_jit_unknown_count"
            ] += 1
        snapshot = None
        proof = boundary.proof
        if (
            boundary.quality == "exact"
            and proof is not None
            and boundary.audio_range == notification.audio_range
        ):
            snapshot = self._asr_provider_boundary_proofs.get(proof.proof_id)
        await self._handle_independent_asr_endpoint(
            epoch,
            provider_key=key,
            provider_snapshot=snapshot,
            deadline=deadline,
        )

    async def _handle_provider_final(
        self,
        key: ProviderUtteranceKey,
        text: str,
        epoch: int,
        provider: str,
        *,
        received_at: float | None = None,
        admission_deadline: float | None = None,
    ) -> None:
        # Compatibility for existing private test/integration callers. The
        # production session callback always supplies the first-receipt pair;
        # therefore an out-of-order final never regains this budget here.
        if received_at is None and admission_deadline is None:
            received_at = time.monotonic()
            admission_deadline = (
                received_at + _PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS
            )
        if received_at is None or admission_deadline is None:
            return
        if admission_deadline < received_at:
            return
        final_deadline = admission_deadline
        if (
            type(key) is not ProviderUtteranceKey
            or epoch != self._asr_session_epoch
            or not self._accept_provider_timeline(key)
        ):
            return
        correlator = self._asr_provider_correlator
        if correlator is None or correlator.is_completed(key):
            return
        if self._asr_sealed_provider_key != key:
            await self._handle_ordered_provider_endpoint(
                ProviderEndpointNotification(
                    phase="ordered",
                    generation=key.generation,
                    buffer_epoch=key.buffer_epoch,
                    utterance_id=key.utterance_id,
                    boundary_quality="unknown",
                    audio_range=None,
                ),
                epoch,
                deadline=final_deadline,
            )
        if (
            epoch != self._asr_session_epoch
            or self._asr_sealed_provider_key != key
        ):
            return
        settled = await self._handle_independent_asr_final(
            text,
            epoch,
            provider,
            provider_key=key,
            received_at=received_at,
            deadline=final_deadline,
        )
        if settled is not None:
            await settled.wait()

    async def _seal_independent_asr_provider_turn_transaction(
        self,
        epoch: int,
        *,
        provider_key: ProviderUtteranceKey | None,
        provider_snapshot: ProviderSpeakerBoundarySnapshot | None,
        deadline: float | None,
    ) -> tuple[
        _ProviderTurnSealTransaction | None,
        str | None,
        _AsrRuntimeIdentity | None,
    ]:
        """Linearize Provider seal and optional reject under Core -> Detector."""

        async with self._asr_final_lock:
            lifecycle = self._asr_lifecycle
            detector = self._asr_detector
            if (
                epoch != self._asr_session_epoch
                or lifecycle is None
                or detector is None
                or self._asr_session is None
                or lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
                or not self._asr_turn_prepared
                or (
                    provider_key is not None
                    and not self._provider_key_timeline_is_current(provider_key)
                )
            ):
                return None, None, None
            try:
                turn_token = self._capture_turn_token(lifecycle)
            except Exception:
                return None, None, None
            identity = self._capture_runtime_identity(
                ingress_token=turn_token.ingress,
                turn_token=turn_token,
            )
            if not self._asr_endpointing_ready(lifecycle, detector, turn_token):
                return None, "ASR_BLOCKED_ENDPOINTING", identity
            final_key = FinalKey.from_turn(turn_token)
            if provider_key is not None:
                correlator = self._asr_provider_correlator
                if correlator is None:
                    await self._post_admission_event(turn_token, Reset())
                    return None, "ASR_ENDPOINTING_FAILED", identity
                try:
                    correlator.bind_ordered(provider_key, turn_token)
                    await self._post_admission_event(
                        turn_token,
                        ProviderBound(provider_key),
                    )
                except (ProviderAliasConflictError, KeyError):
                    await self._post_admission_event(turn_token, Reset())
                    return None, "ASR_ENDPOINTING_FAILED", identity

            provider_fence: ProviderCandidateFence | None = None
            if not _uses_smart_turn_endpointing(lifecycle.provider_policy):
                seal_started_at = time.monotonic()
                seal_budget = (
                    max(0.0, deadline - seal_started_at)
                    if deadline is not None
                    else None
                )
                try:
                    if provider_key is None:
                        provider_fence = await detector.seal_provider_candidate(
                            turn_token,
                            deadline=deadline,
                        )
                    else:
                        provider_fence = await detector.seal_provider_candidate(
                            turn_token,
                            speaker_snapshot=provider_snapshot,
                            deadline=deadline,
                        )
                except asyncio.CancelledError:
                    await self._post_admission_event(turn_token, Reset())
                    raise
                except Exception:
                    provider_fence = None
                    logger.warning(
                        "[%s] provider candidate seal failed",
                        self.display_name,
                    )
                if (
                    not self._runtime_identity_matches(identity)
                    or self._asr_lifecycle is not lifecycle
                    or self._asr_detector is not detector
                    or lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
                    or (
                        provider_key is not None
                        and not self._provider_key_timeline_is_current(provider_key)
                    )
                ):
                    await self._post_admission_event(turn_token, Reset())
                    return None, None, None
                seal_timed_out = bool(
                    provider_key is not None
                    and deadline is not None
                    and provider_fence is None
                    and (
                        time.monotonic() >= deadline
                        or seal_budget == 0.0
                        # Windows event-loop timers may wake slightly before
                        # the monotonic target.  Distinguish that bounded wait
                        # from an immediate stale/no-candidate None without
                        # broadening every seal failure into fail-open.
                        or time.monotonic() - seal_started_at
                        >= max(0.0, (seal_budget or 0.0) - 0.02)
                    )
                )
                if provider_fence is None and not seal_timed_out:
                    await self._post_admission_event(turn_token, Reset())
                    return None, "ASR_ENDPOINTING_FAILED", identity
                self._asr_provider_candidate_fence = provider_fence
                self._asr_sealed_provider_key = provider_key

            lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
            sealed_token = self._capture_transport_token(lifecycle)
            self._asr_sealed_turn_token = sealed_token
            capability: RejectionCapability | None = None
            if provider_fence is not None and provider_key is not None:
                try:
                    admission_record = await self._asr_admission.get_record(
                        turn_token
                    )
                    candidate = (
                        admission_record.speaker_candidate
                        if admission_record is not None
                        else None
                    )
                    if candidate is None:
                        candidate = detector.pending_provider_speaker_candidate(
                            provider_fence
                        )
                    lease = (
                        await detector.prepare_candidate_rejection(candidate)
                        if candidate is not None
                        else None
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    lease = None
                if lease is not None:
                    self._asr_admission_candidate_turns[
                        lease.shadow_candidate
                    ] = turn_token
                    capability = self._register_admission_capability(
                        lease,
                        kind=RejectionCapabilityKind.SEALED,
                        provider_key=provider_key,
                    )
            if capability is None and provider_key is not None:
                await self._post_admission_event(
                    turn_token,
                    BoundaryUnknown(provider_key),
                )
            await self._post_admission_event(
                turn_token,
                TurnSealed(capability),
            )
            if provider_fence is not None:
                await self._post_admission_event(
                    turn_token,
                    MicroEventPending(),
                )
            sealed_wait_event = self._asr_admission_turn_sealed_events.get(
                turn_token
            )
            if sealed_wait_event is not None:
                sealed_wait_event.set()

            return (
                _ProviderTurnSealTransaction(
                    lifecycle=lifecycle,
                    turn_token=turn_token,
                    sealed_token=sealed_token,
                    final_key=final_key,
                    identity=identity,
                ),
                None,
                None,
            )

    async def _handle_independent_asr_endpoint(
        self,
        epoch: int,
        *,
        provider_key: ProviderUtteranceKey | None = None,
        provider_snapshot: ProviderSpeakerBoundarySnapshot | None = None,
        deadline: float | None = None,
    ) -> None:
        """Seal the current turn immediately at its semantic endpoint."""

        if epoch != self._asr_session_epoch:
            return
        provider = self._asr_provider or "unknown"
        lifecycle = self._asr_lifecycle
        if lifecycle is None:
            return
        provider_identity = (
            self._capture_runtime_identity()
            if provider_key is not None
            else None
        )

        def provider_key_is_current() -> bool:
            return bool(
                provider_key is None
                or (
                    provider_identity is not None
                    and self._runtime_identity_matches(provider_identity)
                    and self._provider_key_timeline_is_current(provider_key)
                )
            )

        if (
            provider_key is not None
            and lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
        ):
            # The Provider key, not a local resume/pause credit, authorizes a
            # new logical text turn. Existing onset metadata is only a timing
            # hint; if it is absent or stale, wake the turn without it.
            ingress_token = self._asr_current_ingress_token
            if (
                ingress_token is None
                or not self._ingress_token_matches(ingress_token)
            ):
                return
            if self._asr_overlap_completed_token != ingress_token:
                self._asr_overlap_completed_token = ingress_token
                self._asr_overlap_completed_onsets.clear()
                self._asr_overlap_completed_turns = 0
            if self._asr_overlap_completed_turns <= 0:
                onset_at = self._asr_overlap_onset_at
                if onset_at is not None:
                    self._asr_overlap_completed_onsets.append(onset_at)
                self._asr_overlap_completed_turns = 1
        if (
            lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
            and self._asr_overlap_completed_turns > 0
        ):
            completed_token = self._asr_overlap_completed_token
            if (
                completed_token is None
                or lifecycle.provider_policy.endpoint_authority != "provider"
                or completed_token != self._asr_current_ingress_token
                or not self._ingress_token_matches(completed_token)
            ):
                # The credit belongs to a superseded ingress generation (hard
                # mute, abort, or route swap rotated the token), so drop it
                # instead of waking a stale replacement turn.
                self._asr_overlap_completed_token = None
                self._asr_overlap_completed_turns = 0
                return
            # A provider endpoint reaching Core in WARM_IDLE means the ordered
            # FIFO holds a turn whose local onset and pause both happened while
            # the previous turn was still ACTIVE (its endpoint was queued
            # behind that turn's delayed final). Redeem one completed-overlap
            # credit: replay the onset so the lifecycle is ACTIVE and prepared,
            # then fall through to seal immediately, letting the queued final
            # right behind this endpoint find a DRAINING turn.
            # ⚠️ 先重放、确认真的醒过来了，**再**记账。重放可能唤不醒这一轮
            # （会话暂时不可用时停在 PREWARMING）；此时若 credit 已经扣掉，这张
            # credit 对应的 endpoint 就再也封不了口，紧随其后的 final 会被整条
            # 丢弃，而被弹出的 onset 还会被更晚的回合继承（拿错视觉窗口）。
            replay_onset_at = (
                self._asr_overlap_completed_onsets[0]
                if self._asr_overlap_completed_onsets
                else None
            )
            # 把真实开口时刻交给重放：直接确认分支会优先取 pending onset，于是
            # SPEECH_CONFIRMED 打上的是用户当初开口的时刻，而不是这次重放的时刻。
            _lent_pending_onset = False
            if (
                replay_onset_at is not None
                and self._asr_pending_speech_onset_at is None
            ):
                self._asr_pending_speech_onset_at = replay_onset_at
                _lent_pending_onset = True
            pending_before = self._asr_pending_speech_confirmed
            credit_consumed = False
            await self._handle_independent_asr_activity(
                SpeechActivityEvent.SPEECH_RESUMED,
                epoch,
            )
            if (
                epoch != self._asr_session_epoch
                or self._asr_lifecycle is not lifecycle
                or not provider_key_is_current()
            ):
                return
            if (
                not pending_before
                and self._asr_pending_speech_confirmed
                and lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
            ):
                # 重放被"传输未就绪"挡住了，停在 PREWARMING 并挂起了确认。
                # （provider 权威下 SOFT_WAKE→PREWARMING 之后拦路的就是
                # asr_session.is_ready —— _ensure_smart_turn_ready 在 provider
                # 权威下无 await 直接返回 True；PREWARMING 的 lifecycle 广播没送达
                # 是同态的另一种成因。别在注释里写死"唯一成因"。）
                #
                # 但这一轮**不需要**传输：它的音频早在上一轮还 ACTIVE 时就已经过
                # 线，endpoint 和它自己的 final 已经排在有序 FIFO 里、正要到达。
                # 而且能走到这里就说明老 session 还被认领着——_restart_transport
                # 和 _close_transport_only 都是先把 _asr_session 置 None 再 close，
                # 之后 is_adopted_candidate() 会丢掉它的全部回调——也就是说重连
                # **还没开始**，那条 final 就排在后面。等重连救不回它：重连会换新
                # session，老队列里那条 final 必定在 is_adopted_candidate() 上被
                # 丢掉。就地补完确认，让紧随其后的 final 找到一个 DRAINING 的回合。
                #
                # 这里刻意**不**走 _handle_independent_asr_error：那条出口会 bump
                # epoch、拆掉整个 session、cancel 掉正在跑的重连任务，并把语音路由
                # fail-closed 到本次会话结束——为一句其实救得回来的话把整场语音判
                # 死，违反"绝不丢用户的句子"。真丢的情况（final 始终不来）由下面封
                # 口时装上的 provider-final watchdog 兜底：10s 硬顶，且不受
                # _voice_input_resource_optimization_enabled 开关影响（那个开关会
                # 让 _schedule_transport_warm_expiry 直接 return，所以不能靠它）。
                #
                # 门里 not pending_before 是刻意的：只补偿**这次重放自己造出来的**
                # 那笔挂起确认，不吞别人的。
                #
                # 刻意不做的两件事：不调 _activate_asr_audio_dispatcher /
                # drain_active_start_audio（重连确认分支有，但这一轮的音频早已过
                # 线，本地没有待发缓冲）；不武装 _schedule_transport_warm_expiry
                # （忙窗口的界由上面那个 watchdog 提供）。将来若有人让这条路承接
                # 未发出的 PCM，必须回来补第一条。
                lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
                self._asr_turn_onset_at = (
                    self._asr_pending_speech_onset_at
                    if self._asr_pending_speech_onset_at is not None
                    else time.monotonic()
                )
                # 与 _restart_transport 的补确认块同序：确认一落地就把挂起状态
                # 清掉。真实开口时刻已经装进 _asr_turn_onset_at，下面那个 await
                # 无论怎么返回都不会把它丢掉。
                self._asr_pending_speech_confirmed = False
                self._asr_pending_speech_onset_at = None
                # 这张 credit 就是被这次确认兑走的，账要跟着确认一起落。留到
                # 下面记的话，身份漂移那条 return 会把它跳过：这一轮照常在替换后的
                # 传输上封口，而陈旧的 credit 与 onset 还压在队列里 —— 后面真实的
                # overlap 排在它后面，兑付时拿到错的 onset，多出来的那张还会让某个
                # endpoint 重放到不属于它的回合上，把一条 final 丢掉。
                self._consume_overlap_completed_credit()
                credit_consumed = True
                self._asr_turn_audio_started_at = time.monotonic()
                self._asr_first_partial_recorded = False
                confirm_identity = self._capture_runtime_identity(
                    ingress_token=self._asr_current_ingress_token,
                )
                delivered = await self._send_asr_lifecycle_state(
                    VoiceLifecycleState.ACTIVE,
                    provider=provider,
                    session_epoch=epoch,
                    expected_identity=confirm_identity,
                )
                if (
                    not delivered
                    or epoch != self._asr_session_epoch
                    or self._asr_lifecycle is not lifecycle
                ):
                    # delivered 为假只可能是运行时身份漂移：_send_asr_lifecycle_state
                    # 吞掉回调异常之后，返回的就是 _runtime_identity_matches。而
                    # _restart_transport / _close_transport_only 换掉 _asr_session 与
                    # transport_generation 时都不走 _reset_asr_turn_state，所以这里留下
                    # 的挂起状态没人回收：上面 transition(SPEECH_CONFIRMED) 已经把它兑付
                    # 进 _asr_turn_onset_at，两个兑付点又都以 PREWARMING 为闸、ACTIVE 下
                    # 一律跳过。残留下去会被后面某个不相干的回合当成自己的开口时刻，还会
                    # 把补偿门 not pending_before 恒假化，让重叠补偿此后静默失效。
                    return
            if lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE:
                # 没唤醒。credit 原样留着等下一次兑付；借出去的 onset 也要收回，
                # 免得它被后面某个不相干的回合当成自己的起点。
                # ⚠️ 只有在**没有**挂起的确认时才收回。session 未就绪时
                # _handle_independent_asr_activity 会停在 PREWARMING、置上
                # _asr_pending_speech_confirmed 并**特意留着**这个 onset 等重连后
                # 的确认去取；此时收回等于让那次确认退回用新的 detected_at，把用户
                # 真实开口以来的帧全排除掉。
                if (
                    _lent_pending_onset
                    and not self._asr_pending_speech_confirmed
                    and self._asr_pending_speech_onset_at == replay_onset_at
                ):
                    self._asr_pending_speech_onset_at = None
                return
            # 确认 ACTIVE 之后才记账（补确认那条路已经在上面记过了）。
            if not credit_consumed:
                self._consume_overlap_completed_credit()
        if lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE:
            if not provider_key_is_current():
                return
            if not self._asr_turn_prepared:
                # A rejected preparation keeps the lifecycle ACTIVE so the
                # utterance can retry (SPEECH_RESUMED re-prepares), but Core
                # never ran the interruption/external-turn pause for this
                # turn. Re-prepare before sealing; without a successful
                # preparation the final must never reach Core, so fail
                # closed instead of sealing an unprepared turn.
                await self._prepare_independent_asr_turn(epoch)
                if (
                    epoch != self._asr_session_epoch
                    or self._asr_lifecycle is not lifecycle
                    or lifecycle.snapshot.state is not VoiceLifecycleState.ACTIVE
                    or not provider_key_is_current()
                ):
                    return
                if not self._asr_turn_prepared:
                    await self._handle_independent_asr_error(
                        epoch,
                        provider,
                        status_code="ASR_CORE_TURN_REJECTED",
                    )
                    return
            transaction, failure_status, failure_identity = (
                await self._seal_independent_asr_provider_turn_transaction(
                    epoch,
                    provider_key=provider_key,
                    provider_snapshot=provider_snapshot,
                    deadline=deadline,
                )
            )
            if failure_status is not None:
                await self._handle_independent_asr_error(
                    epoch,
                    provider,
                    status_code=failure_status,
                    expected_identity=failure_identity,
                )
                return
            if transaction is None:
                return
            lifecycle = transaction.lifecycle
            turn_token = transaction.turn_token
            self._asr_turn_endpointed_at = time.monotonic()
            self._asr_last_turn_endpointed_at = self._asr_turn_endpointed_at
            # 与 Core 侧 record.turn_id 同构（asr_runtime.py 的
            # external_turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"），
            # 好让冻结时能直接判"这个封口是不是这条 record 的"。
            self._asr_last_turn_endpointed_key = (
                f"asr-{turn_token.ingress.session_epoch}-{turn_token.turn_id}"
            )
            self._schedule_provider_final_watchdog(
                epoch,
                lifecycle,
                transaction.sealed_token,
            )
            await self._send_asr_lifecycle_state(
                VoiceLifecycleState.DRAINING,
                provider=provider,
                session_epoch=epoch,
                expected_identity=transaction.identity,
            )

    async def _activate_pending_independent_turn(self, epoch: int) -> None:
        """Start the pending turn after the previous final completes."""

        if epoch != self._asr_session_epoch:
            return
        lifecycle = self._asr_lifecycle
        if lifecycle is None or not lifecycle.has_pending_turn:
            # has_pending_turn 还要求 pending buffer 里真有音频：speech 先到、或者
            # 对应 PCM 被丢弃时会走到这里。不清的话这个 onset 会被**下一个**真实
            # pending turn 复用，把那一轮的起点提前到上一轮，视觉帧绑错回合。
            self._asr_pending_turn_onset_at = None
            if lifecycle is not None:
                lifecycle.discard_unconfirmed_pending_audio()
            return
        if lifecycle.snapshot.state is not VoiceLifecycleState.WARM_IDLE:
            lifecycle.discard_pending_turn()
            self._asr_pending_turn_onset_at = None
            self._asr_pending_detector_candidate = None
            return
        payload = lifecycle.begin_pending_turn()
        # begin_pending_turn() 内部完成 SPEECH_CONFIRMED 迁移（lifecycle.py），是第
        # 五个迁移点 —— 之前给另外四处补 onset 打点时漏了它，因为守卫只扫本模块的
        # 字面量。不补的话 _asr_turn_onset_at 还留着**上一轮**的值（它只在
        # close/abort/error 才清），Core 会拿上一轮的 onset 当本回合 started_at，于是
        # 上一轮保留的封口时刻反过来成了本回合的截止点，本回合之后拍的每一帧都被
        # accepts() 拒掉 —— 整轮退化成纯文本。
        self._asr_turn_onset_at = (
            self._asr_pending_turn_onset_at
            if self._asr_pending_turn_onset_at is not None
            else time.monotonic()
        )
        self._asr_pending_turn_onset_at = None
        if not payload:
            return
        turn_token = self._capture_turn_token(lifecycle)
        pending_candidate = self._asr_pending_detector_candidate
        self._asr_pending_detector_candidate = None
        identity = self._capture_runtime_identity(
            ingress_token=turn_token.ingress,
            turn_token=turn_token,
        )
        self._asr_turn_audio_started_at = time.monotonic()
        self._asr_first_partial_recorded = False
        if not await self._ensure_smart_turn_ready(lifecycle, epoch):
            return
        if not self._runtime_identity_matches(identity):
            return
        delivered = await self._send_asr_lifecycle_state(
            VoiceLifecycleState.ACTIVE,
            provider=identity.provider or "unknown",
            session_epoch=epoch,
            expected_identity=identity,
        )
        if not delivered:
            return
        await self._prepare_independent_asr_turn(epoch)
        if not self._runtime_identity_matches(identity):
            return
        asr_session = identity.session
        if asr_session is None or not getattr(asr_session, "is_ready", True):
            await self._handle_independent_asr_error(
                identity.session_epoch,
                identity.provider or "unknown",
                expected_identity=identity,
            )
            return
        detector = identity.detector
        if not self._asr_endpointing_ready(lifecycle, detector, turn_token):
            await self._handle_independent_asr_error(
                identity.session_epoch,
                identity.provider or "unknown",
                status_code="ASR_BLOCKED_ENDPOINTING",
                expected_identity=identity,
            )
            return
        if pending_candidate is not None:
            assert detector is not None
            try:
                bound = await detector.bind_candidate(pending_candidate, turn_token)
            except asyncio.CancelledError:
                raise
            except Exception:
                bound = None
            if not self._runtime_identity_matches(identity):
                return
            if bound is None and _uses_smart_turn_endpointing(
                lifecycle.provider_policy
            ):
                await self._handle_independent_asr_error(
                    identity.session_epoch,
                    identity.provider or "unknown",
                    status_code="ASR_ENDPOINTING_FAILED",
                    expected_identity=identity,
                )
                return
        elif not self._runtime_identity_matches(identity):
            return
        if not await self._activate_asr_audio_dispatcher(
            lifecycle,
            turn_token,
            buffered_pcm16=payload,
        ):
            await self._handle_independent_asr_error(
                identity.session_epoch,
                identity.provider or "unknown",
                status_code="ASR_AUDIO_ORDERING_FAILED",
                expected_identity=identity,
            )
            return
        if not self._runtime_identity_matches(identity):
            return
        self._asr_received_audio = True
        self._asr_audio_bytes += len(payload)

    async def _send_independent_asr_preview(self, text: str, epoch: int) -> None:
        """Send display-only ASR partials without writing conversation history."""

        clean = str(text or "").strip()
        if not clean or epoch != self._asr_session_epoch:
            return
        turn_token = self._asr_partial_turn_token
        if turn_token is None or not self._partial_turn_is_current(turn_token):
            return
        settlement = self._asr_partial_settlements.get(turn_token)
        if settlement is not None:
            if settlement[1] is AdmissionDisposition.FORWARD:
                await self._deliver_independent_asr_preview(turn_token, clean)
            return
        if (
            self._speaker_verifier_enforces_admission
            or turn_token in self._asr_speaker_authoritative_turns
        ):
            self._asr_quarantined_partials[turn_token] = clean
            self._speaker_rejection_metrics[
                "speaker_partial_quarantined_count"
            ] += 1
            return
        await self._deliver_independent_asr_preview(turn_token, clean)

    async def _handle_independent_asr_final(
        self,
        text: str,
        epoch: int,
        provider: str,
        *,
        provider_key: ProviderUtteranceKey | None = None,
        received_at: float | None = None,
        deadline: float | None = None,
    ) -> asyncio.Event | None:
        """Publish one immutable final; admission owns every disposition."""

        if epoch != self._asr_session_epoch:
            return
        if received_at is None:
            received_at = (
                deadline - _PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS
                if deadline is not None
                else time.monotonic()
            )
        if deadline is None:
            deadline = (
                received_at + _PROVIDER_BOUNDARY_SETTLEMENT_TIMEOUT_SECONDS
            )
        if deadline < received_at:
            return
        async with self._asr_final_lock:
            lifecycle = self._asr_lifecycle
            sealed_token = self._asr_sealed_turn_token
            if (
                epoch != self._asr_session_epoch
                or lifecycle is None
                or sealed_token is None
                or lifecycle.snapshot.state is not VoiceLifecycleState.DRAINING
                or not self._transport_token_matches(sealed_token, lifecycle)
                or (
                    provider_key is not None
                    and self._asr_sealed_provider_key != provider_key
                )
            ):
                return
            turn_token = sealed_token.turn
            admission_record = await self._asr_admission.get_record(turn_token)
            if (
                epoch != self._asr_session_epoch
                or self._asr_lifecycle is not lifecycle
                or self._asr_sealed_turn_token != sealed_token
            ):
                return
            if (
                admission_record is None
                or admission_record.terminal_disposition is not None
            ):
                settled = asyncio.Event()
                settled.set()
                return settled
            if turn_token in self._asr_admission_final_contexts:
                return
            final_key = FinalKey.from_turn(turn_token)
            pending = PendingProviderFinal(
                provider_key=provider_key,
                provider=provider,
                text=str(text or "").strip(),
                received_at=received_at,
                admission_deadline=deadline,
            )
            correlator = self._asr_provider_correlator
            if provider_key is not None:
                if correlator is None:
                    return
                try:
                    correlator.record_final(provider_key, pending)
                except ProviderAliasConflictError:
                    return
            detector = self._asr_detector
            provider_fence = self._asr_provider_candidate_fence
            if (
                detector is not None
                and provider_fence is not None
                and not _uses_smart_turn_endpointing(lifecycle.provider_policy)
            ):
                try:
                    micro = detector.sealed_provider_micro_event_decision(
                        provider_fence
                    )
                except Exception:
                    micro = None
                if type(micro) is ProviderMicroEventDecision:
                    await self._post_admission_event(
                        turn_token,
                        (
                            MicroEventSuppressed()
                            if micro.suppress
                            else MicroEventAllowed(
                                shadow_would_suppress=micro.would_suppress
                            )
                        ),
                    )
                else:
                    await self._post_admission_event(
                        turn_token,
                        MicroEventUnavailable(),
                    )
            identity = self._capture_runtime_identity(
                ingress_token=turn_token.ingress,
                turn_token=turn_token,
            )
            context = _AdmissionFinalContext(
                turn_token=turn_token,
                final_key=final_key,
                epoch=epoch,
                provider=provider,
                provider_key=provider_key,
                lifecycle=lifecycle,
                detector=detector,
                correlator=correlator,
                sealed_token=sealed_token,
                provider_fence=provider_fence,
                runtime_identity=identity,
                has_pending_turn=lifecycle.has_pending_turn,
            )
            self._asr_admission_final_contexts[turn_token] = context
            watchdog = self._asr_final_watchdog_task
            self._asr_final_watchdog_task = None
            if watchdog is not None and watchdog is not asyncio.current_task():
                watchdog.cancel()
            try:
                effects = await self._post_admission_event(
                    turn_token,
                    ProviderFinalReceived(pending),
                    now=received_at,
                )
            except (AdmissionIngressClosedError, KeyError):
                if self._asr_admission_final_contexts.get(turn_token) is context:
                    self._asr_admission_final_contexts.pop(turn_token, None)
                context.settled.set()
                return context.settled
            except Exception:
                if self._asr_admission_final_contexts.get(turn_token) is context:
                    self._asr_admission_final_contexts.pop(turn_token, None)
                context.settled.set()
                raise
            if not any(isinstance(effect, ResolveReserved) for effect in effects):
                admission_record = await self._asr_admission.get_record(turn_token)
                if admission_record is None:
                    if (
                        self._asr_admission_final_contexts.get(turn_token)
                        is context
                    ):
                        self._asr_admission_final_contexts.pop(turn_token, None)
                    context.settled.set()
                elif admission_record.terminal_disposition is not None:
                    ticket = admission_record.resolution_ticket
                    if (
                        ticket is not None
                        and ticket.disposition
                        in {
                            AdmissionDisposition.DROP,
                            AdmissionDisposition.ABANDON,
                        }
                    ):
                        # The terminal transition may have raced this final
                        # through another ingress consumer. Re-submit the same
                        # immutable ticket so whichever executor wins owns the
                        # late reservation and context; setdefault makes the
                        # duplicate idempotent.
                        resolver = asyncio.create_task(
                            self._resolve_admission_reservation(
                                ResolveReserved(ticket=ticket, final=None)
                            ),
                            name="voice-turn-admission-late-final-resolution",
                        )
                        self._track_admission_effect_task(resolver, turn_token)
                        resolver.add_done_callback(self._admission_effect_done)
                    else:
                        if (
                            self._asr_admission_final_contexts.get(turn_token)
                            is context
                        ):
                            self._asr_admission_final_contexts.pop(
                                turn_token,
                                None,
                            )
                        context.settled.set()
            return context.settled

    async def _dispatch_asr_transcript_envelope(
        self,
        envelope: TranscriptEnvelope,
    ) -> None:
        ingress_token = envelope.turn_token.ingress
        degraded = False
        if not self._ingress_token_matches(ingress_token):
            # The envelope was accepted before the audio generation moved on,
            # so neither on_final nor a teardown path will run for this turn.
            # Release the Core-side pause keyed to it instead of leaking the
            # pause until the next turn.
            await self._notify_asr_turn_abandoned(envelope.turn_token)
            degraded = True
            execution = self._asr_admission_resolutions.get(envelope.final_key)
            if execution is not None and not execution.core_settled:
                execution.core_settled = True
                try:
                    await self._post_admission_event(
                        envelope.turn_token,
                        CoreSettled(execution.ticket, degraded=True),
                    )
                except KeyError:
                    pass
                if execution.settled.is_set():
                    self._asr_admission_resolutions.pop(
                        envelope.final_key,
                        None,
                    )
        else:
            identity = self._capture_runtime_identity(
                ingress_token=ingress_token,
                turn_token=envelope.turn_token,
            )
            try:
                await self._callbacks.on_final(
                    VoiceTranscriptEvent(
                        turn_token=envelope.turn_token,
                        provider=envelope.provider,
                        text=envelope.text,
                    )
                )
            except asyncio.CancelledError:
                degraded = True
                raise
            except Exception:
                degraded = True
                await self._send_asr_status(
                    "ASR_INDEPENDENT_INJECTION_FAILED",
                    envelope.provider,
                    session_epoch=ingress_token.session_epoch,
                    expected_identity=identity,
                )
            finally:
                execution = self._asr_admission_resolutions.get(
                    envelope.final_key
                )
                if execution is not None and not execution.core_settled:
                    execution.core_settled = True
                    try:
                        await self._post_admission_event(
                            envelope.turn_token,
                            CoreSettled(execution.ticket, degraded=degraded),
                        )
                    except KeyError:
                        pass
                    if execution.settled.is_set():
                        self._asr_admission_resolutions.pop(
                            envelope.final_key,
                            None,
                        )

    async def _handle_independent_asr_error(
        self,
        epoch: int,
        provider: str,
        *,
        status_code: str = "ASR_INDEPENDENT_FAILED",
        expected_identity: _AsrRuntimeIdentity | None = None,
    ) -> None:
        if epoch != self._asr_session_epoch or (
            expected_identity is not None
            and not self._runtime_identity_matches(expected_identity)
        ):
            return
        # The provider callback that reported failure must not be allowed to
        # deliver a queued final into the surviving Omni session.
        self._asr_session_epoch += 1
        failure_epoch = self._asr_session_epoch
        self._asr_audio_generation += 1
        transcript_dispatcher = self._asr_transcript_dispatcher
        detector_dispatcher = self._asr_detector_dispatcher
        audio_dispatcher = self._asr_audio_dispatcher
        if self._asr_admission_ingress_started:
            await self._finish_admission_invalidation(
                self._asr_admission_ingress.invalidate_all_nowait(
                    RouteReplaced()
                ),
                transcript_dispatcher,
                self._asr_provider_correlator,
                self._asr_provider_correlator_namespace,
                self._asr_detector,
            )
        else:
            transcript_dispatcher.invalidate_all()
        detector_dispatcher.invalidate_all()
        audio_dispatcher.abort()
        self._asr_transcript_dispatcher = self._new_asr_transcript_dispatcher()
        self._asr_detector_dispatcher = AsrDetectorDispatcher(
            self._dispatch_asr_detector_event,
            on_failure=self._handle_asr_detector_dispatcher_failure,
        )
        self._asr_audio_dispatcher = AsrAudioDispatcher(
            validator=self._asr_audio_command_is_valid,
            on_wire_audio=self._record_asr_dispatcher_wire_audio,
            on_failure=self._handle_asr_audio_dispatcher_failure,
        )
        asr_session, self._asr_session = self._asr_session, None
        lifecycle, self._asr_lifecycle = self._asr_lifecycle, None
        detector, self._asr_detector = self._asr_detector, None
        lease, self._asr_smart_turn_lease = self._asr_smart_turn_lease, None
        self._asr_provider = None
        self._asr_session_factory = None
        self._asr_transport_selection = None
        self._asr_provider_speaker_sequence = 0
        self._asr_buffered_provider_speaker_observation = None
        self._reset_asr_turn_state()
        for task_name in (
            "_asr_transport_task",
            "_asr_warm_expiry_task",
            "_asr_final_watchdog_task",
        ):
            task = getattr(self, task_name, None)
            setattr(self, task_name, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        if lifecycle is not None:
            lifecycle.stop()
        if detector is not None:
            task = asyncio.create_task(detector.close())
            self._asr_close_tasks.add(task)
            task.add_done_callback(self._asr_close_tasks.discard)
        if lease is not None:
            task = asyncio.create_task(lease.release())
            self._asr_close_tasks.add(task)
            task.add_done_callback(self._asr_close_tasks.discard)
        if asr_session is not None:
            task = asyncio.create_task(self._close_asr_session(asr_session))
            self._asr_close_tasks.add(task)
            task.add_done_callback(self._asr_close_tasks.discard)
        failure_identity = self._capture_runtime_identity()
        try:
            delivered = await self._send_asr_lifecycle_state(
                VoiceLifecycleState.BLOCKED,
                provider=provider,
                session_epoch=failure_epoch,
                expected_identity=failure_identity,
            )
            if not delivered or not self._runtime_identity_matches(failure_identity):
                return
            try:
                await self._callbacks.on_failure(
                    AsrFailureEvent(
                        code=status_code,
                        provider=provider,
                        session_epoch=failure_epoch,
                    )
                )
            except Exception:
                logger.debug(
                    "[%s] independent ASR failure callback failed",
                    self.display_name,
                )
            if not self._runtime_identity_matches(failure_identity):
                return
            await self._send_asr_status(
                status_code,
                provider,
                session_epoch=failure_epoch,
                expected_identity=failure_identity,
            )
        finally:
            # A dispatcher can report its own failure from inside its worker.
            # Let lifecycle/failure/status delivery finish before closing that
            # worker, otherwise close() can cancel the authoritative callback.
            for dispatcher in (detector_dispatcher, audio_dispatcher):
                task = asyncio.create_task(dispatcher.close())
                self._asr_close_tasks.add(task)
                task.add_done_callback(self._asr_close_tasks.discard)

    async def _close_asr_session(self, asr_session: Any) -> None:
        try:
            await asr_session.close()
        except Exception:
            logger.warning(
                "[%s] independent ASR background close failed",
                self.display_name,
            )

    async def _send_asr_status(
        self,
        code: str,
        provider: str,
        *,
        session_epoch: int,
        expected_identity: _AsrRuntimeIdentity,
    ) -> bool:
        if (
            session_epoch != expected_identity.session_epoch
            or not self._runtime_identity_matches(expected_identity)
        ):
            return False
        try:
            await self._callbacks.on_status(
                AsrStatusEvent(
                    code=code,
                    provider=provider,
                    session_epoch=session_epoch,
                )
            )
        except Exception:
            logger.debug(
                "[%s] independent ASR status delivery failed",
                self.display_name,
            )
        return self._runtime_identity_matches(expected_identity)

    async def _send_asr_lifecycle_state(
        self,
        state: VoiceLifecycleState,
        *,
        provider: str,
        session_epoch: int,
        expected_identity: _AsrRuntimeIdentity,
    ) -> bool:
        if (
            session_epoch != expected_identity.session_epoch
            or not self._runtime_identity_matches(expected_identity)
        ):
            return False
        try:
            await self._callbacks.on_lifecycle(
                AsrLifecycleNotification(
                    state=state.value,
                    provider=provider,
                    session_epoch=session_epoch,
                )
            )
        except Exception:
            logger.debug(
                "[%s] ASR lifecycle status delivery failed",
                self.display_name,
            )
        return self._runtime_identity_matches(expected_identity)
