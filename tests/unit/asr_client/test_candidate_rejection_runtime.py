from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import main_logic.asr_client.runtime as runtime_module
from main_logic.asr_client.admission.contracts import (
    AbortProviderTransport,
    AdmissionDisposition,
    AdmissionResolutionTicket,
    AdmissionState,
    ApplyRejection,
    BoundaryExact,
    CandidateBound,
    CaptureState,
    CaptureClosed,
    FinalDeadlineExpired,
    LifecycleSettled,
    MicroEventAllowed,
    MicroEventPending,
    MicroEventSuppressed,
    MicroEventUnavailable,
    PendingProviderFinal,
    ProviderBound,
    ProviderFinalReceived,
    RejectionApplied,
    RejectionCapability,
    RejectionCapabilityKind,
    ResolveReserved,
    ScheduleFinalDeadline,
    SettlePartial,
    SpeakerCheckpointKind,
    SpeakerHigh,
    SpeakerLow,
    TransportSettled,
)
from main_logic.asr_client.admission.coordinator import (
    VoiceTurnAdmissionCoordinator,
)
from main_logic.asr_client.admission.ingress import AdmissionIngressLane
from main_logic.asr_client._provider_events import (
    ProviderAudioRange,
    ProviderEndpointNotification,
    ProviderUtteranceKey,
)
from main_logic.asr_client.candidate_control import CandidateRejectionOutcome
from main_logic.asr_client.endpointing.detector import (
    DetectorCandidateKey,
    DetectorIngressIdentity,
    ProviderCandidateFence,
)
from main_logic.asr_client.endpointing.detector_runtime import (
    DetectorCandidateRejectionCommitResult,
    DetectorFeedResult,
    DetectorRuntime,
)
from main_logic.asr_client.endpointing.micro_event_policy import (
    ProviderMicroEventDecision,
)
from main_logic.asr_client.lifecycle import (
    AudioDisposition,
    FinalKey,
    VoiceInputLifecycleController,
    VoiceLifecycleEvent,
    VoiceLifecycleState,
    VoiceRouteMode,
    VoiceTransportToken,
)
from main_logic.asr_client.provider_policy import resolve_provider_policy
from main_logic.asr_client.runtime import AsrRuntimeCallbacks, IndependentAsrRuntime
from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
)
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from main_logic.voice_turn.contracts import (
    AsrSubmitStatus,
    SpeechActivityEvent,
    VoiceIngressToken,
    VoiceTurnToken,
)


class _RejectionLease:
    def __init__(
        self,
        detector: object,
        turn_token: VoiceTurnToken,
        *,
        provider_fence: ProviderCandidateFence | None = None,
    ) -> None:
        self.candidate = DetectorCandidateKey(7, 11)
        self.shadow_candidate = _shadow_candidate()
        self.turn_token = turn_token
        self.provider_fence = provider_fence
        self.provider_preseal_verdict = None
        self._detector = detector
        self.commit_calls = 0
        self.commit_result = True

    def belongs_to(self, detector: object) -> bool:
        return detector is self._detector

    def commit(self) -> bool:
        self.commit_calls += 1
        return self.commit_result

    async def commit_async(
        self,
        *,
        deadline: float | None = None,
    ) -> DetectorCandidateRejectionCommitResult:
        detector = self._detector
        detector.commit_entered.set()
        if deadline is not None and runtime_module.time.monotonic() >= deadline:
            return DetectorCandidateRejectionCommitResult.STALE
        if detector.block_commit:
            try:
                if deadline is None:
                    await detector.commit_release.wait()
                else:
                    await asyncio.wait_for(
                        detector.commit_release.wait(),
                        timeout=max(
                            0.0,
                            deadline - runtime_module.time.monotonic(),
                        ),
                    )
            except TimeoutError:
                return DetectorCandidateRejectionCommitResult.STALE
        if deadline is not None and runtime_module.time.monotonic() >= deadline:
            return DetectorCandidateRejectionCommitResult.STALE
        self.commit_calls += 1
        if not self.commit_result:
            return DetectorCandidateRejectionCommitResult.STALE
        if self.provider_fence is not None:
            return DetectorCandidateRejectionCommitResult.SEALED_APPLIED
        if self.provider_preseal_verdict is not None:
            return DetectorCandidateRejectionCommitResult.PRESEAL_READY
        return DetectorCandidateRejectionCommitResult.ACTIVE_APPLIED


class _RejectionDetector:
    def __init__(self) -> None:
        self.detector_epoch = 7
        self.lease: _RejectionLease | None = None
        self.prepare_entered = asyncio.Event()
        self.prepare_release = asyncio.Event()
        self.block_prepare = False
        self.commit_entered = asyncio.Event()
        self.commit_release = asyncio.Event()
        self.block_commit = False
        self.seal_entered = asyncio.Event()
        self.seal_release = asyncio.Event()
        self.block_seal = False
        self.reset = AsyncMock()
        self.replace_speaker_verifier = AsyncMock()
        self.close = AsyncMock()
        self.complete_provider_candidate = AsyncMock(return_value=False)
        self.sealed_provider_micro_event_decision = MagicMock(return_value=None)
        self.release_deferred_turn = AsyncMock()
        self.release_speaker_candidate_binding = MagicMock(return_value=True)
        self.endpointing_ready = MagicMock(return_value=True)
        self.observe_provider_audio_ordered = AsyncMock()
        self.observe_provider_audio = MagicMock()
        self.provisional_pending = False
        self.ready_rejection = False
        self.replace_preseal_lease_on_seal = False
        self.seal_turn_tokens: list[VoiceTurnToken | None] = []

    async def prepare_candidate_rejection(self, _candidate):
        self.prepare_entered.set()
        if self.block_prepare:
            await self.prepare_release.wait()
        return self.lease

    async def seal_provider_candidate(
        self,
        turn_token: VoiceTurnToken | None = None,
        *,
        speaker_snapshot=None,
        deadline: float | None = None,
    ):
        del speaker_snapshot
        self.seal_turn_tokens.append(turn_token)
        self.seal_entered.set()
        if self.block_seal:
            try:
                if deadline is None:
                    await self.seal_release.wait()
                else:
                    await asyncio.wait_for(
                        self.seal_release.wait(),
                        timeout=max(
                            0.0,
                            deadline - runtime_module.time.monotonic(),
                        ),
                    )
            except TimeoutError:
                return None
        if deadline is not None and runtime_module.time.monotonic() >= deadline:
            return None
        lease = self.lease
        if lease is None:
            return None
        if turn_token is not None and turn_token != lease.turn_token:
            return None
        fence = ProviderCandidateFence(7, 11, 23)
        if (
            self.replace_preseal_lease_on_seal
            and lease.provider_preseal_verdict is not None
        ):
            sealed_lease = _RejectionLease(
                self,
                lease.turn_token,
                provider_fence=fence,
            )
            self.lease = sealed_lease
            return fence
        lease.provider_fence = fence
        lease.provider_preseal_verdict = None
        return fence

    def ready_provider_speaker_rejection(self, provider_fence):
        lease = self.lease
        if (
            not self.ready_rejection
            or lease is None
            or lease.provider_fence != provider_fence
        ):
            return None
        return lease.shadow_candidate

    def pending_provider_speaker_candidate(self, provider_fence):
        lease = self.lease
        if (
            not self.provisional_pending
            or lease is None
            or lease.provider_fence != provider_fence
        ):
            return None
        return lease.shadow_candidate


def _callbacks(*, abandoned: AsyncMock | None = None) -> AsrRuntimeCallbacks:
    return AsrRuntimeCallbacks(
        display_name=lambda: "candidate-rejection-test",
        on_prepare_turn=AsyncMock(return_value=True),
        on_partial=AsyncMock(),
        on_final=AsyncMock(),
        on_turn_abandoned=abandoned or AsyncMock(),
        on_failure=AsyncMock(),
        on_status=AsyncMock(),
        on_lifecycle=AsyncMock(),
    )


def test_terminal_runtime_rejects_late_speaker_fact_before_ingress_post() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    runtime._asr_terminal_close_requested = True
    runtime._speaker_verifier_activation_generation = "terminal-generation"
    runtime._asr_admission_ingress_started = True
    runtime._asr_admission_ingress.post_nowait = MagicMock()
    candidate = _shadow_candidate()

    accepted = runtime._accept_speaker_evidence_fact(
        SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        activation_generation="terminal-generation",
        enforce=True,
    )

    assert accepted is False
    runtime._asr_admission_ingress.post_nowait.assert_not_called()
    assert runtime._asr_admission_candidate_turns == {}


def _install_active_candidate(
    runtime: IndependentAsrRuntime,
    detector: _RejectionDetector,
    *,
    provider: str = "glm",
    endpointing_mode: str = "manual",
) -> tuple[SimpleNamespace, VoiceInputLifecycleController, VoiceTurnToken]:
    lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, endpointing_mode),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    session = SimpleNamespace(
        is_ready=True,
        close=AsyncMock(),
        signal_user_activity_end=AsyncMock(),
        stream_audio=AsyncMock(),
    )
    runtime._asr_session = session
    runtime._asr_provider = provider
    runtime._asr_lifecycle = lifecycle
    runtime._asr_detector = detector
    runtime._asr_current_ingress_token = runtime.capture_ingress_token(
        connection_id="connection",
        lease_generation=1,
        route_generation=1,
    )
    turn_token = runtime._capture_turn_token(lifecycle)
    detector.lease = _RejectionLease(detector, turn_token)
    runtime._asr_partial_turn_token = turn_token
    runtime._asr_turn_prepared = True
    runtime._speaker_verifier_activation_generation = "profile-generation"
    runtime._speaker_verifier_enforces_admission = True
    assert runtime._asr_audio_dispatcher.activate(turn_token, session, b"") is True
    final_key = FinalKey.from_turn(turn_token)
    assert runtime._asr_transcript_dispatcher.try_reserve(final_key) is True
    runtime._ensure_transport_restart_task = MagicMock()
    return session, lifecycle, turn_token


async def _seal_provider_candidate(
    runtime: IndependentAsrRuntime,
    detector: _RejectionDetector,
) -> tuple[
    SimpleNamespace,
    VoiceInputLifecycleController,
    VoiceTurnToken,
    ProviderCandidateFence,
]:
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    if not runtime._asr_admission_ingress_started:
        await runtime._asr_admission_ingress.start()
        runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    runtime._asr_admission_reservation_dispatchers[FinalKey.from_turn(turn_token)] = (
        runtime._asr_transcript_dispatcher
    )
    provider_fence = _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    return session, lifecycle, turn_token, provider_fence


def _seal_installed_provider_candidate(
    runtime: IndependentAsrRuntime,
    detector: _RejectionDetector,
    session: SimpleNamespace,
    lifecycle: VoiceInputLifecycleController,
    turn_token: VoiceTurnToken,
) -> ProviderCandidateFence:
    provider_fence = ProviderCandidateFence(7, 11, 23)
    assert detector.lease is not None
    detector.lease.provider_fence = provider_fence
    assert runtime._asr_audio_dispatcher.seal(
        turn_token,
        session,
        after_sequence=0,
    )
    lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    runtime._asr_sealed_turn_token = VoiceTransportToken(
        turn=turn_token,
        transport_generation=lifecycle.snapshot.transport_generation,
    )
    runtime._asr_provider_candidate_fence = provider_fence
    return provider_fence


def _shadow_candidate() -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(7, 3, "provider_candidate")


def _smart_turn_shadow_candidate() -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(7, 3, "smart_turn_turn")


async def _open_admission_turn(
    *,
    provider_bound: bool,
    candidate: SpeakerShadowCandidateKey | None = None,
) -> tuple[
    VoiceTurnAdmissionCoordinator,
    AdmissionIngressLane,
    VoiceTurnToken,
    SpeakerShadowCandidateKey,
    ProviderUtteranceKey | None,
]:
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    turn_token = VoiceTurnToken(
        VoiceIngressToken(1, "socket", 2, 3, 4),
        1,
    )
    speaker_candidate = candidate or _shadow_candidate()
    provider_key = ProviderUtteranceKey(1, 0, 1) if provider_bound else None
    await lane.open_turn(turn_token)
    await lane.post(turn_token, CandidateBound(speaker_candidate))
    if provider_key is not None:
        await lane.post(turn_token, ProviderBound(provider_key))
    return coordinator, lane, turn_token, speaker_candidate, provider_key


def _admission_capability(
    turn_token: VoiceTurnToken,
    candidate: SpeakerShadowCandidateKey,
    *,
    provider_key: ProviderUtteranceKey | None,
    kind: RejectionCapabilityKind,
) -> RejectionCapability:
    return RejectionCapability(
        capability_id=1,
        owner_generation=7,
        kind=kind,
        turn_token=turn_token,
        candidate=candidate,
        provider_key=provider_key,
    )


def _admission_final(
    provider_key: ProviderUtteranceKey | None,
    *,
    text: str = "final",
) -> PendingProviderFinal:
    return PendingProviderFinal(provider_key, "qwen", text, 10.0, 10.2)


async def _drain_runtime_admission(runtime: IndependentAsrRuntime) -> None:
    for _ in range(20):
        tasks = {
            *runtime._asr_admission_effect_tasks,
            *runtime._asr_admission_candidate_tasks.values(),
        }
        pending = tuple(task for task in tasks if not task.done())
        if pending:
            await asyncio.gather(*pending)
            continue
        await asyncio.sleep(0)
        if not runtime._asr_admission_effect_tasks and not (
            runtime._asr_admission_candidate_tasks
        ):
            return
    raise AssertionError("admission tasks did not become idle")


async def _open_runtime_admission_turn(
    runtime: IndependentAsrRuntime,
    turn_token: VoiceTurnToken,
) -> None:
    if not runtime._asr_admission_ingress_started:
        await runtime._asr_admission_ingress.start()
        runtime._asr_admission_ingress_started = True
    if await runtime._asr_admission.get_record(turn_token) is None:
        await runtime._asr_admission_ingress.open_turn(turn_token)


async def test_enforced_partial_is_quarantined_until_forward_verdict() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )

    await runtime._send_independent_asr_preview(
        "first draft",
        runtime._asr_session_epoch,
    )
    await runtime._send_independent_asr_preview(
        "latest draft",
        runtime._asr_session_epoch,
    )

    callbacks.on_partial.assert_not_awaited()
    assert runtime._asr_quarantined_partials == {
        turn_token: "latest draft"
    }

    await runtime._execute_admission_effect(
        SettlePartial(turn_token, 1, AdmissionDisposition.FORWARD)
    )

    callbacks.on_partial.assert_awaited_once()
    assert callbacks.on_partial.await_args.args[0].text == "latest draft"
    assert runtime._asr_quarantined_partials == {}

    await runtime._send_independent_asr_preview(
        "admitted draft",
        runtime._asr_session_epoch,
    )
    assert callbacks.on_partial.await_count == 2
    assert callbacks.on_partial.await_args.args[0].text == "admitted draft"
    await _close_dispatchers(runtime)


async def test_denied_partial_is_discarded_and_never_revived() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )

    await runtime._send_independent_asr_preview(
        "blocked draft",
        runtime._asr_session_epoch,
    )
    await runtime._execute_admission_effect(
        SettlePartial(turn_token, 1, AdmissionDisposition.DROP)
    )
    await runtime._send_independent_asr_preview(
        "late blocked draft",
        runtime._asr_session_epoch,
    )

    callbacks.on_partial.assert_not_awaited()
    assert runtime._asr_quarantined_partials == {}
    await _close_dispatchers(runtime)


async def test_forward_verdict_drops_cached_partial_when_final_already_pending() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )

    await runtime._send_independent_asr_preview(
        "obsolete draft",
        runtime._asr_session_epoch,
    )
    runtime._asr_admission_final_contexts[turn_token] = SimpleNamespace()

    await runtime._execute_admission_effect(
        SettlePartial(turn_token, 1, AdmissionDisposition.FORWARD)
    )

    callbacks.on_partial.assert_not_awaited()
    assert runtime._asr_quarantined_partials == {}
    await _close_dispatchers(runtime)


async def test_unenforced_partial_preserves_immediate_preview_contract() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token = _install_active_candidate(
        runtime,
        detector,
    )
    runtime._speaker_verifier_enforces_admission = False

    await runtime._send_independent_asr_preview(
        "ordinary draft",
        runtime._asr_session_epoch,
    )

    callbacks.on_partial.assert_awaited_once()
    assert callbacks.on_partial.await_args.args[0].text == "ordinary draft"
    await _close_dispatchers(runtime)


async def test_candidate_binding_is_published_before_first_score() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )
    candidate = detector.lease.shadow_candidate
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    try:
        assert runtime._accept_speaker_candidate_binding(
            candidate,
            turn_token,
            detector=detector,
            activation_generation="profile-generation",
        )
        await _drain_runtime_admission(runtime)

        record = await runtime._asr_admission.get_record(turn_token)
        assert record is not None
        assert record.speaker_candidate == candidate
        assert runtime._asr_admission_candidate_turns[candidate] == turn_token
    finally:
        await _close_dispatchers(runtime)


async def test_deny_before_final_returns_settled_without_installing_context() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, _provider_fence = (
        await _seal_provider_candidate(runtime, detector)
    )
    candidate = detector.lease.shadow_candidate
    try:
        await runtime._asr_admission_ingress.post(
            turn_token,
            CandidateBound(candidate),
        )
        await runtime._asr_admission_ingress.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        await runtime._asr_admission_ingress.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )

        settled = await runtime._handle_independent_asr_final(
            "blocked final",
            runtime._asr_session_epoch,
            "qwen",
            received_at=10.0,
            deadline=10.2,
        )

        assert settled is not None and settled.is_set()
        assert turn_token not in runtime._asr_admission_final_contexts
        callbacks.on_final.assert_not_awaited()
    finally:
        await _close_dispatchers(runtime)


async def test_deny_interleaved_with_final_post_retires_late_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, _provider_fence = (
        await _seal_provider_candidate(runtime, detector)
    )
    candidate = detector.lease.shadow_candidate
    await runtime._asr_admission_ingress.post(
        turn_token,
        CandidateBound(candidate),
    )
    original_post = runtime._post_admission_event

    async def interleaved_post(token, event, *, now=None):
        if isinstance(event, ProviderFinalReceived):
            await runtime._asr_admission_ingress.post(
                token,
                SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
            )
            await runtime._asr_admission_ingress.post(
                token,
                SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
            )
            record = await runtime._asr_admission.get_record(token)
            assert record is not None
            assert record.resolution_ticket is not None
            runtime._asr_admission_resolutions[FinalKey.from_turn(token)] = (
                runtime_module._AdmissionResolutionExecution(
                    record.resolution_ticket,
                    core_resolution_succeeded=False,
                    owner_done=True,
                )
            )
        return await original_post(token, event, now=now)

    monkeypatch.setattr(runtime, "_post_admission_event", interleaved_post)
    try:
        settled = await runtime._handle_independent_asr_final(
            "blocked final",
            runtime._asr_session_epoch,
            "qwen",
            received_at=10.0,
            deadline=10.2,
        )

        assert settled is not None
        await _drain_runtime_admission(runtime)
        await asyncio.wait_for(settled.wait(), 0.2)
        assert turn_token not in runtime._asr_admission_final_contexts
        callbacks.on_final.assert_not_awaited()
    finally:
        await _close_dispatchers(runtime)


async def test_same_ticket_executor_adopts_context_attached_before_owner_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )
    final_key = FinalKey.from_turn(turn_token)
    dispatcher = MagicMock()
    dispatcher.resolve_reserved.return_value = True
    runtime._asr_admission_reservation_dispatchers[final_key] = dispatcher
    first_context = SimpleNamespace(settled=asyncio.Event())
    second_context = SimpleNamespace(settled=asyncio.Event())
    runtime._asr_admission_final_contexts[turn_token] = first_context
    first_settle_entered = asyncio.Event()
    release_first_settle = asyncio.Event()
    settled_contexts = []

    async def settle_context(_ticket, context) -> None:
        settled_contexts.append(context)
        if context is first_context:
            first_settle_entered.set()
            await release_first_settle.wait()

    monkeypatch.setattr(runtime, "_settle_admission_final", settle_context)
    ticket = AdmissionResolutionTicket(
        turn_token=turn_token,
        record_generation=1,
        resolution_nonce=1,
        disposition=AdmissionDisposition.DROP,
    )
    effect = ResolveReserved(ticket=ticket, final=None)
    owner = asyncio.create_task(runtime._resolve_admission_reservation(effect))
    await asyncio.wait_for(first_settle_entered.wait(), 1)
    runtime._asr_admission_final_contexts[turn_token] = second_context

    await runtime._resolve_admission_reservation(effect)
    execution = runtime._asr_admission_resolutions[final_key]
    assert execution.owner_done is False
    assert execution.late_context is second_context
    release_first_settle.set()
    await asyncio.wait_for(owner, 1)

    assert settled_contexts == [first_context, second_context]
    assert first_context.settled.is_set()
    assert second_context.settled.is_set()
    await _close_dispatchers(runtime)
















async def test_second_low_survives_capture_completion_during_arming() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        first = lane.post_nowait(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        second = lane.post_nowait(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )
        completed = lane.post_nowait(turn_token, CaptureClosed(candidate, 2))
        first_effects = await first
        second_effects = await second
        completed_effects = await completed

        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.evidence_state.value == "deny_latched"
        assert record.admission_state is AdmissionState.DROPPED
        assert sum(
            isinstance(effect, ResolveReserved)
            for effect in (*first_effects, *second_effects, *completed_effects)
        ) == 1
        assert any(
            isinstance(effect, AbortProviderTransport)
            for effect in second_effects
        )
        effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(isinstance(effect, ApplyRejection) for effect in effects)
    finally:
        await lane.close()


async def test_reject_requested_survives_completion_before_ordered_seal() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        deny_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )
        await lane.post(turn_token, CaptureClosed(candidate, 2))
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.evidence_state.value == "deny_latched"
        assert record.admission_state is AdmissionState.DROPPED
        assert [
            effect.disposition
            for effect in deny_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.DROP]

        effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(isinstance(effect, ApplyRejection) for effect in effects)
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.evidence_state.value == "deny_latched"
    finally:
        await lane.close()


async def test_pending_exact_receipt_applies_before_final_deadline() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key)),
            now=10.0,
        )
        deadline = next(
            effect
            for effect in final_effects
            if isinstance(effect, ScheduleFinalDeadline)
        )
        assert deadline.absolute_deadline == 10.2

        exact_effects = await lane.post(
            turn_token,
            BoundaryExact(capability),
            now=10.1,
        )
        assert not any(
            isinstance(effect, ApplyRejection) for effect in exact_effects
        )
        timeout_effects = await lane.post(
            turn_token,
            FinalDeadlineExpired(deadline.ticket, deadline.absolute_deadline),
            now=deadline.absolute_deadline,
        )
        assert not any(
            isinstance(effect, ResolveReserved) for effect in timeout_effects
        )
        pending = await coordinator.get_record(turn_token)
        assert pending is not None
        assert pending.admission_state is AdmissionState.PENDING

        denied_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
            now=10.21,
        )
        assert [
            effect.disposition
            for effect in denied_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.DROP]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()


@pytest.mark.parametrize(
    ("event", "expected_split", "expected_evidence_complete"),
    [
        (SpeechActivityEvent.SPEECH_STARTED, False, True),
        (SpeechActivityEvent.SPEECH_RESUMED, False, False),
    ],
)
async def test_provider_submit_observes_admitted_audio_in_dispatch_order(
    event: SpeechActivityEvent,
    expected_split: bool,
    expected_evidence_complete: bool,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=41,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(event,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    pcm16 = b"\x09\x00" * 160

    result = await runtime.submit(
        ProcessedVoiceFrame(pcm16, 16_000, 0.9, True),
        ingress_token=turn_token.ingress,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    assert result.status is AsrSubmitStatus.ACCEPTED
    detector.observe_provider_audio_ordered.assert_awaited_once_with(
        pcm16,
        sample_rate_hz=16_000,
        identity=detector_identity,
        sequence_no=1,
        split_before_audio=expected_split,
        evidence_complete=expected_evidence_complete,
    )
    detector.observe_provider_audio.assert_not_called()
    session.stream_audio.assert_awaited_once_with(pcm16, sample_rate_hz=16_000)
    await _close_dispatchers(runtime)


async def test_buffered_activation_arms_speaker_before_provider_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._asr_audio_dispatcher.abort(turn_token)
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    observation_entered = asyncio.Event()
    observation_release = asyncio.Event()

    async def block_observation(*_args, **_kwargs) -> bool:
        observation_entered.set()
        await observation_release.wait()
        return True

    monkeypatch.setattr(
        runtime,
        "_observe_admitted_provider_audio",
        block_observation,
    )
    activation = asyncio.create_task(
        runtime._activate_asr_audio_dispatcher(
            lifecycle,
            turn_token,
            buffered_pcm16=b"\x01\x00" * 160,
        )
    )
    await asyncio.wait_for(observation_entered.wait(), 1)

    record = await runtime._asr_admission.get_record(turn_token)
    assert record is not None
    assert record.candidate_binding_state.value == "arming"
    final_effects = await runtime._asr_admission_ingress.post(
        turn_token,
        ProviderFinalReceived(_admission_final(None)),
        now=10.0,
    )
    assert not any(isinstance(effect, ResolveReserved) for effect in final_effects)

    observation_release.set()
    assert await asyncio.wait_for(activation, 1) is True
    await runtime._unarm_speaker_authority_after_observation(
        turn_token,
        "profile-generation",
    )
    await _drain_runtime_admission(runtime)
    await _close_dispatchers(runtime)


async def test_active_enqueue_holds_fast_final_until_two_lows_drop() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    runtime._voice_input_resource_optimization_enabled = True
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=81,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(SpeechActivityEvent.SPEECH_STARTED,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    observation_entered = asyncio.Event()
    observation_release = asyncio.Event()
    provider_wire_seen = asyncio.Event()

    async def block_observation(*_args, **_kwargs) -> None:
        observation_entered.set()
        await observation_release.wait()

    async def record_provider_wire(*_args, **_kwargs) -> None:
        provider_wire_seen.set()

    detector.observe_provider_audio_ordered.side_effect = block_observation
    session.stream_audio.side_effect = record_provider_wire
    submission = asyncio.create_task(
        runtime.submit(
            ProcessedVoiceFrame(b"\x12\x00" * 160, 16_000, 0.9, True),
            ingress_token=turn_token.ingress,
        )
    )
    await asyncio.wait_for(observation_entered.wait(), 1)
    await asyncio.wait_for(provider_wire_seen.wait(), 1)

    final_effects = await runtime._asr_admission_ingress.post(
        turn_token,
        ProviderFinalReceived(_admission_final(None)),
        now=10.0,
    )
    assert not any(isinstance(effect, ResolveReserved) for effect in final_effects)
    candidate = detector.lease.shadow_candidate
    assert runtime._accept_speaker_candidate_binding(
        candidate,
        turn_token,
        detector=detector,
        activation_generation="profile-generation",
    )
    assert runtime._accept_speaker_evidence_fact(
        SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        activation_generation="profile-generation",
        enforce=True,
    )
    assert runtime._accept_speaker_evidence_fact(
        SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        activation_generation="profile-generation",
        enforce=True,
    )
    await _drain_runtime_admission(runtime)
    record = await runtime._asr_admission.get_record(turn_token)
    assert record is None
    assert runtime._speaker_rejection_metrics["speaker_deny_latched_count"] == 1
    runtime._callbacks.on_final.assert_not_awaited()

    observation_release.set()
    await asyncio.wait_for(submission, 1)
    await _close_dispatchers(runtime)


async def test_observation_without_candidate_unarms_pending_authority() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    owner_generation = await runtime._arm_speaker_authority_for_provider_audio(
        turn_token
    )
    assert owner_generation == "profile-generation"

    assert await runtime._observe_admitted_provider_audio(
        lifecycle,
        detector,
        b"\x13\x00" * 160,
        sample_rate_hz=16_000,
        identity=DetectorIngressIdentity(
            ingress_token=turn_token.ingress,
            detector_epoch=7,
            sequence_no=82,
        ),
        split_before_audio=False,
        evidence_complete=True,
        turn_token=turn_token,
    )
    await _drain_runtime_admission(runtime)

    record = await runtime._asr_admission.get_record(turn_token)
    assert record is not None
    assert record.candidate_binding_state.value == "retired"
    assert record.evidence_state.value == "unavailable"
    assert turn_token not in runtime._asr_speaker_authority_pending_turns
    await _close_dispatchers(runtime)


async def test_arm_failure_never_wires_buffered_provider_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._asr_audio_dispatcher.abort(turn_token)
    await _open_runtime_admission_turn(runtime, turn_token)
    original_post = runtime._post_admission_event

    async def fail_pending(token, event, *, now=None):
        if isinstance(event, runtime_module.SpeakerAuthorityPending):
            raise KeyError(token)
        return await original_post(token, event, now=now)

    monkeypatch.setattr(runtime, "_post_admission_event", fail_pending)

    assert not await runtime._activate_asr_audio_dispatcher(
        lifecycle,
        turn_token,
        buffered_pcm16=b"\x14\x00" * 160,
    )
    session.stream_audio.assert_not_awaited()
    assert runtime._asr_audio_dispatcher.active_turn is None
    await _close_dispatchers(runtime)


async def test_arm_post_await_identity_drift_never_wires_provider_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._asr_audio_dispatcher.abort(turn_token)
    await _open_runtime_admission_turn(runtime, turn_token)
    original_post = runtime._post_admission_event
    pending_reduced = asyncio.Event()
    release_post = asyncio.Event()

    async def block_after_pending(token, event, *, now=None):
        effects = await original_post(token, event, now=now)
        if isinstance(event, runtime_module.SpeakerAuthorityPending):
            pending_reduced.set()
            await release_post.wait()
        return effects

    monkeypatch.setattr(runtime, "_post_admission_event", block_after_pending)
    activation = asyncio.create_task(
        runtime._activate_asr_audio_dispatcher(
            lifecycle,
            turn_token,
            buffered_pcm16=b"\x16\x00" * 160,
        )
    )
    await asyncio.wait_for(pending_reduced.wait(), 1)
    runtime._asr_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._speaker_verifier_enforces_admission = False
    release_post.set()

    assert not await asyncio.wait_for(activation, 1)
    session.stream_audio.assert_not_awaited()
    assert runtime._asr_audio_dispatcher.active_turn is None
    await _close_dispatchers(runtime)


async def test_cancelled_observation_still_unarms_pending_authority() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    await _open_runtime_admission_turn(runtime, turn_token)
    assert await runtime._arm_speaker_authority_for_provider_audio(turn_token)
    observation_entered = asyncio.Event()

    async def block_observation(*_args, **_kwargs) -> None:
        observation_entered.set()
        await asyncio.Event().wait()

    detector.observe_provider_audio_ordered.side_effect = block_observation
    observation = asyncio.create_task(
        runtime._observe_admitted_provider_audio(
            lifecycle,
            detector,
            b"\x15\x00" * 160,
            sample_rate_hz=16_000,
            identity=DetectorIngressIdentity(
                ingress_token=turn_token.ingress,
                detector_epoch=7,
                sequence_no=83,
            ),
            split_before_audio=False,
            evidence_complete=True,
            turn_token=turn_token,
        )
    )
    await asyncio.wait_for(observation_entered.wait(), 1)
    observation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observation
    await _drain_runtime_admission(runtime)

    record = await runtime._asr_admission.get_record(turn_token)
    assert record is not None
    assert record.evidence_state.value == "unavailable"
    assert turn_token not in runtime._asr_speaker_authority_pending_turns
    await _close_dispatchers(runtime)


async def test_hot_swap_cannot_bypass_authoritative_turn_partial_drop() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )
    runtime._asr_speaker_authoritative_turns.add(turn_token)
    runtime._speaker_verifier_enforces_admission = False

    await runtime._send_independent_asr_preview(
        "quarantined across swap",
        runtime._asr_session_epoch,
    )
    callbacks.on_partial.assert_not_awaited()
    await runtime._execute_admission_effect(
        SettlePartial(turn_token, 1, AdmissionDisposition.DROP)
    )
    await runtime._send_independent_asr_preview(
        "late denied partial",
        runtime._asr_session_epoch,
    )
    callbacks.on_partial.assert_not_awaited()
    await _close_dispatchers(runtime)


async def test_provider_resumed_audio_splits_after_initial_pre_roll() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=41,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(SpeechActivityEvent.SPEECH_STARTED,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    ingress = turn_token.ingress

    first = await runtime.submit(
        ProcessedVoiceFrame(b"\x09\x00" * 160, 16_000, 0.9, True),
        ingress_token=ingress,
    )
    detector.observe_provider_audio_ordered.reset_mock()
    resumed_identity = DetectorIngressIdentity(
        ingress_token=ingress,
        detector_epoch=7,
        sequence_no=42,
    )
    detector.feed.return_value = DetectorFeedResult(
        events=(SpeechActivityEvent.SPEECH_RESUMED,),
        throttle_available=True,
        identity=resumed_identity,
        candidate=DetectorCandidateKey(7, 11),
    )
    resumed_pcm = b"\x0a\x00" * 160

    second = await runtime.submit(
        ProcessedVoiceFrame(resumed_pcm, 16_000, 0.9, True),
        ingress_token=ingress,
    )

    assert first.status is AsrSubmitStatus.ACCEPTED
    assert second.status is AsrSubmitStatus.ACCEPTED
    detector.observe_provider_audio_ordered.assert_awaited_once_with(
        resumed_pcm,
        sample_rate_hz=16_000,
        identity=resumed_identity,
        sequence_no=2,
        split_before_audio=True,
        evidence_complete=True,
    )
    await _close_dispatchers(runtime)


async def test_provider_split_with_pre_roll_marks_ordered_evidence_incomplete() -> (
    None
):
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=42,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(SpeechActivityEvent.SPEECH_RESUMED,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    pre_roll = b"\x0c\x00" * 320
    lifecycle.accept_audio = MagicMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            disposition=AudioDisposition.FORWARD_WITH_PRE_ROLL,
            pre_roll=pre_roll,
        )
    )

    result = await runtime.submit(
        ProcessedVoiceFrame(b"\x0d\x00" * 160, 16_000, 0.9, True),
        ingress_token=turn_token.ingress,
    )

    assert result.status is AsrSubmitStatus.ACCEPTED
    detector.observe_provider_audio_ordered.assert_awaited_once_with(
        pre_roll,
        sample_rate_hz=16_000,
        identity=detector_identity,
        sequence_no=1,
        split_before_audio=False,
        evidence_complete=False,
    )
    await _close_dispatchers(runtime)


async def test_provider_ordered_observation_drift_is_stale_after_audio_admission() -> (
    None
):
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=51,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(SpeechActivityEvent.SPEECH_STARTED,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )

    async def drift_after_observation(*_args, **_kwargs) -> None:
        runtime._asr_audio_generation += 1

    detector.observe_provider_audio_ordered.side_effect = drift_after_observation

    result = await runtime.submit(
        ProcessedVoiceFrame(b"\x0a\x00" * 160, 16_000, 0.9, True),
        ingress_token=turn_token.ingress,
    )

    assert result.status is AsrSubmitStatus.STALE
    detector.observe_provider_audio_ordered.assert_awaited_once()
    detector.observe_provider_audio.assert_not_called()
    await _close_dispatchers(runtime)


async def test_provider_ordered_observation_missing_identity_uses_legacy_fallback() -> (
    None
):
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    first_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=52,
    )
    third_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=54,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        side_effect=(
            DetectorFeedResult(
                events=(),
                throttle_available=True,
                identity=first_identity,
                candidate=DetectorCandidateKey(7, 11),
            ),
            DetectorFeedResult(
                events=(),
                throttle_available=True,
                identity=None,
                candidate=DetectorCandidateKey(7, 11),
            ),
            DetectorFeedResult(
                events=(),
                throttle_available=True,
                identity=third_identity,
                candidate=DetectorCandidateKey(7, 11),
            ),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    first_pcm16 = b"\x0a\x00" * 160
    fallback_pcm16 = b"\x0b\x00" * 160
    third_pcm16 = b"\x0c\x00" * 160

    results = [
        await runtime.submit(
            ProcessedVoiceFrame(pcm16, 16_000, 0.9, True),
            ingress_token=turn_token.ingress,
        )
        for pcm16 in (first_pcm16, fallback_pcm16, third_pcm16)
    ]

    assert [result.status for result in results] == [
        AsrSubmitStatus.ACCEPTED,
        AsrSubmitStatus.ACCEPTED,
        AsrSubmitStatus.ACCEPTED,
    ]
    assert detector.observe_provider_audio_ordered.await_args_list == [
        call(
            first_pcm16,
            sample_rate_hz=16_000,
            identity=first_identity,
            sequence_no=1,
            split_before_audio=False,
            evidence_complete=True,
        ),
        call(
            third_pcm16,
            sample_rate_hz=16_000,
            identity=third_identity,
            sequence_no=2,
            split_before_audio=False,
            evidence_complete=True,
        ),
    ]
    detector.observe_provider_audio.assert_called_once_with(
        fallback_pcm16,
        sample_rate_hz=16_000,
    )
    await _close_dispatchers(runtime)


async def test_provider_ordered_observation_failure_keeps_admitted_audio() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._voice_input_resource_optimization_enabled = True
    await _open_runtime_admission_turn(runtime, turn_token)
    detector_identity = DetectorIngressIdentity(
        ingress_token=turn_token.ingress,
        detector_epoch=7,
        sequence_no=61,
    )
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(SpeechActivityEvent.SPEECH_STARTED,),
            throttle_available=True,
            identity=detector_identity,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    detector.observe_provider_audio_ordered.side_effect = (
        RuntimeError("first private observer failure"),
        RuntimeError("second private observer failure"),
        None,
    )
    pcm_frames = (
        b"\x0e\x00" * 160,
        b"\x0f\x00" * 160,
        b"\x10\x00" * 160,
    )

    results = [
        await runtime.submit(
            ProcessedVoiceFrame(pcm16, 16_000, 0.9, True),
            ingress_token=turn_token.ingress,
        )
        for pcm16 in pcm_frames
    ]
    await runtime._asr_audio_dispatcher.wait_idle()

    assert [result.status for result in results] == [
        AsrSubmitStatus.ACCEPTED,
        AsrSubmitStatus.ACCEPTED,
        AsrSubmitStatus.ACCEPTED,
    ]
    assert [
        awaited.kwargs["sequence_no"]
        for awaited in detector.observe_provider_audio_ordered.await_args_list
    ] == [1, 2, 3]
    detector.observe_provider_audio.assert_not_called()
    assert session.stream_audio.await_args_list == [
        call(pcm16, sample_rate_hz=16_000) for pcm16 in pcm_frames
    ]
    await _close_dispatchers(runtime)


async def test_concurrent_ordered_observers_reserve_unique_sequences() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    entered_sequences: list[int] = []
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_observer(
        _pcm16: bytes,
        *,
        sample_rate_hz: int,
        identity: DetectorIngressIdentity,
        sequence_no: int,
        split_before_audio: bool,
        evidence_complete: bool,
    ) -> None:
        assert sample_rate_hz == 16_000
        assert identity.ingress_token == turn_token.ingress
        assert split_before_audio is False
        assert evidence_complete is True
        entered_sequences.append(sequence_no)
        if len(entered_sequences) == 2:
            both_entered.set()
        await release.wait()

    detector.observe_provider_audio_ordered.side_effect = blocked_observer
    identities = (
        DetectorIngressIdentity(
            ingress_token=turn_token.ingress,
            detector_epoch=7,
            sequence_no=71,
        ),
        DetectorIngressIdentity(
            ingress_token=turn_token.ingress,
            detector_epoch=7,
            sequence_no=72,
        ),
    )
    tasks = tuple(
        asyncio.create_task(
            runtime._observe_admitted_provider_audio(
                lifecycle,
                detector,
                bytes((value, 0)) * 160,
                sample_rate_hz=16_000,
                identity=identity,
                split_before_audio=False,
                evidence_complete=True,
                turn_token=turn_token,
            )
        )
        for value, identity in zip((17, 18), identities, strict=True)
    )

    await asyncio.wait_for(both_entered.wait(), 1)
    assert entered_sequences == [1, 2]
    assert len(set(entered_sequences)) == 2
    release.set()
    assert await asyncio.wait_for(asyncio.gather(*tasks), 1) == [True, True]
    await _close_dispatchers(runtime)




async def _close_dispatchers(runtime: IndependentAsrRuntime) -> None:
    if runtime._asr_admission_ingress_started:
        await runtime._asr_admission_ingress.close()
        runtime._asr_admission_ingress_started = False
    await runtime._asr_audio_dispatcher.close()
    runtime._asr_transcript_dispatcher.invalidate_all()


async def test_verifier_factory_hot_replaces_active_detector_and_closes_old() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    runtime._asr_detector = detector
    old_factory = MagicMock()
    old_factory.close = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"
    shadow = SimpleNamespace(close=AsyncMock())
    factory = MagicMock(return_value=shadow)
    factory.enforces_admission = True

    updated = await runtime.set_speaker_verifier_factory(
        factory,
        activation_generation="new-profile",
    )

    assert updated is True
    detector.replace_speaker_verifier.assert_awaited_once_with(
        shadow,
        owner_generation="new-profile",
    )
    assert runtime._speaker_verifier_factory is factory
    assert runtime._speaker_verifier_activation_generation == "new-profile"
    assert runtime._speaker_verifier_enforces_admission is True
    old_factory.close.assert_called_once_with()




async def test_verifier_factory_failure_fences_old_activation_fail_open() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.replace_speaker_verifier.side_effect = RuntimeError("swap failed")
    runtime._asr_detector = detector
    old_factory = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"
    shadow = SimpleNamespace(close=AsyncMock())
    factory = MagicMock(return_value=shadow)
    factory.enforces_admission = True

    updated = await runtime.set_speaker_verifier_factory(
        factory,
        activation_generation="new-profile",
    )

    assert updated is False
    shadow.close.assert_not_awaited()
    assert runtime._speaker_verifier_factory is None
    assert runtime._speaker_verifier_activation_generation == "new-profile"
    assert runtime._speaker_verifier_enforces_admission is False
    old_factory.close.assert_called_once_with()


async def test_verifier_failure_after_new_binding_retires_collecting_authority() -> (
    None
):
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
    )
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    old_factory = MagicMock()
    old_factory.close = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"
    shadow = SimpleNamespace(close=AsyncMock())
    factory = MagicMock(return_value=shadow)
    factory.enforces_admission = True
    new_candidate = SpeakerShadowCandidateKey(7, 999, "provider_candidate")

    async def publish_then_fail(new_shadow, *, owner_generation) -> None:
        assert new_shadow is shadow
        assert owner_generation == "new-profile"
        assert runtime._accept_speaker_candidate_binding(
            new_candidate,
            turn_token,
            detector=detector,
            activation_generation=owner_generation,
        )
        await _drain_runtime_admission(runtime)
        record = await runtime._asr_admission.get_record(turn_token)
        assert record is not None
        assert record.capture_state is CaptureState.COLLECTING
        raise RuntimeError("swap failed after install")

    detector.replace_speaker_verifier.side_effect = publish_then_fail
    try:
        updated = await runtime.set_speaker_verifier_factory(
            factory,
            activation_generation="new-profile",
        )
        await _drain_runtime_admission(runtime)

        assert updated is False
        record = await runtime._asr_admission.get_record(turn_token)
        assert (
            record is None
            or record.capture_state is not CaptureState.COLLECTING
        )
        assert runtime._asr_admission_candidate_turns == {}
        assert runtime._asr_admission_capabilities == {}
        shadow.close.assert_not_awaited()
        old_factory.close.assert_called_once_with()
    finally:
        await _close_dispatchers(runtime)


async def test_cancelled_verifier_swap_keeps_transferred_shadow_owned_by_detector() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    runtime._asr_detector = detector
    old_factory = MagicMock()
    old_factory.close = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"
    shadow = SimpleNamespace(close=AsyncMock())
    factory = MagicMock(return_value=shadow)
    factory.enforces_admission = True
    entered = asyncio.Event()

    async def block_after_handoff(new_shadow, *, owner_generation) -> None:
        assert new_shadow is shadow
        assert owner_generation == "new-profile"
        assert runtime._speaker_verifier_activation_generation == "new-profile"
        assert runtime._speaker_verifier_enforces_admission is True
        entered.set()
        await asyncio.Event().wait()

    detector.replace_speaker_verifier.side_effect = block_after_handoff
    task = asyncio.create_task(
        runtime.set_speaker_verifier_factory(
            factory,
            activation_generation="new-profile",
        )
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    shadow.close.assert_not_awaited()
    assert runtime._speaker_verifier_factory is None
    assert runtime._speaker_verifier_activation_generation == "new-profile"
    assert runtime._speaker_verifier_enforces_admission is False
    old_factory.close.assert_called_once_with()






async def test_sealed_provider_rejection_suppresses_only_exact_final() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        deny_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )
        boundary_effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(
            isinstance(effect, ApplyRejection) for effect in boundary_effects
        )
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key)),
            now=10.0,
        )
        assert not any(
            isinstance(effect, ResolveReserved) for effect in final_effects
        )
        resolutions = [
            effect
            for effect in deny_effects
            if isinstance(effect, ResolveReserved)
        ]
        assert [effect.disposition for effect in resolutions] == [
            AdmissionDisposition.DROP
        ]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()


async def test_provider_micro_event_shadow_forwards_non_empty_final() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    try:
        await lane.post(turn_token, SpeakerHigh(candidate, 1))
        await lane.post(turn_token, MicroEventPending())
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key, text="嗯")),
            now=10.0,
        )
        assert not any(
            isinstance(effect, ResolveReserved) for effect in final_effects
        )
        allowed_effects = await lane.post(turn_token, MicroEventAllowed(), now=10.1)
        assert [
            effect.disposition
            for effect in allowed_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.FORWARD]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.FORWARDED
    finally:
        await lane.close()


@pytest.mark.parametrize("enforce", [False, True])
async def test_provider_micro_event_empty_final_is_never_suppressed_or_counted(
    enforce: bool,
) -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    try:
        await lane.post(turn_token, SpeakerHigh(candidate, 1))
        await lane.post(turn_token, MicroEventPending())
        await lane.post(
            turn_token,
            MicroEventSuppressed() if enforce else MicroEventAllowed(),
        )
        effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key, text="")),
            now=10.0,
        )
        assert [
            effect.disposition
            for effect in effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.FORWARD]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.FORWARDED
    finally:
        await lane.close()


async def test_provider_micro_event_enforce_suppresses_exact_non_empty_final() -> None:
    coordinator, lane, turn_token, _candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    try:
        await lane.post(turn_token, MicroEventPending())
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key, text="嗯")),
            now=10.0,
        )
        assert not any(
            isinstance(effect, ResolveReserved) for effect in final_effects
        )
        suppressed_effects = await lane.post(
            turn_token,
            MicroEventSuppressed(),
            now=10.1,
        )
        assert [
            effect.disposition
            for effect in suppressed_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.DROP]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()


async def test_exact_speaker_and_micro_event_suppress_only_once() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        deny_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )
        boundary_effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(
            isinstance(effect, ApplyRejection) for effect in boundary_effects
        )
        await lane.post(turn_token, MicroEventPending())
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key, text="嗯")),
            now=10.0,
        )
        micro_effects = await lane.post(
            turn_token,
            MicroEventSuppressed(),
            now=10.11,
        )
        assert sum(
                isinstance(effect, ResolveReserved)
                for effect in (*deny_effects, *final_effects, *micro_effects)
        ) == 1
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()


async def test_provider_micro_event_query_failure_fails_open() -> None:
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    try:
        await lane.post(turn_token, SpeakerHigh(candidate, 1))
        await lane.post(turn_token, MicroEventPending())
        await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key, text="嗯")),
            now=10.0,
        )
        effects = await lane.post(turn_token, MicroEventUnavailable(), now=10.1)
        assert [
            effect.disposition
            for effect in effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.FORWARD]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.FORWARDED
    finally:
        await lane.close()


async def test_provider_micro_event_decision_is_stale_after_completion_drift() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, _provider_fence = await _seal_provider_candidate(
        runtime,
        detector,
    )
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    async def drift_during_completion(_provider_fence) -> bool:
        runtime._asr_audio_generation += 1
        return False

    detector.complete_provider_candidate.side_effect = drift_during_completion

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")

    callbacks.on_final.assert_not_awaited()
    assert not runtime._asr_transcript_dispatcher.try_reserve(
        FinalKey.from_turn(turn_token)
    )
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 0
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_completion_rejects_session_replacement_during_await() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, lifecycle, turn_token, _provider_fence = await _seal_provider_candidate(
        runtime, detector
    )
    posted_events: list[object] = []
    post_admission_event = runtime._post_admission_event

    async def capture_admission_event(token, event, **kwargs):
        posted_events.append(event)
        return await post_admission_event(token, event, **kwargs)

    runtime._post_admission_event = capture_admission_event

    async def replace_session_during_completion(_provider_fence) -> bool:
        runtime._asr_session = SimpleNamespace(
            is_ready=True,
            close=AsyncMock(),
            signal_user_activity_end=AsyncMock(),
            stream_audio=AsyncMock(),
        )
        return False

    detector.complete_provider_candidate.side_effect = (
        replace_session_during_completion
    )

    settled = await runtime._handle_independent_asr_final(
        "stale-final",
        0,
        "qwen",
    )
    assert settled is not None
    await asyncio.wait_for(settled.wait(), 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert any(
        isinstance(event, TransportSettled) and event.degraded
        for event in posted_events
    )
    assert any(
        isinstance(event, LifecycleSettled) and event.degraded
        for event in posted_events
    )
    final_key = FinalKey.from_turn(turn_token)
    assert not runtime._asr_transcript_dispatcher.try_reserve(final_key)
    await _close_dispatchers(runtime)




async def test_provider_micro_event_suppression_preserves_same_text_successor() -> (
    None
):
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, first_turn, _provider_fence = await _seal_provider_candidate(
        runtime,
        detector,
    )
    lifecycle.mark_pending_turn_speech()
    pending_pcm = b"\x01\x00" * 320
    buffered = lifecycle.accept_audio(pending_pcm, sample_rate_hz=16_000)
    assert buffered.disposition.value == "buffer"
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime._asr_audio_dispatcher.wait_idle()

    assert callbacks.on_final.await_count == 0
    abandoned.assert_awaited_once_with(first_turn)
    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    successor_turn = runtime._asr_partial_turn_token
    assert successor_turn is not None and successor_turn != first_turn
    detector.lease = _RejectionLease(detector, successor_turn)
    await runtime._handle_independent_asr_endpoint(0)
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(False, False, "silero_span_exceeded")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await _drain_runtime_admission(runtime)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert callbacks.on_final.await_args.args[0].text == "嗯"
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 1
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("enforce", [False, True])
async def test_duplicate_micro_event_final_counts_once_and_preserves_next_turn(
    enforce: bool,
) -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, first_turn, provider_fence = await _seal_provider_candidate(
        runtime,
        detector,
    )
    lifecycle.mark_pending_turn_speech()
    pending_pcm = b"\x02\x00" * 320
    buffered = lifecycle.accept_audio(pending_pcm, sample_rate_hz=16_000)
    assert buffered.disposition.value == "buffer"
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, enforce, "eligible_micro_event")
    )
    complete_entered = asyncio.Event()
    complete_release = asyncio.Event()

    async def blocked_complete(received_fence) -> bool:
        assert received_fence == provider_fence
        complete_entered.set()
        await complete_release.wait()
        return False

    detector.complete_provider_candidate.side_effect = blocked_complete
    first_final = asyncio.create_task(
        runtime._handle_independent_asr_final("嗯", 0, "qwen")
    )
    await asyncio.wait_for(complete_entered.wait(), 1)
    duplicate_final = asyncio.create_task(
        runtime._handle_independent_asr_final("嗯", 0, "qwen")
    )
    await asyncio.sleep(0)
    assert duplicate_final.done() is False
    complete_release.set()

    settlements = await asyncio.wait_for(
        asyncio.gather(first_final, duplicate_final),
        1,
    )
    assert sum(settled is not None for settled in settlements) == 1
    settled = next(settled for settled in settlements if settled is not None)
    await asyncio.wait_for(settled.wait(), 1)
    await runtime._asr_audio_dispatcher.wait_idle()

    detector.sealed_provider_micro_event_decision.assert_called_once_with(
        provider_fence
    )
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == int(enforce)
    assert diagnostics["micro_event_shadow_forward_count"] == int(not enforce)
    assert callbacks.on_final.await_count == int(not enforce)
    assert abandoned.await_count == int(enforce)
    if enforce:
        abandoned.assert_awaited_once_with(first_turn)

    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    successor_turn = runtime._asr_partial_turn_token
    assert successor_turn is not None and successor_turn != first_turn
    detector.lease = _RejectionLease(detector, successor_turn)
    detector.complete_provider_candidate.side_effect = None
    detector.complete_provider_candidate.return_value = False
    await runtime._handle_independent_asr_endpoint(0)
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(False, False, "silero_span_exceeded")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await _drain_runtime_admission(runtime)
    await runtime.wait_transcript_idle()

    assert callbacks.on_final.await_count == int(not enforce) + 1
    assert callbacks.on_final.await_args.args[0].text == "嗯"
    assert detector.sealed_provider_micro_event_decision.call_count == 2
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == int(enforce)
    assert diagnostics["micro_event_shadow_forward_count"] == int(not enforce)
    await _close_dispatchers(runtime)


async def test_smart_turn_final_does_not_query_provider_micro_event() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="manual",
    )
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    runtime._asr_admission_reservation_dispatchers[FinalKey.from_turn(turn_token)] = (
        runtime._asr_transcript_dispatcher
    )
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    # This test isolates SmartTurn's micro-event contract. Speaker admission
    # is disabled so a missing verdict cannot hold the unrelated final.
    runtime._speaker_verifier_enforces_admission = False
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await _drain_runtime_admission(runtime)
    await runtime.wait_transcript_idle()

    detector.sealed_provider_micro_event_decision.assert_not_called()
    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 0
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


























async def test_exact_pause_merge_scores_before_seal_and_suppresses_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from main_logic.asr_client.speaker_shadow.asset_manifest import (
        CAMPPLUS_MODEL_ID,
        CAMPPLUS_MODEL_REVISION,
    )
    from main_logic.asr_client.speaker_shadow.campplus import (
        CAMPPLUS_EMBEDDING_DIM,
    )
    from main_logic.voice_identity.contracts import SpeakerModelIdentity
    from main_logic.voice_identity.profile import SpeakerProfile
    from main_logic.voice_identity.reference import SpeakerReference
    from main_logic.voice_identity_service.asr_composition import (
        OwnerVoiceAsrCompositionFactory,
    )

    class _Vad:
        def load(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class _Gate:
        def feed(self, _pcm16: bytes) -> tuple[()]:
            return ()

        def reset(self) -> None:
            return None

    class _LowScoreHost:
        alive = True

        def __init__(self) -> None:
            self.score_count = 0

        async def score(
            self,
            _pcm16: bytes,
            *,
            timeout_seconds: float,
        ) -> float:
            assert timeout_seconds > 0
            self.score_count += 1
            return 0.20

    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    placeholder = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        placeholder,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    await runtime._asr_admission_ingress.start()
    runtime._asr_admission_ingress_started = True
    await runtime._asr_admission_ingress.open_turn(turn_token)
    runtime._asr_admission_reservation_dispatchers[final_key] = (
        runtime._asr_transcript_dispatcher
    )

    model_identity = SpeakerModelIdentity(
        CAMPPLUS_MODEL_ID,
        CAMPPLUS_MODEL_REVISION,
        CAMPPLUS_EMBEDDING_DIM,
    )
    embedding = np.arange(1, CAMPPLUS_EMBEDDING_DIM + 1, dtype=np.float32)
    reference = SpeakerReference(model_identity, embedding)
    embedding.fill(0.0)
    try:
        profile = SpeakerProfile("profile-generation", reference)
    finally:
        reference.close()
    composition = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="profile-generation",
        enforce=True,
    )
    shadow = composition()
    scoring_host = _LowScoreHost()
    monkeypatch.setattr(
        shadow,
        "_ensure_backend",
        AsyncMock(return_value=scoring_host),
    )
    detector = DetectorRuntime(
        vad=_Vad(),
        gate=_Gate(),
        provider_policy=resolve_provider_policy("qwen", "provider"),
        speaker_shadow=shadow,
    )
    runtime._asr_detector = detector
    runtime._asr_provider_exact_session = session

    first_feed = await detector.feed(
        b"\x01\x00" * 160,
        ingress_token=turn_token.ingress,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    assert first_feed.candidate is not None
    assert first_feed.identity is not None
    assert await detector.bind_candidate(
            first_feed.candidate,
            turn_token,
    ) is not None
    second_feed = await detector.feed(
        b"\x02\x00" * 160,
        ingress_token=turn_token.ingress,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    assert second_feed.identity is not None
    await detector.observe_provider_audio_ordered(
        b"\x11\x00" * 12_800,
        sample_rate_hz=16_000,
        identity=first_feed.identity,
        sequence_no=1,
        split_before_audio=False,
    )
    await detector.observe_provider_audio_ordered(
        b"\x12\x00" * 40_000,
        sample_rate_hz=16_000,
        identity=second_feed.identity,
        sequence_no=2,
        split_before_audio=True,
    )

    audio_range = ProviderAudioRange(0, 52_800)
    key = ProviderUtteranceKey(0, 0, 1)
    await runtime._handle_provider_endpoint_notification(
        ProviderEndpointNotification(
            phase="boundary",
            generation=key.generation,
            buffer_epoch=key.buffer_epoch,
            utterance_id=key.utterance_id,
            boundary_quality="exact",
            audio_range=audio_range,
        ),
        runtime._asr_session_epoch,
    )
    await shadow.wait_idle()
    await _drain_runtime_admission(runtime)

    assert scoring_host.score_count == 2
    admission_record = await runtime._asr_admission.get_record(turn_token)
    # Terminal settlement may already retire the reducer record. The durable
    # contract evidence is the single deny latch plus zero transcript output.
    assert admission_record is None
    assert (
        runtime._speaker_verifier_diagnostics()["speaker_deny_latched_count"]
        == 1
    )
    assert runtime._asr_session is session

    # Provider cleanup may retire the correlator alias before the ordered
    # notification arrives. That cannot revise the already-latched deny.
    await runtime._handle_provider_endpoint_notification(
        ProviderEndpointNotification(
            phase="ordered",
            generation=key.generation,
            buffer_epoch=key.buffer_epoch,
            utterance_id=key.utterance_id,
            boundary_quality="exact",
            audio_range=audio_range,
        ),
        runtime._asr_session_epoch,
    )
    await _drain_runtime_admission(runtime)
    admission_record = await runtime._asr_admission.get_record(turn_token)
    assert admission_record is None
    assert (
        runtime._speaker_verifier_diagnostics()["speaker_deny_latched_count"]
        == 1
    )

    await runtime._handle_provider_final(
        key,
        "pause-merged-rejected-final",
        runtime._asr_session_epoch,
        "qwen",
    )
    await _drain_runtime_admission(runtime)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    # Cleanup may retire the exact Provider transport; it cannot forward the
    # denied final or transfer that denial into a successor turn.
    assert runtime._asr_session is None
    diagnostics = detector.speaker_rejection_diagnostics_snapshot()
    assert diagnostics["provider_speaker_segment_merged_resume_count"] == 1
    assert diagnostics["provider_speaker_segment_exact_snapshot_count"] == 0
    assert diagnostics["provider_preseal_verdict_stored_count"] == 1
    # Authoritative denial can retire the Provider transport before an exact
    # seal consumes the cleanup-only preseal verdict.
    assert diagnostics["provider_preseal_verdict_consumed_count"] == 0
    assert diagnostics["provider_rejection_applied_count"] == 0
    assert diagnostics["reconciliation_batch_admitted_count"] == 1
    assert diagnostics["reconciliation_batch_applied_count"] == 1
    assert diagnostics["provider_namespace_poison_count"] == 0
    assert runtime._asr_admission_turn_sealed_events == {}

    await detector.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)












































async def test_provider_gate_timeout_forwards_final_and_rejects_late_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    coordinator, lane, turn_token, candidate, provider_key = (
        await _open_admission_turn(provider_bound=True)
    )
    assert provider_key is not None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=provider_key,
        kind=RejectionCapabilityKind.SEALED,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        boundary_effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(
            isinstance(effect, ApplyRejection) for effect in boundary_effects
        )
        final_effects = await lane.post(
            turn_token,
            ProviderFinalReceived(_admission_final(provider_key)),
            now=10.0,
        )
        deadline = next(
            effect
            for effect in final_effects
            if isinstance(effect, ScheduleFinalDeadline)
        )
        assert deadline.absolute_deadline == 10.2

        timeout_effects = await lane.post(
            turn_token,
            FinalDeadlineExpired(deadline.ticket, deadline.absolute_deadline),
            now=deadline.absolute_deadline,
        )
        assert not any(
            isinstance(effect, ResolveReserved) for effect in timeout_effects
        )
        pending = await coordinator.get_record(turn_token)
        assert pending is not None
        assert pending.admission_state is AdmissionState.PENDING

        denied_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
            now=10.21,
        )
        assert [
            effect.disposition
            for effect in denied_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.DROP]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()
async def test_smart_turn_rejection_does_not_require_provider_gate() -> None:
    candidate = _smart_turn_shadow_candidate()
    coordinator, lane, turn_token, _, provider_key = await _open_admission_turn(
        provider_bound=False,
        candidate=candidate,
    )
    assert provider_key is None
    capability = _admission_capability(
        turn_token,
        candidate,
        provider_key=None,
        kind=RejectionCapabilityKind.ACTIVE,
    )
    try:
        await lane.post(
            turn_token,
            SpeakerLow(candidate, 1, SpeakerCheckpointKind.FIRST),
        )
        denied_effects = await lane.post(
            turn_token,
            SpeakerLow(candidate, 2, SpeakerCheckpointKind.SECOND),
        )
        boundary_effects = await lane.post(turn_token, BoundaryExact(capability))
        assert not any(
            isinstance(effect, ApplyRejection) for effect in boundary_effects
        )

        assert any(
            isinstance(effect, AbortProviderTransport)
            for effect in denied_effects
        )
        assert [
            effect.disposition
            for effect in denied_effects
            if isinstance(effect, ResolveReserved)
        ] == [AdmissionDisposition.DROP]
        record = await coordinator.get_record(turn_token)
        assert record is not None
        assert record.admission_state is AdmissionState.DROPPED
    finally:
        await lane.close()
