from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import main_logic.asr_client.runtime as runtime_module
from main_logic.asr_client.candidate_control import CandidateRejectionOutcome
from main_logic.asr_client.endpointing.detector import (
    DetectorCandidateKey,
    DetectorIngressIdentity,
    ProviderCandidateFence,
)
from main_logic.asr_client.endpointing.detector_runtime import (
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
        self._detector = detector
        self.commit_calls = 0
        self.commit_result = True

    def belongs_to(self, detector: object) -> bool:
        return detector is self._detector

    def commit(self) -> bool:
        self.commit_calls += 1
        return self.commit_result


class _RejectionDetector:
    def __init__(self) -> None:
        self.lease: _RejectionLease | None = None
        self.prepare_entered = asyncio.Event()
        self.prepare_release = asyncio.Event()
        self.block_prepare = False
        self.reset = AsyncMock()
        self.replace_speaker_verifier = AsyncMock()
        self.close = AsyncMock()
        self.complete_provider_candidate = AsyncMock(return_value=False)
        self.sealed_provider_micro_event_decision = MagicMock(return_value=None)
        self.release_deferred_turn = AsyncMock()
        self.observe_provider_audio_ordered = AsyncMock()
        self.observe_provider_audio = MagicMock()
        self.provisional_pending = False
        self.seal_turn_tokens: list[VoiceTurnToken | None] = []

    async def prepare_candidate_rejection(self, _candidate):
        self.prepare_entered.set()
        if self.block_prepare:
            await self.prepare_release.wait()
        return self.lease

    async def seal_provider_candidate(
        self,
        turn_token: VoiceTurnToken | None = None,
    ):
        self.seal_turn_tokens.append(turn_token)
        lease = self.lease
        if lease is None:
            return None
        if turn_token is not None and turn_token != lease.turn_token:
            return None
        fence = ProviderCandidateFence(7, 11, 23)
        lease.provider_fence = fence
        return fence

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
    turn_token = VoiceTurnToken(
        ingress=runtime._asr_current_ingress_token,
        turn_id=lifecycle.snapshot.turn_id,
    )
    detector.lease = _RejectionLease(detector, turn_token)
    runtime._asr_partial_turn_token = turn_token
    runtime._asr_turn_prepared = True
    runtime._speaker_verifier_activation_generation = "profile-generation"
    assert runtime._asr_audio_dispatcher.activate(turn_token, session, b"") is True
    final_key = FinalKey.from_turn(turn_token)
    assert runtime._asr_transcript_dispatcher.try_reserve(final_key) is True
    runtime._asr_reserved_final_key = final_key
    runtime._ensure_transport_restart_task = MagicMock()
    return session, lifecycle, turn_token


def _seal_provider_candidate(
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


def _rewind_installed_candidate_to_unprepared(
    runtime: IndependentAsrRuntime,
) -> None:
    final_key = runtime._asr_reserved_final_key
    assert final_key is not None
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    runtime._asr_turn_prepared = False
    runtime._asr_partial_turn_token = None


def test_rejection_request_outside_event_loop_fails_open() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    runtime._speaker_verifier_activation_generation = "profile-generation"

    assert not runtime.request_speaker_candidate_rejection(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    assert runtime._speaker_verifier_diagnostics()["rejection_request_failed_count"] == 1


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
    detector.feed = AsyncMock(  # type: ignore[attr-defined]
        return_value=DetectorFeedResult(
            events=(),
            throttle_available=True,
            identity=None,
            candidate=DetectorCandidateKey(7, 11),
        )
    )
    runtime._bind_provider_detector_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=True
    )
    pcm16 = b"\x0b\x00" * 160

    result = await runtime.submit(
        ProcessedVoiceFrame(pcm16, 16_000, 0.9, True),
        ingress_token=turn_token.ingress,
    )

    assert result.status is AsrSubmitStatus.ACCEPTED
    detector.observe_provider_audio_ordered.assert_not_awaited()
    detector.observe_provider_audio.assert_called_once_with(
        pcm16,
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
    detector.observe_provider_audio_ordered.side_effect = RuntimeError(
        "private observer failure"
    )
    pcm16 = b"\x0e\x00" * 160

    result = await runtime.submit(
        ProcessedVoiceFrame(pcm16, 16_000, 0.9, True),
        ingress_token=turn_token.ingress,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    assert result.status is AsrSubmitStatus.ACCEPTED
    detector.observe_provider_audio_ordered.assert_awaited_once()
    detector.observe_provider_audio.assert_not_called()
    session.stream_audio.assert_awaited_once_with(pcm16, sample_rate_hz=16_000)
    await _close_dispatchers(runtime)


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            CandidateRejectionOutcome.APPLIED,
            {
                "rejection_task_applied_count": 1,
                "rejection_task_stale_count": 0,
                "rejection_task_cleanup_degraded_count": 0,
            },
        ),
        (
            CandidateRejectionOutcome.STALE,
            {
                "rejection_task_applied_count": 0,
                "rejection_task_stale_count": 1,
                "rejection_task_cleanup_degraded_count": 0,
            },
        ),
        (
            CandidateRejectionOutcome.APPLIED_CLEANUP_DEGRADED,
            {
                "rejection_task_applied_count": 1,
                "rejection_task_stale_count": 0,
                "rejection_task_cleanup_degraded_count": 1,
            },
        ),
    ],
)
async def test_rejection_diagnostics_record_terminal_outcome(
    outcome: CandidateRejectionOutcome,
    expected: dict[str, int],
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    runtime._speaker_verifier_activation_generation = "profile-generation"
    runtime._reject_speaker_candidate = AsyncMock(return_value=outcome)  # type: ignore[method-assign]

    assert runtime.request_speaker_candidate_rejection(
        _smart_turn_shadow_candidate(),
        activation_generation="profile-generation",
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["rejection_task_scheduled_count"] == 1
    assert diagnostics["rejection_task_pending_count"] == 0
    for name, value in expected.items():
        assert diagnostics[name] == value


async def _close_dispatchers(runtime: IndependentAsrRuntime) -> None:
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

    updated = await runtime.set_speaker_verifier_factory(
        factory,
        activation_generation="new-profile",
    )

    assert updated is True
    detector.replace_speaker_verifier.assert_awaited_once_with(shadow)
    assert runtime._speaker_verifier_factory is factory
    assert runtime._speaker_verifier_activation_generation == "new-profile"
    old_factory.close.assert_called_once_with()


async def test_verifier_factory_failure_is_detached_and_preserves_activation() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.replace_speaker_verifier.side_effect = RuntimeError("swap failed")
    runtime._asr_detector = detector
    old_factory = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"
    shadow = SimpleNamespace(close=AsyncMock())
    factory = MagicMock(return_value=shadow)

    updated = await runtime.set_speaker_verifier_factory(
        factory,
        activation_generation="new-profile",
    )

    assert updated is False
    shadow.close.assert_awaited_once_with()
    assert runtime._speaker_verifier_factory is old_factory
    assert runtime._speaker_verifier_activation_generation == "old-profile"


async def test_verifier_detach_failure_still_revokes_old_activation() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.replace_speaker_verifier.side_effect = RuntimeError("detach failed")
    runtime._asr_detector = detector
    old_factory = MagicMock()
    old_factory.close = MagicMock()
    runtime._speaker_verifier_factory = old_factory
    runtime._speaker_verifier_activation_generation = "old-profile"

    updated = await runtime.set_speaker_verifier_factory(
        None,
        activation_generation="revoked-profile",
    )

    assert updated is False
    assert runtime._speaker_verifier_factory is None
    assert runtime._speaker_verifier_activation_generation == "revoked-profile"
    old_factory.close.assert_called_once_with()
    assert not runtime.request_speaker_candidate_rejection(
        _shadow_candidate(),
        activation_generation="old-profile",
    )


async def test_candidate_rejection_applies_and_recovers_next_transport() -> None:
    abandoned = AsyncMock()
    runtime = IndependentAsrRuntime(_callbacks(abandoned=abandoned))
    detector = _RejectionDetector()
    session, _lifecycle, turn_token = _install_active_candidate(runtime, detector)

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    assert outcome is CandidateRejectionOutcome.APPLIED
    assert detector.lease is not None and detector.lease.commit_calls == 1
    session.close.assert_awaited_once_with()
    detector.reset.assert_awaited_once_with()
    abandoned.assert_awaited_once_with(turn_token)
    assert runtime._asr_candidate_rejection is None
    runtime._ensure_transport_restart_task.assert_called_once_with()
    await _close_dispatchers(runtime)


async def test_sealed_provider_rejection_suppresses_only_exact_final() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token, provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    assert outcome is CandidateRejectionOutcome.APPLIED
    assert runtime._asr_suppressed_final_key == FinalKey.from_turn(turn_token)
    assert detector.lease is not None and detector.lease.commit_calls == 1
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    runtime._ensure_transport_restart_task.assert_not_called()
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert runtime._asr_sealed_turn_token is not None
    assert runtime._asr_provider_candidate_fence == provider_fence

    await runtime._handle_independent_asr_final("not-owner", 0, "qwen")
    await runtime.wait_transcript_idle()

    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    assert lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_suppressed_final_key is None
    assert runtime._asr_provider_candidate_fence is None
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    await _close_dispatchers(runtime)


async def test_provider_micro_event_shadow_forwards_non_empty_final() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token, provider_fence = (
        _seal_provider_candidate(runtime, detector)
    )
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, False, "micro_event_shadow")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime.wait_transcript_idle()

    detector.sealed_provider_micro_event_decision.assert_called_once_with(
        provider_fence
    )
    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_shadow_forward_count"] == 1
    assert diagnostics["micro_event_suppressed_count"] == 0
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("enforce", [False, True])
async def test_provider_micro_event_empty_final_is_never_suppressed_or_counted(
    enforce: bool,
) -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _seal_provider_candidate(runtime, detector)
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, enforce, "eligible_micro_event")
    )

    await runtime._handle_independent_asr_final("", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    abandoned.assert_not_awaited()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 0
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_micro_event_enforce_suppresses_exact_non_empty_final() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime.wait_transcript_idle()

    detector.sealed_provider_micro_event_decision.assert_called_once_with(
        provider_fence
    )
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    assert runtime._asr_suppressed_final_key is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 1
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_exact_speaker_and_micro_event_suppress_only_once() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_suppressed_final_key = final_key
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime.wait_transcript_idle()

    detector.sealed_provider_micro_event_decision.assert_called_once_with(
        provider_fence
    )
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    assert runtime._asr_suppressed_final_key is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 1
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_micro_event_query_failure_fails_open() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _seal_provider_candidate(runtime, detector)
    detector.sealed_provider_micro_event_decision.side_effect = RuntimeError(
        "query failed"
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 0
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_micro_event_decision_is_stale_after_completion_drift() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, _provider_fence = _seal_provider_candidate(
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
    assert runtime._asr_transcript_dispatcher.try_reserve(
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
    _session, _lifecycle, turn_token, _provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )

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

    await runtime._handle_independent_asr_final("stale-final", 0, "qwen")

    callbacks.on_final.assert_not_awaited()
    final_key = FinalKey.from_turn(turn_token)
    assert runtime._asr_transcript_dispatcher.try_reserve(final_key)
    runtime._asr_transcript_dispatcher.release(final_key)
    await _close_dispatchers(runtime)


async def test_provider_micro_event_waits_for_existing_speaker_gate() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token, _provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None

    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("嗯", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)
    assert final_task.done() is False
    detector.sealed_provider_micro_event_decision.assert_not_called()

    assert runtime._resolve_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
        rejected=False,
    )
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["micro_event_suppressed_count"] == 1
    await _close_dispatchers(runtime)


async def test_provider_micro_event_suppression_preserves_same_text_successor() -> (
    None
):
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, first_turn, _provider_fence = _seal_provider_candidate(
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
    session, lifecycle, first_turn, provider_fence = _seal_provider_candidate(
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

    await asyncio.wait_for(asyncio.gather(first_final, duplicate_final), 1)
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
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    detector.sealed_provider_micro_event_decision.return_value = (
        ProviderMicroEventDecision(True, True, "micro_event_enforced")
    )

    await runtime._handle_independent_asr_final("嗯", 0, "qwen")
    await runtime.wait_transcript_idle()

    detector.sealed_provider_micro_event_decision.assert_not_called()
    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["micro_event_suppressed_count"] == 0
    assert diagnostics["micro_event_shadow_forward_count"] == 0
    await _close_dispatchers(runtime)


async def test_first_low_provider_gate_waits_outside_final_lock_for_rejection() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()

    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.deadline is None
    provider_fence = _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    detector.prepare_entered = asyncio.Event()
    detector.prepare_release = asyncio.Event()
    detector.block_prepare = True
    assert runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)

    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("not-owner", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)
    assert gate.deadline is not None
    await asyncio.wait_for(runtime._asr_final_lock.acquire(), 0.05)
    runtime._asr_final_lock.release()
    assert not final_task.done()

    detector.prepare_release.set()
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    assert runtime._asr_speaker_candidate_decision_gate is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_count"] == 1
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    assert diagnostics["speaker_gate_resolved_forward_count"] == 0
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["speaker_gate_stale_count"] == 0
    await _close_dispatchers(runtime)


async def test_provisional_gate_holds_early_final_until_owner_resolution() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    detector.provisional_pending = True

    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        provisional=True,
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.first_observation_pending is True
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("owner-final", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)
    assert final_task.done() is False

    detector.provisional_pending = False
    assert runtime._resolve_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        rejected=False,
    )
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_provisional_armed_count"] == 1
    assert diagnostics["speaker_gate_final_before_first_observation_count"] == 1
    assert diagnostics["speaker_gate_resolved_forward_count"] == 1
    assert diagnostics["speaker_gate_provisional_timeout_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_endpoint_installs_provisional_gate_before_returning() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    detector.provisional_pending = True

    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)

    assert detector.seal_turn_tokens == [turn_token]
    gate = runtime._asr_speaker_candidate_decision_gate
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert gate is not None
    assert gate.candidate == _shadow_candidate()
    assert gate.first_observation_pending is True
    assert gate.deadline is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_provisional_armed_count"] == 1
    assert diagnostics["speaker_gate_arm_provisional_stale_count"] == 0

    assert runtime._resolve_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
        rejected=False,
    )
    await _close_dispatchers(runtime)


async def test_provider_endpoint_seal_identity_drift_does_not_publish_fence() -> (
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
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None

    async def drift_while_sealing(
        observed_turn: VoiceTurnToken | None = None,
    ) -> ProviderCandidateFence:
        detector.seal_turn_tokens.append(observed_turn)
        runtime._asr_audio_generation += 1
        return ProviderCandidateFence(7, 11, 23)

    detector.seal_provider_candidate = drift_while_sealing  # type: ignore[method-assign]

    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)

    assert detector.seal_turn_tokens == [turn_token]
    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_provider_candidate_fence is None
    assert runtime._asr_sealed_turn_token is None
    assert runtime._asr_reserved_final_key is None
    assert runtime._asr_transcript_dispatcher.try_reserve(final_key)
    runtime._asr_transcript_dispatcher.release(final_key)
    await _close_dispatchers(runtime)


async def test_provider_final_waits_while_provisional_gate_is_preparing() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    detector.provisional_pending = True
    detector.block_prepare = True

    endpoint_task = asyncio.create_task(
        runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    preparation = runtime._asr_speaker_candidate_decision_preparation
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert runtime._asr_speaker_candidate_decision_gate is None
    assert preparation is not None and preparation.retired is False

    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("early-provider-final", 0, "qwen")
    )
    duplicate_final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("early-provider-final", 0, "qwen")
    )
    while not preparation.wait_started:
        await asyncio.sleep(0)
    assert final_task.done() is False
    assert duplicate_final_task.done() is False
    callbacks.on_final.assert_not_awaited()

    detector.prepare_release.set()
    await asyncio.wait_for(endpoint_task, 1)
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.deadline == preparation.deadline
    assert runtime._resolve_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    await asyncio.wait_for(final_task, 1)
    await asyncio.wait_for(duplicate_final_task, 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._asr_speaker_candidate_decision_preparation is None
    await _close_dispatchers(runtime)


async def test_provisional_prepare_identity_drift_releases_reserved_final() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    detector.provisional_pending = True
    detector.block_prepare = True

    endpoint_task = asyncio.create_task(
        runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    runtime._asr_audio_generation += 1
    detector.prepare_release.set()
    await asyncio.wait_for(endpoint_task, 1)

    assert runtime._asr_reserved_final_key is None
    assert runtime._asr_speaker_candidate_decision_preparation is None
    assert runtime._asr_transcript_dispatcher.try_reserve(final_key)
    runtime._asr_transcript_dispatcher.release(final_key)
    await _close_dispatchers(runtime)


async def test_provisional_prepare_timeout_forwards_without_late_rearm(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS",
        0.01,
    )
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    detector.provisional_pending = True
    detector.block_prepare = True

    endpoint_task = asyncio.create_task(
        runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    await runtime._handle_independent_asr_final("timed-out-final", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._asr_speaker_candidate_decision_gate is None
    detector.prepare_release.set()
    await asyncio.wait_for(endpoint_task, 1)
    assert runtime._asr_speaker_candidate_decision_gate is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_timeout_count"] == 1
    assert diagnostics["speaker_gate_provisional_timeout_count"] == 1
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("resolve_before_provisional_resume", [False, True])
async def test_first_low_racing_provisional_prepare_promotes_exact_gate(
    resolve_before_provisional_resume: bool,
) -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None
    detector.provisional_pending = True
    first_prepare_entered = asyncio.Event()
    release_first_prepare = asyncio.Event()
    prepare_count = 0

    async def prepare_candidate_rejection(_candidate):
        nonlocal prepare_count
        prepare_count += 1
        if prepare_count == 1:
            first_prepare_entered.set()
            await release_first_prepare.wait()
        return detector.lease

    detector.prepare_candidate_rejection = prepare_candidate_rejection
    endpoint_task = asyncio.create_task(
        runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(first_prepare_entered.wait(), 1)

    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.first_observation_pending is False
    if resolve_before_provisional_resume:
        assert runtime._resolve_speaker_candidate_decision(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    release_first_prepare.set()
    await asyncio.wait_for(endpoint_task, 1)

    assert runtime._asr_speaker_candidate_decision_gate is (
        None if resolve_before_provisional_resume else gate
    )
    assert runtime._asr_speaker_candidate_decision_preparation is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_count"] == 1
    assert diagnostics["speaker_gate_provisional_promoted_count"] == 0
    if not resolve_before_provisional_resume:
        assert runtime._resolve_speaker_candidate_decision(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    await _close_dispatchers(runtime)


async def test_provisional_gate_promotes_and_completion_cannot_beat_rejection() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    provider_fence = _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    detector.provisional_pending = True
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        provisional=True,
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("non-owner-final", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)

    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    assert gate.first_observation_pending is False
    detector.prepare_entered.clear()
    detector.block_prepare = True
    assert runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    assert gate.rejection_task is not None
    assert not runtime._resolve_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        rejected=False,
    )

    detector.prepare_release.set()
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_not_awaited()
    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    abandoned.assert_awaited_once_with(turn_token)
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_provisional_promoted_count"] == 1
    assert diagnostics["speaker_gate_resolved_forward_count"] == 0
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    await _close_dispatchers(runtime)


async def test_gate_arm_diagnostics_report_detector_prepare_cancellation() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    detector.block_prepare = True

    task = asyncio.create_task(
        runtime._arm_speaker_candidate_decision(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_arm_prepare_cancelled_count"] == 1
    assert diagnostics["speaker_gate_arm_final_lock_cancelled_count"] == 0
    assert diagnostics["speaker_gate_armed_count"] == 0
    await _close_dispatchers(runtime)


async def test_gate_arm_diagnostics_report_final_lock_cancellation() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    await runtime._asr_final_lock.acquire()
    try:
        task = asyncio.create_task(
            runtime._arm_speaker_candidate_decision(
                _shadow_candidate(),
                activation_generation="profile-generation",
            )
        )
        await asyncio.wait_for(detector.prepare_entered.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        runtime._asr_final_lock.release()

    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_arm_prepare_cancelled_count"] == 0
    assert diagnostics["speaker_gate_arm_final_lock_cancelled_count"] == 1
    assert diagnostics["speaker_gate_armed_count"] == 0
    await _close_dispatchers(runtime)


async def test_gate_arm_diagnostics_report_common_authority_failure() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._asr_reserved_final_key = None

    assert not await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_arm_common_authority_count"] == 1
    assert diagnostics["speaker_gate_arm_prepare_cancelled_count"] == 0
    assert diagnostics["speaker_gate_arm_final_lock_cancelled_count"] == 0
    assert diagnostics["speaker_gate_armed_count"] == 0
    await _close_dispatchers(runtime)


async def test_degraded_verifier_blocks_new_gate_before_detector_prepare() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )

    runtime._mark_speaker_verifier_degraded()

    assert not await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    assert not detector.prepare_entered.is_set()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_arm_degraded_count"] == 1
    assert diagnostics["speaker_gate_armed_count"] == 0
    await _close_dispatchers(runtime)


async def test_degraded_health_transition_during_prepare_cannot_late_arm() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    detector.block_prepare = True
    arm_task = asyncio.create_task(
        runtime._arm_speaker_candidate_decision(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)

    runtime._mark_speaker_verifier_degraded()
    runtime._mark_speaker_verifier_healthy()
    detector.prepare_release.set()

    assert not await asyncio.wait_for(arm_task, 1)
    assert runtime._asr_speaker_candidate_decision_gate is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_arm_degraded_count"] == 1
    assert diagnostics["speaker_gate_armed_count"] == 0
    await _close_dispatchers(runtime)


async def test_degraded_verifier_retires_preparation_and_unowned_gate() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token, _provider_fence = (
        _seal_provider_candidate(runtime, detector)
    )
    candidate = _shadow_candidate()
    preparation = runtime._begin_speaker_candidate_decision_preparation(
        candidate,
        activation_generation="profile-generation",
    )
    assert preparation is not None

    runtime._mark_speaker_verifier_degraded()

    assert preparation.retired
    assert preparation.resolved.done()
    assert runtime._asr_speaker_candidate_decision_preparation is None
    runtime._mark_speaker_verifier_healthy()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None

    runtime._mark_speaker_verifier_degraded()

    assert runtime._asr_speaker_candidate_decision_gate is None
    assert gate.resolved.done()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_released_verifier_degraded_count"] == 1
    await _close_dispatchers(runtime)


async def test_degraded_verifier_does_not_steal_rejection_owned_gate() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    rejection_owner = asyncio.create_task(asyncio.Event().wait())
    gate.rejection_task = rejection_owner

    runtime._mark_speaker_verifier_degraded()

    assert runtime._asr_speaker_candidate_decision_gate is gate
    assert not gate.resolved.done()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_released_verifier_degraded_count"] == 0
    rejection_owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await rejection_owner
    gate.rejection_task = None
    assert runtime._resolve_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    await _close_dispatchers(runtime)


async def test_first_low_provider_gate_arms_while_turn_preparation_is_pending() -> None:
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    callbacks = _callbacks()

    async def delayed_prepare(_turn_token: VoiceTurnToken) -> bool:
        prepare_started.set()
        await prepare_release.wait()
        return True

    callbacks.on_prepare_turn.side_effect = delayed_prepare  # type: ignore[attr-defined]
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    _rewind_installed_candidate_to_unprepared(runtime)
    prepare_task = asyncio.create_task(
        runtime._prepare_independent_asr_turn(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)

    assert runtime._asr_turn_prepared is True
    assert runtime._asr_partial_turn_token is None
    assert runtime._asr_reserved_final_key == FinalKey.from_turn(turn_token)
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    assert gate.armed_while_preparing is True
    assert gate.turn_token == turn_token
    assert gate.final_key == FinalKey.from_turn(turn_token)

    prepare_release.set()
    await asyncio.wait_for(prepare_task, 1)

    assert runtime._asr_partial_turn_token == turn_token
    assert runtime._asr_speaker_candidate_decision_gate is gate
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_count"] == 1
    assert diagnostics["speaker_gate_armed_while_preparing_count"] == 1
    assert diagnostics["speaker_gate_released_prepare_failure_count"] == 0
    assert runtime._resolve_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
        rejected=False,
    )
    await _close_dispatchers(runtime)


async def test_provisional_gate_holds_early_final_until_prepare_then_rejects() -> None:
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)

    async def delayed_prepare(_turn_token: VoiceTurnToken) -> bool:
        prepare_started.set()
        await prepare_release.wait()
        return True

    callbacks.on_prepare_turn.side_effect = delayed_prepare  # type: ignore[attr-defined]
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    _rewind_installed_candidate_to_unprepared(runtime)
    prepare_task = asyncio.create_task(
        runtime._prepare_independent_asr_turn(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.armed_while_preparing is True
    provider_fence = _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )

    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("not-owner-final", 0, "qwen")
    )

    async def wait_for_gate_waiter() -> None:
        while not gate.wait_started:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_gate_waiter(), 0.2)
    assert final_task.done() is False
    callbacks.on_final.assert_not_awaited()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_stale_count"] == 0

    async def complete_preparation_and_reject() -> None:
        prepare_release.set()
        await prepare_task
        assert runtime._asr_partial_turn_token == turn_token
        assert runtime.request_speaker_candidate_rejection(
            candidate,
            activation_generation="profile-generation",
        )
        rejection_tasks = tuple(runtime._asr_rejection_tasks)
        assert len(rejection_tasks) == 1
        await asyncio.gather(*rejection_tasks)
        await final_task
        await runtime.wait_transcript_idle()

    await asyncio.wait_for(complete_preparation_and_reject(), 0.2)

    assert detector.lease is not None and detector.lease.commit_calls == 1
    callbacks.on_final.assert_not_awaited()
    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    abandoned.assert_awaited_once_with(turn_token)
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["speaker_gate_stale_count"] == 0
    assert diagnostics["rejection_task_applied_count"] == 1
    assert runtime._asr_speaker_candidate_decision_gate is None
    await _close_dispatchers(runtime)


async def test_real_speaker_shadow_rejects_final_after_first_low_during_prepare(
    monkeypatch,
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

    class _ScoringHost:
        alive = True

        async def score(
            self,
            _pcm16: bytes,
            *,
            timeout_seconds: float,
        ) -> float:
            assert timeout_seconds > 0
            return 0.20

    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)

    async def delayed_prepare(_turn_token: VoiceTurnToken) -> bool:
        prepare_started.set()
        await prepare_release.wait()
        return True

    callbacks.on_prepare_turn.side_effect = delayed_prepare  # type: ignore[attr-defined]
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    _rewind_installed_candidate_to_unprepared(runtime)

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
    monkeypatch.setattr(
        shadow,
        "_ensure_backend",
        AsyncMock(return_value=_ScoringHost()),
    )
    candidate = _shadow_candidate()
    checkpoint_pcm16 = b"\x21\x00" * (16_000 * 1_500 // 1_000)

    prepare_task = asyncio.create_task(
        runtime._prepare_independent_asr_turn(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)
    assert shadow.submit(
        checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    await shadow.wait_idle()

    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.armed_while_preparing is True
    assert runtime._asr_partial_turn_token is None

    prepare_release.set()
    await asyncio.wait_for(prepare_task, 1)
    provider_fence = _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    assert shadow.submit(
        checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    await shadow.wait_idle()

    async def wait_for_rejection() -> None:
        while not runtime._speaker_verifier_diagnostics()[
            "rejection_task_applied_count"
        ]:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_rejection(), 1)
    await runtime._handle_independent_asr_final("not-owner-final", 0, "qwen")
    await runtime.wait_transcript_idle()

    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_while_preparing_count"] == 1
    assert diagnostics["rejection_task_scheduled_count"] == 1
    assert diagnostics["rejection_task_applied_count"] == 1
    assert diagnostics["rejection_task_stale_count"] == 0
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["speaker_gate_stale_count"] == 0
    callbacks.on_final.assert_not_awaited()
    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    abandoned.assert_awaited_once_with(turn_token)
    assert runtime._asr_suppressed_final_key is None
    assert runtime._asr_speaker_candidate_decision_gate is None
    shadow_metrics = shadow.snapshot()
    assert shadow_metrics["callback_failure_count"] == 0
    assert shadow_metrics["stale_result_count"] == 0

    await shadow.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)


@pytest.mark.parametrize(
    ("completion_similarity", "expect_rejected"),
    [(0.20, True), (0.80, False)],
    ids=["stable-mismatch", "owner-recovery"],
)
async def test_real_speaker_shadow_2999ms_completion_confirms_waiting_final(
    monkeypatch,
    completion_similarity: float,
    expect_rejected: bool,
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

    class _ScoringHost:
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
            if self.score_count == 1:
                return 0.20
            return completion_similarity

    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
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
    scoring_host = _ScoringHost()
    monkeypatch.setattr(
        shadow,
        "_ensure_backend",
        AsyncMock(return_value=scoring_host),
    )
    candidate = _shadow_candidate()
    first_checkpoint_pcm16 = b"\x21\x00" * (16_000 * 1_500 // 1_000)
    below_second_checkpoint_pcm16 = b"\x22\x00" * (
        16_000 * 1_499 // 1_000
    )

    assert shadow.submit(
        first_checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    await shadow.wait_idle()
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("short-non-owner-final", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)

    assert shadow.submit(
        below_second_checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    assert shadow.finish_candidate(candidate)
    await shadow.wait_idle()
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    assert scoring_host.score_count == 2
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["rejection_request_failed_count"] == 0
    if expect_rejected:
        callbacks.on_final.assert_not_awaited()
        assert diagnostics["speaker_gate_resolved_reject_count"] == 1
        assert diagnostics["speaker_gate_resolved_forward_count"] == 0
        assert diagnostics["rejection_task_scheduled_count"] == 1
        assert diagnostics["rejection_task_applied_count"] == 1
        assert diagnostics["rejection_task_stale_count"] == 0
    else:
        callbacks.on_final.assert_awaited_once()
        assert diagnostics["speaker_gate_resolved_reject_count"] == 0
        assert diagnostics["speaker_gate_resolved_forward_count"] == 1
        assert diagnostics["rejection_task_scheduled_count"] == 0
    assert runtime._asr_speaker_candidate_decision_gate is None
    assert not composition._armed_candidates
    composition_diagnostics = composition.diagnostics_snapshot()
    assert composition_diagnostics["speaker_completion_count"] == 1
    assert (
        composition_diagnostics[
            "speaker_completion_after_first_checkpoint_count"
        ]
        == 1
    )
    shadow_metrics = shadow.snapshot()
    assert shadow_metrics["completion_count"] == 1
    assert shadow_metrics["completion_after_first_checkpoint_count"] == 1

    await shadow.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)


async def test_real_detector_fifo_2999ms_tail_rejects_only_second_turn(
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

    class _BlockingSecondScoreHost:
        alive = True

        def __init__(self) -> None:
            self.score_count = 0
            self.second_score_started = asyncio.Event()
            self.second_score_release = asyncio.Event()

        async def score(
            self,
            _pcm16: bytes,
            *,
            timeout_seconds: float,
        ) -> float:
            assert timeout_seconds > 0
            self.score_count += 1
            if self.score_count == 2:
                self.second_score_started.set()
                await self.second_score_release.wait()
            return 0.20

    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    placeholder = _RejectionDetector()
    session, lifecycle, turn_1 = _install_active_candidate(
        runtime,
        placeholder,
        provider="qwen",
        endpointing_mode="provider",
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
    scoring_host = _BlockingSecondScoreHost()
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

    result_a = await detector.feed(
        b"\x11\x00" * 160,
        ingress_token=turn_1.ingress,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    assert result_a.candidate is not None
    assert result_a.identity is not None
    assert await detector.bind_candidate(result_a.candidate, turn_1) is not None
    await detector.observe_provider_audio_ordered(
        b"\x11\x00" * 1_600,
        sample_rate_hz=16_000,
        identity=result_a.identity,
        sequence_no=1,
        split_before_audio=False,
    )
    pcm_2999ms = b"\x22\x00" * (16_000 * 2_999 // 1_000)
    await detector.observe_provider_audio_ordered(
        pcm_2999ms,
        sample_rate_hz=16_000,
        identity=result_a.identity,
        sequence_no=2,
        split_before_audio=True,
    )
    shadow_a = detector._provider_speaker_segments[0].candidate
    shadow_b = detector._provider_speaker_segments[1].candidate
    assert shadow_a is not None and shadow_b is not None

    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    await runtime._asr_audio_dispatcher.wait_idle()
    await shadow.wait_idle()
    fence_1 = runtime._asr_provider_candidate_fence
    assert fence_1 is not None
    assert detector._sealed_provider_candidate_rejection is None
    await runtime._handle_independent_asr_final("turn-1-final", 0, "qwen")
    await runtime.wait_transcript_idle()
    callbacks.on_final.assert_awaited_once()
    assert callbacks.on_final.await_args.args[0].text == "turn-1-final"

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )
    turn_2 = runtime._asr_partial_turn_token
    assert turn_2 is not None and turn_2 != turn_1
    result_b = await detector.feed(
        b"\x23\x00" * 160,
        ingress_token=turn_2.ingress,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    assert result_b.candidate == DetectorCandidateKey(0, 1)
    assert await detector.bind_candidate(result_b.candidate, turn_2) is not None

    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)
    fence_2 = runtime._asr_provider_candidate_fence
    assert fence_2 is not None and fence_2 != fence_1
    sealed_b = detector._sealed_provider_candidate_rejection
    assert sealed_b is not None
    assert sealed_b.shadow_candidate == shadow_b
    assert sealed_b.turn_token == turn_2
    await asyncio.wait_for(scoring_host.second_score_started.wait(), 1)
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    assert gate.candidate == shadow_b
    assert gate.turn_token == turn_2
    assert gate.lease.provider_fence == fence_2
    final_2 = asyncio.create_task(
        runtime._handle_independent_asr_final("turn-2-final", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)
    scoring_host.second_score_release.set()
    await shadow.wait_idle()
    await asyncio.wait_for(final_2, 1)
    await runtime.wait_transcript_idle()

    assert scoring_host.score_count == 2
    assert callbacks.on_final.await_count == 1
    abandoned.assert_awaited_once_with(turn_2)
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["rejection_task_applied_count"] == 1
    detector_diagnostics = detector.speaker_rejection_diagnostics_snapshot()
    assert detector_diagnostics["provider_speaker_segment_exact_snapshot_count"] == 1

    await detector.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("candidate_audio_ms", [2_999, 3_000])
async def test_real_speaker_shadow_pre_gate_race_rejects_before_final(
    monkeypatch,
    candidate_audio_ms: int,
) -> None:
    import numpy as np

    monkeypatch.setattr(
        runtime_module,
        "_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS",
        5.0,
    )

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

    class _BlockingFirstScoreHost:
        alive = True

        def __init__(self) -> None:
            self.score_count = 0
            self.first_score_started = asyncio.Event()
            self.first_score_release = asyncio.Event()

        async def score(
            self,
            _pcm16: bytes,
            *,
            timeout_seconds: float,
        ) -> float:
            assert timeout_seconds > 0
            self.score_count += 1
            if self.score_count == 1:
                self.first_score_started.set()
                await self.first_score_release.wait()
            return 0.20

    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
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
    scoring_host = _BlockingFirstScoreHost()
    monkeypatch.setattr(
        shadow,
        "_ensure_backend",
        AsyncMock(return_value=scoring_host),
    )
    candidate = _shadow_candidate()
    first_checkpoint_pcm16 = b"\x21\x00" * (16_000 * 1_500 // 1_000)
    remaining_candidate_pcm16 = b"\x22\x00" * (
        16_000 * (candidate_audio_ms - 1_500) // 1_000
    )

    assert shadow.submit(
        first_checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    await asyncio.wait_for(scoring_host.first_score_started.wait(), 1)
    assert shadow.submit(
        remaining_candidate_pcm16,
        sample_rate_hz=16_000,
        candidate=candidate,
    )
    assert shadow.finish_candidate(candidate)
    detector.pending_provider_speaker_candidate = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda provider_fence: (
            candidate
            if detector.lease is not None
            and detector.lease.provider_fence == provider_fence
            and shadow.requires_provisional_decision(candidate)
            else None
        )
    )
    final_key = FinalKey.from_turn(turn_token)
    runtime._asr_transcript_dispatcher.release(final_key)
    runtime._asr_reserved_final_key = None

    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)

    gate = runtime._asr_speaker_candidate_decision_gate
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert gate is not None and gate.first_observation_pending is True
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("early-provider-final", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)
    assert final_task.done() is False
    callbacks.on_final.assert_not_awaited()

    scoring_host.first_score_release.set()
    await asyncio.wait_for(shadow.wait_idle(), 1)
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    assert scoring_host.score_count == 2
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(turn_token)
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_provisional_armed_count"] == 1
    assert diagnostics["speaker_gate_provisional_promoted_count"] == 1
    assert diagnostics["speaker_gate_final_before_first_observation_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    assert diagnostics["rejection_task_applied_count"] == 1

    await shadow.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)


async def test_late_short_candidate_completion_cannot_release_successor_gate(
    monkeypatch,
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

    confirmation_started = asyncio.Event()
    confirmation_release = asyncio.Event()

    class _ScoringHost:
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
            if self.score_count == 2:
                confirmation_started.set()
                await confirmation_release.wait()
            return 0.20

    monkeypatch.setattr(
        runtime_module,
        "_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS",
        0.01,
    )
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
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
    monkeypatch.setattr(
        shadow,
        "_ensure_backend",
        AsyncMock(return_value=_ScoringHost()),
    )
    old_candidate = _shadow_candidate()
    first_checkpoint_pcm16 = b"\x21\x00" * (16_000 * 1_500 // 1_000)
    below_second_checkpoint_pcm16 = b"\x22\x00" * (
        16_000 * 1_499 // 1_000
    )
    assert shadow.submit(
        first_checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=old_candidate,
    )
    await shadow.wait_idle()
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    assert shadow.submit(
        below_second_checkpoint_pcm16,
        sample_rate_hz=16_000,
        candidate=old_candidate,
    )
    assert shadow.finish_candidate(old_candidate)
    await asyncio.wait_for(confirmation_started.wait(), 1)

    await runtime._handle_independent_asr_final("timed-out-final", 0, "qwen")
    await runtime.wait_transcript_idle()
    assert runtime._speaker_verifier_diagnostics()["speaker_gate_timeout_count"] == 1
    callbacks.on_final.assert_awaited_once()

    successor_detector = _RejectionDetector()
    _install_active_candidate(
        runtime,
        successor_detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    runtime._asr_accepted_final_keys.clear()
    successor_candidate = SpeakerShadowCandidateKey(
        old_candidate.detector_epoch + 1,
        old_candidate.shadow_generation + 1,
        "provider_candidate",
    )
    assert successor_detector.lease is not None
    successor_detector.lease.shadow_candidate = successor_candidate
    assert await runtime._arm_speaker_candidate_decision(
        successor_candidate,
        activation_generation="profile-generation",
    )
    successor_gate = runtime._asr_speaker_candidate_decision_gate
    assert successor_gate is not None

    confirmation_release.set()
    await shadow.wait_idle()
    rejection_tasks = tuple(runtime._asr_rejection_tasks)
    if rejection_tasks:
        await asyncio.gather(*rejection_tasks)

    assert runtime._asr_speaker_candidate_decision_gate is successor_gate
    assert not successor_gate.resolved.done()
    assert runtime._resolve_speaker_candidate_decision(
        successor_candidate,
        activation_generation="profile-generation",
        rejected=False,
    )

    await shadow.close()
    composition.close()
    profile.close()
    await _close_dispatchers(runtime)


async def test_provider_gate_never_treats_wrong_nonempty_partial_token_as_preparing() -> (
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
    runtime._asr_partial_turn_token = VoiceTurnToken(
        ingress=turn_token.ingress,
        turn_id=turn_token.turn_id + 1,
    )

    assert not await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    assert runtime._asr_speaker_candidate_decision_gate is None
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_count"] == 0
    assert diagnostics["speaker_gate_armed_while_preparing_count"] == 0
    assert detector.lease is not None and detector.lease.commit_calls == 0
    await _close_dispatchers(runtime)


async def test_provisional_gate_cannot_commit_before_partial_token_is_established() -> (
    None
):
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    callbacks = _callbacks()

    async def delayed_prepare(_turn_token: VoiceTurnToken) -> bool:
        prepare_started.set()
        await prepare_release.wait()
        return True

    callbacks.on_prepare_turn.side_effect = delayed_prepare  # type: ignore[attr-defined]
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    _rewind_installed_candidate_to_unprepared(runtime)
    prepare_task = asyncio.create_task(
        runtime._prepare_independent_asr_turn(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.armed_while_preparing is True

    outcome = await runtime._reject_speaker_candidate(
        candidate,
        activation_generation="profile-generation",
        decision_gate=gate,
    )

    assert outcome is CandidateRejectionOutcome.STALE
    assert detector.lease is not None and detector.lease.commit_calls == 0
    assert runtime._asr_suppressed_final_key is None
    assert runtime._asr_candidate_rejection is None
    prepare_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prepare_task
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("prepare_outcome", ["rejected", "cancelled"])
async def test_pending_prepare_failure_releases_only_its_provisional_gate(
    prepare_outcome: str,
) -> None:
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    callbacks = _callbacks()

    async def delayed_prepare(_turn_token: VoiceTurnToken) -> bool:
        prepare_started.set()
        await prepare_release.wait()
        return False

    callbacks.on_prepare_turn.side_effect = delayed_prepare  # type: ignore[attr-defined]
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    _rewind_installed_candidate_to_unprepared(runtime)
    prepare_task = asyncio.create_task(
        runtime._prepare_independent_asr_turn(runtime._asr_session_epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.armed_while_preparing is True

    if prepare_outcome == "cancelled":
        prepare_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prepare_task
    else:
        prepare_release.set()
        await asyncio.wait_for(prepare_task, 1)

    assert runtime._asr_speaker_candidate_decision_gate is None
    assert runtime._asr_partial_turn_token is None
    assert runtime._asr_turn_prepared is False
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_released_prepare_failure_count"] == 1
    assert detector.lease is not None and detector.lease.commit_calls == 0
    await _close_dispatchers(runtime)


async def test_authoritative_gate_does_not_downgrade_after_partial_token_is_cleared() -> (
    None
):
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None and gate.armed_while_preparing is False
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    runtime._asr_partial_turn_token = None

    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._asr_speaker_candidate_decision_gate is None
    assert detector.lease is not None and detector.lease.commit_calls == 0
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_stale_count"] == 1
    assert diagnostics["speaker_gate_waited_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_gate_timeout_forwards_final_and_rejects_late_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS",
        0.01,
    )
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )

    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._asr_speaker_candidate_decision_gate is None
    assert not runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_armed_count"] == 1
    assert diagnostics["speaker_gate_waited_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 0
    await _close_dispatchers(runtime)


async def test_provisional_gate_timeout_forwards_and_cannot_rearm_late_score(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_SPEAKER_CANDIDATE_DECISION_TIMEOUT_SECONDS",
        0.01,
    )
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    detector.provisional_pending = True
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        provisional=True,
    )

    await runtime._handle_independent_asr_final("timed-out-owner-final", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._asr_speaker_candidate_decision_gate is None
    assert not await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_timeout_count"] == 1
    assert diagnostics["speaker_gate_provisional_timeout_count"] == 1
    assert diagnostics["speaker_gate_final_before_first_observation_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_gate_forward_resolution_releases_waiting_final() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("owner", 0, "qwen")
    )
    while not gate.wait_started:
        await asyncio.sleep(0)

    assert runtime._resolve_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        rejected=False,
    )
    await asyncio.wait_for(final_task, 1)
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["speaker_gate_resolved_forward_count"] == 1
    assert diagnostics["speaker_gate_timeout_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_gate_profile_change_is_stale_and_fails_open() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    runtime._speaker_verifier_activation_generation = "replacement-profile"

    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._speaker_verifier_diagnostics()["speaker_gate_stale_count"] == 1
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("stale_cause", ["runtime", "turn"])
async def test_provider_gate_runtime_or_turn_change_releases_without_rejection(
    stale_cause: str,
) -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    if stale_cause == "runtime":
        runtime._asr_audio_generation += 1
    else:
        runtime._asr_sealed_turn_token = VoiceTransportToken(
            turn=VoiceTurnToken(
                ingress=turn_token.ingress,
                turn_id=turn_token.turn_id + 1,
            ),
            transport_generation=lifecycle.snapshot.transport_generation,
        )

    await runtime._handle_independent_asr_final("stale", 0, "qwen")

    assert runtime._asr_speaker_candidate_decision_gate is None
    assert detector.lease is not None and detector.lease.commit_calls == 0
    assert runtime._speaker_verifier_diagnostics()["speaker_gate_stale_count"] == 1
    callbacks.on_final.assert_not_awaited()
    await _close_dispatchers(runtime)


async def test_provider_gate_resolution_exception_fails_open() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    assert await runtime._arm_speaker_candidate_decision(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    gate.resolved.set_exception(RuntimeError("decision failed"))

    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    assert runtime._speaker_verifier_diagnostics()["speaker_gate_stale_count"] == 1
    await _close_dispatchers(runtime)


async def test_provider_rejection_task_exception_releases_gate_fail_open() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    runtime._reject_speaker_candidate = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("reject failed")
    )
    assert runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()

    callbacks.on_final.assert_awaited_once()
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["rejection_task_failure_count"] == 1
    assert diagnostics["speaker_gate_stale_count"] == 1
    assert diagnostics["speaker_gate_resolved_reject_count"] == 0
    await _close_dispatchers(runtime)


async def test_provider_rejection_cannot_commit_after_gate_deadline() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, lifecycle, turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    _seal_installed_provider_candidate(
        runtime,
        detector,
        session,
        lifecycle,
        turn_token,
    )
    gate.deadline = runtime_module.time.monotonic() - 1.0

    assert runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )
    task = next(iter(runtime._asr_rejection_tasks))
    assert await asyncio.wait_for(task, 1) is CandidateRejectionOutcome.STALE
    await asyncio.sleep(0)

    assert detector.lease is not None and detector.lease.commit_calls == 0
    assert runtime._asr_suppressed_final_key is None
    assert runtime._speaker_verifier_diagnostics()["speaker_gate_stale_count"] == 1
    await _close_dispatchers(runtime)


async def test_active_provider_rejection_is_not_limited_by_final_deadline() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    _session, _lifecycle, _turn_token = _install_active_candidate(
        runtime,
        detector,
        provider="qwen",
        endpointing_mode="provider",
    )
    candidate = _shadow_candidate()
    assert await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    gate = runtime._asr_speaker_candidate_decision_gate
    assert gate is not None
    gate.deadline = runtime_module.time.monotonic() - 1.0

    outcome = await runtime._reject_speaker_candidate(
        candidate,
        activation_generation="profile-generation",
        decision_gate=gate,
    )

    assert outcome is CandidateRejectionOutcome.APPLIED
    assert detector.lease is not None and detector.lease.commit_calls == 1
    runtime._resolve_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
        rejected=True,
    )
    await _close_dispatchers(runtime)


async def test_smart_turn_rejection_does_not_require_provider_gate() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    runtime._speaker_verifier_activation_generation = "profile-generation"
    runtime._reject_speaker_candidate = AsyncMock(  # type: ignore[method-assign]
        return_value=CandidateRejectionOutcome.APPLIED
    )
    candidate = _smart_turn_shadow_candidate()

    assert not await runtime._arm_speaker_candidate_decision(
        candidate,
        activation_generation="profile-generation",
    )
    assert runtime.request_speaker_candidate_rejection(
        candidate,
        activation_generation="profile-generation",
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    runtime._reject_speaker_candidate.assert_awaited_once()  # type: ignore[attr-defined]
    diagnostics = runtime._speaker_verifier_diagnostics()
    assert diagnostics["rejection_task_applied_count"] == 1
    assert diagnostics["speaker_gate_armed_count"] == 0
    assert diagnostics["speaker_gate_resolved_reject_count"] == 0


async def test_sealed_provider_rejection_is_stale_after_final_wins_lock() -> None:
    callbacks = _callbacks()
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, _lifecycle, _turn_token, provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )

    await runtime._handle_independent_asr_final("forwarded", 0, "qwen")
    await runtime.wait_transcript_idle()
    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    assert outcome is CandidateRejectionOutcome.STALE
    callbacks.on_final.assert_awaited_once()
    detector.complete_provider_candidate.assert_awaited_once_with(provider_fence)
    assert detector.lease is not None and detector.lease.commit_calls == 0
    assert runtime._asr_suppressed_final_key is None
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("stale_cause", ["profile", "runtime", "turn"])
async def test_sealed_provider_rejection_rechecks_exact_authority_after_prepare(
    stale_cause: str,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.block_prepare = True
    session, lifecycle, turn_token, _provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )
    task = asyncio.create_task(
        runtime._reject_speaker_candidate(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    if stale_cause == "profile":
        runtime._speaker_verifier_activation_generation = "replacement-profile"
    elif stale_cause == "runtime":
        runtime._asr_audio_generation += 1
    else:
        runtime._asr_sealed_turn_token = VoiceTransportToken(
            turn=VoiceTurnToken(
                ingress=turn_token.ingress,
                turn_id=turn_token.turn_id + 1,
            ),
            transport_generation=lifecycle.snapshot.transport_generation,
        )
    detector.prepare_release.set()

    outcome = await asyncio.wait_for(task, 1)

    assert outcome is CandidateRejectionOutcome.STALE
    assert detector.lease is not None and detector.lease.commit_calls == 0
    assert runtime._asr_suppressed_final_key is None
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    await _close_dispatchers(runtime)


async def test_sealed_provider_rejection_preserves_pending_successor() -> None:
    abandoned = AsyncMock()
    callbacks = _callbacks(abandoned=abandoned)
    runtime = IndependentAsrRuntime(callbacks)
    detector = _RejectionDetector()
    session, lifecycle, rejected_turn, _provider_fence = _seal_provider_candidate(
        runtime,
        detector,
    )
    lifecycle.mark_pending_turn_speech()
    pending_pcm = b"\x01\x00" * 320
    decision = lifecycle.accept_audio(pending_pcm, sample_rate_hz=16_000)
    assert decision.disposition.value == "buffer"

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )
    await runtime._handle_independent_asr_final("not-owner", 0, "qwen")
    await runtime._asr_audio_dispatcher.wait_idle()

    assert outcome is CandidateRejectionOutcome.APPLIED
    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert lifecycle.snapshot.turn_id != rejected_turn.turn_id
    assert runtime._asr_partial_turn_token is not None
    assert runtime._asr_partial_turn_token.turn_id == lifecycle.snapshot.turn_id
    session.stream_audio.assert_awaited_with(pending_pcm, sample_rate_hz=16_000)
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    callbacks.on_final.assert_not_awaited()
    abandoned.assert_awaited_once_with(rejected_turn)
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("stale_cause", ["final", "provider_endpoint", "profile"])
async def test_candidate_rejection_forwards_when_authority_is_stale(
    stale_cause: str,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, lifecycle, _turn_token = _install_active_candidate(runtime, detector)
    if stale_cause == "final":
        lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    elif stale_cause == "provider_endpoint":
        runtime._asr_provider_candidate_fence = object()
    else:
        runtime._speaker_verifier_activation_generation = "replacement-profile"

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    assert outcome is CandidateRejectionOutcome.STALE
    diagnostics = runtime._speaker_verifier_diagnostics()
    expected_counter = (
        "rejection_stale_initial_count"
        if stale_cause == "profile"
        else "rejection_stale_candidate_fence_count"
    )
    assert diagnostics[expected_counter] == 1
    assert detector.lease is not None and detector.lease.commit_calls == 0
    session.close.assert_not_awaited()
    detector.reset.assert_not_awaited()
    await _close_dispatchers(runtime)


@pytest.mark.parametrize("stale_cause", ["transport", "profile"])
async def test_candidate_rejection_rechecks_fences_after_prepare_await(
    stale_cause: str,
) -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.block_prepare = True
    session, lifecycle, _turn_token = _install_active_candidate(runtime, detector)
    task = asyncio.create_task(
        runtime._reject_speaker_candidate(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    if stale_cause == "transport":
        lifecycle.invalidate_transport()
    else:
        runtime._speaker_verifier_activation_generation = "replacement-profile"
    detector.prepare_release.set()

    outcome = await asyncio.wait_for(task, 1)

    assert outcome is CandidateRejectionOutcome.STALE
    assert detector.lease is not None and detector.lease.commit_calls == 0
    session.close.assert_not_awaited()
    await _close_dispatchers(runtime)


async def test_cleanup_failure_keeps_drop_and_watchdog_releases_suppression(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_CANDIDATE_REJECTION_WATCHDOG_SECONDS",
        0.0,
    )
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.reset.side_effect = [RuntimeError("reset failed"), None]
    session, _lifecycle, _turn_token = _install_active_candidate(runtime, detector)
    shadow = SimpleNamespace(close=AsyncMock())
    runtime._speaker_verifier_factory = MagicMock(return_value=shadow)

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    async def wait_until_released() -> None:
        while runtime._asr_candidate_rejection is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_released(), 1.0)
    await asyncio.sleep(0)

    assert outcome is CandidateRejectionOutcome.APPLIED_CLEANUP_DEGRADED
    assert detector.lease is not None and detector.lease.commit_calls == 1
    session.close.assert_awaited_once_with()
    assert detector.replace_speaker_verifier.await_args_list == [
        call(None),
        call(shadow),
    ]
    assert detector.reset.await_count == 2
    assert runtime._asr_candidate_rejection is None
    assert runtime._speaker_verifier_diagnostics()["rejection_task_failure_count"] == 0
    runtime._ensure_transport_restart_task.assert_called_once_with()
    await _close_dispatchers(runtime)


async def test_rejection_watchdog_retries_verifier_reinstall(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_CANDIDATE_REJECTION_WATCHDOG_SECONDS",
        0.0,
    )
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    first_shadow = SimpleNamespace(close=AsyncMock())
    second_shadow = SimpleNamespace(close=AsyncMock())
    detector.reset.side_effect = [RuntimeError("reset failed"), None]
    detector.replace_speaker_verifier.side_effect = [
        None,
        RuntimeError("reinstall failed"),
        None,
    ]
    session, _lifecycle, _turn_token = _install_active_candidate(runtime, detector)
    runtime._speaker_verifier_factory = MagicMock(
        side_effect=[first_shadow, second_shadow]
    )

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    async def wait_until_released() -> None:
        while runtime._asr_candidate_rejection is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_released(), 1.0)
    assert outcome is CandidateRejectionOutcome.APPLIED_CLEANUP_DEGRADED
    assert detector.replace_speaker_verifier.await_args_list == [
        call(None),
        call(first_shadow),
        call(second_shadow),
    ]
    first_shadow.close.assert_awaited_once_with()
    second_shadow.close.assert_not_awaited()
    assert not runtime._speaker_verifier_degraded
    assert runtime._asr_candidate_rejection is None
    session.close.assert_awaited_once_with()
    await _close_dispatchers(runtime)


async def test_rejection_watchdog_bounds_stuck_recovery_and_resumes_asr(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_CANDIDATE_REJECTION_WATCHDOG_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        runtime_module,
        "_CANDIDATE_REJECTION_RECOVERY_STEP_TIMEOUT_SECONDS",
        0.02,
    )
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.reset.side_effect = [RuntimeError("reset failed"), None]

    async def never_replace(_shadow) -> None:
        await asyncio.Event().wait()

    detector.replace_speaker_verifier.side_effect = never_replace
    _install_active_candidate(runtime, detector)
    runtime._speaker_verifier_factory = MagicMock(
        side_effect=lambda: SimpleNamespace(close=AsyncMock())
    )
    loop = asyncio.get_running_loop()
    started_at = loop.time()

    outcome = await runtime._reject_speaker_candidate(
        _shadow_candidate(),
        activation_generation="profile-generation",
    )

    async def wait_until_released() -> None:
        while runtime._asr_candidate_rejection is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_released(), 0.5)
    assert loop.time() - started_at < 0.2
    assert outcome is CandidateRejectionOutcome.APPLIED_CLEANUP_DEGRADED
    assert runtime._speaker_verifier_degraded
    assert runtime._asr_candidate_rejection is None
    runtime._ensure_transport_restart_task.assert_called_once_with()
    await _close_dispatchers(runtime)


async def test_late_rejection_cleanup_does_not_reset_recovered_detector(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_CANDIDATE_REJECTION_WATCHDOG_SECONDS",
        0.0,
    )
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    session, _lifecycle, _turn_token = _install_active_candidate(runtime, detector)
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def blocking_close() -> None:
        close_started.set()
        await close_release.wait()

    session.close = AsyncMock(side_effect=blocking_close)
    rejection_task = asyncio.create_task(
        runtime._reject_speaker_candidate(
            _shadow_candidate(),
            activation_generation="profile-generation",
        )
    )
    await asyncio.wait_for(close_started.wait(), 1.0)

    async def wait_until_recovered() -> None:
        while runtime._asr_candidate_rejection is not None:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_recovered(), 1.0)
    close_release.set()
    outcome = await asyncio.wait_for(rejection_task, 1.0)

    assert outcome is CandidateRejectionOutcome.APPLIED_CLEANUP_DEGRADED
    assert detector.reset.await_count == 1
    runtime._ensure_transport_restart_task.assert_called_once_with()
    await _close_dispatchers(runtime)


async def test_close_cancels_and_joins_owned_rejection_task() -> None:
    runtime = IndependentAsrRuntime(_callbacks())
    detector = _RejectionDetector()
    detector.block_prepare = True
    _session, _lifecycle, turn_token = _install_active_candidate(runtime, detector)
    runtime._asr_suppressed_final_key = FinalKey.from_turn(turn_token)

    assert runtime.request_speaker_candidate_rejection(
        _smart_turn_shadow_candidate(),
        activation_generation="profile-generation",
    )
    await asyncio.wait_for(detector.prepare_entered.wait(), 1)
    tasks = tuple(runtime._asr_rejection_tasks)

    await asyncio.wait_for(runtime.close(), 1)

    assert tasks and all(task.done() for task in tasks)
    assert runtime._asr_rejection_tasks == set()
    assert runtime._asr_suppressed_final_key is None
    detector.close.assert_awaited_once_with()
