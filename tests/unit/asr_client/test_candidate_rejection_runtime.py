from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import main_logic.asr_client.runtime as runtime_module
from main_logic.asr_client.candidate_control import CandidateRejectionOutcome
from main_logic.asr_client.endpointing.detector import (
    DetectorCandidateKey,
    ProviderCandidateFence,
)
from main_logic.asr_client.lifecycle import (
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
from main_logic.voice_turn.contracts import VoiceTurnToken


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
        self.release_deferred_turn = AsyncMock()

    async def prepare_candidate_rejection(self, _candidate):
        self.prepare_entered.set()
        if self.block_prepare:
            await self.prepare_release.wait()
        return self.lease


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
