import ast
import asyncio
import hashlib
import inspect
import json
import logging
import textwrap
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from main_logic.asr_client import VoiceIdentityActivationResult
from main_logic.core import LLMSessionManager
from main_logic.core.asr_runtime import (
    AsrRuntimeMixin,
    _HotSwapAudioFrame,
    _ONSET_TRUST_WINDOW_S,
)
from main_logic.core.multimodal_turn import (
    _MAX_LIVE_TURN_RECORDS,
    _MAX_PRERECORD_VISUAL_VALIDATIONS,
)
from main_logic.asr_client.runtime import (
    AsrRuntimeCallbacks,
    AsrStartResult,
    AsrStartStatus,
    IndependentAsrRuntime,
)
from main_logic.asr_client.endpointing.detector_runtime import DetectorFeedResult, DetectorRuntime
from main_logic.voice_input import VoiceInputDispatchResult
from main_logic.voice_input.activation import ActivationState
from main_logic.voice_input.consumers import CoreChatTurnContext
from main_logic.asr_client.lifecycle import (
    AudioDisposition,
    VoiceLifecycleConfig,
    VoiceLifecycleEvent,
    VoiceLifecycleState,
    VoiceTurnToken,
    VoiceRouteMode,
)
from main_logic.asr_client.lifecycle import VoiceInputLifecycleController
from main_logic.asr_client.provider_policy import resolve_provider_policy
from main_logic.voice_turn.activity_evidence import RnnoiseEvidence
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from main_logic.voice_identity_service.activation_runtime import (
    VoiceSessionActivationRuntime,
)
from main_logic.voice_identity_service.activation_scoring import (
    ActivationScoreResult,
    ActivationScoreStatus,
)
from main_logic.voice_turn.contracts import (
    AsrFailureEvent,
    AsrLifecycleNotification,
    AsrStatusEvent,
    AsrSubmitResult,
    AsrSubmitStatus,
    SpeechActivityEvent,
    VoiceIngressToken,
    VoicePartialEvent,
    VoiceTranscriptEvent,
)
from main_logic.voice_turn.contracts import EvaluationStatus, TurnDecision
from main_logic.asr_client.endpointing.coordinator import CoordinatorState
from main_logic.asr_client.endpointing.detector import (
    BoundDetectorTurn,
    CoreDetectorEventEnvelope,
    DetectorCandidateKey,
    DetectorIngressIdentity,
    ProviderCandidateFence,
    DetectorRuntimeEvent,
    DetectorTurnEvent,
    DetectorSubmitResult,
    DetectorSubmitStatus,
)
import main_logic.core.asr_runtime as core_asr_runtime_module
import main_logic.core as core_module
import main_logic.voice_turn.audio_input as audio_input_module
from utils import preferences


pytestmark = pytest.mark.asyncio


class _Runtime(AsrRuntimeMixin):
    def __init__(self) -> None:
        self._init_asr_runtime_state()
        self._voice_lease_synchronized = True
        self._voice_lease_owner = "core"
        self._voice_input_suppressed = False
        self.lanlan_name = "Test"
        self.session = type("Omni", (), {})()
        self.session.create_response = AsyncMock()
        self.session.handle_interruption = AsyncMock()
        self.handle_new_message = AsyncMock()
        self.handle_input_transcript = AsyncMock(return_value=True)
        self.send_status = AsyncMock()

    def __getattr__(self, name: str):
        component = self.__dict__.get("_asr_runtime")
        if component is not None and hasattr(component, name):
            return getattr(component, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        component = self.__dict__.get("_asr_runtime")
        if name in {
            "_asr_route_mode",
            # Keep the operation generation on the instance so reads observe
            # bumps, matching production (writes routed to the component
            # would freeze it at its initial value for every read).
            "_asr_route_operation_generation",
            "_microphone_route_generation",
            "_independent_asr_provider",
            "_independent_asr_route_key",
            "_voice_input_audio_pipeline",
        }:
            object.__setattr__(self, name, value)
            return
        if component is not None and (
            name.startswith("_asr_")
            or name
            in {
                "_voice_input_resource_optimization_enabled",
            }
        ):
            setattr(component, name, value)
            if name == "_asr_lifecycle" and value is not None:
                component._asr_current_ingress_token = self._capture_ingress_token()
            return
        object.__setattr__(self, name, value)


class _GateAsyncLock:
    def __init__(self) -> None:
        self.requested = asyncio.Event()
        self.release = asyncio.Event()

    async def __aenter__(self):
        self.requested.set()
        await self.release.wait()
        return self

    async def __aexit__(self, *_exc_info) -> None:
        return None

async def test_external_voice_suppression_aborts_once_and_restores_pcm_gate() -> None:
    runtime = _Runtime()
    runtime._invalidate_voice_pcm_sync = MagicMock()
    runtime._abort_independent_asr = AsyncMock()
    assert runtime._voice_input_accepts_pcm() is True

    await runtime.set_voice_input_suppressed(
        "voice_identity_enrollment",
        suppressed=True,
    )
    await runtime.set_voice_input_suppressed(
        "voice_identity_enrollment",
        suppressed=True,
    )

    assert runtime._voice_input_accepts_pcm() is False
    runtime._abort_independent_asr.assert_awaited_once_with(
        "voice_identity_enrollment"
    )
    assert runtime._invalidate_voice_pcm_sync.call_count == 1

    await runtime.set_voice_input_suppressed(
        "voice_identity_enrollment",
        suppressed=False,
    )

    assert runtime._voice_input_accepts_pcm() is True
    assert runtime._invalidate_voice_pcm_sync.call_count == 2

async def test_independent_route_sends_pcm_to_asr_only() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    await _install_active_smart_turn(runtime)

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    assert consumed is True
    asr.stream_audio.assert_awaited_once_with(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    assert runtime._asr_audio_bytes == 320
    assert runtime._omni_mic_audio_bytes == 0

async def test_stale_submit_drops_only_current_frame() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.STALE)
    )

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert runtime._asr_route_mode == "independent"

async def test_unavailable_submit_blocks_core_route() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "qwen"
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
    )
    clear_queue = MagicMock(wraps=runtime._clear_audio_stream_queue)
    clear_cache = MagicMock(wraps=runtime.hot_swap_audio_cache.clear)
    runtime._clear_audio_stream_queue = clear_queue
    runtime.hot_swap_audio_cache.clear = clear_cache

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert runtime._asr_route_mode == "blocked"
    clear_queue.assert_called_once_with("independent_asr_unavailable")
    clear_cache.assert_called_once_with()

async def test_stale_unavailable_submit_cannot_block_replacement_route() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "provider-a"
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()

    async def unavailable_after_replacement(*_args, **_kwargs):
        submit_started.set()
        await release_submit.wait()
        return AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)

    runtime._asr_runtime.submit = AsyncMock(side_effect=unavailable_after_replacement)
    clear_queue = MagicMock(wraps=runtime._clear_audio_stream_queue)
    clear_cache = MagicMock(wraps=runtime.hot_swap_audio_cache.clear)
    runtime._clear_audio_stream_queue = clear_queue
    runtime.hot_swap_audio_cache.clear = clear_cache
    routed = asyncio.create_task(
        runtime._route_microphone_audio(
            b"\x01\x00" * 160,
            sample_rate_hz=16_000,
        )
    )
    await asyncio.wait_for(submit_started.wait(), 1)

    new_core_session = SimpleNamespace(stream_audio=AsyncMock())
    new_asr_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime.session = new_core_session
    runtime._asr_runtime._asr_audio_generation += 1
    runtime._asr_session = new_asr_session
    runtime._independent_asr_provider = "provider-b"
    runtime._asr_runtime._asr_current_ingress_token = runtime._capture_ingress_token()
    release_submit.set()
    await asyncio.wait_for(routed, 1)

    assert runtime._asr_route_mode == "independent"
    assert runtime._independent_asr_provider == "provider-b"
    assert runtime.session is new_core_session
    assert runtime._asr_session is new_asr_session
    clear_queue.assert_not_called()
    clear_cache.assert_not_called()

async def test_async_detector_orders_pre_roll_before_smart_turn_seal() -> None:
    class Vad:
        def load(self) -> bool:
            return True

        def close(self) -> None:
            return None

    class Gate:
        def feed(self, _pcm16: bytes):
            return (
                SpeechActivityEvent.SPEECH_STARTED,
                SpeechActivityEvent.CANDIDATE_PAUSE,
            )

        def reset(self) -> None:
            return None

    class Coordinator:
        state = CoordinatorState.IDLE

        def push_audio(self, _pcm16: bytes) -> None:
            return None

        async def on_activity_event(self, event) -> None:
            self.state = (
                CoordinatorState.PAUSE_CANDIDATE
                if event is SpeechActivityEvent.CANDIDATE_PAUSE
                else CoordinatorState.SPEECH_ACTIVE
            )

        async def evaluate_buffered(self):
            return SimpleNamespace(
                status=EvaluationStatus.OK,
                decision=TurnDecision.COMPLETE,
            )

        async def prepare_predictor(self) -> bool:
            return True

        async def reset(self) -> None:
            self.state = CoordinatorState.IDLE

        async def close(self) -> None:
            self.state = CoordinatorState.CLOSED

        async def unload_predictor(self) -> None:
            return None

    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    asr.signal_user_activity_end = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = "glm"
    runtime._asr_route_mode = "independent"
    lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("glm", "manual"),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_lifecycle = lifecycle
    detector: DetectorRuntime

    async def on_event(event) -> None:
        assert runtime._asr_detector_dispatcher.submit_nowait(
            CoreDetectorEventEnvelope(
                event=event,
                detector_ref=detector,
                lifecycle_ref=lifecycle,
                session_epoch=runtime._asr_session_epoch,
            )
        )

    detector = DetectorRuntime(
        vad=Vad(),
        gate=Gate(),
        provider_policy=resolve_provider_policy("glm", "manual"),
        coordinator=Coordinator(),
        on_event=on_event,
    )
    runtime._asr_detector = detector
    pcm16 = b"\x01\x00" * 160

    assert await runtime._route_microphone_audio(
        pcm16,
        sample_rate_hz=16_000,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    for _ in range(200):
        if asr.signal_user_activity_end.await_count:
            break
        await asyncio.sleep(0.001)
    await runtime._asr_detector_dispatcher.wait_idle()
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once_with(pcm16, sample_rate_hz=16_000)
    asr.signal_user_activity_end.assert_awaited_once()
    assert runtime._omni_mic_audio_bytes == 0
    await detector.close()

@pytest.mark.parametrize("provider", ["qwen", "soniox"])
async def test_manual_streaming_provider_waits_for_smart_turn(
    provider: str,
) -> None:
    runtime = _Runtime()
    lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, "manual"),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _FailedSmartTurnDetector()
    turn_token = VoiceTurnToken(
        VoiceIngressToken(1, "socket", 1, 1, 1),
        turn_id=1,
    )

    assert runtime._asr_endpointing_ready(lifecycle, detector, turn_token) is False

async def test_enforced_lifecycle_suppresses_local_silence_upload() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = type(
        "Detector",
        (),
        {"feed": AsyncMock(return_value=DetectorFeedResult((), True))},
    )()

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    assert consumed is True
    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_lifecycle.pre_roll_bytes == 320

async def test_local_speech_wake_uploads_pre_roll_to_independent_asr() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _ReadyDetector()
    detector.feed = AsyncMock(
        side_effect=[
            DetectorFeedResult((), True),
            DetectorFeedResult((SpeechActivityEvent.SPEECH_STARTED,), True),
        ]
    )
    runtime._asr_detector = detector

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._route_microphone_audio(
        b"\x02\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once_with(
        (b"\x01\x00" * 160) + (b"\x02\x00" * 160),
        sample_rate_hz=16_000,
    )
    runtime.session.handle_interruption.assert_awaited_once_with()

async def test_game_consumer_accepts_real_pcm_through_pipeline(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = False
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.is_game_route_active",
        lambda _name: True,
    )
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.get_active_game_route_identity",
        lambda _name: ("game", "session-a"),
    )
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="game",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "qwen"
    route_audio = AsyncMock(return_value=True)
    runtime._route_microphone_audio = route_audio
    evidence = RnnoiseEvidence(True, 3, 0.9, 0.6, 0.2, 0.55)
    processed = ProcessedVoiceFrame(
        pcm16=b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        speech_probability=0.8,
        rnnoise_available=True,
        rnnoise_evidence=evidence,
    )
    runtime._voice_input_audio_pipeline.process = AsyncMock(return_value=processed)
    token = runtime._capture_ingress_token()

    await runtime._process_microphone_stream_data(
        {
            "input_type": "audio",
            "sample_rate_hz": 16_000,
            "data": [1] * 160,
        },
        ingress_token=token,
        captured_at=1234.5,
    )

    runtime._voice_input_audio_pipeline.process.assert_awaited_once()
    route_audio.assert_awaited_once_with(
        processed.pcm16,
        sample_rate_hz=processed.sample_rate_hz,
        speech_probability=processed.speech_probability,
        rnnoise_available=processed.rnnoise_available,
        rnnoise_evidence=evidence,
        ingress_token=token,
        received_at=ANY,
        captured_at=1234.5,
    )

async def test_game_consumer_submit_preserves_owner_identity(monkeypatch) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.is_game_route_active",
        lambda _name: True,
    )
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="game",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "qwen"
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
    )
    token = runtime._capture_ingress_token()
    evidence = RnnoiseEvidence(True, 3, 0.9, 0.6, 0.2, 0.55)
    processed = ProcessedVoiceFrame(
        pcm16=b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        speech_probability=0.8,
        rnnoise_available=True,
        rnnoise_evidence=evidence,
    )

    await runtime._route_microphone_audio(
        processed.pcm16,
        sample_rate_hz=processed.sample_rate_hz,
        speech_probability=processed.speech_probability,
        rnnoise_available=processed.rnnoise_available,
        rnnoise_evidence=evidence,
        ingress_token=token,
    )

    runtime._asr_runtime.submit.assert_awaited_once_with(
        processed,
        ingress_token=token,
    )

async def test_fresh_blocked_route_consumes_pcm_without_omni() -> None:
    runtime = _Runtime()

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert consumed is True
    assert runtime._asr_audio_bytes == 0
    assert runtime._omni_mic_audio_bytes == 0

async def test_voice_session_activation_gates_independent_asr_before_submit() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
    )
    factory = _CoreActivationFactory(similarity=0.1)
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
    )

    pcm16 = b"\xd0\x07" * 1_600
    await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    await asyncio.sleep(0)
    for _ in range(14):
        await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    for _ in range(20):
        await asyncio.sleep(0)
    runtime._asr_runtime.submit.assert_not_awaited()
    assert factory.scorers[0].calls == 1
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

async def test_voice_pcm_invalidation_retires_activation_without_another_frame() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    factory = _CoreActivationFactory()
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
    )
    await runtime._route_microphone_audio(
        b"\xd0\x07" * 1_600,
        sample_rate_hz=16_000,
    )
    await asyncio.sleep(0)
    assert runtime._voice_session_activation_runtime is factory.runtimes[0]

    runtime._invalidate_voice_pcm_sync("microphone_stopped")
    cleanup = tuple(runtime._core_asr_cleanup_tasks)
    if cleanup:
        await asyncio.gather(*cleanup)

    assert runtime._voice_session_activation_runtime is None
    assert factory.scorers[0].closed is True
    assert factory.closed is False

async def test_session_activation_detach_failure_blocks_microphone_pcm() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    runtime._speaker_shadow_factory = MagicMock()
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(return_value=False)

    result = await runtime.set_voice_session_activation_factory(
        _CoreActivationFactory(),
        activation_generation="profile",
    )
    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert result is VoiceIdentityActivationResult.RUNTIME_DEGRADED
    assert consumed is True
    assert runtime._speaker_shadow_factory is not None
    assert runtime._voice_session_activation_factory is None
    assert runtime._voice_session_activation_degraded is True
    runtime.session.stream_audio.assert_not_awaited()

async def test_cancelled_session_activation_detach_blocks_microphone_pcm() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    runtime._speaker_shadow_factory = MagicMock()
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(
        side_effect=asyncio.CancelledError
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.set_voice_session_activation_factory(
            _CoreActivationFactory(),
            activation_generation="profile",
        )
    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert runtime._speaker_shadow_factory is not None
    assert runtime._voice_session_activation_factory is None
    assert runtime._voice_session_activation_degraded is True
    runtime.session.stream_audio.assert_not_awaited()

async def test_inflight_session_activation_detach_blocks_microphone_pcm() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    runtime._speaker_shadow_factory = MagicMock()
    detach_entered = asyncio.Event()
    release_detach = asyncio.Event()

    async def delayed_detach(*_args, **_kwargs) -> bool:
        detach_entered.set()
        await release_detach.wait()
        return False

    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(
        side_effect=delayed_detach
    )
    transition = asyncio.create_task(
        runtime.set_voice_session_activation_factory(
            _CoreActivationFactory(),
            activation_generation="profile",
        )
    )
    await detach_entered.wait()

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert runtime._voice_session_activation_degraded is True
    runtime.session.stream_audio.assert_not_awaited()
    release_detach.set()
    assert await transition is VoiceIdentityActivationResult.RUNTIME_DEGRADED

async def test_turn_endpoint_seals_immediately_before_provider_final() -> None:
    runtime = _Runtime()
    runtime._asr_session = type("Asr", (), {"is_ready": True})()
    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )

    await runtime._handle_independent_asr_endpoint(epoch)

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING

async def test_provider_final_watchdog_blocks_only_independent_asr() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"is_ready": True, "close": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    policy = replace(
        resolve_provider_policy("qwen", "manual"),
        provider_final_timeout_ms=10,
    )
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _ReadyDetector()

    await _start_and_seal_turn(runtime)
    # 10ms 超时跟 Windows 的 15.6ms 定时器分辨率同量级，固定 sleep 到底睡多久
    # 完全看运气；守护任务跑完才是「已开火」的权威信号，直接等它。
    watchdog = runtime._asr_final_watchdog_task
    assert watchdog is not None
    await asyncio.wait_for(watchdog, 5)

    assert runtime._asr_route_mode == "blocked"
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    assert runtime._omni_mic_audio_bytes == 0

async def test_provider_final_watchdog_honors_per_provider_policy_timeout() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"is_ready": True, "close": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_provider = "glm"
    runtime._asr_route_mode = "independent"
    # Segmented providers resolve a longer final timeout than the streaming
    # default; scale both down so the watchdog must track the policy value.
    policy = replace(
        resolve_provider_policy("glm", "manual"),
        # 500ms 而不是 80ms：下面「还没到点」那半只能靠时间证明，被测窗口必须
        # 远大于 Windows 的 15.6ms 定时器分辨率，否则余量不到一个 tick。
        provider_final_timeout_ms=500,
    )
    assert resolve_provider_policy("glm", "manual").provider_final_timeout_ms == 40_000
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _ReadyDetector()

    await _start_and_seal_turn(runtime, "glm")
    armed_at = time.monotonic()
    watchdog = runtime._asr_final_watchdog_task
    assert watchdog is not None

    # 单个 sleep 睡了多久在 Windows 上不可信（15.6ms 分辨率，既会提前弹出也会
    # 超发），所以只信真实时钟：轮询到确实过了 150ms —— 远超一个 tick，也远小于
    # 上面的 500ms 窗口。醒来后必须先复查挂钟再断言：这一觉可能被别的任务拖长而
    # 睡过了观察窗口（甚至睡过 500ms 守护窗口），那时守护任务改成 blocked 是合法的，
    # 先断言就会把「事后才发生」误报成「窗口内提前开火」。
    # A watchdog stuck on the shared default (10 ms in the scaled test above)
    # would have fired by now; the per-provider override keeps it armed.
    deadline = armed_at + 0.15
    while True:
        await asyncio.sleep(0.005)
        if time.monotonic() >= deadline:
            break
        assert runtime._asr_route_mode == "independent"

    # 正向那半不猜时间：守护任务自己跑完（内部 await 完错误处理才结束）即同步点。
    await asyncio.wait_for(watchdog, 5)
    elapsed = time.monotonic() - armed_at

    assert runtime._asr_route_mode == "blocked"
    # 上界也必须钉住，否则这条用例只主张「五秒内会开火」：把 per-provider 超时写成
    # 常量 2s、或者把 ms 当成 s 换算错的回归，在 150ms 观察窗口里同样还是
    # independent，然后在 5s 内跑完，照样通过。配置是 500ms，给 3 倍余量。
    assert elapsed < 1.5, (
        f"守护任务没有按 per-provider 的 500ms 超时开火，实际 {elapsed:.3f}s"
    )

async def test_optimization_disabled_streaming_uploads_without_smart_turn() -> None:
    runtime = _Runtime()
    runtime._voice_input_resource_optimization_enabled = False
    asr = type("Asr", (), {"is_ready": True, "stream_audio": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "provider"),
        shadow_mode=False,
        resource_optimization_enabled=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _ReadyDetector()

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        rnnoise_available=False,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_smart_turn_lease is None
    assert runtime._asr_detector._token is None
    assert runtime._omni_mic_audio_bytes == 0

async def test_active_onset_before_delayed_provider_final_starts_next_turn() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    assert runtime._asr_turn_prepared is True

    # Provider VAD already ended turn 1, but its ordered endpoint callback is
    # delivered only right before the delayed final. The local detector sees
    # the next turn's onset while Core is still ACTIVE and prepared.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )

    # Turn 1's ordered callbacks arrive: endpoint immediately before final.
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # The remembered onset was replayed: turn 2 is ACTIVE and prepared.
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_turn_prepared is True

    # Turn 2's ordered callbacks now seal and deliver instead of no-oping.
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second"]
    assert runtime.handle_new_message.await_count == 2

async def test_submit_without_lifecycle_returns_typed_unavailable() -> None:
    runtime = _Runtime()
    result = await runtime._asr_runtime.submit(
        ProcessedVoiceFrame(b"\x01\x00" * 160, 16_000, 0.0, False),
        ingress_token=VoiceIngressToken(0, "socket", 0, 0, 0),
    )

    assert result == AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
    assert not isinstance(result, bool)

async def test_submit_has_only_typed_top_level_return_paths() -> None:
    source = textwrap.dedent(inspect.getsource(IndependentAsrRuntime.submit))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.AsyncFunctionDef)
    returns: list[ast.Return] = []

    def collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Return):
                returns.append(child)
            else:
                collect(child)

    for statement in function.body:
        collect(statement)

    assert returns
    assert all(
        isinstance(return_node.value, ast.Call)
        and isinstance(return_node.value.func, ast.Name)
        and return_node.value.func.id == "AsrSubmitResult"
        for return_node in returns
    )

async def test_hard_mute_during_detector_await_invalidates_inflight_pcm() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = "qwen"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)

    feed_started = asyncio.Event()
    release_feed = asyncio.Event()

    class _BlockingDetector(_ReadyDetector):
        async def feed(self, _pcm16: bytes, **_kwargs) -> DetectorFeedResult:
            feed_started.set()
            await release_feed.wait()
            return DetectorFeedResult((), True)

    runtime._asr_detector = _BlockingDetector()
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )

    route_task = asyncio.create_task(
        runtime._route_microphone_audio(
            b"\x01\x00" * 160,
            sample_rate_hz=16_000,
        )
    )
    await asyncio.wait_for(feed_started.wait(), 1)
    await runtime._handle_voice_input_control(
        "lease_sync",
        1,
        owner="core",
        hard_muted=True,
        focus_suppressed=False,
    )
    release_feed.set()

    assert await route_task is True
    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_audio_bytes == 0
    assert runtime._omni_mic_audio_bytes == 0

async def test_final_does_not_double_count_sampled_streaming_wire_audio() -> None:
    runtime = _Runtime()
    session = SimpleNamespace(is_ready=True, provider_wire_audio_ms=480)
    runtime._asr_session = session
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "qwen")
    # Streaming sessions advance the counter inside stream_audio, so the
    # per-chunk dispatcher sample has already recorded the full amount.
    runtime._sync_provider_wire_metrics(session)
    assert runtime._asr_lifecycle.metrics.provider_wire_audio_ms == 480

    await runtime._handle_independent_asr_final("hello", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    metrics = runtime._asr_lifecycle.metrics
    assert metrics.provider_wire_audio_ms == 480
    assert metrics.cloud_audio_ms == 480
    assert runtime._asr_last_provider_wire_audio_ms == 480

async def test_asr_stream_failure_never_replays_the_failed_frame_to_omni() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock(side_effect=RuntimeError("sensitive provider body"))
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    await _install_active_smart_turn(runtime, "qwen")

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    assert consumed is True
    assert runtime._asr_route_mode == "blocked"
    assert runtime._asr_session is None
    assert "sensitive provider body" not in str(runtime.send_status.await_args)

async def test_websocket_core_submits_one_external_turn_after_local_history() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    runtime.session.submit_external_voice_turn = AsyncMock()
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "qwen")

    await runtime._handle_independent_asr_final(" hello ", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_awaited_once_with(
        "hello",
        is_voice_source=True,
        source="independent_asr",
        metadata={"provider": "qwen"},
        source_game_route_identity=None,
    )
    runtime.session.submit_external_voice_turn.assert_awaited_once()
    call = runtime.session.submit_external_voice_turn.await_args
    assert call.args == ("hello",)
    assert call.kwargs["turn_id"].startswith("asr-")
    runtime.session.create_response.assert_not_awaited()

@pytest.mark.parametrize("route_mode", ["independent", "native"])
async def test_same_core_session_promotion_resyncs_visual_delivery_mode(
    route_mode: str,
) -> None:
    """A promoted session inherits the live route even when provider key is unchanged."""
    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    runtime.input_mode = "audio"
    runtime._asr_route_mode = route_mode
    runtime._independent_asr_route_key = "qwen"
    runtime._start_independent_asr_if_enabled = AsyncMock()
    replacement_session = type("ReplacementOmni", (), {})()
    replacement_session._supports_native_image = True
    replacement_session.set_visual_delivery_mode = MagicMock()
    replacement_session.block_raw_visual_delivery = MagicMock()
    runtime.session = replacement_session

    await runtime._reconcile_independent_asr_after_core_change()

    if route_mode == "independent":
        replacement_session.set_visual_delivery_mode.assert_not_called()
        replacement_session.block_raw_visual_delivery.assert_called_once_with()
    else:
        replacement_session.set_visual_delivery_mode.assert_called_once_with("native")
    runtime._start_independent_asr_if_enabled.assert_not_awaited()

async def test_provider_hot_swap_drops_cached_pcm_from_old_asr_generation(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    bridge = _HotSwapRuntimeStub(start_status=AsrStartStatus.READY)
    object.__setattr__(runtime, "_asr_runtime", bridge)
    runtime.core_api_type = "glm"
    runtime.input_mode = "audio"
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
    runtime.session.stream_audio = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "provider-a"
    runtime._independent_asr_route_key = "gemini"
    old_token = runtime._capture_ingress_token()
    assert runtime.hot_swap_audio_cache.append(
        _HotSwapAudioFrame(
            pcm16=b"\x01\x00" * 160,
            token=old_token,
            audio_stream_epoch=runtime._audio_stream_epoch,
        )
    )
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )

    await runtime._reconcile_independent_asr_after_core_change()

    assert len(runtime.hot_swap_audio_cache) == 1
    assert runtime._asr_route_mode == "independent"
    await runtime._flush_hot_swap_audio_cache()

    assert bridge.submissions == []
    runtime.session.stream_audio.assert_not_awaited()

async def test_failed_provider_hot_swap_blocks_and_discards_cached_pcm(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    bridge = _HotSwapRuntimeStub(start_status=AsrStartStatus.UNAVAILABLE)
    object.__setattr__(runtime, "_asr_runtime", bridge)
    runtime.core_api_type = "glm"
    runtime.input_mode = "audio"
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
    runtime.session.stream_audio = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "provider-a"
    runtime._independent_asr_route_key = "gemini"
    assert runtime.hot_swap_audio_cache.append(
        _HotSwapAudioFrame(
            pcm16=b"\x01\x00" * 160,
            token=runtime._capture_ingress_token(),
            audio_stream_epoch=runtime._audio_stream_epoch,
        )
    )
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )

    await runtime._reconcile_independent_asr_after_core_change()
    assert runtime._asr_route_mode == "blocked"
    await runtime._flush_hot_swap_audio_cache()

    assert bridge.submissions == []
    assert not runtime.hot_swap_audio_cache
    runtime.session.stream_audio.assert_not_awaited()

async def test_current_audio_pipeline_failure_blocks_once_without_pcm() -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = False
    runtime.is_flushing_hot_swap_cache = False
    runtime.session.stream_audio = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    runtime._asr_runtime.abort = AsyncMock()
    runtime._asr_runtime.submit = AsyncMock()
    runtime._voice_input_audio_pipeline.process = AsyncMock(
        side_effect=RuntimeError("soxr failed")
    )
    token = runtime._capture_ingress_token()
    message = {
        "input_type": "audio",
        "sample_rate_hz": 48_000,
        "data": [1] * 480,
    }

    await runtime._process_microphone_stream_data(
        message,
        ingress_token=token,
    )
    await runtime._process_microphone_stream_data(
        message,
        ingress_token=token,
    )

    assert runtime._asr_route_mode == "blocked"
    runtime._asr_runtime.abort.assert_awaited_once_with("audio_preprocessing_failed")
    runtime._asr_runtime.submit.assert_not_awaited()
    runtime.session.stream_audio.assert_not_awaited()
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert statuses == [
        {
            "code": "ASR_AUDIO_PREPROCESSING_FAILED",
            "details": {"provider": "glm", "session_epoch": 0},
        }
    ]

async def test_pipeline_failure_stops_accepting_pcm_at_ingress() -> None:
    """The latch drops frames before the queue, not after the worker dequeues.

    NATIVE route and the fixture's DEFAULT lease state, both load-bearing.
    The independent route reaches `_abort_independent_asr` on the way here and
    that invalidates the voice PCM sync, which closes the lease gate one step
    below the latch; and `_begin_voice_input_connection` also leaves the lease
    in a state that refuses PCM. Either one makes this test pass without ever
    exercising the latch -- the first version of it did exactly that, and the
    mutant survived.

    On the native route with a live lease nothing else stands in the way:
    `_voice_input_accepts_pcm` is lease-only and reads neither the route mode
    nor the latch, so a backpressured status send left the client free to fill
    the bounded queue -- and overflowing it takes the QueueFull path, which
    aborts the run all over again.
    """

    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("native")
    runtime._asr_runtime.abort = AsyncMock()
    status_started = asyncio.Event()
    release_status = asyncio.Event()

    async def backpressured_status(_payload) -> None:
        status_started.set()
        await release_status.wait()

    runtime.send_status = AsyncMock(side_effect=backpressured_status)
    # Keep the worker from draining what the queue accepts, so the depth below
    # measures what ingress ADMITTED rather than what survived a race with it.
    runtime._ensure_audio_stream_worker = lambda: None

    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=runtime._capture_ingress_token(),
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(status_started.wait(), 1)

    # Premise: everything DOWNSTREAM of the latch would still take this PCM.
    # Without this the test can pass for the wrong reason.
    assert runtime._voice_input_accepts_pcm() is True

    for _ in range(4):
        await runtime._enqueue_audio_stream_data(
            {"input_type": "audio", "sample_rate_hz": 48_000, "data": [1] * 480}
        )

    assert runtime._audio_stream_queue.qsize() == 0, (
        "PCM arriving during the failure notice must be dropped at ingress "
        "rather than queued behind it"
    )

    release_status.set()
    await asyncio.wait_for(failure, 1)
    assert runtime._asr_route_mode == "blocked"

async def test_status_delivery_failure_never_breaks_audio_runtime() -> None:
    runtime = _Runtime()
    runtime.send_status.side_effect = RuntimeError("socket closed")
    runtime._set_microphone_route("native")
    runtime.session.stream_audio = AsyncMock()
    identity = runtime._asr_runtime._capture_runtime_identity()

    await runtime._send_asr_status(
        "ASR_INDEPENDENT_READY",
        "glm",
        session_epoch=runtime._asr_session_epoch,
        expected_identity=identity,
    )
    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    runtime.send_status.assert_awaited_once()
    runtime.session.stream_audio.assert_awaited_once()
    assert runtime._voice_input_pipeline_failed is False

@pytest.mark.parametrize(
    "newer_transition",
    [
        "game_takeover",
        "hard_mute",
        "focus_suppress",
        "lease_generation",
        "connection_replacement",
    ],
)
async def test_game_release_resume_only_survives_pcm_gating_transitions(
    newer_transition: str,
) -> None:
    """Ownership loss during the release abort must skip resume; PCM-gating
    transitions (mute/focus/lease bump) must not, because resume has no other
    call site and skipping it leaves the runtime SUSPENDED for the session."""
    runtime = _Runtime()
    runtime._voice_lease_connection_id = "connection"
    runtime._voice_lease_generation = 1
    runtime._voice_lease_owner = "game"
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def abort(reason: str) -> None:
        if reason == "game_release":
            abort_started.set()
            await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=abort)
    runtime._asr_runtime.resume = AsyncMock()
    runtime._asr_runtime.suspend = AsyncMock()
    releasing = asyncio.create_task(
        runtime._apply_voice_lease_state(
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
            reason="game_release",
            force_abort=True,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    if newer_transition == "connection_replacement":
        runtime._begin_voice_input_connection("replacement")
    elif newer_transition == "lease_generation":
        runtime._voice_lease_generation += 1
    elif newer_transition == "game_takeover":
        await runtime._apply_voice_lease_state(
            owner="game",
            hard_muted=False,
            focus_suppressed=False,
            reason="game_takeover",
            force_abort=True,
        )
    elif newer_transition == "hard_mute":
        await runtime._apply_voice_lease_state(
            owner="core",
            hard_muted=True,
            focus_suppressed=False,
            reason="hard_mute",
            force_abort=True,
        )
    else:
        await runtime._apply_voice_lease_state(
            owner="core",
            hard_muted=False,
            focus_suppressed=True,
            reason="focus_suppress",
            force_abort=True,
        )
    release_abort.set()
    await asyncio.wait_for(releasing, 1)

    if newer_transition in {"game_takeover", "connection_replacement"}:
        runtime._asr_runtime.resume.assert_not_awaited()
    else:
        runtime._asr_runtime.resume.assert_awaited_once_with("game_release")

async def test_unsynchronized_pcm_signals_lease_resync_once_per_state() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True

    for _ in range(3):
        await runtime._enqueue_audio_stream_data(_mic_frame())

    resync = _lease_resync_statuses(runtime)
    assert len(resync) == 1
    assert resync[0]["details"]["reason"] == "lease_unsynchronized"
    assert runtime._audio_stream_queue.empty()
    assert runtime._audio_stream_worker_task is None

    assert runtime._begin_voice_input_connection("pet-window") is True
    for _ in range(2):
        await runtime._enqueue_audio_stream_data(_mic_frame())

    assert len(_lease_resync_statuses(runtime)) == 2
    assert runtime._audio_stream_queue.empty()

async def test_voice_control_status_resolves_owner_after_display_delivery() -> None:
    runtime = _Runtime()
    voice_owner = None

    async def deliver_display(_message: str) -> bool:
        nonlocal voice_owner
        voice_owner = object()
        return True

    runtime.send_status = AsyncMock(side_effect=deliver_display)
    runtime._voice_owner_socket = MagicMock(side_effect=lambda: voice_owner)
    runtime._send_to_voice_owner = AsyncMock(side_effect=lambda _payload: voice_owner)

    delivered = await runtime._send_voice_control_status("lease changed")

    assert delivered == (True, True)
    runtime._send_to_voice_owner.assert_awaited_once_with(
        {"type": "status", "message": "lease changed"}
    )

async def test_synchronized_none_owner_pcm_signals_lease_resync() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="none",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )

    for _ in range(2):
        await runtime._enqueue_audio_stream_data(_mic_frame())

    resync = _lease_resync_statuses(runtime)
    assert len(resync) == 1
    assert resync[0]["details"]["reason"] == "owner_none"
    assert runtime._audio_stream_queue.empty()

async def test_hard_muted_pcm_never_signals_lease_resync() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="core",
            hard_muted=True,
            focus_suppressed=False,
        )
        is True
    )

    for _ in range(2):
        await runtime._enqueue_audio_stream_data(_mic_frame())

    assert _lease_resync_statuses(runtime) == []
    assert runtime._audio_stream_queue.empty()

async def test_game_owner_pcm_never_signals_lease_resync() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="game",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )

    for _ in range(2):
        await runtime._enqueue_audio_stream_data(_mic_frame())

    assert _lease_resync_statuses(runtime) == []
    assert runtime._audio_stream_queue.empty()

async def test_failed_lease_release_does_not_skip_accepted_final_delivery() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "glm")
    component = runtime._asr_runtime
    component._asr_lifecycle.provider_policy = replace(
        component._asr_lifecycle.provider_policy,
        warm_transport_ms=60_000,
    )
    epoch = component._asr_session_epoch
    await _start_and_seal_turn(runtime, "glm")
    lease = component._asr_smart_turn_lease
    assert lease is not None

    async def raising_release() -> None:
        raise RuntimeError("release boom")

    lease.release = raising_release

    await runtime._handle_independent_asr_final("hello", epoch, "glm")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert component._asr_smart_turn_lease is None
    assert (
        component._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    )
    runtime.handle_input_transcript.assert_awaited_once_with(
        "hello",
        is_voice_source=True,
        source="independent_asr",
        metadata={"provider": "glm"},
        source_game_route_identity=None,
    )
    assert component._asr_warm_expiry_task is not None
    component._asr_warm_expiry_task.cancel()

async def test_provider_final_preserves_unconfirmed_successor_pcm_as_pre_roll() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    detector.complete_provider_candidate.return_value = True
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    successor_pcm = b"\x02\x00" * 160
    assert runtime._asr_lifecycle.accept_audio(
        successor_pcm,
        sample_rate_hz=16_000,
    ).disposition is AudioDisposition.BUFFER

    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    decision = runtime._asr_lifecycle.accept_audio(
        b"\x03\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert decision.disposition is AudioDisposition.FORWARD_WITH_PRE_ROLL
    assert decision.pre_roll.startswith(successor_pcm)
    detector.complete_provider_candidate.assert_awaited_once()

async def test_provider_final_lock_then_overflow_preserves_accepted_final() -> None:
    runtime = _Runtime()
    asr = type(
        "Asr",
        (),
        {
            "is_ready": True,
            "stream_audio": AsyncMock(),
            "close": AsyncMock(),
        },
    )()
    runtime._asr_session = asr
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    completion_started = asyncio.Event()
    completion_release = asyncio.Event()

    async def complete_provider_candidate(_fence) -> bool:
        completion_started.set()
        await completion_release.wait()
        return False

    detector.complete_provider_candidate.side_effect = complete_provider_candidate
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    runtime._asr_lifecycle.accept_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    ingress_token = runtime._asr_runtime._asr_current_ingress_token
    assert ingress_token is not None

    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("first", epoch, "openai")
    )
    await asyncio.wait_for(completion_started.wait(), 1)
    overflow_task = asyncio.create_task(
        runtime._handle_audio_ingress_backpressure(
            ingress_token,
            observed_state=VoiceLifecycleState.DRAINING,
        )
    )
    await asyncio.sleep(0)
    assert final_task.done() is False
    assert overflow_task.done() is False
    assert runtime._asr_final_lock.locked()
    completion_release.set()
    await asyncio.gather(final_task, overflow_task)
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_awaited_once_with(
        "first",
        is_voice_source=True,
        source="independent_asr",
        metadata={"provider": "openai"},
        source_game_route_identity=None,
    )
    assert runtime._asr_lifecycle.has_pending_turn is False
    assert runtime._asr_sealed_turn_token is None

@pytest.mark.unit
async def test_microphone_route_syncs_provider_neutral_visual_delivery_mode() -> None:
    """Independent ASR must fail closed for raw vision during every route state."""
    runtime = _Runtime()
    runtime.session._supports_native_image = True
    runtime.session.set_visual_delivery_mode = MagicMock()
    runtime.session.block_raw_visual_delivery = MagicMock()
    runtime.session.allow_raw_visual_delivery = MagicMock()

    runtime._set_microphone_route("independent")
    runtime._set_microphone_route("blocked")
    runtime._set_microphone_route("native")

    delivered_modes = [
        getattr(item.args[0], "value", item.args[0])
        for item in runtime.session.set_visual_delivery_mode.call_args_list
    ]
    assert delivered_modes == ["native"]
    assert runtime.session.block_raw_visual_delivery.call_count >= 2
    runtime.session.allow_raw_visual_delivery.assert_called_once_with()

@pytest.mark.unit
async def test_direct_multimodal_final_submits_raw_image_once() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="direct_atomic"
    )
    # 在**调用发生的那一刻**取一次所有权判据的值。它是个活闭包，事后再调时
    # 这一轮早已结束、所有权已释放，所以只能在这里记。
    owned_at_call: list = []

    async def _record_ownership(*_args, **kwargs):
        cb = kwargs.get("visual_still_owned")
        owned_at_call.append(cb() if callable(cb) else None)

    runtime.session.submit_multimodal_turn = AsyncMock(
        side_effect=_record_ownership
    )
    runtime.session.submit_external_voice_turn = AsyncMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    async def validate_frame() -> None:
        await asyncio.sleep(0)
        assert runtime._stage_independent_visual_frame(
            "raw-frame",
            source="screen",
            request_id="screen-1",
            captured_at=record.started_at,
        )

    validation_task = asyncio.create_task(validate_frame())
    assert runtime._track_independent_visual_validation_task(
        validation_task,
        captured_at=record.started_at,
    )

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=token,
            provider="openai",
            text="look here",
        )
    )
    await validation_task

    runtime.session.submit_multimodal_turn.assert_awaited_once_with(
        "look here",
        ("raw-frame",),
        turn_id=turn_id,
        # 帧总线的频道标签，与这批帧一起冻结。会话侧读活状态会在裁剪 / arbiter
        # 排队 / SDK send 那几段 await 里漂到后继发声的通道上。
        source="screen",
        # Gemini 那条路在真正送出之前还有一段压缩 await，所有权判据必须跟着进去。
        visual_still_owned=ANY,
    )
    # 传的是这一轮 record 自己的 source，不是某个字面量碰巧相等。
    assert runtime.session.submit_multimodal_turn.await_args.kwargs["source"] == (
        record.source if hasattr(record, "source") else "screen"
    )
    # 穿进去的必须是活的判据，且在真正调用 provider 的那一刻仍持有所有权。
    assert owned_at_call == [True]
    runtime.session.submit_external_voice_turn.assert_not_awaited()
    assert turn_id not in runtime._core_multimodal_turns

@pytest.mark.unit
async def test_final_superseded_after_freeze_submits_text_without_frames() -> None:
    """Freezing the frames is not the last word; the submit is.

    The record is retained past a successor prepare so this final keeps its
    transcript, which means the route self-check still finds the same record
    object and passes. But the successor now owns the visuals, so the frozen
    frames belong to the newer utterance. The sentence still has to be
    submitted -- as plain text, the ordinary no-image path.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="direct_atomic"
    )
    runtime.session.submit_multimodal_turn = AsyncMock()
    runtime.session.submit_external_voice_turn = AsyncMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    assert runtime._stage_independent_visual_frame(
        "frame-of-the-old-turn",
        source="screen",
        request_id="screen-1",
        captured_at=record.started_at,
    )

    accepted = runtime.handle_input_transcript

    async def accept_then_let_a_successor_start(*args, **kwargs):
        result = await accepted(*args, **kwargs)
        # 冻结之后、提交之前：后继发声 prepare，视觉所有权交出去。
        successor = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(),
            turn_id=token.turn_id + 1,
        )
        runtime._begin_core_multimodal_turn(
            f"asr-{successor.ingress.session_epoch}-{successor.turn_id}",
            successor,
        )
        return result

    runtime.handle_input_transcript = accept_then_let_a_successor_start

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=token,
            provider="openai",
            text="look here",
        )
    )

    assert record.invalidated.is_set()
    runtime.session.submit_multimodal_turn.assert_not_awaited()
    runtime.session.submit_external_voice_turn.assert_awaited_once()
    assert "look here" in runtime.session.submit_external_voice_turn.await_args.args

@pytest.mark.unit
async def test_provider_admission_rejection_submits_the_transcript_as_text() -> None:
    """Losing the provider's admission window must not lose the sentence.

    The arbiter rejects a multimodal ticket once a newer turn has armed its
    pause, and deletes the committed item on the way out -- nothing of this
    request survives provider-side. Propagating that error drops the user's
    whole utterance; the frames are gone but the transcript still has to be
    answered, exactly as when Core detects the supersession itself.
    """
    from main_logic.omni_realtime_client._response_arbiter import (
        ResponseAdmissionRejected,
    )

    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="direct_atomic"
    )
    runtime.session.submit_multimodal_turn = AsyncMock(
        side_effect=ResponseAdmissionRejected(
            "response dispatch admission rejected after commit"
        )
    )
    runtime.session.submit_external_voice_turn = AsyncMock()
    admission_token = runtime._asr_runtime._capture_turn_token(
        runtime._asr_lifecycle
    )
    admission_turn_id = (
        f"asr-{admission_token.ingress.session_epoch}-{admission_token.turn_id}"
    )
    runtime._begin_core_multimodal_turn(admission_turn_id, admission_token)
    admission_record = runtime._core_multimodal_turns[admission_turn_id]
    assert runtime._stage_independent_visual_frame(
        "frame-of-this-turn",
        source="screen",
        request_id="screen-1",
        captured_at=admission_record.started_at,
    )

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=admission_token,
            provider="openai",
            text="这句话不能消失",
        )
    )

    runtime.session.submit_multimodal_turn.assert_awaited_once()
    runtime.session.submit_external_voice_turn.assert_awaited_once()
    assert (
        "这句话不能消失"
        in runtime.session.submit_external_voice_turn.await_args.args
    )

@pytest.mark.unit
async def test_endpoint_cutoff_survives_provider_final_clearing_the_live_field() -> None:
    """PROVIDER_FINAL clears the live timestamp before Core freezes the turn."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=97)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "spoken-frame",
        source="screen",
        request_id="screen-spoken",
        captured_at=record.started_at,
    )

    # 封口 -> provider final：runtime 清掉了 live 字段，lifecycle 也已经离开
    # DRAINING，只剩下不随 final 清除的那个副本。
    sealed_at = record.started_at + 1.0
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = sealed_at
    runtime._asr_lifecycle = SimpleNamespace(
        snapshot=SimpleNamespace(state=VoiceLifecycleState.WARM_IDLE)
    )

    # 端点之后拍的帧在 final 派发期间才校验完。
    runtime._stage_independent_visual_frame(
        "post-endpoint-frame",
        source="screen",
        request_id="screen-post",
        captured_at=sealed_at + 0.5,
    )

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert record.endpoint_at == sealed_at
    assert turn is not None
    assert turn.images == ("spoken-frame",)

@pytest.mark.unit
def test_speech_onset_is_stamped_at_the_transition_not_after_delivery() -> None:
    """The onset stamp must not sit behind an awaited lifecycle notification.

    Two production SPEECH_CONFIRMED paths stamp ``_asr_turn_audio_started_at``
    only after awaiting ``_send_asr_lifecycle_state()``. Visual ownership uses
    the onset as its lower bound, so a stamp taken after that await turns every
    frame captured during delivery into a "not this utterance" frame. The
    invariant is syntactic: the stamp follows the transition with no await in
    between.
    """
    import inspect

    from main_logic.asr_client import lifecycle as asr_lifecycle_module
    from main_logic.asr_client import runtime as asr_runtime_module

    source = inspect.getsource(asr_runtime_module).splitlines()

    # ⚠️ 这个守卫的第一版只扫 runtime.py 里的字面量
    # `lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)`，因此完全看不见
    # lifecycle.py 自己的 `self.transition(...)`（begin_pending_turn 里那一处）——
    # 第五个迁移点就是这么漏掉的，还给了"五处都打点了"的假绿。清单式守卫必须自己
    # 证明清单是全的：先跨模块把所有迁移点数出来，再逐个查。
    lifecycle_source = inspect.getsource(asr_lifecycle_module).splitlines()
    lifecycle_sites = [
        index
        for index, line in enumerate(lifecycle_source)
        if "transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)" in line
    ]
    # lifecycle 侧的迁移点没有 runtime 字段可写，只能要求它的**调用方**补打点。
    for index in lifecycle_sites:
        owner = None
        for back in range(index, -1, -1):
            stripped = lifecycle_source[back].strip()
            if stripped.startswith("def "):
                owner = stripped[4:].split("(")[0]
                break
        assert owner is not None
        callers = [
            i for i, line in enumerate(source) if f"lifecycle.{owner}()" in line
        ]
        assert callers, (
            f"lifecycle.{owner}() performs a SPEECH_CONFIRMED transition but no "
            f"runtime call site was found to stamp the onset"
        )
        for caller in callers:
            window = chr(10).join(source[caller : caller + 12])
            assert "self._asr_turn_onset_at" in window, (
                f"runtime line {caller + 1}: lifecycle.{owner}() transitions to "
                f"SPEECH_CONFIRMED, so its caller must stamp the onset; got: "
                f"{window!r}"
            )
    transition = "lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)"
    stamp = "self._asr_turn_onset_at ="
    sites = [i for i, line in enumerate(source) if transition in line]

    assert sites, "no SPEECH_CONFIRMED transition found"
    for index in sites:
        # 赋值必须**紧接**转换那一行开始（注释和空行不算，它们引入不了 await）。
        # 值本身可以是多行表达式：几条路径都要在"暂存的 onset"和"进函数时刻"之间选。
        first = next(
            offset
            for offset in range(1, 12)
            if source[index + offset].strip()
            and not source[index + offset].strip().startswith("#")
        )
        assert source[index + first].strip().startswith(stamp), (
            f"line {index + 1}: SPEECH_CONFIRMED must start stamping the onset "
            f"before anything else, got: {source[index + first].strip()!r}"
        )

    # 每一条路径的 onset 赋值都必须**优先取暂存的 pending onset**，只有它为空时才
    # 用进函数时刻。session 先未就绪、随后又 ready 时，真实开口时刻就是当初记下的
    # 那个值；就地取时钟会把整段重连等待算成「开口之后」，期间拍的帧全被排除。
    #
    # 规则对所有迁移点一视同仁，因此不再需要"哪条是延迟路径"这种启发式识别 ——
    # 之前那版靠往上扫若干行找条件语句，既会跨函数误标，也挡不住直接分支退化。
    for index in sites:
        begin = next(
            offset
            for offset in range(1, 12)
            if source[index + offset].strip()
            and not source[index + offset].strip().startswith("#")
        )
        statement = []
        depth = 0
        for offset in range(begin, begin + 9):
            line = source[index + offset]
            statement.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                break
        window = chr(10).join(statement)
        assert "self._asr_pending_speech_onset_at" in window, (
            f"line {index + 1}: the onset assignment must prefer the pending "
            f"onset captured before the reconnect, got: {window!r}"
        )

    # detected_at 本身必须在函数里任何 await 之前捕获。
    for index, line in enumerate(source):
        if line.strip() != "detected_at = time.monotonic()":
            continue
        for back in range(index, -1, -1):
            stripped = source[back].strip()
            if stripped.startswith(("async def ", "def ")):
                break
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("await ") and " await " not in stripped, (
                f"line {index + 1}: detected_at must be captured before any await; "
                f"line {back + 1} is {stripped!r}"
            )

    # 暂存的 pending turn onset 也必须用进函数时刻。函数入口已经存了 detected_at
    # （上面那条规则保证它在任何 await 之前），DRAINING 分支再读一次时钟等于把
    # 「进函数 → 走到这一行」之间拍的帧排除在这段发声之外，而这个字段正是后面
    # begin_pending_turn 那处 _asr_turn_onset_at 的来源。
    for index, line in enumerate(source):
        stripped = line.strip()
        if not stripped.startswith("self._asr_pending_turn_onset_at = "):
            continue
        rhs = stripped.split(" = ", 1)[1]
        if rhs == "None":
            continue
        captures_detected_at = False
        for back in range(index, -1, -1):
            # 只在**方法**定义处收边（4 空格缩进）。这些函数里 detected_at 与
            # DRAINING 分支之间隔着 event_is_current / wake_is_current 这类嵌套
            # def，按 "任意 def" 收边会提前停下，规则对这两处直接失效。
            if source[back].startswith(("    def ", "    async def ")):
                break
            if source[back].strip() == "detected_at = time.monotonic()":
                captures_detected_at = True
                break
        if not captures_detected_at:
            continue
        assert rhs == "detected_at", (
            f"line {index + 1}: the pending turn onset must carry the entry "
            f"timestamp its function already captured, got: {rhs!r}"
        )

class _TestSmartTurnLease:
    def __init__(self, token) -> None:
        self.token = token
        self.released = False

    async def release(self) -> None:
        self.released = True

class _ReadyDetector:
    def __init__(self, feed_result: DetectorFeedResult | None = None) -> None:
        self.detector_epoch = 1
        self._token = None
        self._feed_result = feed_result or DetectorFeedResult((), True)
        self.bind_candidate = AsyncMock(return_value=object())
        self.reset = AsyncMock(side_effect=self._reset)
        self.close = AsyncMock()
        self.release_deferred_turn = AsyncMock()
        self.seal_provider_candidate = AsyncMock(
            return_value=ProviderCandidateFence(1, 0, 0)
        )
        self.complete_provider_candidate = AsyncMock(return_value=False)
        self.discard_provider_successor = AsyncMock(return_value=True)
        self.observe_provider_audio = MagicMock()

    async def prepare_endpointing(self, token):
        self._token = token
        return _TestSmartTurnLease(token)

    def endpointing_ready(self, token) -> bool:
        return self._token == token

    async def feed(self, _pcm16: bytes, **_kwargs) -> DetectorFeedResult:
        return self._feed_result

    async def _reset(self) -> None:
        self._token = None

class _FailedSmartTurnDetector(_ReadyDetector):
    async def prepare_endpointing(self, token):
        self._token = None
        return None

    def endpointing_ready(self, token) -> bool:
        return False

class _QueuedSmartTurnDetector(_ReadyDetector):
    def __init__(self) -> None:
        super().__init__()
        self.queued_audio_ms = 0
        self.smart_turn_evaluation_ms = 0
        self.smart_turn_stale_result_count = 0
        self.smart_turn_coalesced_evaluation_count = 0
        self.force_speech_started = AsyncMock(return_value=True)
        self._sequence_no = 0

    async def submit_audio(
        self,
        _pcm16: bytes,
        *,
        ingress_token,
        **_kwargs,
    ) -> DetectorSubmitResult:
        self._sequence_no += 1
        return DetectorSubmitResult(
            status=DetectorSubmitStatus.ACCEPTED,
            throttle_available=False,
            endpointing_available=True,
            identity=DetectorIngressIdentity(
                ingress_token=ingress_token,
                detector_epoch=1,
                sequence_no=self._sequence_no,
            ),
            candidate=DetectorCandidateKey(1, 0),
        )

def _selection(provider_key: str, endpointing_mode: str = "manual"):
    return type(
        "Selection",
        (),
        {
            "provider_key": provider_key,
            "endpointing_mode": endpointing_mode,
            "soniox_region": None,
        },
    )()

async def _start_runtime_with_callback_candidates(
    monkeypatch,
    *,
    candidate_count: int = 2,
):
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    selection = _selection("qwen", "provider")
    selection_ref = selection
    detector = _ReadyDetector()
    callbacks: list[dict[str, object]] = []
    sessions = [
        SimpleNamespace(
            is_ready=True,
            connect=AsyncMock(),
            close=AsyncMock(),
        )
        for _ in range(candidate_count)
    ]

    def create_candidate(_core_type, *, selection: object, **kwargs):
        assert selection is selection_ref
        callbacks.append(kwargs)
        return sessions[len(callbacks) - 1]

    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(
            return_value={
                "independentAsrEnabled": True,
                "voiceInputResourceOptimizationEnabled": False,
            }
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=selection),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        create_candidate,
    )
    monkeypatch.setattr(
        runtime_module,
        "DetectorRuntime",
        MagicMock(return_value=detector),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_session is sessions[0]
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    assert runtime._asr_detector is detector
    assert runtime._asr_route_mode == "independent"
    return runtime, sessions, callbacks, detector

def _install_ready_lifecycle(
    runtime: _Runtime,
    provider: str = "qwen",
) -> None:
    if runtime._asr_session is None:
        runtime._asr_session = type("Asr", (), {"is_ready": True})()
    runtime._asr_provider = provider
    runtime._set_microphone_route("independent")
    endpointing_mode = "provider" if provider == "openai" else "manual"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, endpointing_mode),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _ReadyDetector()
    runtime._asr_runtime._asr_current_ingress_token = runtime._capture_ingress_token()

def _install_replacement_runtime_generation(
    runtime: _Runtime,
    provider: str = "qwen",
):
    component = runtime._asr_runtime
    component._asr_session_epoch += 1
    component._asr_audio_generation += 1
    component._asr_transcript_dispatcher.invalidate_all()
    component._asr_detector_dispatcher.invalidate_all()
    component._asr_audio_dispatcher.abort()
    session = SimpleNamespace(
        is_ready=True,
        close=AsyncMock(),
        signal_user_activity_end=AsyncMock(),
    )
    lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, "manual"),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _QueuedSmartTurnDetector()
    detector.detector_epoch = 1
    component._asr_session = session
    component._asr_provider = provider
    component._asr_lifecycle = lifecycle
    component._asr_detector = detector
    runtime._set_microphone_route("independent")
    component._asr_current_ingress_token = runtime._capture_ingress_token()
    return session, lifecycle, detector

async def _install_active_smart_turn(runtime: _Runtime, provider: str = "qwen") -> None:
    _install_ready_lifecycle(runtime, provider)
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )

async def _start_and_seal_turn(
    runtime: _Runtime,
    provider: str = "qwen",
) -> None:
    if runtime._asr_lifecycle is None:
        _install_ready_lifecycle(runtime, provider)
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )
    await runtime._handle_independent_asr_endpoint(runtime._asr_session_epoch)

class _CoreActivationScorer:
    profile_generation = "profile"
    scorer_generation = 1

    def __init__(self, *, similarity: float = 0.8) -> None:
        self.similarity = similarity
        self.calls = 0
        self.closed = False

    async def prepare(self) -> ActivationScoreStatus:
        return ActivationScoreStatus.READY

    async def score(self, identity, pcm16: bytes, *, sample_rate_hz: int):
        assert pcm16
        assert sample_rate_hz == 16_000
        self.calls += 1
        return ActivationScoreResult(
            identity,
            ActivationScoreStatus.READY,
            self.similarity,
        )

    async def close(self) -> None:
        self.closed = True

class _CoreActivationFactory:
    activation_generation = "profile"

    def __init__(self, *, similarity: float = 0.8) -> None:
        self.similarity = similarity
        self.runtimes: list[VoiceSessionActivationRuntime] = []
        self.scorers: list[_CoreActivationScorer] = []
        self.closed = False

    def create(self, generation, output, *, status_callback=None):
        scorer = _CoreActivationScorer(similarity=self.similarity)
        scorer.profile_generation = self.activation_generation
        runtime = VoiceSessionActivationRuntime(
            generation,
            scorer,  # type: ignore[arg-type]
            output,
            status_callback=status_callback,
        )
        self.scorers.append(scorer)
        self.runtimes.append(runtime)
        return runtime

    def close(self) -> None:
        self.closed = True

async def _wait_for_activation_output(
    call_count,
    *,
    expected_count: int,
) -> None:
    try:
        async with asyncio.timeout(1.0):
            while call_count() < expected_count:
                await asyncio.sleep(0)
    except TimeoutError as exc:
        raise AssertionError(
            "activation replay did not drain: "
            f"expected {expected_count}, received {call_count()}"
        ) from exc

async def _start_bridge_and_capture_builder_call(monkeypatch, runtime):
    import main_logic.asr_client.runtime as runtime_module

    asr = type("Asr", (), {})()
    asr.connect = AsyncMock()
    asr.close = AsyncMock()
    builder = MagicMock(return_value=asr)
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=_selection("gemini")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        builder,
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "independent"
    return builder.call_args.kwargs

def _install_failing_restart_candidates(
    runtime: _Runtime,
    provider: str,
    *,
    failure_count: int,
) -> list[SimpleNamespace]:
    runtime._asr_session = SimpleNamespace(is_ready=False, close=AsyncMock())
    _install_ready_lifecycle(runtime, provider)
    candidates: list[SimpleNamespace] = []

    def build_candidate(_selection):
        candidate = SimpleNamespace(
            is_ready=True,
            connect=AsyncMock(
                side_effect=RuntimeError("private restart connect detail")
            ),
            close=AsyncMock(),
        )
        candidates.append(candidate)
        assert len(candidates) <= failure_count
        return candidate

    runtime._asr_session_factory = MagicMock(side_effect=build_candidate)
    runtime._asr_transport_selection = _selection(provider)
    return candidates

class _HotSwapRuntimeStub:
    def __init__(self, *, start_status: AsrStartStatus) -> None:
        self.session_epoch = 1
        self.audio_generation = 1
        self.active_provider: str | None = "provider-a"
        self.start_status = start_status
        self.submissions: list[tuple[str | None, bytes, object]] = []
        self.abort = AsyncMock()

    def capture_ingress_token(
        self,
        *,
        connection_id: str,
        lease_generation: int,
        route_generation: int,
    ):
        from main_logic.voice_turn.contracts import VoiceIngressToken

        return VoiceIngressToken(
            self.session_epoch,
            connection_id,
            lease_generation,
            route_generation,
            self.audio_generation,
        )

    async def close(self) -> None:
        self.session_epoch += 1
        self.audio_generation += 1
        self.active_provider = None

    async def start(
        self,
        *,
        route_key: str,
        resource_optimization_enabled: bool,
        user_language: str | None = None,
    ) -> AsrStartResult:
        _ = (route_key, resource_optimization_enabled, user_language)
        self.active_provider = (
            "provider-b" if self.start_status is AsrStartStatus.READY else None
        )
        return AsrStartResult(
            self.start_status,
            provider="provider-b",
            session_epoch=self.session_epoch,
        )

    async def submit(self, frame, *, ingress_token) -> AsrSubmitResult:
        self.submissions.append((self.active_provider, frame.pcm16, ingress_token))
        return AsrSubmitResult(AsrSubmitStatus.ACCEPTED)

def _lease_resync_statuses(runtime: _Runtime) -> list[dict]:
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    return [
        status
        for status in statuses
        if status["code"] == "VOICE_INPUT_LEASE_RESYNC_REQUIRED"
    ]

def _mic_frame() -> dict:
    return {"input_type": "audio", "sample_rate_hz": 16_000, "data": [1] * 160}

def _seal_utterance(runtime) -> None:
    runtime._asr_lifecycle = SimpleNamespace(
        snapshot=SimpleNamespace(state=VoiceLifecycleState.DRAINING)
    )
