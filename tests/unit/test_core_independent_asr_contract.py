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

async def test_native_route_installs_future_verifier_but_reports_unsupported() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(return_value=True)
    factory = MagicMock()

    result = await runtime.set_speaker_verifier_factory(
        factory,
        activation_generation="profile-generation",
    )

    assert result is VoiceIdentityActivationResult.UNSUPPORTED_ASR_ROUTE
    assert runtime._speaker_shadow_factory is factory

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

async def test_game_takeover_wins_even_if_provider_clear_fails() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock(side_effect=RuntimeError("provider abort failed"))
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)

    await runtime._suspend_independent_voice_input_for_game()

    assert runtime._asr_lifecycle.snapshot.state.value == "suspended"

async def test_rejected_voice_input_final_is_observable(monkeypatch) -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    event = VoiceTranscriptEvent(
        turn_token=runtime._asr_runtime._capture_turn_token(
            runtime._asr_lifecycle
        ),
        provider="qwen",
        text="hello",
    )
    runtime._voice_input_registry.dispatch_final = AsyncMock(
        return_value=VoiceInputDispatchResult.REJECTED
    )
    debug = MagicMock()
    monkeypatch.setattr(core_asr_runtime_module.logger, "debug", debug)

    await runtime._dispatch_voice_input_final(event)

    debug.assert_called_once()
    assert "voice input final rejected" in debug.call_args.args[0]

async def test_stale_audio_epoch_rejects_processed_rnnoise_evidence() -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = False
    runtime.is_flushing_hot_swap_cache = False
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "qwen"
    evidence = RnnoiseEvidence(True, 3, 0.9, 0.6, 0.2, 0.55)
    processed = ProcessedVoiceFrame(
        pcm16=b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        speech_probability=evidence.peak,
        rnnoise_available=True,
        rnnoise_evidence=evidence,
    )
    runtime._voice_input_audio_pipeline.process = AsyncMock(return_value=processed)
    route_audio = AsyncMock(return_value=True)
    runtime._route_microphone_audio = route_audio

    await runtime._process_microphone_stream_data(
        {
            "input_type": "audio",
            "sample_rate_hz": 16_000,
            "data": [1] * 160,
        },
        ingress_token=runtime._capture_ingress_token(),
        audio_stream_epoch=runtime._audio_stream_epoch + 1,
    )

    runtime._voice_input_audio_pipeline.process.assert_awaited_once()
    route_audio.assert_not_awaited()

async def test_game_owner_without_consumer_remains_fail_closed() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")

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

    assert runtime._voice_input_accepts_pcm() is False
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.SUSPENDED
    assert runtime._omni_mic_audio_bytes == 0

async def test_required_activation_token_rejects_waiting_older_factory() -> None:
    runtime = _Runtime()
    old_factory = _CoreActivationFactory()
    old_token = runtime.voice_session_activation_policy_token()
    await runtime._core_voice_session_swap_lock.acquire()
    replacement = asyncio.create_task(
        runtime.set_voice_session_activation_factory(
            old_factory,
            activation_generation=old_factory.activation_generation,
            activation_required=True,
            expected_policy_revision=old_token,
        )
    )
    await asyncio.sleep(0)

    runtime.require_voice_session_activation(
        activation_generation="new-required-intent",
    )
    runtime._core_voice_session_swap_lock.release()

    assert (
        await replacement
        is VoiceIdentityActivationResult.RUNTIME_DEGRADED
    )
    assert runtime._voice_session_activation_factory is None
    assert runtime._voice_session_activation_required is True

async def test_native_idle_reconnect_survives_activation_authority_replacement() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.is_active = True
    runtime.session.instructions = "stay in character"
    runtime.session._connection_generation = 12
    delivered: list[bytes] = []

    async def stream_audio(pcm16: bytes) -> None:
        delivered.append(pcm16)

    async def reconnect(_instructions: str, *, native_audio: bool) -> None:
        assert native_audio is True
        runtime.session._connection_generation += 1

    runtime.session.stream_audio = AsyncMock(side_effect=stream_audio)
    runtime.session.connect = AsyncMock(side_effect=reconnect)
    runtime.session.close = AsyncMock()
    runtime.message_handler_task = None
    runtime._restart_message_handler_after_session_reconnect = AsyncMock(
        return_value=True
    )

    retired_factory = _CoreActivationFactory()
    await runtime.set_voice_session_activation_factory(
        retired_factory,
        activation_generation="profile",
    )
    await runtime._route_microphone_audio(
        b"\xd0\x07" * 1_600,
        sample_rate_hz=16_000,
    )
    await asyncio.sleep(0)
    retired_generation = retired_factory.runtimes[0].generation
    runtime.session_closed_by_server = True
    runtime._native_activation_idle_reconnect_identity = (retired_generation, 12)

    replacement_factory = _CoreActivationFactory()
    replacement_factory.activation_generation = "replacement-profile"
    await runtime.set_voice_session_activation_factory(
        replacement_factory,
        activation_generation="replacement-profile",
    )
    replacement_generation = runtime._capture_voice_session_activation_generation()
    assert runtime._native_activation_idle_reconnect_identity == (
        replacement_generation,
        12,
    )

    frames = [b"\xd1\x07" * 1_600 for _ in range(15)]
    await runtime._route_microphone_audio(frames[0], sample_rate_hz=16_000)
    await asyncio.sleep(0)
    for frame in frames[1:]:
        await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)
    for _ in range(100):
        if replacement_factory.scorers[0].calls:
            break
        await asyncio.sleep(0)
    assert replacement_factory.scorers[0].calls == 1
    await _wait_for_activation_output(
        lambda: len(delivered),
        expected_count=len(frames),
    )

    runtime.session.connect.assert_awaited_once_with(
        "stay in character",
        native_audio=True,
    )
    assert replacement_factory.scorers[0].calls == 1
    assert delivered == frames
    assert runtime.session_closed_by_server is False
    assert runtime._native_activation_idle_reconnect_identity is None
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

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

async def test_speech_started_prepares_external_voice_turn() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.prepare_external_voice_turn = AsyncMock()

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )

    runtime.session.prepare_external_voice_turn.assert_awaited_once_with(
        turn_id=f"asr-{runtime._asr_session_epoch}-1"
    )
    runtime.handle_new_message.assert_awaited_once_with()

async def test_stale_core_prepare_restores_previous_preview_owner() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()

    async def block_prepare(*, turn_id: str) -> None:
        del turn_id
        prepare_started.set()
        await release_prepare.wait()

    runtime.session.prepare_external_voice_turn = AsyncMock(side_effect=block_prepare)
    runtime.session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    previous_token = replace(token, turn_id=token.turn_id + 100)
    previous_turn_id = (
        f"asr-{previous_token.ingress.session_epoch}-{previous_token.turn_id}"
    )
    runtime._core_asr_preview_turn_id = previous_turn_id
    runtime._core_asr_preview_turn_token = previous_token
    runtime._core_asr_preview_text = "previous partial"

    prepare_task = asyncio.create_task(runtime._prepare_core_voice_turn(token))
    await asyncio.wait_for(prepare_started.wait(), 1)
    runtime._voice_input_transition_generation += 1
    release_prepare.set()

    assert await asyncio.wait_for(prepare_task, 1) is False
    assert runtime._core_asr_preview_turn_id == previous_turn_id
    assert runtime._core_asr_preview_turn_token == previous_token
    assert runtime._core_asr_preview_text == "previous partial"
    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_empty_final_completes_turn_without_core_injection() -> None:
    runtime = _Runtime()
    runtime.session.prepare_external_voice_turn = AsyncMock()
    runtime.session.abandon_external_voice_turn = MagicMock()
    await _start_and_seal_turn(runtime)
    turn_id = runtime.session.prepare_external_voice_turn.await_args.kwargs["turn_id"]

    await runtime._handle_independent_asr_final(
        "",
        runtime._asr_session_epoch,
        "qwen",
    )
    # Teardown racing the queued empty final may win or lose, but both paths
    # terminate the same pinned route. Repeated invalidation and a duplicate
    # provider final must not produce a second cancellation/abandonment.
    runtime._invalidate_voice_pcm_sync("duplicate_after_empty_final")
    runtime._invalidate_voice_pcm_sync("duplicate_after_empty_final")
    await runtime._handle_independent_asr_final(
        "",
        runtime._asr_session_epoch,
        "qwen",
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_lifecycle.metrics.false_wake_count == 1
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    runtime.session.abandon_external_voice_turn.assert_called_once_with(turn_id)
    assert runtime._omni_mic_audio_bytes == 0

async def test_registry_cancelled_prepare_releases_keyed_external_turn_pause() -> (
    None
):
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime.handle_new_message = AsyncMock(side_effect=asyncio.CancelledError)
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)

    with pytest.raises(asyncio.CancelledError):
        await runtime._prepare_voice_input_turn(token)
    await runtime._voice_input_registry.wait_idle()

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_final_swap_barrier_timeout_drops_without_blocking_dispatcher() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime._core_voice_session_swap_barrier_timeout_s = 0.01
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)

    await runtime._core_voice_session_swap_lock.acquire()
    try:
        await asyncio.wait_for(
            runtime._dispatch_core_asr_transcript(
                VoiceTranscriptEvent(
                    turn_token=token,
                    provider="qwen",
                    text="bounded",
                )
            ),
            timeout=0.5,
        )
    finally:
        runtime._core_voice_session_swap_lock.release()

    runtime.session.create_response.assert_not_awaited()

async def test_cancelled_preview_clear_still_releases_keyed_external_turn_pause() -> None:
    runtime = _Runtime()
    session = runtime.session
    session.abandon_external_voice_turn = MagicMock()
    runtime._send_core_asr_preview_clear = AsyncMock(
        side_effect=asyncio.CancelledError
    )
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=7)
    context = CoreChatTurnContext(
        token=token,
        external_turn_id="asr-cancelled-preview",
        session_ref=session,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime._cancel_core_chat_voice_turn(context, "takeover")

    session.abandon_external_voice_turn.assert_called_once_with(
        "asr-cancelled-preview"
    )

async def test_voice_lease_advances_runtime_barrier_before_waiting_for_registry(
    previous_owner: str,
    owner: str,
    reason: str,
    barrier_method: str,
) -> None:
    runtime = _Runtime()
    runtime._voice_lease_owner = previous_owner
    order: list[str] = []
    runtime._invalidate_voice_pcm_sync = MagicMock(
        side_effect=lambda _reason: order.append("invalidate")
    )
    runtime._asr_runtime.suspend = AsyncMock(
        side_effect=lambda _reason: order.append("suspend")
    )
    runtime._asr_runtime.abort = AsyncMock(
        side_effect=lambda _reason: order.append("abort")
    )
    runtime._asr_runtime.resume = AsyncMock()
    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=lambda: order.append("wait_idle")
    )

    await runtime._apply_voice_lease_state(
        owner=owner,
        hard_muted=False,
        focus_suppressed=False,
        reason=reason,
        force_abort=True,
    )

    assert order == ["invalidate", barrier_method, "wait_idle"]

async def test_runtime_close_preserves_manager_lifetime_registry_builtins() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    registry = runtime._voice_input_registry
    core_registration = runtime._core_chat_voice_input_registration
    game_registration = runtime._game_voice_input_registration
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    assert await runtime._prepare_voice_input_turn(token) is True
    runtime._asr_runtime.close = AsyncMock()

    await runtime._close_independent_asr(next_route_mode="blocked")
    runtime._ensure_asr_runtime_state()
    runtime._ensure_asr_runtime_state()

    assert runtime._voice_input_registry is registry
    assert runtime._core_chat_voice_input_registration is core_registration
    assert runtime._game_voice_input_registration is game_registration
    assert core_registration.closed is False
    assert game_registration.closed is False
    assert len(registry._records) == 2

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

async def test_optimization_disabled_provider_route_never_prepares_smart_turn(
    provider: str,
) -> None:
    runtime = _Runtime()
    runtime._voice_input_resource_optimization_enabled = False
    asr = type("Asr", (), {"is_ready": True, "stream_audio": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_provider = provider
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, "provider"),
        shadow_mode=False,
        resource_optimization_enabled=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _ReadyDetector()
    detector.prepare_endpointing = AsyncMock()
    runtime._asr_detector = detector

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        rnnoise_available=False,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once()
    detector.prepare_endpointing.assert_not_awaited()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_smart_turn_lease is None
    assert runtime._omni_mic_audio_bytes == 0

async def test_final_without_observed_pending_preserves_racing_next_onset() -> None:
    runtime = _Runtime()
    await _start_and_seal_turn(runtime, "gemini")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)

    await runtime._handle_independent_asr_final(
        "first",
        runtime._asr_session_epoch,
        "gemini",
    )

    # A next onset may be admitted after final acceptance but before cleanup.
    # Releasing the completed turn preserves that audio; a full reset loses it.
    detector.reset.assert_not_awaited()
    detector.release_deferred_turn.assert_awaited_once_with()

async def test_completed_overlap_before_delayed_final_delivers_both_finals() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    assert runtime._asr_turn_prepared is True

    # Turn 2 both starts and reaches local silence while turn 1 is still
    # ACTIVE and prepared: its provider endpoint and final are queued in the
    # ordered FIFO behind turn 1's delayed final.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )

    # Turn 1's ordered callbacks arrive: endpoint immediately before final.
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # The completed overlap is not replayed yet: only turn 2's own provider
    # endpoint proves a queued turn exists.
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_turn_prepared is False

    # Turn 2's queued endpoint redeems the credit: the turn activates,
    # prepares, and seals so the final right behind it can deliver.
    await runtime._handle_independent_asr_endpoint(epoch)
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second"]
    assert runtime.handle_new_message.await_count == 2
    assert runtime._asr_overlap_completed_turns == 0

async def test_smart_turn_active_resumed_is_not_recorded_for_replay() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # SmartTurn authority orders activity and endpoint through one detector
    # queue, so a mid-turn resume is same-turn speech and must stay a no-op.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("hello", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_turn_prepared is False
    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["hello"]
    assert runtime.handle_new_message.await_count == 1

async def test_initial_ready_transport_also_expires_from_local_listen() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    policy = replace(resolve_provider_policy("qwen", "manual"), warm_transport_ms=1_000)
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        config=VoiceLifecycleConfig(default_warm_transport_ms=0),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)

    runtime._schedule_transport_warm_expiry(
        runtime._asr_session_epoch,
        expected_state=VoiceLifecycleState.LOCAL_LISTEN,
    )
    # default_warm_transport_ms=0 会立刻到期，固定 sleep 赌不起；
    # 到期任务本身就是「已过期并关闭」的同步点。
    expiry = runtime._asr_warm_expiry_task
    assert expiry is not None
    await asyncio.wait_for(expiry, 5)

    assert runtime._asr_session is None
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DEEP_SLEEP
    asr.close.assert_awaited_once_with()

async def test_prewarm_expiry_rechecks_identity_after_detector_reset() -> None:
    runtime = _Runtime()
    original_session = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = original_session
    runtime._asr_route_mode = "independent"
    policy = replace(resolve_provider_policy("openai", "provider"), warm_transport_ms=1_000)
    lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        config=VoiceLifecycleConfig(default_warm_transport_ms=0),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_lifecycle = lifecycle
    reset_started = asyncio.Event()
    reset_release = asyncio.Event()
    detector = _ReadyDetector()

    async def reset() -> None:
        reset_started.set()
        await reset_release.wait()

    detector.reset.side_effect = reset
    runtime._asr_detector = detector
    runtime._schedule_transport_warm_expiry(
        runtime._asr_session_epoch,
        expected_state=VoiceLifecycleState.PREWARMING,
    )
    await asyncio.wait_for(reset_started.wait(), 1)

    successor_session = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = successor_session
    reset_release.set()
    expiry = runtime._asr_warm_expiry_task
    assert expiry is not None
    await asyncio.wait_for(expiry, 1)

    assert lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
    original_session.close.assert_not_awaited()
    successor_session.close.assert_not_awaited()

async def test_smart_turn_fail_open_buffers_until_deep_sleep_transport_reconnects(
    provider: str,
) -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = provider
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.WARM_EXPIRED)
    runtime._asr_detector = _QueuedSmartTurnDetector()
    new_asr = type("Asr", (), {})()
    new_asr.is_ready = True
    connect_started = asyncio.Event()
    connect_release = asyncio.Event()

    async def connect() -> None:
        connect_started.set()
        await connect_release.wait()

    new_asr.connect = AsyncMock(side_effect=connect)
    new_asr.stream_audio = AsyncMock()
    runtime._asr_session_factory = MagicMock(return_value=new_asr)
    runtime._asr_transport_selection = _selection(provider)
    pcm16 = b"\x03\x00" * 160

    await runtime._route_microphone_audio(
        pcm16,
        sample_rate_hz=16_000,
        rnnoise_available=False,
    )
    await asyncio.wait_for(connect_started.wait(), 1)

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
    assert runtime._asr_route_mode == "independent"
    assert runtime._asr_lifecycle.pending_connect_bytes == len(pcm16)
    connect_release.set()
    await runtime._asr_detector_dispatcher.wait_idle()
    await runtime._asr_audio_dispatcher.wait_idle()

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_route_mode == "independent"
    assert runtime._omni_mic_audio_bytes == 0
    new_asr.connect.assert_awaited_once_with()
    new_asr.stream_audio.assert_awaited_once_with(
        pcm16,
        sample_rate_hz=16_000,
    )
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert all(status.get("code") != "ASR_BLOCKED_ENDPOINTING" for status in statuses)

async def test_hard_mute_suppresses_stale_audio_dispatcher_failure() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    runtime._asr_session = asr
    _install_ready_lifecycle(runtime, "qwen")
    runtime._asr_detector = _ReadyDetector()
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    turn_token = VoiceTurnToken(
        ingress=runtime._capture_ingress_token(lifecycle),
        turn_id=lifecycle.snapshot.turn_id,
    )

    assert await runtime._handle_voice_input_control(
        "lease_sync",
        1,
        owner="core",
        hard_muted=True,
        focus_suppressed=False,
    )
    runtime.send_status.reset_mock()
    await runtime._handle_asr_audio_dispatcher_failure(
        turn_token,
        RuntimeError("old provider write failed after hard mute"),
    )

    assert runtime._asr_route_mode == "independent"
    assert runtime._asr_lifecycle is lifecycle
    runtime.send_status.assert_not_awaited()

async def test_explicit_owner_none_cannot_be_overridden_by_legacy_authorization() -> (
    None
):
    runtime = _Runtime()
    runtime._asr_runtime.abort = AsyncMock()

    assert runtime._begin_voice_input_connection("explicit-socket") is True
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
    runtime._asr_runtime.abort.reset_mock()

    assert (
        await runtime._ensure_voice_input_session_authorized("explicit-socket") is True
    )
    assert runtime._voice_lease_generation == 1
    assert runtime._voice_lease_owner == "none"
    assert runtime._voice_input_accepts_pcm() is False
    runtime._asr_runtime.abort.assert_not_awaited()

async def test_accepted_final_is_recorded_and_injected_once() -> None:
    runtime = _Runtime()
    runtime._asr_provider = "glm"
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "glm")

    await asyncio.gather(
        runtime._handle_independent_asr_final(" hello ", epoch, "glm"),
        runtime._handle_independent_asr_final(" hello ", epoch, "glm"),
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_awaited_once_with(
        "hello",
        is_voice_source=True,
        source="independent_asr",
        metadata={"provider": "glm"},
        source_game_route_identity=None,
    )
    runtime.session.create_response.assert_awaited_once_with("hello")

async def test_blocked_core_response_does_not_block_next_asr_turn() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch
    response_started = asyncio.Event()
    release_response = asyncio.Event()

    async def block_response(_text: str) -> None:
        response_started.set()
        await release_response.wait()

    runtime.session.create_response.side_effect = block_response
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "qwen")
    await response_started.wait()

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    release_response.set()
    await runtime._wait_asr_transcript_dispatch_idle()

async def test_consumed_or_suppressed_final_does_not_create_response() -> None:
    runtime = _Runtime()
    runtime.handle_input_transcript.return_value = False
    await _start_and_seal_turn(runtime, "gemini")

    await runtime._handle_independent_asr_final(
        "echo",
        runtime._asr_session_epoch,
        "gemini",
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.session.create_response.assert_not_awaited()

async def test_cancelled_core_close_waiting_for_pipeline_lock_stays_owned() -> None:
    runtime = _Runtime()
    gate = _GateAsyncLock()
    runtime._voice_input_pipeline_transition_lock = gate
    old_pipeline = SimpleNamespace(close=AsyncMock())
    runtime._voice_input_audio_pipeline = old_pipeline
    runtime._independent_asr_provider = "old-provider"
    runtime._independent_asr_route_key = "old-core"
    runtime._voice_input_registry.wait_idle = AsyncMock()
    runtime._asr_runtime.close = AsyncMock()

    closing = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await asyncio.wait_for(gate.requested.wait(), 1)
    close_cleanup = next(
        task
        for task in runtime._core_asr_cleanup_tasks
        if task.get_name() == "core-independent-asr-close"
    )

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert close_cleanup.cancelled() is False
    assert runtime._voice_input_audio_pipeline is old_pipeline
    gate.release.set()
    await asyncio.wait_for(asyncio.shield(close_cleanup), 1)

    assert runtime._voice_input_audio_pipeline is not old_pipeline
    assert runtime._independent_asr_provider is None
    assert runtime._independent_asr_route_key is None
    old_pipeline.close.assert_awaited_once_with()
    runtime._voice_input_registry.wait_idle.assert_awaited_once_with()
    runtime._asr_runtime.close.assert_awaited_once_with()

async def test_stale_start_waiting_for_pipeline_lock_cannot_replace_successor(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime._close_independent_asr = AsyncMock()
    gate = _GateAsyncLock()
    runtime._voice_input_pipeline_transition_lock = gate
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(
            return_value={
                "independentAsrEnabled": False,
                "noiseReductionEnabled": False,
            }
        ),
    )

    starting = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(gate.requested.wait(), 1)
    runtime._begin_asr_route_operation()
    successor_pipeline = SimpleNamespace(
        nr_enabled=True,
        close=AsyncMock(),
    )
    runtime._voice_input_audio_pipeline = successor_pipeline
    gate.release.set()
    await asyncio.wait_for(starting, 1)

    assert runtime._voice_input_audio_pipeline is successor_pipeline
    assert runtime._voice_input_noise_reduction_enabled is True
    successor_pipeline.close.assert_not_awaited()

async def test_close_failure_keeps_the_requested_blocked_route() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.close = AsyncMock(side_effect=RuntimeError("close failed"))
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"

    await runtime._close_independent_asr(next_route_mode="blocked")

    assert runtime._asr_route_mode == "blocked"
    assert not hasattr(runtime._asr_runtime, "_asr_route_mode")
    assert (
        await runtime._route_microphone_audio(b"\x00\x00", sample_rate_hz=16_000)
        is True
    )

async def test_independent_asr_setting_is_persisted_as_a_boolean() -> None:
    assert "independentAsrEnabled" in preferences._ALLOWED_CONVERSATION_SETTINGS
    assert (
        "voiceInputResourceOptimizationEnabled"
        in preferences._ALLOWED_CONVERSATION_SETTINGS
    )
    assert (
        "voice_input_resource_optimization_enabled"
        not in preferences._ALLOWED_CONVERSATION_SETTINGS
    )

async def test_start_without_user_language_builds_session_without_hint(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    assert getattr(runtime, "user_language", None) is None

    kwargs = await _start_bridge_and_capture_builder_call(monkeypatch, runtime)

    assert kwargs["user_language"] is None

async def test_soniox_connect_retries_exhausted_blocks_without_provider_fallback(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.session.stream_audio = AsyncMock()
    soniox_selection = _selection("soniox", "provider")
    forbidden_core_resolver = MagicMock(
        side_effect=AssertionError("Soniox recovery must not resolve another provider")
    )
    sleep = AsyncMock()
    sessions = []
    for attempt in range(3):
        session = type("Soniox", (), {})()
        session.connect = AsyncMock(
            side_effect=RuntimeError(f"private provider detail {attempt}")
        )
        session.close = AsyncMock()
        sessions.append(session)
    built_selections = []

    def create_candidate(_core_type, *, selection, **_kwargs):
        built_selections.append(selection)
        assert selection is soniox_selection
        return sessions[len(built_selections) - 1]

    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=soniox_selection),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_core_follow_selection",
        forbidden_core_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        create_candidate,
    )
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)

    await runtime._start_independent_asr_if_enabled("audio")

    consumed = await runtime._route_microphone_audio(
        b"\x00\x00",
        sample_rate_hz=16_000,
    )
    for session in sessions:
        session.connect.assert_awaited_once_with()
        session.close.assert_awaited_once_with()
    forbidden_core_resolver.assert_not_called()
    assert built_selections == [soniox_selection] * 3
    assert [call.args for call in sleep.await_args_list] == [(0.25,), (0.5,)]
    assert runtime._asr_session is None
    assert runtime._asr_provider is None
    assert runtime._asr_route_mode == "blocked"
    assert consumed is True
    runtime.session.stream_audio.assert_not_awaited()
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert statuses[-1] == {
        "code": "ASR_INDEPENDENT_PROVIDER_UNAVAILABLE",
        "details": {
            "provider": "soniox",
            "session_epoch": runtime._asr_session_epoch,
        },
    }
    assert "private provider detail" not in str(runtime.send_status.await_args_list)

async def test_restart_closes_not_ready_session_before_replacement() -> None:
    runtime = _Runtime()
    events: list[str] = []

    async def close_old() -> None:
        events.append("old.close")

    async def connect_new() -> None:
        events.append("new.connect")

    old_session = SimpleNamespace(
        is_ready=False,
        close=AsyncMock(side_effect=close_old),
    )
    candidate = SimpleNamespace(
        is_ready=True,
        connect=AsyncMock(side_effect=connect_new),
        close=AsyncMock(),
    )
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    runtime._asr_session_factory = MagicMock(return_value=candidate)
    runtime._asr_transport_selection = _selection("qwen")

    await runtime._restart_transport(max_attempts=1)

    assert events == ["old.close", "new.connect"]
    old_session.close.assert_awaited_once_with()
    candidate.connect.assert_awaited_once_with()
    candidate.close.assert_not_awaited()
    assert runtime._asr_session is candidate

async def test_restart_rejects_non_positive_attempt_override() -> None:
    runtime = _Runtime()

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        await runtime._restart_transport(max_attempts=0)

async def test_selection_failure_is_reported_without_escaping_session_start(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(side_effect=ValueError("invalid provider configuration")),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "blocked"
    assert not hasattr(runtime._asr_runtime, "_asr_required")
    assert runtime._asr_session is None
    assert runtime._asr_provider is None
    assert "ASR_INDEPENDENT_FAILED" in runtime.send_status.await_args.args[0]
    assert "invalid provider configuration" not in str(
        runtime.send_status.await_args_list
    )

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

async def test_start_failure_blocks_omni_without_leaking_error(monkeypatch) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "glm"
    asr = type("Asr", (), {})()
    asr.connect = AsyncMock(side_effect=RuntimeError("secret provider response"))
    asr.close = AsyncMock()
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=_selection("glm")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        MagicMock(return_value=asr),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "blocked"
    assert runtime._asr_session is None
    assert "secret provider response" not in str(runtime.send_status.await_args)

async def test_hot_swap_starts_independent_asr_after_core_route_change(
    core_type: str,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = core_type
    runtime.input_mode = "audio"
    runtime._asr_route_mode = "blocked"
    runtime._independent_asr_route_key = "free"
    runtime._start_independent_asr_if_enabled = AsyncMock()

    await runtime._reconcile_independent_asr_after_core_change()

    runtime._start_independent_asr_if_enabled.assert_awaited_once_with(
        "audio",
        preserve_hot_swap_audio=True,
    )

async def test_native_to_blocked_fences_raw_frames_during_route_reconciliation() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("native")
    replacement_session = type("ReplacementOmni", (), {})()
    replacement_session.set_visual_delivery_mode = MagicMock()
    replacement_session.block_raw_visual_delivery = MagicMock()
    runtime.session = replacement_session

    runtime._set_microphone_route("blocked")

    replacement_session.set_visual_delivery_mode.assert_called_once_with("native")
    replacement_session.block_raw_visual_delivery.assert_called_once_with()

async def test_resource_optimization_handshake_false_overrides_persisted_enabled(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(
            return_value={
                "independentAsrEnabled": True,
                "voiceInputResourceOptimizationEnabled": True,
            }
        ),
    )
    start_mock = AsyncMock(
        return_value=AsrStartResult(
            status=AsrStartStatus.FAILED,
            failure_code="ASR_START_STALE",
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    runtime.set_voice_input_resource_optimization_handshake(False)
    await runtime._start_independent_asr_if_enabled("audio")

    assert start_mock.await_args.kwargs["resource_optimization_enabled"] is False
    assert runtime._speaker_shadow_factory is None
    assert "speaker_shadow_factory" not in start_mock.await_args.kwargs

async def test_connect_budget_stops_a_connect_it_cannot_finish(
    monkeypatch,
) -> None:
    # The other half: independent ASR IS wanted, so the decision would connect --
    # and a verdict produced after the frontend's deadline is worse than none,
    # because the client's timeout tears down the session that did start. Leave
    # the route on the blocked placeholder, which is what the caller would have
    # re-acked without re-deciding at all.
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    start_mock = AsyncMock()
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    await runtime._start_independent_asr_if_enabled(
        "audio",
        handshake_override=True,
        connect_budget_seconds=0.0,
    )

    assert runtime._asr_route_mode == "blocked"
    start_mock.assert_not_awaited()

async def test_start_session_handshake_missing_falls_back_to_persisted(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    start_mock = AsyncMock(
        return_value=AsrStartResult(
            status=AsrStartStatus.FAILED,
            failure_code="ASR_START_STALE",
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    # An absent field (forwarded as None by the router) clears any override a
    # previous session left behind, restoring the persisted-setting behavior.
    runtime.set_independent_asr_handshake(True)
    runtime.set_independent_asr_handshake(None)
    await runtime._start_independent_asr_if_enabled("audio")

    start_mock.assert_awaited_once()

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

async def test_backpressured_status_send_does_not_block_pipeline_transitions() -> None:
    """The frontend socket is unbounded; the transition lock must not wait on it.

    ``_fail_closed_voice_route`` writes the failure notice to the voice owner,
    and a throttled or backpressured client can stall that write for as long
    as it likes. Session restart, independent-ASR close and the
    noise-reduction toggle all need the same pipeline transition lock, so
    holding it across the notify phase parked every recovery operation behind
    one unrelated client write.
    """

    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    assert runtime._begin_voice_input_connection("socket-a") is True
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_synchronized = True
    runtime._asr_runtime.abort = AsyncMock()
    status_started = asyncio.Event()
    release_status = asyncio.Event()

    async def backpressured_status(_payload) -> None:
        status_started.set()
        await release_status.wait()

    runtime.send_status = AsyncMock(side_effect=backpressured_status)
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=runtime._capture_ingress_token(),
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(status_started.wait(), 1)

    source_pipeline = runtime._voice_input_audio_pipeline
    # The client is still absorbing the notice. A toggle must not queue behind
    # it: this is the whole point of shrinking the lock.
    assert await asyncio.wait_for(
        runtime.apply_voice_input_noise_reduction(False),
        1,
    ) is True
    assert runtime._voice_input_audio_pipeline is not source_pipeline

    release_status.set()
    await asyncio.wait_for(failure, 1)

    # ...and the failure still finishes fail-closed once the client catches up.
    assert runtime._asr_route_mode == "blocked"
    assert runtime._voice_lease_connection_id == ""
    assert runtime._voice_lease_owner == "none"

async def test_stale_failure_abort_does_not_clear_successor_route_audio() -> None:
    """A restart landing inside the abort keeps its own queued audio.

    `_abort_independent_asr` invalidates the voice PCM sync AFTER awaiting the
    runtime abort. That await is no longer covered by the pipeline transition
    lock, so a session restart can install a newer route inside it -- and the
    old failure would then clear the successor's queued and hot-swap audio and
    drop its microphone input.
    """

    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    assert runtime._begin_voice_input_connection("socket-a") is True
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_synchronized = True
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def blocking_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=blocking_abort)
    invalidated: list[str] = []
    runtime._invalidate_voice_pcm_sync = lambda reason: invalidated.append(reason)

    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=runtime._capture_ingress_token(),
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    # A newer route operation claims the route while the abort is in flight.
    object.__setattr__(
        runtime,
        "_asr_route_operation_generation",
        runtime._asr_route_operation_generation + 1,
    )
    release_abort.set()
    await asyncio.wait_for(failure, 1)

    assert invalidated == [], (
        "the successor route owns the voice PCM sync now; this failure must "
        "not clear it"
    )
    runtime.send_status.assert_not_awaited()

async def test_old_abort_release_cannot_close_replacement_session() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    old_lifecycle = runtime._asr_lifecycle
    old_detector = runtime._asr_detector
    release_started = asyncio.Event()
    release_old_lease = asyncio.Event()

    class BlockingLease:
        async def release(self) -> None:
            release_started.set()
            await release_old_lease.wait()

    runtime._asr_smart_turn_lease = BlockingLease()
    abort_task = asyncio.create_task(runtime._asr_runtime.abort("test_abort"))
    await asyncio.wait_for(release_started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    release_old_lease.set()
    await asyncio.wait_for(abort_task, 1)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert runtime._asr_lifecycle is not old_lifecycle
    assert runtime._asr_detector is not old_detector
    old_session.close.assert_awaited_once_with()
    new_session.close.assert_not_awaited()

async def test_concurrent_smart_turn_readiness_callers_share_installed_lease() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "glm")
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()

    class _Lease:
        def __init__(self, token, detector) -> None:
            self.token = token
            self._detector = detector
            self.released = False

        async def release(self) -> None:
            self.released = True
            self._detector.token = None

    class _BlockingDetector:
        def __init__(self) -> None:
            self.token = None
            self.prepare_calls = 0

        async def prepare_endpointing(self, token):
            self.prepare_calls += 1
            prepare_started.set()
            await release_prepare.wait()
            self.token = token
            return _Lease(token, self)

        def endpointing_ready(self, token) -> bool:
            return self.token == token

    detector = _BlockingDetector()
    runtime._asr_detector = detector
    component = runtime._asr_runtime
    epoch = component._asr_session_epoch
    first = asyncio.create_task(
        component._ensure_smart_turn_ready(lifecycle, epoch)
    )
    await asyncio.wait_for(prepare_started.wait(), 1)
    second_started = asyncio.Event()

    async def ensure_from_speech_caller() -> bool:
        second_started.set()
        return await component._ensure_smart_turn_ready(lifecycle, epoch)

    second = asyncio.create_task(ensure_from_speech_caller())
    await asyncio.wait_for(second_started.wait(), 1)
    release_prepare.set()

    assert await asyncio.wait_for(first, 1) is True
    assert await asyncio.wait_for(second, 1) is True
    assert detector.prepare_calls == 1
    lease = component._asr_smart_turn_lease
    assert lease is not None
    assert lease.released is False
    assert detector.endpointing_ready(lease.token) is True

async def test_close_unwind_cannot_clear_new_generation_owned_fields() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    old_detector = runtime._asr_detector
    assert old_detector is not None
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def close_detector() -> None:
        close_started.set()
        await release_close.wait()

    old_detector.close = AsyncMock(side_effect=close_detector)
    runtime._asr_session_factory = object()
    runtime._asr_transport_selection = object()
    old_transport = asyncio.create_task(asyncio.Event().wait())
    runtime._asr_transport_task = old_transport
    closing = asyncio.create_task(runtime._asr_runtime._close_independent_asr())
    await asyncio.wait_for(close_started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    new_factory = object()
    new_selection = object()
    keep_transport = asyncio.Event()
    new_transport = asyncio.create_task(keep_transport.wait())
    runtime._asr_session_factory = new_factory
    runtime._asr_transport_selection = new_selection
    runtime._asr_transport_task = new_transport
    new_token = runtime._capture_ingress_token()
    runtime._asr_runtime._asr_current_ingress_token = new_token
    new_transcript_dispatcher = runtime._asr_transcript_dispatcher
    new_detector_dispatcher = runtime._asr_detector_dispatcher
    new_audio_dispatcher = runtime._asr_audio_dispatcher
    release_close.set()
    await asyncio.wait_for(closing, 1)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert runtime._asr_current_ingress_token == new_token
    assert runtime._asr_session_factory is new_factory
    assert runtime._asr_transport_selection is new_selection
    assert runtime._asr_transport_task is new_transport
    assert runtime._asr_transcript_dispatcher is new_transcript_dispatcher
    assert runtime._asr_detector_dispatcher is new_detector_dispatcher
    assert runtime._asr_audio_dispatcher is new_audio_dispatcher
    old_detector.close.assert_awaited_once_with()
    old_session.close.assert_awaited_once_with()
    new_session.close.assert_not_awaited()
    keep_transport.set()
    await new_transport

async def test_disabled_or_text_session_never_creates_provider(monkeypatch) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    factory = MagicMock()
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": False}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        factory,
    )

    await runtime._start_independent_asr_if_enabled("audio")
    await runtime._start_independent_asr_if_enabled("text")

    factory.assert_not_called()
    assert runtime._asr_route_mode == "blocked"
    assert not hasattr(runtime._asr_runtime, "_asr_route_mode")

async def test_unknown_core_capability_remains_fail_closed(monkeypatch) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "unknown"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "blocked"
    assert any(
        "ASR_INDEPENDENT_FAILED" in status_call.args[0]
        for status_call in runtime.send_status.await_args_list
    )

async def test_settings_read_failure_blocks_omni(monkeypatch) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(side_effect=RuntimeError("settings unavailable")),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "blocked"
    assert (
        await runtime._route_microphone_audio(b"\x00\x00", sample_rate_hz=16_000)
        is True
    )

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

async def test_current_native_send_failure_still_closes_route_once() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("native")
    runtime.session_closed_by_server = False
    runtime.session.stream_audio = AsyncMock(
        side_effect=RuntimeError("connection closed")
    )

    await runtime._route_microphone_audio(
        b"\x01\x00",
        sample_rate_hz=16_000,
        ingress_token=runtime._capture_native_ingress_token(),
    )

    runtime.session.stream_audio.assert_awaited_once_with(b"\x01\x00")
    assert runtime.session_closed_by_server is True
    assert runtime._omni_mic_audio_bytes == 0

async def test_runtime_failure_from_a_live_route_still_revokes_the_lease() -> None:
    # Codex P2. The chokepoint refactor passed the PRE-transition identity tuple
    # as still_current, but that tuple carries _asr_route_mode (and
    # _microphone_route_generation, inside the ingress token) while the handler
    # sets the route to "blocked" two lines earlier. The predicate was therefore
    # false on ENTRY -- against the handler's own step -- so
    # _fail_closed_voice_route returned before revoking, leaving the recording
    # socket holding a live hardware microphone on a dead route. Only reachable
    # from a LIVE route, which is exactly the real runtime-failure case; an
    # already-blocked route happened to compare equal and masked it.
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._voice_lease_connection_id = "socket-a"
    runtime._voice_input_websocket = object()

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="current-provider",
            session_epoch=runtime._asr_session_epoch,
        )
    )

    assert runtime._asr_route_mode == "blocked"
    assert runtime._voice_lease_connection_id == ""

async def test_settings_result_is_stale_after_connection_replacement(
    monkeypatch,
) -> None:
    settings_started = asyncio.Event()
    release_settings = asyncio.Event()

    async def load_settings(**_kwargs):
        settings_started.set()
        await release_settings.wait()
        return {"independentAsrEnabled": True}

    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    runtime._voice_lease_connection_id = "connection"
    runtime._voice_lease_generation = 0
    runtime._asr_runtime.start = AsyncMock(
        return_value=AsrStartResult(
            AsrStartStatus.READY,
            provider="qwen",
            session_epoch=runtime._asr_session_epoch,
        )
    )
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        load_settings,
    )

    starting = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(settings_started.wait(), 1)
    runtime._begin_voice_input_connection("replacement")
    release_settings.set()
    await asyncio.wait_for(starting, 1)

    runtime._asr_runtime.start.assert_not_awaited()
    assert runtime._asr_route_mode == "blocked"
    assert runtime._independent_asr_provider is None

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

async def test_idle_backpressure_trailing_activity_is_dropped_cleanly(
    monkeypatch,
) -> None:
    runtime, sessions, callbacks, detector = (
        await _start_runtime_with_callback_candidates(
            monkeypatch,
            candidate_count=1,
        )
    )
    component = runtime._asr_runtime
    lifecycle = component._asr_lifecycle
    assert lifecycle is not None
    token = runtime._capture_ingress_token()
    component._asr_current_ingress_token = token
    on_activity = callbacks[0]["on_speech_activity"]
    assert callable(on_activity)

    await component._handle_audio_ingress_backpressure(token)

    # The idle branch bumps the audio generation but keeps the session
    # adopted so genuinely new speech keeps working.
    assert lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    assert component._asr_session is sessions[0]
    assert not component._ingress_token_matches(token)

    # Trailing session-side speech events with the stale generation must be
    # dropped cleanly: without the identity gate the first event corrupts the
    # lifecycle toward ACTIVE and the second raises an uncaught
    # ASR_INGRESS_TOKEN_REQUIRED into the provider adapter.
    await on_activity(SpeechActivityEvent.SPEECH_STARTED)
    await on_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    assert component._asr_turn_prepared is False
    runtime.session.handle_interruption.assert_not_awaited()
    runtime.handle_new_message.assert_not_awaited()
    assert all(
        "ASR_INDEPENDENT_FAILED" not in call.args[0]
        for call in runtime.send_status.await_args_list
    )

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

async def test_start_installs_latest_verifier_published_during_connect(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    selection = _selection("qwen", "provider")
    connect_started = asyncio.Event()
    connect_release = asyncio.Event()

    async def connect() -> None:
        connect_started.set()
        await connect_release.wait()

    session = SimpleNamespace(
        is_ready=True,
        connect=connect,
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        lambda _core_type: selection,
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        lambda _core_type, **_kwargs: session,
    )
    detector_factory = MagicMock(return_value=_ReadyDetector())
    monkeypatch.setattr(runtime_module, "DetectorRuntime", detector_factory)
    stale_shadow = SimpleNamespace(close=AsyncMock())
    current_shadow = SimpleNamespace(close=AsyncMock())
    stale_factory = MagicMock(return_value=stale_shadow)
    current_factory = MagicMock(return_value=current_shadow)

    start_task = asyncio.create_task(
        runtime._asr_runtime.start(
            route_key="qwen",
            resource_optimization_enabled=True,
            speaker_shadow_factory=stale_factory,
        )
    )
    await asyncio.wait_for(connect_started.wait(), 1.0)
    assert await runtime._asr_runtime.set_speaker_verifier_factory(
        current_factory,
        activation_generation="current-profile",
    )
    connect_release.set()
    result = await asyncio.wait_for(start_task, 1.0)

    assert result.status is AsrStartStatus.READY
    stale_factory.assert_not_called()
    current_factory.assert_called_once_with()
    assert detector_factory.call_args.kwargs["speaker_shadow"] is current_shadow

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

async def test_independent_multimodal_turn_never_reuses_prior_turn_frame() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    captured_at = time.monotonic()
    assert runtime._stage_independent_visual_frame(
        "prior-turn-frame",
        source="screen",
        request_id="screen-prior",
        captured_at=captured_at,
    )
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=78)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "new question")

    assert turn is None

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

async def test_route_close_drops_the_staged_visual_caches() -> None:
    """Staged originals belong to the route, not to the process.

    Their only other clearing point is the NEXT turn starting, so an episode
    that ends while screen sharing is on -- with no further utterance -- leaves
    full-size base64 originals pinned on a long-lived character manager, and the
    next episode starts with a buffer already full of the previous one's frames.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    assert runtime._stage_independent_visual_frame(
        "frame-with-no-utterance",
        source="screen",
        request_id="screen-1",
        captured_at=time.monotonic(),
    )
    assert runtime._prerecord_visual_frames
    assert runtime._latest_independent_visual_frame is not None

    await runtime._close_independent_asr(next_route_mode="blocked")

    assert runtime._prerecord_visual_frames == []
    assert runtime._latest_independent_visual_frame is None

async def test_direct_multimodal_failure_reports_status_without_text_fallback() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "openai"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="direct_atomic"
    )
    runtime.session.submit_multimodal_turn = AsyncMock(
        side_effect=RuntimeError("provider rejected image")
    )
    runtime.session.submit_external_voice_turn = AsyncMock()
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "openai")
    record = runtime._active_multimodal_turn_record()
    assert record is not None
    # The frame was captured during speech and validated after the endpoint.
    # Stamping it with the current clock can exclude it from this sealed turn.
    assert runtime._stage_independent_visual_frame(
        "raw-frame",
        source="screen",
        request_id="screen-1",
        captured_at=record.started_at,
    )

    await runtime._handle_independent_asr_final(
        "look here",
        epoch,
        "openai",
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.session.submit_multimodal_turn.assert_awaited_once()
    runtime.session.submit_external_voice_turn.assert_not_awaited()
    status_payloads = [call.args[0] for call in runtime.send_status.await_args_list]
    assert any("ASR_INDEPENDENT_INJECTION_FAILED" in item for item in status_payloads)
    assert "provider rejected image" not in str(status_payloads)

async def test_frames_captured_after_the_endpoint_are_not_folded_in() -> None:
    """Screen state from after the user stopped talking is not this turn."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=92)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "spoken-frame",
        source="screen",
        request_id="screen-spoken",
        captured_at=record.started_at,
    )

    _seal_utterance(runtime)
    runtime._mark_independent_asr_endpoint_if_sealed()
    assert record.endpoint_at is not None
    runtime._stage_independent_visual_frame(
        "post-endpoint-frame",
        source="screen",
        request_id="screen-post",
        captured_at=record.endpoint_at + 0.5,
    )

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("spoken-frame",)

async def test_live_seal_between_onset_and_registration_still_binds() -> None:
    """The live field floors on started_at, not registered_at.

    started_at is rolled back to the speech onset (an overlapping successor can
    even predate the previous turn's seal), so a real window exists between the
    seal and the registration: a very short utterance can be sealed by ASR
    before its record is built. Flooring the live field on registered_at would
    leave such a turn without a cutoff forever, folding everything captured
    after the user stopped talking into this turn.

    Found by mutation: flipping the live branch to registered_at turned nothing
    red in this whole file before this case existed.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    # 把语音起点回拨，制造 started_at < registered_at 的真实窗口。
    onset = time.monotonic() - 0.5
    runtime._asr_turn_onset_at = onset
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=99)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    assert record.started_at < record.registered_at, "夹具没造出那段窗口"

    # 在飞字段：封口发生在开口之后、record 建立之前。
    sealed_at = record.started_at + 0.1
    assert sealed_at < record.registered_at
    runtime._asr_turn_endpointed_at = sealed_at

    runtime._stage_independent_visual_frame(
        "post-seal-frame",
        source="screen",
        request_id="screen-post-seal",
        captured_at=sealed_at + 0.05,
    )

    assert record.endpoint_at == sealed_at

async def test_stale_seal_instant_from_a_previous_turn_is_not_this_turn_cutoff() -> None:
    """A leftover timestamp predates this record and must not seal it early."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    runtime._asr_turn_endpointed_at = time.monotonic() - 30.0
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=96)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "spoken-frame",
        source="screen",
        request_id="screen-spoken",
        captured_at=record.started_at,
    )

    assert record.endpoint_at is None
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("spoken-frame",)

async def test_prerecord_validation_task_is_attached_to_the_onset_record() -> None:
    """A frame task created before the record exists must not be dropped."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset

    gate = asyncio.Event()

    async def pending_validation() -> None:
        await gate.wait()

    task = asyncio.create_task(pending_validation())
    await asyncio.sleep(0)

    # record 还没建出来：这一步在旧实现里等于永久丢弃这个任务。
    assert runtime._track_independent_visual_validation_task(
        task,
        captured_at=onset + 0.01,
    ) is False

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=100)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert task in record.pending_visual_validations
    assert runtime._prerecord_visual_validations == {}

    gate.set()
    await task

async def test_pending_turn_does_not_inherit_the_previous_turn_endpoint() -> None:
    """A turn started while the previous one drained must not be sealed by it.

    ``_asr_turn_onset_at`` survives a normal turn end (only close/abort/error
    clear it), and ``_asr_last_turn_endpointed_at`` is never cleared. If the
    pending-turn activation forgets to re-stamp the onset, Core takes the
    PREVIOUS turn's onset as this record's ``started_at``, the previous seal
    then satisfies ``sealed_at >= started_at``, and every frame captured for
    the new utterance is rejected as post-endpoint — a silent text-only turn.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    previous_onset = time.monotonic() - 2.0
    previous_seal = previous_onset + 1.0
    runtime._asr_turn_onset_at = previous_onset          # 上一轮遗留，没人清
    runtime._asr_turn_endpointed_at = None               # PROVIDER_FINAL 已清
    runtime._asr_last_turn_endpointed_at = previous_seal  # 永不清
    # pending turn 在上一轮排空期间被标记，之后才激活。
    runtime._asr_turn_onset_at = previous_seal + 0.2

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=101)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 截止点是在第一次 staging（或 final 冻结）时才认领的，所以要先喂一帧再判。
    assert runtime._stage_independent_visual_frame(
        "new-utterance-frame",
        source="screen",
        request_id="screen-new",
        captured_at=record.started_at + 0.1,
    )
    assert record.endpoint_at is None, (
        "the previous turn's seal must not become this turn's cutoff"
    )
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "and this one?")

    assert turn is not None
    assert turn.images == ("new-utterance-frame",)

async def test_prerecord_frames_from_a_previous_route_are_not_adopted() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset

    assert runtime._stage_independent_visual_frame(
        "prerecord-frame",
        source="screen",
        request_id="screen-0",
        captured_at=onset + 0.01,
    )
    # 路由换代之后，那一帧不再属于这条链路。
    runtime._voice_input_transition_generation += 1

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=104)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 关键断言打在"有没有被并进 record"上。只断言 snapshot 为 None 是不够的 ——
    # accepts() 在冻结时还会按 route_generation 再过滤一次，采纳环节即使漏判也照样
    # 返回 None，那样这条用例就是假绿（实测：去掉采纳侧的 route 过滤仍然通过）。
    assert record.last_frame is None
    assert record.first_frame is None
    assert runtime._snapshot_core_multimodal_turn(turn_id, "lost") is None

async def test_prerecord_buffer_trims_in_capture_order_not_arrival_order() -> None:
    """Concurrent validation means arrival order is not capture order.

    The cap evicts the most redundant INTERIOR point and keeps both ends. If the
    buffer is held in arrival order, those "ends" are not the temporal first and
    last, so the eviction can drop the actual start of the utterance — the same
    trap already fixed once for the middle-frame candidates.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset

    # 落地顺序把两端交替喂进来：0, 19, 1, 18, 2, 17, ...
    capture_order = [i if i % 2 == 0 else 19 - i for i in range(20)]
    for generation, index in enumerate(capture_order):
        runtime._stage_independent_visual_frame(
            f"f{index}",
            source="screen",
            request_id=f"screen-{generation}",
            captured_at=onset + 0.001 * (index + 1),
        )

    kept = [frame.image_b64 for frame in runtime._prerecord_visual_frames]
    assert len(kept) <= 8
    # 时间上的首尾必须活着，而不是"最先/最后落地的那两帧"。
    assert kept[0] == "f0"
    assert kept[-1] == f"f{max(capture_order)}"
    captured = [frame.captured_at for frame in runtime._prerecord_visual_frames]
    assert captured == sorted(captured)

async def test_successor_prepares_do_not_evict_a_still_running_final() -> None:
    """A record is removed by its own dispatch, never by a successor's prepare.

    An accepted final can sit inside handle_input_transcript for a while (bounded
    visual-validation join, provider submit). Meanwhile provider VAD can prepare
    several successor utterances. Evicting the oldest record to make room drops
    the identity that in-flight final needs, so the user's whole sentence is
    neither stored nor submitted.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    running = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=401)
    running_id = f"asr-{running.ingress.session_epoch}-{running.turn_id}"
    runtime._begin_core_multimodal_turn(running_id, running)
    running_record = runtime._core_multimodal_turns[running_id]

    for turn_id in (402, 403, 404):
        token = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(), turn_id=turn_id
        )
        runtime._begin_core_multimodal_turn(
            f"asr-{token.ingress.session_epoch}-{token.turn_id}", token
        )

    assert runtime._core_multimodal_turns.get(running_id) is running_record

    # 它自己的 dispatch 收尾时才该消失。
    runtime._abandon_core_voice_turn(running_id, session_ref=None)
    assert running_id not in runtime._core_multimodal_turns

async def test_validation_tracking_picks_the_active_record_not_a_retained_one() -> None:
    """Retained records exist only so an in-flight final keeps its transcript.

    They are invalidated; the active turn is the newest live one. Selecting
    "whichever record happens to be first" binds new frame validations to a
    superseded turn.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    first = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=301)
    first_id = f"asr-{first.ingress.session_epoch}-{first.turn_id}"
    runtime._begin_core_multimodal_turn(first_id, first)
    first_record = runtime._core_multimodal_turns[first_id]

    second = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=302)
    second_id = f"asr-{second.ingress.session_epoch}-{second.turn_id}"
    runtime._begin_core_multimodal_turn(second_id, second)
    second_record = runtime._core_multimodal_turns[second_id]

    gate = asyncio.Event()

    async def pending_validation() -> None:
        await gate.wait()

    task = asyncio.create_task(pending_validation())
    await asyncio.sleep(0)
    assert runtime._track_independent_visual_validation_task(
        task,
        captured_at=second_record.started_at,
    ) is True

    assert task in second_record.pending_visual_validations
    assert task not in first_record.pending_visual_validations

    gate.set()
    await task

async def test_endpoint_marking_skips_invalidated_records() -> None:
    """A successor's seal has no business landing on a superseded record."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    first = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=401)
    first_id = f"asr-{first.ingress.session_epoch}-{first.turn_id}"
    runtime._begin_core_multimodal_turn(first_id, first)
    retained = runtime._core_multimodal_turns[first_id]

    second = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=402)
    second_id = f"asr-{second.ingress.session_epoch}-{second.turn_id}"
    runtime._begin_core_multimodal_turn(second_id, second)
    active = runtime._core_multimodal_turns[second_id]

    runtime._asr_turn_endpointed_at = time.monotonic()
    runtime._mark_independent_asr_endpoint_if_sealed()

    assert active.endpoint_at is not None
    assert retained.endpoint_at is None

async def test_a_stale_onset_does_not_evict_the_prerecord_buffer() -> None:
    """Trimming is only safe against an onset this turn actually owns.

    A leftover value from an older turn would otherwise look like "speech began
    long ago" and evict every frame captured since -- the opposite of what the
    trim exists for. The buffer uses the same trust window as the record, so
    the two agree on where the turn began.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    now = time.monotonic()

    # 不可信的 onset。取**未来**时刻这一侧：那是真正危险的方向 —— 拿它去裁，
    # 每一帧都「早于开口」，整个缓冲会被清空。（过去那一侧的残值只会裁掉比它
    # 更早的帧，多数情况下是空操作，判别不出这条守卫。）
    runtime._asr_runtime._asr_turn_onset_at = now + 30.0

    for i in range(3):
        assert runtime._stage_independent_visual_frame(
            f"frame-{i}",
            source="screen",
            request_id=f"screen-{i}",
            captured_at=now - 1.0 + i * 0.2,
        )

    # onset 不可信 → 不裁，三张都留着（若采信，三张会被全部清掉）。
    assert [f.image_b64 for f in runtime._prerecord_visual_frames] == [
        "frame-0",
        "frame-1",
        "frame-2",
    ]

async def test_direct_overlap_replay_reclaims_its_lent_onset_when_it_never_wakes() -> None:
    """A direct replay that never reaches ACTIVE must take its onset back.

    Both overlap replay paths lend the recorded onset to the confirmation
    branch. The credit-redemption path reclaims it when the wake-up fails; the
    direct path (driven by the delayed provider final) did not, so the stale
    timestamp stayed in the pending slot and the NEXT, unrelated utterance
    adopted it as its visual ownership boundary -- pulling in frames that
    belong to nobody and rejecting the ones it is actually about.

    The carve-out is identical to the credit path: an onset held for a PENDING
    confirmation is deliberately kept, because clearing it would send that
    confirmation back to a fresh detected_at and drop every frame since the
    user actually started speaking. The dual below pins that half.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # A successor spoke while the first turn was still ACTIVE and prepared:
    # its onset is remembered for the direct replay after the delayed final.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    assert runtime._asr_overlap_onset_at is not None
    await runtime._handle_independent_asr_endpoint(epoch)

    # The replay cannot wake the turn, and leaves no pending confirmation
    # behind (Smart Turn lease unavailable / lifecycle broadcast undelivered).
    async def refuse_to_wake(*_args, **_kwargs):
        return None

    runtime._asr_runtime._handle_independent_asr_activity = refuse_to_wake

    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_pending_speech_onset_at is None, (
        "the lent onset stayed behind and a later unrelated turn will adopt it"
    )

async def test_an_unwoken_redemption_never_parks_in_an_untimed_busy_state() -> None:
    """Invariant: a busy state must always carry a timer, whatever the fix is.

    Asserts the absence of the combination "busy state AND both timers None"
    rather than any particular implementation, so it survives a different
    compensation strategy later. HEAD lands squarely in that forbidden
    combination.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch
    component = runtime._asr_runtime

    for event in (
        SpeechActivityEvent.SPEECH_STARTED,
        SpeechActivityEvent.SPEECH_RESUMED,
        SpeechActivityEvent.CANDIDATE_PAUSE,
    ):
        await runtime._handle_independent_asr_activity(event, epoch)
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    component._asr_session.is_ready = False
    await runtime._handle_independent_asr_endpoint(epoch)

    state = (
        runtime._asr_lifecycle.snapshot.state
        if runtime._asr_lifecycle is not None
        else None
    )
    busy = {
        VoiceLifecycleState.PREWARMING,
        VoiceLifecycleState.ACTIVE,
        VoiceLifecycleState.DRAINING,
    }
    assert not (
        state in busy
        and component._asr_warm_expiry_task is None
        and component._asr_final_watchdog_task is None
    ), "忙标志停在了没有任何定时器兜底的状态上"
