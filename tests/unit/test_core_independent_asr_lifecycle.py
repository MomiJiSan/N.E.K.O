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

async def test_external_voice_suppression_reasons_are_independent() -> None:
    runtime = _Runtime()
    runtime._invalidate_voice_pcm_sync = MagicMock()
    runtime._abort_independent_asr = AsyncMock()

    await runtime.set_voice_input_suppressed("enrollment", suppressed=True)
    await runtime.set_voice_input_suppressed("maintenance", suppressed=True)
    await runtime.set_voice_input_suppressed("enrollment", suppressed=False)

    assert runtime._voice_input_accepts_pcm() is False

    await runtime.set_voice_input_suppressed("maintenance", suppressed=False)

    assert runtime._voice_input_accepts_pcm() is True
    assert runtime._abort_independent_asr.await_count == 2

async def test_independent_asr_activity_probe_is_provider_neutral() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")

    assert runtime._independent_asr_user_turn_active() is False

    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._independent_asr_user_turn_active() is True

    runtime._asr_route_mode = "native"
    assert runtime._independent_asr_user_turn_active() is False

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

async def test_provider_endpoint_does_not_wait_for_smart_turn(
    provider: str,
) -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    asr.close = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = provider
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy(provider, "provider"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _FailedSmartTurnDetector(
        DetectorFeedResult((SpeechActivityEvent.SPEECH_STARTED,), True)
    )
    pcm16 = b"\x01\x00" * 160

    assert await runtime._route_microphone_audio(
        pcm16,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once_with(pcm16, sample_rate_hz=16_000)
    assert runtime._asr_route_mode == "independent"
    assert runtime._omni_mic_audio_bytes == 0

async def test_detector_failure_fails_open_to_same_independent_asr() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    await _install_active_smart_turn(runtime)
    runtime._asr_detector.feed = AsyncMock(return_value=DetectorFeedResult((), False))

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()

    asr.stream_audio.assert_awaited_once_with(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    assert runtime._asr_route_mode == "independent"

async def test_game_consumer_ignores_empty_final(monkeypatch) -> None:
    runtime = _Runtime()
    route_transcript = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.is_game_route_active",
        lambda _name: True,
    )
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.get_active_game_route_identity",
        lambda _name: ("game", "session-a"),
    )
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.route_external_voice_transcript",
        route_transcript,
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
    _install_ready_lifecycle(runtime, "qwen")
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    event = VoiceTranscriptEvent(
        turn_token=runtime._asr_runtime._capture_turn_token(lifecycle),
        provider="qwen",
        text="",
    )

    assert await runtime._prepare_voice_input_turn(event.turn_token) is True
    await runtime._dispatch_voice_input_final(event)
    await runtime._voice_input_registry.wait_idle()

    route_transcript.assert_not_awaited()
    runtime.handle_new_message.assert_not_awaited()
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()

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

async def test_game_final_cannot_cross_lease_back_to_core(monkeypatch) -> None:
    runtime = _Runtime()
    route_transcript = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.is_game_route_active",
        lambda _name: True,
    )
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.get_active_game_route_identity",
        lambda _name: ("game", "session-a"),
    )
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.route_external_voice_transcript",
        route_transcript,
    )
    await runtime._handle_voice_input_control(
        "lease_sync",
        1,
        owner="game",
        hard_muted=False,
        focus_suppressed=False,
    )
    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "qwen")

    await runtime._handle_voice_input_control(
        "lease_sync",
        2,
        owner="core",
        hard_muted=False,
        focus_suppressed=False,
    )
    await runtime._handle_independent_asr_final("stale", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    route_transcript.assert_not_awaited()
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    assert runtime._omni_mic_audio_bytes == 0

async def test_native_route_is_sufficient_to_authorize_omni_audio() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert consumed is True
    assert runtime._asr_route_mode == "native"
    runtime.session.stream_audio.assert_awaited_once()
    assert not hasattr(runtime._asr_runtime, "_asr_required")

async def test_native_idle_disconnect_reconnects_before_activation_replay() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.is_active = True
    runtime.session.instructions = "stay in character"
    runtime.session._connection_generation = 4
    delivered: list[bytes] = []
    old_listener_started = asyncio.Event()
    release_old_listener = asyncio.Event()

    async def old_listener() -> None:
        old_listener_started.set()
        await release_old_listener.wait()

    old_listener_task = asyncio.create_task(old_listener())
    await old_listener_started.wait()

    async def stream_audio(pcm16: bytes) -> None:
        delivered.append(pcm16)

    async def reconnect(_instructions: str, *, native_audio: bool) -> None:
        assert native_audio is True
        assert old_listener_task.done()
        runtime.session._connection_generation += 1

    runtime.session.stream_audio = AsyncMock(side_effect=stream_audio)
    runtime.session.connect = AsyncMock(side_effect=reconnect)
    runtime.session.close = AsyncMock()
    runtime.message_handler_task = old_listener_task
    runtime._restart_message_handler_after_session_reconnect = AsyncMock(
        return_value=True
    )
    factory = _CoreActivationFactory()
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
    )

    frames = [
        int(2_000 + sequence).to_bytes(2, "little", signed=True) * 1_600
        for sequence in range(15)
    ]
    await runtime._route_microphone_audio(frames[0], sample_rate_hz=16_000)
    await asyncio.sleep(0)
    generation = factory.runtimes[0].generation
    runtime.session_closed_by_server = True
    runtime._native_activation_idle_reconnect_identity = (generation, 4)

    for frame in frames[1:]:
        await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)

    for _ in range(10):
        await asyncio.sleep(0)
    runtime.session.connect.assert_not_awaited()
    assert delivered == []
    release_old_listener.set()

    await _wait_for_activation_output(
        lambda: len(delivered),
        expected_count=len(frames),
    )

    runtime.session.connect.assert_awaited_once_with(
        "stay in character",
        native_audio=True,
    )
    runtime._restart_message_handler_after_session_reconnect.assert_awaited_once_with(
        runtime.session
    )
    assert runtime.session_closed_by_server is False
    assert runtime._native_activation_idle_reconnect_identity is None
    assert delivered == frames
    assert factory.scorers[0].calls == 1

    live_frame = b"\x01\x00" * 160
    await runtime._route_microphone_audio(live_frame, sample_rate_hz=16_000)
    await asyncio.sleep(0)
    assert delivered == [*frames, live_frame]
    assert runtime.session.connect.await_count == 1
    assert factory.scorers[0].calls == 1
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

async def test_activation_authority_replacement_waits_for_native_reconnect() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.is_active = True
    runtime.session.instructions = "stay in character"
    runtime.session._connection_generation = 24
    connect_entered = asyncio.Event()
    release_connect = asyncio.Event()
    delivered: list[bytes] = []

    async def reconnect(_instructions: str, *, native_audio: bool) -> None:
        assert native_audio is True
        connect_entered.set()
        await release_connect.wait()
        runtime.session._connection_generation += 1

    runtime.session.connect = AsyncMock(side_effect=reconnect)
    runtime.session.stream_audio = AsyncMock(
        side_effect=lambda pcm16: delivered.append(pcm16)
    )
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
    runtime._native_activation_idle_reconnect_identity = (retired_generation, 24)

    reconnect_task = asyncio.create_task(
        runtime._reconnect_native_voice_session_for_activation(
            retired_generation,
            runtime._capture_native_ingress_token(),
        )
    )
    await connect_entered.wait()

    replacement_factory = _CoreActivationFactory()
    replacement_factory.activation_generation = "replacement-profile"
    replacement_task = asyncio.create_task(
        runtime.set_voice_session_activation_factory(
            replacement_factory,
            activation_generation="replacement-profile",
        )
    )
    await asyncio.sleep(0)
    assert replacement_task.done() is False

    release_connect.set()
    assert await reconnect_task is True
    assert await replacement_task is VoiceIdentityActivationResult.READY
    assert runtime.session_closed_by_server is False
    assert runtime._native_activation_idle_reconnect_identity is None

    frames = [b"\xd1\x07" * 1_600 for _ in range(15)]
    await runtime._route_microphone_audio(frames[0], sample_rate_hz=16_000)
    await asyncio.sleep(0)
    for frame in frames[1:]:
        await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)
    await _wait_for_activation_output(
        lambda: len(delivered),
        expected_count=len(frames),
    )

    assert runtime.session.connect.await_count == 1
    assert delivered == frames
    assert replacement_factory.scorers[0].calls == 1
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

async def test_factory_audio_contract_blocks_until_session_pipeline_matches() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    factory = _CoreActivationFactory()
    factory.noise_reduction_enabled = False
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
        activation_required=True,
    )
    frame = b"\x01\x00" * 160

    assert await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)
    runtime.session.stream_audio.assert_not_awaited()
    assert factory.runtimes == []

    runtime._voice_input_noise_reduction_enabled = False
    assert await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)
    assert len(factory.runtimes) == 1

async def test_session_activation_swap_timeout_preserves_legacy_verifier() -> None:
    runtime = _Runtime()
    legacy_factory = MagicMock()
    runtime._speaker_shadow_factory = legacy_factory
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(return_value=True)
    runtime._core_voice_session_swap_barrier_timeout_s = 0.01
    await runtime._core_voice_session_swap_lock.acquire()
    try:
        result = await runtime.set_voice_session_activation_factory(
            _CoreActivationFactory(),
            activation_generation="profile",
        )
    finally:
        runtime._core_voice_session_swap_lock.release()

    assert result is VoiceIdentityActivationResult.RUNTIME_DEGRADED
    assert runtime._speaker_shadow_factory is legacy_factory
    assert runtime._voice_session_activation_factory is None
    runtime._asr_runtime.set_speaker_verifier_factory.assert_not_awaited()

async def test_session_activation_rejects_mismatched_factory_generation() -> None:
    runtime = _Runtime()
    factory = _CoreActivationFactory()

    with pytest.raises(ValueError, match="generation does not match"):
        await runtime.set_voice_session_activation_factory(
            factory,
            activation_generation="stale-profile",
        )
    assert runtime._voice_session_activation_factory is None

async def test_reconnect_listener_replacement_cancels_retired_receive_task() -> None:
    manager = LLMSessionManager.__new__(LLMSessionManager)
    manager.lock = asyncio.Lock()
    manager.is_active = True
    replacement_started = asyncio.Event()

    class Session:
        async def handle_messages(self):
            replacement_started.set()
            await asyncio.Event().wait()

    session = Session()
    manager.session = session
    retired_cancelled = asyncio.Event()

    async def retired_receive_loop():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            retired_cancelled.set()
            raise

    retired_task = asyncio.create_task(retired_receive_loop())
    manager.message_handler_task = retired_task
    await asyncio.sleep(0)

    assert await manager._restart_message_handler_after_session_reconnect(session)
    await asyncio.wait_for(replacement_started.wait(), 1)

    assert retired_cancelled.is_set()
    assert retired_task.done()
    assert manager.message_handler_task is not retired_task
    manager.message_handler_task.cancel()
    await asyncio.gather(manager.message_handler_task, return_exceptions=True)

async def test_rejected_prepare_fails_closed_instead_of_sealing_turn() -> None:
    runtime = _Runtime()
    runtime._asr_session = type(
        "Asr", (), {"is_ready": True, "close": AsyncMock()}
    )()
    _install_ready_lifecycle(runtime, "qwen")
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime.handle_new_message.side_effect = RuntimeError("prepare rejected")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    assert runtime._asr_turn_prepared is False
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE

    await runtime._handle_independent_asr_endpoint(epoch)

    # A persistently rejected preparation must never seal the turn: sealing
    # is the only gate through which a provider final reaches Core, and Core
    # does not re-run the interruption/external-turn pause at dispatch time.
    assert runtime._asr_route_mode == "blocked"
    assert "ASR_CORE_TURN_REJECTED" in str(runtime.send_status.await_args_list)

    await runtime._handle_independent_asr_final("hello", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()

async def test_prepare_failure_releases_keyed_external_turn_pause() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.prepare_external_voice_turn = AsyncMock(
        side_effect=RuntimeError("prepare failed")
    )
    runtime.session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)

    assert await runtime._prepare_core_voice_turn(token) is False

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_pre_dispatch_hot_swap_reprepares_turn_on_promoted_session() -> None:
    """A same-route hot swap transfers the final off the closed old arbiter."""
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    prepared_session = runtime.session
    prepared_session.create_response.side_effect = RuntimeError("closed arbiter")

    replacement = type("Omni", (), {})()
    replacement.create_response = AsyncMock()
    replacement.submit_external_voice_turn = AsyncMock()
    replacement.prepare_external_voice_turn = AsyncMock()
    replacement.abandon_external_voice_turn = MagicMock()

    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    runtime.session = replacement
    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(turn_token=token, provider="qwen", text="prepared"),
        session_ref=prepared_session,
    )

    prepared_session.create_response.assert_not_awaited()
    replacement.prepare_external_voice_turn.assert_awaited_once_with(
        turn_id=f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )
    replacement.submit_external_voice_turn.assert_awaited_once_with(
        "prepared",
        turn_id=f"asr-{token.ingress.session_epoch}-{token.turn_id}",
    )

async def test_final_transcript_is_dropped_when_the_route_leaves_core_mid_restore() -> None:
    # Codex P2, the other half of the case above. Pinning session_ref protects
    # only the SESSION: a game or text takeover landing inside the preview
    # restore's websocket send moves _voice_lease_owner off "core" WITHOUT
    # necessarily replacing self.session, and the transcript was still injected
    # and an ordinary Core response started after the route had left Core.
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    timed_session = runtime.session

    takeover_ran = False

    async def _game_takeover_mid_restore(*_args, **_kwargs) -> None:
        nonlocal takeover_ran
        takeover_ran = True
        runtime._voice_lease_owner = "game"

    runtime._restore_core_asr_preview_after_final = _game_takeover_mid_restore

    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(turn_token=token, provider="qwen", text="hello"),
    )

    # Pin from OUTSIDE the hook that the race was actually manufactured; without
    # this the case degrades into an ordinary final that never left Core.
    assert takeover_ran
    assert runtime._voice_lease_owner == "game"
    # Session identity never moved, so only the route check can catch this.
    assert runtime.session is timed_session
    # No Core response is started for a route that has moved on.
    timed_session.create_response.assert_not_awaited()

async def test_abort_bumps_generation_before_waiting_for_registry_cancel() -> None:
    runtime = _Runtime()
    order: list[str] = []
    runtime._asr_runtime.abort = AsyncMock(
        side_effect=lambda _reason: order.append("abort")
    )
    runtime._invalidate_voice_pcm_sync = MagicMock(
        side_effect=lambda _reason: order.append("invalidate")
    )
    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=lambda: order.append("wait_idle")
    )

    await runtime._abort_independent_asr("ingress_backpressure")

    assert order == ["abort", "invalidate", "wait_idle"]

async def test_current_asr_failure_force_releases_external_turn_pause() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    runtime.session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    assert await runtime._prepare_voice_input_turn(token) is True

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="qwen",
            session_epoch=runtime._asr_session_epoch,
        )
    )

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}",
    )

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

async def test_native_connection_close_is_latched_and_not_retried() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("native")
    runtime.session_closed_by_server = False
    runtime.last_audio_send_error_time = 0.0
    runtime.audio_error_log_interval = 2.0
    runtime.session.stream_audio = AsyncMock(
        side_effect=AttributeError("connection already closed")
    )

    assert (
        await runtime._route_microphone_audio(b"\x01\x00", sample_rate_hz=16_000)
        is True
    )
    assert (
        await runtime._route_microphone_audio(b"\x01\x00", sample_rate_hz=16_000)
        is True
    )

    assert runtime.session_closed_by_server is True
    runtime.session.stream_audio.assert_awaited_once()

async def test_warm_idle_pending_speech_does_not_reenter_draining_guard() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    _install_ready_lifecycle(runtime, "qwen")
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
    await runtime._route_microphone_audio(
        b"\x02\x00" * 160,
        sample_rate_hz=16_000,
    )
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    assert lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert lifecycle.has_pending_turn is True

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )

    assert lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert lifecycle.has_pending_turn is True

async def test_stale_overlap_onset_is_not_replayed_after_final() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )

    # The audio generation moves on before the delayed final, so the recorded
    # onset belongs to a stale ingress and must not wake a replacement turn.
    component = runtime._asr_runtime
    component._asr_audio_generation += 1
    component._asr_current_ingress_token = runtime._capture_ingress_token()

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_turn_prepared is False
    # The prepared Registry route retains the original full VoiceTurnToken.
    # Rotating audio_generation makes the later final a different route, so
    # strict routing drops it together with the stale overlap onset.
    runtime.handle_input_transcript.assert_not_awaited()
    assert runtime.handle_new_message.await_count == 1

async def test_hard_mute_clears_completed_overlap_credit() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    _install_ready_lifecycle(runtime, "openai")
    runtime._clear_audio_stream_queue = MagicMock()
    runtime.hot_swap_audio_cache = []
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    assert runtime._asr_overlap_completed_turns == 1

    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            12,
            owner="core",
            hard_muted=True,
            focus_suppressed=False,
        )
        is True
    )
    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    # Hard mute tears the turn state down: neither the onset nor the credit
    # may survive to wake a replacement turn.
    assert runtime._asr_overlap_onset_token is None
    assert runtime._asr_overlap_completed_token is None
    assert runtime._asr_overlap_completed_turns == 0

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("ghost", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # Turn 1's preparation before the mute awaited handle_new_message once;
    # the muted ghost final must not deliver a transcript or a second turn.
    assert runtime.handle_input_transcript.await_count == 0
    assert runtime.handle_new_message.await_count == 1

async def test_active_ingress_backpressure_releases_keyed_core_turn_without_blocking(
    monkeypatch,
) -> None:
    runtime, sessions, callbacks, detector = (
        await _start_runtime_with_callback_candidates(
            monkeypatch,
            candidate_count=1,
        )
    )
    current_pause_id: str | None = None

    async def prepare_external_voice_turn(*, turn_id: str) -> None:
        nonlocal current_pause_id
        current_pause_id = turn_id

    def abandon_external_voice_turn(turn_id: str | None = None) -> None:
        nonlocal current_pause_id
        if turn_id is not None and turn_id != current_pause_id:
            return
        current_pause_id = None

    runtime.session.prepare_external_voice_turn = AsyncMock(
        side_effect=prepare_external_voice_turn
    )
    runtime.session.abandon_external_voice_turn = MagicMock(
        side_effect=abandon_external_voice_turn
    )
    core_session = runtime.session
    component = runtime._asr_runtime
    lifecycle = component._asr_lifecycle
    assert lifecycle is not None
    current_ingress = runtime._capture_ingress_token()
    component._asr_current_ingress_token = current_ingress
    on_activity = callbacks[0]["on_speech_activity"]
    assert callable(on_activity)

    await on_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    prepared_turn_id = lifecycle.snapshot.turn_id
    runtime.session.prepare_external_voice_turn.assert_awaited_once()
    external_turn_id = (
        runtime.session.prepare_external_voice_turn.await_args.kwargs["turn_id"]
    )
    assert current_pause_id == external_turn_id
    backpressure_status_started = asyncio.Event()
    release_backpressure_status = asyncio.Event()

    async def block_backpressure_status(payload: str) -> None:
        if json.loads(payload).get("code") == "ASR_INGRESS_BACKPRESSURE":
            backpressure_status_started.set()
            await release_backpressure_status.wait()

    runtime.send_status.side_effect = block_backpressure_status
    backpressure_task = asyncio.create_task(
        component._handle_audio_ingress_backpressure(current_ingress)
    )
    await asyncio.wait_for(backpressure_status_started.wait(), 1)

    next_turn = VoiceTurnToken(
        ingress=runtime._capture_ingress_token(),
        turn_id=lifecycle.snapshot.turn_id,
    )
    assert next_turn.turn_id != prepared_turn_id
    assert await runtime._prepare_core_voice_turn(next_turn) is True
    assert runtime.session.prepare_external_voice_turn.await_count == 2
    next_external_turn_id = (
        runtime.session.prepare_external_voice_turn.await_args.kwargs["turn_id"]
    )
    assert next_external_turn_id != external_turn_id
    assert current_pause_id == next_external_turn_id

    release_backpressure_status.set()
    await asyncio.wait_for(backpressure_task, 1)

    assert runtime.session is core_session
    assert runtime._asr_route_mode == "independent"
    assert component._asr_lifecycle is lifecycle
    assert lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    assert component._asr_session is None
    assert component._asr_detector is detector
    assert component._asr_current_ingress_token is None
    sessions[0].close.assert_awaited_once_with()
    detector.reset.assert_awaited_once_with()
    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        external_turn_id
    )
    assert current_pause_id == next_external_turn_id
    assert all(
        "ASR_INDEPENDENT_FAILED" not in call.args[0]
        for call in runtime.send_status.await_args_list
    )

async def test_warm_idle_uses_provider_transport_ttl() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    policy = replace(resolve_provider_policy("openai", "provider"), warm_transport_ms=0)
    lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        config=VoiceLifecycleConfig(default_warm_transport_ms=1_000),
        shadow_mode=False,
    )
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    runtime._asr_lifecycle = lifecycle

    runtime._schedule_transport_warm_expiry(
        runtime._asr_session_epoch,
        expected_state=VoiceLifecycleState.WARM_IDLE,
    )
    expiry = runtime._asr_warm_expiry_task
    assert expiry is not None
    await asyncio.wait_for(expiry, 1)

    assert runtime._asr_session is None
    assert lifecycle.snapshot.state is VoiceLifecycleState.DEEP_SLEEP
    asr.close.assert_awaited_once_with()

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

async def test_hard_mute_is_backend_authoritative_and_rejects_stale_lease_events() -> (
    None
):
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = type("Detector", (), {})()
    detector.reset = AsyncMock()
    detector.feed = AsyncMock(return_value=DetectorFeedResult((), True))
    runtime._asr_detector = detector
    runtime._clear_audio_stream_queue = MagicMock()
    runtime.hot_swap_audio_cache = [b"old-pcm"]
    old_token = runtime._capture_ingress_token(runtime._asr_lifecycle)

    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            12,
            owner="core",
            hard_muted=True,
            focus_suppressed=False,
        )
        is True
    )
    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    asr.close.assert_awaited_once_with()
    runtime._clear_audio_stream_queue.assert_called_once_with("lease_sync")
    assert runtime.hot_swap_audio_cache == []
    assert runtime._ingress_token_matches(old_token) is False
    detector.reset.assert_awaited_once_with()
    detector.feed.assert_not_awaited()
    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_lifecycle.pre_roll_bytes == 0

    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            11,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is False
    )
    assert runtime._voice_input_suppressed is True
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            13,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )
    assert runtime._voice_input_suppressed is False

async def test_new_websocket_connection_resets_mic_lease_generation_once() -> None:
    runtime = _Runtime()
    runtime._voice_lease_generation = 12

    assert runtime._begin_voice_input_connection("socket-a") is True
    assert runtime._voice_lease_generation == -1
    assert runtime._voice_lease_control_seen is False
    assert runtime._voice_input_accepts_pcm() is False
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
    assert runtime._voice_lease_control_seen is True
    assert runtime._voice_input_accepts_pcm() is False

    assert runtime._begin_voice_input_connection("socket-a") is False
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is False
    )

    assert runtime._begin_voice_input_connection("socket-b") is True
    assert runtime._voice_lease_control_seen is False
    assert runtime._voice_input_accepts_pcm() is False
    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )
    assert runtime._voice_input_accepts_pcm() is True

async def test_legacy_authorization_loses_race_to_new_connection_identity() -> None:
    runtime = _Runtime()
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def _block_old_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    old_abort = AsyncMock(side_effect=_block_old_abort)
    runtime._asr_runtime.abort = old_abort
    assert runtime._begin_voice_input_connection("socket-a") is True

    authorize_task = asyncio.create_task(
        runtime._ensure_voice_input_session_authorized("socket-a")
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    assert runtime._begin_voice_input_connection("socket-b") is True
    runtime._asr_runtime.abort = AsyncMock()
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
    release_abort.set()

    assert await authorize_task is False
    assert runtime._voice_lease_connection_id == "socket-b"
    assert runtime._voice_lease_generation == 1
    assert runtime._voice_lease_control_seen is True
    assert runtime._voice_lease_owner == "none"
    assert runtime._voice_input_accepts_pcm() is False
    old_abort.assert_awaited_once_with("legacy_session_start")

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

async def test_late_first_final_then_second_final_recovers_in_linear_order() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch
    events: list[str] = []

    runtime.session.handle_interruption.side_effect = lambda: events.append(
        "interruption"
    )
    runtime.handle_new_message.side_effect = lambda: events.append("prepare")
    runtime.handle_input_transcript.side_effect = lambda text, **_kwargs: (
        events.append(f"transcript:{text}") or True
    )
    runtime.session.create_response.side_effect = lambda text: events.append(
        f"response:{text}"
    )

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first fragment", epoch, "openai")
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("second fragment", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert events.count("interruption") == 2
    assert events.count("prepare") == 2
    assert [event for event in events if event.startswith("transcript:")] == [
        "transcript:first fragment",
        "transcript:second fragment",
    ]
    assert [event for event in events if event.startswith("response:")] == [
        "response:first fragment",
        "response:second fragment",
    ]

async def test_close_releases_independent_audio_pipeline() -> None:
    runtime = _Runtime()
    pipeline = type("Pipeline", (), {})()
    pipeline.close = AsyncMock()
    runtime._voice_input_audio_pipeline = pipeline

    await runtime._close_independent_asr(next_route_mode="blocked")

    pipeline.close.assert_awaited_once_with()
    assert runtime._voice_input_audio_pipeline is not pipeline

async def test_cancelled_successor_close_owns_runtime_cleanup_after_old_close() -> None:
    runtime = _Runtime()
    first_wait_started = asyncio.Event()
    second_wait_started = asyncio.Event()
    release_registry_wait = asyncio.Event()
    wait_calls = 0

    async def block_registry_wait() -> None:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            first_wait_started.set()
        elif wait_calls == 2:
            second_wait_started.set()
        await release_registry_wait.wait()

    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=block_registry_wait
    )
    runtime._asr_runtime.close = AsyncMock()

    retired_close = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await first_wait_started.wait()

    successor_close = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await second_wait_started.wait()
    successor_cleanup = tuple(runtime._core_asr_cleanup_tasks)
    successor_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await successor_close

    release_registry_wait.set()
    await retired_close
    await asyncio.gather(*successor_cleanup)

    runtime._asr_runtime.close.assert_awaited_once_with()

async def test_cancelled_start_settings_swap_keeps_pipeline_cleanup_owned(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def block_pipeline_close() -> None:
        close_started.set()
        await release_close.wait()

    stale_pipeline = SimpleNamespace(
        nr_enabled=True,
        close=AsyncMock(side_effect=block_pipeline_close),
    )
    runtime._voice_input_audio_pipeline = stale_pipeline
    runtime._close_independent_asr = AsyncMock()
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

    starting = asyncio.create_task(
        runtime._start_independent_asr_if_enabled("audio")
    )
    await asyncio.wait_for(close_started.wait(), 1)
    replacement = runtime._voice_input_audio_pipeline
    cleanup = next(
        task
        for task in runtime._core_asr_cleanup_tasks
        if task.get_name() == "core-voice-input-pipeline-close"
    )

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    assert replacement is not stale_pipeline
    assert replacement.nr_enabled is False
    assert cleanup.cancelled() is False
    release_close.set()
    await asyncio.wait_for(cleanup, 1)
    stale_pipeline.close.assert_awaited_once_with()

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

async def test_runtime_builds_primary_candidate_from_its_single_selection(
    monkeypatch,
) -> None:
    import main_logic.asr_client as asr_client
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    selection = asr_client._AsrSelection(
        provider_key="gemini",
        endpointing_mode="manual",
    )
    resolver = MagicMock(return_value=selection)
    asr = type("Asr", (), {})()
    asr.connect = AsyncMock()
    asr.close = AsyncMock()
    builder = MagicMock(return_value=asr)

    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(runtime_module, "_resolve_asr_selection", resolver)
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        builder,
        raising=False,
    )
    assert not hasattr(runtime_module, "create_asr_session")

    await runtime._start_independent_asr_if_enabled("audio")

    resolver.assert_called_once_with("gemini")
    assert builder.call_args.kwargs["selection"] is selection
    asr.connect.assert_awaited_once_with()
    assert runtime._asr_session is asr
    assert runtime._asr_provider == "gemini"
    assert runtime._asr_route_mode == "independent"

async def test_explicit_intl_soniox_is_selected_before_audio(monkeypatch) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    asr = type("Asr", (), {})()
    asr.connect = AsyncMock()
    asr.close = AsyncMock()
    factory = MagicMock(return_value=asr)
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=_selection("soniox", "provider")),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        factory,
    )

    await runtime._start_independent_asr_if_enabled("audio")

    asr.connect.assert_awaited_once_with()
    assert runtime._asr_session is asr
    assert runtime._asr_provider == "soniox"
    assert runtime._asr_received_audio is False

async def test_adopted_start_activity_callback_survives_idle_audio_generation_bump(
    monkeypatch,
) -> None:
    runtime, sessions, callbacks, detector = (
        await _start_runtime_with_callback_candidates(
            monkeypatch,
            candidate_count=1,
        )
    )
    component = runtime._asr_runtime
    original_audio_generation = component._asr_audio_generation
    current_ingress = runtime._capture_ingress_token()
    component._asr_current_ingress_token = current_ingress

    await component._handle_audio_ingress_backpressure(current_ingress)

    assert component._asr_audio_generation == original_audio_generation + 1
    assert component._asr_session is sessions[0]
    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    detector.reset.assert_awaited_once_with()

    updated_ingress = runtime._capture_ingress_token()
    component._asr_current_ingress_token = updated_ingress
    on_activity = callbacks[0]["on_speech_activity"]
    assert callable(on_activity)
    await on_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert component._asr_current_ingress_token == updated_ingress
    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    sessions[0].close.assert_not_awaited()

async def test_restart_default_attempts_follow_soniox_policy_ladder(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    sleep = AsyncMock()
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
    candidates = _install_failing_restart_candidates(
        runtime, "soniox", failure_count=3
    )
    assert runtime._asr_lifecycle.provider_policy.connect_max_attempts == 3

    await runtime._restart_transport()
    while runtime._asr_runtime._asr_close_tasks:
        await asyncio.gather(
            *tuple(runtime._asr_runtime._asr_close_tasks),
            return_exceptions=True,
        )

    assert len(candidates) == 3
    for candidate in candidates:
        candidate.connect.assert_awaited_once_with()
        candidate.close.assert_awaited_once_with()
    assert [call.args for call in sleep.await_args_list] == [(0.25,), (0.5,)]
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert statuses[-1]["code"] == "ASR_INDEPENDENT_FAILED"

async def test_adopted_restart_cancellation_fails_closed_and_propagates(
    monkeypatch,
) -> None:
    runtime, sessions, _callbacks, detector = (
        await _start_runtime_with_callback_candidates(monkeypatch)
    )
    component = runtime._asr_runtime
    lifecycle = component._asr_lifecycle
    assert lifecycle is not None
    started_epoch = component._asr_session_epoch
    on_failure = AsyncMock(side_effect=component._callbacks.on_failure)
    component._callbacks = replace(component._callbacks, on_failure=on_failure)

    await component._close_transport_only()
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    component._asr_pending_speech_confirmed = True
    component._asr_current_ingress_token = runtime._capture_ingress_token()
    prepare_started = asyncio.Event()
    keep_preparing = asyncio.Event()

    async def block_prepare(_epoch: int) -> None:
        prepare_started.set()
        await keep_preparing.wait()

    component._prepare_independent_asr_turn = AsyncMock(side_effect=block_prepare)
    restarting = asyncio.create_task(component._restart_transport(max_attempts=3))
    await asyncio.wait_for(prepare_started.wait(), 1)
    assert component._asr_session is sessions[1]

    # Authoritative cancellation targets the shared owner, not one waiter.
    assert component._asr_transport_task is not None
    component._asr_transport_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restarting
    while component._asr_close_tasks:
        await asyncio.gather(
            *tuple(component._asr_close_tasks),
            return_exceptions=True,
        )

    on_failure.assert_awaited_once()
    failure = on_failure.await_args.args[0]
    assert failure.code == "ASR_INDEPENDENT_FAILED"
    assert failure.session_epoch == started_epoch + 1
    sessions[1].close.assert_awaited_once_with()
    detector.close.assert_awaited_once_with()
    assert component._asr_session is None
    assert component._asr_lifecycle is None
    assert component._asr_detector is None
    assert component._asr_session_factory is None
    assert component._asr_transport_selection is None
    assert component._asr_session_epoch == started_epoch + 1
    assert runtime._asr_route_mode == "blocked"
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert {
        "code": "ASR_INDEPENDENT_FAILED",
        "details": {
            "provider": "qwen",
            "session_epoch": started_epoch + 1,
        },
    } in statuses

async def test_qwen_core_starts_independent_asr_with_external_turn_support(
    monkeypatch,
    core_type: str,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = core_type
    runtime.session.set_visual_delivery_mode = MagicMock()
    runtime.session.block_raw_visual_delivery = MagicMock()
    asr = type("Asr", (), {})()

    async def connect_after_visual_fail_closed() -> None:
        delivered_modes = [
            getattr(call.args[0], "value", call.args[0])
            for call in runtime.session.set_visual_delivery_mode.call_args_list
        ]
        assert "external_description" not in delivered_modes
        runtime.session.block_raw_visual_delivery.assert_called()

    asr.connect = AsyncMock(side_effect=connect_after_visual_fail_closed)
    asr.close = AsyncMock()
    factory = MagicMock(return_value=asr)
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=_selection("qwen")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        factory,
    )

    await runtime._start_independent_asr_if_enabled("audio")

    factory.assert_called_once()
    asr.connect.assert_awaited_once_with()
    assert runtime._asr_route_mode == "independent"
    assert runtime._asr_session is asr
    assert runtime._asr_provider == "qwen"

async def test_partial_preview_is_display_only_and_epoch_guarded() -> None:
    runtime = _Runtime()
    websocket = type("WebSocket", (), {})()
    websocket.send_json = AsyncMock()
    runtime.websocket = websocket
    runtime.current_speech_id = "speech-current"
    runtime._set_microphone_route("independent")
    await _install_active_smart_turn(runtime)
    epoch = runtime._asr_session_epoch
    token = runtime._asr_runtime._asr_partial_turn_token
    assert token is not None
    assert runtime._activate_asr_audio_dispatcher(runtime._asr_lifecycle, token)

    await runtime._send_independent_asr_preview(" draft ", epoch)
    await runtime._send_independent_asr_preview("stale", epoch + 1)

    websocket.send_json.assert_awaited_once_with(
        {
            "type": "user_transcript_preview",
            "text": "draft",
            "turn_id": "speech-current",
            "asr_turn_id": f"asr-{epoch}-1",
        }
    )
    runtime.handle_input_transcript.assert_not_awaited()

async def test_hot_swap_reuses_matching_asr_provider() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = "gemini"
    runtime._independent_asr_route_key = "gemini"
    runtime._start_independent_asr_if_enabled = AsyncMock()

    await runtime._reconcile_independent_asr_after_core_change()

    runtime._start_independent_asr_if_enabled.assert_not_awaited()

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

async def test_start_session_handshake_true_overrides_persisted_disabled(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": False}),
    )
    start_mock = AsyncMock(
        return_value=AsrStartResult(
            status=AsrStartStatus.FAILED,
            failure_code="ASR_START_STALE",
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    runtime.set_independent_asr_handshake(True)
    await runtime._start_independent_asr_if_enabled("audio")

    # The handshake beats the stale persisted value: the independent runtime
    # start is attempted instead of the native fallback.
    start_mock.assert_awaited_once()
    assert runtime._asr_route_mode != "native"

async def test_failed_independent_start_preserves_external_visual_route_memory(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.session.set_visual_delivery_mode = MagicMock()
    runtime.session.block_raw_visual_delivery = MagicMock()
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    start_mock = AsyncMock(
        return_value=AsrStartResult(
            status=AsrStartStatus.FAILED,
            failure_code="ASR_CONNECT_FAILED",
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "blocked"
    assert runtime._visual_route_mode == "independent"
    runtime.session.block_raw_visual_delivery.assert_called()

async def test_provider_restart_reuses_accepted_session_optimization(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
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
            status=AsrStartStatus.READY,
            provider="qwen",
            session_epoch=0,
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    await runtime._start_independent_asr_if_enabled(
        "audio",
        resource_optimization_override=False,
    )
    assert runtime._voice_input_resource_optimization_session_value is False

    # A losing/deduplicated request may overwrite the shared handshake, but a
    # provider-changing restart still belongs to the already accepted session.
    runtime.set_voice_input_resource_optimization_handshake(True)
    runtime.core_api_type = "openai"
    await runtime._reconcile_independent_asr_after_core_change()

    assert start_mock.await_count == 2
    assert all(
        call.kwargs["resource_optimization_enabled"] is False
        for call in start_mock.await_args_list
    )

async def test_start_session_handshake_malformed_value_is_ignored(
    monkeypatch,
    malformed,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": False}),
    )
    start_mock = AsyncMock()
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    # Strict bool typing: truthy non-bool values never enable the route.
    runtime.set_independent_asr_handshake(malformed)
    await runtime._start_independent_asr_if_enabled("audio")

    start_mock.assert_not_awaited()
    assert runtime._asr_route_mode == "native"

async def test_noise_reduction_replacement_waits_for_pipeline_failure_revoke() -> None:
    class _ObservedAsyncLock:
        def __init__(self) -> None:
            self._lock = asyncio.Lock()
            self._requests = 0
            self.second_request = asyncio.Event()

        async def __aenter__(self):
            self._requests += 1
            if self._requests == 2:
                self.second_request.set()
            await self._lock.acquire()
            return self

        async def __aexit__(self, *_exc_info) -> None:
            self._lock.release()

    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    assert runtime._begin_voice_input_connection("socket-a") is True
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_synchronized = True
    transition_lock = _ObservedAsyncLock()
    runtime._voice_input_pipeline_transition_lock = transition_lock
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def block_first_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=block_first_abort)
    source_pipeline = runtime._voice_input_audio_pipeline
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=runtime._capture_ingress_token(),
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=source_pipeline,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    replacement = asyncio.create_task(runtime.apply_voice_input_noise_reduction(False))
    await asyncio.wait_for(transition_lock.second_request.wait(), 1)
    release_abort.set()
    failure_result, replacement_result = await asyncio.wait_for(
        asyncio.gather(failure, replacement),
        1,
    )

    assert failure_result is None
    assert replacement_result is True
    assert runtime._asr_route_mode == "blocked"
    assert runtime._voice_lease_connection_id == ""
    assert runtime._voice_lease_owner == "none"
    assert runtime._voice_input_audio_pipeline is not source_pipeline
    assert runtime._voice_input_audio_pipeline.nr_enabled is False
    assert runtime._voice_input_pipeline_failed is False

async def test_a_live_route_releases_the_pipeline_failure_ingress_latch() -> None:
    """The latch is fail-closed, not permanent: a live route clears it."""

    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    runtime._asr_runtime.abort = AsyncMock()
    runtime._voice_input_audio_pipeline.process = AsyncMock(
        side_effect=RuntimeError("soxr failed")
    )
    token = runtime._capture_ingress_token()
    await runtime._process_microphone_stream_data(
        {"input_type": "audio", "sample_rate_hz": 48_000, "data": [1] * 480},
        ingress_token=token,
    )
    assert runtime._asr_route_mode == "blocked"
    assert runtime._voice_input_pipeline_failure_token is not None

    runtime._set_microphone_route("independent")
    assert runtime._voice_input_pipeline_failure_token is None

async def test_stale_pipeline_failure_never_reports_to_current_identity(
    changed_identity: str,
) -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def delayed_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=delayed_abort)
    token = runtime._capture_ingress_token()
    source_pipeline = runtime._voice_input_audio_pipeline
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=token,
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=source_pipeline,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    if changed_identity == "lease_generation":
        runtime._voice_lease_generation += 1
    elif changed_identity == "hard_mute":
        runtime._voice_lease_hard_muted = True
        runtime._voice_input_transition_generation += 1
    elif changed_identity == "focus_suppression":
        runtime._voice_lease_focus_suppressed = True
        runtime._voice_input_transition_generation += 1
    elif changed_identity == "game_takeover":
        runtime._voice_lease_owner = "game"
        runtime._voice_input_transition_generation += 1
    elif changed_identity == "route_operation":
        object.__setattr__(
            runtime,
            "_asr_route_operation_generation",
            runtime._asr_route_operation_generation + 1,
        )
    elif changed_identity == "core_session":
        runtime.session = SimpleNamespace(stream_audio=AsyncMock())
    else:
        raise AssertionError(changed_identity)

    release_abort.set()
    await asyncio.wait_for(failure, 1)

    runtime.send_status.assert_not_awaited()
    runtime._asr_runtime.abort.assert_awaited_once_with("audio_preprocessing_failed")

async def test_old_detector_endpoint_cannot_seal_replacement_runtime() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    detector = _QueuedSmartTurnDetector()
    detector.detector_epoch = 1
    runtime._asr_detector = detector
    runtime._asr_runtime._asr_current_ingress_token = runtime._capture_ingress_token()
    turn_token = runtime._asr_runtime._capture_turn_token(lifecycle)
    detector._token = turn_token
    candidate = DetectorCandidateKey(detector.detector_epoch, 1)
    envelope = CoreDetectorEventEnvelope(
        event=DetectorTurnEvent(
            ingress=DetectorIngressIdentity(
                ingress_token=turn_token.ingress,
                detector_epoch=detector.detector_epoch,
                sequence_no=1,
            ),
            bound_turn=BoundDetectorTurn(
                candidate=candidate,
                turn_token=turn_token,
            ),
            kind="complete",
        ),
        detector_ref=detector,
        lifecycle_ref=lifecycle,
        session_epoch=runtime._asr_session_epoch,
    )
    draining_started = asyncio.Event()
    release_draining = asyncio.Event()

    async def block_old_lifecycle(payload: str) -> None:
        status = json.loads(payload)
        if (
            status.get("code") == "ASR_LIFECYCLE_STATE"
            and status.get("details", {}).get("state") == "draining"
        ):
            draining_started.set()
            await release_draining.wait()

    runtime.send_status.side_effect = block_old_lifecycle
    endpoint_task = asyncio.create_task(
        runtime._asr_runtime._dispatch_asr_detector_event(envelope)
    )
    await asyncio.wait_for(draining_started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    release_draining.set()
    await asyncio.wait_for(endpoint_task, 1)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    new_session.close.assert_not_awaited()
    assert runtime._asr_route_mode == "independent"
    statuses = [
        json.loads(call.args[0]).get("code")
        for call in runtime.send_status.await_args_list
    ]
    assert "ASR_AUDIO_ORDERING_FAILED" not in statuses

async def test_current_detector_feed_exception_fails_closed_once() -> None:
    runtime = _Runtime()
    runtime._asr_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    _install_ready_lifecycle(runtime, "qwen")
    runtime._asr_detector.feed = AsyncMock(
        side_effect=RuntimeError("current detector failed")
    )
    ingress = runtime._capture_ingress_token()
    runtime._asr_runtime._asr_current_ingress_token = ingress

    result = await runtime._asr_runtime.submit(
        ProcessedVoiceFrame(b"\x01\x00" * 160, 16_000, 0.8, True),
        ingress_token=ingress,
    )
    await asyncio.sleep(0)

    assert result.status is AsrSubmitStatus.UNAVAILABLE
    codes = [
        json.loads(call.args[0])["code"] for call in runtime.send_status.await_args_list
    ]
    assert codes.count("ASR_INDEPENDENT_STREAM_FAILED") == 1
    assert runtime._asr_session is None
    assert runtime._asr_lifecycle is None
    assert runtime._asr_detector is None

async def test_old_pipeline_failure_does_not_report_replacement_provider() -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
    runtime.is_flushing_hot_swap_cache = False
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "provider-a"
    abort_started = asyncio.Event()
    release_abort = asyncio.Event()

    async def block_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=block_abort)
    old_pipeline = runtime._voice_input_audio_pipeline
    old_pipeline.process = AsyncMock(side_effect=RuntimeError("soxr failed"))
    processing = asyncio.create_task(
        runtime._process_microphone_stream_data(
            {
                "input_type": "audio",
                "sample_rate_hz": 48_000,
                "data": [1] * 480,
            },
            ingress_token=runtime._capture_ingress_token(),
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    runtime._voice_input_audio_pipeline = SimpleNamespace(
        process=AsyncMock(),
        close=AsyncMock(),
    )
    runtime._voice_input_pipeline_failed = False
    runtime._independent_asr_provider = "provider-b"
    runtime._set_microphone_route("independent")
    release_abort.set()
    await asyncio.wait_for(processing, 1)

    assert runtime._asr_route_mode == "independent"
    assert runtime._independent_asr_provider == "provider-b"
    assert runtime._voice_input_pipeline_failed is False
    runtime.send_status.assert_not_awaited()
    runtime._asr_runtime.abort.assert_awaited_once_with("audio_preprocessing_failed")

async def test_free_core_uses_native_asr_when_preferences_are_unreadable(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "free"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(side_effect=OSError("preferences unavailable")),
    )
    start_mock = AsyncMock()
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)
    runtime.set_independent_asr_handshake(True)

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "native"
    start_mock.assert_not_awaited()
    assert "ASR_INDEPENDENT_DISABLED" in runtime.send_status.await_args.args[0]

async def test_blocked_route_consumes_audio_without_an_asr_or_omni_send() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "blocked"

    assert (
        await runtime._route_microphone_audio(
            b"\x00\x00",
            sample_rate_hz=16_000,
        )
        is True
    )
    assert runtime._asr_route_mode == "blocked"

async def test_session_swap_during_transcript_reprepares_promoted_final() -> None:
    runtime = _Runtime()
    old_session = runtime.session
    old_session.create_response.side_effect = RuntimeError("closed arbiter")
    new_session = type(
        "Omni",
        (),
        {
            "create_response": AsyncMock(),
            "prepare_external_voice_turn": AsyncMock(),
            "submit_external_voice_turn": AsyncMock(),
            "abandon_external_voice_turn": MagicMock(),
        },
    )()

    async def swap_session(*_args, **_kwargs) -> bool:
        runtime.session = new_session
        return True

    runtime.handle_input_transcript.side_effect = swap_session
    await _start_and_seal_turn(runtime, "glm")

    await runtime._handle_independent_asr_final(
        "belongs to old role",
        runtime._asr_session_epoch,
        "glm",
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    old_session.create_response.assert_not_awaited()
    new_session.create_response.assert_not_awaited()
    new_session.prepare_external_voice_turn.assert_awaited_once()
    new_session.submit_external_voice_turn.assert_awaited_once()

async def test_stale_runtime_ready_result_cannot_replace_new_route(
    monkeypatch,
) -> None:
    start_entered = asyncio.Event()
    release_start = asyncio.Event()

    class BlockingBridge:
        def __init__(self) -> None:
            self.session_epoch = 0
            self.audio_generation = 0

        def capture_ingress_token(
            self,
            *,
            connection_id,
            lease_generation,
            route_generation,
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

        async def start(self, **_kwargs) -> AsrStartResult:
            source_epoch = self.session_epoch
            start_entered.set()
            await release_start.wait()
            return AsrStartResult(
                AsrStartStatus.READY,
                provider="old-provider",
                session_epoch=source_epoch,
            )

    runtime = _Runtime()
    bridge = BlockingBridge()
    object.__setattr__(runtime, "_asr_runtime", bridge)
    runtime.core_api_type = "old-core"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )

    starting = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(start_entered.wait(), 1)
    runtime._begin_asr_route_operation()
    bridge.session_epoch += 1
    runtime.core_api_type = "new-core"
    runtime._independent_asr_provider = "new-provider"
    runtime._independent_asr_route_key = "new-core"
    runtime._set_microphone_route("independent")
    release_start.set()
    await asyncio.wait_for(starting, 1)

    assert runtime._independent_asr_provider == "new-provider"
    assert runtime._independent_asr_route_key == "new-core"
    assert runtime._asr_route_mode == "independent"

async def test_old_notifications_cannot_override_new_generation() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    first_send_entered = asyncio.Event()
    release_first_send = asyncio.Event()
    payloads = []

    async def ordered_send_status(payload: str) -> None:
        payloads.append(json.loads(payload))
        if len(payloads) == 1:
            first_send_entered.set()
            await release_first_send.wait()

    runtime.send_status = AsyncMock(side_effect=ordered_send_status)
    old_epoch = runtime._asr_session_epoch
    old_event = AsrLifecycleNotification(
        state="local_listen",
        provider="old-provider",
        session_epoch=old_epoch,
    )
    old_delivery = asyncio.create_task(runtime._send_core_asr_lifecycle(old_event))
    await asyncio.wait_for(first_send_entered.wait(), 1)

    runtime._asr_session_epoch += 1
    new_epoch = runtime._asr_session_epoch
    new_event = AsrLifecycleNotification(
        state="blocked",
        provider="new-provider",
        session_epoch=new_epoch,
    )
    new_delivery = asyncio.create_task(runtime._send_core_asr_lifecycle(new_event))
    release_first_send.set()
    await asyncio.wait_for(
        asyncio.gather(old_delivery, new_delivery),
        1,
    )
    await runtime._send_core_asr_lifecycle(old_event)
    await runtime._send_core_asr_status(
        AsrStatusEvent(
            code="ASR_OLD_READY",
            provider="old-provider",
            session_epoch=old_epoch,
        )
    )
    await runtime._send_core_asr_status(
        AsrStatusEvent(
            code="ASR_NEW_READY",
            provider="new-provider",
            session_epoch=new_epoch,
        )
    )

    assert [
        payload["details"]["state"]
        for payload in payloads
        if payload["code"] == "ASR_LIFECYCLE_STATE"
    ] == ["local_listen", "blocked"]
    assert payloads[-1] == {
        "code": "ASR_NEW_READY",
        "details": {
            "provider": "new-provider",
            "session_epoch": new_epoch,
        },
    }
    assert payloads[1]["details"]["session_epoch"] == new_epoch

async def test_runtime_failure_leaves_the_game_lease_alone() -> None:
    # The galgame route holds the mic through its built-in consumer route and tears
    # down via GAME_ROUTE_ENDED; re-basing the identity must not start
    # collaterally revoking it.
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._voice_lease_connection_id = "socket-a"
    runtime._voice_lease_owner = "game"

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="current-provider",
            session_epoch=runtime._asr_session_epoch,
        )
    )

    assert runtime._voice_lease_connection_id == "socket-a"

async def test_stale_start_abort_does_not_clobber_newer_start_placeholder(
    monkeypatch,
) -> None:
    """A stale start parked in its abort must not clear the blocked
    placeholder a newer start installed meanwhile: clearing it would make
    the newer start's fence fail before it even reaches the native
    fallback, leaving the route blocked with no failure status."""

    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    runtime._voice_lease_connection_id = "conn-A"
    runtime._voice_lease_generation = 0

    a_start_parked = asyncio.Event()
    release_a_start = asyncio.Event()
    a_abort_parked = asyncio.Event()
    release_a_abort = asyncio.Event()
    b_settings_parked = asyncio.Event()
    release_b_settings = asyncio.Event()
    start_calls: list[dict] = []

    async def fake_runtime_start(**kwargs):
        start_calls.append(kwargs)
        if len(start_calls) == 1:
            a_start_parked.set()
            await release_a_start.wait()
            return AsrStartResult(
                AsrStartStatus.FAILED,
                failure_code="ASR_START_STALE",
                session_epoch=runtime._asr_session_epoch,
            )
        return AsrStartResult(
            AsrStartStatus.READY,
            provider="qwen",
            session_epoch=runtime._asr_session_epoch,
        )

    async def fake_abort(reason):
        a_abort_parked.set()
        await release_a_abort.wait()

    runtime._asr_runtime.start = fake_runtime_start
    runtime._asr_runtime.abort = fake_abort
    runtime._asr_runtime.close = AsyncMock()

    settings_calls = 0

    async def load_settings(**_kwargs):
        nonlocal settings_calls
        settings_calls += 1
        if settings_calls == 2:
            b_settings_parked.set()
            await release_b_settings.wait()
        return {"independentAsrEnabled": True}

    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        load_settings,
    )

    task_a = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(a_start_parked.wait(), 1)
    runtime._begin_voice_input_connection("conn-B")
    release_a_start.set()
    await asyncio.wait_for(a_abort_parked.wait(), 1)

    task_b = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(b_settings_parked.wait(), 1)
    assert runtime._independent_asr_route_key == "qwen"

    release_a_abort.set()
    await asyncio.wait_for(task_a, 1)
    assert runtime._independent_asr_route_key == "qwen"

    release_b_settings.set()
    await asyncio.wait_for(task_b, 1)

    assert len(start_calls) == 2
    assert runtime._asr_route_mode == "independent"
    assert runtime._independent_asr_provider == "qwen"

async def test_notification_waiting_on_lock_drops_same_epoch_stale_identity(
    notification: str,
) -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    current_epoch = runtime._asr_session_epoch
    await runtime._asr_notification_lock.acquire()
    if notification == "status":
        event = AsrStatusEvent(
            code="ASR_OLD_READY",
            provider="old-provider",
            session_epoch=current_epoch,
        )
        delivery = asyncio.create_task(runtime._send_core_asr_status(event))
    elif notification == "lifecycle":
        event = AsrLifecycleNotification(
            state="local_listen",
            provider="old-provider",
            session_epoch=current_epoch,
        )
        delivery = asyncio.create_task(runtime._send_core_asr_lifecycle(event))
    else:
        event = AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="old-provider",
            session_epoch=current_epoch,
        )
        delivery = asyncio.create_task(runtime._handle_core_asr_failure(event))
    await asyncio.sleep(0)

    runtime._asr_audio_generation += 1
    runtime._asr_notification_lock.release()
    await asyncio.wait_for(delivery, 1)

    runtime.send_status.assert_not_awaited()
    assert runtime._asr_route_mode == "independent"

async def test_lease_resync_rearms_for_new_microphone_route_generation() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True

    await runtime._maybe_signal_voice_lease_resync()
    first_episode = runtime._voice_lease_resync_signal_state
    assert first_episode is not None
    assert first_episode[-1] == runtime._microphone_route_generation

    runtime._set_microphone_route("native")
    assert runtime._voice_lease_resync_signal_state is None
    await runtime._maybe_signal_voice_lease_resync()
    native_episode = runtime._voice_lease_resync_signal_state
    assert native_episode is not None

    runtime._set_microphone_route("native")
    await runtime._maybe_signal_voice_lease_resync()
    assert runtime._voice_lease_resync_signal_state == native_episode
    assert runtime.send_status.await_count == 2

    runtime._set_microphone_route("blocked")

    await runtime._maybe_signal_voice_lease_resync()
    second_episode = runtime._voice_lease_resync_signal_state
    assert second_episode is not None
    assert second_episode != first_episode
    assert second_episode[-1] == runtime._microphone_route_generation
    assert runtime.send_status.await_count == 3

async def test_cancelled_blocked_text_notice_retries_same_episode() -> None:
    runtime = _Runtime()
    runtime.input_mode = "text"
    assert runtime._begin_voice_input_connection("chat-window") is True
    runtime._set_microphone_route("blocked")
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def block_send(_message: str) -> None:
        send_started.set()
        await release_send.wait()

    runtime.send_status = AsyncMock(side_effect=block_send)
    first = asyncio.create_task(
        runtime._maybe_signal_blocked_text_mode_microphone()
    )
    await asyncio.wait_for(send_started.wait(), 1)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert runtime._blocked_text_mode_microphone_signal_state is None
    runtime.send_status = AsyncMock()
    await runtime._maybe_signal_blocked_text_mode_microphone()

    runtime.send_status.assert_awaited_once()
    assert runtime._blocked_text_mode_microphone_signal_state is not None

async def test_noise_reduction_disabled_reaches_pipeline_audio_processor(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    created: list[dict] = []

    class _RecordingProcessor:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)
            self.speech_probability = 0.0
            self.rnnoise_available = False

        def process_chunk(self, _audio_bytes: bytes) -> bytes:
            return b""

        def close(self) -> None:
            return None

    monkeypatch.setattr(audio_input_module, "AudioProcessor", _RecordingProcessor)
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

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._voice_input_noise_reduction_enabled is False
    assert runtime._voice_input_audio_pipeline.nr_enabled is False
    await runtime._voice_input_audio_pipeline.process(
        b"\x01\x00" * 480,
        sample_rate_hz=48_000,
    )
    assert created[-1]["noise_reduce_enabled"] is False

    started_pipeline = runtime._voice_input_audio_pipeline
    await runtime._close_independent_asr(next_route_mode="blocked")

    assert runtime._voice_input_audio_pipeline is not started_pipeline
    assert runtime._voice_input_audio_pipeline.nr_enabled is False
    await runtime._voice_input_audio_pipeline.process(
        b"\x01\x00" * 480,
        sample_rate_hz=48_000,
    )
    assert len(created) == 2
    assert created[-1]["noise_reduce_enabled"] is False

async def test_accepted_final_dropped_by_generation_bump_abandons_turn(
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
    runtime.session.abandon_external_voice_turn = MagicMock()
    component._asr_current_ingress_token = runtime._capture_ingress_token()
    epoch = component._asr_session_epoch
    on_activity = callbacks[0]["on_speech_activity"]
    on_final = callbacks[0]["on_input_transcript"]

    await on_activity(SpeechActivityEvent.SPEECH_STARTED)
    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    sealed_turn_id = lifecycle.snapshot.turn_id
    await component._handle_independent_asr_endpoint(epoch)
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING

    await on_final("hello world")

    # The final was accepted, but the generation moves on before the serial
    # transcript dispatcher delivers the queued envelope.
    component._asr_audio_generation += 1
    await component.wait_transcript_idle()

    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{epoch}-{sealed_turn_id}"
    )

async def test_start_resolves_selection_off_event_loop(monkeypatch) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    selection = _selection("qwen", "provider")
    resolver_threads: list[threading.Thread] = []

    def resolver(core_type: str):
        assert core_type == "qwen"
        resolver_threads.append(threading.current_thread())
        return selection

    session = SimpleNamespace(
        is_ready=True,
        connect=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(runtime_module, "_resolve_asr_selection", resolver)
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        lambda _core_type, **_kwargs: session,
    )
    detector_factory = MagicMock(return_value=_ReadyDetector())
    monkeypatch.setattr(runtime_module, "DetectorRuntime", detector_factory)

    result = await runtime._asr_runtime.start(
        route_key="qwen",
        resource_optimization_enabled=False,
    )

    assert result.status is AsrStartStatus.READY
    assert len(resolver_threads) == 1
    assert resolver_threads[0] is not threading.main_thread()
    assert (
        detector_factory.call_args.kwargs["resource_optimization_enabled"] is False
    )
    assert detector_factory.call_args.kwargs["speaker_shadow"] is None

async def test_teardown_routines_share_one_turn_state_reset() -> None:
    import ast
    import inspect as inspect_module

    from main_logic.asr_client import runtime as runtime_module

    source = inspect_module.getsource(runtime_module.IndependentAsrRuntime)
    tree = ast.parse(source)
    class_node = tree.body[0]
    for method_name in (
        "_detach_independent_asr",
        "_abort_transport",
        "_handle_independent_asr_error",
    ):
        method = next(
            node
            for node in class_node.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
            and node.name == method_name
        )
        calls = {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert "_reset_asr_turn_state" in calls, method_name

async def test_stale_provider_endpoint_releases_local_final_reservation() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    component = runtime._asr_runtime
    component._runtime_identity_matches = MagicMock(return_value=False)
    epoch = component._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )

    await runtime._handle_independent_asr_endpoint(epoch)

    assert component._asr_reserved_final_key is None
    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE

async def test_provider_overflow_waiting_on_final_lock_is_identity_fenced(
    replacement: str,
) -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    ingress_token = runtime._asr_runtime._asr_current_ingress_token
    assert ingress_token is not None

    await runtime._asr_final_lock.acquire()
    overflow_task = asyncio.create_task(
        runtime._handle_audio_ingress_backpressure(
            ingress_token,
            observed_state=VoiceLifecycleState.DRAINING,
        )
    )
    await asyncio.sleep(0)
    if replacement == "epoch":
        runtime._asr_session_epoch += 1
    elif replacement == "lifecycle":
        replacement_lifecycle = VoiceInputLifecycleController(
            provider_policy=resolve_provider_policy("openai", "provider"),
            shadow_mode=False,
        )
        replacement_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
        runtime._asr_lifecycle = replacement_lifecycle
    else:
        runtime._asr_detector = _ReadyDetector()
    runtime._asr_final_lock.release()
    await overflow_task

    detector.discard_provider_successor.assert_not_awaited()
    assert "ASR_INGRESS_BACKPRESSURE" not in str(runtime.send_status.await_args_list)
    watchdog = runtime._asr_final_watchdog_task
    if watchdog is not None:
        watchdog.cancel()

async def test_independent_visual_sync_failure_blocks_raw_images_without_stopping_asr() -> None:
    runtime = _Runtime()
    call_order: list[str] = []

    def block_raw_visual_delivery() -> None:
        call_order.append("block")

    def fail_visual_mode_sync(_mode: str) -> None:
        call_order.append("sync")
        raise RuntimeError("stale realtime session")

    runtime.session.block_raw_visual_delivery = block_raw_visual_delivery
    runtime.session.set_visual_delivery_mode = fail_visual_mode_sync

    runtime._set_microphone_route("independent")

    assert runtime._asr_route_mode == "independent"
    assert call_order == ["block"]

async def test_independent_multimodal_turn_rejects_owned_frame_expired_at_final() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=80)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    assert runtime._stage_independent_visual_frame(
        "expired-owned-frame",
        source="screen",
        request_id="screen-expired",
        captured_at=record.started_at,
    )

    with patch(
        "main_logic.core.asr_runtime.time.monotonic",
        return_value=(
            record.started_at + runtime._independent_visual_frame_ttl_s + 1.0
        ),
    ):
        turn = runtime._snapshot_core_multimodal_turn(turn_id, "delayed final")

    assert turn is None

async def test_dispatch_hands_the_ownership_predicate_to_the_handoff() -> None:
    """Checking before the handoff is not enough; it must check inside too.

    Connecting and promoting the Offline candidate, starting TTS and syncing
    tools are the longest awaits on the path, and the handoff's own
    ``operation_is_current`` covers route identity only. A guard that merely
    exercises the predicate in isolation still passes when the dispatch stops
    handing it over, so assert the call site itself.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="handoff_required"
    )
    runtime.session.submit_external_voice_turn = AsyncMock()
    seen: dict = {}

    async def observe_predicate_inside_the_handoff(_turn, **kwargs):
        still_owned = kwargs["visual_still_owned"]
        seen["before"] = still_owned()
        # 后继发声在交接进行中 prepare —— 谓词必须立刻反映出来，而不是停在
        # 进入交接那一刻的快照。
        seen["record"].invalidated.set()
        seen["after"] = still_owned()
        return True

    runtime._handoff_to_offline_vlm_and_submit = AsyncMock(
        side_effect=observe_predicate_inside_the_handoff
    )
    handoff_token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    handoff_turn_id = (
        f"asr-{handoff_token.ingress.session_epoch}-{handoff_token.turn_id}"
    )
    runtime._begin_core_multimodal_turn(handoff_turn_id, handoff_token)
    handoff_record = runtime._core_multimodal_turns[handoff_turn_id]
    seen["record"] = handoff_record
    assert runtime._stage_independent_visual_frame(
        "frame-of-this-turn",
        source="screen",
        request_id="screen-1",
        captured_at=handoff_record.started_at,
    )

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=handoff_token,
            provider="openai",
            text="look here",
        )
    )

    runtime._handoff_to_offline_vlm_and_submit.assert_awaited_once()
    assert seen["before"] is True
    assert seen["after"] is False

async def test_new_turn_wakes_visual_validation_wait_without_cancelling_task() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    first_token = VoiceTurnToken(
        ingress=runtime._capture_ingress_token(),
        turn_id=82,
    )
    first_turn_id = (
        f"asr-{first_token.ingress.session_epoch}-{first_token.turn_id}"
    )
    runtime._begin_core_multimodal_turn(first_turn_id, first_token)
    first_record = runtime._core_multimodal_turns[first_turn_id]
    release = asyncio.Event()
    validation_task = asyncio.create_task(release.wait())
    assert runtime._track_independent_visual_validation_task(
        validation_task,
        captured_at=first_record.started_at,
    )
    waiting = asyncio.create_task(
        runtime._await_independent_visual_validation_tasks(first_turn_id)
    )
    await asyncio.sleep(0)

    second_token = VoiceTurnToken(
        ingress=runtime._capture_ingress_token(),
        turn_id=83,
    )
    runtime._begin_core_multimodal_turn(
        f"asr-{second_token.ingress.session_epoch}-{second_token.turn_id}",
        second_token,
    )

    await asyncio.wait_for(waiting, timeout=0.1)
    assert not validation_task.done()
    release.set()
    await validation_task

async def test_native_visual_sync_failure_keeps_raw_images_blocked() -> None:
    runtime = _Runtime()
    call_order: list[str] = []

    def allow_raw_visual_delivery() -> None:
        call_order.append("allow")

    def block_raw_visual_delivery() -> None:
        call_order.append("block")

    def fail_visual_mode_sync(_mode: str) -> None:
        call_order.append("sync")
        raise RuntimeError("stale realtime session")

    runtime.session.allow_raw_visual_delivery = allow_raw_visual_delivery
    runtime.session.block_raw_visual_delivery = block_raw_visual_delivery
    runtime.session.set_visual_delivery_mode = fail_visual_mode_sync

    runtime._set_microphone_route("native")

    assert runtime._asr_route_mode == "native"
    assert call_order == ["sync", "block"]

async def test_post_endpoint_cache_frame_cannot_seed_an_empty_turn() -> None:
    """The empty-record fallback must respect the endpoint cutoff too."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=94)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    _seal_utterance(runtime)
    runtime._mark_independent_asr_endpoint_if_sealed()
    runtime._stage_independent_visual_frame(
        "post-endpoint-frame",
        source="screen",
        request_id="screen-post",
        captured_at=record.endpoint_at + 0.5,
    )
    # 缓存里有这一帧（主动搭话观察还要用），但本回合一帧都没收到。
    assert runtime._latest_independent_visual_frame.image_b64 == "post-endpoint-frame"
    assert record.last_frame is None

    assert runtime._snapshot_core_multimodal_turn(turn_id, "what is that") is None

async def test_this_turn_seal_in_the_same_tick_is_still_its_cutoff() -> None:
    """The other direction: this turn's own seal must survive a tick collision.

    A very short utterance can seal inside the same ~15ms tick its record was
    registered in; PROVIDER_FINAL then clears the live field, leaving only the
    retained copy. A pure timestamp test is wrong in one direction or the
    other, and this is the half where "equality belongs to the previous turn"
    is wrong: this turn loses its cutoff and post-speech frames get folded into
    its transcript.

    Hence the criterion is turn identity, not the timestamp -- the runtime
    records which turn the retained seal belongs to.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=101)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 本轮自己的封口，恰好与注册落在同一个 tick 上。
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = record.registered_at
    runtime._asr_last_turn_endpointed_key = record.turn_id

    runtime._stage_independent_visual_frame(
        "opening-frame",
        source="screen",
        request_id="screen-opening",
        captured_at=record.started_at,
    )

    assert record.endpoint_at == record.registered_at, (
        "本轮自己的封口被当成上一轮残值丢掉了：相等时必须靠身份而不是时间戳"
    )

async def test_frame_validated_during_lifecycle_notification_joins_the_turn() -> None:
    """Speech onset, not record creation, is the ownership boundary."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    # 刻意只设 _asr_turn_onset_at：_asr_turn_audio_started_at 在两条生产路径上是
    # 投递完成之后才打的，用它当起点正是被修掉的那个缺陷，所以这条用例不能靠它。
    runtime._asr_turn_onset_at = onset

    # 语音已确认，Core 还卡在 _send_asr_lifecycle_state 的投递里；这一帧就是这段
    # 发声的开头（用户开口时指的东西），它先于 record 落地。
    assert runtime._stage_independent_visual_frame(
        "onset-frame",
        source="screen",
        request_id="screen-onset",
        captured_at=onset + 0.01,
    )

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=98)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("onset-frame",)

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

async def test_all_prerecord_frames_join_the_turn_not_just_the_newest() -> None:
    """Frames validated before the record exists must survive as a span.

    The single-slot cache keeps only the newest frame, and the pending-task
    stash drops a task the moment it completes. If lifecycle delivery is slow
    enough for several validations to land first, keeping only the newest one
    silently loses the actual first/middle frames of the utterance.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset

    for index in range(3):
        assert runtime._stage_independent_visual_frame(
            f"prerecord-{index}",
            source="screen",
            request_id=f"screen-{index}",
            captured_at=onset + 0.01 * (index + 1),
        )
    assert len(runtime._prerecord_visual_frames) == 3

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=103)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    # 开头 / 中间 / 结尾都在，而不是只剩最新那张。
    assert turn.images == ("prerecord-0", "prerecord-1", "prerecord-2")
    # 消费即清空，不会漏进下一轮。
    assert runtime._prerecord_visual_frames == []

async def test_overlapping_successor_is_not_sealed_by_its_predecessor() -> None:
    """The successor's onset predates the predecessor's seal — by design.

    A provider-VAD successor utterance begins while the previous turn is still
    ACTIVE, so its recorded onset is EARLIER than the previous turn's endpoint.
    Comparing the retained seal against ``started_at`` would therefore bind the
    predecessor's endpoint to the successor and reject every frame it captures.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    successor_onset = time.monotonic() - 1.0
    predecessor_seal = successor_onset + 0.3
    runtime._asr_turn_onset_at = successor_onset
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = predecessor_seal

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=105)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    assert record.started_at < predecessor_seal

    assert runtime._stage_independent_visual_frame(
        "successor-frame",
        source="screen",
        request_id="screen-successor",
        captured_at=predecessor_seal + 0.4,
    )
    assert record.endpoint_at is None

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "and this?")

    assert turn is not None
    assert turn.images == ("successor-frame",)

async def test_new_prepare_does_not_erase_a_preceding_turn_record() -> None:
    """An in-flight accepted final must still find its own record.

    The preceding final can still be running in TranscriptDispatcher (for
    example awaiting the bounded visual-validation join) when the successor is
    prepared. Clearing every record there makes that dispatch fail its identity
    self-check and return without recording OR submitting the transcript — the
    overlapping utterance erases a complete user turn.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    first = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=201)
    first_id = f"asr-{first.ingress.session_epoch}-{first.turn_id}"
    runtime._begin_core_multimodal_turn(first_id, first)
    first_record = runtime._core_multimodal_turns[first_id]

    second = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=202)
    second_id = f"asr-{second.ingress.session_epoch}-{second.turn_id}"
    runtime._begin_core_multimodal_turn(second_id, second)

    # 前一条的记录仍在，且仍是同一个对象 —— 身份自检因此不会误判。
    assert runtime._core_multimodal_turns.get(first_id) is first_record
    # 但它已被标记作废：图归新回合，旧 final 只是别被整句丢掉。
    assert first_record.invalidated.is_set()
    assert runtime._core_multimodal_turns.get(second_id) is not None

async def test_all_records_mid_dispatch_keeps_them_past_the_cap() -> None:
    """When nothing is evictable the cap yields, it does not pick a victim.

    Every record in the dict belongs to a final that is still being dispatched,
    so evicting any of them drops a sentence the user already finished. Going
    over the cap is the lesser failure: unbounded growth would mean a dispatch
    that never returns, which is a different bug and must not be papered over
    by discarding speech.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    for turn_id in range(701, 701 + _MAX_LIVE_TURN_RECORDS + 4):
        token = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(), turn_id=turn_id
        )
        record_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
        runtime._begin_core_multimodal_turn(record_id, token)
        # 每一条都立刻进入派发，于是永远没有可淘汰的记录。
        runtime._core_multimodal_turns[record_id].dispatch_started = True

    assert len(runtime._core_multimodal_turns) == _MAX_LIVE_TURN_RECORDS + 4
    assert all(
        record.dispatch_started
        for record in runtime._core_multimodal_turns.values()
    )
    # 各自的 dispatch 收尾时才回落到界内。
    for record_id in list(runtime._core_multimodal_turns)[:4]:
        runtime._abandon_core_voice_turn(record_id, session_ref=None)
    assert len(runtime._core_multimodal_turns) == _MAX_LIVE_TURN_RECORDS

async def test_prerecord_stash_still_arms_while_older_records_are_retained() -> None:
    """The dict is no longer empty between turns, so 'no records' is the wrong test."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    first = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=305)
    first_id = f"asr-{first.ingress.session_epoch}-{first.turn_id}"
    runtime._begin_core_multimodal_turn(first_id, first)
    runtime._core_multimodal_turns[first_id].invalidated.set()

    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset
    assert runtime._stage_independent_visual_frame(
        "between-turns-frame",
        source="screen",
        request_id="screen-between",
        captured_at=onset,
    )

    # 当前这一轮还没建起来，这帧必须被暂存下来等它。
    assert [f.image_b64 for f in runtime._prerecord_visual_frames] == [
        "between-turns-frame"
    ]

async def test_a_late_registration_still_adopts_its_real_onset() -> None:
    """Waiting behind a provider final does not make an onset stale.

    An overlapping utterance registers only after the previous turn's final
    lands, and that provider timeout reaches 40s in the registry. Judging the
    onset by the FRAME freshness window (5s) rejects it, resets ``started_at``
    to registration time and drops every frame captured since the user actually
    started speaking -- the turn goes text-only while the screen was streaming
    the whole time. Frame freshness is enforced separately, at freeze time.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    now = time.monotonic()
    real_onset = now - 30.0  # 排在一个 provider final 后面，远超帧的 5s TTL
    runtime._asr_runtime._asr_turn_onset_at = real_onset

    frame_at = real_onset + 1.0
    assert runtime._stage_independent_visual_frame(
        "frame-from-the-real-onset",
        source="screen",
        request_id="screen-late",
        captured_at=frame_at,
    )

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=902)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 起点是真实开口时刻，不是注册时刻。
    assert record.started_at == pytest.approx(real_onset, abs=0.01)
    # 那一刻以来的帧被采纳了，而不是整轮退化成纯文本。
    assert [f.image_b64 for f in record.sampled_frames()] == [
        "frame-from-the-real-onset"
    ]

async def test_credit_redemption_drops_the_pending_slot_when_the_transport_identity_moves_on() -> None:
    """Dual of the direct-replay case for the completed-overlap credit path.

    Both compensation blocks force-confirm the same way, so both strand the
    pending slot the same way when the runtime identity drifts across the
    ACTIVE broadcast. Covering only one leaves the other free to regress.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch
    component = runtime._asr_runtime

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # 后继在上一轮还 ACTIVE 时开口又停顿：攒下一张 completed-overlap credit。
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._asr_overlap_completed_turns == 1
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE

    # 同上：注入一个明显靠前的时刻，后面那条继承断言才有分辨力。
    recorded_onset = time.monotonic() - 5.0
    component._asr_overlap_completed_onsets[0] = recorded_onset

    component._asr_session.is_ready = False
    lifecycle_ref = runtime._asr_lifecycle
    real_on_lifecycle = component._callbacks.on_lifecycle
    drifted = False

    async def _drift_transport_midflight(note: AsrLifecycleNotification) -> None:
        nonlocal drifted
        if note.state == VoiceLifecycleState.ACTIVE.value and not drifted:
            drifted = True
            component._asr_session = None
            lifecycle_ref.invalidate_transport()
        await real_on_lifecycle(note)

    component._callbacks = replace(
        component._callbacks,
        on_lifecycle=_drift_transport_midflight,
    )

    # 后继自己的 endpoint 兑付这张 credit，重放停在 PREWARMING 后就地补确认。
    await runtime._handle_independent_asr_endpoint(epoch)

    assert drifted is True
    assert runtime._asr_session_epoch == epoch
    assert runtime._asr_lifecycle is lifecycle_ref
    assert component._asr_pending_speech_confirmed is False
    assert component._asr_pending_speech_onset_at is None
    assert component._asr_turn_onset_at == recorded_onset
    # 确认已经落地（lifecycle 是 ACTIVE，这一轮会照常封口），所以那张 credit
    # 必须跟着确认一起记掉，不能被身份漂移那条 return 跳过。
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_overlap_completed_turns == 0
    assert not component._asr_overlap_completed_onsets

    # 身份漂移让这一轮停在 ACTIVE（那次 return 越过了随后的封口）。恢复身份、
    # 把它正常走完，才谈得上「下一次不相干的开口」。
    component._asr_session = type("Asr", (), {"is_ready": True})()
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE

    fresh_floor = time.monotonic()
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    assert component._asr_turn_onset_at != recorded_onset
    assert component._asr_turn_onset_at >= fresh_floor

    # 行为层：后面一次**真实**的 overlap 兑付必须拿到它自己的 onset。credit 若
    # 被漏记，这张陈旧的会按 FIFO 排在前面先被兑走，这一轮就拿错了开口时刻。
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("third", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._asr_overlap_completed_turns == 1
    later_onset = component._asr_overlap_completed_onsets[0]
    assert later_onset != recorded_onset

    await runtime._handle_independent_asr_endpoint(epoch)
    assert component._asr_turn_onset_at == later_onset
    assert runtime._asr_overlap_completed_turns == 0

async def test_overlap_credit_survives_a_replay_that_never_activates() -> None:
    """Spend the credit on a successful wake-up, not on the attempt.

    The replay can leave the lifecycle short of ACTIVE when the session is
    momentarily unavailable. Deducting the credit first strands that turn: its
    endpoint can no longer seal, the final queued right behind it is discarded,
    and the popped onset goes on to be inherited by an unrelated later turn.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    assert runtime._asr_overlap_completed_turns == 1
    onset_before = list(runtime._asr_overlap_completed_onsets)

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE

    # 重放唤不醒这一轮（会话暂时不可用）。
    async def refuse_to_wake(*_args, **_kwargs):
        return None

    runtime._asr_runtime._handle_independent_asr_activity = refuse_to_wake
    await runtime._handle_independent_asr_endpoint(epoch)

    # credit 和 onset 都原样留着，等下一次兑付。
    assert runtime._asr_overlap_completed_turns == 1
    assert list(runtime._asr_overlap_completed_onsets) == onset_before
    # 借出去的 onset 也收回了，不会被后面不相干的回合继承。
    assert runtime._asr_pending_speech_onset_at is None

async def test_lease_resync_does_not_hand_a_successor_the_replaced_episode() -> None:
    """A takeover inside the display send must not reach the new recorder.

    The display push is an await, so the voice-owner lookup that follows it
    can resolve the SUCCESSOR's socket. Withholding the ledger commit
    afterwards is not enough -- a delivered status cannot be retracted.
    """

    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True

    send_started = asyncio.Event()
    release_send = asyncio.Event()
    owner_payloads: list[dict] = []

    async def stalling_send_status(_message: str) -> bool:
        send_started.set()
        await release_send.wait()
        return True

    async def record_owner_send(payload: dict):
        owner_payloads.append(payload)
        return successor_socket

    successor_socket = object()
    runtime.send_status = AsyncMock(side_effect=stalling_send_status)
    runtime._voice_owner_socket = lambda: successor_socket
    runtime._send_to_voice_owner = record_owner_send

    signal = asyncio.create_task(runtime._maybe_signal_voice_lease_resync())
    await asyncio.wait_for(send_started.wait(), timeout=1)

    # A different window claims the microphone while the display push is stuck.
    assert runtime._begin_voice_input_connection("recorder-window") is True
    release_send.set()
    await asyncio.wait_for(signal, timeout=1)

    assert owner_payloads == []
    assert runtime._voice_lease_resync_signal_state is None
