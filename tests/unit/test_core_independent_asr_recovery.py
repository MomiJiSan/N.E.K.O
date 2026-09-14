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

async def test_external_voice_suppression_resets_native_audio_turn() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime._invalidate_voice_pcm_sync = MagicMock()
    runtime._abort_independent_asr = AsyncMock()
    runtime.session.clear_audio_buffer = AsyncMock()

    await runtime.set_voice_input_suppressed(
        "voice_identity_enrollment",
        suppressed=True,
    )

    runtime.session.clear_audio_buffer.assert_awaited_once_with()
    runtime._abort_independent_asr.assert_not_awaited()

async def test_activity_probe_tracks_accepted_final_until_dispatch_completes() -> None:
    runtime = _Runtime()
    await _start_and_seal_turn(runtime, "qwen")
    sealed_token = runtime._asr_sealed_turn_token
    assert sealed_token is not None
    release_started = asyncio.Event()
    release_lease = asyncio.Event()
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    class _BlockingLease:
        token = sealed_token.turn

        async def release(self) -> None:
            release_started.set()
            await release_lease.wait()

    async def block_dispatch(*_args, **_kwargs) -> bool:
        dispatch_started.set()
        await release_dispatch.wait()
        return True

    runtime._asr_smart_turn_lease = _BlockingLease()
    runtime.handle_input_transcript.side_effect = block_dispatch
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final(
            "短语音",
            runtime._asr_session_epoch,
            "qwen",
        )
    )

    await release_started.wait()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._independent_asr_user_turn_active() is True

    release_lease.set()
    await dispatch_started.wait()
    assert runtime._independent_asr_user_turn_active() is True

    release_dispatch.set()
    await final_task
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._independent_asr_user_turn_active() is False

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

async def test_game_takeover_clears_provider_audio_and_suspends_lifecycle() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = type("Detector", (), {"reset": AsyncMock()})()
    runtime._asr_detector = detector

    await runtime._suspend_independent_voice_input_for_game()

    asr.close.assert_awaited_once_with()
    detector.reset.assert_awaited_once_with()
    assert runtime._asr_lifecycle.snapshot.state.value == "suspended"

    await runtime._resume_independent_voice_input_after_game()
    assert runtime._asr_lifecycle.snapshot.state.value == "local_listen"

async def test_game_takeover_pre_abort_window_rejects_stale_core_turn(
    monkeypatch,
) -> None:
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
    runtime.session.abandon_external_voice_turn = MagicMock()
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
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_session.close = AsyncMock()
    runtime._asr_session.signal_user_activity_end = AsyncMock()

    preview_clear_started = asyncio.Event()
    release_preview_clear = asyncio.Event()

    async def block_preview_clear(payload: dict[str, object]) -> None:
        if (
            payload.get("type") == "user_transcript_preview"
            and payload.get("text") == ""
        ):
            preview_clear_started.set()
            await release_preview_clear.wait()

    runtime.websocket = SimpleNamespace(
        send_json=AsyncMock(side_effect=block_preview_clear),
    )
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "openai")
    sealed = runtime._asr_runtime._asr_sealed_turn_token
    assert sealed is not None
    stale_ingress = sealed.turn.ingress

    await runtime._handle_independent_asr_final("", epoch, "openai")
    await asyncio.wait_for(preview_clear_started.wait(), 1)

    takeover = asyncio.create_task(
        runtime._handle_voice_input_control("game_takeover", 2)
    )
    endpoint: asyncio.Task[None] | None = None
    try:
        for _ in range(100):
            if runtime._voice_lease_owner == "game":
                break
            await asyncio.sleep(0)
        assert runtime._voice_lease_owner == "game"
        assert runtime._voice_lease_generation == 2
        assert takeover.done() is False

        await runtime._handle_independent_asr_activity(
            SpeechActivityEvent.SPEECH_STARTED,
            epoch,
        )
        prepared_token = runtime._asr_runtime._asr_partial_turn_token
        endpoint = asyncio.create_task(
            runtime._handle_independent_asr_endpoint(epoch)
        )
        for _ in range(100):
            if endpoint.done():
                break
            await asyncio.sleep(0)
        await runtime._handle_independent_asr_final(
            "stale core audio",
            epoch,
            "openai",
        )
        await runtime._asr_runtime.wait_transcript_idle()

        assert stale_ingress.lease_generation == 1
        assert prepared_token is None
        route_transcript.assert_not_awaited()
    finally:
        release_preview_clear.set()
        pending = [takeover]
        if endpoint is not None:
            pending.append(endpoint)
        results = await asyncio.wait_for(asyncio.gather(*pending), 1)
        assert results[0] is True
        await runtime._voice_input_registry.wait_idle()

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{epoch}-1"
    )

async def test_hot_swap_cache_replay_preserves_rnnoise_evidence() -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
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
    token = runtime._capture_ingress_token()

    await runtime._process_microphone_stream_data(
        {
            "input_type": "audio",
            "sample_rate_hz": 16_000,
            "data": [1] * 160,
        },
        ingress_token=token,
        captured_at=2345.6,
    )

    assert len(runtime.hot_swap_audio_cache) == 1
    route_audio.assert_not_awaited()
    runtime.is_hot_swap_imminent = False
    await runtime._flush_hot_swap_audio_cache()

    route_audio.assert_awaited_once_with(
        processed.pcm16,
        sample_rate_hz=processed.sample_rate_hz,
        speech_probability=processed.speech_probability,
        rnnoise_available=processed.rnnoise_available,
        rnnoise_evidence=evidence,
        ingress_token=token,
        received_at=ANY,
        captured_at=2345.6,
    )

async def test_hard_mute_overrides_game_consumer(monkeypatch) -> None:
    runtime = _Runtime()
    monkeypatch.setattr(
        "main_logic.voice_input.consumers.game.is_game_route_active",
        lambda _name: True,
    )

    await runtime._handle_voice_input_control(
        "lease_sync",
        1,
        owner="game",
        hard_muted=True,
        focus_suppressed=False,
    )

    assert runtime._voice_input_accepts_pcm() is False
    assert runtime._voice_input_suppression_reasons == {"hard_mute"}
    assert runtime._omni_mic_audio_bytes == 0

async def test_required_activation_revoke_retires_runtime_before_async_replace() -> None:
    runtime = _Runtime()
    factory = _CoreActivationFactory()
    old_runtime = SimpleNamespace(close=AsyncMock())
    runtime._voice_session_activation_factory = factory
    runtime._voice_session_activation_runtime = old_runtime
    before_permission = runtime._voice_session_activation_permission_revision

    token = runtime.require_voice_session_activation(
        activation_generation="required-empty",
    )

    assert token == runtime._voice_session_activation_policy_revision
    assert runtime._voice_session_activation_required is True
    assert runtime._voice_session_activation_degraded is True
    assert runtime._voice_session_activation_factory is None
    assert runtime._voice_session_activation_runtime is None
    assert factory.closed is True
    assert runtime._voice_session_activation_permission_revision > before_permission
    await asyncio.sleep(0)
    old_runtime.close.assert_awaited_once()

async def test_native_idle_reconnect_failure_keeps_replay_for_one_safe_retry() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.is_active = True
    runtime.session.instructions = "stay in character"
    runtime.session._connection_generation = 9
    delivered: list[bytes] = []
    reconnect_attempts = 0

    async def stream_audio(pcm16: bytes) -> None:
        if len(delivered) == len(frames):
            # Keep the final live frame pending for one scheduling turn. A wait
            # for only the 15 replay frames would return before delivery ends.
            await asyncio.sleep(0)
        delivered.append(pcm16)

    async def reconnect(_instructions: str, *, native_audio: bool) -> None:
        nonlocal reconnect_attempts
        assert native_audio is True
        reconnect_attempts += 1
        if reconnect_attempts == 1:
            raise RuntimeError("still offline")
        runtime.session._connection_generation += 1

    runtime.session.stream_audio = AsyncMock(side_effect=stream_audio)
    runtime.session.connect = AsyncMock(side_effect=reconnect)
    runtime.session.close = AsyncMock()
    runtime.message_handler_task = None
    runtime._restart_message_handler_after_session_reconnect = AsyncMock(
        return_value=True
    )
    factory = _CoreActivationFactory()
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
    )

    frames = [b"\xd0\x07" * 1_600 for _ in range(15)]
    await runtime._route_microphone_audio(frames[0], sample_rate_hz=16_000)
    await asyncio.sleep(0)
    generation = factory.runtimes[0].generation
    runtime.session_closed_by_server = True
    runtime._native_activation_idle_reconnect_identity = (generation, 9)
    for frame in frames[1:]:
        await runtime._route_microphone_audio(frame, sample_rate_hz=16_000)

    for _ in range(100):
        if (
            runtime.session.close.await_count
            and factory.runtimes[0]._output_task is None
        ):
            break
        await asyncio.sleep(0)
    assert reconnect_attempts == 1
    assert delivered == []
    assert factory.runtimes[0].state is ActivationState.REPLAYING
    runtime.session.close.assert_awaited_once_with()

    retry_frame = b"\xd1\x07" * 1_600
    await runtime._route_microphone_audio(retry_frame, sample_rate_hz=16_000)
    await _wait_for_activation_output(
        lambda: len(delivered),
        expected_count=len(frames) + 1,
    )

    assert reconnect_attempts == 2
    assert delivered == [*frames, retry_frame]
    assert factory.runtimes[0].state is ActivationState.ACTIVE
    assert factory.scorers[0].calls == 1
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

async def test_voice_session_activation_keeps_original_monotonic_capture_time() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    captured_frames = []

    class _CapturingRuntime:
        state = ActivationState.WAITING

        def __init__(self, generation) -> None:
            self.generation = generation

        async def prepare(self):
            return None

        async def feed(self, frame, *, voice_activity: bool):
            assert voice_activity is True
            captured_frames.append(frame)
            return None

        async def close(self) -> None:
            return None

    class _CapturingFactory:
        activation_generation = "profile"

        def create(self, generation, output, *, status_callback=None):
            return _CapturingRuntime(generation)

        def close(self) -> None:
            return None

    await runtime.set_voice_session_activation_factory(
        _CapturingFactory(),
        activation_generation="profile",
    )

    await runtime._route_microphone_audio(
        b"\xd0\x07" * 1_600,
        sample_rate_hz=16_000,
        received_at=29.9,
        captured_at=1_725_000_000.0,
    )

    assert len(captured_frames) == 1
    assert captured_frames[0].captured_at == 29.9
    assert captured_frames[0].context.captured_at == 1_725_000_000.0

async def test_voice_session_activation_route_change_retires_old_authority() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    factory = _CoreActivationFactory()
    await runtime.set_voice_session_activation_factory(
        factory,
        activation_generation="profile",
    )

    pcm16 = b"\xd0\x07" * 1_600
    await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    await asyncio.sleep(0)
    runtime._set_microphone_route("independent")
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
    )
    await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    await asyncio.sleep(0)

    assert len(factory.runtimes) == 2
    assert factory.runtimes[0].generation != factory.runtimes[1].generation
    assert factory.scorers[0].closed is True
    runtime.session.stream_audio.assert_not_awaited()
    runtime._asr_runtime.submit.assert_not_awaited()
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

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

async def test_speech_started_interrupts_and_prepares_turn_once() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )

    runtime.session.handle_interruption.assert_awaited_once_with()
    runtime.handle_new_message.assert_awaited_once_with()
    assert runtime._asr_turn_prepared is True

async def test_game_takeover_during_core_prepare_drops_stale_message() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()

    async def block_prepare(*, turn_id: str) -> None:
        prepare_started.set()
        await release_prepare.wait()

    runtime.session.prepare_external_voice_turn = AsyncMock(side_effect=block_prepare)
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime._asr_runtime.suspend = AsyncMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    prepare_task = asyncio.create_task(runtime._prepare_core_voice_turn(token))
    await asyncio.wait_for(prepare_started.wait(), 1)

    await runtime._suspend_independent_voice_input_for_game()
    release_prepare.set()

    assert await asyncio.wait_for(prepare_task, 1) is False
    runtime.handle_new_message.assert_not_awaited()
    external_turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    assert runtime.session.abandon_external_voice_turn.call_args_list == [
        call(external_turn_id),
    ]

async def test_endpoint_reprepares_turn_after_transient_prepare_rejection() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime.handle_new_message.side_effect = [RuntimeError("transient"), None]
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    assert runtime._asr_turn_prepared is False
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE

    await runtime._handle_independent_asr_endpoint(epoch)

    # The retry-able recovery path: the endpoint re-runs preparation, so the
    # interruption/external-turn pause is established before the seal and the
    # provider final is injected normally.
    assert runtime._asr_turn_prepared is True
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert runtime.handle_new_message.await_count == 2

    await runtime._handle_independent_asr_final("hello", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_awaited_once()
    runtime.session.create_response.assert_awaited_once_with("hello")

async def test_registry_prepare_rejection_releases_keyed_external_turn_pause() -> (
    None
):
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime.handle_new_message = AsyncMock(side_effect=RuntimeError("history failed"))
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)

    assert await runtime._prepare_voice_input_turn(token) is False

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_final_waits_for_shared_swap_barrier_then_uses_promoted_session() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    prepared_session = runtime.session
    prepared_session.abandon_external_voice_turn = MagicMock()

    replacement = type("Omni", (), {})()
    replacement.submit_external_voice_turn = AsyncMock()
    replacement.prepare_external_voice_turn = AsyncMock()
    replacement.abandon_external_voice_turn = MagicMock()

    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    await runtime._core_voice_session_swap_lock.acquire()
    dispatch = asyncio.create_task(
        runtime._dispatch_core_asr_transcript(
            VoiceTranscriptEvent(
                turn_token=token,
                provider="qwen",
                text="after swap",
            ),
            session_ref=prepared_session,
        )
    )
    try:
        await asyncio.sleep(0)
        assert dispatch.done() is False
        runtime.session = replacement
    finally:
        runtime._core_voice_session_swap_lock.release()
    await dispatch

    replacement.prepare_external_voice_turn.assert_awaited_once_with(
        turn_id=f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )
    replacement.submit_external_voice_turn.assert_awaited_once_with(
        "after swap",
        turn_id=f"asr-{token.ingress.session_epoch}-{token.turn_id}",
    )

async def test_transcript_dispatch_failure_releases_keyed_external_turn_pause() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    runtime.handle_input_transcript.side_effect = RuntimeError("history failed")
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    event = VoiceTranscriptEvent(
        turn_token=token,
        provider="qwen",
        text="hello",
    )

    with pytest.raises(RuntimeError, match="history failed"):
        await runtime._dispatch_core_asr_transcript(event)

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_suspend_advances_runtime_barrier_before_waiting_for_registry_cancel() -> (
    None
):
    runtime = _Runtime()
    order: list[str] = []
    runtime._invalidate_voice_pcm_sync = MagicMock(
        side_effect=lambda _reason: order.append("invalidate")
    )
    runtime._asr_runtime.suspend = AsyncMock(
        side_effect=lambda _reason: order.append("suspend")
    )
    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=lambda: order.append("wait_idle")
    )

    await runtime._suspend_independent_asr("game_takeover")

    assert order == ["suspend", "invalidate", "wait_idle"]

async def test_registry_cancellation_abandons_the_prepared_session_after_swap() -> (
    None
):
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    original_session = runtime.session
    original_session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    assert await runtime._prepare_voice_input_turn(token) is True

    replacement = type("Omni", (), {})()
    replacement.abandon_external_voice_turn = MagicMock()
    runtime.session = replacement
    assert runtime._voice_input_registry.invalidate_utterance(
        token,
        reason="session_hot_swap",
    )
    await runtime._voice_input_registry.wait_idle()

    original_session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}",
    )
    replacement.abandon_external_voice_turn.assert_not_called()

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

async def test_native_audio_failure_log_is_rate_limited(monkeypatch) -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("native")
    runtime.session_closed_by_server = False
    runtime.last_audio_send_error_time = 0.0
    runtime.audio_error_log_interval = 2.0
    runtime.session.stream_audio = AsyncMock(side_effect=RuntimeError("send failed"))
    log_error = MagicMock()
    monkeypatch.setattr(core_asr_runtime_module.logger, "error", log_error)

    await runtime._route_microphone_audio(b"\x01\x00", sample_rate_hz=16_000)
    await runtime._route_microphone_audio(b"\x01\x00", sample_rate_hz=16_000)

    assert runtime.session.stream_audio.await_count == 2
    log_error.assert_called_once()

async def test_stale_pending_activation_discards_confirmed_candidate() -> None:
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
    assert lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert lifecycle.has_pending_turn is True

    await runtime._activate_pending_independent_turn(epoch)

    assert lifecycle.pending_turn_bytes == 0
    assert lifecycle.has_pending_turn is False
    assert runtime._asr_pending_detector_candidate is None

async def test_candidate_pause_defers_overlap_onset_without_ghost_wake() -> None:
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
    # Local VAD then observes a pause: the provider final that follows may be
    # the current utterance ending, so replaying the onset at that final would
    # wake a ghost turn. The onset converts into a completed-overlap credit
    # that only a later provider endpoint in WARM_IDLE can redeem.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    assert runtime._asr_overlap_onset_token is None
    assert runtime._asr_overlap_completed_turns == 1

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("hello", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # No second endpoint arrived, so the credit must not wake anything.
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_turn_prepared is False
    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["hello"]
    assert runtime.handle_new_message.await_count == 1

async def test_stale_completed_overlap_is_dropped_at_next_endpoint() -> None:
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

    # The audio generation moves on before the delayed final, so the credit
    # belongs to a stale ingress and must not wake a replacement turn.
    component = runtime._asr_runtime
    component._asr_audio_generation += 1
    component._asr_current_ingress_token = runtime._capture_ingress_token()

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    await runtime._handle_independent_asr_endpoint(epoch)
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_turn_prepared is False
    assert runtime._asr_overlap_completed_turns == 0
    await runtime._handle_independent_asr_final("ghost", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # Both the overlap credit and the final belong to the superseded full
    # VoiceTurnToken once audio_generation rotates.
    runtime.handle_input_transcript.assert_not_awaited()
    assert runtime.handle_new_message.await_count == 1

async def test_transport_only_close_enters_deep_sleep_without_closing_detector() -> (
    None
):
    runtime = _Runtime()
    asr = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    detector = type("Detector", (), {"close": AsyncMock()})()
    runtime._asr_detector = detector

    await runtime._close_transport_only()

    assert runtime._asr_session is None
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DEEP_SLEEP
    assert runtime._asr_detector is detector
    assert runtime._asr_route_mode == "independent"
    asr.close.assert_awaited_once_with()
    detector.close.assert_not_awaited()

async def test_stale_transport_expiry_never_closes_successor(
    replacement: str,
) -> None:
    runtime = _Runtime()
    original_session = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = original_session
    runtime._asr_route_mode = "independent"
    policy = replace(resolve_provider_policy("openai", "provider"), warm_transport_ms=0)
    original_lifecycle = VoiceInputLifecycleController(
        provider_policy=policy,
        shadow_mode=False,
    )
    original_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    original_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    original_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    original_lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    original_lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    runtime._asr_lifecycle = original_lifecycle
    runtime._schedule_transport_warm_expiry(
        runtime._asr_session_epoch,
        expected_state=VoiceLifecycleState.WARM_IDLE,
    )

    successor_session = type("Asr", (), {"close": AsyncMock()})()
    if replacement == "epoch":
        runtime._asr_session_epoch += 1
    elif replacement == "lifecycle":
        successor_lifecycle = VoiceInputLifecycleController(
            provider_policy=policy,
            shadow_mode=False,
        )
        successor_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
        runtime._asr_lifecycle = successor_lifecycle
    elif replacement == "session":
        runtime._asr_session = successor_session
    elif replacement == "transport":
        original_lifecycle.invalidate_transport()
    else:
        original_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
        original_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)

    expected_current_session = (
        successor_session if replacement == "session" else original_session
    )
    assert runtime._asr_session is expected_current_session
    expiry = runtime._asr_warm_expiry_task
    assert expiry is not None
    await asyncio.wait_for(expiry, 1)

    original_session.close.assert_not_awaited()
    expected_current_session.close.assert_not_awaited()

async def test_deep_sleep_speech_reconnects_and_flushes_pending_audio() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = "qwen"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.SPEECH_CONFIRMED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.TURN_SEALED)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    runtime._asr_lifecycle.transition(VoiceLifecycleEvent.WARM_EXPIRED)
    detector = _ReadyDetector()
    detector.feed = AsyncMock(
        return_value=DetectorFeedResult((SpeechActivityEvent.SPEECH_STARTED,), True)
    )
    runtime._asr_detector = detector
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
    runtime._asr_transport_selection = _selection("qwen")

    await runtime._route_microphone_audio(
        b"\x03\x00" * 160,
        sample_rate_hz=16_000,
    )
    await asyncio.wait_for(connect_started.wait(), 1)

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
    assert runtime._asr_lifecycle.pending_connect_bytes == 320
    connect_release.set()
    assert runtime._asr_transport_task is not None
    await runtime._asr_transport_task

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    new_asr.connect.assert_awaited_once_with()
    new_asr.stream_audio.assert_awaited_once_with(
        b"\x03\x00" * 160,
        sample_rate_hz=16_000,
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

async def test_legacy_audio_session_authorization_is_one_shot() -> None:
    runtime = _Runtime()
    runtime._asr_runtime.abort = AsyncMock()

    assert runtime._begin_voice_input_connection("legacy-socket") is True
    assert await runtime._ensure_voice_input_session_authorized("legacy-socket") is True
    assert runtime._voice_lease_generation == 0
    assert runtime._voice_lease_synchronized is True
    assert runtime._voice_lease_owner == "core"
    assert runtime._voice_lease_hard_muted is False
    assert runtime._voice_lease_focus_suppressed is False
    assert runtime._voice_input_accepts_pcm() is True
    runtime._asr_runtime.abort.assert_awaited_once_with("legacy_session_start")

    runtime._asr_runtime.abort.reset_mock()
    assert await runtime._ensure_voice_input_session_authorized("legacy-socket") is True
    assert runtime._voice_lease_generation == 0
    runtime._asr_runtime.abort.assert_not_awaited()

async def test_game_owner_and_hard_mute_remain_simultaneously_authoritative() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")

    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="game",
            hard_muted=True,
            focus_suppressed=False,
        )
        is True
    )

    assert runtime._voice_lease_owner == "game"
    assert runtime._voice_lease_hard_muted is True
    assert runtime._voice_input_suppression_reasons == {"game", "hard_mute"}
    assert runtime._voice_input_accepts_pcm() is False
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.SUSPENDED

async def test_identical_text_in_consecutive_turns_is_delivered_twice() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch

    for _ in range(2):
        await runtime._handle_independent_asr_activity(
            SpeechActivityEvent.SPEECH_STARTED,
            epoch,
        )
        await runtime._handle_independent_asr_endpoint(epoch)
        await runtime._handle_independent_asr_final("嗯", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["嗯", "嗯"]
    assert [
        call.args[0] for call in runtime.session.create_response.await_args_list
    ] == [
        "嗯",
        "嗯",
    ]

async def test_three_pending_finals_recover_without_request_multiplication() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    for text in ("first", "second", "third"):
        await runtime._handle_independent_asr_activity(
            SpeechActivityEvent.SPEECH_STARTED,
            epoch,
        )
        await runtime._handle_independent_asr_endpoint(epoch)
        await runtime._handle_independent_asr_final(text, epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime.session.handle_interruption.await_count == 3
    assert runtime.handle_new_message.await_count == 3
    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == [
        "first",
        "second",
        "third",
    ]
    assert [
        call.args[0] for call in runtime.session.create_response.await_args_list
    ] == [
        "first",
        "second",
        "third",
    ]

async def test_cancelled_core_close_keeps_detached_cleanup_owned() -> None:
    runtime = _Runtime()
    pipeline_close_started = asyncio.Event()
    release_pipeline_close = asyncio.Event()
    registry_wait_started = asyncio.Event()
    release_registry_wait = asyncio.Event()

    async def block_pipeline_close() -> None:
        pipeline_close_started.set()
        await release_pipeline_close.wait()

    async def block_registry_wait() -> None:
        registry_wait_started.set()
        await release_registry_wait.wait()

    pipeline = SimpleNamespace(close=AsyncMock(side_effect=block_pipeline_close))
    runtime._voice_input_audio_pipeline = pipeline
    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=block_registry_wait
    )
    runtime._asr_runtime.close = AsyncMock()

    closing = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await asyncio.wait_for(pipeline_close_started.wait(), 1)
    await asyncio.wait_for(registry_wait_started.wait(), 1)
    replacement = runtime._voice_input_audio_pipeline
    cleanup_tasks = set(runtime._core_asr_cleanup_tasks)

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert replacement is not pipeline
    assert all(task.cancelled() is False for task in cleanup_tasks)
    release_pipeline_close.set()
    release_registry_wait.set()
    await asyncio.wait_for(asyncio.gather(*cleanup_tasks), 1)

    pipeline.close.assert_awaited_once_with()
    runtime._asr_runtime.close.assert_awaited_once_with()

async def test_start_pipeline_construction_failure_preserves_audio_contract(
    monkeypatch, initial_nr: bool,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime._close_independent_asr = AsyncMock()
    await runtime.apply_voice_input_noise_reduction(initial_nr)
    original = runtime._voice_input_audio_pipeline
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={
            "independentAsrEnabled": False,
            "noiseReductionEnabled": not initial_nr,
        }),
    )
    try:
        with monkeypatch.context() as failing:
            failing.setattr(
                core_asr_runtime_module,
                "VoiceInputAudioPipeline",
                MagicMock(side_effect=RuntimeError("pipeline construction failed")),
            )
            with pytest.raises(RuntimeError, match="pipeline construction failed"):
                await runtime._start_independent_asr_if_enabled("audio")

        assert runtime._voice_input_audio_pipeline is original
        assert runtime._voice_input_noise_reduction_enabled is initial_nr
        assert runtime._asr_route_mode == "blocked"
        frame = await original.process(b"\x01\x00" * 160, sample_rate_hz=16_000)
        assert frame.pcm16 == b"\x01\x00" * 160

        await runtime._start_independent_asr_if_enabled("audio")
        assert runtime._voice_input_audio_pipeline.nr_enabled is not initial_nr
        assert runtime._voice_input_noise_reduction_enabled is not initial_nr
        with pytest.raises(RuntimeError, match="VOICE_AUDIO_PIPELINE_CLOSED"):
            await original.process(b"\x01\x00" * 160, sample_rate_hz=16_000)
    finally:
        await runtime._voice_input_audio_pipeline.close()
        await asyncio.gather(*runtime._core_asr_cleanup_tasks, return_exceptions=True)

async def test_cancelled_noise_reduction_swap_keeps_pipeline_cleanup_owned() -> None:
    runtime = _Runtime()
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

    applying = asyncio.create_task(
        runtime.apply_voice_input_noise_reduction(False)
    )
    await asyncio.wait_for(close_started.wait(), 1)
    replacement = runtime._voice_input_audio_pipeline
    cleanup = next(
        task
        for task in runtime._core_asr_cleanup_tasks
        if task.get_name() == "core-voice-input-pipeline-close"
    )

    applying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await applying

    assert replacement is not stale_pipeline
    assert replacement.nr_enabled is False
    assert cleanup.cancelled() is False
    release_close.set()
    await asyncio.wait_for(cleanup, 1)
    stale_pipeline.close.assert_awaited_once_with()

async def test_asr_backpressure_reports_specific_blocking_status() -> None:
    runtime = _Runtime()
    blocking_status_sent = asyncio.Event()

    async def record_status(message: str) -> None:
        if "ASR_STREAM_BACKPRESSURE" in message:
            blocking_status_sent.set()

    runtime.send_status.side_effect = record_status
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock(
        side_effect=RuntimeError("ASR_STREAM_BACKPRESSURE: queue full")
    )
    asr.close = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    await _install_active_smart_turn(runtime, "qwen")

    await runtime._route_microphone_audio(
        b"\x00\x00" * 160,
        sample_rate_hz=16_000,
    )
    await runtime._asr_audio_dispatcher.wait_idle()
    await asyncio.wait_for(blocking_status_sent.wait(), 1)

    assert "ASR_STREAM_BACKPRESSURE" in runtime.send_status.await_args.args[0]
    assert runtime._asr_route_mode == "blocked"

async def test_start_forwards_core_user_language_to_session_builder(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.user_language = "ja"

    kwargs = await _start_bridge_and_capture_builder_call(monkeypatch, runtime)

    assert kwargs["user_language"] == "ja"

async def test_soniox_connect_failure_retries_same_selection_before_audio(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    soniox_selection = _selection("soniox", "provider")
    primary_resolver = MagicMock(return_value=soniox_selection)
    forbidden_core_resolver = MagicMock(
        side_effect=AssertionError("Soniox recovery must not resolve another provider")
    )
    save_settings = MagicMock(
        side_effect=AssertionError("Provider recovery must not rewrite user settings")
    )
    sleep = AsyncMock()
    sessions = []
    for side_effect in (
        RuntimeError("provider detail 1"),
        RuntimeError("provider detail 2"),
        None,
    ):
        session = type("Soniox", (), {})()
        session.connect = AsyncMock(side_effect=side_effect)
        session.close = AsyncMock()
        sessions.append(session)
    built_selections = []
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        primary_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_core_follow_selection",
        forbidden_core_resolver,
        raising=False,
    )
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(
        preferences,
        "save_global_conversation_settings",
        save_settings,
    )

    def build_candidate(_core_type, *, selection, **_kwargs):
        assert runtime._asr_provider == "soniox"
        built_selections.append(selection)
        assert selection is soniox_selection
        return sessions[len(built_selections) - 1]

    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        build_candidate,
    )

    await runtime._start_independent_asr_if_enabled("audio")

    sessions[0].close.assert_awaited_once_with()
    sessions[1].close.assert_awaited_once_with()
    sessions[2].close.assert_not_awaited()
    for session in sessions:
        session.connect.assert_awaited_once_with()
    primary_resolver.assert_called_once_with("gemini")
    forbidden_core_resolver.assert_not_called()
    save_settings.assert_not_called()
    assert built_selections == [soniox_selection] * 3
    assert [call.args for call in sleep.await_args_list] == [(0.25,), (0.5,)]
    assert runtime._asr_session is sessions[2]
    assert runtime._asr_provider == "soniox"
    assert runtime._asr_transport_selection is soniox_selection
    assert runtime._asr_lifecycle.provider_policy.endpoint_authority == "provider"
    assert runtime._asr_route_mode == "independent"
    assert "provider detail" not in str(runtime.send_status.await_args_list)
    assert "ASR_INDEPENDENT_PROVIDER_UNAVAILABLE" not in str(
        runtime.send_status.await_args_list
    )

async def test_reconnected_start_callback_survives_abort_start_generation_change(
    monkeypatch,
) -> None:
    runtime, sessions, callbacks, _detector = (
        await _start_runtime_with_callback_candidates(monkeypatch)
    )
    component = runtime._asr_runtime
    original_start_generation = component._asr_start_generation

    await component.abort("hard_mute")

    assert component._asr_start_generation > original_start_generation
    assert component._asr_session is None
    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN
    sessions[0].close.assert_awaited_once_with()

    await component._restart_transport(max_attempts=1)

    assert len(callbacks) == 2
    assert component._asr_session is sessions[1]
    updated_ingress = runtime._capture_ingress_token()
    component._asr_current_ingress_token = updated_ingress
    old_activity = callbacks[0]["on_speech_activity"]
    new_activity = callbacks[1]["on_speech_activity"]
    assert callable(old_activity)
    assert callable(new_activity)

    await old_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.LOCAL_LISTEN

    await new_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert component._asr_current_ingress_token == updated_ingress
    assert component._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    sessions[1].close.assert_not_awaited()

async def test_restart_explicit_attempt_override_beats_policy(monkeypatch) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    sleep = AsyncMock()
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
    candidates = _install_failing_restart_candidates(
        runtime, "soniox", failure_count=1
    )

    await runtime._restart_transport(max_attempts=1)
    while runtime._asr_runtime._asr_close_tasks:
        await asyncio.gather(
            *tuple(runtime._asr_runtime._asr_close_tasks),
            return_exceptions=True,
        )

    assert len(candidates) == 1
    candidates[0].connect.assert_awaited_once_with()
    sleep.assert_not_awaited()

async def test_adopted_restart_exception_fails_closed_without_retry(
    monkeypatch,
) -> None:
    runtime, sessions, callbacks, detector = (
        await _start_runtime_with_callback_candidates(monkeypatch)
    )
    component = runtime._asr_runtime
    lifecycle = component._asr_lifecycle
    assert lifecycle is not None
    started_epoch = component._asr_session_epoch

    await component._close_transport_only()

    assert lifecycle.snapshot.state is VoiceLifecycleState.DEEP_SLEEP
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    assert lifecycle.snapshot.state is VoiceLifecycleState.PREWARMING
    component._asr_pending_speech_confirmed = True
    component._asr_current_ingress_token = runtime._capture_ingress_token()
    component._prepare_independent_asr_turn = AsyncMock(
        side_effect=RuntimeError("post-adoption recovery failed")
    )

    await component._restart_transport(max_attempts=3)
    while component._asr_close_tasks:
        await asyncio.gather(
            *tuple(component._asr_close_tasks),
            return_exceptions=True,
        )

    assert len(callbacks) == 2
    sessions[0].close.assert_awaited_once_with()
    sessions[1].connect.assert_awaited_once_with()
    sessions[1].close.assert_awaited_once_with()
    component._prepare_independent_asr_turn.assert_awaited_once_with(started_epoch)
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

async def test_stale_settings_failure_cannot_refence_replacement_session(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    settings_read_started = asyncio.Event()
    release_stale_read = asyncio.Event()
    read_count = 0

    async def load_settings(*, strict: bool = False) -> dict:
        nonlocal read_count
        assert strict is True
        read_count += 1
        if read_count == 1:
            settings_read_started.set()
            await release_stale_read.wait()
            raise OSError("stale settings read failed")
        return {"independentAsrEnabled": False}

    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        load_settings,
    )
    stale_start = asyncio.create_task(
        runtime._start_independent_asr_if_enabled(
            "audio",
            handshake_override=True,
        )
    )
    await settings_read_started.wait()

    replacement_session = MagicMock()
    replacement_session.set_visual_delivery_mode = MagicMock()
    replacement_session.block_raw_visual_delivery = MagicMock()
    runtime.session = replacement_session
    runtime.core_api_type = "gemini"
    await runtime._start_independent_asr_if_enabled(
        "audio",
        handshake_override=False,
    )
    assert runtime._asr_route_mode == "native"

    release_stale_read.set()
    await stale_start

    delivered_modes = [
        getattr(call.args[0], "value", call.args[0])
        for call in replacement_session.set_visual_delivery_mode.call_args_list
    ]
    assert delivered_modes
    assert set(delivered_modes) == {"native"}

async def test_partial_preview_keeps_prepared_token_and_rejects_after_abort() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    on_partial = AsyncMock()
    runtime._asr_runtime._callbacks = replace(
        runtime._asr_runtime._callbacks,
        on_partial=on_partial,
    )
    await _install_active_smart_turn(runtime)
    epoch = runtime._asr_session_epoch
    captured_token = runtime._asr_runtime._asr_partial_turn_token
    assert captured_token is not None
    assert runtime._activate_asr_audio_dispatcher(
        runtime._asr_lifecycle,
        captured_token,
    )

    await runtime._send_independent_asr_preview("current", epoch)

    event = on_partial.await_args.args[0]
    assert event.turn_token is captured_token
    assert event.session_epoch == epoch
    on_partial.reset_mock()

    runtime._asr_audio_dispatcher.abort(captured_token)
    await runtime._send_independent_asr_preview("late", epoch)

    on_partial.assert_not_awaited()

async def test_hot_swap_replaces_asr_before_cached_audio_for_new_core() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "glm"
    runtime.input_mode = "audio"
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = "gemini"
    runtime._independent_asr_route_key = "gemini"
    runtime._start_independent_asr_if_enabled = AsyncMock()

    await runtime._reconcile_independent_asr_after_core_change()

    runtime._start_independent_asr_if_enabled.assert_awaited_once_with(
        "audio",
        preserve_hot_swap_audio=True,
    )

async def test_blocked_replacement_session_preserves_external_visual_policy_and_fence() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._set_microphone_route("blocked")
    replacement_session = type("ReplacementOmni", (), {})()
    replacement_session.set_visual_delivery_mode = MagicMock()
    replacement_session.block_raw_visual_delivery = MagicMock()
    runtime.session = replacement_session

    runtime._set_microphone_route("blocked")

    replacement_session.set_visual_delivery_mode.assert_not_called()
    replacement_session.block_raw_visual_delivery.assert_called()

async def test_start_session_handshake_false_overrides_persisted_enabled(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    start_mock = AsyncMock()
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    runtime.set_independent_asr_handshake(False)
    await runtime._start_independent_asr_if_enabled("audio")

    start_mock.assert_not_awaited()
    assert runtime._asr_route_mode == "native"

async def test_connect_budget_does_not_block_a_free_native_route(
    monkeypatch,
) -> None:
    # Codex P2. The budget bounds the PROVIDER CONNECT, nothing else. A request
    # whose handshake disables independent ASR settles on native without talking
    # to anyone, so refusing it over a connect budget would leave the route on
    # its blocked placeholder and abort a microphone start that had nothing to
    # wait for.
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
        handshake_override=False,
        connect_budget_seconds=0.0,
    )

    assert runtime._asr_route_mode == "native"
    start_mock.assert_not_awaited()

async def test_resource_optimization_handshake_malformed_falls_back_to_persisted(
    monkeypatch,
    malformed,
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

    runtime.set_voice_input_resource_optimization_handshake(malformed)
    await runtime._start_independent_asr_if_enabled("audio")

    assert start_mock.await_args.kwargs["resource_optimization_enabled"] is True

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

async def test_pipeline_failure_still_revokes_after_a_bare_pipeline_swap() -> None:
    """A replacement that does not end the route must not skip the revoke.

    The mirror image of the case above. Replacing the pipeline clears
    ``_voice_input_pipeline_failed`` -- that is all a noise-reduction toggle
    does -- but it neither unblocks the route nor revokes the lease, so
    reading it as "someone else owns this failure now" leaves the microphone
    blocked forever with the lease still held. That is the race commit
    94c26715 was written for, and it is why the notify phase fences on the
    failure's own token instead: a replacement that genuinely retires this
    failure (a start, a close) advances the route operation generation, which
    ``_fail_closed_voice_route`` checks on its own.
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

    async def delayed_abort(_reason: str) -> None:
        abort_started.set()
        await release_abort.wait()

    runtime._asr_runtime.abort = AsyncMock(side_effect=delayed_abort)
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=runtime._capture_ingress_token(),
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    runtime._voice_input_audio_pipeline = SimpleNamespace(
        process=AsyncMock(),
        close=AsyncMock(),
    )
    runtime._voice_input_pipeline_failed = False

    release_abort.set()
    await asyncio.wait_for(failure, 1)

    runtime.send_status.assert_awaited()
    assert runtime._voice_lease_connection_id == ""
    assert runtime._voice_lease_owner == "none"

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

async def test_replaced_audio_pipeline_late_failure_is_silent() -> None:
    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
    runtime.is_flushing_hot_swap_cache = False
    runtime.session.stream_audio = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._asr_runtime.abort = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def fail_late(*_args, **_kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("old pipeline failed")

    old_pipeline = runtime._voice_input_audio_pipeline
    old_pipeline.process = AsyncMock(side_effect=fail_late)
    token = runtime._capture_ingress_token()
    processing = asyncio.create_task(
        runtime._process_microphone_stream_data(
            {
                "input_type": "audio",
                "sample_rate_hz": 48_000,
                "data": [1] * 480,
            },
            ingress_token=token,
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    runtime._voice_input_audio_pipeline = type(
        "ReplacementPipeline",
        (),
        {"process": AsyncMock(), "close": AsyncMock()},
    )()
    runtime._set_microphone_route("blocked")
    runtime._set_microphone_route("independent")
    release.set()
    await asyncio.wait_for(processing, 1)

    runtime._asr_runtime.abort.assert_not_awaited()
    runtime.session.stream_audio.assert_not_awaited()
    runtime.send_status.assert_not_awaited()
    assert runtime._voice_input_pipeline_failed is False

async def test_old_smart_turn_release_cannot_clear_replacement_lease() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "glm")
    lifecycle = runtime._asr_lifecycle
    detector = runtime._asr_detector
    assert lifecycle is not None
    assert detector is not None
    release_started = asyncio.Event()
    release_old_lease = asyncio.Event()

    class BlockingLease:
        token = object()

        async def release(self) -> None:
            release_started.set()
            await release_old_lease.wait()

    old_lease = BlockingLease()
    runtime._asr_smart_turn_lease = old_lease
    prepare_task = asyncio.create_task(
        runtime._asr_runtime._ensure_smart_turn_ready(
            lifecycle,
            runtime._asr_session_epoch,
        )
    )
    await asyncio.wait_for(release_started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "glm"
    )
    new_lease = _TestSmartTurnLease(
        runtime._asr_runtime._capture_turn_token(new_lifecycle)
    )
    runtime._asr_smart_turn_lease = new_lease
    release_old_lease.set()

    assert await asyncio.wait_for(prepare_task, 1) is False
    assert runtime._asr_smart_turn_lease is new_lease
    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert new_lease.released is False

async def test_stale_connect_failure_cannot_fail_new_generation() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    lifecycle.transition(VoiceLifecycleEvent.SOFT_WAKE)
    runtime._asr_session = None
    started = asyncio.Event()
    release = asyncio.Event()
    candidate = SimpleNamespace(close=AsyncMock())

    async def connect() -> None:
        started.set()
        await release.wait()
        raise RuntimeError("old candidate failed")

    candidate.connect = AsyncMock(side_effect=connect)
    runtime._asr_session_factory = MagicMock(return_value=candidate)
    runtime._asr_transport_selection = _selection("qwen")
    old_restart = runtime._asr_runtime._ensure_transport_restart_task()
    await asyncio.wait_for(started.wait(), 1)

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
    release.set()
    await asyncio.wait_for(old_restart, 1)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert runtime._asr_session_factory is new_factory
    assert runtime._asr_transport_selection is new_selection
    assert runtime._asr_transport_task is new_transport
    candidate.close.assert_awaited_once_with()
    new_session.close.assert_not_awaited()
    runtime.send_status.assert_not_awaited()
    keep_transport.set()
    await new_transport

async def test_session_activation_resolves_asr_before_frontend_ack() -> None:
    order: list[str] = []
    manager = LLMSessionManager.__new__(LLMSessionManager)
    manager.lock = asyncio.Lock()
    manager.input_cache_lock = asyncio.Lock()
    manager.is_active = False
    manager._session_turn_count = 0
    manager.session_start_failure_count = 1
    manager.session_start_last_failure_time = 1.0
    manager._memory_error_retry_after = 1.0
    manager._session_start_circuit_open = True
    manager.pending_agent_callbacks = []
    manager._activity_tracker = type(
        "Tracker", (), {"on_voice_mode": lambda self, value: None}
    )()
    manager.is_goodbye_silent = lambda: False
    manager._drain_pending_context_appends_before_ready = AsyncMock()
    manager._flush_pending_input_data = AsyncMock()
    manager._consume_next_session_context_messages = MagicMock()
    manager._start_independent_asr_if_enabled = AsyncMock(
        side_effect=lambda _mode, **_kwargs: order.append("asr")
    )
    manager.send_session_started = AsyncMock(
        side_effect=lambda _mode, **_kwargs: order.append("started")
    )

    stop = asyncio.Event()

    class _Session:
        async def handle_messages(self) -> None:
            await stop.wait()

    manager.session = _Session()

    await LLMSessionManager._start_session_activate(
        manager,
        "audio",
        0,
        time.time(),
    )

    assert order == ["asr", "started"]
    stop.set()
    await manager.message_handler_task

async def test_unreadable_independent_setting_preserves_visual_route_on_hot_swap(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.session.set_visual_delivery_mode = MagicMock()
    runtime.session.block_raw_visual_delivery = MagicMock()
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(side_effect=OSError("preferences unavailable")),
    )
    runtime.set_independent_asr_handshake(True)

    await runtime._start_independent_asr_if_enabled("audio")
    await runtime._reconcile_independent_asr_after_core_change()

    assert runtime._asr_route_mode == "blocked"
    assert runtime._visual_route_mode == "independent"
    runtime.session.block_raw_visual_delivery.assert_called()

async def test_independent_route_without_ready_session_blocks_omni() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"is_ready": False})()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"

    assert (
        await runtime._route_microphone_audio(b"\x00\x00", sample_rate_hz=16_000)
        is True
    )
    assert runtime._asr_route_mode == "blocked"

async def test_game_takeover_during_transcript_drops_stale_core_final() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "glm")
    transcript_started = asyncio.Event()
    release_transcript = asyncio.Event()

    async def block_transcript(*_args, **_kwargs) -> bool:
        transcript_started.set()
        await release_transcript.wait()
        return True

    runtime.handle_input_transcript.side_effect = block_transcript
    runtime.session.submit_external_voice_turn = AsyncMock()
    runtime._asr_runtime.suspend = AsyncMock()
    turn_token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    event = VoiceTranscriptEvent(
        turn_token=turn_token,
        provider="glm",
        text="belongs to Core",
    )
    dispatch_task = asyncio.create_task(runtime._dispatch_core_asr_transcript(event))
    await asyncio.wait_for(transcript_started.wait(), 1)

    await runtime._suspend_independent_voice_input_for_game()
    release_transcript.set()
    await asyncio.wait_for(dispatch_task, 1)

    runtime.session.submit_external_voice_turn.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()

async def test_old_native_send_failure_cannot_close_new_session() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("native")
    runtime.session_closed_by_server = False
    send_entered = asyncio.Event()
    release_send = asyncio.Event()

    async def fail_old_send(_pcm16) -> None:
        send_entered.set()
        await release_send.wait()
        raise RuntimeError("connection closed")

    old_session = type("OldOmni", (), {})()
    old_session.stream_audio = AsyncMock(side_effect=fail_old_send)
    runtime.session = old_session
    old_token = runtime._capture_native_ingress_token()
    old_send = asyncio.create_task(
        runtime._route_microphone_audio(
            b"\x01\x00",
            sample_rate_hz=16_000,
            ingress_token=old_token,
        )
    )
    await asyncio.wait_for(send_entered.wait(), 1)

    new_session = type("NewOmni", (), {})()
    new_session.stream_audio = AsyncMock()
    runtime.session = new_session
    release_send.set()
    await asyncio.wait_for(old_send, 1)

    assert runtime.session_closed_by_server is False
    assert runtime._omni_mic_audio_bytes == 0
    await runtime._route_microphone_audio(
        b"\x02\x00",
        sample_rate_hz=16_000,
        ingress_token=runtime._capture_native_ingress_token(),
    )
    new_session.stream_audio.assert_awaited_once_with(b"\x02\x00")
    assert runtime._omni_mic_audio_bytes == 2

async def test_failure_event_only_blocks_current_generation() -> None:
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    current_epoch = runtime._asr_session_epoch

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="old-provider",
            session_epoch=current_epoch - 1,
        )
    )
    assert runtime._asr_route_mode == "independent"

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="current-provider",
            session_epoch=current_epoch,
        )
    )
    assert runtime._asr_route_mode == "blocked"

async def test_core_start_is_invalidated_by_mic_lease_transition(
    monkeypatch,
    transition: str,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    release_connect = asyncio.Event()

    class Candidate:
        def __init__(self) -> None:
            self.connect_started = asyncio.Event()
            self.is_ready = True
            self.close = AsyncMock()

        async def connect(self) -> None:
            self.connect_started.set()
            await release_connect.wait()

    candidate = Candidate()
    selection = SimpleNamespace(
        provider_key="qwen",
        endpointing_mode="provider",
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(return_value=selection),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        MagicMock(return_value=candidate),
    )
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    runtime = _Runtime()
    runtime.core_api_type = "qwen"
    runtime._voice_lease_connection_id = "connection"
    runtime._voice_lease_generation = 0

    starting = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(candidate.connect_started.wait(), 1)
    if transition == "connection_replacement":
        runtime._begin_voice_input_connection("replacement")
    elif transition == "lease_sync":
        await runtime._handle_voice_input_control(
            "lease_sync",
            1,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
    else:
        await runtime._handle_voice_input_control(transition, 1)
    release_connect.set()
    await asyncio.wait_for(starting, 1)

    assert runtime._asr_route_mode == "blocked"
    assert runtime._independent_asr_provider is None
    assert "ASR_INDEPENDENT_READY" not in str(runtime.send_status.await_args_list)
    candidate.close.assert_awaited_once_with()

async def test_core_start_survives_benign_lease_transition(monkeypatch) -> None:
    """Owner flip / mute toggle / lease bump during the settings await are
    PCM-gating changes, not route operations; they must not abort the start
    (there is no retry or failure status on that path)."""

    settings_started = asyncio.Event()
    release_settings = asyncio.Event()

    async def load_settings(**_kwargs):
        settings_started.set()
        await release_settings.wait()
        return {"independentAsrEnabled": True}

    runtime = _Runtime()
    runtime._voice_lease_owner = "none"
    runtime._voice_lease_synchronized = True
    runtime.core_api_type = "qwen"

    async def ready_start(**_kwargs) -> AsrStartResult:
        return AsrStartResult(
            AsrStartStatus.READY,
            provider="qwen",
            session_epoch=runtime._asr_session_epoch,
        )

    runtime._asr_runtime.start = AsyncMock(side_effect=ready_start)
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        load_settings,
    )

    starting = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(settings_started.wait(), 1)
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_hard_muted = True
    runtime._voice_lease_generation += 1
    release_settings.set()
    await asyncio.wait_for(starting, 1)

    runtime._asr_runtime.start.assert_awaited_once()
    assert runtime._asr_route_mode == "independent"
    assert runtime._independent_asr_provider == "qwen"

async def test_failure_cancellation_can_publish_without_notification_deadlock() -> (
    None
):
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    current_epoch = runtime._asr_session_epoch

    async def cancellation_wait_idle() -> None:
        assert runtime._asr_notification_lock.locked() is False
        await runtime._send_core_asr_status(
            AsrStatusEvent(
                code="ASR_CANCEL_CLEANUP",
                provider="plugin-consumer",
                session_epoch=current_epoch,
            )
        )

    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=cancellation_wait_idle
    )

    await asyncio.wait_for(
        runtime._handle_core_asr_failure(
            AsrFailureEvent(
                code="ASR_INDEPENDENT_FAILED",
                provider="current-provider",
                session_epoch=current_epoch,
            )
        ),
        1,
    )

    assert "ASR_CANCEL_CLEANUP" in str(runtime.send_status.await_args_list)

async def test_blocked_text_notice_commits_only_for_current_connection() -> None:
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

    assert runtime._begin_voice_input_connection("pet-window") is True
    release_send.set()
    await asyncio.wait_for(first, 1)

    assert runtime._blocked_text_mode_microphone_signal_state is None
    runtime.send_status = AsyncMock()
    await runtime._maybe_signal_blocked_text_mode_microphone()

    runtime.send_status.assert_awaited_once()
    assert runtime._blocked_text_mode_microphone_signal_state is not None

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

async def test_settings_read_failure_keeps_noise_reduction_enabled(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(side_effect=RuntimeError("settings unavailable")),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._voice_input_noise_reduction_enabled is True
    assert runtime._voice_input_audio_pipeline.nr_enabled is True

async def test_accepted_final_identity_loss_before_dispatch_abandons_turn() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "glm")
    runtime.session.abandon_external_voice_turn = MagicMock()
    component = runtime._asr_runtime
    epoch = component._asr_session_epoch
    await _start_and_seal_turn(runtime, "glm")
    sealed_turn_id = component._asr_lifecycle.snapshot.turn_id
    lease = component._asr_smart_turn_lease
    assert lease is not None

    async def bumping_release() -> None:
        component._asr_audio_generation += 1

    lease.release = bumping_release

    await runtime._handle_independent_asr_final("hello", epoch, "glm")
    await runtime._wait_asr_transcript_dispatch_idle()

    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{epoch}-{sealed_turn_id}"
    )

async def test_speaker_shadow_factory_is_lightweight_sync_and_fail_open(
    monkeypatch,
    factory_fails: bool,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    selection = _selection("qwen", "provider")
    session = SimpleNamespace(
        is_ready=True,
        connect=AsyncMock(),
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
    shadow = SimpleNamespace(close=AsyncMock())
    factory_threads: list[threading.Thread] = []

    def factory():
        factory_threads.append(threading.current_thread())
        if factory_fails:
            raise RuntimeError("missing shadow backend")
        return shadow

    result = await runtime._asr_runtime.start(
        route_key="qwen",
        resource_optimization_enabled=True,
        speaker_shadow_factory=factory,
    )

    assert result.status is AsrStartStatus.READY
    assert factory_threads == [threading.main_thread()]
    assert detector_factory.call_args.kwargs["speaker_shadow"] is (
        None if factory_fails else shadow
    )

def test_hot_swap_replay_damage_accounts_for_rebound_frames() -> None:
    # Codex P2. Cached pre-swap frames carry a stale route generation, so replay
    # rebinds them onto the new session -- but only the local SEND token is
    # rebound; the frame objects appended to damaged_frames keep their original
    # token. The final `any(_ingress_token_matches(frame.token) ...)` check was
    # therefore false, _invalidate_interrupted_voice_turn was skipped, and a
    # prefix that had already reached the new provider stayed in place: later
    # speech got concatenated across the missing tail instead of the damaged
    # turn being cleared.
    #
    # Structural, and deliberately so: driving _flush_hot_swap_audio_cache to a
    # mid-replay failure needs a cache, live session, route mode and token
    # generations. What this pins is that the rebind records current-route
    # damage and that the damage check consults it.
    import inspect

    from main_logic.core import asr_runtime as asr_runtime_module

    source = inspect.getsource(asr_runtime_module.AsrRuntimeMixin._flush_hot_swap_audio_cache)

    assert "rebound_to_current_route = False" in source, (
        "the flush must track whether any frame was rebound onto the live route"
    )
    assert "nonlocal rebound_to_current_route" in source, (
        "replay_frames must be able to record the rebind"
    )
    # Set at the rebind, consulted at the damage check, in that order.
    set_at = source.index("rebound_to_current_route = True")
    # Anchor on the damage condition itself: a bare-name match also hits the
    # `nonlocal` declaration, which sits BEFORE the rebind and inverted this
    # ordering assertion into a false failure.
    checked_at = source.index("if damaged_frames and (")
    assert source.index("token = rebound") < set_at, (
        "the flag belongs with the rebind it records"
    )
    assert set_at < checked_at
    assert "_invalidate_interrupted_voice_turn" in source[checked_at:], (
        "the damage check must still be what gates the invalidation"
    )

async def test_provider_successor_discard_failure_fails_closed_once() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    detector.discard_provider_successor.side_effect = RuntimeError("private failure")
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    ingress_token = runtime._asr_runtime._asr_current_ingress_token
    assert ingress_token is not None

    await runtime._handle_audio_ingress_backpressure(
        ingress_token,
        observed_state=VoiceLifecycleState.DRAINING,
    )

    statuses = [json.loads(call.args[0]) for call in runtime.send_status.await_args_list]
    codes = [payload["code"] for payload in statuses]
    assert codes.count("ASR_ENDPOINTING_FAILED") == 1
    assert codes.count("ASR_INGRESS_BACKPRESSURE") == 0
    assert runtime._asr_session_epoch == epoch + 1
    assert runtime._asr_route_mode == "blocked"
    assert "private failure" not in str(runtime.send_status.await_args_list)

async def test_speaker_shadow_abba_cannot_change_provider_authority(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    quiet = b"\x00\x00" * 160
    started = b"\x01\x00" * 160
    continued = b"\x02\x00" * 160
    paused = b"\x03\x00" * 160
    successor = b"\x04\x00" * 160
    candidate_frames = (quiet, started, continued, paused)
    candidate_probabilities = (0.0, 0.9, 0.1, 0.1)
    candidate_rnnoise_available = (False, True, True, True)
    candidate_processed_frames = tuple(
        ProcessedVoiceFrame(
            pcm16=pcm16,
            sample_rate_hz=16_000,
            speech_probability=probability,
            rnnoise_available=rnnoise_available,
        )
        for pcm16, probability, rnnoise_available in zip(
            candidate_frames,
            candidate_probabilities,
            candidate_rnnoise_available,
            strict=True,
        )
    )
    successor_processed_frame = ProcessedVoiceFrame(
        pcm16=successor,
        sample_rate_hz=16_000,
        speech_probability=0.9,
        rnnoise_available=True,
    )
    gate_events = (
        (),
        (SpeechActivityEvent.SPEECH_STARTED,),
        (),
        (SpeechActivityEvent.CANDIDATE_PAUSE,),
        (SpeechActivityEvent.SPEECH_STARTED,),
    )
    # Metrics derived from wall-clock, excluded from the snapshot comparison
    # because their value depends on how the runner happened to schedule us.
    #
    # ⚠️ Audio-duration metrics (local_audio_ms / cloud_audio_ms /
    # provider_wire_audio_ms / suppressed_silence_ms / shadow_suppressed_audio_ms)
    # are deliberately NOT here: they are computed from the frames fed in, so they
    # are deterministic and are part of what this test asserts.
    #
    # ⚠️ Any new wall-clock metric must be added here. Nothing on the dataclass
    # marks a field as wall-clock, so this list cannot be derived — which is how
    # asr_audio_command_queue_ms was missed when it landed: it measures
    # `time.monotonic() - queued_at` (asr_client/audio.py), reads 0 on an idle
    # machine, and came back as 16 (the Windows timer granularity) on a busy CI
    # runner, failing whole-snapshot equality on a single value.
    volatile_metric_names = frozenset(
        {
            "connect_latency_ms",
            "first_partial_latency_ms",
            "final_latency_ms",
            "smart_turn_load_ms",
            "smart_turn_inference_ms",
            "detector_submit_latency_ms",
            "detector_queue_audio_ms",
            "detector_queue_high_water_ms",
            "detector_oldest_frame_age_ms",
            "asr_audio_command_queue_ms",
        }
    )
    real_detector_runtime = DetectorRuntime
    selection = _selection("qwen", "provider")

    class _Vad:
        def __init__(self) -> None:
            self.load_count = 0
            self.close_count = 0

        def load(self) -> bool:
            self.load_count += 1
            return True

        def close(self) -> None:
            self.close_count += 1

    class _Gate:
        def __init__(self) -> None:
            self.frames: list[bytes] = []
            self._events = iter(gate_events)

        def feed(self, pcm16: bytes):
            self.frames.append(pcm16)
            return next(self._events)

        def reset(self) -> None:
            return None

    class _ProviderSession:
        def __init__(self, callbacks: dict[str, object]) -> None:
            self.callbacks = callbacks
            self.is_ready = True
            self.connect_count = 0
            self.close_count = 0
            self.signal_end_count = 0
            self.provider_wire_audio_ms = 0
            self.wire_pcm: list[bytes] = []

        async def connect(self) -> None:
            self.connect_count += 1

        async def close(self) -> None:
            self.close_count += 1

        async def stream_audio(
            self,
            pcm16: bytes,
            *,
            sample_rate_hz: int,
        ) -> None:
            assert sample_rate_hz == 16_000
            self.wire_pcm.append(pcm16)
            self.provider_wire_audio_ms += len(pcm16) * 1_000 // (16_000 * 2)

        async def signal_user_activity_end(self) -> None:
            self.signal_end_count += 1

    class _Shadow:
        enabled = True

        def __init__(self, *, raises: bool) -> None:
            self.raises = raises
            self.submissions: list[tuple[bytes, int, object]] = []
            self.finishes: list[object] = []
            self.reset_count = 0
            self.close_count = 0

        def submit(
            self,
            pcm16: bytes,
            *,
            sample_rate_hz: int,
            candidate,
        ) -> bool:
            self.submissions.append((pcm16, sample_rate_hz, candidate))
            if self.raises and len(self.submissions) == 2:
                raise RuntimeError("shadow submit failure")
            return False

        def finish_candidate(self, candidate) -> bool:
            self.finishes.append(candidate)
            if self.raises:
                raise RuntimeError("shadow finish failure")
            return False

        async def reset(self) -> None:
            self.reset_count += 1

        async def close(self) -> None:
            self.close_count += 1

    def turn_identity(turn_token: VoiceTurnToken) -> tuple[object, ...]:
        ingress = turn_token.ingress
        return (
            turn_token.turn_id,
            ingress.session_epoch,
            ingress.connection_id,
            ingress.lease_generation,
            ingress.route_generation,
            ingress.audio_generation,
        )

    def transcript_identity(event: VoiceTranscriptEvent) -> tuple[object, ...]:
        return (event.text, event.provider, *turn_identity(event.turn_token))

    async def replay(
        shadow_mode: str | None,
    ) -> tuple[dict[str, object], _Shadow | None]:
        lifecycle_notifications: list[AsrLifecycleNotification] = []
        statuses: list[AsrStatusEvent] = []
        failures: list[AsrFailureEvent] = []
        prepared_turns: list[VoiceTurnToken] = []
        finals: list[VoiceTranscriptEvent] = []
        abandoned_turns: list[VoiceTurnToken] = []
        partials: list[VoicePartialEvent] = []
        gate = _Gate()
        vad = _Vad()
        provider_sessions: list[_ProviderSession] = []
        detector_shadows: list[object | None] = []
        factory_calls = 0
        shadow = (
            None
            if shadow_mode is None
            else _Shadow(raises=shadow_mode == "raises")
        )

        async def on_prepare_turn(turn_token: VoiceTurnToken) -> bool:
            prepared_turns.append(turn_token)
            return True

        async def on_partial(event: VoicePartialEvent) -> None:
            partials.append(event)

        async def on_final(event: VoiceTranscriptEvent) -> None:
            finals.append(event)

        async def on_turn_abandoned(turn_token: VoiceTurnToken) -> None:
            abandoned_turns.append(turn_token)

        async def on_failure(event: AsrFailureEvent) -> None:
            failures.append(event)

        async def on_status(event: AsrStatusEvent) -> None:
            statuses.append(event)

        async def on_lifecycle(event: AsrLifecycleNotification) -> None:
            lifecycle_notifications.append(event)

        runtime = IndependentAsrRuntime(
            AsrRuntimeCallbacks(
                display_name=lambda: "ABBA",
                on_prepare_turn=on_prepare_turn,
                on_partial=on_partial,
                on_final=on_final,
                on_turn_abandoned=on_turn_abandoned,
                on_failure=on_failure,
                on_status=on_status,
                on_lifecycle=on_lifecycle,
            )
        )

        def create_session(_core_type: str, **kwargs) -> _ProviderSession:
            assert kwargs["selection"] is selection
            session = _ProviderSession(kwargs)
            provider_sessions.append(session)
            return session

        def create_detector(**kwargs) -> DetectorRuntime:
            detector_shadows.append(kwargs.get("speaker_shadow"))
            return real_detector_runtime(vad=vad, gate=gate, **kwargs)

        def create_shadow():
            nonlocal factory_calls
            factory_calls += 1
            return shadow

        monkeypatch.setattr(
            runtime_module,
            "_resolve_asr_selection",
            lambda _core_type: selection,
        )
        monkeypatch.setattr(
            runtime_module,
            "_create_asr_session_from_selection",
            create_session,
        )
        monkeypatch.setattr(runtime_module, "DetectorRuntime", create_detector)

        start_result = await runtime.start(
            route_key="qwen",
            resource_optimization_enabled=True,
            speaker_shadow_factory=(create_shadow if shadow is not None else None),
        )
        assert start_result.status is AsrStartStatus.READY
        assert len(provider_sessions) == 1
        provider_session = provider_sessions[0]
        lifecycle = runtime._asr_lifecycle
        assert lifecycle is not None
        ingress_token = runtime.capture_ingress_token(
            connection_id="abba-connection",
            lease_generation=7,
            route_generation=11,
        )
        snapshots: list[tuple[object, ...]] = []

        def record_snapshot(step: str) -> None:
            snapshot = lifecycle.snapshot
            snapshots.append(
                (
                    step,
                    snapshot.state.value,
                    snapshot.route_mode.value,
                    snapshot.route_generation,
                    snapshot.transport_generation,
                    snapshot.turn_id,
                    tuple(
                        sorted(
                            (name, value)
                            for name, value in lifecycle.metrics.snapshot().items()
                            if name not in volatile_metric_names
                        )
                    ),
                )
            )

        record_snapshot("started")
        submit_statuses: list[AsrSubmitStatus] = []
        for index, processed_frame in enumerate(candidate_processed_frames):
            result = await runtime.submit(
                processed_frame,
                ingress_token=ingress_token,
            )
            await runtime._asr_audio_dispatcher.wait_idle()
            submit_statuses.append(result.status)
            record_snapshot(f"candidate-frame-{index}")

        endpoint_result = await provider_session.callbacks["on_turn_endpointed"]()
        record_snapshot("endpoint")
        provider_wire_before_successor = tuple(provider_session.wire_pcm)
        shadow_submissions_before_successor = (
            () if shadow is None else tuple(shadow.submissions)
        )

        successor_result = await runtime.submit(
            successor_processed_frame,
            ingress_token=ingress_token,
        )
        await runtime._asr_audio_dispatcher.wait_idle()
        submit_statuses.append(successor_result.status)
        record_snapshot("successor-buffered")
        provider_wire_before_final = tuple(provider_session.wire_pcm)
        shadow_submissions_before_final = (
            () if shadow is None else tuple(shadow.submissions)
        )

        final_result = await provider_session.callbacks["on_input_transcript"](
            "abba-final"
        )
        await runtime._asr_audio_dispatcher.wait_idle()
        await runtime.wait_transcript_idle()
        record_snapshot("final-successor-active")

        await runtime.close()
        record_snapshot("closed")
        trace = {
            "start_result": (
                start_result.status.value,
                start_result.provider,
                start_result.failure_code,
                start_result.session_epoch,
            ),
            "wire_pcm": tuple(provider_session.wire_pcm),
            "wire_sha256": tuple(
                hashlib.sha256(payload).hexdigest()
                for payload in provider_session.wire_pcm
            ),
            "wire_concat_sha256": hashlib.sha256(
                b"".join(provider_session.wire_pcm)
            ).hexdigest(),
            "wire_order": tuple(
                (
                    index,
                    hashlib.sha256(payload).hexdigest(),
                )
                for index, payload in enumerate(provider_session.wire_pcm, start=1)
            ),
            "wire_before_successor": provider_wire_before_successor,
            "wire_before_final": provider_wire_before_final,
            "wire_object_provenance": (
                provider_session.wire_pcm[0]
                is candidate_processed_frames[0].pcm16,
                provider_session.wire_pcm[0]
                is candidate_processed_frames[1].pcm16,
                provider_session.wire_pcm[1]
                is candidate_processed_frames[2].pcm16,
                provider_session.wire_pcm[2]
                is candidate_processed_frames[3].pcm16,
                provider_session.wire_pcm[3] is successor_processed_frame.pcm16,
            ),
            "submit_statuses": tuple(status.value for status in submit_statuses),
            "provider_callbacks": (
                ("endpoint", 1, endpoint_result),
                ("final", 1, final_result),
            ),
            "prepared_turns": tuple(
                turn_identity(token) for token in prepared_turns
            ),
            "finals": tuple(transcript_identity(event) for event in finals),
            "lifecycle": tuple(
                (event.state, event.provider, event.session_epoch)
                for event in lifecycle_notifications
            ),
            "snapshots": tuple(snapshots),
            "statuses": tuple(
                (event.code, event.provider, event.session_epoch)
                for event in statuses
            ),
            "failures": tuple(
                (event.code, event.provider, event.session_epoch)
                for event in failures
            ),
            "abandoned": tuple(
                turn_identity(token) for token in abandoned_turns
            ),
            "partials": tuple(
                (event.text, *turn_identity(event.turn_token)) for event in partials
            ),
            "provider_signal_end_count": provider_session.signal_end_count,
            "provider_connect_count": provider_session.connect_count,
            "provider_close_count": provider_session.close_count,
            "gate_pcm": tuple(gate.frames),
            "vad_load_count": vad.load_count,
            "vad_close_count": vad.close_count,
            "closed": (
                runtime._asr_session is None,
                runtime._asr_detector is None,
                runtime._asr_lifecycle is None,
                runtime._asr_warm_expiry_task is None,
            ),
        }

        assert detector_shadows == [shadow]
        assert factory_calls == (0 if shadow is None else 1)
        assert provider_wire_before_successor == (
            quiet + started,
            continued,
            paused,
        )
        assert provider_session.wire_pcm[1] is candidate_processed_frames[2].pcm16
        assert provider_session.wire_pcm[2] is candidate_processed_frames[3].pcm16
        assert provider_wire_before_final == provider_wire_before_successor
        if shadow is not None:
            assert shadow_submissions_before_successor == tuple(shadow.submissions[:3])
            assert shadow_submissions_before_final == shadow_submissions_before_successor
            assert len(shadow.submissions) == 4
            assert all(
                submission[0] is provider_payload
                for submission, provider_payload in zip(
                    shadow.submissions,
                    provider_session.wire_pcm,
                    strict=True,
                )
            )
            assert [submission[0] for submission in shadow.submissions] == [
                quiet + started,
                continued,
                paused,
                successor,
            ]
            assert quiet not in [
                submission[0] for submission in shadow.submissions
            ]
            assert all(submission[1] == 16_000 for submission in shadow.submissions)
            candidates = [submission[2] for submission in shadow.submissions]
            assert candidates[0] == candidates[1] == candidates[2]
            assert candidates[3] != candidates[0]
            assert all(
                getattr(candidate, "scope") == "provider_candidate"
                for candidate in candidates
            )
            assert getattr(candidates[3], "shadow_generation") > getattr(
                candidates[0],
                "shadow_generation",
            )
            assert shadow.finishes == [candidates[0]]
            assert shadow.reset_count == 0
            assert shadow.close_count == 1
        return trace, shadow

    disabled_a, _ = await replay(None)
    observed_false, false_shadow = await replay("false")
    observed_raising, raising_shadow = await replay("raises")
    disabled_b, _ = await replay(None)

    assert false_shadow is not None
    assert raising_shadow is not None
    assert disabled_a == observed_false == observed_raising == disabled_b
    assert disabled_a["wire_pcm"] == (
        quiet + started,
        continued,
        paused,
        successor,
    )
    assert disabled_a["wire_concat_sha256"] == hashlib.sha256(
        quiet + started + continued + paused + successor
    ).hexdigest()
    assert disabled_a["wire_object_provenance"] == (
        False,
        False,
        True,
        True,
        False,
    )
    assert disabled_a["submit_statuses"] == ("accepted",) * 5
    assert disabled_a["provider_callbacks"] == (
        ("endpoint", 1, None),
        ("final", 1, None),
    )
    assert disabled_a["provider_signal_end_count"] == 0
    assert disabled_a["provider_connect_count"] == 1
    assert disabled_a["provider_close_count"] == 1
    assert len(disabled_a["finals"]) == 1

async def test_independent_multimodal_turn_samples_the_utterance_span() -> None:
    """One utterance carries first/middle/last; identity fields name the last."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=77)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    captured_at = time.monotonic()

    assert runtime._stage_independent_visual_frame(
        "first-frame",
        source="screen",
        request_id="frame-1",
        captured_at=captured_at,
    )
    assert runtime._stage_independent_visual_frame(
        "latest-frame",
        source="camera",
        request_id="frame-2",
        captured_at=captured_at + 0.1,
    )
    assert not runtime._stage_independent_visual_frame(
        "stale-frame",
        source="screen",
        request_id="frame-stale",
        captured_at=captured_at - 0.1,
    )

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    # 两帧都在本回合窗口内：开头那张不能因为"不是最新"被丢掉——用户开口时指的
    # 东西就在那张上。source / request_id 仍然描述最新那张（回合的收尾身份）。
    assert turn.images == ("first-frame", "latest-frame")
    assert turn.source == "camera"
    assert turn.request_id == "frame-2"
    assert turn.image_generation > turn.start_image_generation

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

async def test_offline_image_free_voice_turn_retries_tts_after_failure() -> None:
    runtime = _Runtime()
    runtime.response_backend = "offline_vlm"
    runtime.ensure_tts_pipeline_alive = AsyncMock(
        side_effect=[RuntimeError("tts unavailable"), None]
    )
    runtime.session.submit_external_voice_turn = AsyncMock()

    with pytest.raises(RuntimeError, match="tts unavailable"):
        await runtime._submit_core_voice_turn(
            "first",
            turn_id="turn-1",
            session_ref=runtime.session,
        )
    runtime.session.submit_external_voice_turn.assert_not_awaited()

    await runtime._submit_core_voice_turn(
        "second",
        turn_id="turn-2",
        session_ref=runtime.session,
    )

    assert runtime.ensure_tts_pipeline_alive.await_count == 2
    runtime.session.submit_external_voice_turn.assert_awaited_once_with(
        "second",
        turn_id="turn-2",
    )

async def test_out_of_order_frame_still_joins_the_turn_sample() -> None:
    """A frame that validates late must not be dropped by the latest-frame guard."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=91)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    base = record.started_at

    assert runtime._stage_independent_visual_frame(
        "later-frame",
        source="screen",
        request_id="screen-later",
        captured_at=base + 1.0,
    )
    # 更早拍摄、更晚校验完：不能顶掉最新帧缓存，但必须进本回合抽样。
    assert runtime._stage_independent_visual_frame(
        "earlier-frame",
        source="camera",
        request_id="camera-earlier",
        captured_at=base + 0.1,
    )
    assert runtime._latest_independent_visual_frame.image_b64 == "later-frame"

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("earlier-frame", "later-frame")

async def test_endpoint_cutoff_uses_the_recorded_seal_instant() -> None:
    """A frame captured in the gap before Core looks must still be excluded."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=95)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "spoken-frame",
        source="screen",
        request_id="screen-spoken",
        captured_at=record.started_at,
    )

    # ASR 在这一刻封口，但 Core 要到下一帧 staging 才会去看。
    sealed_at = record.started_at + 1.0
    runtime._asr_turn_endpointed_at = sealed_at
    _seal_utterance(runtime)

    # 这帧拍摄于封口之后、Core 观察之前——按观察时刻当截止值它会被放行。
    runtime._stage_independent_visual_frame(
        "gap-frame",
        source="screen",
        request_id="screen-gap",
        captured_at=sealed_at + 0.5,
    )

    assert record.endpoint_at == sealed_at
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("spoken-frame",)

async def test_a_seal_after_this_record_registered_still_becomes_its_cutoff() -> None:
    """Dual: a retained seal that really belongs to this turn still binds.

    Guards against over-tightening the gate into "never trust a retained copy".
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=98)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    sealed_at = record.registered_at + 1.0
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = sealed_at
    runtime._asr_last_turn_endpointed_key = record.turn_id

    runtime._stage_independent_visual_frame(
        "late-frame",
        source="screen",
        request_id="screen-late",
        captured_at=sealed_at + 0.5,
    )

    assert record.endpoint_at == sealed_at

async def test_frame_captured_before_the_onset_is_still_a_prior_turn_frame() -> None:
    """Widening the window to the onset must not reach into the previous turn."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()

    assert runtime._stage_independent_visual_frame(
        "prior-turn-frame",
        source="screen",
        request_id="screen-prior",
        captured_at=onset - 1.0,
    )
    runtime._asr_turn_onset_at = onset

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=99)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)

    assert runtime._snapshot_core_multimodal_turn(turn_id, "new question") is None

async def test_reconnect_listener_join_is_bounded() -> None:
    """A receive task that swallows cancellation must not wedge the swap lock."""
    from main_logic.core import LLMSessionManager

    manager = LLMSessionManager.__new__(LLMSessionManager)
    manager.lanlan_name = "Test"
    manager.lock = asyncio.Lock()
    manager.is_active = True
    manager._core_voice_listener_cancel_timeout_s = 0.05
    manager.session_ready = True
    manager._close_independent_asr = AsyncMock()
    manager.send_session_ended_by_server = AsyncMock()
    session = SimpleNamespace(handle_messages=AsyncMock(), close=AsyncMock())
    manager.session = session

    stuck_release = asyncio.Event()

    async def stuck_listener() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await stuck_release.wait()

    listener = asyncio.create_task(stuck_listener())
    manager.message_handler_task = listener
    await asyncio.sleep(0)

    installed = await asyncio.wait_for(
        manager._restart_message_handler_after_session_reconnect(session),
        5.0,
    )

    # fail-closed：停不下来的 listener 还绑在退休会话上，不能在它之上再装一个
    # receive 循环；调用方都把 False 当成"放弃这次重连"。
    assert installed is False
    session.handle_messages.assert_not_called()

    # 而且必须把这条会话**退休**掉：只返回 False 会留下一个看起来还活着、实际没有
    # receive 循环的 client，之后每一轮都撞上同一个卡死的 task 再超时一次，语音从此
    # 永远收不到回复。
    assert manager.session is None
    assert manager.message_handler_task is None
    assert manager.is_active is False
    assert manager.session_ready is False

    # 会话没了，麦克风也必须收掉：否则独立 ASR 继续往一个不存在的回答会话投
    # transcript，用户说什么都石沉大海。
    manager._close_independent_asr.assert_awaited_once_with(next_route_mode="blocked")
    manager.send_session_ended_by_server.assert_awaited_once_with()

    stuck_release.set()
    await asyncio.gather(listener, return_exceptions=True)
    for _ in range(50):
        if session.close.await_count:
            break
        await asyncio.sleep(0.01)
    session.close.assert_awaited_once_with()

async def test_prerecord_frame_buffer_is_bounded() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset

    for index in range(40):
        runtime._stage_independent_visual_frame(
            f"prerecord-{index}",
            source="screen",
            request_id=f"screen-{index}",
            captured_at=onset + 0.001 * (index + 1),
        )

    assert len(runtime._prerecord_visual_frames) <= 8
    # 超限时丢的是"最冗余"的内点，**不是队头** —— 队头正是这段发声的开头。
    kept = [frame.image_b64 for frame in runtime._prerecord_visual_frames]
    assert kept[0] == "prerecord-0"
    assert kept[-1] == "prerecord-39"

async def test_live_endpoint_still_seals_its_own_turn() -> None:
    """The live field only ever describes the in-flight turn, so keep it loose."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=106)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "spoken-frame",
        source="screen",
        request_id="screen-spoken",
        captured_at=record.started_at,
    )
    # 极短发声：封口甚至可能早于 record 注册那一刻。live 字段仍然必须绑上。
    sealed_at = record.started_at
    runtime._asr_turn_endpointed_at = sealed_at
    runtime._mark_independent_asr_endpoint_if_sealed()

    assert record.endpoint_at == sealed_at

async def test_retained_turn_records_are_bounded() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    for turn_id in range(210, 230):
        token = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(), turn_id=turn_id
        )
        runtime._begin_core_multimodal_turn(
            f"asr-{token.ingress.session_epoch}-{token.turn_id}", token
        )

    # 记录本该由各自 dispatch 的 finally 移除；这个上限只是内存兜底。
    assert len(runtime._core_multimodal_turns) <= 8
    # 留下的是最近的那些 —— 淘汰绝不能挑到最新那条（它才是当前在跑的）。
    kept = sorted(runtime._core_multimodal_turns)
    assert kept[-1].endswith("-229")

async def test_the_real_dispatch_marks_its_record_before_it_can_be_evicted() -> None:
    """The flag has to be set by the dispatch itself, not only in a test.

    A guard that only checks the eviction predicate passes even when nothing
    ever sets the flag; this drives the actual final through
    ``_dispatch_core_asr_transcript`` and lets a long run of successor prepares
    land while it is suspended.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.submit_external_voice_turn = AsyncMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    accepted = runtime.handle_input_transcript
    seen_mid_dispatch = {}

    async def accept_then_let_successors_pile_up(*args, **kwargs):
        result = await accepted(*args, **kwargs)
        for turn_id_n in range(601, 601 + _MAX_LIVE_TURN_RECORDS * 2):
            successor = VoiceTurnToken(
                ingress=runtime._capture_ingress_token(),
                turn_id=turn_id_n,
            )
            runtime._begin_core_multimodal_turn(
                f"asr-{successor.ingress.session_epoch}-{successor.turn_id}",
                successor,
            )
        seen_mid_dispatch["record"] = runtime._core_multimodal_turns.get(turn_id)
        return result

    runtime.handle_input_transcript = accept_then_let_successors_pile_up

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=token,
            provider="openai",
            text="the sentence that must not be dropped",
        )
    )

    assert seen_mid_dispatch["record"] is record
    runtime.session.submit_external_voice_turn.assert_awaited_once()
    # 自己的 finally 摘掉它。
    assert turn_id not in runtime._core_multimodal_turns

async def test_live_onset_replay_waits_behind_queued_overlap_credits() -> None:
    """FIFO order decides who gets replayed, not who is newest.

    A completed onset/pause cycle (turn 2) and a still-live onset (turn 3) can
    coexist when turn 1's final is delayed. The provider FIFO still delivers
    turn 2's endpoint/final first, so replaying turn 3 right now hands turn 2's
    endpoint a turn-3 record: turn 2's transcript takes turn 3's visual window,
    and turn 3's own endpoint finds no credit left, dropping its final.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # Turn 2: a full onset/pause cycle while turn 1 is ACTIVE -> one credit.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.CANDIDATE_PAUSE,
        epoch,
    )
    assert runtime._asr_overlap_completed_turns == 1
    # Turn 3: onset only -- the user is still speaking, so it stays in the
    # single slot instead of becoming a credit.
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    assert runtime._asr_overlap_onset_token is not None

    # Turn 1's delayed final. Turn 3 must NOT be replayed here.
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_overlap_completed_turns == 1
    assert runtime._asr_overlap_onset_token is not None

    # Turn 2 redeems its own credit, in its own FIFO slot.
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()
    # Credits are drained, so turn 3's onset finally gets its replay.
    assert runtime._asr_overlap_completed_turns == 0

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("third", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second", "third"]
    assert runtime.handle_new_message.await_count == 3
    assert runtime._asr_overlap_onset_token is None

async def test_idle_frames_do_not_consume_the_prerecord_budget() -> None:
    """Frames from before the user spoke are not this turn's to keep.

    Screen sharing fills the eight-slot buffer while nobody is talking. The
    sampler deliberately preserves widely spaced endpoints, so those idle
    frames hold their slots; the few captured between speech confirmation and
    record creation then get sampled together with the whole idle history, and
    the onset filter at record creation discards all of them -- leaving only
    the newest frame and losing this turn's opening and middle views.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    now = time.monotonic()

    # 共享着但没人说话：闲置帧铺满缓冲。
    runtime._asr_runtime._asr_turn_onset_at = None
    for i in range(_MAX_PRERECORD_VISUAL_VALIDATIONS):
        assert runtime._stage_independent_visual_frame(
            f"idle-{i}",
            source="screen",
            request_id=f"screen-idle-{i}",
            captured_at=now - 60.0 + i * 5.0,
        )
    assert len(runtime._prerecord_visual_frames) == _MAX_PRERECORD_VISUAL_VALIDATIONS

    # 用户开口。确认到注册之间又拍了三张。
    onset = now - 2.0
    runtime._asr_runtime._asr_turn_onset_at = onset
    for i in range(3):
        assert runtime._stage_independent_visual_frame(
            f"speech-{i}",
            source="screen",
            request_id=f"screen-speech-{i}",
            captured_at=onset + 0.2 * (i + 1),
        )

    kept = [f.image_b64 for f in runtime._prerecord_visual_frames]
    # 开口之前的一张都不占名额了，这一轮自己的三张全在。
    assert kept == ["speech-0", "speech-1", "speech-2"]

async def test_direct_overlap_replay_seals_when_the_session_is_not_ready() -> None:
    """The direct replay must complete its confirmation in place too.

    Dual of the completed-overlap credit path. Parking in PREWARMING and just
    holding the onset is not enough: this successor's provider endpoint and
    final are already queued in the ordered FIFO and about to arrive, a
    PREWARMING lifecycle cannot seal, and _handle_independent_asr_final()
    requires DRAINING -- so the whole utterance is discarded with no watchdog
    armed.

    Waiting for the reconnect cannot recover it either: a reconnect swaps in a
    new session and is_adopted_candidate() drops every callback still queued on
    the old one (_restart_transport / _close_transport_only both null
    _asr_session before closing it). Reaching this point proves the old session
    is still adopted, i.e. the reconnect has not started.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch
    component = runtime._asr_runtime

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # 后继在上一轮还 ACTIVE 时开口：它的 onset 被记下来等直接重放。
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    recorded_onset = component._asr_overlap_onset_at
    assert recorded_onset is not None
    await runtime._handle_independent_asr_endpoint(epoch)

    # 两条有序回调之间传输掉线：重放会停在 PREWARMING 并挂起确认。
    component._asr_session.is_ready = False

    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    # 重放就地补完了确认：回合醒着，后继排在 FIFO 里的 endpoint 才封得了口。
    # （HEAD 上这里是 PREWARMING，封不了口，那条 final 会被整条丢弃。）
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    # 这一刻还没 prepare 是对的：直接重放只负责唤醒，prepare 由后继自己的
    # endpoint 完成（_handle_independent_asr_endpoint 的 not _asr_turn_prepared
    # 分支）。断言它已 prepare 属于对契约的过度主张。
    # 用的是用户当初真实开口的时刻，不是这次重放的时刻。
    assert component._asr_turn_onset_at == recorded_onset
    assert component._asr_pending_speech_onset_at is None
    # 没走 fail-closed 出口（那条会 bump epoch、拆掉 session）。
    assert runtime._asr_session_epoch == epoch

    # 后继自己的 endpoint 紧随其后到达 —— 这一步才封口。
    await runtime._handle_independent_asr_endpoint(epoch)
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    # 忙窗口有定时器兜底。
    assert component._asr_final_watchdog_task is not None

    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second"]
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE

async def test_an_unwoken_redemption_still_seals_and_delivers_the_queued_final() -> None:
    """An unwoken replay must still seal: its final is already on the way.

    This test REPLACES test_a_pending_confirmation_keeps_the_lent_onset and
    deliberately overturns its reasoning ("hold the onset for the confirmation
    that follows the reconnect"). The reconnect cannot recover this final:
    _restart_transport() nulls _asr_session before closing it, after which
    every provider callback is dropped by is_adopted_candidate(). Reaching this
    point proves the old session is still adopted -- the reconnect has not
    started and the final is right behind this endpoint in the ordered FIFO.
    Completing the confirmation in place, so that final finds a DRAINING turn,
    is the only way not to lose the utterance.

    Measured on HEAD: state=PREWARMING, credit still 1, sealed_token=None, both
    the warm-expiry and provider-final timers None, transcripts only ["first"]
    -- the whole sentence lost AND a busy flag left with no timer behind it.
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
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    component = runtime._asr_runtime
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert component._asr_overlap_completed_turns == 1
    recorded_onset = component._asr_overlap_completed_onsets[0]

    # 不打桩：跑真实控制流，只让传输在两条有序回调之间掉线。
    component._asr_session.is_ready = False

    await runtime._handle_independent_asr_endpoint(epoch)

    # 仍然封口，那条排在后面的 final 才有 DRAINING 可落。
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    # 恰好兑付一次，不多不少。
    assert component._asr_overlap_completed_turns == 0
    assert list(component._asr_overlap_completed_onsets) == []
    assert component._asr_overlap_completed_token is None
    # onset 被本轮消费掉，不会被后面某个不相干的回合当成自己的起点。
    assert component._asr_pending_speech_onset_at is None
    # 用的是用户当初真实开口的时刻，不是这次重放的时刻。
    assert component._asr_turn_onset_at == recorded_onset
    # 忙窗口有定时器兜底（HEAD 上这里是 None）。
    assert component._asr_final_watchdog_task is not None
    # 没走 fail-closed 出口：那条会 bump epoch、拆掉 session、把语音判死。
    # 只有这组断言能区分两条出口——错误出口也会发同名 status。
    assert runtime._asr_session_epoch == epoch
    assert runtime._asr_lifecycle is not None

    await runtime._handle_independent_asr_final("second", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second"]
    # 收尾不留忙标志。
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
