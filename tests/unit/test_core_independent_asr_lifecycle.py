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

async def test_blocked_consumer_callback_does_not_block_next_turn_lifecycle() -> (
    None
):
    runtime = _Runtime()
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    async def block_first_final(*_args, **_kwargs) -> bool:
        callback_started.set()
        await release_callback.wait()
        return True

    runtime.handle_input_transcript.side_effect = block_first_final
    await _start_and_seal_turn(runtime, "qwen")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_final("first", epoch, "qwen")
    await asyncio.wait_for(callback_started.wait(), 1)
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_turn_prepared is True
    release_callback.set()
    await runtime._wait_asr_transcript_dispatch_idle()
    runtime.session.create_response.assert_awaited_once_with("first")

async def test_hot_swap_lifecycle_guards_close_and_promote_with_voice_barrier() -> None:
    source = inspect.getsource(
        core_module.LLMSessionManager._perform_final_swap_sequence
    )

    barrier = source.index("async with core_voice_session_lock")
    close = source.index("old_main_session.close()", barrier)
    promote = source.index("self.session = new_session", close)
    barrier_exit = source.index("if not _promote_allowed", promote)
    assert barrier < close < promote < barrier_exit
    assert "asyncio.timeout_at" in source[barrier:promote]

async def test_submit_without_lifecycle_returns_typed_unavailable() -> None:
    runtime = _Runtime()
    result = await runtime._asr_runtime.submit(
        ProcessedVoiceFrame(b"\x01\x00" * 160, 16_000, 0.0, False),
        ingress_token=VoiceIngressToken(0, "socket", 0, 0, 0),
    )

    assert result == AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
    assert not isinstance(result, bool)

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
