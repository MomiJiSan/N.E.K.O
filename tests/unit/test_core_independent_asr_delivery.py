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

async def test_core_forgets_future_verifier_when_physical_detach_degrades() -> None:
    runtime = _Runtime()
    runtime._speaker_shadow_factory = MagicMock()
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(return_value=False)

    updated = await runtime.set_speaker_verifier_factory(
        None,
        activation_generation="revoked-profile",
    )

    assert updated is False
    assert runtime._speaker_shadow_factory is None

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

async def test_smart_turn_unavailable_blocks_segmented_provider_before_wire_audio(
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
        provider_policy=resolve_provider_policy(
            provider,
            "manual",
        ),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _FailedSmartTurnDetector()

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )

    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_route_mode == "blocked"
    assert runtime._omni_mic_audio_bytes == 0

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

async def test_game_consumer_reuses_smart_turn_asr_without_core(
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
    assert runtime._voice_input_accepts_pcm() is True

    _install_ready_lifecycle(runtime, "qwen")
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "qwen")
    await runtime._handle_independent_asr_final("play", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    route_transcript.assert_awaited_once_with(
        "Test",
        "play",
        request_id=f"asr-{epoch}-1",
        game_type="game",
        session_id="session-a",
    )
    runtime.handle_new_message.assert_not_awaited()
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    assert runtime._omni_mic_audio_bytes == 0

    assert (
        await runtime._handle_voice_input_control(
            "lease_sync",
            2,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is True
    )

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

async def test_game_consumer_failure_never_falls_back_to_core(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    route_transcript = AsyncMock(side_effect=RuntimeError("consumer failed"))
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

    await runtime._handle_independent_asr_final("play", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    route_transcript.assert_awaited_once_with(
        "Test",
        "play",
        request_id=f"asr-{epoch}-1",
        game_type="game",
        session_id="session-a",
    )
    runtime.handle_new_message.assert_not_awaited()
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    assert runtime._omni_mic_audio_bytes == 0

async def test_fresh_blocked_route_consumes_pcm_without_omni() -> None:
    runtime = _Runtime()

    consumed = await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )

    assert consumed is True
    assert runtime._asr_audio_bytes == 0
    assert runtime._omni_mic_audio_bytes == 0

async def test_voice_session_activation_gates_native_then_replays_and_forwards() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.session.stream_audio = AsyncMock()
    factory = _CoreActivationFactory()
    assert (
        await runtime.set_voice_session_activation_factory(
            factory,
            activation_generation="profile",
        )
        is VoiceIdentityActivationResult.READY
    )

    pcm16 = b"\xd0\x07" * 1_600
    await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    await asyncio.sleep(0)
    for _ in range(14):
        await runtime._route_microphone_audio(pcm16, sample_rate_hz=16_000)
    assert runtime.session.stream_audio.await_count == 0

    await _wait_for_activation_output(
        lambda: runtime.session.stream_audio.await_count,
        expected_count=15,
    )
    assert runtime.session.stream_audio.await_count == 15
    assert factory.scorers[0].calls == 1

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    )
    await asyncio.sleep(0)
    assert runtime.session.stream_audio.await_count == 16
    assert factory.scorers[0].calls == 1
    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="disabled",
    )

async def test_disabling_activation_reconnects_idle_native_session_on_next_frame() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "native"
    runtime.is_active = True
    runtime.session.instructions = "stay in character"
    runtime.session._connection_generation = 18
    delivered: list[bytes] = []

    async def reconnect(_instructions: str, *, native_audio: bool) -> None:
        assert native_audio is True
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
    runtime.session_closed_by_server = True
    runtime._native_activation_idle_reconnect_identity = (
        factory.runtimes[0].generation,
        18,
    )

    assert (
        await runtime.set_voice_session_activation_factory(
            None,
            activation_generation="disabled",
        )
        is VoiceIdentityActivationResult.READY
    )
    disabled_generation = runtime._capture_voice_session_activation_generation()
    assert runtime._native_activation_idle_reconnect_identity == (
        disabled_generation,
        18,
    )

    live_frame = b"\x01\x00" * 160
    await runtime._route_microphone_audio(live_frame, sample_rate_hz=16_000)

    runtime.session.connect.assert_awaited_once_with(
        "stay in character",
        native_audio=True,
    )
    assert delivered == [live_frame]
    assert runtime.session_closed_by_server is False
    assert runtime._native_activation_idle_reconnect_identity is None

async def test_required_activation_without_factory_blocks_both_audio_routes(
    route_mode: str,
) -> None:
    runtime = _Runtime()
    runtime.session.stream_audio = AsyncMock()
    runtime._asr_runtime.submit = AsyncMock(
        return_value=AsrSubmitResult(AsrSubmitStatus.ACCEPTED)
    )
    runtime._set_microphone_route(route_mode)

    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="required-unavailable",
        activation_required=True,
    )
    frame = b"\x01\x00" * 160
    assert await runtime._route_microphone_audio(
        frame,
        sample_rate_hz=16_000,
    )

    runtime.session.stream_audio.assert_not_awaited()
    runtime._asr_runtime.submit.assert_not_awaited()

    await runtime.set_voice_session_activation_factory(
        None,
        activation_generation="explicitly-disabled",
        activation_required=False,
    )
    assert await runtime._route_microphone_audio(
        frame,
        sample_rate_hz=16_000,
    )
    if route_mode == "native":
        runtime.session.stream_audio.assert_awaited_once_with(frame)
        runtime._asr_runtime.submit.assert_not_awaited()
    else:
        runtime.session.stream_audio.assert_not_awaited()
        runtime._asr_runtime.submit.assert_awaited_once()

async def test_session_activation_detaches_preexisting_utterance_verifier() -> None:
    runtime = _Runtime()
    legacy_factory = MagicMock()
    runtime._speaker_shadow_factory = legacy_factory
    runtime._asr_runtime.set_speaker_verifier_factory = AsyncMock(return_value=True)
    factory = _CoreActivationFactory()

    assert (
        await runtime.set_voice_session_activation_factory(
            factory,
            activation_generation="profile",
        )
        is VoiceIdentityActivationResult.READY
    )
    runtime._asr_runtime.set_speaker_verifier_factory.assert_awaited_once_with(
        None,
        activation_generation="profile",
    )
    assert runtime._speaker_shadow_factory is None
    assert runtime._voice_session_activation_factory is factory

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

async def test_gemini_prepare_reconnect_replaces_core_receive_task() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.prepare_external_voice_turn = AsyncMock(return_value=True)
    runtime._restart_message_handler_after_session_reconnect = AsyncMock(
        return_value=True
    )

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        runtime._asr_session_epoch,
    )

    runtime._restart_message_handler_after_session_reconnect.assert_awaited_once_with(
        runtime.session
    )
    runtime.handle_new_message.assert_awaited_once_with()

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

async def test_final_transcript_drops_new_conversation_swap_mid_restore() -> None:
    """A real conversation transition still invalidates the prepared final."""
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    timed_session = runtime.session

    replacement = type("Omni", (), {})()
    replacement.create_response = AsyncMock()
    replacement.submit_external_voice_turn = AsyncMock()
    replacement.abandon_external_voice_turn = MagicMock()

    async def _hot_swap_mid_restore(*_args, **_kwargs) -> None:
        runtime._voice_input_transition_generation += 1
        runtime.session = replacement

    runtime._restore_core_asr_preview_after_final = _hot_swap_mid_restore

    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(turn_token=token, provider="qwen", text="hello"),
    )

    # CodeRabbit: the race is manufactured inside a hook, so if that hook ever
    # stops being called -- preview restore skipped, moved, or bypassed on the
    # accepted branch -- runtime.session would never move, and all three
    # assertions below would pass while modelling an ordinary final with no hot
    # swap at all. Pin that the swap really happened first.
    assert runtime.session is replacement
    timed_session.create_response.assert_not_awaited()
    replacement.create_response.assert_not_awaited()
    replacement.submit_external_voice_turn.assert_not_awaited()

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

async def test_stale_final_guard_releases_keyed_external_turn_pause(
    stale_guard: str,
) -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime)
    runtime.session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    event = VoiceTranscriptEvent(
        turn_token=token,
        provider="qwen",
        text="hello",
    )
    if stale_guard == "ingress":
        runtime._asr_audio_generation += 1
    else:
        runtime._voice_lease_owner = "game"

    await runtime._dispatch_core_asr_transcript(event)

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    )

async def test_core_asr_teardown_force_releases_external_turn_pause(
    operation: str,
) -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    runtime.session.abandon_external_voice_turn = MagicMock()
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    assert await runtime._prepare_voice_input_turn(token) is True

    if operation == "abort":
        runtime._asr_runtime.abort = AsyncMock()
        await runtime._abort_independent_asr("test_abort")
        runtime._asr_runtime.abort.assert_awaited_once_with("test_abort")
    else:
        runtime._asr_runtime.close = AsyncMock()
        await runtime._close_independent_asr(next_route_mode="blocked")
        runtime._asr_runtime.close.assert_awaited_once_with()

    runtime.session.abandon_external_voice_turn.assert_called_once_with(
        f"asr-{token.ingress.session_epoch}-{token.turn_id}",
    )

async def test_runtime_state_initializes_and_backfills_phase4a_fields() -> None:
    runtime = _Runtime()

    assert runtime._voice_input_resource_optimization_handshake_override is None
    assert runtime._voice_input_resource_optimization_session_value is None
    assert runtime._core_asr_preview_turn_token is None
    assert runtime._voice_input_external_suppressions == set()

    del runtime._voice_input_resource_optimization_handshake_override
    del runtime._voice_input_resource_optimization_session_value
    del runtime._core_asr_preview_turn_token
    del runtime._voice_input_external_suppressions
    runtime._ensure_asr_runtime_state()

    assert runtime._voice_input_resource_optimization_handshake_override is None
    assert runtime._voice_input_resource_optimization_session_value is None
    assert runtime._core_asr_preview_turn_token is None
    assert runtime._voice_input_external_suppressions == set()

async def test_segmented_fail_open_uses_continuous_wake_without_fake_speech() -> None:
    runtime = _Runtime()
    runtime._voice_input_resource_optimization_enabled = False
    asr = type("Asr", (), {"is_ready": True, "stream_audio": AsyncMock()})()
    runtime._asr_session = asr
    runtime._asr_provider = "glm"
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("glm", "manual"),
        shadow_mode=False,
        resource_optimization_enabled=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _QueuedSmartTurnDetector()
    runtime._asr_detector = detector

    await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
        rnnoise_available=False,
    )
    await runtime._asr_detector_dispatcher.wait_idle()
    await runtime._asr_audio_dispatcher.wait_idle()

    detector.force_speech_started.assert_not_awaited()
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    asr.stream_audio.assert_awaited_once()
    assert runtime._asr_route_mode == "independent"

async def test_draining_next_speech_waits_for_old_final_then_starts_new_turn() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "manual"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    detector = _ReadyDetector()
    detector.feed = AsyncMock(return_value=DetectorFeedResult((), True))
    detector.reset = AsyncMock()
    detector.release_deferred_turn = AsyncMock()
    runtime._asr_detector = detector
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    old_turn = runtime._asr_lifecycle.identity.turn_id
    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    await runtime._route_microphone_audio(
        b"\x02\x00" * 160,
        sample_rate_hz=16_000,
    )

    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_lifecycle.pending_turn_bytes == 320

    await runtime._handle_independent_asr_final("first", epoch, "qwen")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert runtime._asr_lifecycle.identity.turn_id == old_turn + 1
    asr.stream_audio.assert_awaited_once_with(
        b"\x02\x00" * 160,
        sample_rate_hz=16_000,
    )
    assert runtime.handle_new_message.await_count == 2
    detector.reset.assert_not_awaited()
    detector.release_deferred_turn.assert_awaited_once_with()

    runtime.handle_input_transcript.reset_mock()
    await runtime._handle_independent_asr_final("stale-old-turn", epoch, "qwen")
    runtime.handle_input_transcript.assert_not_awaited()

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

async def test_two_completed_overlaps_replay_in_order_after_delayed_final() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    # Turns 2 and 3 each start and reach local silence while turn 1 is still
    # ACTIVE: one completed-overlap credit accumulates per onset+pause cycle.
    for _ in range(2):
        await runtime._handle_independent_asr_activity(
            SpeechActivityEvent.SPEECH_RESUMED,
            epoch,
        )
        await runtime._handle_independent_asr_activity(
            SpeechActivityEvent.CANDIDATE_PAUSE,
            epoch,
        )
    assert runtime._asr_overlap_completed_turns == 2

    await runtime._handle_independent_asr_endpoint(epoch)
    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    for text in ("second", "third"):
        await runtime._handle_independent_asr_endpoint(epoch)
        await runtime._handle_independent_asr_final(text, epoch, "openai")
        await runtime._wait_asr_transcript_dispatch_idle()

    assert [
        call.args[0] for call in runtime.handle_input_transcript.await_args_list
    ] == ["first", "second", "third"]
    assert runtime.handle_new_message.await_count == 3
    assert runtime._asr_overlap_completed_turns == 0

async def test_draining_pending_turn_overflow_discards_candidate_and_reports_backpressure() -> (
    None
):
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.stream_audio = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_provider = "qwen"
    runtime._asr_route_mode = "independent"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("qwen", "provider"),
        shadow_mode=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    runtime._asr_detector = _ReadyDetector()
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
        b"\x01\x00" * (16_000 * 9),
        sample_rate_hz=16_000,
    )

    asr.stream_audio.assert_not_awaited()
    assert runtime._asr_session is asr
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.DRAINING
    assert runtime._asr_sealed_turn_token is not None
    assert runtime._asr_lifecycle.pending_turn_bytes == 0
    assert runtime._asr_lifecycle.has_pending_turn is False
    runtime._asr_detector.reset.assert_not_awaited()
    runtime._asr_detector.discard_provider_successor.assert_awaited_once_with(
        runtime._asr_provider_candidate_fence
    )
    assert any(
        "ASR_INGRESS_BACKPRESSURE" in call.args[0]
        for call in runtime.send_status.await_args_list
    )
    assert runtime._omni_mic_audio_bytes == 0

async def test_prewarming_uses_idle_transport_ttl() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {"close": AsyncMock()})()
    runtime._asr_session = asr
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

    runtime._schedule_transport_warm_expiry(
        runtime._asr_session_epoch,
        expected_state=VoiceLifecycleState.PREWARMING,
    )
    expiry = runtime._asr_warm_expiry_task
    assert expiry is not None
    await asyncio.wait_for(expiry, 1)

    assert runtime._asr_session is None
    assert lifecycle.snapshot.state is VoiceLifecycleState.DEEP_SLEEP
    asr.close.assert_awaited_once_with()

async def test_submit_without_lifecycle_returns_typed_unavailable() -> None:
    runtime = _Runtime()
    result = await runtime._asr_runtime.submit(
        ProcessedVoiceFrame(b"\x01\x00" * 160, 16_000, 0.0, False),
        ingress_token=VoiceIngressToken(0, "socket", 0, 0, 0),
    )

    assert result == AsrSubmitResult(AsrSubmitStatus.UNAVAILABLE)
    assert not isinstance(result, bool)

async def test_optimization_disabled_buffers_until_initial_transport_is_ready() -> None:
    runtime = _Runtime()
    runtime._voice_input_resource_optimization_enabled = False
    runtime._asr_route_mode = "independent"
    runtime._asr_provider = "glm"
    runtime._asr_lifecycle = VoiceInputLifecycleController(
        provider_policy=resolve_provider_policy("glm", "manual"),
        shadow_mode=False,
        resource_optimization_enabled=False,
    )
    runtime._asr_lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
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
    runtime._asr_transport_selection = _selection("glm")
    pcm16 = b"\x04\x00" * 160

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

async def test_game_takeover_suppresses_stale_detector_dispatcher_failure() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.is_ready = True
    asr.close = AsyncMock()
    runtime._asr_session = asr
    _install_ready_lifecycle(runtime, "qwen")
    detector = _ReadyDetector()
    detector.detector_epoch = 1
    runtime._asr_detector = detector
    lifecycle = runtime._asr_lifecycle
    assert lifecycle is not None
    ingress_token = runtime._capture_ingress_token(lifecycle)
    envelope = CoreDetectorEventEnvelope(
        event=DetectorRuntimeEvent(
            ingress=DetectorIngressIdentity(
                ingress_token=ingress_token,
                detector_epoch=detector.detector_epoch,
                sequence_no=1,
            ),
            candidate=DetectorCandidateKey(detector.detector_epoch, 1),
            kind="control_lane_failed",
        ),
        detector_ref=detector,
        lifecycle_ref=lifecycle,
        session_epoch=runtime._asr_session_epoch,
    )

    assert await runtime._handle_voice_input_control(
        "game_takeover",
        1,
    )
    runtime.send_status.reset_mock()
    await runtime._handle_asr_detector_dispatcher_failure(
        envelope,
        RuntimeError("old detector callback failed after game takeover"),
    )

    assert runtime._asr_route_mode == "independent"
    assert runtime._asr_lifecycle is lifecycle
    assert lifecycle.snapshot.state is VoiceLifecycleState.SUSPENDED
    runtime.send_status.assert_not_awaited()

async def test_rejected_explicit_control_permanently_disables_legacy_fallback(
    event: str,
    generation: int,
) -> None:
    runtime = _Runtime()
    runtime._asr_runtime.abort = AsyncMock()

    assert runtime._begin_voice_input_connection("explicit-socket") is True
    assert (
        await runtime._handle_voice_input_control(
            event,
            generation,
            owner="core",
            hard_muted=False,
            focus_suppressed=False,
        )
        is False
    )
    assert runtime._voice_lease_control_seen is True
    assert (
        await runtime._ensure_voice_input_session_authorized("explicit-socket") is False
    )
    await runtime._enqueue_audio_stream_data(
        {
            "input_type": "audio",
            "sample_rate_hz": 16_000,
            "data": [1] * 160,
        }
    )

    assert runtime._voice_lease_synchronized is False
    assert runtime._voice_lease_owner == "none"
    assert runtime._voice_input_accepts_pcm() is False
    assert runtime._audio_stream_queue.empty()
    runtime._asr_runtime.abort.assert_not_awaited()

async def test_final_records_segmented_wire_audio_committed_at_seal() -> None:
    runtime = _Runtime()
    session = SimpleNamespace(is_ready=True, provider_wire_audio_ms=0)
    runtime._asr_session = session
    epoch = runtime._asr_session_epoch
    await _start_and_seal_turn(runtime, "glm")
    # Segmented sessions advance the cumulative counter only at the seal-time
    # physical-segment commit, after the dispatcher's last per-chunk sample.
    session.provider_wire_audio_ms = 480

    await runtime._handle_independent_asr_final("hello", epoch, "glm")
    await runtime._wait_asr_transcript_dispatch_idle()

    metrics = runtime._asr_lifecycle.metrics
    assert metrics.provider_wire_audio_ms == 480
    assert metrics.cloud_audio_ms == 480
    assert runtime._asr_last_provider_wire_audio_ms == 480

async def test_core_swap_cancels_blocked_old_final_without_touching_new_state() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    old_epoch = runtime._asr_session_epoch
    old_core_session = runtime.session
    transcript_started = asyncio.Event()
    release_transcript = asyncio.Event()

    async def block_transcript(_text: str, **_kwargs: object) -> bool:
        transcript_started.set()
        await release_transcript.wait()
        return True

    runtime.handle_input_transcript.side_effect = block_transcript
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        old_epoch,
    )
    await runtime._handle_independent_asr_endpoint(old_epoch)
    await runtime._handle_independent_asr_final("old", old_epoch, "qwen")
    await transcript_started.wait()

    await runtime._close_independent_asr(next_route_mode="blocked")
    new_core_session = type("NewCore", (), {})()
    new_core_session.create_response = AsyncMock()
    new_core_session.handle_interruption = AsyncMock()
    runtime.session = new_core_session
    _install_ready_lifecycle(runtime, "qwen")
    new_lifecycle = runtime._asr_lifecycle
    assert new_lifecycle is not None
    expected_state = new_lifecycle.snapshot.state

    release_transcript.set()
    await asyncio.sleep(0)

    old_core_session.create_response.assert_not_awaited()
    new_core_session.create_response.assert_not_awaited()
    assert runtime._asr_lifecycle is new_lifecycle
    assert new_lifecycle.snapshot.state is expected_state
    assert runtime._asr_sealed_turn_token is None

async def test_close_invalidates_late_final_before_waiting_for_provider() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.close = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    old_epoch = runtime._asr_session_epoch

    await runtime._close_independent_asr(next_route_mode="blocked")
    await runtime._handle_independent_asr_final("late", old_epoch, "glm")

    asr.close.assert_awaited_once_with()
    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.create_response.assert_not_awaited()
    assert runtime._asr_route_mode == "blocked"

async def test_core_close_detaches_shared_state_before_registry_wait() -> None:
    runtime = _Runtime()
    registry_wait_started = asyncio.Event()
    release_registry_wait = asyncio.Event()

    async def block_registry_wait() -> None:
        registry_wait_started.set()
        await release_registry_wait.wait()

    old_pipeline = SimpleNamespace(close=AsyncMock())
    runtime._voice_input_audio_pipeline = old_pipeline
    runtime._independent_asr_provider = "old-provider"
    runtime._independent_asr_route_key = "old-core"
    runtime._voice_input_registry.wait_idle = AsyncMock(
        side_effect=block_registry_wait
    )
    runtime._asr_runtime.close = AsyncMock()

    closing = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await asyncio.wait_for(registry_wait_started.wait(), 1)

    detached_replacement = runtime._voice_input_audio_pipeline
    assert detached_replacement is not old_pipeline
    assert runtime._independent_asr_provider is None
    assert runtime._independent_asr_route_key is None

    runtime._begin_asr_route_operation()
    runtime._independent_asr_provider = "new-provider"
    runtime._independent_asr_route_key = "new-core"
    runtime._set_microphone_route("independent")
    release_registry_wait.set()
    await asyncio.wait_for(closing, 1)

    assert runtime._voice_input_audio_pipeline is detached_replacement
    assert runtime._independent_asr_provider == "new-provider"
    assert runtime._independent_asr_route_key == "new-core"
    assert runtime._asr_route_mode == "independent"
    runtime._asr_runtime.close.assert_not_awaited()

async def test_stale_close_waiting_for_pipeline_lock_cannot_replace_successor() -> None:
    runtime = _Runtime()
    runtime._asr_runtime.close = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "old-provider"
    runtime._independent_asr_route_key = "old-core"
    gate = _GateAsyncLock()
    runtime._voice_input_pipeline_transition_lock = gate

    closing = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await asyncio.wait_for(gate.requested.wait(), 1)
    runtime._begin_asr_route_operation()
    successor_pipeline = SimpleNamespace(
        nr_enabled=True,
        close=AsyncMock(),
    )
    runtime._voice_input_audio_pipeline = successor_pipeline
    runtime._independent_asr_provider = "new-provider"
    runtime._independent_asr_route_key = "new-core"
    runtime._set_microphone_route("independent")
    gate.release.set()
    await asyncio.wait_for(closing, 1)

    assert runtime._voice_input_audio_pipeline is successor_pipeline
    assert runtime._independent_asr_provider == "new-provider"
    assert runtime._independent_asr_route_key == "new-core"
    assert runtime._asr_route_mode == "independent"
    successor_pipeline.close.assert_not_awaited()
    runtime._asr_runtime.close.assert_not_awaited()

async def test_close_requires_callers_to_declare_the_next_route() -> None:
    parameter = inspect.signature(AsrRuntimeMixin._close_independent_asr).parameters[
        "next_route_mode"
    ]

    assert parameter.default is inspect.Parameter.empty

async def test_start_uses_current_core_route_only_after_provider_ready(
    monkeypatch,
) -> None:
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
        MagicMock(return_value=_selection("gemini")),
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        factory,
    )

    await runtime._start_independent_asr_if_enabled("audio")

    asr.connect.assert_awaited_once_with()
    assert runtime._asr_session is asr
    assert runtime._asr_provider == "gemini"
    assert runtime._asr_route_mode == "independent"
    assert factory.call_args.args == ("gemini",)
    assert factory.call_args.kwargs["selection"].provider_key == "gemini"

async def test_startup_close_window_is_blocked_before_settings_resolution(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class _OldAsr:
        is_ready = True

        async def close(self) -> None:
            close_started.set()
            await release_close.wait()

    runtime._asr_session = _OldAsr()
    runtime._asr_route_mode = "independent"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": False}),
    )

    start_task = asyncio.create_task(runtime._start_independent_asr_if_enabled("audio"))
    await asyncio.wait_for(close_started.wait(), 1)

    assert runtime._asr_route_mode == "blocked"
    assert (
        await runtime._route_microphone_audio(b"\x00\x00", sample_rate_hz=16_000)
        is True
    )

    release_close.set()
    await asyncio.wait_for(start_task, 1)
    assert runtime._asr_route_mode == "native"
    assert not hasattr(runtime._asr_runtime, "_asr_required")

async def test_failed_soniox_candidate_cannot_invalidate_successful_successor(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.websocket = type("WebSocket", (), {"send_json": AsyncMock()})()
    callbacks: list[dict[str, object]] = []

    failed_session = type("Soniox", (), {})()
    failed_session.connect = AsyncMock(side_effect=RuntimeError("provider detail"))
    failed_session.close = AsyncMock()
    successful_session = type("Soniox", (), {})()
    successful_session.connect = AsyncMock()
    successful_session.close = AsyncMock()
    soniox_selection = _selection("soniox", "provider")
    sessions = [failed_session, successful_session]

    def capture_partial(session, callback) -> None:
        session.partial_callback = callback

    def create_candidate(_core_type, *, selection, **kwargs):
        assert selection is soniox_selection
        callbacks.append(kwargs)
        return sessions[len(callbacks) - 1]

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
        MagicMock(
            side_effect=AssertionError(
                "Soniox recovery must not resolve another provider"
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "_create_asr_session_from_selection",
        create_candidate,
    )
    monkeypatch.setattr(runtime_module, "_attach_partial_callback", capture_partial)
    monkeypatch.setattr(runtime_module.asyncio, "sleep", AsyncMock())

    await runtime._start_independent_asr_if_enabled("audio")
    adopted_epoch = runtime._asr_session_epoch

    await callbacks[0]["on_input_transcript"]("late soniox final")
    await callbacks[0]["on_speech_activity"](SpeechActivityEvent.SPEECH_STARTED)
    await failed_session.partial_callback("late soniox preview")
    await callbacks[0]["on_connection_error"]("late soniox error")
    await asyncio.sleep(0)

    runtime.handle_input_transcript.assert_not_awaited()
    runtime.session.handle_interruption.assert_not_awaited()
    runtime.handle_new_message.assert_not_awaited()
    runtime.websocket.send_json.assert_not_awaited()
    successful_session.close.assert_not_awaited()
    assert runtime._asr_session is successful_session
    assert runtime._asr_provider == "soniox"
    assert runtime._asr_route_mode == "independent"
    assert runtime._asr_session_epoch == adopted_epoch

async def test_restart_default_attempts_follow_single_attempt_policy(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    sleep = AsyncMock()
    monkeypatch.setattr(runtime_module.asyncio, "sleep", sleep)
    candidates = _install_failing_restart_candidates(runtime, "qwen", failure_count=1)
    assert runtime._asr_lifecycle.provider_policy.connect_max_attempts == 1

    await runtime._restart_transport()
    while runtime._asr_runtime._asr_close_tasks:
        await asyncio.gather(
            *tuple(runtime._asr_runtime._asr_close_tasks),
            return_exceptions=True,
        )

    assert len(candidates) == 1
    candidates[0].connect.assert_awaited_once_with()
    candidates[0].close.assert_awaited_once_with()
    sleep.assert_not_awaited()
    statuses = [
        json.loads(call.args[0]) for call in runtime.send_status.await_args_list
    ]
    assert statuses[-1]["code"] == "ASR_INDEPENDENT_FAILED"
    assert "private restart connect detail" not in str(
        runtime.send_status.await_args_list
    )

async def test_not_ready_close_cannot_overwrite_replacement_generation() -> None:
    runtime = _Runtime()
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def close_old() -> None:
        close_started.set()
        await release_close.wait()

    old_session = SimpleNamespace(
        is_ready=False,
        close=AsyncMock(side_effect=close_old),
    )
    old_factory = MagicMock()
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    runtime._asr_session_factory = old_factory
    runtime._asr_transport_selection = _selection("qwen")

    restarting = asyncio.create_task(runtime._restart_transport(max_attempts=1))
    await asyncio.wait_for(close_started.wait(), 1)
    assert runtime._asr_session is None

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    new_factory = object()
    new_selection = object()
    runtime._asr_session_factory = new_factory
    runtime._asr_transport_selection = new_selection
    release_close.set()
    await asyncio.wait_for(restarting, 1)

    old_session.close.assert_awaited_once_with()
    old_factory.assert_not_called()
    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert runtime._asr_session_factory is new_factory
    assert runtime._asr_transport_selection is new_selection
    new_session.close.assert_not_awaited()

async def test_selection_failure_during_core_change_stays_blocked(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    runtime._independent_asr_route_key = "openai"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_asr_selection",
        MagicMock(side_effect=ValueError("invalid region configuration")),
    )

    await runtime._reconcile_independent_asr_after_core_change()

    assert runtime._independent_asr_route_key == "gemini"
    assert runtime._asr_route_mode == "blocked"
    assert runtime._asr_session is None
    assert "ASR_INDEPENDENT_FAILED" in runtime.send_status.await_args.args[0]

async def test_audio_activation_mirrors_only_dispatcher_accepted_provider_payload(
    accepted: bool,
    observer_raises: bool,
) -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    component = runtime._asr_runtime
    lifecycle = component._asr_lifecycle
    detector = component._asr_detector
    assert lifecycle is not None
    assert isinstance(detector, _ReadyDetector)
    if observer_raises:
        detector.observe_provider_audio.side_effect = RuntimeError("observer failed")
    token = component._capture_turn_token(lifecycle)
    payload = b"\x01\x00" * 320
    activate = MagicMock(return_value=accepted)
    component._asr_audio_dispatcher = SimpleNamespace(
        active_turn=None,
        activate=activate,
    )

    result = component._activate_asr_audio_dispatcher(
        lifecycle,
        token,
        buffered_pcm16=payload,
    )

    assert result is accepted
    activate.assert_called_once()
    assert activate.call_args.args[2] is payload
    if accepted:
        detector.observe_provider_audio.assert_called_once()
        assert detector.observe_provider_audio.call_args.args[0] is payload
        assert detector.observe_provider_audio.call_args.kwargs == {
            "sample_rate_hz": 16_000,
        }
    else:
        detector.observe_provider_audio.assert_not_called()

async def test_builder_failure_stays_blocked_and_never_sends_audio_to_omni(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.session.stream_audio = AsyncMock()
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
        MagicMock(side_effect=RuntimeError("private provider detail")),
    )

    await runtime._start_independent_asr_if_enabled("audio")
    consumed = await runtime._route_microphone_audio(
        b"\x00\x00",
        sample_rate_hz=16_000,
    )
    if not consumed:
        await runtime.session.stream_audio(b"\x00\x00")

    assert runtime._asr_route_mode == "blocked"
    assert runtime._asr_session is None
    assert consumed is True
    runtime.session.stream_audio.assert_not_awaited()
    assert "private provider detail" not in str(runtime.send_status.await_args)

async def test_hot_swap_does_not_retry_failed_same_core_route() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    runtime._asr_route_mode = "blocked"
    runtime._independent_asr_route_key = "gemini"
    runtime._start_independent_asr_if_enabled = AsyncMock()

    await runtime._reconcile_independent_asr_after_core_change()

    runtime._start_independent_asr_if_enabled.assert_not_awaited()

async def test_disabled_native_route_key_prevents_same_core_reconcile(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": False}),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "native"
    assert runtime._independent_asr_route_key == "gemini"
    runtime.is_active = True
    runtime.is_hot_swap_imminent = True
    runtime.session.stream_audio = AsyncMock()
    old_token = runtime._capture_ingress_token()
    assert runtime.hot_swap_audio_cache.append(
        _HotSwapAudioFrame(
            pcm16=b"\x01\x00" * 160,
            token=old_token,
            audio_stream_epoch=runtime._audio_stream_epoch,
        )
    )
    runtime._set_microphone_route("blocked")
    runtime._set_microphone_route("native")
    runtime._start_independent_asr_if_enabled = AsyncMock()
    await runtime._reconcile_independent_asr_after_core_change()
    runtime._start_independent_asr_if_enabled.assert_not_awaited()
    await runtime._flush_hot_swap_audio_cache()
    runtime.session.stream_audio.assert_awaited_once_with(b"\x01\x00" * 160)
    assert runtime._omni_mic_audio_bytes == 320

async def test_core_passes_only_configured_speaker_shadow_factory(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    factory = MagicMock()
    runtime._speaker_shadow_factory = factory
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

    await runtime._start_independent_asr_if_enabled("audio")

    assert start_mock.await_args.kwargs["speaker_shadow_factory"] is factory
    factory.assert_not_called()

async def test_connect_budget_is_opt_in(monkeypatch) -> None:
    # Every other caller (hot-swap, device change, the ordinary start) passes no
    # budget and must keep connecting exactly as before.
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": True}),
    )
    async def _ready(**_kwargs):
        # Epoch read at call time: the teardown that precedes the connect bumps
        # it, and a result stamped with the pre-call value reads as stale.
        return AsrStartResult(
            status=AsrStartStatus.READY,
            provider="qwen",
            session_epoch=runtime._capture_ingress_token().session_epoch,
        )

    start_mock = AsyncMock(side_effect=_ready)
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    await runtime._start_independent_asr_if_enabled("audio", handshake_override=True)

    assert runtime._asr_route_mode == "independent"
    start_mock.assert_awaited_once()

async def test_missing_independent_asr_setting_defaults_disabled(monkeypatch) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={}),
    )
    start_mock = AsyncMock(
        return_value=AsrStartResult(
            status=AsrStartStatus.FAILED,
            failure_code="ASR_START_STALE",
        )
    )
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)

    await runtime._start_independent_asr_if_enabled("audio")

    start_mock.assert_not_awaited()
    assert runtime._asr_route_mode == "native"

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

async def test_pipeline_toggle_during_failure_notify_keeps_ingress_closed() -> None:
    """A toggle must not reopen the microphone while the route is failing.

    `_voice_input_pipeline_failed` is the ingress gate, and ANY pipeline
    replacement clears it -- a noise-reduction toggle included. Once the
    notify phase left the transition lock, such a toggle can land while a
    backpressured status send is still in flight, i.e. while this failure
    still owns a blocked route whose lease has not been revoked. Frames would
    then be parsed, queued and run through the replacement DSP until the route
    discards them: bounded queue refilled, preprocessing burnt, ingress
    backpressure tripped, all during what must stay a fail-closed interval.
    """

    runtime = _Runtime()
    runtime.is_active = True
    runtime.is_hot_swap_imminent = False
    runtime.is_flushing_hot_swap_cache = False
    runtime.session.stream_audio = AsyncMock()
    runtime._set_microphone_route("independent")
    runtime._independent_asr_provider = "glm"
    assert runtime._begin_voice_input_connection("socket-a") is True
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_synchronized = True
    runtime._asr_runtime.abort = AsyncMock()
    runtime._asr_runtime.submit = AsyncMock()
    token = runtime._capture_ingress_token()
    status_started = asyncio.Event()
    release_status = asyncio.Event()

    async def backpressured_status(_payload) -> None:
        status_started.set()
        await release_status.wait()

    runtime.send_status = AsyncMock(side_effect=backpressured_status)
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=token,
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(status_started.wait(), 1)

    assert await asyncio.wait_for(
        runtime.apply_voice_input_noise_reduction(False),
        1,
    ) is True
    # Premise: the toggle really did clear the old gate, so anything still
    # holding ingress closed is the committed failure's own latch.
    assert runtime._voice_input_pipeline_failed is False
    replacement = runtime._voice_input_audio_pipeline
    replacement.process = AsyncMock()

    # Captured HERE, not before the failure. A live client keeps sending PCM
    # with a current token, so `_ingress_token_matches` passes and the frame
    # reaches the DSP without any lease check in between -- which is the whole
    # point. A token snapshotted before the route was blocked would be dropped
    # by the token fence instead, and this test would prove nothing.
    live_token = runtime._capture_ingress_token()
    assert runtime._ingress_token_matches(live_token) is True

    await runtime._process_microphone_stream_data(
        {"input_type": "audio", "sample_rate_hz": 48_000, "data": [1] * 480},
        ingress_token=live_token,
    )

    replacement.process.assert_not_awaited()
    runtime._asr_runtime.submit.assert_not_awaited()
    runtime.session.stream_audio.assert_not_awaited()

    release_status.set()
    await asyncio.wait_for(failure, 1)
    assert runtime._asr_route_mode == "blocked"
    assert runtime._voice_lease_connection_id == ""

async def test_pipeline_failure_from_replaced_connection_is_silent() -> None:
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
    failure = asyncio.create_task(
        runtime._fail_voice_input_pipeline(
            ingress_token=token,
            session_ref=runtime.session,
            audio_epoch=runtime._audio_stream_epoch,
            pipeline_ref=runtime._voice_input_audio_pipeline,
        )
    )
    await asyncio.wait_for(abort_started.wait(), 1)

    assert runtime._begin_voice_input_connection("replacement-connection")
    replacement_lease_state = (
        runtime._voice_lease_connection_id,
        runtime._voice_lease_generation,
        runtime._voice_lease_owner,
        runtime._voice_lease_synchronized,
    )
    release_abort.set()
    await asyncio.wait_for(failure, 1)

    assert (
        runtime._voice_lease_connection_id,
        runtime._voice_lease_generation,
        runtime._voice_lease_owner,
        runtime._voice_lease_synchronized,
    ) == replacement_lease_state
    runtime.send_status.assert_not_awaited()
    runtime._asr_runtime.abort.assert_awaited_once_with("audio_preprocessing_failed")

async def test_old_failure_callback_cannot_detach_replacement_runtime() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    old_lifecycle = runtime._asr_lifecycle
    old_detector = runtime._asr_detector
    blocked_started = asyncio.Event()
    release_blocked = asyncio.Event()

    async def block_old_lifecycle(payload: str) -> None:
        status = json.loads(payload)
        if (
            status.get("code") == "ASR_LIFECYCLE_STATE"
            and status.get("details", {}).get("state") == "blocked"
        ):
            blocked_started.set()
            await release_blocked.wait()

    runtime.send_status.side_effect = block_old_lifecycle
    old_epoch = runtime._asr_session_epoch
    failure_task = asyncio.create_task(
        runtime._handle_independent_asr_error(old_epoch, "qwen")
    )
    await asyncio.wait_for(blocked_started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    release_blocked.set()
    await asyncio.wait_for(failure_task, 1)
    await asyncio.sleep(0)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    assert old_lifecycle.snapshot.state is VoiceLifecycleState.OFF
    old_detector.close.assert_awaited_once_with()
    new_session.close.assert_not_awaited()
    new_detector.close.assert_not_awaited()
    assert runtime._asr_route_mode == "independent"

async def test_stale_detector_feed_exception_cannot_fail_new_generation() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingDetector(_ReadyDetector):
        async def feed(self, _pcm16: bytes, **_kwargs):
            started.set()
            await release.wait()
            raise RuntimeError("old detector failed")

    runtime._asr_detector = _BlockingDetector()
    ingress = runtime._capture_ingress_token()
    runtime._asr_runtime._asr_current_ingress_token = ingress
    submit = asyncio.create_task(
        runtime._asr_runtime.submit(
            ProcessedVoiceFrame(b"\x01\x00" * 160, 16_000, 0.8, True),
            ingress_token=ingress,
        )
    )
    await asyncio.wait_for(started.wait(), 1)

    new_session, new_lifecycle, new_detector = _install_replacement_runtime_generation(
        runtime, "qwen"
    )
    release.set()
    result = await asyncio.wait_for(submit, 1)

    assert result.status is AsrSubmitStatus.STALE
    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is new_lifecycle
    assert runtime._asr_detector is new_detector
    new_session.close.assert_not_awaited()
    runtime.send_status.assert_not_awaited()

async def test_same_epoch_reconnect_survives_old_abort_release() -> None:
    runtime = _Runtime()
    old_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    runtime._asr_session = old_session
    _install_ready_lifecycle(runtime, "qwen")
    lifecycle = runtime._asr_lifecycle
    old_detector = runtime._asr_detector
    assert lifecycle is not None
    assert old_detector is not None
    release_started = asyncio.Event()
    release_old_lease = asyncio.Event()

    class _BlockingLease:
        async def release(self) -> None:
            release_started.set()
            await release_old_lease.wait()

    runtime._asr_smart_turn_lease = _BlockingLease()
    aborting = asyncio.create_task(runtime._asr_runtime.abort("test_abort"))
    await asyncio.wait_for(release_started.wait(), 1)

    new_session = SimpleNamespace(is_ready=True, close=AsyncMock())
    new_detector = _ReadyDetector()
    new_lease = _TestSmartTurnLease(object())
    runtime._asr_session = new_session
    runtime._asr_detector = new_detector
    runtime._asr_smart_turn_lease = new_lease
    lifecycle.invalidate_transport()
    runtime._asr_runtime._asr_current_ingress_token = runtime._capture_ingress_token()
    release_old_lease.set()
    await asyncio.wait_for(aborting, 1)

    assert runtime._asr_session is new_session
    assert runtime._asr_lifecycle is lifecycle
    assert runtime._asr_detector is new_detector
    assert runtime._asr_smart_turn_lease is new_lease
    old_session.close.assert_awaited_once_with()
    new_session.close.assert_not_awaited()
    new_detector.reset.assert_not_awaited()
    assert new_lease.released is False

async def test_free_core_always_uses_native_asr_regardless_of_toggle(
    monkeypatch,
    persisted_enabled: bool,
    handshake_enabled: bool | None,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "free"
    runtime.session.stream_audio = AsyncMock()
    monkeypatch.setattr(
        core_module,
        "aload_global_conversation_settings",
        AsyncMock(return_value={"independentAsrEnabled": persisted_enabled}),
    )
    start_mock = AsyncMock()
    monkeypatch.setattr(runtime._asr_runtime, "start", start_mock)
    runtime.set_independent_asr_handshake(handshake_enabled)

    await runtime._start_independent_asr_if_enabled("audio")

    assert runtime._asr_route_mode == "native"
    assert runtime._independent_asr_route_key == "free"
    assert runtime._independent_asr_provider is None
    start_mock.assert_not_awaited()
    assert "ASR_INDEPENDENT_DISABLED" in runtime.send_status.await_args.args[0]
    assert "ASR_INDEPENDENT_UNAVAILABLE" not in runtime.send_status.await_args.args[0]

    assert await runtime._route_microphone_audio(
        b"\x01\x00" * 160,
        sample_rate_hz=16_000,
    ) is True
    runtime.session.stream_audio.assert_awaited_once_with(b"\x01\x00" * 160)

async def test_provider_error_without_audio_closes_and_blocks_omni() -> None:
    runtime = _Runtime()
    asr = type("Asr", (), {})()
    asr.close = AsyncMock()
    runtime._asr_session = asr
    runtime._asr_route_mode = "independent"
    epoch = runtime._asr_session_epoch

    await runtime._handle_independent_asr_error(epoch, "glm")
    await asyncio.sleep(0)

    assert runtime._asr_session_epoch == epoch + 1
    assert runtime._asr_route_mode == "blocked"
    asr.close.assert_awaited_once_with()

async def test_injection_failure_is_reported_once_without_provider_body() -> None:
    runtime = _Runtime()
    runtime.session.create_response.side_effect = RuntimeError("sensitive response")
    await _start_and_seal_turn(runtime, "gemini")

    await runtime._handle_independent_asr_final(
        "hello",
        runtime._asr_session_epoch,
        "gemini",
    )
    await runtime._wait_asr_transcript_dispatch_idle()

    status_payloads = [call.args[0] for call in runtime.send_status.await_args_list]
    assert any("ASR_INDEPENDENT_INJECTION_FAILED" in item for item in status_payloads)
    assert "sensitive response" not in str(status_payloads)
    runtime.session.create_response.assert_awaited_once_with("hello")

async def test_old_core_close_cannot_clear_new_pipeline_or_provider() -> None:
    runtime = _Runtime()
    runtime_close_entered = asyncio.Event()
    release_runtime_close = asyncio.Event()

    async def block_runtime_close() -> None:
        runtime_close_entered.set()
        await release_runtime_close.wait()

    runtime._asr_runtime.close = AsyncMock(side_effect=block_runtime_close)
    old_pipeline = runtime._voice_input_audio_pipeline
    old_pipeline.close = AsyncMock()
    runtime._independent_asr_provider = "old-provider"
    runtime._independent_asr_route_key = "old-core"
    runtime._set_microphone_route("independent")

    closing = asyncio.create_task(
        runtime._close_independent_asr(next_route_mode="blocked")
    )
    await asyncio.wait_for(runtime_close_entered.wait(), 1)
    new_pipeline = runtime._voice_input_audio_pipeline
    runtime._begin_asr_route_operation()
    runtime._independent_asr_provider = "new-provider"
    runtime._independent_asr_route_key = "new-core"
    runtime._set_microphone_route("independent")
    release_runtime_close.set()
    await asyncio.wait_for(closing, 1)

    old_pipeline.close.assert_awaited_once_with()
    assert runtime._voice_input_audio_pipeline is new_pipeline
    assert runtime._independent_asr_provider == "new-provider"
    assert runtime._independent_asr_route_key == "new-core"
    assert runtime._asr_route_mode == "independent"

async def test_partial_preview_requires_current_core_lease() -> None:
    runtime = _Runtime()
    websocket = type("WebSocket", (), {})()
    websocket.send_json = AsyncMock()
    runtime.websocket = websocket
    runtime._set_microphone_route("independent")
    epoch = runtime._asr_session_epoch
    token = VoiceTurnToken(
        ingress=runtime._capture_ingress_token(),
        turn_id=1,
    )
    stale_token = VoiceTurnToken(
        ingress=replace(token.ingress, session_epoch=epoch + 1),
        turn_id=token.turn_id,
    )

    runtime._voice_lease_owner = "game"
    await runtime._send_core_asr_preview(
        VoicePartialEvent(turn_token=token, text="game")
    )
    runtime._voice_lease_owner = "core"
    runtime._voice_lease_hard_muted = True
    await runtime._send_core_asr_preview(
        VoicePartialEvent(turn_token=token, text="muted")
    )
    runtime._voice_lease_hard_muted = False
    runtime._voice_lease_focus_suppressed = True
    await runtime._send_core_asr_preview(
        VoicePartialEvent(turn_token=token, text="focused")
    )
    runtime._voice_lease_focus_suppressed = False
    await runtime._send_core_asr_preview(
        VoicePartialEvent(turn_token=stale_token, text="stale")
    )
    await runtime._send_core_asr_preview(
        VoicePartialEvent(turn_token=token, text="current")
    )

    websocket.send_json.assert_awaited_once_with(
        {
            "type": "user_transcript_preview",
            "text": "current",
            "turn_id": f"asr-preview-{epoch}",
        }
    )

async def test_runtime_failure_still_fences_a_competing_newer_operation() -> None:
    # Re-basing the identity must not weaken the fence it exists for: a NEWER
    # route operation landing during this handler's own transition still has to
    # stop the revoke, because _revoke_voice_input_connection calls
    # _invalidate_asr_start() and would cancel that newer start.
    runtime = _Runtime()
    runtime._set_microphone_route("independent")
    runtime._voice_lease_connection_id = "socket-a"
    original_set_route = runtime._set_microphone_route

    def _set_route_then_supersede(mode: str) -> None:
        original_set_route(mode)
        runtime._begin_asr_route_operation()

    runtime._set_microphone_route = _set_route_then_supersede

    await runtime._handle_core_asr_failure(
        AsrFailureEvent(
            code="ASR_INDEPENDENT_FAILED",
            provider="current-provider",
            session_epoch=runtime._asr_session_epoch,
        )
    )

    assert runtime._voice_lease_connection_id == "socket-a"

async def test_cold_start_with_unclaimed_lease_still_routes(
    monkeypatch, enabled: bool
) -> None:
    """The bundled frontend flips the lease owner to "core" only after
    session_started, so route setup must not require owner=="core": gating
    the start on lease state would leave every cold start blocked."""

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
        AsyncMock(return_value={"independentAsrEnabled": enabled}),
    )

    await runtime._start_independent_asr_if_enabled("audio")

    if enabled:
        assert runtime._asr_route_mode == "independent"
        assert runtime._independent_asr_provider == "qwen"
    else:
        assert runtime._asr_route_mode == "native"

async def test_current_game_release_still_aborts_and_resumes_once() -> None:
    runtime = _Runtime()
    runtime._voice_lease_connection_id = "connection"
    runtime._voice_lease_generation = 1
    runtime._voice_lease_owner = "game"
    runtime._asr_runtime.abort = AsyncMock()
    runtime._asr_runtime.resume = AsyncMock()

    await runtime._apply_voice_lease_state(
        owner="core",
        hard_muted=False,
        focus_suppressed=False,
        reason="game_release",
        force_abort=True,
    )

    runtime._asr_runtime.abort.assert_awaited_once_with("game_release")
    runtime._asr_runtime.resume.assert_awaited_once_with("game_release")

async def test_cancelled_lease_resync_send_retries_same_episode() -> None:
    runtime = _Runtime()
    assert runtime._begin_voice_input_connection("chat-window") is True
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def block_send(_message: str) -> None:
        send_started.set()
        await release_send.wait()

    runtime.send_status = AsyncMock(side_effect=block_send)
    first = asyncio.create_task(runtime._maybe_signal_voice_lease_resync())
    await asyncio.wait_for(send_started.wait(), 1)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert runtime._voice_lease_resync_signal_state is None
    runtime.send_status = AsyncMock()
    await runtime._maybe_signal_voice_lease_resync()

    runtime.send_status.assert_awaited_once()
    assert runtime._voice_lease_resync_signal_state is not None

async def test_blocked_text_episode_keeps_session_identity_reference() -> None:
    runtime = _Runtime()
    runtime.input_mode = "text"
    runtime._set_microphone_route("blocked")
    session = runtime.session

    episode = runtime._blocked_text_mode_microphone_episode()

    assert episode is not None
    assert episode[-1] is session

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

async def test_idle_backpressure_new_speech_still_wakes_adopted_session(
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
    stale_token = runtime._capture_ingress_token()
    component._asr_current_ingress_token = stale_token
    on_activity = callbacks[0]["on_speech_activity"]

    await component._handle_audio_ingress_backpressure(stale_token)

    # New speech re-arms the current ingress token through submit() before
    # the provider observes it; the adopted session must then wake normally.
    component._asr_current_ingress_token = runtime._capture_ingress_token()
    await on_activity(SpeechActivityEvent.SPEECH_STARTED)

    assert lifecycle.snapshot.state is VoiceLifecycleState.ACTIVE
    assert component._asr_turn_prepared is True
    runtime.handle_new_message.assert_awaited_once()

async def test_transport_restart_task_failure_is_logged(caplog) -> None:
    runtime = _Runtime()
    component = runtime._asr_runtime

    async def failing_restart(_operation) -> None:
        raise RuntimeError("restart boom")

    component._run_transport_connect_operation = failing_restart
    with caplog.at_level(logging.ERROR, logger="main_logic.asr_client._infra"):
        component._ensure_transport_restart_task()
        task = component._asr_transport_task
        assert task is not None
        await asyncio.wait([task])
        await asyncio.sleep(0)

    assert "independent-asr-transport-restart" in caplog.text
    assert "restart boom" in caplog.text

async def test_failed_detector_construction_closes_created_speaker_shadow(
    monkeypatch,
) -> None:
    import main_logic.asr_client.runtime as runtime_module

    runtime = _Runtime()
    selection = _selection("qwen", "provider")
    session = SimpleNamespace(
        is_ready=True,
        connect=AsyncMock(),
        close=AsyncMock(),
    )
    shadow = SimpleNamespace(close=AsyncMock())
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
    monkeypatch.setattr(
        runtime_module,
        "DetectorRuntime",
        MagicMock(side_effect=RuntimeError("detector construction failed")),
    )

    result = await runtime._asr_runtime.start(
        route_key="qwen",
        resource_optimization_enabled=True,
        speaker_shadow_factory=lambda: shadow,
    )

    assert result.status in {AsrStartStatus.FAILED, AsrStartStatus.UNAVAILABLE}
    shadow.close.assert_awaited_once_with()

async def test_provider_fence_failure_does_not_accept_final() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    detector.complete_provider_candidate.return_value = None
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)

    await runtime._handle_independent_asr_final(
        "must-not-publish",
        epoch,
        "openai",
    )

    statuses = [json.loads(call.args[0]) for call in runtime.send_status.await_args_list]
    codes = [payload["code"] for payload in statuses]
    assert codes.count("ASR_ENDPOINTING_FAILED") == 1
    assert runtime._asr_accepted_final_keys == {}
    runtime.handle_input_transcript.assert_not_awaited()
    assert runtime._asr_route_mode == "blocked"

async def test_provider_overflow_lock_then_final_preserves_accepted_final() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    detector = runtime._asr_detector
    assert isinstance(detector, _ReadyDetector)
    discard_started = asyncio.Event()
    discard_release = asyncio.Event()

    async def discard_provider_successor(_fence) -> bool:
        discard_started.set()
        await discard_release.wait()
        return True

    detector.discard_provider_successor.side_effect = discard_provider_successor
    detector.complete_provider_candidate.return_value = False
    epoch = runtime._asr_session_epoch
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_endpoint(epoch)
    ingress_token = runtime._asr_runtime._asr_current_ingress_token
    assert ingress_token is not None

    overflow_task = asyncio.create_task(
        runtime._handle_audio_ingress_backpressure(
            ingress_token,
            observed_state=VoiceLifecycleState.DRAINING,
        )
    )
    await asyncio.wait_for(discard_started.wait(), 1)
    final_task = asyncio.create_task(
        runtime._handle_independent_asr_final("first", epoch, "openai")
    )
    await asyncio.sleep(0)
    assert final_task.done() is False
    discard_release.set()
    await asyncio.gather(overflow_task, final_task)
    await runtime._wait_asr_transcript_dispatch_idle()

    detector.discard_provider_successor.assert_awaited_once()
    detector.complete_provider_candidate.assert_awaited_once()
    runtime.handle_input_transcript.assert_awaited_once_with(
        "first",
        is_voice_source=True,
        source="independent_asr",
        metadata={"provider": "openai"},
        source_game_route_identity=None,
    )
    assert runtime._asr_lifecycle.snapshot.state is VoiceLifecycleState.WARM_IDLE
    assert runtime._asr_accepted_final_keys

async def test_native_route_leaves_provider_capability_routing_inside_session() -> None:
    """Core selects the ASR strategy, while session capability keeps legacy behavior."""
    runtime = _Runtime()
    runtime.session._supports_native_image = False
    runtime.session.set_visual_delivery_mode = MagicMock()

    runtime._set_microphone_route("native")

    delivered_mode = runtime.session.set_visual_delivery_mode.call_args.args[0]
    assert getattr(delivered_mode, "value", delivered_mode) == "native"

async def test_independent_multimodal_turn_rejects_delayed_prior_capture() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    captured_before_turn = time.monotonic() - 1.0
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=79)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # Validation completes after prepare, so generation alone looks current;
    # the ingress capture time must keep this prior image out of the new turn.
    assert runtime._stage_independent_visual_frame(
        "delayed-prior-frame",
        source="screen",
        request_id="screen-delayed",
        captured_at=captured_before_turn,
    )
    assert record.last_frame is None
    assert runtime._snapshot_core_multimodal_turn(turn_id, "new question") is None

    assert runtime._stage_independent_visual_frame(
        "current-turn-frame",
        source="camera",
        request_id="camera-current",
        captured_at=record.started_at,
    )
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "new question")

    assert turn is not None
    assert turn.images == ("current-turn-frame",)
    assert turn.captured_at == record.started_at

async def test_ownership_lost_between_the_freeze_check_and_the_provider_call(
    delivery,
) -> None:
    """One check up front is not enough; every await is another window.

    Between the post-freeze check and the actual provider call there is still
    the transcript send, preview restoration, the swap barrier and (on the
    handoff path) preparing a replacement session. A successor prepared in any
    of those windows owns the frames, so the last synchronous point before the
    call has to look again.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    runtime._asr_route_mode = "independent"
    runtime.session.submit_multimodal_turn = AsyncMock()
    runtime.session.submit_external_voice_turn = AsyncMock()
    runtime._handoff_to_offline_vlm_and_submit = AsyncMock(return_value=True)
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

    def _take_ownership_then_report_delivery():
        # 这一步排在冻结后那次检查**之后**、真正调 provider 之前。
        successor = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(),
            turn_id=token.turn_id + 1,
        )
        runtime._begin_core_multimodal_turn(
            f"asr-{successor.ingress.session_epoch}-{successor.turn_id}",
            successor,
        )
        return delivery

    runtime.session.get_multimodal_turn_delivery = MagicMock(
        side_effect=_take_ownership_then_report_delivery
    )

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=token,
            provider="openai",
            text="look here",
        )
    )

    assert record.invalidated.is_set()
    runtime.session.submit_multimodal_turn.assert_not_awaited()
    runtime._handoff_to_offline_vlm_and_submit.assert_not_awaited()
    runtime.session.submit_external_voice_turn.assert_awaited_once()
    assert "look here" in runtime.session.submit_external_voice_turn.await_args.args

async def test_visual_validation_wait_timeout_does_not_cancel_image_task() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    runtime._independent_visual_frame_ttl_s = 0.01
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=81)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    release = asyncio.Event()
    validation_task = asyncio.create_task(release.wait())
    assert runtime._track_independent_visual_validation_task(
        validation_task,
        captured_at=record.started_at,
    )

    await runtime._await_independent_visual_validation_tasks(turn_id)

    assert not validation_task.done()
    release.set()
    await validation_task

async def test_handoff_failure_never_falls_back_to_transcript_only() -> None:
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "qwen")
    runtime._asr_route_mode = "independent"
    runtime.session.get_multimodal_turn_delivery = MagicMock(
        return_value="handoff_required"
    )
    runtime.session.submit_external_voice_turn = AsyncMock()
    runtime._handoff_to_offline_vlm_and_submit = AsyncMock(return_value=False)
    runtime.is_preparing_new_session = True
    runtime.message_cache_for_new_session = [
        {"role": "Test", "text": "earlier reply"}
    ]

    async def cache_current_final(*_args, **_kwargs) -> bool:
        runtime.message_cache_for_new_session.append(
            {"role": "master", "text": "what is this"}
        )
        return True

    runtime.handle_input_transcript.side_effect = cache_current_final
    token = runtime._asr_runtime._capture_turn_token(runtime._asr_lifecycle)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    assert runtime._stage_independent_visual_frame(
        "raw-frame",
        source="camera",
        request_id="camera-1",
        captured_at=time.monotonic(),
    )

    await runtime._dispatch_core_asr_transcript(
        VoiceTranscriptEvent(
            turn_token=token,
            provider="qwen",
            text="what is this",
        )
    )

    runtime._handoff_to_offline_vlm_and_submit.assert_awaited_once()
    handoff_kwargs = (
        runtime._handoff_to_offline_vlm_and_submit.await_args.kwargs
    )
    assert handoff_kwargs["prepared_session"] is runtime.session
    assert handoff_kwargs["cached_turns_before_final"] == [
        {"role": "Test", "text": "earlier reply"}
    ]
    runtime.session.submit_external_voice_turn.assert_not_awaited()
    assert "ASR_MULTIMODAL_TURN_FAILED" in str(
        runtime.send_status.await_args_list
    )

async def test_frame_captured_before_the_endpoint_survives_late_validation() -> None:
    """Validation finishing after DRAINING must not discard a spoken-window frame."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=93)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]
    captured_while_speaking = record.started_at

    # 端点先到，这帧的校验任务才跑完 —— 拍摄时它还在说话，必须留下。
    _seal_utterance(runtime)
    runtime._mark_independent_asr_endpoint_if_sealed()
    assert runtime._stage_independent_visual_frame(
        "late-validated-frame",
        source="screen",
        request_id="screen-late",
        captured_at=captured_while_speaking,
    )

    turn = runtime._snapshot_core_multimodal_turn(turn_id, "what is that")

    assert turn is not None
    assert turn.images == ("late-validated-frame",)

async def test_previous_turn_seal_in_the_same_tick_is_not_this_turn_cutoff() -> None:
    """A previous turn's seal in the same tick is not this turn's cutoff.

    monotonic is ~15ms coarse on Windows (_begin_core_multimodal_turn in this
    same module already falls back to a generation criterion for exactly this
    reason), so the previous turn's seal and the successor record's
    registration can land in one tick and compare equal. Stamping it onto the
    successor makes every later frame fail accepts(); once the opening frame
    expires, a slightly longer utterance degrades to text-only and the user
    sees "she only caught the instant I started talking".

    The criterion is turn identity, not the timestamp -- see the dual below.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=97)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 上一轮的封口副本，时刻与本轮 record 的注册时刻**相等**（同一个 tick），
    # 但身份是上一轮的。live 字段是空的——PROVIDER_FINAL 已经把它清掉了，这正是
    # 保留副本存在的原因。
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = record.registered_at
    runtime._asr_last_turn_endpointed_key = "asr-0-96"
    assert runtime._asr_last_turn_endpointed_key != record.turn_id

    assert runtime._stage_independent_visual_frame(
        "opening-frame",
        source="screen",
        request_id="screen-opening",
        captured_at=record.started_at,
    )
    # 发声中段拍的帧——如果上一轮的封口被误绑成本轮截止点，它会被 accepts() 拒掉。
    assert runtime._stage_independent_visual_frame(
        "middle-frame",
        source="screen",
        request_id="screen-middle",
        captured_at=record.registered_at + 1.0,
    )

    assert record.endpoint_at is None, (
        "上一轮的封口被盖到了后继回合上：相等必须归上一轮"
    )
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "这是什么")

    assert turn is not None
    assert "middle-frame" in turn.images

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

async def test_prerecord_validation_stash_is_bounded() -> None:
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset
    gate = asyncio.Event()

    async def pending_validation() -> None:
        await gate.wait()

    tasks = [asyncio.create_task(pending_validation()) for _ in range(40)]
    await asyncio.sleep(0)
    for task in tasks:
        runtime._track_independent_visual_validation_task(
            task,
            captured_at=onset + 0.01,
        )

    assert len(runtime._prerecord_visual_validations) <= 8

    gate.set()
    await asyncio.gather(*tasks)

async def test_retained_seal_predating_registration_is_not_this_turn_cutoff() -> None:
    """Second line of defence behind the onset stamp.

    A retained seal survives across turns, so "is it >= started_at" cannot tell
    whether it belongs to this turn — an overlapping successor's onset is even
    recorded BEFORE the predecessor sealed. The floor for the retained copy is
    therefore the moment the record was registered: the previous turn's seal
    necessarily happened before that.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    previous_onset = time.monotonic() - 1.0
    previous_seal = previous_onset + 0.5
    # 模拟"激活 pending turn 时忘了补 onset"：留着上一轮的 onset。
    runtime._asr_turn_onset_at = previous_onset
    runtime._asr_turn_endpointed_at = None
    runtime._asr_last_turn_endpointed_at = previous_seal

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=102)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    assert runtime._stage_independent_visual_frame(
        "post-seal-frame",
        source="screen",
        request_id="screen-post",
        captured_at=previous_seal + 0.5,
    )
    # 即使 onset 是上一轮的残值（第一道防线失效），上一轮的封口也不能成为本轮的
    # 截止点 —— 它发生在本 record 注册之前。
    assert record.endpoint_at is None
    turn = runtime._snapshot_core_multimodal_turn(turn_id, "survives")

    assert turn is not None
    assert turn.images == ("post-seal-frame",)

def test_overlap_replay_carries_the_real_onset_not_the_replay_instant() -> None:
    """The overlap replay happens long after the user actually resumed speaking.

    A provider-VAD successor utterance can reach Core while the previous turn
    is still ACTIVE; its onset is remembered and replayed only once the delayed
    final arrives. Stamping the replay instant as the onset would classify
    everything captured in between as "after the user spoke", so the successor
    utterance loses the frames it was actually about.
    """
    import inspect

    from main_logic.asr_client import runtime as asr_runtime_module

    source = inspect.getsource(asr_runtime_module).splitlines()

    record = [
        index
        for index, line in enumerate(source)
        if "self._asr_overlap_onset_token = self._asr_current_ingress_token" in line
    ]
    assert record, "overlap onset token is never recorded"
    for index in record:
        window = chr(10).join(source[index : index + 3])
        assert "self._asr_overlap_onset_at = detected_at" in window, (
            f"line {index + 1}: the overlap onset instant must be recorded "
            f"alongside its token, got: {window!r}"
        )

    # 只认「把 SPEECH_RESUMED 重放给 _handle_independent_asr_activity」那一处，
    # 不要把无关的集合字面量里出现的同名枚举也算进来。
    # 只认 overlap **重放**那一处：它由「兑付一次 completed-overlap credit」的那段
    # 代码驱动。同名枚举在别处也会被正常派发（那些是真实发生的时刻，用进函数时钟
    # 是对的），不能一并要求它们交接 onset。
    # overlap 有**两条**重放路径：credit 兑付那条，和 provider final 到达时的直接
    # 重放。两条都必须把真实开口时刻交给确认分支 —— 只修其中一条正是上一轮的漏。
    replay = [
        index
        for index, line in enumerate(source)
        if "await self._handle_independent_asr_activity(" in line
        and "SpeechActivityEvent.SPEECH_RESUMED," in source[index + 1]
    ]
    assert len(replay) >= 2, f"expected both overlap replay paths, got {len(replay)}"
    for index in replay:
        window = chr(10).join(source[max(0, index - 30) : index])
        # credit 兑付那条按队列 popleft（每张 credit 一个时刻），直接重放那条用它
        # 自己捕获的 overlap_onset_at。两条都必须交接。
        assert (
            "self._asr_pending_speech_onset_at = replay_onset_at" in window
            or "self._asr_pending_speech_onset_at = overlap_onset_at" in window
        ), (
            f"line {index + 1}: every overlap replay must hand the recorded "
            f"onset to the confirmation path, got: {window!r}"
        )

async def test_prerecord_task_stash_keeps_the_earliest_validation() -> None:
    """Evicting the oldest task drops the opening frame of the utterance.

    The router registers validation tasks in capture order, so the oldest entry
    is the earliest capture. If a short utterance reaches final before that
    evicted task completes, the final freeze cannot wait for it and the record
    is abandoned before the opening frame lands.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    onset = time.monotonic()
    runtime._asr_turn_onset_at = onset
    gate = asyncio.Event()

    async def pending_validation() -> None:
        await gate.wait()

    tasks = [asyncio.create_task(pending_validation()) for _ in range(30)]
    await asyncio.sleep(0)
    for index, task in enumerate(tasks):
        runtime._track_independent_visual_validation_task(
            task,
            captured_at=onset + 0.001 * index,
        )

    stash = runtime._prerecord_visual_validations
    assert len(stash) <= 8
    kept = sorted(stash.values())
    # 时间上的首尾都必须活着 —— 淘汰只能发生在中间。
    assert kept[0] == onset
    assert kept[-1] == onset + 0.001 * 29

    gate.set()
    await asyncio.gather(*tasks)

async def test_a_dispatching_record_outlives_the_cap() -> None:
    """The cap must never be the thing that drops an accepted final.

    Raising the limit only moves the failure to a higher overlap count. What
    decides eviction is whether that record's own dispatch has finished -- the
    dict is bounded by removals from each dispatch's own finally, and a run of
    prepares long enough to hit the cap must skip anything mid-dispatch.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    running = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=501)
    running_id = f"asr-{running.ingress.session_epoch}-{running.turn_id}"
    runtime._begin_core_multimodal_turn(running_id, running)
    running_record = runtime._core_multimodal_turns[running_id]
    running_record.dispatch_started = True

    # 远多于上限的后继 prepare。
    for turn_id in range(502, 502 + _MAX_LIVE_TURN_RECORDS * 3):
        token = VoiceTurnToken(
            ingress=runtime._capture_ingress_token(), turn_id=turn_id
        )
        runtime._begin_core_multimodal_turn(
            f"asr-{token.ingress.session_epoch}-{token.turn_id}", token
        )

    assert runtime._core_multimodal_turns.get(running_id) is running_record
    # 没在派发的那些仍然有界。
    assert len(runtime._core_multimodal_turns) <= _MAX_LIVE_TURN_RECORDS

async def test_invalidated_record_does_not_hand_over_its_frames() -> None:
    """A superseded turn keeps its words but not the successor's frames."""
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    first = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=303)
    first_id = f"asr-{first.ingress.session_epoch}-{first.turn_id}"
    runtime._begin_core_multimodal_turn(first_id, first)
    record = runtime._core_multimodal_turns[first_id]
    assert runtime._stage_independent_visual_frame(
        "first-turn-frame",
        source="screen",
        request_id="screen-first",
        captured_at=record.started_at,
    )
    assert runtime._snapshot_core_multimodal_turn(first_id, "first") is not None

    second = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=304)
    runtime._begin_core_multimodal_turn(
        f"asr-{second.ingress.session_epoch}-{second.turn_id}", second
    )

    # 记录还在（话要留住），但视觉所有权已经交给后继回合 —— 走纯文本提交。
    assert runtime._core_multimodal_turns.get(first_id) is record
    assert runtime._snapshot_core_multimodal_turn(first_id, "first") is None

async def test_frames_after_a_sealed_turn_are_kept_for_the_successor() -> None:
    """A sealed record is done taking frames, so it must not block the buffer.

    Between the endpoint and the provider final, the record is sealed but not
    yet invalidated -- the successor cannot be prepared until that final lands.
    Frames captured in that window fail the sealed record's ``accepts()``
    (they are past its endpoint), so if it still counts as the active record
    they are neither attached nor retained: the successor turn loses its
    opening and middle frames and keeps only the latest-frame cache.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"

    token = VoiceTurnToken(ingress=runtime._capture_ingress_token(), turn_id=901)
    turn_id = f"asr-{token.ingress.session_epoch}-{token.turn_id}"
    runtime._begin_core_multimodal_turn(turn_id, token)
    record = runtime._core_multimodal_turns[turn_id]

    # 这一轮说完了：封口，但 provider final 还没回来，所以还没作废。
    record.endpoint_at = record.started_at + 1.0
    assert not record.invalidated.is_set()
    assert runtime._active_multimodal_turn_record() is None

    # 后继开口，帧在封口之后拍到。
    assert runtime._stage_independent_visual_frame(
        "successor-opening-frame",
        source="screen",
        request_id="screen-successor",
        captured_at=record.endpoint_at + 0.5,
    )

    # 它进不了已封口的那条记录，但必须被留住给后继。
    assert [f.image_b64 for f in runtime._prerecord_visual_frames] == [
        "successor-opening-frame"
    ]

async def test_replay_drops_the_pending_slot_when_the_transport_identity_moves_on() -> None:
    """A drifted runtime identity must not strand the pending confirmation.

    _send_asr_lifecycle_state() swallows delivery exceptions and returns
    _runtime_identity_matches(), so a false return means the runtime identity
    moved on -- and _restart_transport / _close_transport_only swap
    _asr_session and bump transport_generation without bumping the epoch or
    running _reset_asr_turn_state(). Holding the pending confirmation across
    that return strands it: the compensation already transitioned to ACTIVE,
    and both redemption sites gate on PREWARMING, so nothing ever collects it.
    The next unrelated utterance then adopts the stale onset as its visual
    ownership boundary, and the poisoned flag pins pending_before True so the
    overlap compensation silently stops firing.

    The real onset is already committed to _asr_turn_onset_at before the
    broadcast, so clearing the slot on confirmation loses nothing.
    """
    runtime = _Runtime()
    _install_ready_lifecycle(runtime, "openai")
    epoch = runtime._asr_session_epoch
    component = runtime._asr_runtime

    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_STARTED,
        epoch,
    )
    await runtime._handle_independent_asr_activity(
        SpeechActivityEvent.SPEECH_RESUMED,
        epoch,
    )
    assert component._asr_overlap_onset_at is not None
    await runtime._handle_independent_asr_endpoint(epoch)

    # monotonic 在这台机器上一整个测试跑下来只走一格，靠时钟自然推进区分不了
    # 「陈旧 onset」和「新回合 onset」。按仓库既有做法直接注入一个明显靠前的
    # 时刻，后面那条继承断言才有分辨力。
    recorded_onset = time.monotonic() - 5.0
    component._asr_overlap_onset_at = recorded_onset

    component._asr_session.is_ready = False
    lifecycle_ref = runtime._asr_lifecycle

    # ACTIVE 广播飞在半空时来一次「仅关传输」：换掉 _asr_session、bump
    # transport_generation，epoch 与 lifecycle 对象都不动 —— 这正是
    # _close_transport_only 干的事，也是唯一能让 delivered 为假的那条腿。
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

    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert drifted is True
    # 走的确实是「传输身份漂移」这条腿，不是 detach / fail-closed 那条
    # （那两条会 bump epoch、换 lifecycle，并且自己会跑 _reset_asr_turn_state）。
    assert runtime._asr_session_epoch == epoch
    assert runtime._asr_lifecycle is lifecycle_ref

    # 挂起槽必须已经腾空 —— 没人会再来兑付它。
    assert component._asr_pending_speech_confirmed is False
    assert component._asr_pending_speech_onset_at is None
    # 而用户真实开口的时刻一点没丢：它在 await 之前就装进了 _asr_turn_onset_at。
    assert component._asr_turn_onset_at == recorded_onset

    # 行为层：走完这一轮，下一次**不相干**的开口不能继承那个陈旧时刻。
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

async def test_direct_overlap_replay_keeps_the_onset_for_a_pending_confirmation() -> None:
    """Dual: an onset held for a pending confirmation must NOT be reclaimed.

    When the session is momentarily unavailable the replay parks in PREWARMING
    with the confirmation pending and deliberately holds the onset for it.
    Reclaiming it there sends that confirmation back to a fresh detected_at and
    every frame since the user started speaking is excluded.
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
    await runtime._handle_independent_asr_endpoint(epoch)

    async def park_with_pending_confirmation(*_args, **_kwargs):
        runtime._asr_runtime._asr_pending_speech_confirmed = True

    runtime._asr_runtime._handle_independent_asr_activity = (
        park_with_pending_confirmation
    )

    await runtime._handle_independent_asr_final("first", epoch, "openai")
    await runtime._wait_asr_transcript_dispatch_idle()

    assert runtime._asr_pending_speech_onset_at is not None

async def test_overlap_prerecord_trims_against_the_pending_turn_onset() -> None:
    """During an overlap the successor's boundary lives in the pending slot.

    Speech that starts while the previous turn is still DRAINING records its
    onset in ``_asr_pending_turn_onset_at``; it is only copied into
    ``_asr_turn_onset_at`` once the previous provider final activates that turn.
    Reading only the latter means the whole overlap window is judged against the
    PRECEDING turn's onset, so frames from after its endpoint still count as
    "this turn's" and fill the bounded buffer, evicting the successor's real
    opening and middle views.
    """
    runtime = _Runtime()
    runtime._asr_route_mode = "independent"
    now = time.monotonic()

    # 前一轮的 onset 很早；后继在它还没收场时开口，边界记在 pending 槽。
    runtime._asr_runtime._asr_turn_onset_at = now - 40.0
    runtime._asr_runtime._asr_pending_turn_onset_at = now - 2.0

    # 前一轮封口之后、后继开口之前的帧。
    for i in range(3):
        assert runtime._stage_independent_visual_frame(
            f"between-{i}",
            source="screen",
            request_id=f"screen-between-{i}",
            captured_at=now - 30.0 + i,
        )
    # 后继自己的帧。
    for i in range(2):
        assert runtime._stage_independent_visual_frame(
            f"successor-{i}",
            source="screen",
            request_id=f"screen-successor-{i}",
            captured_at=now - 1.5 + i * 0.3,
        )

    # 只有后继自己的帧留下，中间那些没占名额。
    assert [f.image_b64 for f in runtime._prerecord_visual_frames] == [
        "successor-0",
        "successor-1",
    ]
