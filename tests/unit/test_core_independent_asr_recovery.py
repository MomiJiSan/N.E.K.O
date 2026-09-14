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


pytestmark = [pytest.mark.asyncio, pytest.mark.runtime]


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

@pytest.mark.parametrize(
    "replacement",
    ["epoch", "lifecycle", "session", "transport", "state"],
)
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

async def test_close_releases_independent_audio_pipeline() -> None:
    runtime = _Runtime()
    pipeline = type("Pipeline", (), {})()
    pipeline.close = AsyncMock()
    runtime._voice_input_audio_pipeline = pipeline

    await runtime._close_independent_asr(next_route_mode="blocked")

    pipeline.close.assert_awaited_once_with()
    assert runtime._voice_input_audio_pipeline is not pipeline

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

@pytest.mark.parametrize("initial_nr", [True, False])
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

@pytest.mark.parametrize(
    "changed_identity",
    [
        "lease_generation",
        "hard_mute",
        "focus_suppression",
        "game_takeover",
        "route_operation",
        "core_session",
    ],
)
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

@pytest.mark.parametrize("notification", ["status", "lifecycle", "failure"])
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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
