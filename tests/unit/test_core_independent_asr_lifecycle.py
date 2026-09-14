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

@pytest.mark.parametrize(
    "provider",
    ["dummy", "glm", "gemini"],
)
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

@pytest.mark.parametrize("provider", ["qwen", "grok", "soniox"])
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

@pytest.mark.parametrize("route_mode", ["native", "independent"])
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

@pytest.mark.parametrize("stale_guard", ["ingress", "owner"])
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

@pytest.mark.parametrize("operation", ["abort", "close"])
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

@pytest.mark.parametrize("provider", ["qwen", "openai"])
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

@pytest.mark.parametrize("provider", ["glm", "gemini"])
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

@pytest.mark.parametrize("core_type", ["qwen", "qwen_intl"])
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

@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("observer_raises", [False, True])
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

@pytest.mark.parametrize("core_type", ["openai", "glm", "gemini"])
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

async def test_hot_swap_does_not_retry_failed_same_core_route() -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.input_mode = "audio"
    runtime._asr_route_mode = "blocked"
    runtime._independent_asr_route_key = "gemini"
    runtime._start_independent_asr_if_enabled = AsyncMock()

    await runtime._reconcile_independent_asr_after_core_change()

    runtime._start_independent_asr_if_enabled.assert_not_awaited()

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

@pytest.mark.parametrize("enabled", [False, True])
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

@pytest.mark.unit
async def test_native_route_leaves_provider_capability_routing_inside_session() -> None:
    """Core selects the ASR strategy, while session capability keeps legacy behavior."""
    runtime = _Runtime()
    runtime.session._supports_native_image = False
    runtime.session.set_visual_delivery_mode = MagicMock()

    runtime._set_microphone_route("native")

    delivered_mode = runtime.session.set_visual_delivery_mode.call_args.args[0]
    assert getattr(delivered_mode, "value", delivered_mode) == "native"

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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
