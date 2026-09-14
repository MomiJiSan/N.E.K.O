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

@pytest.mark.parametrize(
    ("previous_owner", "owner", "reason", "barrier_method"),
    [
        ("core", "game", "game_takeover", "suspend"),
        ("game", "core", "game_release", "abort"),
        ("core", "none", "connection_closed", "abort"),
    ],
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

@pytest.mark.parametrize(
    ("event", "generation"),
    [
        ("invalid-control", 0),
        ("lease_sync", -1),
    ],
)
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

async def test_start_forwards_core_user_language_to_session_builder(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    runtime.user_language = "ja"

    kwargs = await _start_bridge_and_capture_builder_call(monkeypatch, runtime)

    assert kwargs["user_language"] == "ja"

async def test_start_without_user_language_builds_session_without_hint(
    monkeypatch,
) -> None:
    runtime = _Runtime()
    runtime.core_api_type = "gemini"
    assert getattr(runtime, "user_language", None) is None

    kwargs = await _start_bridge_and_capture_builder_call(monkeypatch, runtime)

    assert kwargs["user_language"] is None

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

async def test_restart_rejects_non_positive_attempt_override() -> None:
    runtime = _Runtime()

    with pytest.raises(ValueError, match="max_attempts must be positive"):
        await runtime._restart_transport(max_attempts=0)

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

@pytest.mark.parametrize("malformed", ["false", 0, 1, [False], {"enabled": False}])
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

@pytest.mark.parametrize("malformed", ["true", 1, 0, [True], {"enabled": True}])
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

@pytest.mark.parametrize(
    ("persisted_enabled", "handshake_enabled"),
    [
        (True, None),
        (False, None),
        (False, True),
        (True, False),
    ],
)
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

@pytest.mark.parametrize(
    "transition",
    [
        "hard_mute",
        "focus_suppress",
        "game_takeover",
        "lease_sync",
        "connection_replacement",
    ],
)
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

async def test_blocked_text_episode_keeps_session_identity_reference() -> None:
    runtime = _Runtime()
    runtime.input_mode = "text"
    runtime._set_microphone_route("blocked")
    session = runtime.session

    episode = runtime._blocked_text_mode_microphone_episode()

    assert episode is not None
    assert episode[-1] is session

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

@pytest.mark.parametrize("factory_fails", [False, True])
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

@pytest.mark.parametrize("replacement", ["epoch", "lifecycle", "detector"])
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

@pytest.mark.unit
@pytest.mark.parametrize("delivery", ["direct_atomic", "handoff_required"])
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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
