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
