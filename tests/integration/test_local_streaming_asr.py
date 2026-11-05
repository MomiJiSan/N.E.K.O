"""Opt-in real ASR / Smart Turn / transcript admission / Core handoff.

Start the standalone service and set NEKO_ASR_URL/NEKO_ASR_TOKEN plus
NEKO_ASR_E2E_WAV to a public mono PCM16 16 kHz recording. Endpoint assets must
already be installed. Existing admission, echo guard, transcript notifications,
synchronization queue and candidate-session cache run their production code.
LLM/browser/plugin/telemetry/directive sinks are recorders: no cloud request,
permanent chat store, browser rendering, microphone or TTS is exercised.
The route harness also records reply-start hooks; it does not start a reply.
"""
import asyncio
import json
import os
import queue
import wave
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from starlette.websockets import WebSocketState

from main_logic.core.context_append import ContextAppendMixin
from main_logic.core.proactive import ProactiveMixin
from main_logic.core.turn import TurnMixin
from main_logic.omni_realtime_client import OmniRealtimeClient
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from tests.support.asr_fakes import _Runtime

pytestmark = [pytest.mark.asyncio, pytest.mark.integration_serial]


@pytest.fixture(scope="module")
def chain_reports():
    return []


@pytest.fixture
async def service_preparation():
    if os.getenv("NEKO_ASR_E2E_MANAGED") != "1" or not os.getenv("NEKO_ASR_E2E_WAV"):
        yield {"managed": False}
        return
    from main_logic.asr_client import local_service
    request_id = str(uuid4())
    started = asyncio.get_running_loop().time()
    async def release():
        await local_service.cancel_request(request_id)
    await local_service.startup()
    try:
        async with asyncio.timeout(20):
            prepared = await local_service.prepare(request_id)
            while not prepared.get("ready"):
                assert prepared.get("state") not in {"failed", "cancelled", "expired"}, prepared
                await asyncio.sleep(0.05)
                prepared = await local_service.status(prepared["operation_id"])
        yield {"managed": True, "prepare_seconds": asyncio.get_running_loop().time() - started,
               "release": release}
    finally:
        await release()
        await local_service.close()


@pytest.mark.parametrize("optimization", [False, True], ids=["optimization-off", "optimization-on"])
async def test_real_smart_turn_to_preview_and_core_chat(tmp_path, monkeypatch, optimization, chain_reports, service_preparation):
    filename = os.getenv("NEKO_ASR_E2E_WAV")
    if not filename:
        pytest.skip("Set NEKO_ASR_E2E_WAV and start the standalone ASR service")
    def read():
        with wave.open(filename, "rb") as source:
            assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
            return source.readframes(source.getnframes())
    pcm = await asyncio.to_thread(read)
    assert 32000 <= len(pcm) <= 32000 * 30
    core = _Runtime()
    core.core_api_type = "qwen"
    core.websocket = SimpleNamespace(
        client_state=WebSocketState.CONNECTED, send_json=AsyncMock(),
    )
    # A typed external sink takes the production realtime transcript branch,
    # without constructing a client or reading its keys/connecting to an LLM.
    core.session = Mock(spec=OmniRealtimeClient)
    core.session._response_arbiter = None
    core.session._is_gemini = False
    core.session.prepare_external_voice_turn = AsyncMock(return_value=False)
    core.session.submit_external_voice_turn = AsyncMock()
    core.session.abandon_external_voice_turn = Mock()
    core._takeover_input_dispatcher = None
    core._activity_tracker = SimpleNamespace(on_voice_rms=Mock(), on_user_message=Mock())
    core.last_user_engagement_time = None
    core._session_turn_count = 0
    core.sync_message_queue = queue.Queue()
    core.master_name = "Local test user"
    core.is_preparing_new_session = True
    core.message_cache_for_new_session = []
    core._config_manager = None  # No persistence into the developer's chat store.
    core._bg_tasks = set()
    core.handle_input_transcript = AsyncMock(wraps=MethodType(TurnMixin.handle_input_transcript, core))
    core._should_suppress_dirty_voice_transcript = MethodType(TurnMixin._should_suppress_dirty_voice_transcript, core)
    core._note_user_turn = MethodType(TurnMixin._note_user_turn, core)
    core.note_user_engagement = MethodType(ProactiveMixin.note_user_engagement, core)
    core._fire_task = MethodType(ContextAppendMixin._fire_task, core)
    # These observers leave the chat-admission path; use local recorders rather
    # than touching plugin state, directives or application telemetry.
    core._broadcast_voice_transcript_observed = AsyncMock()
    core._publish_user_utterance_to_plugin_bus = Mock()
    core._inject_pending_user_directives = AsyncMock()
    core._dispatch_mini_game_invite_keyword = AsyncMock()
    tracker = SimpleNamespace(note_first_user_message=Mock(), note_user_message=Mock())
    monkeypatch.setattr("utils.token_tracker.TokenTracker.get_instance", lambda: tracker)
    core._voice_lease_connection_id = "local-e2e"
    core._set_microphone_route("independent")
    core.current_speech_id = "local-e2e"
    runtime = core._asr_runtime
    try:
        result = await runtime.start(
            route_key="qwen", resource_optimization_enabled=optimization,
            provider_preference="local_streaming", user_language="zh-CN",
        )
        assert result.provider == "local_streaming", result
        if not optimization:
            assert runtime._asr_session is not None and runtime._asr_session.is_ready, result
        if service_preparation["managed"]:
            # Mimic the frontend releasing its prepare waiter after start ack.
            # The existing voice runtime now owns the renewable resource lease.
            await service_preparation["release"]()
            assert runtime._asr_resource_lease is not None
        # Two sequential sentences exercise the same host route. No test calls
        # commit: real Silero + Smart Turn owns both sentence boundaries.
        audio = b"\0" * 16000 + pcm + b"\0" * 96000
        frames = 0
        for sentence in range(2):
            token = core._capture_ingress_token()
            started = asyncio.get_running_loop().time()
            for offset in range(0, len(audio), 1024):
                frame = audio[offset:offset + 1024]
                await runtime.submit(ProcessedVoiceFrame(frame, 16000, None), ingress_token=token)
                frames += 1
                deadline = started + (offset + len(frame)) / 32000
                await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))
            async with asyncio.timeout(15):
                while core.session.submit_external_voice_turn.await_count < sentence + 1:
                    await asyncio.sleep(0.02)
                await runtime.wait_transcript_idle()
            assert core.session.submit_external_voice_turn.await_count == sentence + 1
        assert runtime._asr_session is not None and runtime._asr_session.is_ready
        previews = [call.args[0] for call in core.websocket.send_json.await_args_list
                    if call.args and call.args[0].get("type") == "user_transcript_preview"
                    and call.args[0].get("text")]
        assert previews, "Real partial must reach the existing subtitle event"
        final_calls = core.handle_input_transcript.await_args_list
        assert len(final_calls) == core._session_turn_count == 2
        final_texts = [call.args[0] for call in final_calls]
        assert all(text.strip() for text in final_texts)
        assert all(call.kwargs["source"] == "independent_asr" for call in final_calls)
        assert all(call.kwargs["metadata"]["provider"] == "local_streaming" for call in final_calls)
        handoffs = core.session.submit_external_voice_turn.await_args_list
        assert [call.args for call in handoffs] == [(text,) for text in final_texts]
        turn_ids = [call.kwargs["turn_id"] for call in handoffs]
        assert len(set(turn_ids)) == 2 and all(identity.startswith("asr-") for identity in turn_ids)
        assert {preview["asr_turn_id"] for preview in previews} == set(turn_ids)
        notifications = [call.args[0] for call in core.websocket.send_json.await_args_list
                         if call.args and call.args[0].get("type") == "user_transcript"]
        assert [event["text"] for event in notifications] == final_texts
        queued = list(core.sync_message_queue.queue)
        assert [message["data"]["data"] for message in queued] == final_texts
        assert all(message["data"]["source"] == "independent_asr" for message in queued)
        assert core.message_cache_for_new_session == [{"role": core.master_name, "text": "".join(final_texts)}]
        assert core._activity_tracker.on_voice_rms.call_count == 2
        assert [call.kwargs["text"] for call in core._activity_tracker.on_user_message.call_args_list] == final_texts
        detector = runtime._asr_detector
        assert detector is not None
        if service_preparation["managed"]:
            assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
            assert runtime._asr_resource_lease is None
        chain_reports.append({
            "passed": True, "realtime": True, "input_frames": frames,
            "managed_service": service_preparation["managed"],
            "prepare_seconds": service_preparation.get("prepare_seconds"),
            "normal_capture_end_released_resource": service_preparation["managed"],
            "resource_optimization_enabled": optimization, "sentences": 2,
            "asr": "real Paraformer INT8 service", "endpoint": "real Silero + Smart Turn",
            "preview_events": len(previews), "user_transcript_events": len(notifications),
            "sync_messages": len(queued), "candidate_cache_updates": 2,
            "chat_handoffs": len(handoffs), "turn_ids": turn_ids, "final_texts": final_texts,
        })
        report = {"passed": True, "cases": chain_reports,
                  "boundary": "Real transcript admission/echo guard/user_transcript/sync queue/candidate cache and Core dispatch; external LLM, browser, plugin, directive and telemetry sinks are recorders; no persistent history/LLM reply/TTS/microphone"}
        output = Path(os.getenv("NEKO_ASR_E2E_REPORT", str(tmp_path / "local-e2e.json")))
        await asyncio.to_thread(output.write_text, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        await runtime.close()
        if core._bg_tasks:
            await asyncio.gather(*core._bg_tasks)
