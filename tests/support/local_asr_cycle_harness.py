"""Real Core admission with bounded per-cycle external sink recorders."""
import asyncio
import queue
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

from starlette.websockets import WebSocketState

from main_logic.core.context_append import ContextAppendMixin
from main_logic.core.proactive import ProactiveMixin
from main_logic.core.turn import TurnMixin
from main_logic.omni_realtime_client import OmniRealtimeClient
from tests.support.asr_fakes import _Runtime


def make_core(monkeypatch):
    core = _Runtime()
    core.core_api_type = "qwen"
    core.websocket = SimpleNamespace(client_state=WebSocketState.CONNECTED, send_json=AsyncMock())
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
    core.master_name = "Cycle test user"
    core.is_preparing_new_session = True
    core.message_cache_for_new_session = []
    core._config_manager = None
    core._bg_tasks = set()
    core.handle_input_transcript = AsyncMock(wraps=MethodType(TurnMixin.handle_input_transcript, core))
    core._should_suppress_dirty_voice_transcript = MethodType(TurnMixin._should_suppress_dirty_voice_transcript, core)
    core._note_user_turn = MethodType(TurnMixin._note_user_turn, core)
    core.note_user_engagement = MethodType(ProactiveMixin.note_user_engagement, core)
    core._fire_task = MethodType(ContextAppendMixin._fire_task, core)
    core._broadcast_voice_transcript_observed = AsyncMock()
    core._publish_user_utterance_to_plugin_bus = Mock()
    core._inject_pending_user_directives = AsyncMock()
    core._dispatch_mini_game_invite_keyword = AsyncMock()
    tracker = SimpleNamespace(note_first_user_message=Mock(), note_user_message=Mock())
    monkeypatch.setattr("utils.token_tracker.TokenTracker.get_instance", lambda: tracker)
    core._voice_lease_connection_id = "m3-cycle"
    core._set_microphone_route("independent")
    core.current_speech_id = "m3-cycle"
    return core


async def close_core(core):
    await core._asr_runtime.close()
    if core._bg_tasks:
        await asyncio.gather(*core._bg_tasks)


def assert_core_delivery(core, finals):
    texts = [event["text"] for event in finals]
    calls = core.session.submit_external_voice_turn.await_args_list
    assert [call.args for call in calls] == [(text,) for text in texts]
    assert core.handle_input_transcript.await_count == core._session_turn_count == len(finals)
    assert len({call.kwargs["turn_id"] for call in calls}) == len(finals)
    notifications = [call.args[0] for call in core.websocket.send_json.await_args_list
                     if call.args and call.args[0].get("type") == "user_transcript"]
    assert [event["text"] for event in notifications] == texts
    assert [message["data"]["data"] for message in core.sync_message_queue.queue] == texts
    if texts:
        previews = [call.args[0] for call in core.websocket.send_json.await_args_list
                    if call.args and call.args[0].get("type") == "user_transcript_preview"
                    and call.args[0].get("text")]
        assert {event["asr_turn_id"] for event in previews} == {call.kwargs["turn_id"] for call in calls}
