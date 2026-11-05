"""Real managed M2 reload through Smart Turn, subtitles and Core admission.

Only external LLM/browser/plugin/directive/telemetry sinks are recorders.
A transparent WebSocket observer delegates every actual read and write.
Same pinned weights under distinct trusted manifests verify lifecycle, not WER.
"""
import asyncio
import json
import os
import queue
import subprocess
import time
import traceback
import wave
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import psutil
import pytest
from starlette.websockets import WebSocketState

from main_logic.asr_client import local_service, reload_control
from main_logic.asr_client.workers import local_streaming
from main_logic.core.context_append import ContextAppendMixin
from main_logic.core.proactive import ProactiveMixin
from main_logic.core.turn import TurnMixin
from main_logic.omni_realtime_client import OmniRealtimeClient
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from tests.support.asr_fakes import _Runtime

pytestmark = [pytest.mark.asyncio, pytest.mark.integration_serial]


async def _until(function, timeout=25):
    async with asyncio.timeout(timeout):
        while True:
            value = function()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return value
            await asyncio.sleep(0.01)


def _catalog(config, root):
    script = """
import hashlib,json,sys
from pathlib import Path
from neko_asr.bundles import build_pinned_manifest,canonical_json
model,root=map(Path,sys.argv[1:])
root.mkdir()
raw=canonical_json(build_pinned_manifest(model,'host-reload-v2'))
manifest=root/'v2.manifest.json'
manifest.write_bytes(raw)
(root/'catalog.json').write_bytes(canonical_json({'schema_version':1,'candidates':{
'v2':{'path':str(model),'manifest':str(manifest),'manifest_sha256':hashlib.sha256(raw).hexdigest()}}}))
"""
    env = dict(os.environ, PYTHONPATH=str(Path(config.root) / "src"))
    subprocess.run([config.python, "-c", script, config.model_dir, str(root)], env=env, check=True,
                   capture_output=True, timeout=20)
    return root / "catalog.json"


class _ObservedSocket:
    def __init__(self, socket, events, writes, connection):
        self.socket, self.events, self.writes, self.connection = socket, events, writes, connection

    async def recv(self):
        value = await self.socket.recv()
        self.events.append((time.monotonic(), self.connection, json.loads(value)))
        return value

    async def send(self, value):
        if isinstance(value, bytes):
            self.writes.append((time.monotonic(), self.connection, str(UUID(bytes=value[4:20]))))
        await self.socket.send(value)

    async def close(self, *args, **kwargs):
        return await self.socket.close(*args, **kwargs)


@pytest.mark.parametrize("hot_reload", [False, True], ids=["baseline", "reload"])
async def test_real_managed_reload_smart_turn_subtitles_and_core_handoff(tmp_path, monkeypatch, hot_reload):
    if os.getenv("NEKO_ASR_REAL_RELOAD") != "1":
        pytest.skip("Set NEKO_ASR_REAL_RELOAD=1 plus four managed-service paths")
    config = local_service.configuration()
    assert isinstance(config, local_service.ManagedConfig)
    catalog = await asyncio.to_thread(_catalog, config, tmp_path / "catalog")
    monkeypatch.setenv("NEKO_ASR_CATALOG", str(catalog))
    monkeypatch.setenv("NEKO_ASR_STATE_DIR", str(tmp_path / "catalog" / "state"))
    monkeypatch.setenv("NEKO_ASR_IDLE_SECONDS", "0.4")
    with wave.open(config.smoke_wav, "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        pcm = source.readframes(source.getnframes())
    events, writes = [], []
    connects = 0
    connect = local_streaming.websockets.connect

    async def observed_connect(*args, **kwargs):
        nonlocal connects
        connects += 1
        return _ObservedSocket(await connect(*args, **kwargs), events, writes, connects)

    monkeypatch.setattr(local_streaming.websockets, "connect", observed_connect)
    await local_service.startup()
    manager = local_service.get_manager()
    preparation = str(uuid4())
    observed = {}
    report = None
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
    failures = []
    handle_error = runtime._handle_independent_asr_error

    async def observed_error(*args, **kwargs):
        detector = runtime._asr_detector
        failures.append({"code": kwargs.get("status_code"),
                         "stack": [f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}" for frame in traceback.extract_stack()[-8:]],
                         "detector_closed": getattr(detector, "_closed", None),
                         "detector_readiness": str(getattr(detector, "_smart_turn_readiness", None)),
                         "detector_token": str(getattr(detector, "_smart_turn_token", None)),
                         "lifecycle_snapshot": str(getattr(runtime._asr_lifecycle, "snapshot", None)),
                         "adapter_failure": str(getattr(getattr(detector, "_semantic_adapter", None), "failure", None))})
        return await handle_error(*args, **kwargs)

    monkeypatch.setattr(runtime, "_handle_independent_asr_error", observed_error)
    try:
        await manager.acquire_ready(preparation)
        supervisor = manager._owned.process

        def snapshot():
            try:
                root = psutil.Process(supervisor.pid)
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                return
            for process in processes:
                try:
                    observed[(process.pid, process.create_time())] = process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass

        snapshot()
        result = await runtime.start(route_key="qwen", resource_optimization_enabled=False,
                                     provider_preference="local_streaming", user_language="zh-CN")
        assert result.provider == "local_streaming", result
        assert runtime._asr_session.is_ready and runtime._asr_resource_lease is not None
        await manager.cancel_request(preparation)
        control = reload_control.get_control()
        candidates = await control.candidates()
        assert any(value["candidate_id"] == "v2" for value in candidates["candidates"])
        audio = b"\0" * 16000 + pcm + b"\0" * 96000
        operation, published_at = None, None
        old_identity = None
        frames = 0

        async def feed_sentence():
            nonlocal frames
            token = core._capture_ingress_token()
            started = time.monotonic()
            for offset in range(0, len(audio), 1024):
                frame = audio[offset:offset + 1024]
                await runtime.submit(ProcessedVoiceFrame(frame, 16000, None), ingress_token=token)
                frames += 1
                deadline = started + (offset + len(frame)) / 32000
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))

        for sentence in range(3):
            feeding = asyncio.create_task(feed_sentence())
            try:
                if sentence == 0 and hot_reload:
                    old_open = await _until(lambda: next((value for _, _, value in events if value["type"] == "opened"), None))
                    old_identity = old_open["stream_id"]
                    started = time.monotonic()
                    operation = await control.reload(str(uuid4()), "v2")

                    async def published():
                        value = await control.status(operation["operation_id"])
                        assert value["state"] not in {"failed", "cancelled", "requires_pause"}, value
                        return value if value["active_changed"] else None

                    publication = await _until(published)
                    published_at = time.monotonic()
                    publish_seconds = published_at - started
                    assert publication["worker_instance"] != old_open["worker_instance"]
                    assert not any(value["type"] == "final" and value.get("stream_id") == old_identity for _, _, value in events)
                    snapshot()
                await feeding
            finally:
                feeding.cancel()
                await asyncio.gather(feeding, return_exceptions=True)
            assert not failures, failures
            await _until(lambda: core.session.submit_external_voice_turn.await_count == sentence + 1)
            await runtime.wait_transcript_idle()
            assert core.session.submit_external_voice_turn.await_count == sentence + 1
        assert runtime._asr_session.is_ready
        assert connects == 1, "All three sentences must share the actual same WebSocket"
        all_opened = [value for _, _, value in events if value["type"] == "opened"]
        finals = [value for _, _, value in events if value["type"] == "final"]
        assert len(finals) == 3
        # The existing runtime may eagerly open the next silent/prewarm turn.
        # Count completed sentences, then cancel any successor through normal
        # capture_end; an open is not itself a chat-delivery claim.
        completed = {value["stream_id"] for value in finals}
        opened = [value for value in all_opened if value["stream_id"] in completed]
        assert len(opened) == 3
        assert len({value["stream_id"] for value in opened}) == 3
        if hot_reload:
            assert opened[0]["worker_instance"] != opened[1]["worker_instance"] == opened[2]["worker_instance"]
            assert opened[0]["route_generation"] == 1 and all(value["route_generation"] == 2 for value in opened[1:])
            assert any(stamp > published_at and identity == old_identity for stamp, _, identity in writes)
        else:
            assert len({value["worker_instance"] for value in opened}) == 1
            assert all(value["route_generation"] == 1 for value in opened)
            publish_seconds = None
        for opening in opened:
            stream_events = [value for _, _, value in events if value.get("stream_id") == opening["stream_id"]]
            assert sum(value["type"] == "final" for value in stream_events) == 1
            for value in stream_events:
                assert tuple(value[key] for key in ("worker_instance", "route_generation", "bundle_id")) == tuple(
                    opening[key] for key in ("worker_instance", "route_generation", "bundle_id"))
        final_calls = core.handle_input_transcript.await_args_list
        final_texts = [call.args[0] for call in final_calls]
        assert len(final_calls) == core._session_turn_count == 3
        assert final_texts == [value["text"] for value in finals]
        handoffs = core.session.submit_external_voice_turn.await_args_list
        assert [call.args for call in handoffs] == [(text,) for text in final_texts]
        turn_ids = [call.kwargs["turn_id"] for call in handoffs]
        assert len(set(turn_ids)) == 3
        notifications = [call.args[0] for call in core.websocket.send_json.await_args_list
                         if call.args and call.args[0].get("type") == "user_transcript"]
        previews = [call.args[0] for call in core.websocket.send_json.await_args_list
                    if call.args and call.args[0].get("type") == "user_transcript_preview" and call.args[0].get("text")]
        assert [event["text"] for event in notifications] == final_texts
        assert {event["asr_turn_id"] for event in previews} == set(turn_ids)
        assert [message["data"]["data"] for message in core.sync_message_queue.queue] == final_texts
        assert core.message_cache_for_new_session == [{"role": core.master_name, "text": "".join(final_texts)}]
        if hot_reload:
            await control.action(operation["operation_id"], "finalize")

            async def retired():
                value = await control.status(operation["operation_id"])
                return value if value["state"] == "completed" and not value["retirement_pending"] else None

            await _until(retired)
        assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
        assert runtime._asr_resource_lease is None
        report = {"passed": True, "managed_service": True, "same_websocket_sentences": 3,
                  "real_smart_turn": True, "publish_seconds": publish_seconds,
                  "hot_reload": hot_reload, "old_sentence_audio_continues_after_publish": hot_reload,
                  "wire_finals": 3, "user_transcript_events": len(notifications),
                  "core_handoffs": len(handoffs), "preview_events": len(previews),
                  "final_texts": final_texts, "turn_ids": turn_ids,
                  "worker_generations": [{key: value[key] for key in ("worker_instance", "route_generation", "bundle_id")} for value in opened],
                  "normal_capture_end_released_resource": True, "input_frames": frames,
                  "warm_successor_opens": len(all_opened) - len(opened),
                  "boundary": "Real managed CPU ASR/Silero/SmartTurn/host worker/subtitle events/Core admission and dispatch; same weights different trusted manifests. LLM/browser/plugin/directive/telemetry are recorders; no microphone, TTS, persistent history or accuracy comparison."}
    finally:
        diagnostic_root = Path(os.getenv("NEKO_ASR_RELOAD_HOST_REPORT", str(tmp_path / "reload-host-core.json"))).parent
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        diagnostic = {"hot_reload": hot_reload, "failures": failures,
                      "opened": [value for _, _, value in events if value["type"] == "opened"],
                      "terminals": [value for _, _, value in events if value["type"] in {"final", "error", "cancelled"}],
                      "core_handoffs": core.session.submit_external_voice_turn.await_count}
        (diagnostic_root / ("host-diagnostic-" + ("reload" if hot_reload else "baseline") + ".json")).write_text(
            json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        await runtime.close()
        if core._bg_tasks:
            await asyncio.gather(*core._bg_tasks)
        await local_service.close()
    assert supervisor.returncode is not None
    def all_exited():
        for pid, created in observed:
            try:
                process = psutil.Process(pid)
                if process.create_time() != created:
                    continue
                if os.name == "nt":
                    import win32api
                    import win32con
                    import win32event
                    handle = win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)
                    try:
                        if win32event.WaitForSingleObject(handle, 0) != win32event.WAIT_OBJECT_0:
                            return False
                    finally:
                        handle.Close()
                elif process.is_running():
                    return False
            except psutil.NoSuchProcess:
                continue
        return True

    await _until(all_exited, timeout=3)
    report["all_observed_owned_processes_exited"] = True
    report["observed_owned_processes"] = [{"pid": pid, "created": created, "rss": rss} for (pid, created), rss in observed.items()]
    output = Path(os.getenv("NEKO_ASR_RELOAD_HOST_REPORT", str(tmp_path / "reload-host-core.json")))
    if not hot_reload:
        output = output.with_stem(output.stem + "-baseline")
    output.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(output.write_text, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
