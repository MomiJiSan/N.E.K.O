"""Opt-in wall-clock M3 soak: real CPU/Smart Turn/Core, bounded external sinks.

NEKO_ASR_M3_SOAK=1; NEKO_ASR_M3_SECONDS defaults to 1800 (60 for pilot).
No microphone/LLM/TTS/user recordings; A/B share pinned public weights.
"""
import asyncio
import json
import os
import queue
import statistics
import subprocess
import time
import traceback
import wave
from collections import Counter, deque
from pathlib import Path
from types import MethodType, SimpleNamespace
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
THRESHOLDS = {"settled_growth_absolute_bytes": 64 * 1024**2, "settled_growth_fraction": 0.20,
              "native_workers_max": 2, "host_audio_queue_hold_max_bytes": 64000,
              "errors_max": 0, "final_delivery_ratio": 1, "steady_exclude_seconds_after_publish": 12,
              "steady_window_seconds": 300, "checkpoint_seconds": 30,
              "scope": "Frozen diagnostic acceptance thresholds, not a product SLA"}


def noop(*args, **kwargs):
    return None


async def anoop(*args, **kwargs):
    return None


async def afalse(*args, **kwargs):
    return False


async def until(function, timeout=25):
    async with asyncio.timeout(timeout):
        while True:
            value = function()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return value
            await asyncio.sleep(0.02)


def catalog(config, root):
    script = """
import hashlib,sys
from pathlib import Path
from neko_asr.bundles import build_pinned_manifest,canonical_json
model,root=map(Path,sys.argv[1:]); root.mkdir()
candidates={}
for name in ('A','B'):
 raw=canonical_json(build_pinned_manifest(model,'m3-soak-'+name))
 manifest=root/(name+'.json'); manifest.write_bytes(raw)
 candidates[name]={'path':str(model),'manifest':str(manifest),'manifest_sha256':hashlib.sha256(raw).hexdigest()}
(root/'catalog.json').write_bytes(canonical_json({'schema_version':1,'candidates':candidates}))
"""
    subprocess.run([config.python, "-c", script, config.model_dir, str(root)], check=True,
                   capture_output=True, timeout=20,
                   env=dict(os.environ, PYTHONPATH=str(Path(config.root) / "src")))
    return root / "catalog.json"


class State:
    def __init__(self):
        self.counts = Counter()
        self.streams = {}
        self.retired = deque(maxlen=128)
        self.final_queue = deque(maxlen=4)
        self.dispatch = None
        self.handoff_queue = deque(maxlen=4)
        self.all_stream_ids = set()
        self.all_turn_ids = set()
        self.last_turn = 0
        self.first_final_text = None
        self.latencies = {key: deque(maxlen=4096) for key in (
            "first_audio_to_first_partial_seconds", "commit_to_wire_final_seconds",
            "wire_final_to_core_handoff_seconds", "reload_request_to_publish_seconds")}
        self.errors = deque(maxlen=16)
        self.published_at = 0.0
        self.expected_generation = 1
        self.expected_version = None
        self.reload_old_stream = None
        self.pending_cross = False
        self.crossed = 0
        self.last_binding = None
        self.endpoint_decisions = deque(maxlen=16)

    def error(self, code, **detail):
        self.counts["errors"] += 1
        self.errors.append({"code": code, **detail})

    def receive(self, event):
        now = time.monotonic()
        kind, identity = event["type"], event.get("stream_id")
        self.counts[kind] += 1
        if kind == "hello":
            assert event["per_utterance_binding"] is True
            return
        if kind == "error":
            self.error(event.get("code", "WIRE_ERROR"))
            return
        if kind == "finished":
            return
        binding = tuple(event.get(key) for key in ("worker_instance", "route_generation", "bundle_id"))
        if kind == "opened":
            assert identity not in self.all_stream_ids and len(self.all_stream_ids) < 4096
            self.all_stream_ids.add(identity)
            assert len(self.streams) <= 1
            assert binding[1] >= self.expected_generation
            if self.expected_version and binding[1] == self.expected_generation:
                assert binding[2].endswith(":" + self.expected_version)
            if self.last_binding:
                assert binding[1] >= self.last_binding[1]
                assert binding[1] != self.last_binding[1] or binding == self.last_binding
            self.last_binding = binding
            self.streams[identity] = {"binding": binding, "first_audio": None, "first_partial": False,
                                      "commit": None, "last_audio": None, "frames": {}, "accept_seq": 0,
                                      "consume_seq": 0, "samples": 0}
            return
        stream = self.streams.get(identity)
        if stream is None:
            assert kind == "cancelled" and identity in self.retired, (kind, identity)
            return
        assert stream["binding"] == binding
        if kind == "accepted":
            assert event["seq"] == stream["accept_seq"]
            stream["accept_seq"] += 1
        if kind == "consumed":
            assert event["seq"] == stream["consume_seq"]
            stream["consume_seq"] += 1
            end, size = stream["frames"].pop(event["seq"])
            assert event["sample_end"] == end
            self.counts["consumed_pcm_bytes"] += size
        if kind == "partial" and not stream["first_partial"]:
            stream["first_partial"] = True
            if stream["first_audio"] is not None:
                self.latencies["first_audio_to_first_partial_seconds"].append(now - stream["first_audio"])
        if kind == "final":
            assert stream["commit"] is not None and not stream["frames"]
            assert event["sample_end"] == stream["samples"]
            assert len(self.final_queue) < self.final_queue.maxlen
            if self.first_final_text is None:
                self.first_final_text = event["text"]
            assert event["text"].strip()
            if event["text"] != self.first_final_text:
                self.counts["final_text_context_variations"] += 1
            self.final_queue.append({"text": event["text"], "at": now, "stream_id": identity})
            self.latencies["commit_to_wire_final_seconds"].append(now - stream["commit"])
            self.retired.append(identity)
            del self.streams[identity]
        elif kind == "cancelled":
            self.counts["cancelled_inflight_pcm_bytes"] += sum(size for _, size in stream["frames"].values())
            self.retired.append(identity)
            del self.streams[identity]

    def send(self, payload):
        now = time.monotonic()
        if isinstance(payload, bytes):
            identity = str(UUID(bytes=payload[4:20]))
            stream = self.streams[identity]
            sequence = int.from_bytes(payload[20:24], "big")
            start = int.from_bytes(payload[24:32], "big")
            size = len(payload) - 32
            assert start == stream["samples"] and sequence not in stream["frames"]
            assert len(stream["frames"]) < 256
            stream["samples"] += size // 2
            stream["frames"][sequence] = (stream["samples"], size)
            self.counts["sent_pcm_bytes"] += size
            if stream["first_audio"] is None:
                stream["first_audio"] = now
            stream["last_audio"] = now
            if self.pending_cross and identity == self.reload_old_stream and now > self.published_at:
                self.pending_cross = False
                self.crossed += 1
        else:
            value = json.loads(payload)
            if value.get("type") == "commit":
                stream = self.streams[value["stream_id"]]
                assert value["sample_end"] == stream["samples"] and stream["commit"] is None
                stream["commit"] = now
                self.counts["commit_commands"] += 1

    def summary(self):
        percentiles = {}
        for key, values in self.latencies.items():
            ordered = sorted(values)
            percentiles[key] = {"count": len(ordered), **{name: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]
                                if ordered else None for name, q in (("p50", .5), ("p95", .95), ("p99", .99))}}
        return {"counts": dict(self.counts), "open_streams": len(self.streams),
                "pending_finals": len(self.final_queue), "reloads_with_old_audio_after_publish": self.crossed,
                "latency_seconds": percentiles, "errors": list(self.errors),
                "recent_endpoint_decisions": list(self.endpoint_decisions)}


class Socket:
    def __init__(self, socket, state):
        self.socket, self.state = socket, state

    async def recv(self):
        raw = await self.socket.recv()
        self.state.receive(json.loads(raw))
        return raw

    async def send(self, payload):
        self.state.send(payload)
        await self.socket.send(payload)

    async def close(self, *args, **kwargs):
        return await self.socket.close(*args, **kwargs)


class Session(OmniRealtimeClient):
    def __init__(self, state):
        self.state = state
        self._response_arbiter = None
        self._is_gemini = False

    prepare_external_voice_turn = staticmethod(afalse)
    abandon_external_voice_turn = staticmethod(noop)
    create_response = staticmethod(anoop)
    handle_interruption = staticmethod(anoop)

    async def submit_external_voice_turn(self, text, *, turn_id, **kwargs):
        state = self.state
        assert state.handoff_queue and state.handoff_queue[0]["text"] == text
        expected = state.handoff_queue.popleft()
        assert turn_id not in state.all_turn_ids and len(state.all_turn_ids) < 4096
        state.all_turn_ids.add(turn_id)
        number = int(turn_id.rsplit("-", 1)[1])
        assert number > state.last_turn
        state.last_turn = number
        state.counts["core_handoffs"] += 1
        assert state.counts["formal_notifications"] == state.counts["core_handoffs"]
        state.latencies["wire_final_to_core_handoff_seconds"].append(time.monotonic() - expected["at"])


def core_fixture(state, monkeypatch):
    core = _Runtime()
    for name in ("handle_new_message", "send_status", "_close_interrupted_offline_turn"):
        setattr(core, name, anoop if name != "_close_interrupted_offline_turn" else lambda *args, **kwargs: True)
    core.core_api_type = "qwen"

    async def notify(event):
        if event.get("type") == "user_transcript":
            assert state.dispatch and state.dispatch["text"] == event["text"]
            state.counts["formal_notifications"] += 1
        elif event.get("type") == "user_transcript_preview" and event.get("text"):
            state.counts["subtitle_previews"] += 1

    core.websocket = SimpleNamespace(client_state=WebSocketState.CONNECTED, send_json=notify)
    core.session = Session(state)
    core._takeover_input_dispatcher = None
    core._activity_tracker = SimpleNamespace(on_voice_rms=noop, on_user_message=noop)
    core.last_user_engagement_time = None
    core._session_turn_count = 0
    core.sync_message_queue = queue.Queue(maxsize=8)
    core.master_name = "Public fixture"
    core.is_preparing_new_session = False
    core.message_cache_for_new_session = []
    core._config_manager = None
    core._bg_tasks = set()
    real_transcript = MethodType(TurnMixin.handle_input_transcript, core)

    async def transcript(text, **kwargs):
        assert state.final_queue and state.dispatch is None
        state.dispatch = state.final_queue.popleft()
        assert len(state.handoff_queue) < state.handoff_queue.maxlen
        state.handoff_queue.append(state.dispatch)
        assert text == state.dispatch["text"]
        state.counts["core_transcript_calls"] += 1
        try:
            return await real_transcript(text, **kwargs)
        finally:
            state.dispatch = None
            while not core.sync_message_queue.empty():
                queued = core.sync_message_queue.get_nowait()
                assert queued["data"]["data"] == text
                state.counts["sync_messages"] += 1
                core.sync_message_queue.task_done()

    core.handle_input_transcript = transcript
    core._should_suppress_dirty_voice_transcript = MethodType(TurnMixin._should_suppress_dirty_voice_transcript, core)
    core._note_user_turn = MethodType(TurnMixin._note_user_turn, core)
    core.note_user_engagement = MethodType(ProactiveMixin.note_user_engagement, core)
    core._fire_task = MethodType(ContextAppendMixin._fire_task, core)
    for name in ("_broadcast_voice_transcript_observed", "_inject_pending_user_directives", "_dispatch_mini_game_invite_keyword"):
        setattr(core, name, anoop)
    core._publish_user_utterance_to_plugin_bus = noop
    tracker = SimpleNamespace(note_first_user_message=noop, note_user_message=noop)
    monkeypatch.setattr("utils.token_tracker.TokenTracker.get_instance", lambda: tracker)
    core._voice_lease_connection_id = "m3-soak"
    core._set_microphone_route("independent")
    core.current_speech_id = "m3-soak"
    runtime = core._asr_runtime
    original_error = runtime._handle_independent_asr_error

    async def observed_error(*args, **kwargs):
        state.error(kwargs.get("status_code", "RUNTIME_ERROR"), stack=[f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                    for frame in traceback.extract_stack()[-8:]])
        return await original_error(*args, **kwargs)

    runtime._handle_independent_asr_error = observed_error
    return core


class Monitor:
    def __init__(self, manager, runtime, state, root):
        self.manager, self.runtime, self.state, self.root = manager, runtime, state, root
        self.start = time.monotonic()
        self.stop = asyncio.Event()
        self.identities = {}
        self.steady = deque(maxlen=7200)
        self.latest = {}
        self.peaks = Counter()
        self.last_checkpoint = 0
        self.prev_cpu = None
        self.resource_task = None

    def sample(self):
        root = psutil.Process(self.manager._owned.process.pid)
        processes = [psutil.Process(os.getpid()), root, *root.children(recursive=True)]
        result = {"host_rss": 0, "host_private": 0, "tree_rss": 0, "tree_private": 0,
                  "host_handles": 0, "tree_handles": 0, "host_threads": 0, "tree_threads": 0,
                  "native_workers": 0, "tree_processes": 0, "cpu_seconds": 0}
        seen = set()
        for process in processes:
            try:
                key = (process.pid, process.create_time())
                if key in seen:
                    continue
                seen.add(key)
                memory, times = process.memory_info(), process.cpu_times()
                host = process.pid == os.getpid()
                prefix = "host_" if host else "tree_"
                result[prefix + "rss"] += memory.rss
                result[prefix + "private"] += getattr(memory, "private", getattr(memory, "vms", 0))
                result[prefix + "threads"] += process.num_threads()
                result[prefix + "handles"] += process.num_handles() if os.name == "nt" else process.num_fds()
                if not host:
                    self.identities[key] = {"pid": key[0], "created": key[1], "cpu": times.user + times.system}
                    result["tree_processes"] += 1
                    if any("spawn_main" in part or part == "--multiprocessing-fork" for part in process.cmdline()):
                        result["native_workers"] += 1
                else:
                    host_cpu = times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        result["cpu_seconds"] = host_cpu + sum(item["cpu"] for item in self.identities.values())
        return result

    def queues(self):
        session = self.runtime._asr_session
        requests = getattr(session, "_request_queue", None)
        if requests is None:
            requests = getattr(session, "_requests", None)
        assert requests is not None, "Audio hold measurement must use the actual session queue"
        held = requests.waiting_audio_bytes
        self.peaks["host_audio_queue_hold_bytes"] = max(self.peaks["host_audio_queue_hold_bytes"], held)
        assert held <= THRESHOLDS["host_audio_queue_hold_max_bytes"]
        return {"audio_queue_hold_bytes": held,
                "detector_queue_audio_ms": getattr(self.runtime._asr_detector, "queued_audio_ms", 0),
                "pending_core_finals": len(self.state.final_queue)}

    async def run(self):
        with (self.root / "checkpoints.jsonl").open("a", encoding="utf-8", buffering=1) as output:
            while not self.stop.is_set():
                sample = await asyncio.to_thread(self.sample)
                now = time.monotonic()
                sample.update(self.queues())
                assert sample["native_workers"] <= THRESHOLDS["native_workers_max"], sample
                for key, value in sample.items():
                    self.peaks[key] = max(self.peaks[key], value)
                if self.prev_cpu:
                    sample["cpu_one_core_percent"] = 100 * (sample["cpu_seconds"] - self.prev_cpu[1]) / (now - self.prev_cpu[0])
                self.prev_cpu = now, sample["cpu_seconds"]
                sample["elapsed_seconds"] = now - self.start
                self.latest = sample
                if (sample["native_workers"] == 1 and now - self.state.published_at >= THRESHOLDS["steady_exclude_seconds_after_publish"]):
                    self.steady.append(dict(sample))
                if now - self.last_checkpoint >= THRESHOLDS["checkpoint_seconds"]:
                    checkpoint = {"phase": "running", "resources": sample, **self.state.summary()}
                    output.write(json.dumps(checkpoint, ensure_ascii=False) + "\n")
                    output.flush()
                    atomic_json(self.root / "progress.json", checkpoint)
                    self.last_checkpoint = now
                try:
                    await asyncio.wait_for(self.stop.wait(), 1)
                except TimeoutError:
                    pass

    async def close(self):
        self.stop.set()
        if self.resource_task:
            await self.resource_task

    def growth(self):
        if not self.steady:
            return {}
        first_end = self.steady[0]["elapsed_seconds"] + 300
        last_start = self.steady[-1]["elapsed_seconds"] - 300
        result = {}
        for key in ("host_rss", "host_private", "tree_rss", "tree_private"):
            first = [sample[key] for sample in self.steady if sample["elapsed_seconds"] <= first_end]
            last = [sample[key] for sample in self.steady if sample["elapsed_seconds"] >= last_start]
            baseline, final = statistics.median(first), statistics.median(last)
            threshold = max(THRESHOLDS["settled_growth_absolute_bytes"], baseline * THRESHOLDS["settled_growth_fraction"])
            xs = [sample["elapsed_seconds"] for sample in self.steady]
            ys = [sample[key] for sample in self.steady]
            xm, ym = statistics.mean(xs), statistics.mean(ys)
            denominator = sum((x - xm)**2 for x in xs)
            trend = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denominator if denominator else 0
            result[key] = {"first_window_median_bytes": baseline, "last_window_median_bytes": final,
                           "growth_bytes": final - baseline, "threshold_bytes": threshold,
                           "within_frozen_threshold": final - baseline <= threshold,
                           "linear_trend_bytes_per_hour": trend * 3600,
                           "first_window_samples": len(first), "last_window_samples": len(last)}
        return result


def atomic_json(filename, value):
    temporary = filename.with_suffix(".new.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, filename)


def exited(identities):
    for item in identities.values():
        try:
            process = psutil.Process(item["pid"])
            if process.create_time() != item["created"]:
                continue
            if os.name == "nt":
                import win32api
                import win32con
                import win32event
                handle = win32api.OpenProcess(win32con.SYNCHRONIZE, False, process.pid)
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


async def test_real_cpu_host_periodic_reload_soak(tmp_path, monkeypatch):
    if os.getenv("NEKO_ASR_M3_SOAK") != "1":
        pytest.skip("Opt in with NEKO_ASR_M3_SOAK=1 and managed-service paths")
    duration = float(os.getenv("NEKO_ASR_M3_SECONDS", "1800"))
    assert 60 <= duration <= 7200
    root = Path(os.getenv("NEKO_ASR_M3_REPORT_DIR", str(tmp_path / "m3")))
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "thresholds.json", {"frozen_before_run": True, "duration_seconds": duration, **THRESHOLDS})
    config = local_service.configuration()
    assert isinstance(config, local_service.ManagedConfig)
    filename = await asyncio.to_thread(catalog, config, tmp_path / "catalog")
    monkeypatch.setenv("NEKO_ASR_CATALOG", str(filename))
    monkeypatch.setenv("NEKO_ASR_STATE_DIR", str(tmp_path / "catalog" / "state"))
    with wave.open(config.smoke_wav, "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        pcm = source.readframes(source.getnframes())
    audio = b"\0" * 16000 + pcm + b"\0" * 96000
    state = State()
    connect = local_streaming.websockets.connect

    async def observed_connect(*args, **kwargs):
        state.counts["connections"] += 1
        return Socket(await connect(*args, **kwargs), state)

    monkeypatch.setattr(local_streaming.websockets, "connect", observed_connect)
    core = core_fixture(state, monkeypatch)
    runtime = core._asr_runtime
    await local_service.startup()
    manager = local_service.get_manager()
    prepare_id = str(uuid4())
    monitor = None
    reloader = None
    stopped = asyncio.Event()
    run_started = None
    active_elapsed = None
    failure = None
    try:
        await manager.acquire_ready(prepare_id)
        result = await runtime.start(route_key="qwen", resource_optimization_enabled=False,
                                     provider_preference="local_streaming", user_language="zh-CN")
        assert result.provider == "local_streaming" and runtime._asr_session.is_ready
        await manager.cancel_request(prepare_id)
        control = reload_control.get_control()
        coordinator = runtime._asr_detector._semantic_adapter._coordinator
        evaluate = coordinator.evaluate

        async def observe_evaluate(*args, **kwargs):
            value = await evaluate(*args, **kwargs)
            decision = str(value.decision)
            state.counts["endpoint_" + decision] += 1
            state.endpoint_decisions.append({"decision": decision, "status": str(value.status),
                                             "reason": str(value.reason), "coordinator_state": str(coordinator.state)})
            return value

        monkeypatch.setattr(coordinator, "evaluate", observe_evaluate)
        monitor = Monitor(manager, runtime, state, root)
        monitor.resource_task = asyncio.create_task(monitor.run())
        run_started = time.monotonic()

        async def reloads():
            next_reload = run_started
            while not stopped.is_set():
                if time.monotonic() < next_reload:
                    try:
                        await asyncio.wait_for(stopped.wait(), next_reload - time.monotonic())
                        return
                    except TimeoutError:
                        pass
                old_id, old_stream = await until(lambda: next(((identity, stream) for identity, stream in state.streams.items()
                    if stream["last_audio"] and time.monotonic() - stream["last_audio"] < .2), None))
                candidate = "B" if state.counts["reloads"] % 2 == 0 else "A"
                started = time.monotonic()
                operation = await control.reload(str(uuid4()), candidate)

                async def published():
                    value = await control.status(operation["operation_id"])
                    assert value["state"] not in {"failed", "cancelled", "requires_pause"}, value
                    return value if value["active_changed"] else None

                publication = await until(published)
                state.published_at = time.monotonic()
                assert publication["route_generation"] > old_stream["binding"][1]
                state.expected_generation = publication["route_generation"]
                state.expected_version = "m3-soak-" + candidate
                state.reload_old_stream = old_id
                state.pending_cross = old_id in state.streams
                state.latencies["reload_request_to_publish_seconds"].append(state.published_at - started)
                state.counts["reloads"] += 1
                await control.action(operation["operation_id"], "finalize")

                async def completed():
                    value = await control.status(operation["operation_id"])
                    assert value["state"] != "failed", value
                    return value["state"] == "completed" and not value["retirement_pending"]

                await until(completed)
                next_reload = started + 90

        reloader = asyncio.create_task(reloads())
        clips = 0
        while time.monotonic() - run_started < duration:
            token = core._capture_ingress_token()
            sentence_started = time.monotonic()
            for offset in range(0, len(audio), 1024):
                assert not state.errors, list(state.errors)
                assert not monitor.resource_task.done(), monitor.resource_task.exception() if monitor.resource_task.done() else None
                if reloader.done():
                    reloader.result()
                frame = audio[offset:offset + 1024]
                await runtime.submit(ProcessedVoiceFrame(frame, 16000, None), ingress_token=token)
                state.counts["input_frames"] += 1
                monitor.queues()
                deadline = sentence_started + (offset + len(frame)) / 32000
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            clips += 1
            state.counts["input_clips"] = clips
            # Input clips are not semantic turns. Keep feeding the public
            # recording even when Smart Turn legally asks for continuation.
            # Every actual wire final must still traverse all delivery stages.
            await runtime.wait_transcript_idle()
            await until(lambda: state.counts["final"] == state.counts["formal_notifications"]
                        == state.counts["core_handoffs"] == state.counts["sync_messages"], timeout=5)
            core.message_cache_for_new_session.clear()
        # Finish with real-rate silence so the last voiced semantic candidate
        # can naturally settle. Do not manufacture commit commands per clip.
        async with asyncio.timeout(20):
            while any(stream["first_partial"] for stream in state.streams.values()) or state.final_queue or state.handoff_queue:
                assert not state.errors, list(state.errors)
                await runtime.submit(ProcessedVoiceFrame(b"\0" * 1024, 16000, None), ingress_token=token)
                state.counts["input_frames"] += 1
                state.counts["tail_silence_frames"] += 1
                monitor.queues()
                await asyncio.sleep(.032)
        await runtime.wait_transcript_idle()
        assert state.counts["commit_commands"] == state.counts["final"]
        assert state.counts["final"] == state.counts["core_handoffs"] == state.counts["formal_notifications"] == state.counts["sync_messages"]
        active_elapsed = time.monotonic() - run_started
        stopped.set()
        if reloader:
            await reloader
        assert active_elapsed >= duration
        assert not state.errors
        if duration >= 1800:
            assert state.counts["connections"] >= 2 and state.counts["final"] >= 65
            assert 15 <= state.counts["reloads"] <= 30
        assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
        assert runtime._asr_resource_lease is None
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        stopped.set()
        if reloader:
            reloader.cancel()
            await asyncio.gather(reloader, return_exceptions=True)
        if monitor:
            await monitor.close()
        await runtime.close()
        if core._bg_tasks:
            await asyncio.gather(*core._bg_tasks)
        await local_service.close()
        cleanup_passed = False
        if monitor:
            await until(lambda: exited(monitor.identities), timeout=5)
            cleanup_passed = True
        growth = monitor.growth() if monitor else {}
        threshold_passed = all(value["within_frozen_threshold"] for value in growth.values())
        report = {"passed": failure is None and threshold_passed and cleanup_passed,
                  "semantic_turn_definition": "Actual SmartTurn-owned commit/final, independent of public input clip count; an uncommitted silent warm successor is cancelled at normal close",
                  "failure": failure, "duration_requested_seconds": duration,
                  "actual_active_monotonic_seconds": active_elapsed,
                  "active_plus_teardown_monotonic_seconds": time.monotonic() - run_started if run_started else None,
                  "optimization": False, "thresholds": THRESHOLDS, "settled_growth": growth,
                  "resource_peaks": dict(monitor.peaks) if monitor else {},
                  "last_resource_sample": monitor.latest if monitor else {},
                  "owned_process_exit_verified": cleanup_passed, "owned_processes": list(monitor.identities.values()) if monitor else [],
                  "latency_clock": "time.monotonic; Windows Python3.11 clock may quantize small intervals around 16ms",
                  "latency_definitions": {"first_audio_to_first_partial_seconds": "First PCM write to first nonempty service partial, including VAD-prefix context",
                                          "commit_to_wire_final_seconds": "Smart Turn owned commit write to matching service final receive",
                                          "wire_final_to_core_handoff_seconds": "Matching wire final receive to production Core external-turn dispatch sink",
                                          "reload_request_to_publish_seconds": "Host reload-control request to observed active_changed true"},
                  "observer_limits": {"wire_active_streams": 2, "terminal_ids": 128, "pending_finals": 4,
                                      "all_stream_and_turn_ids_each": 4096, "handoff_pending": 4,
                                      "latency_samples_each": 4096, "resource_samples": 7200, "errors": 16,
                                      "mock_audio_history": 0, "pytest_logging_capture_disabled_by_runner": True},
                  "rss_scope": "RSS sums double-count shared pages; host pytest plus separately owned service tree; private bytes also recorded. Startup/teardown excluded from active duration and growth windows.",
                  "boundary": "Real managed CPU ASR/SmartTurn/Silero/host worker/Core admission/subtitle events; trusted A/B identical pinned weights. External LLM/browser/plugin/telemetry are bounded counters; public WAV realtime playback, no microphone/TTS/real user data or WER validation.",
                  **state.summary()}
        atomic_json(root / "report.json", report)
        atomic_json(root / "progress.json", {"phase": "completed" if report["passed"] else "failed", **report})
        assert threshold_passed, growth
