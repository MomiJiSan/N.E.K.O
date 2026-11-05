"""Opt-in real CPU voice start/stop, cancellation, idle and supervisor cycles."""
import asyncio
import gc
import json
import os
import statistics
import time
import traceback
import wave
from collections import Counter
from pathlib import Path
from uuid import uuid4

import psutil
import pytest

from main_logic.asr_client import local_service
from main_logic.asr_client.workers import local_streaming
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from tests.integration.test_local_streaming_asr_reload import _ObservedSocket, _until
from tests.integration.test_local_streaming_asr_soak import State, core_fixture
from tests.support.local_asr_cycle_harness import close_core

pytestmark = [pytest.mark.asyncio, pytest.mark.integration_serial]

MIB = 1024 * 1024


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))] if ordered else None


def memory(process):
    with process.oneshot():
        info = process.memory_info()
        cpu = process.cpu_times()
        return {"rss": info.rss, "private_bytes": getattr(info, "private", getattr(info, "data", 0)),
                "handles": process.num_handles() if os.name == "nt" else process.num_fds(),
                "threads": process.num_threads(), "cpu_seconds": cpu.user + cpu.system}


class ProcessSampler:
    """Bound live exact handles; retain only small terminal identity records."""
    def __init__(self):
        self.manager = None
        self.live, self.retired = {}, []
        self.samples = []
        self.peak_native = 0
        self.closed = False
        self.failure = None
        self.task = asyncio.create_task(self.run())

    def snapshot(self):
        import win32api
        import win32event
        import win32process
        for key, item in list(self.live.items()):
            if win32event.WaitForSingleObject(item["handle"], 0) == win32event.WAIT_OBJECT_0:
                item["handle"].Close()
                self.retired.append({key: value for key, value in item.items() if key != "handle"})
                del self.live[key]
        manager = self.manager
        owned = manager._owned if manager is not None else None
        if owned is not None and owned.process.returncode is None:
            try:
                root = psutil.Process(owned.process.pid)
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            for process in processes:
                try:
                    created = process.create_time()
                    key = (process.pid, created)
                    if key not in self.live and not any(value["pid"] == process.pid and value["created"] == created for value in self.retired):
                        handle = win32api.OpenProcess(0x1000 | 0x100000, False, process.pid)
                        exact_created = win32process.GetProcessTimes(handle)["CreationTime"].timestamp()
                        assert abs(exact_created - created) <= .001
                        command = " ".join(process.cmdline())
                        self.live[key] = {"pid": process.pid, "created": created, "handle": handle,
                                          "native": "spawn_main" in command, "name": process.name(),
                                          "supervisor_root": owned.process.pid}
                except psutil.NoSuchProcess:
                    continue
        native = 0
        stats = []
        for item in self.live.values():
            try:
                process = psutil.Process(item["pid"])
                assert process.create_time() == item["created"]
                stats.append({"pid": item["pid"], "native": item["native"], **memory(process)})
                native += int(item["native"])
            except psutil.NoSuchProcess:
                continue
        self.peak_native = max(self.peak_native, native)
        assert native <= 2, "More than two native model workers"
        self.samples.append({"monotonic": time.monotonic(), "host": memory(psutil.Process()),
                             "native_workers": native, "processes": stats})

    async def run(self):
        try:
            while not self.closed:
                self.snapshot()
                await asyncio.sleep(1)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                self.failure = repr(error)
            raise

    def check(self):
        assert self.failure is None, self.failure

    async def retired_group(self, root_pid):
        async def check():
            self.snapshot()
            return not any(item["supervisor_root"] == root_pid for item in self.live.values())
        await _until(check, 8)

    async def close(self):
        self.closed = True
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        def all_signaled():
            self.snapshot()
            return not self.live
        await _until(all_signaled, 8)
        self.check()


async def test_real_repeated_voice_cycles(tmp_path, monkeypatch):
    if os.getenv("NEKO_ASR_REAL_CYCLES") != "1":
        pytest.skip("Set NEKO_ASR_REAL_CYCLES=1 and managed model/Smart Turn paths")
    assert os.name == "nt", "Exact-handle cycles currently validate Windows"
    minimum = int(os.getenv("NEKO_ASR_CYCLE_COUNT", "40"))
    duration = float(os.getenv("NEKO_ASR_CYCLE_SECONDS", "1800"))
    assert minimum >= 1 and duration >= 0
    monkeypatch.setenv("NEKO_ASR_IDLE_SECONDS", "0.4")
    config = local_service.configuration()
    assert isinstance(config, local_service.ManagedConfig)
    with wave.open(config.smoke_wav, "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        pcm = source.readframes(source.getnframes())
    audio = b"\0" * 16000 + pcm + b"\0" * 96000
    output = Path(os.getenv("NEKO_ASR_CYCLES_REPORT", str(tmp_path / "cycles.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    thresholds = {"min_cycles": minimum, "min_wall_seconds": duration,
                  "host_settled_rss_growth_bytes": 64 * MIB, "host_settled_rss_growth_fraction": .20,
                  "max_native_workers": 2, "duplicate_final": 0, "orphan": 0,
                  "idle_test_seconds": .4, "manager_restart_every": 10,
                  "settled_baseline_after_cycle": 3}
    report = {"passed": False, "thresholds": thresholds, "cycles": [], "expected_cancelled_sentences": 0,
              "normal_finals": 0, "expected_warm_successor_cancellations": 0,
              "supervisor_restarts": 0, "unexpected_errors": [],
              "boundary": "Real managed ASR/Silero/Smart Turn/Core admission/subtitles; external LLM/browser/plugin sinks are bounded per-cycle recorders; no microphone/TTS/accuracy claim."}
    sampler = ProcessSampler()
    connect = local_streaming.websockets.connect
    events, writes, sockets = [], [], []
    state = None
    class CycleSocket(_ObservedSocket):
        def __init__(self, socket, cycle_state):
            super().__init__(socket, events, writes, len(sockets) + 1)
            self.cycle_state = cycle_state

        async def recv(self):
            raw = await super().recv()
            self.cycle_state.receive(json.loads(raw))
            return raw

        async def send(self, payload):
            self.cycle_state.send(payload)
            await super().send(payload)

    async def observed_connect(*args, **kwargs):
        sock = CycleSocket(await connect(*args, **kwargs), state)
        sockets.append(sock)
        return sock
    monkeypatch.setattr(local_streaming.websockets, "connect", observed_connect)
    started = time.monotonic()
    last_checkpoint = started
    cold, voice, idle, settled = [], [], [], []
    workers = set()
    core = None
    cycle_patch = None
    await local_service.startup()
    manager = local_service.get_manager()
    sampler.manager = manager

    def save():
        elapsed = time.monotonic() - started
        first_window = [item["rss"] for item in settled if item["wall_seconds"] <= 300]
        last_window = [item["rss"] for item in settled if item["wall_seconds"] >= elapsed - 300]
        report["settled_five_minute_windows"] = {
            "first_samples": len(first_window), "last_samples": len(last_window),
            "first_median_rss": statistics.median(first_window) if first_window else None,
            "last_median_rss": statistics.median(last_window) if last_window else None,
            "growth_bytes": statistics.median(last_window) - statistics.median(first_window) if first_window and last_window else None,
            "complete_nonoverlapping_windows": elapsed >= 600}
        report.update(wall_seconds=time.monotonic() - started, completed_cycles=len(report["cycles"]),
                      peak_native_workers=sampler.peak_native, observed_processes=sampler.retired,
                      resource_samples=sampler.samples, settled_samples=settled,
                      latencies={key: {"p50": percentile(values, .5), "p95": percentile(values, .95)}
                                 for key, values in (("prepare_seconds", cold), ("voice_start_seconds", voice), ("idle_release_seconds", idle))})
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    async def feed(runtime, payload, stop=None):
        token = core._capture_ingress_token()
        began = time.monotonic()
        for offset in range(0, len(payload), 1024):
            frame = payload[offset:offset + 1024]
            await runtime.submit(ProcessedVoiceFrame(frame, 16000, None), ingress_token=token)
            if stop is not None and stop():
                return
            await asyncio.sleep(max(0, began + (offset + len(frame)) / 32000 - time.monotonic()))

    try:
        cycle = 0
        while cycle < minimum or time.monotonic() - started < duration:
            cycle += 1
            events.clear()
            writes.clear()
            sockets.clear()
            prepare_id = str(uuid4())
            began = time.monotonic()
            preparation = await manager.acquire_ready(prepare_id)
            cold.append(time.monotonic() - began)
            endpoint = manager.endpoint()
            instance = preparation["worker_instance"]
            assert instance not in workers, "Idle did not produce a new native worker"
            workers.add(instance)
            sampler.snapshot()
            optimization = cycle % 4 != 0 and not (minimum == 3 and cycle == 3)
            cycle_patch = pytest.MonkeyPatch()
            state = State()
            core = core_fixture(state, cycle_patch)
            runtime = core._asr_runtime
            began = time.monotonic()
            result = await runtime.start(route_key="qwen", resource_optimization_enabled=optimization,
                                         provider_preference="local_streaming", user_language="zh-CN")
            voice.append(time.monotonic() - began)
            assert result.provider == "local_streaming" and runtime._asr_session.is_ready
            coordinator = runtime._asr_detector._semantic_adapter._coordinator
            evaluate = coordinator.evaluate
            async def observed_evaluate(*args, **kwargs):
                value = await evaluate(*args, **kwargs)
                state.endpoint_decisions.append({"decision": str(value.decision), "status": str(value.status),
                                                 "reason": str(value.reason), "coordinator_state": str(coordinator.state)})
                return value
            cycle_patch.setattr(coordinator, "evaluate", observed_evaluate)
            leases = list(manager._leases.values())
            await manager.cancel_request(prepare_id)
            cancelled_id = None
            if cycle % 4 == 2:
                # Real partial utterance, normal runtime abort -> native cancel.
                await feed(runtime, b"\0" * 16000 + pcm,
                           stop=lambda: any(value["type"] == "consumed" for _, _, value in events))
                cancelled_id = next(value["stream_id"] for _, _, value in events if value["type"] == "opened")
                assert not any(value["type"] == "final" for _, _, value in events)
                await runtime.abort("m3_expected_cancel")
                await _until(lambda: any(value["type"] == "cancelled" and value.get("stream_id") == cancelled_id for _, _, value in events))
                assert state.counts["core_handoffs"] == 0
                report["expected_cancelled_sentences"] += 1
            target_finals = 2 if cycle % 5 == 0 else 1
            clips = 0
            # WAV boundaries are input accounting, never semantic endpoints.
            # Keep ingress alive while Smart Turn legitimately continues a turn.
            async with asyncio.timeout(90):
                while True:
                    await feed(runtime, audio)
                    clips += 1
                    assert not state.errors, list(state.errors)
                    await runtime.wait_transcript_idle()
                    actual = state.counts["final"]
                    paired = actual == state.counts["formal_notifications"] == state.counts["core_handoffs"] == state.counts["sync_messages"]
                    voiced_pending = any(stream["first_partial"] for stream in state.streams.values())
                    if actual >= target_finals and paired and not voiced_pending and not state.final_queue and not state.handoff_queue:
                        break
            finals = [value for _, _, value in events if value["type"] == "final"]
            assert len(finals) >= target_finals
            terminal_counts = Counter(value["stream_id"] for value in finals)
            assert all(count == 1 for count in terminal_counts.values())
            if cancelled_id is not None:
                assert cancelled_id not in terminal_counts
            assert state.counts["commit_commands"] == len(finals) == state.counts["core_handoffs"]
            assert state.counts["formal_notifications"] == state.counts["sync_messages"] == len(finals)
            assert state.counts["subtitle_previews"] >= len(finals)
            report["normal_finals"] += len(finals)
            assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
            await close_core(core)
            assert runtime._asr_resource_lease is None
            assert not manager._leases and not manager._active
            assert not any(manager._request_waiters.values())
            assert all(task.done() for task in manager._requests.values())
            assert all(item.renew_task is None or item.renew_task.done() for item in leases)
            for dispatcher in (runtime._asr_audio_dispatcher, runtime._asr_detector_dispatcher,
                               runtime._asr_transcript_dispatcher):
                assert dispatcher._queue.empty() and dispatcher._queue._unfinished_tasks == 0
            assert not runtime._asr_owned_cleanup_tasks and not runtime._asr_transcript_dispatcher.has_pending_delivery
            assert all(socket.socket.state.name == "CLOSED" for socket in sockets)
            assert not state.streams and not state.final_queue and not state.handoff_queue
            assert state.counts["final"] == len(finals)
            warm_cancelled = {value["stream_id"] for _, _, value in events
                              if value["type"] == "cancelled" and value.get("terminal_state") == "cancelled"
                              and value["stream_id"] != cancelled_id}
            report["expected_warm_successor_cancellations"] += len(warm_cancelled)
            began = time.monotonic()
            async def unloaded():
                value = await manager._request(endpoint, "GET", "/v1/health", statuses=(200, 503))
                return value.get("state") == "stopped" and not value.get("ready")
            await _until(unloaded)
            idle.append(time.monotonic() - began)
            def native_retired():
                sampler.snapshot()
                return not any(item["native"] for item in sampler.live.values())
            await _until(native_retired, 8)
            report["cycles"].append({"cycle": cycle, "optimization": optimization, "worker_instance": instance,
                                      "normal_finals": len(finals), "actual_final_target": target_finals, "input_clips": clips,
                                      "wire_counts": dict(state.counts), "recent_endpoint_decisions": list(state.endpoint_decisions),
                                      "warm_successor_cancellations": len(warm_cancelled),
                                      "cancelled_sentence": cancelled_id is not None,
                                      "leases_zero": True, "ws_closed": True, "queues_zero": True,
                                      "idle_unloaded": True, "prepare_seconds": cold[-1], "voice_start_seconds": voice[-1]})
            # Undo per-cycle observers before GC: pytest's outer fixture would
            # otherwise retain each runtime bound method and its Core graph.
            cycle_patch.undo()
            cycle_patch = None
            core = None
            del runtime, coordinator, evaluate, observed_evaluate
            gc.collect()
            settled.append({"cycle": cycle, "wall_seconds": time.monotonic() - started, **memory(psutil.Process())})
            if len(settled) >= 3:
                baseline = settled[2]["rss"]
                limit = max(64 * MIB, baseline * .20)
                assert settled[-1]["rss"] - baseline <= limit, "Settled host RSS exceeded frozen limit"
            if cycle % 10 == 0:
                root_pid = manager._owned.process.pid
                await local_service.close()
                await sampler.retired_group(root_pid)
                assert not manager._leases and not manager._cleanups and not manager._spawn_tasks
                report["supervisor_restarts"] += 1
                await local_service.startup()
                manager = local_service.get_manager()
                sampler.manager = manager
            sampler.check()
            if time.monotonic() - last_checkpoint >= 120 or cycle <= 3:
                save()
                print(json.dumps({"checkpoint": "cycles", "cycle": cycle,
                                  "wall_seconds": report["wall_seconds"], "rss": settled[-1]["rss"],
                                  "expected_cancellations": report["expected_cancelled_sentences"]}), flush=True)
                last_checkpoint = time.monotonic()
        report["workload_completed"] = True
    except BaseException as error:
        detector = core._asr_runtime._asr_detector if core is not None else None
        report["unexpected_errors"].append({"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc(), "failed_cycle": cycle,
                                             "state": state.summary() if state is not None else None,
                                             "lifecycle": str(getattr(core._asr_runtime._asr_lifecycle, "snapshot", None)) if core is not None else None,
                                             "detector_readiness": str(getattr(detector, "_smart_turn_readiness", None)),
                                             "wire_events": events, "wire_writes": writes})
        raise
    finally:
        try:
            if core is not None:
                await close_core(core)
        finally:
            if cycle_patch is not None:
                cycle_patch.undo()
            await local_service.close()
            sampler.manager = None
            try:
                await sampler.close()
                report["passed"] = bool(report.get("workload_completed") and not report["unexpected_errors"])
            except BaseException as error:
                report["unexpected_errors"].append({"type": type(error).__name__, "stage": "exact_handle_cleanup", "message": str(error)})
                raise
            finally:
                save()
