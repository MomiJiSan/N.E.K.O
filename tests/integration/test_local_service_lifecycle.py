"""Opt-in real CPU managed-service ownership, prepare, renewal and idle test.

Set NEKO_ASR_REAL_LIFECYCLE=1 plus the four managed-service configuration
paths. Uses existing public weights/speech and never downloads a model.
The short idle/lease intervals are test-only launcher overrides; production
defaults remain 300/60 seconds. This does not claim an uncached disk read.
"""
import asyncio
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import psutil
import pytest

from main_logic.asr_client.local_service import (
    LocalServiceManager,
    ManagedConfig,
    _WindowsJob,
    configuration,
)
from tests.support.asr_fakes import _Runtime

pytestmark = [pytest.mark.asyncio, pytest.mark.integration_serial]


def _snapshot(pid):
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    result = []
    for process in processes:
        try:
            command = process.cmdline()
            native = any("spawn_main" in item or item == "--multiprocessing-fork" for item in command)
            result.append({"pid": process.pid, "created": process.create_time(),
                           "rss": process.memory_info().rss, "native_model_process": native})
        except psutil.NoSuchProcess:
            pass
    return result


def _still_owned_alive(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.is_running() and process.create_time() == identity["created"]
    except psutil.NoSuchProcess:
        return False


async def _until(function, *, timeout=15):
    async with asyncio.timeout(timeout):
        while True:
            result = await function()
            if result:
                return result
            await asyncio.sleep(0.025)


async def test_real_managed_prepare_renew_idle_reload_and_owned_exit(tmp_path):
    if os.getenv("NEKO_ASR_REAL_LIFECYCLE") != "1":
        pytest.skip("Opt in with NEKO_ASR_REAL_LIFECYCLE=1 and managed-service paths")
    config = configuration()
    assert isinstance(config, ManagedConfig)
    config = replace(config, idle_seconds=0.4)
    spawn_count = 0
    startup_failures = []

    async def spawn(*args, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        # Enough time for actual CPU cold loading; short enough to observe
        # the real background renewer during this integration test.
        try:
            return await asyncio.create_subprocess_exec(*args, "--lease-seconds", "10", **kwargs)
        except Exception as error:
            startup_failures.append(("spawn", type(error).__name__, str(error)))
            raise

    def job(process):
        try:
            return _WindowsJob(process)
        except Exception as error:
            startup_failures.append(("job", type(error).__name__, str(error)))
            raise

    manager = LocalServiceManager(config, spawn=spawn, job_factory=job)
    stop_sampling = asyncio.Event()
    sampler = None
    observed = {}
    peak_rss = 0
    report = {"passed": False}
    voice = None
    core = None
    supervisor = None
    try:
        started = time.monotonic()
        try:
            endpoint = await manager._endpoint()
        except Exception as error:
            raise AssertionError(f"Managed startup failed; underlying stages: {startup_failures}") from error
        supervisor = manager._owned.process
        listen_seconds = time.monotonic() - started
        health = await manager._request(endpoint, "GET", "/v1/health", statuses=(200, 503))
        initial = await asyncio.to_thread(_snapshot, supervisor.pid)
        assert health["state"] == "stopped" and not health["ready"]
        assert not any(item["native_model_process"] for item in initial), "Lazy listening must not start a native model process"
        observed.update(((item["pid"], item["created"]), item) for item in initial)

        async def sample():
            nonlocal peak_rss
            while not stop_sampling.is_set():
                snapshot = await asyncio.to_thread(_snapshot, supervisor.pid)
                observed.update(((item["pid"], item["created"]), item) for item in snapshot)
                peak_rss = max(peak_rss, sum(item["rss"] for item in snapshot))
                await asyncio.sleep(0.02)

        sampler = asyncio.create_task(sample())
        first, second = str(uuid4()), str(uuid4())
        started = time.monotonic()
        a, b = await asyncio.gather(manager.prepare(first), manager.prepare(second))
        assert a["operation_id"] == b["operation_id"]
        assert a["lease_id"] != b["lease_id"]
        await manager.cancel(a["lease_id"])
        cold = await manager.acquire_ready(second)
        cold_seconds = time.monotonic() - started
        assert cold["ready"] is True and cold["state"] == "ready"
        assert cold["timings"]["smoke_seconds"] > 0
        worker = cold["worker_instance"]
        after_cold = await asyncio.to_thread(_snapshot, supervisor.pid)
        assert any(item["native_model_process"] for item in after_cold), "A ready service must own a real native child"
        observed.update(((item["pid"], item["created"]), item) for item in after_cold)

        started = time.monotonic()
        warm = await manager.acquire_ready(str(uuid4()))
        warm_seconds = time.monotonic() - started
        assert warm["worker_instance"] == worker
        assert warm["operation_id"] == cold["operation_id"]
        assert spawn_count == 1

        voice = manager.retain_ready()
        assert voice is not None
        # Bind the real managed lease to the existing Core/runtime resource
        # slot. Normal microphone stop does not send end_session.
        core = _Runtime()
        core._asr_runtime._asr_resource_lease = voice
        initial_expiry = voice._lease.expires_at
        for identity in (a["lease_id"], b["lease_id"], warm["lease_id"]):
            await manager.cancel(identity)

        async def renewed():
            assert not voice._lease.failed
            return voice._lease.expires_at > initial_expiry + 0.1

        await _until(renewed, timeout=8)
        # No WS is open: a live voice lease, including its renewals, protects
        # the model even after multiple artificial idle deadlines.
        await asyncio.sleep(0.9)
        held = await manager.status(cold["operation_id"])
        assert held["ready"] is True and held["worker_instance"] == worker
        assert supervisor.returncode is None
        renewer = voice._lease.renew_task
        accepted = await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
        assert accepted is True
        assert core._asr_runtime.capture_resource_lease() is None
        assert voice._lease.refs == 0 and voice._identity not in manager._leases
        assert renewer.done(), "Normal capture_end must stop the model lease renewer"
        await core._asr_runtime.close()
        voice = None

        async def idle():
            status = await manager._request(endpoint, "GET", "/v1/health", statuses=(200, 503))
            return status if not status["ready"] and status["state"] == "stopped" else None

        await _until(idle)

        async def native_retired():
            return not any(item["native_model_process"] for item in await asyncio.to_thread(_snapshot, supervisor.pid))

        await _until(native_retired)
        assert supervisor.returncode is None, "Idle unloads the model, not the supervisor"
        for _ in range(3):
            assert not (await manager._request(endpoint, "GET", "/v1/health", statuses=(200, 503)))["ready"]
        assert await native_retired(), "Health queries must not reload the model"

        started = time.monotonic()
        reloaded = await manager.acquire_ready(str(uuid4()))
        reload_seconds = time.monotonic() - started
        assert reloaded["ready"] is True
        assert reloaded["worker_instance"] != worker
        assert spawn_count == 1, "Idle recovery reuses the owned supervisor"
        observed.update(((item["pid"], item["created"]), item) for item in await asyncio.to_thread(_snapshot, supervisor.pid))
        source_hash = await asyncio.to_thread(lambda: hashlib.sha256(Path(config.smoke_wav).read_bytes()).hexdigest())
        report = {
            "passed": True, "platform": os.name, "threads": config.threads,
            "lazy_listen_seconds": listen_seconds, "cold_prepare_seconds": cold_seconds,
            "lazy_owned_process_count": len(initial), "lazy_native_model_process_count": 0,
            "warm_prepare_seconds": warm_seconds, "idle_reload_seconds": reload_seconds,
            "cold_stages": cold["timings"], "reload_stages": reloaded["timings"],
            "sampled_peak_service_tree_rss_bytes": peak_rss, "rss_sample_interval_seconds": 0.02,
            "test_idle_seconds": 0.4, "test_lease_seconds": 10,
            "production_idle_seconds": 300, "production_lease_seconds": 60,
            "single_flight": True, "one_waiter_cancelled_without_cancelling_peer": True,
            "nonempty_speech_smoke_required": True, "warm_reused_worker": True,
            "voice_lease_renewed_without_websocket": True,
            "normal_capture_end_released_voice_lease": True,
            "normal_capture_end_stopped_renewal_without_end_session": True,
            "idle_retired_model_kept_supervisor": True, "health_no_load_side_effect": True,
            "reload_created_new_worker": True, "supervisor_spawns": spawn_count,
            "smoke_source_sha256": source_hash,
            "boundary": "Real Windows managed process/job, CPU model and authenticated HTTP; cached local files, no OS cache eviction, microphone, LLM or TTS",
        }
    finally:
        if core is not None:
            await core._asr_runtime.close()
        if voice is not None:
            await voice.close()
        await manager.close()
        stop_sampling.set()
        if sampler is not None:
            await sampler
        if supervisor is not None:
            assert supervisor.returncode is not None
        async def all_retired():
            alive = await asyncio.to_thread(lambda: [item for item in observed.values() if _still_owned_alive(item)])
            return not alive
        await _until(all_retired, timeout=8)
    assert report["passed"]
    report["all_observed_owned_processes_retired"] = True
    report["observed_owned_process_count"] = len(observed)
    output = Path(os.getenv("NEKO_ASR_LIFECYCLE_REPORT", str(tmp_path / "m1c-lifecycle-real.json")))
    await asyncio.to_thread(output.write_text, json.dumps(report, indent=2), encoding="utf-8")
