import asyncio
import json
import os
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest

from main_logic.asr_client import local_service as local


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    configured = {name: os.getenv(name) for name in local._MANAGED_FIELDS}
    for name in (*local._MANAGED_FIELDS, local._MANAGED_EXE_FIELD, "NEKO_ASR_SERVICE_THREADS", "NEKO_ASR_IDLE_SECONDS", "NEKO_ASR_URL", "NEKO_ASR_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    return configured


def endpoint():
    return local.Endpoint(local.DEFAULT_URL, "x" * 32)


class Process:
    pid = 123

    def __init__(self, event=None):
        self.returncode = None
        self.exited = asyncio.Event()
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(json.dumps(event or {"event": "listening", "addresses": [["127.0.0.1", 50120]],
                                                   "instance_id": str(uuid4())}).encode() + b"\n")
        self.stdout.feed_eof()
        self.killed = False

    async def wait(self):
        await self.exited.wait()
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -1
        self.exited.set()


class Job:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def close(self):
        self.closed = True
        if self.process.returncode is None:
            self.process.kill()


class Service:
    def __init__(self):
        self.calls = []
        self.records = {}
        self.operation = str(uuid4())
        self.ready = True
        self.ttl = 60
        self.post_entered = asyncio.Event()
        self.post_release = asyncio.Event()
        self.post_release.set()
        self.renewed = asyncio.Event()
        self.response_override = None
        self.options = []
        self.processes = []

    def __call__(self, **kwargs):
        self.options.append(kwargs)
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def request(self, method, url, headers, json=None):
        self.calls.append((method, url, headers, json))
        if self.response_override is not None:
            value = self.response_override
            if isinstance(value, Exception):
                raise value
            return value
        if url.endswith("/v1/prepare"):
            self.post_entered.set()
            await self.post_release.wait()
            record = self.records.setdefault(json["request_id"], {"operation_id": self.operation, "lease_id": str(uuid4())})
            result = dict(record, state="ready" if self.ready else "loading", ready=self.ready,
                          expires_in_seconds=self.ttl)
        elif "/v1/operations/" in url:
            result = {"operation_id": self.operation, "state": "ready" if self.ready else "loading", "ready": self.ready}
        elif url.endswith("/renew"):
            self.renewed.set()
            result = {"state": "ready", "expires_in_seconds": self.ttl}
        elif "/v1/leases/" in url:
            result = {"state": "released"}
        elif "/v1/prepare/" in url:
            result = {"state": "cancelled"}
        elif url.endswith("/shutdown"):
            for process in self.processes:
                if process.returncode is None:
                    process.returncode = 0
                    process.exited.set()
            result = {"state": "closing"}
        else:
            raise AssertionError(url)
        return httpx.Response(200, json=result)

    async def spawn(self, *args, **kwargs):
        process = Process()
        self.processes.append(process)
        self.spawn_args = args, kwargs
        return process


def managed(service, monkeypatch, **kwargs):
    config = local.ManagedConfig("C:\\python.exe", "C:\\asr", "C:\\models", "C:\\speech.wav")
    monkeypatch.setattr(local.LocalServiceManager, "_validate_files", staticmethod(lambda _: None))
    return local.LocalServiceManager(config, spawn=service.spawn, client_factory=service,
                                     job_factory=kwargs.pop("job_factory", Job), **kwargs)


async def settle(manager):
    async with asyncio.timeout(2):
        while manager._cleanups:
            await asyncio.gather(*tuple(manager._cleanups), return_exceptions=True)


@pytest.mark.parametrize("url", ["https://127.0.0.1:50120/v1/asr/stream", "ws://evil.test:50120/v1/asr/stream",
                                 "ws://127.0.0.1/v1/asr/stream", "ws://127.0.0.1:0/v1/asr/stream",
                                 "ws://127.0.0.1:50120/v1/asr/stream?q=x", "ws://u:p@127.0.0.1:50120/v1/asr/stream",
                                 "ws://127.0.0.1:50120/v1/asr/stream#x", "ws://127.0.0.1:50120/wrong"])
def test_reject_unsafe_external_configuration(monkeypatch, url):
    monkeypatch.setenv("NEKO_ASR_URL", url)
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    assert not local.configuration_is_available()


@pytest.mark.parametrize("token", ["", "x" * 31, "x" * 257, "a " * 32, "中" * 32])
def test_token_validation(monkeypatch, token):
    monkeypatch.setenv("NEKO_ASR_TOKEN", token)
    assert not local.configuration_is_available()


def test_probe_is_pure_and_redacts_endpoint(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_TOKEN", "secret" * 8)
    assert local.configuration_is_available()
    assert "secret" not in repr(local.configuration())
    assert local.configuration().http_url == "http://127.0.0.1:50120"


@pytest.mark.parametrize("field,value", [("NEKO_ASR_SERVICE_THREADS", "0"), ("NEKO_ASR_SERVICE_THREADS", "bad"),
                                         ("NEKO_ASR_IDLE_SECONDS", "nan"), ("NEKO_ASR_IDLE_SECONDS", "-1"),
                                         ("NEKO_ASR_IDLE_SECONDS", "0")])
def test_invalid_managed_numbers(monkeypatch, field, value):
    for name in local._MANAGED_FIELDS:
        monkeypatch.setenv(name, "C:\\trusted")
    monkeypatch.setenv(field, value)
    assert not local.configuration_is_available()


def test_partial_managed_config_never_falls_back(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_SERVICE_PYTHON", "C:\\python.exe")
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    assert not local.configuration_is_available()
    for name in local._MANAGED_FIELDS:
        monkeypatch.setenv(name, "C:\\trusted")
    assert isinstance(local.configuration(), local.ManagedConfig)
    monkeypatch.setenv("NEKO_ASR_MODEL_DIR", "relative")
    assert not local.configuration_is_available()


@pytest.mark.asyncio
async def test_external_prepare_ready_cancel_never_owns_service():
    service = Service()
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    result = await manager.prepare(str(uuid4()))
    assert (await manager.status(result["operation_id"]))["ready"]
    assert manager.endpoint() == endpoint()
    await manager.cancel(result["lease_id"])
    await manager.close()
    assert not service.processes
    assert not any(call[1].endswith("/shutdown") for call in service.calls)
    assert all(options["trust_env"] is False for options in service.options)


@pytest.mark.asyncio
async def test_managed_single_spawn_independent_waiters(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    one, two = await asyncio.gather(manager.prepare(str(uuid4())), manager.prepare(str(uuid4())))
    assert len(service.processes) == 1 and one["operation_id"] == two["operation_id"]
    assert one["lease_id"] != two["lease_id"]
    assert manager.endpoint().managed
    argv, options = service.spawn_args
    assert "--lazy" in argv and "0" in argv
    assert service.calls[0][2]["Authorization"] == "Bearer " + options["env"]["NEKO_ASR_TOKEN"]
    assert options["env"]["NEKO_ASR_TOKEN"] not in " ".join(argv)
    await manager.cancel(one["lease_id"])
    assert manager.endpoint().managed
    owned = manager._owned
    await manager.close()
    assert owned.job.closed and owned.process.returncode == 0


@pytest.mark.asyncio
async def test_managed_not_prepared_and_dead_process_fail_fast(monkeypatch):
    service = Service()
    service.ready = False
    manager = managed(service, monkeypatch)
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.endpoint()
    result = await manager.prepare(str(uuid4()))
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.retain_ready()
    service.ready = True
    await manager.status(result["operation_id"])
    assert manager.endpoint().managed
    service.processes[0].kill()
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.endpoint()
    second = await manager.prepare(str(uuid4()))
    assert len(service.processes) == 2 and second["ready"]
    await manager.close()


@pytest.mark.asyncio
async def test_voice_reference_retains_front_lease_and_renews(monkeypatch):
    service = Service()
    service.ttl = 0.06
    manager = managed(service, monkeypatch)
    result = await manager.prepare(str(uuid4()))
    first, second = manager.retain_ready(), manager.retain_ready()
    assert (await manager.cancel(result["lease_id"]))["state"] == "retained"
    async with asyncio.timeout(1):
        await service.renewed.wait()
    await first.close()
    assert result["lease_id"] in manager._leases
    await second.close()
    await second.close()
    assert result["lease_id"] not in manager._leases
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.endpoint()
    await manager.close()


@pytest.mark.asyncio
async def test_voice_close_cleans_without_front_release(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    result = await manager.prepare(str(uuid4()))
    voice = manager.retain_ready()
    await voice.close()
    assert result["lease_id"] not in manager._leases
    await manager.close()


@pytest.mark.asyncio
async def test_renew_failure_cannot_publish_stale_ready(monkeypatch):
    service = Service()
    service.ttl = 0.03
    manager = managed(service, monkeypatch)
    result = await manager.prepare(str(uuid4()))
    owner = manager.retain_ready()
    service.response_override = OSError("secret")
    async with asyncio.timeout(1):
        while not manager._leases[result["lease_id"]].failed:
            await asyncio.sleep(0)
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        manager.endpoint()
    service.response_override = None
    await owner.close()
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_before_post_is_tombstoned(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    request = str(uuid4())
    await manager.cancel_request(request)
    assert (await manager.prepare(request))["state"] == "cancelled"
    assert not service.calls and not service.processes
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_one_inflight_prepare_does_not_cancel_other(monkeypatch):
    service = Service()
    service.post_release.clear()
    manager = managed(service, monkeypatch)
    one_id, two_id = str(uuid4()), str(uuid4())
    one = asyncio.create_task(manager.prepare(one_id))
    two = asyncio.create_task(manager.prepare(two_id))
    await service.post_entered.wait()
    one.cancel()
    with pytest.raises(asyncio.CancelledError):
        await one
    service.post_release.set()
    result = await two
    await settle(manager)
    assert result["ready"] and manager._owned.process.returncode is None
    assert {item.request_id for item in manager._leases.values()} == {two_id}
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_last_during_spawn_retires_late_owned_process(monkeypatch):
    service = Service()
    gate, entered = asyncio.Event(), asyncio.Event()
    manager = managed(service, monkeypatch)
    async def spawn(*args, **kwargs):
        entered.set()
        await gate.wait()
        return await service.spawn(*args, **kwargs)
    manager._spawn = spawn
    task = asyncio.create_task(manager.prepare(str(uuid4())))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()
    await settle(manager)
    assert len(service.processes) == 1 and service.processes[0].returncode is not None
    assert not service.calls or all(not call[1].endswith("/prepare") for call in service.calls)
    await manager.close()


@pytest.mark.asyncio
async def test_close_while_post_response_late_releases_lease(monkeypatch):
    service = Service()
    service.post_release.clear()
    manager = managed(service, monkeypatch)
    task = asyncio.create_task(manager.prepare(str(uuid4())))
    await service.post_entered.wait()
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    service.post_release.set()
    result = await task
    await closing
    assert result["state"] == "cancelled" and not manager._leases
    assert service.processes[0].returncode is not None
    with pytest.raises(local.LocalServiceError, match="CLOSED"):
        await manager.prepare(str(uuid4()))


@pytest.mark.asyncio
async def test_failed_spawn_or_job_is_retired_and_retryable(monkeypatch):
    service = Service()
    def bad_job(_):
        raise OSError("do not log")
    manager = managed(service, monkeypatch, job_factory=bad_job)
    with pytest.raises(local.LocalServiceError, match="START_FAILED"):
        await manager.prepare(str(uuid4()))
    assert service.processes[0].killed
    manager._job_factory = Job
    assert (await manager.prepare(str(uuid4())))["ready"]
    await manager.close()


@pytest.mark.parametrize("event", [{"event": "listening", "addresses": [["evil", 50120]], "instance_id": str(uuid4())},
                                  {"event": "listening", "addresses": [], "instance_id": str(uuid4())},
                                  {"event": "wrong"}])
@pytest.mark.asyncio
async def test_invalid_listening_event_cleans_process(monkeypatch, event):
    service = Service()
    manager = managed(service, monkeypatch)
    async def spawn(*args, **kwargs):
        process = Process(event)
        service.processes.append(process)
        return process
    manager._spawn = spawn
    with pytest.raises(local.LocalServiceError, match="PROTOCOL"):
        await manager.prepare(str(uuid4()))
    assert service.processes[0].returncode is not None
    await manager.close()


@pytest.mark.parametrize("response,code", [(httpx.Response(401, json={}), "REJECTED"),
                                         (httpx.Response(200, json=[]), "PROTOCOL"),
                                         (httpx.Response(200, content=b"x" * 32769), "REJECTED"),
                                         (TimeoutError(), "TIMEOUT"), (OSError("secret"), "UNAVAILABLE")])
@pytest.mark.asyncio
async def test_http_faults_fail_with_stable_codes(response, code):
    service = Service()
    service.response_override = response
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    with pytest.raises(local.LocalServiceError, match=code):
        await manager.prepare(str(uuid4()))
    await manager.close()


@pytest.mark.asyncio
async def test_acquire_ready_and_deadline_releases_lease():
    service = Service()
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    assert (await manager.acquire_ready(str(uuid4())))["ready"]
    service.ready = False
    with pytest.raises(TimeoutError):
        await manager.acquire_ready(str(uuid4()), deadline=asyncio.get_running_loop().time() + 0.01)
    await settle(manager)
    assert len(manager._leases) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_retired_ledger_and_unknown_cancel_are_bounded():
    service = Service()
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    for _ in range(300):
        request = str(uuid4())
        result = await manager.prepare(request)
        await manager.cancel_request(request)
        assert result["ready"]
    for _ in range(600):
        await manager.cancel_request(str(uuid4()))
    assert len(manager._requests) <= 256 and len(manager._abandoned) <= 512
    unknown = str(uuid4())
    assert await manager.cancel(unknown) == {"lease_id": unknown, "state": "released"}
    await manager.close()


@pytest.mark.asyncio
async def test_facades_external_legacy_and_close(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    service = Service()
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    local._managers[asyncio.get_running_loop()] = manager
    assert local.get_endpoint() == endpoint()
    assert local.retain_ready() is None
    result = await local.prepare(str(uuid4()))
    assert (await local.status(result["operation_id"]))["ready"]
    await local.cancel_request(manager._leases[result["lease_id"]].request_id)
    with pytest.raises(local.LocalServiceError, match="NOT_FOUND"):
        await manager.status(str(uuid4()))
    with pytest.raises(local.LocalServiceError, match="INVALID_REQUEST"):
        await manager.prepare("bad")
    with pytest.raises(local.LocalServiceError, match="INVALID_REQUEST"):
        await manager.cancel_request("bad")
    await local.close()
    await local.close()
    with pytest.raises(local.LocalServiceError, match="CLOSED"):
        local.get_manager()
    with pytest.raises(local.LocalServiceError, match="CLOSED"):
        local.get_endpoint()
    await local.startup()
    assert local.get_manager() is not manager
    await local.close()


@pytest.mark.asyncio
async def test_cancel_request_reaches_service_before_pending_post_returns(monkeypatch):
    service = Service()
    service.post_release.clear()
    manager = managed(service, monkeypatch)
    identity = str(uuid4())
    post = asyncio.create_task(manager.prepare(identity))
    await service.post_entered.wait()
    await manager.cancel_request(identity)
    assert any(call[0] == "DELETE" and call[1].endswith("/v1/prepare/" + identity) for call in service.calls)
    service.post_release.set()
    assert (await post)["state"] == "cancelled"
    await settle(manager)
    assert not manager._leases
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["cancelled", "expired"])
async def test_cancelled_or_expired_service_request_has_no_usable_lease(state):
    service = Service()
    service.response_override = httpx.Response(202, json={"state": state, "ready": False, "operation_id": None,
                                                         "lease_id": None, "expires_in_seconds": 0})
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    identity = str(uuid4())
    assert (await manager.prepare(identity))["state"] == state
    assert not manager._leases
    assert (await manager.prepare(identity))["state"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl,ready", [(0, True), (60, False), (float("nan"), True)])
async def test_inconsistent_ready_is_rejected_and_remote_request_cancelled(ttl, ready):
    service = Service()
    service.response_override = httpx.Response(202, content=json.dumps({"state": "ready", "ready": ready,
                                                                     "operation_id": str(uuid4()), "lease_id": str(uuid4()),
                                                                     "expires_in_seconds": ttl}).encode())
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    identity = str(uuid4())
    with pytest.raises(local.LocalServiceError, match="PROTOCOL"):
        await manager.prepare(identity)
    await settle(manager)
    assert any(call[0] == "DELETE" and call[1].endswith(identity) for call in service.calls)
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object contract")
async def test_real_windows_job_kills_only_owned_subprocess():
    import psutil

    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c",
        "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)",
        stdout=asyncio.subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004,
    )
    job = None
    try:
        import win32job
        job = await asyncio.to_thread(local._WindowsJob, process)
        members = win32job.QueryInformationJobObject(job.handle, win32job.JobObjectBasicProcessIdList)
        assert process.pid in members
        await asyncio.to_thread(job.resume_primary)
        async with asyncio.timeout(3):
            explicit_child = int((await process.stdout.readline()).decode())
        await asyncio.to_thread(job.adopt_descendants)
        children = psutil.Process(process.pid).children(recursive=True)
        identities = [(item.pid, item.create_time()) for item in children]
        members = win32job.QueryInformationJobObject(job.handle, win32job.JobObjectBasicProcessIdList)
        assert explicit_child in members and all(item.pid in members for item in children)
        await asyncio.to_thread(job.close)
        await asyncio.wait_for(process.wait(), 3)
        assert process.returncode is not None
        async with asyncio.timeout(3):
            while any(psutil.pid_exists(pid) and psutil.Process(pid).create_time() == created for pid, created in identities):
                await asyncio.sleep(0.01)
    finally:
        if job is not None:
            await asyncio.to_thread(job.close)
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_drain_failure_still_closes_exact_job(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    await manager.prepare(str(uuid4()))
    owned = manager._owned
    async def fail():
        raise RuntimeError("drain failed")
    monkeypatch.setattr(manager, "_drain_close", fail)
    with pytest.raises(RuntimeError, match="drain failed"):
        await manager.close()
    assert owned.job.closed and owned.process.returncode is not None


@pytest.mark.asyncio
async def test_cancelled_shutdown_caller_does_not_erase_cleanup_owner(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    service = Service()
    service.post_release.clear()
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    loop = asyncio.get_running_loop()
    local._managers[loop] = manager
    preparing = asyncio.create_task(manager.prepare(str(uuid4())))
    await service.post_entered.wait()
    closing = asyncio.create_task(local.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert local._closing_managers[loop] is manager
    starting = asyncio.create_task(local.startup())
    await asyncio.sleep(0)
    assert not starting.done()
    service.post_release.set()
    assert (await preparing)["state"] == "cancelled"
    await starting
    assert manager._close_task.done()
    await local.close()


@pytest.mark.asyncio
async def test_managed_worker_rejects_a_different_supervisor_identity(monkeypatch):
    from main_logic.asr_client._infra import AsrSessionConfig, _AsrRequestQueue
    from main_logic.asr_client.workers.local_streaming import local_streaming_asr_worker
    from tests.unit.test_asr_local_streaming_worker import _Connect, _event, _Socket

    monkeypatch.setattr(local, "get_endpoint", lambda: local.Endpoint(local.DEFAULT_URL, "x" * 32, str(uuid4()), True))
    requests, responses = _AsrRequestQueue(), asyncio.Queue()
    task = asyncio.create_task(local_streaming_asr_worker(requests, responses, "", AsrSessionConfig(),
                                                         connect_factory=_Connect(_Socket())))
    event = await _event(responses, "error")
    assert event.error_code == "ASR_LOCAL_PROTOCOL_ERROR"
    await task


@pytest.mark.asyncio
async def test_voice_close_during_manager_shutdown_hands_cleanup_to_manager(monkeypatch):
    service = Service()
    manager = managed(service, monkeypatch)
    result = await manager.prepare(str(uuid4()))
    voice = manager.retain_ready()
    record = manager._leases[result["lease_id"]]
    closing = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    await voice.close()
    await closing
    assert record.refs == 0 and not manager._leases


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object contract")
async def test_real_native_job_fallback_when_shutdown_is_rejected(clean_env, monkeypatch, tmp_path):
    if os.getenv("NEKO_ASR_REAL_LIFECYCLE") != "1" or not all(clean_env.values()):
        pytest.skip("Opt in with existing managed ASR Python/root/model/speech paths")
    import psutil
    import win32job

    manager = local.LocalServiceManager(local.ManagedConfig(*(clean_env[name] for name in local._MANAGED_FIELDS)))
    identities = []
    try:
        ready = await manager.acquire_ready(str(uuid4()))
        assert ready["ready"]
        owned = manager._owned
        root = psutil.Process(owned.process.pid)
        processes = [root, *root.children(recursive=True)]
        members = set(win32job.QueryInformationJobObject(owned.job.handle, win32job.JobObjectBasicProcessIdList))
        native_found = False
        for item in processes:
            native = any("spawn_main" in arg or arg == "--multiprocessing-fork" for arg in item.cmdline())
            native_found |= native
            identities.append({"pid": item.pid, "created": item.create_time(), "native": native})
        assert native_found and {item["pid"] for item in identities} <= members
        original = manager._request
        async def reject_shutdown(endpoint, method, path, *args, **kwargs):
            if path == "/v1/shutdown":
                raise local.LocalServiceError("ASR_LOCAL_PREPARE_REJECTED")
            return await original(endpoint, method, path, *args, **kwargs)
        monkeypatch.setattr(manager, "_request", reject_shutdown)
        await manager.close()
        def alive(identity):
            try:
                item = psutil.Process(identity["pid"])
                return item.is_running() and item.create_time() == identity["created"]
            except psutil.NoSuchProcess:
                return False
        async with asyncio.timeout(3):
            while any(alive(item) for item in identities):
                await asyncio.sleep(0.01)
        from pathlib import Path
        output = Path(os.getenv("NEKO_ASR_HARDKILL_REPORT", str(tmp_path / "hardkill.json")))
        report = {"passed": True, "shutdown_rejected": True, "all_descendants_in_job": True,
                  "all_owned_processes_exited": True, "processes": identities}
        await asyncio.to_thread(output.write_text, json.dumps(report, indent=2), encoding="utf-8")
    finally:
        await manager.close()


class LegacyService(Service):
    def __init__(self):
        super().__init__()
        self.health = {
            "ready": True, "state": "ready", "instance_id": str(uuid4()),
            "bundle": {"model_id": "streaming-paraformer-bilingual-zh-en", "backend_family": "sherpa-onnx-paraformer",
                       "bundle_version": "pinned-test-version", "sample_rate_hz": 16000, "channels": 1, "pcm_format": "pcm16le",
                       "capabilities": {"streaming": True, "manual_commit": True, "partial_semantics": "snapshot"},
                       "finish_contract": {"version": 1, "fresh_stream_per_utterance": True}},
        }
        self.health_status = 200
        self.control_status = 501

    async def request(self, method, url, headers, json=None):
        if url.endswith("/prepare") or url.endswith("/health"):
            self.calls.append((method, url, headers, json))
            if url.endswith("/prepare"):
                return httpx.Response(self.control_status, json={"code": "CONTROL_UNSUPPORTED"})
            return httpx.Response(self.health_status, json=self.health)
        return await super().request(method, url, headers, json)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 501])
async def test_external_legacy_eager_ready_has_no_fake_lease_or_shutdown(status):
    service = LegacyService()
    service.control_status = status
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    request_id = str(uuid4())
    ready = await manager.acquire_ready(request_id)
    assert ready["state"] == "ready" and ready["legacy_eager"] is True
    assert ready["lease_id"] is None and ready["operation_id"] is None
    assert manager.retain_ready() is None and not manager._leases
    before = len(service.calls)
    await manager.cancel_request(request_id)
    await manager.close()
    assert len(service.calls) == before
    assert not any(call[1].endswith("/shutdown") for call in service.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 501])
@pytest.mark.parametrize("fault", ["not-ready", "bad-instance", "bad-model", "bad-pcm", "bad-capability"])
async def test_external_legacy_eager_requires_strict_health(fault, status):
    service = LegacyService()
    service.control_status = status
    if fault == "not-ready":
        service.health.update(ready=False, state="failed")
        service.health_status = 503
    elif fault == "bad-instance":
        service.health["instance_id"] = "bad"
    elif fault == "bad-model":
        service.health["bundle"]["model_id"] = "other-model"
    elif fault == "bad-pcm":
        service.health["bundle"]["sample_rate_hz"] = 48000
    else:
        service.health["bundle"]["capabilities"]["manual_commit"] = False
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    with pytest.raises(local.LocalServiceError, match="NOT_PREPARED"):
        await manager.prepare(str(uuid4()))
    await settle(manager)
    assert not manager._leases
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 501])
async def test_managed_control_unsupported_never_uses_legacy_health(monkeypatch, status):
    service = LegacyService()
    service.control_status = status
    manager = managed(service, monkeypatch)
    with pytest.raises(local.LocalServiceError, match="REJECTED"):
        await manager.prepare(str(uuid4()))
    await settle(manager)
    assert not any(call[1].endswith("/health") for call in service.calls)
    await manager.close()


@pytest.mark.asyncio
async def test_external_conflict_is_not_a_legacy_compatibility_signal():
    service = LegacyService()
    service.control_status = 409
    manager = local.LocalServiceManager(endpoint(), client_factory=service)
    with pytest.raises(local.LocalServiceError, match="REJECTED"):
        await manager.prepare(str(uuid4()))
    await settle(manager)
    assert not any(call[1].endswith("/health") for call in service.calls)
    await manager.close()
