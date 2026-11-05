"""Trusted local ASR preparation and exact ownership of an optional supervisor.

This module never downloads models or runs inference. Managed service settings
come from the host environment; HTTP callers cannot supply commands or paths.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import subprocess
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import httpx

DEFAULT_URL = "ws://127.0.0.1:50120/v1/asr/stream"
_MANAGED_FIELDS = ("NEKO_ASR_SERVICE_PYTHON", "NEKO_ASR_SERVICE_ROOT", "NEKO_ASR_MODEL_DIR", "NEKO_ASR_SMOKE_WAV")
_MANAGED_EXE_FIELD = "NEKO_ASR_SERVICE_EXE"
_TERMINAL = {"failed", "cancelled", "expired", "stopped"}


class LocalServiceError(RuntimeError):
    def __init__(self, code="ASR_LOCAL_SERVICE_UNAVAILABLE", *, status=None):
        self.code = code
        self.status = status
        super().__init__(code)


@dataclass(frozen=True)
class Endpoint:
    url: str
    token: str = field(repr=False)
    instance_id: str | None = None
    managed: bool = False

    @property
    def http_url(self):
        return "http://" + urlsplit(self.url).netloc


@dataclass(frozen=True)
class ManagedConfig:
    python: str
    root: str
    model_dir: str
    smoke_wav: str
    threads: int = 1
    idle_seconds: float = 300.0
    catalog: str | None = None
    state_dir: str | None = None
    memory_budget_mb: int = 2048
    executable: str | None = None


def _uuid(value):
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def configuration():
    """Read syntax only; no filesystem, subprocess or network work in UI probes."""
    values = [os.getenv(name, "") for name in _MANAGED_FIELDS]
    executable = os.getenv(_MANAGED_EXE_FIELD, "")
    if executable and values[0]:
        raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID")
    managed_values = [executable, *values[1:]] if executable else values
    catalog, state_dir = os.getenv("NEKO_ASR_CATALOG", ""), os.getenv("NEKO_ASR_STATE_DIR", "")
    if (catalog or state_dir) and not all(managed_values):
        raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID")
    if any(managed_values):
        try:
            threads = int(os.getenv("NEKO_ASR_SERVICE_THREADS", "1"))
            idle = float(os.getenv("NEKO_ASR_IDLE_SECONDS", "300"))
            budget = int(os.getenv("NEKO_ASR_MEMORY_BUDGET_MB", "2048"))
            if (not all(managed_values) or not all(Path(value).is_absolute() for value in managed_values)
                    or threads not in {1, 2, 3, 4} or not math.isfinite(idle) or not 0 < idle <= 86400
                    or not 1 <= budget <= 1048576
                    or bool(catalog) != bool(state_dir)
                    or (catalog and (not Path(catalog).is_absolute() or not Path(state_dir).is_absolute()))):
                raise ValueError
        except (ValueError, TypeError):
            raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID") from None
        return ManagedConfig(*values, threads=threads, idle_seconds=idle,
                             catalog=catalog or None, state_dir=state_dir or None, memory_budget_mb=budget,
                             executable=executable or None)
    url, token = os.getenv("NEKO_ASR_URL", DEFAULT_URL), os.getenv("NEKO_ASR_TOKEN", "")
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "ws" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                 and parsed.path == "/v1/asr/stream" and not parsed.query and not parsed.fragment
                 and not parsed.username and not parsed.password and parsed.port is not None
                 and 1 <= parsed.port <= 65535 and not any(c.isspace() for c in url))
    except (ValueError, TypeError):
        valid = False
    if not valid or not 32 <= len(token) <= 256 or any(not 33 <= ord(c) <= 126 for c in token):
        raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID")
    return Endpoint(url, token)


def configuration_is_available():
    try:
        configuration()
        return True
    except LocalServiceError:
        return False


class _WindowsJob:
    """Retain exact handles, including launcher children born before assignment."""
    def __init__(self, process):
        self.handle = None
        self._handles = {}
        self.identities = {}
        self._guard = threading.RLock()
        if os.name != "nt":
            return
        import psutil
        import win32job
        self._root = psutil.Process(process.pid)
        self._root_created = self._root.create_time()
        self.handle = win32job.CreateJobObject(None, "")
        try:
            limits = win32job.QueryInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation)
            limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation, limits)
            win32job.AssignProcessToJobObject(self.handle, self._retain(self._root))
            self.adopt_descendants()
        except BaseException:
            self.close()
            raise

    def _retain(self, item):
        import win32api
        import win32con
        import win32process
        created = item.create_time()
        if item.pid in self._handles:
            if self.identities[item.pid] != created or not item.is_running():
                raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
            return self._handles[item.pid]
        handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE
                                     | win32con.PROCESS_QUERY_INFORMATION | win32con.SYNCHRONIZE, False, item.pid)
        try:
            exact_created = win32process.GetProcessTimes(handle)["CreationTime"].timestamp()
            # pywin32 datetime has millisecond precision; retain the handle so
            # the PID cannot be recycled after this identity check.
            if not item.is_running() or abs(exact_created - created) > 0.001:
                raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
        except BaseException:
            handle.Close()
            raise
        self._handles[item.pid], self.identities[item.pid] = handle, created
        return handle

    def adopt_descendants(self):
        import psutil
        import win32event
        import win32job
        with self._guard:
            if self.handle is None:
                raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
            for _ in range(4):
                if not self._root.is_running() or self._root.create_time() != self._root_created:
                    raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
                descendants = self._root.children(recursive=True)
                for item in descendants:
                    try:
                        self._retain(item)
                    except psutil.NoSuchProcess:
                        continue
                members = set(win32job.QueryInformationJobObject(self.handle, win32job.JobObjectBasicProcessIdList))
                for pid, handle in self._handles.items():
                    if pid in members or win32job.IsProcessInJob(handle, self.handle) or win32event.WaitForSingleObject(handle, 0) == win32event.WAIT_OBJECT_0:
                        continue
                    try:
                        win32job.AssignProcessToJobObject(self.handle, handle)
                    except Exception:
                        if not win32job.IsProcessInJob(handle, self.handle) and win32event.WaitForSingleObject(handle, 0) != win32event.WAIT_OBJECT_0:
                            raise
                current = self._root.children(recursive=True)
                members = set(win32job.QueryInformationJobObject(self.handle, win32job.JobObjectBasicProcessIdList))
                if all(item.pid in members for item in current):
                    return
            raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")

    def resume_primary(self):
        import ctypes
        from ctypes import wintypes

        import win32api
        import win32con
        import win32process

        with self._guard:
            if self.handle is None or not self._root.is_running():
                raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
            threads = self._root.threads()
            if len(threads) != 1:
                raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
            handle = win32api.OpenThread(win32con.THREAD_SUSPEND_RESUME | win32con.THREAD_QUERY_INFORMATION,
                                        False, threads[0].id)
            try:
                owner = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessIdOfThread
                owner.argtypes = [wintypes.HANDLE]
                owner.restype = wintypes.DWORD
                if owner(int(handle)) != self._root.pid or win32process.ResumeThread(handle) != 1:
                    raise LocalServiceError("ASR_LOCAL_PROCESS_OWNERSHIP_LOST")
            finally:
                handle.Close()

    def close(self):
        with self._guard:
            members = set()
            try:
                if self.handle is not None:
                    import win32job
                    members = set(win32job.QueryInformationJobObject(self.handle, win32job.JobObjectBasicProcessIdList))
                    self.handle.Close()
                    self.handle = None
            finally:
                import win32api
                import win32event
                for pid, handle in self._handles.items():
                    try:
                        if pid not in members and win32event.WaitForSingleObject(handle, 0) != win32event.WAIT_OBJECT_0:
                            try:
                                win32api.TerminateProcess(handle, 1)
                            except Exception:
                                if win32event.WaitForSingleObject(handle, 1000) != win32event.WAIT_OBJECT_0:
                                    raise
                    finally:
                        handle.Close()
                self._handles.clear()


@dataclass(eq=False)
class _Owned:
    process: object
    job: object
    endpoint: Endpoint | None = None
    closing: asyncio.Task | None = None


@dataclass(eq=False)
class _Lease:
    endpoint: Endpoint
    request_id: str
    operation_id: str
    expires_at: float
    ready: bool = False
    refs: int = 0
    released: bool = False
    failed: bool = False
    renew_task: asyncio.Task | None = None


class VoiceLease:
    def __init__(self, manager, identity, lease):
        self._manager, self._identity, self._lease = manager, identity, lease
        self._closing = None

    async def close(self):
        if self._closing is None:
            self._closing = asyncio.create_task(self._manager._release_voice(self._identity, self._lease))
            self._closing.add_done_callback(_observe)
        await asyncio.shield(self._closing)


def _observe(task):
    if not task.cancelled():
        task.exception()


class LocalServiceManager:
    def __init__(self, config=None, *, spawn=None, client_factory=None, job_factory=None):
        self.config = configuration() if config is None else config
        self._spawn = spawn or asyncio.create_subprocess_exec
        self._client_factory = client_factory or httpx.AsyncClient
        self._job_factory = job_factory or _WindowsJob
        self._owned = None
        self._start_task = None
        self._requests = {}
        self._request_waiters = {}
        self._active = set()
        self._abandoned = {}
        self._leases = {}
        self._operations = {}
        self._request_endpoints = {}
        self._legacy = set()
        self._ready_until = 0.0
        self._closing = False
        self._close_task = None
        self._cleanups = set()
        self._spawn_tasks = set()
        self._close_deadline = None

    def _cleanup(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._cleanups.add(task)
        task.add_done_callback(self._cleanups.discard)
        task.add_done_callback(_observe)
        return task

    async def _request(self, endpoint, method, path, body=None, *, deadline=None, statuses=(200, 202)):
        budget = min(5.0, max(0.0, deadline - asyncio.get_running_loop().time())) if deadline else 5.0
        if self._close_deadline is not None:
            budget = min(budget, max(0.01, self._close_deadline - asyncio.get_running_loop().time()))
        if budget <= 0:
            raise LocalServiceError("ASR_LOCAL_PREPARE_TIMEOUT")
        try:
            async with asyncio.timeout(budget):
                async with self._client_factory(trust_env=False, timeout=budget, follow_redirects=False) as client:
                    response = await client.request(method, endpoint.http_url + path,
                                                    headers={"Authorization": "Bearer " + endpoint.token}, json=body)
                    if response.status_code not in statuses or len(response.content) > 32768:
                        raise LocalServiceError("ASR_LOCAL_PREPARE_REJECTED", status=response.status_code)
                    result = response.json()
                    if not isinstance(result, dict):
                        raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
                    return result
        except LocalServiceError:
            raise
        except TimeoutError:
            raise LocalServiceError("ASR_LOCAL_PREPARE_TIMEOUT") from None
        except Exception:
            raise LocalServiceError() from None

    @staticmethod
    def _validate_files(config):
        if config.executable:
            if config.python or not Path(config.executable).is_file() or not Path(config.root).is_dir():
                raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID")
        elif not Path(config.python).is_file() or not (Path(config.root) / "src/neko_asr/cli.py").is_file():
            raise LocalServiceError("ASR_LOCAL_CONFIG_INVALID")

    async def _start(self):
        config = self.config
        owned = None
        try:
            async with asyncio.timeout(10):
                await asyncio.to_thread(self._validate_files, config)
                token = secrets.token_hex(32)
                env = dict(os.environ, NEKO_ASR_TOKEN=token)
                if config.executable:
                    env.pop("PYTHONPATH", None)
                    env.pop("PYTHONHOME", None)
                else:
                    env["PYTHONPATH"] = str(Path(config.root) / "src")
                kwargs = {"cwd": config.root, "env": env, "stdout": asyncio.subprocess.PIPE,
                          "stderr": asyncio.subprocess.DEVNULL, "limit": 16384}
                if os.name == "nt":
                    # The venv redirector must not execute until its root is in
                    # our job; assigning later does not adopt existing children.
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | 0x00000004
                spawn_task = asyncio.create_task(self._spawn_owned(config, kwargs))
                self._spawn_tasks.add(spawn_task)
                spawn_task.add_done_callback(self._spawn_tasks.discard)
                spawn_task.add_done_callback(_observe)
                try:
                    owned = await asyncio.shield(spawn_task)
                except BaseException:
                    self._cleanup(self._retire_late_spawn(spawn_task))
                    raise
                process = owned.process
                self._owned = owned
                for _ in range(16):
                    line = await process.stdout.readline()
                    if not line or len(line) > 8192:
                        raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("event") == "listening":
                        addresses = event.get("addresses")
                        instance = event.get("instance_id")
                        if not isinstance(addresses, list) or not addresses or not _uuid(instance):
                            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
                        host, port, *_ = addresses[0]
                        if host != "127.0.0.1" or type(port) is not int or not 1 <= port <= 65535:
                            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
                        owned.endpoint = Endpoint(f"ws://127.0.0.1:{port}/v1/asr/stream", token, instance, True)
                        adopt = getattr(owned.job, "adopt_descendants", None)
                        if adopt is not None:
                            barrier = asyncio.create_task(asyncio.to_thread(adopt))
                            barrier.add_done_callback(_observe)
                            await asyncio.shield(barrier)
                        if self._closing:
                            raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
                        return owned.endpoint
                raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        except BaseException as error:
            if owned is not None:
                await self._stop(owned)
                if self._owned is owned:
                    self._owned = None
            if isinstance(error, asyncio.CancelledError):
                raise
            if isinstance(error, LocalServiceError):
                raise
            raise LocalServiceError("ASR_LOCAL_SERVICE_START_FAILED") from None

    async def _spawn_owned(self, config, kwargs):
        reload_args = (["--catalog", config.catalog, "--state-dir", config.state_dir,
                        "--memory-budget-mb", str(config.memory_budget_mb)] if config.catalog else [])
        command = [config.executable] if config.executable else [config.python, "-m", "neko_asr.cli"]
        process = await self._spawn(*command, "--lazy", "--host", "127.0.0.1",
                                    "--port", "0", "--model-dir", config.model_dir,
                                    "--smoke-wav", config.smoke_wav, "--threads", str(config.threads),
                                    "--idle-seconds", str(config.idle_seconds), *reload_args, **kwargs)
        owned = _Owned(process, None)
        try:
            owned.job = await asyncio.to_thread(self._job_factory, process)
            resume = getattr(owned.job, "resume_primary", None)
            if resume is not None:
                await asyncio.to_thread(resume)
        except BaseException:
            await self._stop(owned)
            raise
        return owned

    async def _retire_late_spawn(self, task):
        try:
            await self._stop(await asyncio.shield(task))
        except Exception:
            pass

    async def _endpoint(self):
        if self._closing:
            raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
        if isinstance(self.config, Endpoint):
            return self.config
        if self._owned is not None and self._owned.process.returncode is not None:
            old, self._owned = self._owned, None
            self._start_task = None
            self._ready_until = 0
            self._cleanup(self._stop(old))  # noqa: ASYNC_BLOCK # this manager's _stop is async; cleanup schedules its coroutine.
        if self._start_task is None or self._start_task.done() and self._start_task.exception() is not None:
            self._start_task = asyncio.create_task(self._start())
            self._start_task.add_done_callback(_observe)
        return await asyncio.shield(self._start_task)

    def endpoint(self):
        """The worker may connect only after independent preparation succeeded."""
        if isinstance(self.config, Endpoint):
            return self.config
        owned = self._owned
        if (self._closing or owned is None or owned.process.returncode is not None
                or owned.endpoint is None or not self._ready_leases(owned.endpoint)):
            raise LocalServiceError("ASR_LOCAL_NOT_PREPARED")
        return owned.endpoint

    def control_endpoint(self):
        """Read an already-running control plane without preparing a model."""
        if self._closing:
            raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
        if isinstance(self.config, Endpoint):
            return self.config
        owned = self._owned
        if owned is None or owned.process.returncode is not None or owned.endpoint is None:
            raise LocalServiceError("ASR_LOCAL_NOT_PREPARED")
        return owned.endpoint

    async def prepare(self, request_id):
        if not _uuid(request_id):
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        if self._closing:
            raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
        if request_id in self._abandoned:
            return {"request_id": request_id, "state": "cancelled", "ready": False}
        if request_id not in self._requests:
            self._prune()
            if len(self._requests) >= 256:
                raise LocalServiceError("ASR_LOCAL_PREPARE_LIMIT")
            self._active.add(request_id)
            task = asyncio.create_task(self._prepare(request_id))
            task.add_done_callback(_observe)
            self._requests[request_id] = task
        task = self._requests[request_id]
        self._request_waiters[request_id] = self._request_waiters.get(request_id, 0) + 1
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._request_waiters[request_id] == 1:
                self._remember_cancel(request_id)
                self._active.discard(request_id)
                self._cleanup(self._cancel_late(task))
            raise
        finally:
            self._request_waiters[request_id] -= 1

    def _prune(self):
        for request_id, task in tuple(self._requests.items()):
            if len(self._requests) < 192:
                break
            if task.done() and request_id not in self._active and not self._request_waiters.get(request_id):
                self._requests.pop(request_id, None)
                self._request_waiters.pop(request_id, None)
                self._request_endpoints.pop(request_id, None)
                self._legacy.discard(request_id)
        active_ops = {lease.operation_id for lease in self._leases.values()}
        for identity in tuple(self._operations):
            if len(self._operations) < 192:
                break
            if identity not in active_ops:
                self._operations.pop(identity, None)

    def _remember_cancel(self, request_id):
        self._abandoned[request_id] = asyncio.get_running_loop().time()
        for identity in tuple(self._abandoned):
            if len(self._abandoned) <= 512:
                break
            task = self._requests.get(identity)
            if identity != request_id and identity not in self._active and (task is None or task.done()):
                self._abandoned.pop(identity, None)

    async def _prepare(self, request_id):
        endpoint = None
        try:
            endpoint = await self._endpoint()
            if request_id in self._abandoned:
                return {"state": "cancelled", "ready": False}
            self._request_endpoints[request_id] = endpoint
            try:
                result = await self._request(endpoint, "POST", "/v1/prepare", {"request_id": request_id})
            except LocalServiceError as error:
                if isinstance(self.config, Endpoint) and error.status in {404, 501}:
                    return await self._legacy_ready(endpoint, request_id)
                raise
            if result.get("state") in {"cancelled", "expired"} and result.get("ready") is False:
                self._active.discard(request_id)
                self._remember_cancel(request_id)
                return result
            operation, lease = result.get("operation_id"), result.get("lease_id")
            ttl = result.get("expires_in_seconds")
            if (not _uuid(operation) or not _uuid(lease) or type(ttl) not in {int, float}
                    or not math.isfinite(ttl) or not 0 <= ttl <= 86400
                    or (result.get("state") == "ready" and (result.get("ready") is not True or ttl <= 0))):
                raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
            expiry = asyncio.get_running_loop().time() + ttl
            self._leases[lease] = _Lease(endpoint, request_id, operation, expiry,
                                         ready=result.get("state") == "ready" and result.get("ready") is True)
            self._operations[operation] = endpoint
            if result.get("state") == "ready" and result.get("ready") is True:
                self._ready_until = max(self._ready_until, expiry)
            if self._closing or request_id in self._abandoned:
                await self.cancel(lease)
                return dict(result, state="cancelled", ready=False)
            return result
        except BaseException:
            self._active.discard(request_id)
            if endpoint is not None:
                self._cleanup(self._cancel_remote_request(endpoint, request_id))
            raise

    async def _legacy_ready(self, endpoint, request_id):
        health = await self._request(endpoint, "GET", "/v1/health", statuses=(200, 503))
        bundle = health.get("bundle")
        caps = bundle.get("capabilities") if isinstance(bundle, dict) else None
        finish = bundle.get("finish_contract") if isinstance(bundle, dict) else None
        if (health.get("ready") is not True or health.get("state") != "ready" or not _uuid(health.get("instance_id"))
                or not isinstance(bundle, dict) or bundle.get("model_id") != "streaming-paraformer-bilingual-zh-en"
                or bundle.get("backend_family") != "sherpa-onnx-paraformer" or not isinstance(bundle.get("bundle_version"), str)
                or not bundle["bundle_version"] or type(bundle.get("sample_rate_hz")) is not int or bundle["sample_rate_hz"] != 16000
                or type(bundle.get("channels")) is not int or bundle["channels"] != 1 or bundle.get("pcm_format") != "pcm16le"
                or not isinstance(caps, dict) or caps.get("streaming") is not True or caps.get("manual_commit") is not True
                or caps.get("partial_semantics") != "snapshot" or not isinstance(finish, dict)
                or type(finish.get("version")) is not int or finish["version"] != 1
                or finish.get("fresh_stream_per_utterance") is not True):
            raise LocalServiceError("ASR_LOCAL_NOT_PREPARED")
        if self._closing or request_id in self._abandoned:
            self._active.discard(request_id)
            return {"request_id": request_id, "state": "cancelled", "ready": False}
        self._legacy.add(request_id)
        return {"request_id": request_id, "operation_id": None, "lease_id": None,
                "state": "ready", "ready": True, "expires_in_seconds": 0, "legacy_eager": True}

    async def _cancel_remote_request(self, endpoint, request_id):
        try:
            return await self._request(endpoint, "DELETE", "/v1/prepare/" + request_id)
        except Exception:
            # Service TTL remains the bounded fallback if the transport is down.
            return {"request_id": request_id, "state": "cancelled"}

    async def _cancel_late(self, task):
        try:
            result = await asyncio.shield(task)
            if result.get("lease_id"):
                await self.cancel(result["lease_id"])
        except Exception:
            pass
        finally:
            owned = self._owned
            if owned is not None and not self._active:
                self._owned, self._start_task, self._ready_until = None, None, 0
                await self._stop(owned)

    async def status(self, operation_id):
        endpoint = self._operations.get(operation_id)
        if endpoint is None:
            raise LocalServiceError("ASR_LOCAL_OPERATION_NOT_FOUND")
        result = await self._request(endpoint, "GET", "/v1/operations/" + operation_id)
        if result.get("operation_id") != operation_id:
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        if result.get("state") == "ready" and result.get("ready") is True:
            matches = [item for item in self._leases.values() if item.endpoint is endpoint and item.operation_id == operation_id]
            for item in matches:
                item.ready = True
            expiries = [item.expires_at for item in matches]
            if expiries:
                self._ready_until = max(self._ready_until, max(expiries))
        return result

    async def cancel(self, lease_id):
        item = self._leases.get(lease_id)
        if item is None:
            return {"lease_id": lease_id, "state": "released"}
        item.released = True
        if item.refs:
            return {"lease_id": lease_id, "state": "retained"}
        self._leases.pop(lease_id, None)
        endpoint, request_id = item.endpoint, item.request_id
        self._active.discard(request_id)
        if not self._active:
            self._ready_until = 0
        return await self._request(endpoint, "DELETE", "/v1/leases/" + lease_id)

    async def cancel_request(self, request_id):
        if not _uuid(request_id):
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        self._remember_cancel(request_id)
        if request_id in self._legacy:
            self._active.discard(request_id)
            return {"request_id": request_id, "state": "cancelled"}
        task = self._requests.get(request_id)
        leases = [(identity, item) for identity, item in self._leases.items() if item.request_id == request_id]
        if leases:
            return await self.cancel(leases[0][0])
        self._active.discard(request_id)
        endpoint = self._request_endpoints.get(request_id)
        if endpoint is not None:
            await self._cancel_remote_request(endpoint, request_id)
        if task is not None and not task.done():
            self._cleanup(self._cancel_late(task))
        return {"request_id": request_id, "state": "cancelled"}

    def _ready_leases(self, endpoint):
        now = asyncio.get_running_loop().time()
        return [(identity, item) for identity, item in self._leases.items()
                if item.endpoint is endpoint and item.ready and not item.failed and item.expires_at > now
                and (not item.released or item.refs)]

    def retain_ready(self):
        if isinstance(self.config, Endpoint):
            matches = self._ready_leases(self.config)
            if not matches:
                return None  # Legacy eager external services need no host lease.
        else:
            endpoint = self.endpoint()
            matches = self._ready_leases(endpoint)
        identity, item = max(matches, key=lambda pair: pair[1].expires_at)
        item.refs += 1
        if item.renew_task is None or item.renew_task.done():
            item.renew_task = asyncio.create_task(self._renew(identity, item))
            item.renew_task.add_done_callback(_observe)
        return VoiceLease(self, identity, item)

    async def _renew(self, identity, item):
        try:
            while item.refs and self._leases.get(identity) is item and not self._closing:
                remaining = item.expires_at - asyncio.get_running_loop().time()
                await asyncio.sleep(min(20.0, max(0.01, remaining / 3)))
                if not item.refs or self._leases.get(identity) is not item or self._closing:
                    return
                response = await self._request(item.endpoint, "POST", "/v1/leases/" + identity + "/renew", {})
                ttl = response.get("expires_in_seconds")
                if (type(ttl) not in {int, float} or not math.isfinite(ttl) or not 0 < ttl <= 86400
                        or response.get("state") in _TERMINAL):
                    raise LocalServiceError("ASR_LOCAL_LEASE_EXPIRED")
                if self._leases.get(identity) is item and item.refs:
                    item.expires_at = asyncio.get_running_loop().time() + ttl
        except asyncio.CancelledError:
            raise
        except Exception:
            item.failed = True

    async def _release_voice(self, identity, item):
        if self._closing:
            await self.close()
            return
        if self._leases.get(identity) is not item:
            return
        item.refs -= 1
        if item.refs:
            return
        task = item.renew_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._leases.get(identity) is item:
            await self.cancel(identity)

    async def acquire_ready(self, request_id, *, deadline=None):
        deadline = deadline or asyncio.get_running_loop().time() + 15.0
        result = None
        try:
            async with asyncio.timeout_at(deadline):
                result = await self.prepare(request_id)
                while not (result.get("state") == "ready" and result.get("ready") is True):
                    if result.get("state") in _TERMINAL:
                        raise LocalServiceError(result.get("error_code") or "ASR_LOCAL_PREPARE_FAILED")
                    await asyncio.sleep(min(0.1, max(0.0, deadline - asyncio.get_running_loop().time())))
                    result = dict(result, **await self.status(result["operation_id"]))
                return result
        except BaseException:
            if result is not None and result.get("lease_id"):
                self._cleanup(self.cancel(result["lease_id"]))
            raise

    async def _stop(self, owned):
        if owned.closing is None:
            owned.closing = asyncio.create_task(self._stop_impl(owned))
            owned.closing.add_done_callback(_observe)
        await asyncio.shield(owned.closing)

    async def _stop_impl(self, owned):
        process = owned.process
        try:
            if owned.endpoint is not None and process.returncode is None:
                try:
                    await self._request(owned.endpoint, "POST", "/v1/shutdown", {"instance_id": owned.endpoint.instance_id})
                    await asyncio.wait_for(asyncio.shield(process.wait()), self._remaining_close(3))
                except Exception:
                    pass
        finally:
            try:
                if owned.job is not None:
                    await asyncio.to_thread(owned.job.close)
            finally:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await asyncio.wait_for(asyncio.shield(process.wait()), self._remaining_close(3))

    def _remaining_close(self, maximum):
        if self._close_deadline is None:
            return maximum
        return min(maximum, max(0.01, self._close_deadline - asyncio.get_running_loop().time()))

    async def close(self):
        if self._close_task is None:
            self._closing = True
            self._close_deadline = asyncio.get_running_loop().time() + 15
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(_observe)
        await asyncio.shield(self._close_task)

    async def _close(self):
        try:
            await self._drain_close()
        finally:
            owned, self._owned = self._owned, None
            self._ready_until = 0
            if owned is not None:
                await self._stop(owned)

    async def _drain_close(self):
        reload_control = getattr(self, "_reload_control", None)
        if reload_control is not None:
            try:
                await asyncio.wait_for(reload_control.close(), self._remaining_close(6))
            except Exception:
                pass
        for request in self._active.copy():
            self._remember_cancel(request)
        self._active.clear()
        requests = tuple(self._requests.values())
        if requests:
            try:
                await asyncio.wait_for(asyncio.gather(*(asyncio.shield(task) for task in requests), return_exceptions=True), self._remaining_close(12))
            except TimeoutError:
                pass
        renewals = []
        for item in self._leases.values():
            item.refs = 0
            if item.renew_task is not None:
                item.renew_task.cancel()
                renewals.append(item.renew_task)
        if renewals:
            try:
                await asyncio.wait_for(asyncio.gather(*renewals, return_exceptions=True), self._remaining_close(3))
            except TimeoutError:
                pass
        leases = tuple(self._leases)
        if leases:
            await asyncio.gather(*(self.cancel(lease) for lease in leases), return_exceptions=True)
        if self._start_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._start_task), self._remaining_close(12))
            except Exception:
                pass
        pending = tuple(self._cleanups)
        if pending:
            await asyncio.wait_for(asyncio.gather(*(asyncio.shield(task) for task in pending), return_exceptions=True), self._remaining_close(12))


_managers = weakref.WeakKeyDictionary()
_closed_loops = weakref.WeakSet()
_closing_managers = weakref.WeakKeyDictionary()


def get_manager():
    loop = asyncio.get_running_loop()
    if loop in _closed_loops:
        raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
    manager = _managers.get(loop)
    if manager is None:
        manager = LocalServiceManager()
        _managers[loop] = manager
    return manager


def get_endpoint():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop in _closed_loops:
        raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")
    config = configuration()
    if isinstance(config, Endpoint):
        return config
    if loop is None:
        raise LocalServiceError("ASR_LOCAL_NOT_PREPARED")
    return get_manager().endpoint()


async def close():
    loop = asyncio.get_running_loop()
    _closed_loops.add(loop)
    manager = _managers.pop(loop, None)
    if manager is not None:
        _closing_managers[loop] = manager
        try:
            await manager.close()
        finally:
            def finished(_):
                if _closing_managers.get(loop) is manager:
                    _closing_managers.pop(loop, None)
            if manager._close_task.done():
                finished(None)
            else:
                manager._close_task.add_done_callback(finished)


async def startup():
    """Explicit application startup may reset a previously closed event loop."""
    loop = asyncio.get_running_loop()
    manager = _closing_managers.get(loop)
    if manager is not None:
        await manager.close()
    _closed_loops.discard(loop)


async def prepare(request_id):
    return await get_manager().prepare(request_id)


async def status(operation_id):
    return await get_manager().status(operation_id)


async def cancel_request(request_id):
    return await get_manager().cancel_request(request_id)


def retain_ready():
    return get_manager().retain_ready()
