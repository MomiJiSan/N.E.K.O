"""Trusted-catalog reload operations; never retry or replay microphone audio."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from . import local_service
from .local_service import LocalServiceError, _observe, _uuid


@dataclass(eq=False)
class _Request:
    identity: str
    candidate: str
    endpoint: object
    task: asyncio.Task | None = None
    operation: str | None = None
    cancelled: bool = False
    result: dict | None = None


class ReloadControl:
    def __init__(self, manager):
        self.manager = manager
        self.requests = {}
        self.operations = {}
        self.tombstones = {}
        self.tasks = set()
        self.closed = False
        self.closing = None

    def _task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(_observe)
        return task

    def _allowed(self):
        if self.closed or self.manager._closing:
            raise LocalServiceError("ASR_LOCAL_MANAGER_CLOSED")

    @staticmethod
    def _document(value, *, operation=None):
        if (not isinstance(value, dict) or not isinstance(value.get("state"), str)
                or len(value["state"]) > 64
                or type(value.get("active_changed")) is not bool
                or type(value.get("retirement_pending")) is not bool):
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        identity = value.get("operation_id")
        if identity is not None and not _uuid(identity):
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        if operation is not None and identity != operation:
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        return value

    def _remember(self, record, value):
        value = self._document(value)
        identity = value.get("operation_id")
        if record.operation is not None and identity not in {None, record.operation}:
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        if value.get("request_id") != record.identity:
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        if identity is not None:
            record.operation = identity
            self.operations[identity] = record
        record.result = dict(value)
        return dict(value)

    def _prune(self):
        for identity, record in tuple(self.requests.items()):
            if len(self.requests) < 96:
                break
            value = record.result or {}
            if (record.task is not None and record.task.done()
                    and value.get("state") in {"completed", "failed", "cancelled", "rolled_back", "requires_pause"}
                    and value.get("retirement_pending") is False):
                self.requests.pop(identity, None)
                if record.operation is not None:
                    self.operations.pop(record.operation, None)

    async def candidates(self):
        self._allowed()
        # Discovery is read-only and never starts a stopped supervisor/model.
        endpoint = self.manager.control_endpoint()
        value = await self.manager._request(endpoint, "GET", "/v1/candidates")
        if not isinstance(value, dict) or not isinstance(value.get("candidates"), list) or len(value["candidates"]) > 128:
            raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
        fields = {"candidate_id", "bundle_id", "bundle_version", "model_id", "manifest_sha256", "enabled"}
        candidates = []
        for candidate in value["candidates"]:
            if not isinstance(candidate, dict):
                raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
            candidates.append({key: candidate[key] for key in fields if key in candidate})
        return {"enabled": value.get("enabled") is True, "candidates": candidates}

    async def reload(self, request_id, candidate_id):
        self._allowed()
        if (not _uuid(request_id) or not isinstance(candidate_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate_id)):
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        if request_id in self.tombstones:
            value = self.tombstones[request_id]
            if value is None:
                raise LocalServiceError("ASR_LOCAL_CANCEL_PENDING")
            return dict(value)
        record = self.requests.get(request_id)  # noqa: ASYNC_BLOCK # requests is an in-memory dict, not the HTTP module.
        if record is not None and record.candidate != candidate_id:
            raise LocalServiceError("ASR_LOCAL_REQUEST_CONFLICT")
        if record is not None and record.operation is not None and record.task.done():
            return await self.status(record.operation)
        if record is None:
            self._prune()
            if len(self.requests) >= 128 or len(self.tombstones) >= 512:
                raise LocalServiceError("ASR_LOCAL_RELOAD_LIMIT")
            endpoint = self.manager.endpoint()
            record = _Request(request_id, candidate_id, endpoint)
            self.requests[request_id] = record
            record.task = self._task(self._submit(record))
        try:
            return await asyncio.shield(record.task)
        except asyncio.CancelledError:
            # A cancelled HTTP waiter cannot cancel another operation or leave
            # an unknown accepted POST free to publish later.
            self._task(self.cancel_request(request_id))
            raise

    async def _submit(self, record):
        if record.cancelled or self.closed:
            return record.result or {"request_id": record.identity, "operation_id": None,
                                     "state": "cancelled", "active_changed": False, "retirement_pending": False}
        try:
            value = await self.manager._request(record.endpoint, "POST", "/v1/reload",
                                               {"request_id": record.identity, "candidate_id": record.candidate})
            value = self._remember(record, value)
            if record.cancelled or self.closed:
                return await self._compensate(record)
            return value
        except BaseException:
            self._task(self._compensate(record))
            raise

    async def _compensate(self, record):
        value = await self.manager._request(record.endpoint, "DELETE", "/v1/reload-requests/" + record.identity)
        return self._remember(record, value)

    async def cancel_request(self, request_id):
        self._allowed()
        return await asyncio.shield(self._task(self._cancel_request(request_id)))

    async def _cancel_request(self, request_id):
        if not _uuid(request_id):
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        record = self.requests.get(request_id)  # noqa: ASYNC_BLOCK # requests is an in-memory dict, not the HTTP module.
        if record is None:
            self._allowed()
            if request_id not in self.tombstones and len(self.tombstones) >= 512:
                raise LocalServiceError("ASR_LOCAL_RELOAD_LIMIT")
            self.tombstones[request_id] = None
            # Persist the fence in an already-running service too: a previous
            # HTTP owner may have lost the accepted POST response. Never spawn
            # a supervisor just to cancel an unknown request.
            try:
                endpoint = self.manager.control_endpoint()
            except LocalServiceError as error:
                if error.code != "ASR_LOCAL_NOT_PREPARED":
                    raise
            else:
                value = await self.manager._request(endpoint, "DELETE", "/v1/reload-requests/" + request_id)
                self._document(value)
                if value.get("request_id") != request_id:
                    raise LocalServiceError("ASR_LOCAL_PROTOCOL_ERROR")
                self.tombstones[request_id] = dict(value)
                if value.get("operation_id") is not None:
                    # Recover control of a remotely accepted operation whose
                    # original POST response or local record was lost.
                    recovered = _Request(request_id, value.get("candidate_id", ""), endpoint)
                    self._remember(recovered, value)
                return dict(self.tombstones[request_id])
            value = {"request_id": request_id, "operation_id": None, "state": "cancelled",
                     "active_changed": False, "retirement_pending": False}
            self.tombstones[request_id] = value
            return dict(value)
        record.cancelled = True
        # The server may report active_changed after the commit boundary. Never
        # relabel that partial success as an unpublished cancellation.
        return await self._compensate(record)

    def _record(self, operation_id):
        if not _uuid(operation_id):
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        record = self.operations.get(operation_id)
        if record is None:
            raise LocalServiceError("ASR_LOCAL_OPERATION_NOT_FOUND")
        return record

    async def status(self, operation_id):
        self._allowed()
        record = self._record(operation_id)
        value = await self.manager._request(record.endpoint, "GET", "/v1/operations/" + operation_id)
        self._document(value, operation=operation_id)
        return self._remember(record, value)

    async def action(self, operation_id, action):
        self._allowed()
        return await asyncio.shield(self._task(self._action(operation_id, action)))

    async def _action(self, operation_id, action):
        if action not in {"cancel", "finalize", "rollback"}:
            raise LocalServiceError("ASR_LOCAL_INVALID_REQUEST")
        record = self._record(operation_id)
        method, suffix = ("DELETE", "") if action == "cancel" else ("POST", "/" + action)
        value = await self.manager._request(record.endpoint, method, "/v1/reloads/" + operation_id + suffix,
                                           {} if method == "POST" else None)
        self._document(value, operation=operation_id)
        return self._remember(record, value)

    async def close(self):
        if self.closing is None:
            self.closed = True
            self.closing = self._task(self._close())
        await asyncio.shield(self.closing)

    async def _close(self):
        pending = []
        for record in tuple(self.requests.values()):
            value = record.result or {}
            if not value.get("active_changed") and value.get("state") not in {"failed", "cancelled", "requires_pause"}:
                record.cancelled = True
                pending.append(self._task(self._compensate(record)))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        running = [record.task for record in self.requests.values() if record.task is not None and not record.task.done()]
        if running:
            await asyncio.gather(*(asyncio.shield(task) for task in running), return_exceptions=True)
        # A failed POST may schedule its compensation while the above await
        # runs. The close owner must observe those tasks as well.
        remaining = [task for task in self.tasks if task is not asyncio.current_task() and not task.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


def get_control():
    manager = local_service.get_manager()
    control = getattr(manager, "_reload_control", None)
    if control is None:
        control = manager._reload_control = ReloadControl(manager)
    return control
