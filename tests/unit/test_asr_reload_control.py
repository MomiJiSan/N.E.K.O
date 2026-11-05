import asyncio
from uuid import uuid4

import pytest

from main_logic.asr_client import reload_control as reload
from main_logic.asr_client.local_service import LocalServiceError


class Manager:
    def __init__(self):
        self._closing = False
        self.address = object()
        self.calls = []
        self.operation = str(uuid4())
        self.request = None
        self.state = "loading"
        self.changed = False
        self.response = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.stopped = False

    def endpoint(self):
        return self.control_endpoint()

    def control_endpoint(self):
        if self.stopped:
            raise LocalServiceError("ASR_LOCAL_NOT_PREPARED")
        return self.address

    def document(self):
        return {"operation_id": self.operation, "request_id": self.request,
                "state": self.state, "active_changed": self.changed, "retirement_pending": self.changed}

    async def _request(self, endpoint, method, path, body=None):
        self.calls.append((endpoint, method, path, body))
        if path == "/v1/reload":
            self.request = body["request_id"]
            self.entered.set()
            await self.release.wait()
        if self.response is not None:
            if isinstance(self.response, Exception):
                raise self.response
            return self.response
        if method == "DELETE" and not self.changed:
            self.state = "cancelled"
        return self.document()


@pytest.fixture
def pair():
    manager = Manager()
    return manager, reload.ReloadControl(manager)


@pytest.mark.asyncio
async def test_idempotent_request_reads_current_status_without_second_post(pair):
    manager, control = pair
    request = str(uuid4())
    first = await control.reload(request, "version-b")
    assert first["state"] == "loading"
    manager.state, manager.changed = "draining", True
    assert (await control.reload(request, "version-b"))["active_changed"] is True
    assert sum(call[1] == "POST" for call in manager.calls) == 1
    with pytest.raises(LocalServiceError, match="REQUEST_CONFLICT"):
        await control.reload(request, "version-c")
    await control.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action,method,suffix", [("cancel", "DELETE", ""), ("finalize", "POST", "/finalize"),
                                                  ("rollback", "POST", "/rollback")])
async def test_control_actions_keep_exact_original_endpoint(pair, action, method, suffix):
    manager, control = pair
    result = await control.reload(str(uuid4()), "b")
    address = manager.address
    manager.address = object()
    await control.status(result["operation_id"])
    await control.action(result["operation_id"], action)
    assert manager.calls[-1][:3] == (address, method, "/v1/reloads/" + manager.operation + suffix)
    await control.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_fences_late_accepted_post(pair):
    manager, control = pair
    manager.release.clear()
    request = str(uuid4())
    task = asyncio.create_task(control.reload(request, "b"))
    await manager.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    manager.release.set()
    await control.close()
    assert control.requests[request].result["state"] == "cancelled"
    assert any(call[1:3] == ("DELETE", "/v1/reload-requests/" + request) for call in manager.calls)
    assert not control.tasks


@pytest.mark.asyncio
async def test_late_cancel_reports_publication_without_fake_rollback(pair):
    manager, control = pair
    request = str(uuid4())
    await control.reload(request, "b")
    manager.changed, manager.state = True, "draining"
    value = await control.cancel_request(request)
    assert value["active_changed"] and value["state"] == "draining"
    await control.close()
    assert not any(call[2].endswith("rollback") for call in manager.calls)


@pytest.mark.asyncio
async def test_unknown_cancel_contacts_running_service_and_preserves_partial_success(pair):
    manager, control = pair
    manager.request = request = str(uuid4())
    manager.changed, manager.state = True, "draining"
    value = await control.cancel_request(request)
    assert value["active_changed"]
    assert (await control.reload(request, "b"))["active_changed"]
    assert [call[1] for call in manager.calls] == ["DELETE"]
    endpoint = manager.address
    manager.address = object()
    assert (await control.status(manager.operation))["active_changed"]
    await control.action(manager.operation, "rollback")
    assert manager.calls[-1][0] is endpoint
    await control.close()


@pytest.mark.asyncio
async def test_cancel_before_post_never_starts_stopped_service(pair):
    manager, control = pair
    manager.stopped = True
    request = str(uuid4())
    assert (await control.cancel_request(request))["state"] == "cancelled"
    assert (await control.reload(request, "b"))["state"] == "cancelled"
    assert not manager.calls
    with pytest.raises(LocalServiceError, match="NOT_PREPARED"):
        await control.candidates()
    await control.close()


@pytest.mark.asyncio
async def test_unknown_cancel_transport_failure_never_claims_confirmed_cancellation(pair):
    manager, control = pair
    request = str(uuid4())
    manager.response = RuntimeError("lost response")
    with pytest.raises(RuntimeError):
        await control.cancel_request(request)
    with pytest.raises(LocalServiceError, match="CANCEL_PENDING"):
        await control.reload(request, "b")
    await control.close()


@pytest.mark.asyncio
async def test_full_tombstone_table_still_allows_retry_of_pending_cancel(pair):
    manager, control = pair
    request = str(uuid4())
    manager.request = request
    control.tombstones = {str(uuid4()): None for _ in range(511)}
    control.tombstones[request] = None
    assert (await control.cancel_request(request))["state"] == "cancelled"
    with pytest.raises(LocalServiceError, match="RELOAD_LIMIT"):
        await control.cancel_request(str(uuid4()))
    await control.close()


@pytest.mark.asyncio
async def test_unknown_cancel_must_match_request_identity(pair):
    manager, control = pair
    manager.response = manager.document()
    with pytest.raises(LocalServiceError, match="PROTOCOL_ERROR"):
        await control.cancel_request(str(uuid4()))
    await control.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["rollback", "finalize", "cancel_request"])
async def test_close_owns_accepted_public_mutations_even_if_waiter_is_cancelled(pair, action):
    manager, control = pair
    request = str(uuid4())
    value = await control.reload(request, "b")
    manager.changed, manager.state = True, "stabilizing"
    # Record the publication so close need not cancel this completed reload.
    await control.status(value["operation_id"])
    entered, release = asyncio.Event(), asyncio.Event()
    original = manager._request
    async def gated(*args):
        entered.set()
        await release.wait()
        return await original(*args)
    manager._request = gated
    coroutine = (control.cancel_request(request) if action == "cancel_request"
                 else control.action(value["operation_id"], action))
    waiter = asyncio.create_task(coroutine)
    await entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    close = asyncio.create_task(control.close())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await close
    assert not control.tasks


@pytest.mark.asyncio
async def test_lost_post_schedules_request_id_compensation_and_close_observes_it(pair):
    manager, control = pair
    manager.response = RuntimeError("lost POST response")
    with pytest.raises(RuntimeError):
        await control.reload(str(uuid4()), "b")
    manager.response = None
    await control.close()
    assert any(call[1] == "DELETE" for call in manager.calls)
    assert not control.tasks


@pytest.mark.asyncio
async def test_close_waits_for_late_post_and_is_idempotent(pair):
    manager, control = pair
    manager.release.clear()
    task = asyncio.create_task(control.reload(str(uuid4()), "b"))
    await manager.entered.wait()
    close = asyncio.create_task(control.close())
    await asyncio.sleep(0)
    assert not close.done()
    manager.release.set()
    await asyncio.gather(task, close, control.close())
    with pytest.raises(LocalServiceError, match="MANAGER_CLOSED"):
        await control.candidates()
    assert not control.tasks


@pytest.mark.asyncio
async def test_catalog_discovery_filters_paths(pair):
    manager, control = pair
    manager.response = {"enabled": True, "candidates": [{"candidate_id": "b", "path": "private", "token": "secret"}]}
    assert await control.candidates() == {"enabled": True, "candidates": [{"candidate_id": "b"}]}
    await control.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, {}, {"candidates": [None]}, {"candidates": [{}] * 129}])
async def test_bad_catalog_is_protocol_error(pair, value):
    manager, control = pair
    # None is a legitimate fake default, so use a dedicated request stub.
    async def request(*args):
        return value
    manager._request = request
    with pytest.raises(LocalServiceError, match="PROTOCOL_ERROR"):
        await control.candidates()
    await control.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"state": None}, {"active_changed": 1}, {"retirement_pending": None},
                                   {"operation_id": "bad"}, {"request_id": str(uuid4())}])
async def test_malformed_operation_fails_closed(pair, change):
    manager, control = pair
    request = str(uuid4())
    manager.request = request
    manager.response = {**manager.document(), **change}
    with pytest.raises(LocalServiceError, match="PROTOCOL_ERROR"):
        await control.reload(request, "b")
    manager.response = None
    await control.close()


@pytest.mark.asyncio
async def test_operation_identity_cannot_change_between_polls(pair):
    manager, control = pair
    value = await control.reload(str(uuid4()), "b")
    manager.response = {**manager.document(), "operation_id": str(uuid4())}
    with pytest.raises(LocalServiceError, match="PROTOCOL_ERROR"):
        await control.status(value["operation_id"])
    manager.response = None
    await control.close()


@pytest.mark.asyncio
async def test_invalid_inputs_and_closed_manager(pair):
    manager, control = pair
    for request, candidate in [("bad", "b"), (str(uuid4()), "../b"), (str(uuid4()), "")]:
        with pytest.raises(LocalServiceError, match="INVALID_REQUEST"):
            await control.reload(request, candidate)
    with pytest.raises(LocalServiceError, match="INVALID_REQUEST"):
        await control.cancel_request("bad")
    with pytest.raises(LocalServiceError, match="INVALID_REQUEST"):
        await control.status("bad")
    with pytest.raises(LocalServiceError, match="OPERATION_NOT_FOUND"):
        await control.status(str(uuid4()))
    with pytest.raises(LocalServiceError, match="INVALID_REQUEST"):
        await control.action(str(uuid4()), "restart")
    manager._closing = True
    with pytest.raises(LocalServiceError, match="MANAGER_CLOSED"):
        await control.reload(str(uuid4()), "b")
    await control.close()


def test_control_is_owned_by_existing_manager(monkeypatch, pair):
    manager, _ = pair
    monkeypatch.setattr(reload.local_service, "get_manager", lambda: manager)
    assert reload.get_control() is reload.get_control()
