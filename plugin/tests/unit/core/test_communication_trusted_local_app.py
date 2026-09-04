from __future__ import annotations

import asyncio

import pytest

from plugin.core.communication import PluginCommunicationResourceManager


pytestmark = pytest.mark.plugin_unit


class _Transport:
    def __init__(self) -> None:
        self.commands: list[dict[str, object]] = []
        self.sent = asyncio.Event()

    async def send_command(self, message: dict[str, object]) -> None:
        self.commands.append(message)
        self.sent.set()


@pytest.mark.asyncio
async def test_trusted_command_has_dedicated_type_and_routes_response() -> None:
    transport = _Transport()
    manager = PluginCommunicationResourceManager(
        plugin_id="fixture",
        transport=transport,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        manager.trigger_trusted_local_app(
            context={
                "app_id": "app",
                "client_id": "client",
                "session_id": "session",
                "scope": "scope",
                "operation": "operation",
            },
            operation="target-operation",
            payload={"request_id": "request-1"},
            timeout=1.0,
        )
    )
    await transport.sent.wait()
    command = transport.commands[0]
    assert command["type"] == "TRUSTED_LOCAL_APP_INVOKE"
    assert command["operation"] == "target-operation"
    request_id = str(command["req_id"])
    manager._dispatch_result(
        {"req_id": request_id, "success": True, "data": {"accepted": True}}
    )
    assert await task == {"accepted": True}
    assert manager.get_pending_requests_count() == 0


@pytest.mark.asyncio
async def test_cancel_sends_dedicated_cancel_and_drops_pending_future() -> None:
    transport = _Transport()
    manager = PluginCommunicationResourceManager(
        plugin_id="fixture",
        transport=transport,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        manager.trigger_trusted_local_app(
            context={
                "app_id": "app",
                "client_id": "client",
                "session_id": "session",
                "scope": "scope",
                "operation": "operation",
            },
            operation="target-operation",
            payload={},
            timeout=30.0,
        )
    )
    await transport.sent.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [command["type"] for command in transport.commands] == [
        "TRUSTED_LOCAL_APP_INVOKE",
        "CANCEL_TRUSTED_LOCAL_APP",
    ]
    assert transport.commands[1]["req_id"] == transport.commands[0]["req_id"]
    assert manager.get_pending_requests_count() == 0
    # A response arriving after cancellation is isolated by request id and
    # cannot complete another invocation.
    manager._dispatch_result(
        {
            "req_id": transport.commands[0]["req_id"],
            "success": True,
            "data": {"late": True},
        }
    )
    assert manager.get_pending_requests_count() == 0
