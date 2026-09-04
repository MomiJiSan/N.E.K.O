from __future__ import annotations

import pytest

from plugin.server.application.plugins import trusted_local_app_dispatch as module
from plugin.server.local_app_bridge.errors import LocalAppBridgeError


pytestmark = pytest.mark.plugin_unit


class _Health:
    alive = True


class _Host:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls: list[dict[str, object]] = []

    def health_check(self) -> _Health:
        return _Health()

    async def trigger_trusted_local_app(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return {"host": self.marker}


@pytest.mark.asyncio
async def test_each_invocation_resolves_latest_host_after_plugin_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Host("first")
    second = _Host("second")
    snapshots = iter(({"plugin": first}, {"plugin": second}))
    monkeypatch.setattr(
        module.state,
        "get_plugin_hosts_snapshot_cached",
        lambda timeout: next(snapshots),
    )
    dispatch = module.TrustedLocalAppPluginDispatch()
    kwargs = {
        "plugin_id": "plugin",
        "plugin_operation": "operation",
        "context": {
            "app_id": "app",
            "client_id": "client",
            "session_id": "session",
            "scope": "scope",
            "operation": "external-operation",
        },
        "payload": {"request_id": "request-1"},
        "timeout": 2.0,
    }
    assert await dispatch.invoke(**kwargs) == {"host": "first"}
    assert await dispatch.invoke(**kwargs) == {"host": "second"}
    assert len(first.calls) == len(second.calls) == 1


@pytest.mark.asyncio
async def test_plugin_timeout_is_reported_as_unconfirmed_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TimeoutHost(_Host):
        async def trigger_trusted_local_app(self, **kwargs: object) -> object:
            raise TimeoutError

    monkeypatch.setattr(
        module.state,
        "get_plugin_hosts_snapshot_cached",
        lambda timeout: {"plugin": _TimeoutHost("timeout")},
    )
    with pytest.raises(LocalAppBridgeError) as uncertain:
        await module.TrustedLocalAppPluginDispatch().invoke(
            plugin_id="plugin",
            plugin_operation="operation",
            context={
                "app_id": "app",
                "client_id": "client",
                "session_id": "session",
                "scope": "scope",
                "operation": "external-operation",
            },
            payload={"request_id": "request-1"},
            timeout=2.0,
        )
    assert uncertain.value.code == "operation_outcome_unconfirmed"
