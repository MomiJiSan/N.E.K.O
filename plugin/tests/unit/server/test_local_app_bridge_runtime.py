from __future__ import annotations

import asyncio
import contextlib

import pytest

from plugin.server.local_app_bridge import (
    DispatchEnvelope,
    LocalAppBridgeRuntime,
    LocalAppPluginRegistration,
    LocalAppPluginTarget,
    LocalAppPolicy,
)
from plugin.server.local_app_bridge import runtime as runtime_module


pytestmark = pytest.mark.plugin_unit


def _registration() -> LocalAppPluginRegistration:
    return LocalAppPluginRegistration(
        policy=LocalAppPolicy(
            app_id="knowledge_dungeon",
            allowed_operations={
                "study_companion:dungeon": frozenset(
                    {
                        "knowledge_dungeon.bootstrap",
                        "knowledge_dungeon.perform_action",
                    }
                )
            },
        ),
        targets=(
            LocalAppPluginTarget(
                scope="study_companion:dungeon",
                operation="knowledge_dungeon.bootstrap",
                plugin_id="study_companion",
                plugin_operation="knowledge_dungeon.bootstrap",
            ),
            LocalAppPluginTarget(
                scope="study_companion:dungeon",
                operation="knowledge_dungeon.perform_action",
                plugin_id="study_companion",
                plugin_operation="knowledge_dungeon.perform_action",
            ),
        ),
    )


def test_registration_must_exactly_cover_policy() -> None:
    registration = _registration()
    with pytest.raises(ValueError, match="exactly match"):
        LocalAppPluginRegistration(
            policy=registration.policy,
            targets=registration.targets[:1],
        )


@pytest.mark.asyncio
async def test_runtime_maps_verified_session_to_fixed_plugin_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _Dispatch:
        async def invoke(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return {"accepted": True}

    monkeypatch.setattr(runtime_module, "TrustedLocalAppPluginDispatch", _Dispatch)
    runtime = LocalAppBridgeRuntime()
    runtime.register_plugin_app(_registration())

    first_origin = await runtime.start()
    assert first_origin.startswith("http://127.0.0.1:")
    assert await runtime.start() == first_origin
    material = await runtime.issue_launch_material(
        app_id="knowledge_dungeon",
        client_id="electron-1",
        requested_scopes=frozenset({"study_companion:dungeon"}),
    )
    service = runtime._service
    assert service is not None
    paired = await service.pair(
        launch_code=material.launch_code,
        app_id=material.app_id,
        client_id=material.client_id,
    )
    result = await service.dispatch(
        DispatchEnvelope(
            identity=paired.identity,
            scope="study_companion:dungeon",
            operation="knowledge_dungeon.bootstrap",
            payload={"request_id": "request-1"},
        ),
        access_token=paired.access_token,
    )
    assert result == {"accepted": True}
    assert calls == [
        {
            "plugin_id": "study_companion",
            "plugin_operation": "knowledge_dungeon.bootstrap",
            "context": {
                "app_id": "knowledge_dungeon",
                "client_id": "electron-1",
                "session_id": paired.identity.session_id,
                "scope": "study_companion:dungeon",
                "operation": "knowledge_dungeon.bootstrap",
            },
            "payload": {"request_id": "request-1"},
            "timeout": 30.0,
        }
    ]

    await runtime.close()
    assert runtime.is_running is False
    # Registration is configuration, so a lifecycle restart restores it while
    # tokens and sessions from the old service stay invalidated.
    second_origin = await runtime.start()
    assert second_origin.startswith("http://127.0.0.1:")
    await runtime.close()


@pytest.mark.asyncio
async def test_server_lifecycle_stops_listener_before_plugin_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plugin.server import lifecycle

    calls: list[str] = []

    class _State:
        plugin_hosts: dict[str, object] = {}
        plugins: dict[str, object] = {}
        event_handlers: dict[str, object] = {}

        @staticmethod
        def acquire_plugin_hosts_write_lock():
            return contextlib.nullcontext()

        @staticmethod
        def acquire_plugins_write_lock():
            return contextlib.nullcontext()

        @staticmethod
        def acquire_event_handlers_write_lock():
            return contextlib.nullcontext()

        @staticmethod
        def close_plugin_resources() -> None:
            return None

    async def _stop_local() -> None:
        calls.append("local_app_bridge")

    async def _stop_hosts() -> bool:
        calls.append("plugin_hosts")
        return False

    async def _noop_async(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(lifecycle, "state", _State())
    monkeypatch.setattr(lifecycle, "emit_lifecycle_event", lambda _event: None)
    monkeypatch.setattr(lifecycle, "stop_local_app_bridge", _stop_local)
    monkeypatch.setattr(lifecycle, "stop_bridge", lambda: None)
    monkeypatch.setattr(lifecycle, "stop_proactive_bridge", lambda: None)
    monkeypatch.setattr(lifecycle.metrics_collector, "stop", _noop_async)
    monkeypatch.setattr(
        lifecycle.status_manager, "shutdown_status_consumer", _noop_async
    )
    monkeypatch.setattr(lifecycle.bus_subscription_manager, "stop", _noop_async)
    monkeypatch.setattr(lifecycle.plugin_router, "stop", _noop_async)
    service = lifecycle.ServerLifecycleService()
    monkeypatch.setattr(service, "_shutdown_hosts", _stop_hosts)

    result = await service._shutdown_internal()

    assert result.had_errors is False
    assert calls == ["local_app_bridge", "plugin_hosts"]


@pytest.mark.asyncio
async def test_registration_is_frozen_once_listener_bind_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_started = asyncio.Event()
    finish_bind = asyncio.Event()

    class _Server:
        def __init__(self, service) -> None:
            self.service = service
            self.origin = "http://127.0.0.1:49123"

        async def start(self) -> str:
            bind_started.set()
            await finish_bind.wait()
            return self.origin

        async def close(self) -> None:
            await self.service.close()

    monkeypatch.setattr(runtime_module, "LocalAppBridgeHttpServer", _Server)
    runtime = LocalAppBridgeRuntime()
    start_task = asyncio.create_task(runtime.start())
    await bind_started.wait()
    with pytest.raises(RuntimeError, match="frozen"):
        runtime.register_plugin_app(_registration())
    finish_bind.set()
    assert await start_task == "http://127.0.0.1:49123"
    await runtime.close()
