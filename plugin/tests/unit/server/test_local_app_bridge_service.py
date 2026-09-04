from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from plugin.server.local_app_bridge import (
    DispatchEnvelope,
    LocalAppBridgeError,
    LocalAppBridgeService,
    LocalAppPolicy,
    SessionIdentity,
)
from plugin.server.local_app_bridge.contracts import LaunchMaterial
from plugin.server.local_app_bridge.dispatcher import TrustedLocalAppDispatcher
from plugin.server.local_app_bridge.launch import (
    launch_local_app,
    write_launch_material,
)
from plugin.server.local_app_bridge.sessions import LocalAppSessionStore

pytestmark = pytest.mark.plugin_unit


async def _service_and_session(
    handler: Any,
) -> tuple[LocalAppBridgeService, str, SessionIdentity]:
    service = LocalAppBridgeService(operation_timeout=0.2)
    service.register_app(
        LocalAppPolicy(
            app_id="demo.app",
            allowed_operations={"demo:read": frozenset({"echo"})},
        )
    )
    service.register_operation(
        app_id="demo.app",
        scope="demo:read",
        operation="echo",
        handler=handler,
    )
    material = await service.issue_launch_material(
        bridge_origin="http://127.0.0.1:49123",
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    paired = await service.pair(
        launch_code=material.launch_code,
        app_id=material.app_id,
        client_id=material.client_id,
    )
    return service, paired.access_token, paired.identity


@pytest.mark.asyncio
async def test_trusted_dispatch_is_allowlisted_and_receives_bound_identity() -> None:
    seen: list[object] = []

    async def echo(context: object, payload: object) -> object:
        seen.append(context)
        return {"echo": payload}

    service, token, identity = await _service_and_session(echo)
    envelope = DispatchEnvelope(
        identity=identity,
        scope="demo:read",
        operation="echo",
        payload={"value": 3},
    )
    assert await service.dispatch(envelope, access_token=token) == {
        "echo": {"value": 3}
    }
    assert getattr(seen[0], "identity") == identity

    forbidden_envelope = DispatchEnvelope(
        identity=identity,
        scope="demo:read",
        operation="not-registered",
        payload={},
    )
    with pytest.raises(LocalAppBridgeError) as denied:
        await service.dispatch(forbidden_envelope, access_token=token)
    assert denied.value.code == "operation_forbidden"
    await service.close()


@pytest.mark.asyncio
async def test_session_close_during_handler_fails_closed_at_identity_fence() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(_context: object, _payload: object) -> object:
        started.set()
        await release.wait()
        return {"late": True}

    service, token, identity = await _service_and_session(delayed)
    task = asyncio.create_task(
        service.dispatch(
            DispatchEnvelope(identity, "demo:read", "echo", {}),
            access_token=token,
        )
    )
    await started.wait()
    await service.close_session(identity=identity, access_token=token)
    release.set()
    with pytest.raises(LocalAppBridgeError) as stale:
        await task
    assert stale.value.code == "operation_outcome_unconfirmed"
    assert "reconciliation" in stale.value.message
    await service.close()


@pytest.mark.asyncio
async def test_token_rotation_during_handler_requires_idempotent_reconciliation() -> (
    None
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(_context: object, _payload: object) -> object:
        started.set()
        await release.wait()
        return {"possibly_committed": True}

    service, token, identity = await _service_and_session(delayed)
    task = asyncio.create_task(
        service.dispatch(
            DispatchEnvelope(
                identity, "demo:read", "echo", {"request_id": "request-1"}
            ),
            access_token=token,
        )
    )
    await started.wait()
    await service.rotate(identity=identity, access_token=token)
    release.set()
    with pytest.raises(LocalAppBridgeError) as uncertain:
        await task
    assert uncertain.value.code == "operation_outcome_unconfirmed"
    await service.close()


@pytest.mark.asyncio
async def test_dispatch_timeout_and_invalid_response_are_safe() -> None:
    async def slow(_context: object, _payload: object) -> object:
        await asyncio.sleep(1)
        return {}

    service, token, identity = await _service_and_session(slow)
    with pytest.raises(LocalAppBridgeError) as timed_out:
        await service.dispatch(
            DispatchEnvelope(identity, "demo:read", "echo", {}),
            access_token=token,
        )
    assert timed_out.value.code == "operation_outcome_unconfirmed"
    await service.close()

    async def invalid(_context: object, _payload: object) -> object:
        return object()

    service, token, identity = await _service_and_session(invalid)
    with pytest.raises(LocalAppBridgeError) as invalid_response:
        await service.dispatch(
            DispatchEnvelope(identity, "demo:read", "echo", {}),
            access_token=token,
        )
    assert invalid_response.value.code == "invalid_operation_response"
    assert invalid_response.value.status_code == 502
    await service.close()


def test_dispatch_envelope_rejects_extra_fields_and_oversized_payload() -> None:
    valid = {
        "protocol_version": 1,
        "app_id": "demo.app",
        "client_id": "client-1",
        "session_id": "session-1",
        "scope": "demo:read",
        "operation": "echo",
        "payload": {},
    }
    with pytest.raises(LocalAppBridgeError) as extra:
        DispatchEnvelope.from_mapping({**valid, "forged_identity": "yes"})
    assert extra.value.code == "invalid_fields"
    with pytest.raises(LocalAppBridgeError) as too_large:
        DispatchEnvelope.from_mapping({**valid, "payload": {"data": "x" * (64 * 1024)}})
    assert too_large.value.status_code == 413


@pytest.mark.asyncio
async def test_host_authority_is_bound_to_one_dispatcher_instance() -> None:
    first = TrustedLocalAppDispatcher()
    second = TrustedLocalAppDispatcher()
    policy = LocalAppPolicy("demo.app", {"demo:read": frozenset({"echo"})})
    second.register_app(policy)

    async def echo(_context: object, payload: object) -> object:
        return payload

    second.register_operation(
        app_id="demo.app", scope="demo:read", operation="echo", handler=echo
    )
    with pytest.raises(LocalAppBridgeError) as untrusted:
        await second.dispatch(
            authority=first._issue_authority(),
            identity=SessionIdentity("demo.app", "client-1", "session-1"),
            scope="demo:read",
            operation="echo",
            payload={},
        )
    assert untrusted.value.code == "untrusted_dispatch"


@pytest.mark.asyncio
async def test_rotation_during_rate_limit_wait_is_rechecked_before_handler() -> None:
    limiter_started = asyncio.Event()
    release_limiter = asyncio.Event()
    handler_called = False

    class _BlockingLimiter:
        async def check_pair(self, **_kwargs: object) -> None:
            return None

        async def check_session(self, _session_id: str) -> None:
            limiter_started.set()
            await release_limiter.wait()

        async def close(self) -> None:
            return None

    sessions = LocalAppSessionStore()
    service = LocalAppBridgeService(
        sessions=sessions,
        rate_limiter=_BlockingLimiter(),  # type: ignore[arg-type]
    )
    service.register_app(LocalAppPolicy("demo.app", {"demo:read": frozenset({"echo"})}))

    async def echo(_context: object, _payload: object) -> object:
        nonlocal handler_called
        handler_called = True
        return {}

    service.register_operation(
        app_id="demo.app", scope="demo:read", operation="echo", handler=echo
    )
    material = await service.issue_launch_material(
        bridge_origin="http://127.0.0.1:49123",
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    paired = await service.pair(
        launch_code=material.launch_code,
        app_id="demo.app",
        client_id="client-1",
    )
    task = asyncio.create_task(
        service.dispatch(
            DispatchEnvelope(paired.identity, "demo:read", "echo", {}),
            access_token=paired.access_token,
        )
    )
    await limiter_started.wait()
    await sessions.rotate_access_token(
        identity=paired.identity,
        access_token=paired.access_token,
    )
    release_limiter.set()
    with pytest.raises(LocalAppBridgeError) as stale:
        await task
    assert stale.value.code == "session_changed"
    assert handler_called is False
    await service.close()


class _Writer:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, value: bytes) -> None:
        self.data += value

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


@pytest.mark.asyncio
async def test_launch_material_uses_stdin_and_secret_is_not_repr_or_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material = LaunchMaterial(
        "http://127.0.0.1:49123", "demo.app", "client-1", "secret-code"
    )
    writer = _Writer()
    assert "secret-code" not in repr(material)
    await write_launch_material(writer, material)  # type: ignore[arg-type]
    assert b"secret-code" in writer.data and writer.closed

    @dataclass
    class _Process:
        stdin: _Writer
        killed: bool = False

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            return 0

    process = _Process(_Writer())
    captured_args: tuple[object, ...] = ()

    async def create(*args: object, **kwargs: object) -> _Process:
        nonlocal captured_args
        captured_args = args
        assert kwargs["stdin"] is asyncio.subprocess.PIPE
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    returned = await launch_local_app("demo.exe", args=("--safe",), material=material)
    assert returned is process
    assert captured_args == ("demo.exe", "--safe")
    assert all("secret-code" not in str(argument) for argument in captured_args)
    assert b"secret-code" in process.stdin.data

    with pytest.raises(ValueError):
        await launch_local_app(
            "demo.exe", args=("--code=secret-code",), material=material
        )
