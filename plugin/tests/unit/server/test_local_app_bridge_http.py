from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from plugin.server.local_app_bridge import (
    LocalAppBridgeHttpServer,
    LocalAppBridgeService,
    LocalAppPolicy,
)

pytestmark = pytest.mark.plugin_unit


async def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    encoded = json.dumps(body or {}, separators=(",", ":")).encode()
    request_headers = {
        "Host": f"127.0.0.1:{port}",
        "Content-Type": "application/json",
        "Content-Length": str(len(encoded)),
        **(headers or {}),
    }
    lines = [
        f"{method} {path} HTTP/1.1",
        *(f"{key}: {value}" for key, value in request_headers.items()),
        "",
        "",
    ]
    writer.write("\r\n".join(lines).encode("ascii") + encoded)
    await writer.drain()
    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("bridge closed before a response was committed")
    status = int(status_line.split()[1])
    response_headers: dict[bytes, bytes] = {}
    while True:
        line = await reader.readline()
        if line == b"\r\n":
            break
        name, value = line.split(b":", 1)
        response_headers[name.lower()] = value.strip()
    response_body = await reader.readexactly(int(response_headers[b"content-length"]))
    writer.close()
    await writer.wait_closed()
    return status, json.loads(response_body)


async def _running_bridge() -> tuple[LocalAppBridgeService, LocalAppBridgeHttpServer]:
    async def echo(_context: object, payload: object) -> object:
        return {"echo": payload}

    service = LocalAppBridgeService()
    service.register_app(LocalAppPolicy("demo.app", {"demo:read": frozenset({"echo"})}))
    service.register_operation(
        app_id="demo.app", scope="demo:read", operation="echo", handler=echo
    )
    server = LocalAppBridgeHttpServer(service)
    await server.start()
    return service, server


@pytest.mark.asyncio
async def test_http_pair_and_trusted_dispatch_end_to_end() -> None:
    service, server = await _running_bridge()
    assert server.origin.startswith("http://127.0.0.1:")
    material = await service.issue_launch_material(
        bridge_origin=server.origin,
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    status, paired = await _request(
        server.port,
        "POST",
        "/v1/pair",
        body={
            "protocol_version": 1,
            "app_id": "demo.app",
            "client_id": "client-1",
            "launch_code": material.launch_code,
        },
    )
    assert status == 200 and paired["ok"] is True
    assert paired["access_token_expires_in"] == 900

    status, dispatched = await _request(
        server.port,
        "POST",
        "/v1/dispatch",
        body={
            "protocol_version": 1,
            "app_id": "demo.app",
            "client_id": "client-1",
            "session_id": paired["session_id"],
            "scope": "demo:read",
            "operation": "echo",
            "payload": {"value": 7},
        },
        headers={"Authorization": f"Bearer {paired['access_token']}"},
    )
    assert status == 200
    assert dispatched["result"] == {"echo": {"value": 7}}

    identity_body = {
        "protocol_version": 1,
        "app_id": "demo.app",
        "client_id": "client-1",
        "session_id": paired["session_id"],
    }
    status, rotated = await _request(
        server.port,
        "POST",
        "/v1/sessions/rotate",
        body=identity_body,
        headers={"Authorization": f"Bearer {paired['access_token']}"},
    )
    assert status == 200 and rotated["access_token"] != paired["access_token"]
    status, closed = await _request(
        server.port,
        "POST",
        "/v1/sessions/close",
        body=identity_body,
        headers={"Authorization": f"Bearer {rotated['access_token']}"},
    )
    assert status == 200 and closed["ok"] is True
    await server.close()


@pytest.mark.asyncio
async def test_http_rejects_browser_origin_bad_host_extra_fields_and_wrong_token() -> (
    None
):
    service, server = await _running_bridge()
    material = await service.issue_launch_material(
        bridge_origin=server.origin,
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    pair_body = {
        "protocol_version": 1,
        "app_id": "demo.app",
        "client_id": "client-1",
        "launch_code": material.launch_code,
    }
    status, response = await _request(
        server.port,
        "POST",
        "/v1/pair",
        body=pair_body,
        headers={"Origin": "http://127.0.0.1:9999"},
    )
    assert status == 403 and response["error"]["code"] == "browser_origin_forbidden"
    status, response = await _request(
        server.port,
        "POST",
        "/v1/pair",
        body={**pair_body, "extra": True},
    )
    assert status == 400 and response["error"]["code"] == "invalid_fields"
    status, paired = await _request(server.port, "POST", "/v1/pair", body=pair_body)
    assert status == 200

    dispatch_body = {
        "protocol_version": 1,
        "app_id": "demo.app",
        "client_id": "client-1",
        "session_id": paired["session_id"],
        "scope": "demo:read",
        "operation": "echo",
        "payload": {},
    }
    status, response = await _request(
        server.port,
        "POST",
        "/v1/dispatch",
        body=dispatch_body,
        headers={"Authorization": "Bearer wrong"},
    )
    assert status == 401 and response["error"]["code"] == "invalid_access_token"
    status, response = await _request(
        server.port,
        "GET",
        "/v1/health",
        headers={"Host": "localhost:1234"},
    )
    assert status == 403 and response["error"]["code"] == "invalid_host"
    await server.close()


@pytest.mark.asyncio
async def test_http_enforces_body_limit_and_releases_listener() -> None:
    _service, server = await _running_bridge()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        (
            "POST /v1/pair HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.port}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: 65537\r\n\r\n"
        ).encode("ascii")
    )
    await writer.drain()
    status_line = await reader.readline()
    assert status_line.startswith(b"HTTP/1.1 413")
    writer.close()
    await writer.wait_closed()
    port = server.port
    await server.close()
    with pytest.raises((ConnectionRefusedError, OSError)):
        await asyncio.open_connection("127.0.0.1", port)


@pytest.mark.asyncio
async def test_http_rate_limit_returns_retry_after() -> None:
    service, server = await _running_bridge()
    material = await service.issue_launch_material(
        bridge_origin=server.origin,
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    _, paired = await _request(
        server.port,
        "POST",
        "/v1/pair",
        body={
            "protocol_version": 1,
            "app_id": "demo.app",
            "client_id": "client-1",
            "launch_code": material.launch_code,
        },
    )
    body = {
        "protocol_version": 1,
        "app_id": "demo.app",
        "client_id": "client-1",
        "session_id": paired["session_id"],
        "scope": "demo:read",
        "operation": "echo",
        "payload": {},
    }
    for _ in range(30):
        status, _response = await _request(
            server.port,
            "POST",
            "/v1/dispatch",
            body=body,
            headers={"Authorization": f"Bearer {paired['access_token']}"},
        )
        assert status == 200
    status, response = await _request(
        server.port,
        "POST",
        "/v1/dispatch",
        body=body,
        headers={"Authorization": f"Bearer {paired['access_token']}"},
    )
    assert status == 429
    assert response["error"]["code"] == "rate_limited"
    assert response["error"]["retry_after"] > 0
    await server.close()


@pytest.mark.asyncio
async def test_http_rejects_oversized_headers_and_close_cancels_slow_reader() -> None:
    _service, server = await _running_bridge()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    writer.write(
        (
            "GET /v1/health HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.port}\r\n"
            f"X-Fill: {'x' * 9000}\r\n\r\n"
        ).encode("ascii")
    )
    await writer.drain()
    assert (await reader.readline()).startswith(b"HTTP/1.1 431")
    writer.close()
    await writer.wait_closed()

    slow_reader, slow_writer = await asyncio.open_connection("127.0.0.1", server.port)
    slow_writer.write(b"POST /v1/pair HTTP/1.1\r\n")
    await slow_writer.drain()
    await asyncio.wait_for(server.close(), timeout=1)
    assert await asyncio.wait_for(slow_reader.read(), timeout=1) == b""
    slow_writer.close()
    await slow_writer.wait_closed()


@pytest.mark.asyncio
async def test_server_close_cancels_in_flight_dispatch_before_releasing_resources() -> (
    None
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(_context: object, _payload: object) -> object:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    service = LocalAppBridgeService()
    service.register_app(
        LocalAppPolicy("demo.app", {"demo:read": frozenset({"blocked"})})
    )
    service.register_operation(
        app_id="demo.app",
        scope="demo:read",
        operation="blocked",
        handler=blocked,
    )
    server = LocalAppBridgeHttpServer(service)
    await server.start()
    material = await service.issue_launch_material(
        bridge_origin=server.origin,
        app_id="demo.app",
        client_id="client-1",
        requested_scopes=frozenset({"demo:read"}),
    )
    _, paired = await _request(
        server.port,
        "POST",
        "/v1/pair",
        body={
            "protocol_version": 1,
            "app_id": "demo.app",
            "client_id": "client-1",
            "launch_code": material.launch_code,
        },
    )
    request_task = asyncio.create_task(
        _request(
            server.port,
            "POST",
            "/v1/dispatch",
            body={
                "protocol_version": 1,
                "app_id": "demo.app",
                "client_id": "client-1",
                "session_id": paired["session_id"],
                "scope": "demo:read",
                "operation": "blocked",
                "payload": {"request_id": "request-1"},
            },
            headers={"Authorization": f"Bearer {paired['access_token']}"},
        )
    )
    await started.wait()
    await asyncio.wait_for(server.close(), timeout=1)
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    with pytest.raises((asyncio.IncompleteReadError, ConnectionError)):
        await request_task
