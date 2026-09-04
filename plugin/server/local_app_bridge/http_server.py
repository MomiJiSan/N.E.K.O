from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from plugin.server.local_app_bridge.contracts import (
    MAX_BODY_BYTES,
    PROTOCOL_VERSION,
    DispatchEnvelope,
    require_exact_fields,
)
from plugin.server.local_app_bridge.errors import (
    LocalAppBridgeError,
    bad_request,
    forbidden,
    unauthorized,
)
from plugin.server.local_app_bridge.service import (
    LocalAppBridgeService,
    identity_from_mapping,
)

MAX_HEADER_BYTES = 8192
READ_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _HttpRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: object


class LocalAppBridgeHttpServer:
    """Minimal single-request HTTP/1.1 server bound exclusively to IPv4 loopback."""

    def __init__(self, service: LocalAppBridgeService) -> None:
        self._service = service
        self._server: asyncio.Server | None = None
        self._port: int | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("bridge server is not running")
        return self._port

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> str:
        if self._closed:
            raise RuntimeError("bridge server is closed")
        if self._server is not None:
            return self.origin
        self._server = await asyncio.start_server(
            self._accept_connection,
            host="127.0.0.1",
            port=0,
            limit=MAX_HEADER_BYTES + 1,
        )
        sockets = self._server.sockets or ()
        if len(sockets) != 1:
            await self.close()
            raise RuntimeError("bridge must own exactly one loopback listener")
        host, port = sockets[0].getsockname()[:2]
        if host != "127.0.0.1" or not isinstance(port, int) or port <= 0:
            await self.close()
            raise RuntimeError("bridge listener escaped the loopback boundary")
        self._port = port
        return self.origin

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        current = asyncio.current_task()
        tasks = [task for task in self._connections if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connections.clear()
        await self._service.close()
        self._port = None

    async def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            peer = writer.get_extra_info("peername")
            if not isinstance(peer, tuple) or not peer or peer[0] != "127.0.0.1":
                raise forbidden("loopback_required")
            request = await self._read_request(reader)
            self._validate_headers(request.headers)
            status, payload = await self._route(request)
        except asyncio.CancelledError:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
            if task is not None:
                self._connections.discard(task)
            raise
        except LocalAppBridgeError as exc:
            status = exc.status_code
            payload = {"ok": False, "error": {"code": exc.code, "message": exc.message}}
            if exc.retry_after is not None:
                payload["error"]["retry_after"] = round(exc.retry_after, 3)
        except Exception:
            status = 500
            payload = {
                "ok": False,
                "error": {"code": "internal_error", "message": "Internal error"},
            }
        try:
            with contextlib.suppress(ConnectionError, OSError):
                await self._write_response(writer, status, payload)
        finally:
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await writer.wait_closed()
            if task is not None:
                self._connections.discard(task)

    async def _read_request(self, reader: asyncio.StreamReader) -> _HttpRequest:
        try:
            header_block = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except asyncio.LimitOverrunError as exc:
            raise LocalAppBridgeError(
                "headers_too_large", 431, "Request headers are too large"
            ) from exc
        except (asyncio.IncompleteReadError, TimeoutError) as exc:
            raise bad_request("invalid_http", "Invalid HTTP request") from exc
        if len(header_block) > MAX_HEADER_BYTES:
            raise LocalAppBridgeError(
                "headers_too_large", 431, "Request headers are too large"
            )
        try:
            lines = header_block[:-4].decode("ascii").split("\r\n")
            method, path, version = lines[0].split(" ", 2)
        except (UnicodeDecodeError, ValueError) as exc:
            raise bad_request("invalid_http", "Invalid HTTP request") from exc
        if (
            version != "HTTP/1.1"
            or method not in {"GET", "POST"}
            or not path.startswith("/")
        ):
            raise bad_request("invalid_http", "Invalid HTTP request")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                raise bad_request("invalid_http", "Invalid HTTP request")
            name, raw_value = line.split(":", 1)
            name = name.strip().lower()
            if not name or name in headers:
                raise bad_request("invalid_http", "Invalid HTTP request")
            headers[name] = raw_value.strip()
        if "transfer-encoding" in headers:
            raise bad_request(
                "transfer_encoding_forbidden", "Transfer encoding is not supported"
            )
        raw_length = headers.get("content-length", "0")
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise bad_request(
                "invalid_content_length", "Invalid content length"
            ) from exc
        if content_length < 0:
            raise bad_request("invalid_content_length", "Invalid content length")
        if content_length > MAX_BODY_BYTES:
            raise LocalAppBridgeError(
                "body_too_large", 413, "Request body is too large"
            )
        try:
            body_bytes = await asyncio.wait_for(
                reader.readexactly(content_length),
                timeout=READ_TIMEOUT_SECONDS,
            )
        except (asyncio.IncompleteReadError, TimeoutError) as exc:
            raise bad_request("incomplete_body", "Incomplete request body") from exc
        if content_length == 0:
            body: object = {}
        else:
            if (
                headers.get("content-type", "").split(";", 1)[0].strip().lower()
                != "application/json"
            ):
                raise LocalAppBridgeError(
                    "json_required", 415, "JSON content is required"
                )
            try:
                body = json.loads(body_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise bad_request("invalid_json", "Invalid JSON") from exc
        return _HttpRequest(method=method, path=path, headers=headers, body=body)

    def _validate_headers(self, headers: Mapping[str, str]) -> None:
        if headers.get("host") != f"127.0.0.1:{self.port}":
            raise forbidden("invalid_host")
        if "origin" in headers:
            raise forbidden("browser_origin_forbidden")

    async def _route(self, request: _HttpRequest) -> tuple[int, Mapping[str, object]]:
        if request.method == "GET" and request.path == "/v1/health":
            require_exact_fields(request.body, required=frozenset())
            return 200, {"ok": True, "protocol_version": PROTOCOL_VERSION}
        if request.method != "POST":
            raise LocalAppBridgeError("route_not_found", 404, "Route not found")
        if request.path == "/v1/pair":
            body = require_exact_fields(
                request.body,
                required=frozenset(
                    {"protocol_version", "app_id", "client_id", "launch_code"}
                ),
            )
            self._require_protocol(body)
            result = await self._service.pair(
                launch_code=self._text(body["launch_code"]),
                app_id=self._text(body["app_id"]),
                client_id=self._text(body["client_id"]),
            )
            return 200, {"ok": True, **result.to_dict()}
        if request.path == "/v1/dispatch":
            envelope = DispatchEnvelope.from_mapping(request.body)
            result = await self._service.dispatch(
                envelope,
                access_token=self._bearer_token(request.headers),
            )
            return 200, {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "result": result,
            }
        if request.path in {"/v1/sessions/rotate", "/v1/sessions/close"}:
            body = require_exact_fields(
                request.body,
                required=frozenset(
                    {"protocol_version", "app_id", "client_id", "session_id"}
                ),
            )
            self._require_protocol(body)
            identity = identity_from_mapping(body)
            token = self._bearer_token(request.headers)
            if request.path.endswith("/rotate"):
                result = await self._service.rotate(
                    identity=identity, access_token=token
                )
                return 200, {"ok": True, **result.to_dict()}
            await self._service.close_session(identity=identity, access_token=token)
            return 200, {"ok": True, "protocol_version": PROTOCOL_VERSION}
        raise LocalAppBridgeError("route_not_found", 404, "Route not found")

    @staticmethod
    def _require_protocol(body: Mapping[str, Any]) -> None:
        if body["protocol_version"] != PROTOCOL_VERSION:
            raise bad_request("unsupported_protocol", "Unsupported protocol version")

    @staticmethod
    def _text(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise bad_request()
        return value

    @staticmethod
    def _bearer_token(headers: Mapping[str, str]) -> str:
        authorization = headers.get("authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if separator != " " or scheme.lower() != "bearer" or not token or " " in token:
            raise unauthorized("invalid_authorization")
        return token

    @staticmethod
    async def _write_response(
        writer: asyncio.StreamWriter,
        status: int,
        payload: Mapping[str, object],
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        phrase = HTTPStatus(status).phrase
        writer.write(
            (
                f"HTTP/1.1 {status} {phrase}\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Cache-Control: no-store\r\n"
                "Connection: close\r\n"
                "X-Content-Type-Options: nosniff\r\n"
                "\r\n"
            ).encode("ascii")
            + body
        )
        try:
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
