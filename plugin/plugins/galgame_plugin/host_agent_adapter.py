from __future__ import annotations

import json
from typing import Any

import httpx

try:
    from config import TOOL_SERVER_PORT as _TOOL_SERVER_PORT
except Exception:
    _TOOL_SERVER_PORT = 48915

try:
    from utils.internal_http_client import (
        get_internal_http_client as _get_internal_http_client,
    )
except Exception:
    _get_internal_http_client = None


class HostAgentError(RuntimeError):
    pass


class HostAgentAdapter:
    def __init__(self, logger, *, tool_server_port: int | None = None) -> None:
        self._logger = logger
        self._tool_server_port = int(tool_server_port or _TOOL_SERVER_PORT)
        self._fallback_client: httpx.AsyncClient | None = None
        self._prefer_fallback_client = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._tool_server_port}"

    def _get_client(self) -> httpx.AsyncClient:
        if not self._prefer_fallback_client and _get_internal_http_client is not None:
            return _get_internal_http_client()
        if self._fallback_client is None or self._fallback_client.is_closed:
            self._fallback_client = httpx.AsyncClient(
                timeout=5.0,
                proxy=None,
                trust_env=False,
                transport=httpx.AsyncHTTPTransport(verify=False, retries=0),
            )
        return self._fallback_client

    async def shutdown(self) -> None:
        if self._fallback_client is not None and not self._fallback_client.is_closed:
            await self._fallback_client.aclose()
        self._fallback_client = None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        attempts = 2 if not self._prefer_fallback_client and _get_internal_http_client is not None else 1
        for attempt in range(attempts):
            client = self._get_client()
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    json=payload,
                    timeout=timeout,
                )
                break
            except Exception as exc:
                last_exc = exc
                if (
                    attempt == 0
                    and not self._prefer_fallback_client
                    and _get_internal_http_client is not None
                    and self._is_closed_loop_error(exc)
                ):
                    # The shared internal AsyncClient can survive a plugin restart while
                    # still being bound to the previous closed loop. Fall back to a
                    # plugin-local client and retry once on the current loop.
                    self._prefer_fallback_client = True
                    self._logger.warning(
                        "HostAgentAdapter switching to fallback AsyncClient after closed-loop error on {} {}: {}",
                        method,
                        path,
                        exc,
                    )
                    continue
                raise HostAgentError(f"{method} {path} failed: {exc}") from exc
        else:
            assert last_exc is not None
            raise HostAgentError(f"{method} {path} failed: {last_exc}") from last_exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise HostAgentError(
                f"{method} {path} returned non-json payload: HTTP {response.status_code}"
            ) from exc

        if not response.is_success:
            raise HostAgentError(
                f"{method} {path} responded {response.status_code}: "
                f"{data.get('detail') or data.get('error') or data}"
            )
        if not isinstance(data, dict):
            raise HostAgentError(
                f"{method} {path} returned invalid payload type: {type(data)!r}"
            )
        return data

    @staticmethod
    def _is_closed_loop_error(exc: Exception) -> bool:
        return "Event loop is closed" in str(exc or "")

    async def get_computer_use_availability(self, *, timeout: float = 1.5) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            "/computer_use/availability",
            timeout=timeout,
        )

    async def run_computer_use_instruction(
        self,
        instruction: str,
        *,
        lanlan_name: str = "",
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        payload = {"instruction": instruction.strip()}
        if lanlan_name:
            payload["lanlan_name"] = lanlan_name
        return await self._request_json(
            "POST",
            "/computer_use/run",
            payload=payload,
            timeout=timeout,
        )

    async def get_task(self, task_id: str, *, timeout: float = 2.0) -> dict[str, Any]:
        return await self._request_json(
            "GET",
            f"/tasks/{task_id}",
            timeout=timeout,
        )

    async def cancel_task(self, task_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            f"/tasks/{task_id}/cancel",
            timeout=timeout,
        )
