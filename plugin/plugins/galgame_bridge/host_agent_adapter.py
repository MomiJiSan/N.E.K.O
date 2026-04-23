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

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._tool_server_port}"

    def _get_client(self) -> httpx.AsyncClient:
        if _get_internal_http_client is not None:
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
        client = self._get_client()
        url = f"{self.base_url}{path}"
        try:
            response = await client.request(
                method=method,
                url=url,
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:
            raise HostAgentError(f"{method} {path} failed: {exc}") from exc

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
