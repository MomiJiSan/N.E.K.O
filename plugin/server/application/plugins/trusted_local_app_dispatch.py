"""Host-only dispatch from an authenticated local app into a plugin process."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from plugin.core.state import state
from plugin.server.local_app_bridge.contracts import require_identifier
from plugin.server.local_app_bridge.errors import LocalAppBridgeError


@runtime_checkable
class _HostHealth(Protocol):
    alive: bool


@runtime_checkable
class TrustedLocalAppHost(Protocol):
    def health_check(self) -> _HostHealth: ...

    async def trigger_trusted_local_app(
        self,
        *,
        context: Mapping[str, str],
        operation: str,
        payload: Mapping[str, object],
        timeout: float,
    ) -> object: ...


class TrustedLocalAppPluginDispatch:
    """Resolve the current host for every call and invoke its private IPC path."""

    async def invoke(
        self,
        *,
        plugin_id: str,
        plugin_operation: str,
        context: Mapping[str, str],
        payload: Mapping[str, object],
        timeout: float,
    ) -> object:
        plugin_id = require_identifier(plugin_id, "plugin_id")
        plugin_operation = require_identifier(plugin_operation, "plugin_operation")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")

        host = await asyncio.to_thread(self._resolve_host, plugin_id)
        health = await asyncio.to_thread(host.health_check)
        if not health.alive:
            raise LocalAppBridgeError(
                "plugin_not_ready", 503, "Operation target is unavailable"
            )
        try:
            return await host.trigger_trusted_local_app(
                context=dict(context),
                operation=plugin_operation,
                payload=dict(payload),
                timeout=float(timeout),
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise LocalAppBridgeError(
                "operation_outcome_unconfirmed",
                409,
                "Operation outcome requires idempotent reconciliation",
            ) from exc
        except LocalAppBridgeError:
            raise
        except Exception as exc:
            raise LocalAppBridgeError(
                "plugin_operation_failed", 502, "Operation target failed"
            ) from exc

    @staticmethod
    def _resolve_host(plugin_id: str) -> TrustedLocalAppHost:
        hosts = state.get_plugin_hosts_snapshot_cached(timeout=1.0)
        host = hosts.get(plugin_id)
        if not isinstance(host, TrustedLocalAppHost):
            raise LocalAppBridgeError(
                "plugin_not_found", 503, "Operation target is unavailable"
            )
        return host
