from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from plugin.server.local_app_bridge.contracts import LocalAppPolicy, SessionIdentity
from plugin.server.local_app_bridge.errors import LocalAppBridgeError, forbidden

TrustedHandler = Callable[
    ["TrustedDispatchContext", Mapping[str, object]], Awaitable[object]
]


@dataclass(frozen=True, slots=True)
class TrustedDispatchContext:
    identity: SessionIdentity
    scope: str
    operation: str


class HostDispatchAuthority:
    """Opaque capability owned by the bridge service, never serialized."""

    __slots__ = ("_proof",)

    def __init__(self, proof: object) -> None:
        self._proof = proof


class TrustedLocalAppDispatcher:
    """Host-memory-only operation registry, separate from plugin entries/events."""

    def __init__(self) -> None:
        self._policies: dict[str, LocalAppPolicy] = {}
        self._handlers: dict[tuple[str, str, str], TrustedHandler] = {}
        self._proof = object()
        self._closed = False

    def _issue_authority(self) -> HostDispatchAuthority:
        return HostDispatchAuthority(self._proof)

    def register_app(self, policy: LocalAppPolicy) -> None:
        if self._closed:
            raise RuntimeError("dispatcher is closed")
        if policy.app_id in self._policies:
            raise ValueError(f"app already registered: {policy.app_id}")
        self._policies[policy.app_id] = policy

    def register_operation(
        self,
        *,
        app_id: str,
        scope: str,
        operation: str,
        handler: TrustedHandler,
    ) -> None:
        policy = self._policies.get(app_id)
        if policy is None:
            raise ValueError(f"app is not registered: {app_id}")
        if operation not in policy.allowed_operations.get(scope, frozenset()):
            raise ValueError("operation is outside the app policy allowlist")
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError("trusted handler must be async")
        key = (app_id, scope, operation)
        if key in self._handlers:
            raise ValueError("trusted operation is already registered")
        self._handlers[key] = handler

    def allowed_scopes(self, app_id: str) -> frozenset[str]:
        policy = self._policies.get(app_id)
        if policy is None:
            raise forbidden("app_not_registered")
        return frozenset(policy.allowed_operations)

    async def dispatch(
        self,
        *,
        authority: HostDispatchAuthority,
        identity: SessionIdentity,
        scope: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> object:
        if authority._proof is not self._proof:
            raise forbidden("untrusted_dispatch")
        if self._closed:
            raise LocalAppBridgeError("bridge_closed", 503, "Bridge is unavailable")
        policy = self._policies.get(identity.app_id)
        if policy is None or operation not in policy.allowed_operations.get(
            scope, frozenset()
        ):
            raise forbidden("operation_forbidden")
        handler = self._handlers.get((identity.app_id, scope, operation))
        if handler is None:
            raise LocalAppBridgeError(
                "operation_unavailable", 503, "Operation is unavailable"
            )
        try:
            return await handler(
                TrustedDispatchContext(
                    identity=identity, scope=scope, operation=operation
                ),
                dict(payload),
            )
        except asyncio.CancelledError:
            raise
        except LocalAppBridgeError:
            raise
        except Exception as exc:
            raise LocalAppBridgeError(
                "operation_failed", 500, "Operation failed"
            ) from exc

    def close(self) -> None:
        self._closed = True
        self._handlers.clear()
        self._policies.clear()
