from __future__ import annotations

import asyncio
from collections.abc import Mapping

from plugin.server.local_app_bridge.contracts import (
    MAX_BODY_BYTES,
    DispatchEnvelope,
    LaunchMaterial,
    LocalAppPolicy,
    PairResult,
    SessionIdentity,
    json_size,
    require_identifier,
)
from plugin.server.local_app_bridge.dispatcher import (
    TrustedHandler,
    TrustedLocalAppDispatcher,
)
from plugin.server.local_app_bridge.errors import LocalAppBridgeError, forbidden
from plugin.server.local_app_bridge.rate_limit import LocalAppRateLimiter
from plugin.server.local_app_bridge.sessions import LocalAppSessionStore


class LocalAppBridgeService:
    """Provider-neutral orchestration for pairing and trusted local dispatch."""

    def __init__(
        self,
        *,
        sessions: LocalAppSessionStore | None = None,
        dispatcher: TrustedLocalAppDispatcher | None = None,
        operation_timeout: float = 30.0,
        rate_limiter: LocalAppRateLimiter | None = None,
    ) -> None:
        if operation_timeout <= 0:
            raise ValueError("operation_timeout must be positive")
        self._sessions = sessions or LocalAppSessionStore()
        self._dispatcher = dispatcher or TrustedLocalAppDispatcher()
        self._authority = self._dispatcher._issue_authority()
        self._rate_limiter = rate_limiter or LocalAppRateLimiter()
        self._operation_timeout = operation_timeout
        self._closed = False

    def register_app(self, policy: LocalAppPolicy) -> None:
        self._require_open()
        self._dispatcher.register_app(policy)

    def register_operation(
        self,
        *,
        app_id: str,
        scope: str,
        operation: str,
        handler: TrustedHandler,
    ) -> None:
        self._require_open()
        self._dispatcher.register_operation(
            app_id=app_id,
            scope=scope,
            operation=operation,
            handler=handler,
        )

    async def issue_launch_material(
        self,
        *,
        bridge_origin: str,
        app_id: str,
        client_id: str,
        requested_scopes: frozenset[str],
    ) -> LaunchMaterial:
        self._require_open()
        require_identifier(app_id, "app_id")
        require_identifier(client_id, "client_id")
        allowed_scopes = self._dispatcher.allowed_scopes(app_id)
        if not requested_scopes or not requested_scopes.issubset(allowed_scopes):
            raise forbidden("scope_forbidden")
        launch_code = await self._sessions.issue_launch_code(
            app_id=app_id,
            client_id=client_id,
            scopes=requested_scopes,
        )
        return LaunchMaterial(
            bridge_origin=bridge_origin,
            app_id=app_id,
            client_id=client_id,
            launch_code=launch_code,
        )

    async def pair(
        self, *, launch_code: str, app_id: str, client_id: str
    ) -> PairResult:
        self._require_open()
        await self._rate_limiter.check_pair(app_id=app_id, client_id=client_id)
        return await self._sessions.exchange_launch_code(
            launch_code=launch_code,
            app_id=app_id,
            client_id=client_id,
        )

    async def dispatch(
        self, envelope: DispatchEnvelope, *, access_token: str
    ) -> object:
        self._require_open()
        if json_size(envelope.payload) > MAX_BODY_BYTES:
            raise LocalAppBridgeError("payload_too_large", 413, "Payload is too large")
        lease = await self._sessions.authorize(
            identity=envelope.identity,
            scope=envelope.scope,
            access_token=access_token,
        )
        await self._rate_limiter.check_session(envelope.identity.session_id)
        # The limiter is an await point: verify ownership again before any
        # trusted handler is allowed to create externally visible state.
        await self._sessions.confirm_lease(lease)
        try:
            async with asyncio.timeout(self._operation_timeout):
                result = await self._dispatcher.dispatch(
                    authority=self._authority,
                    identity=lease.identity,
                    scope=envelope.scope,
                    operation=envelope.operation,
                    payload=envelope.payload,
                )
        except TimeoutError as exc:
            raise LocalAppBridgeError(
                "operation_outcome_unconfirmed",
                409,
                "Operation outcome requires idempotent reconciliation",
            ) from exc
        try:
            await self._sessions.confirm_lease(lease)
        except LocalAppBridgeError as exc:
            raise LocalAppBridgeError(
                "operation_outcome_unconfirmed",
                409,
                "Operation outcome requires idempotent reconciliation",
            ) from exc
        try:
            result_size = json_size(result)
        except LocalAppBridgeError as exc:
            raise LocalAppBridgeError(
                "invalid_operation_response",
                502,
                "Operation returned an invalid response",
            ) from exc
        if result_size > MAX_BODY_BYTES:
            raise LocalAppBridgeError(
                "response_too_large", 502, "Operation response is too large"
            )
        return result

    async def rotate(
        self, *, identity: SessionIdentity, access_token: str
    ) -> PairResult:
        self._require_open()
        await self._sessions.authenticate_control(
            identity=identity,
            access_token=access_token,
        )
        await self._rate_limiter.check_session(identity.session_id)
        return await self._sessions.rotate_access_token(
            identity=identity,
            access_token=access_token,
        )

    async def close_session(
        self, *, identity: SessionIdentity, access_token: str
    ) -> None:
        self._require_open()
        await self._sessions.authenticate_control(
            identity=identity,
            access_token=access_token,
        )
        await self._rate_limiter.check_session(identity.session_id)
        await self._sessions.close_session(identity=identity, access_token=access_token)

    async def cleanup(self) -> None:
        if not self._closed:
            await self._sessions.cleanup()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._sessions.close()
        await self._rate_limiter.close()
        self._dispatcher.close()

    def _require_open(self) -> None:
        if self._closed:
            raise LocalAppBridgeError("bridge_closed", 503, "Bridge is unavailable")


def identity_from_mapping(value: Mapping[str, object]) -> SessionIdentity:
    return SessionIdentity(
        app_id=require_identifier(value.get("app_id"), "app_id"),
        client_id=require_identifier(value.get("client_id"), "client_id"),
        session_id=require_identifier(value.get("session_id"), "session_id"),
    )
