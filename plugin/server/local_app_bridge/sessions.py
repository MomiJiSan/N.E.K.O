from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from plugin.server.local_app_bridge.contracts import (
    ACCESS_TOKEN_TTL_SECONDS,
    LAUNCH_CODE_TTL_SECONDS,
    MAX_ACTIVE_SESSIONS,
    MAX_PENDING_LAUNCH_CODES,
    SESSION_ABSOLUTE_TTL_SECONDS,
    SESSION_IDLE_TTL_SECONDS,
    PairResult,
    SessionIdentity,
)
from plugin.server.local_app_bridge.errors import (
    LocalAppBridgeError,
    forbidden,
    unauthorized,
)


def _secret_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _LaunchGrant:
    code_hash: str
    app_id: str
    client_id: str
    scopes: frozenset[str]
    expires_at: float


@dataclass(slots=True)
class _Session:
    identity: SessionIdentity
    scopes: frozenset[str]
    token_hash: str
    token_expires_at: float
    created_at: float
    last_activity_at: float
    generation: int = 1


@dataclass(frozen=True, slots=True)
class SessionLease:
    identity: SessionIdentity
    scope: str
    token_hash: str = field(repr=False)
    generation: int = 1


class LocalAppSessionStore:
    """Concurrency-safe in-memory launch grants and bounded sessions."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._launch_grants: dict[str, _LaunchGrant] = {}
        self._spent_launch_codes: dict[str, float] = {}
        self._sessions: dict[str, _Session] = {}
        self._closed = False

    async def issue_launch_code(
        self,
        *,
        app_id: str,
        client_id: str,
        scopes: Iterable[str],
    ) -> str:
        code = secrets.token_urlsafe(32)
        code_hash = _secret_hash(code)
        now = self._clock()
        async with self._lock:
            self._require_open()
            self._cleanup_locked(now)
            if len(self._launch_grants) >= MAX_PENDING_LAUNCH_CODES:
                raise LocalAppBridgeError(
                    "launch_capacity_reached", 503, "Bridge is at capacity"
                )
            self._launch_grants[code_hash] = _LaunchGrant(
                code_hash=code_hash,
                app_id=app_id,
                client_id=client_id,
                scopes=frozenset(scopes),
                expires_at=now + LAUNCH_CODE_TTL_SECONDS,
            )
        return code

    async def exchange_launch_code(
        self,
        *,
        launch_code: str,
        app_id: str,
        client_id: str,
    ) -> PairResult:
        code_hash = _secret_hash(launch_code)
        now = self._clock()
        async with self._lock:
            self._require_open()
            self._cleanup_spent_locked(now)
            if code_hash in self._spent_launch_codes:
                raise LocalAppBridgeError(
                    "launch_code_replayed", 409, "Launch code was already used"
                )
            grant = self._launch_grants.get(code_hash)
            if grant is None:
                self._cleanup_locked(now)
                raise unauthorized("invalid_launch_code")
            if now >= grant.expires_at:
                self._launch_grants.pop(code_hash, None)
                self._mark_spent_locked(code_hash, now)
                raise unauthorized("launch_code_expired")
            if not hmac.compare_digest(grant.app_id, app_id) or not hmac.compare_digest(
                grant.client_id, client_id
            ):
                raise forbidden("launch_identity_mismatch")
            self._cleanup_sessions_locked(now)
            if len(self._sessions) >= MAX_ACTIVE_SESSIONS:
                raise LocalAppBridgeError(
                    "session_capacity_reached", 503, "Bridge is at capacity"
                )
            self._launch_grants.pop(code_hash, None)
            self._mark_spent_locked(code_hash, now)
            session_id = f"las_{secrets.token_urlsafe(24)}"
            token = secrets.token_urlsafe(48)
            identity = SessionIdentity(
                app_id=app_id, client_id=client_id, session_id=session_id
            )
            self._sessions[session_id] = _Session(
                identity=identity,
                scopes=grant.scopes,
                token_hash=_secret_hash(token),
                token_expires_at=now + ACCESS_TOKEN_TTL_SECONDS,
                created_at=now,
                last_activity_at=now,
            )
            return PairResult(
                identity=identity, access_token=token, granted_scopes=grant.scopes
            )

    async def authorize(
        self,
        *,
        identity: SessionIdentity,
        scope: str,
        access_token: str,
    ) -> SessionLease:
        now = self._clock()
        token_hash = _secret_hash(access_token)
        async with self._lock:
            self._require_open()
            session = self._require_live_session_locked(identity.session_id, now)
            self._verify_identity(session, identity)
            if not hmac.compare_digest(session.token_hash, token_hash):
                raise unauthorized("invalid_access_token")
            if now >= session.token_expires_at:
                raise unauthorized("access_token_expired")
            if scope not in session.scopes:
                raise forbidden("scope_forbidden")
            session.last_activity_at = now
            return SessionLease(
                identity=session.identity,
                scope=scope,
                token_hash=token_hash,
                generation=session.generation,
            )

    async def authenticate_control(
        self,
        *,
        identity: SessionIdentity,
        access_token: str,
    ) -> SessionLease:
        now = self._clock()
        token_hash = _secret_hash(access_token)
        async with self._lock:
            self._require_open()
            session = self._require_live_session_locked(identity.session_id, now)
            self._verify_identity(session, identity)
            if not hmac.compare_digest(session.token_hash, token_hash):
                raise unauthorized("invalid_access_token")
            if now >= session.token_expires_at:
                raise unauthorized("access_token_expired")
            session.last_activity_at = now
            return SessionLease(
                identity=session.identity,
                scope="",
                token_hash=token_hash,
                generation=session.generation,
            )

    async def confirm_lease(self, lease: SessionLease) -> None:
        now = self._clock()
        async with self._lock:
            self._require_open()
            session = self._require_live_session_locked(lease.identity.session_id, now)
            self._verify_identity(session, lease.identity)
            if session.generation != lease.generation or not hmac.compare_digest(
                session.token_hash, lease.token_hash
            ):
                raise LocalAppBridgeError(
                    "session_changed", 409, "Session changed during operation"
                )

    async def rotate_access_token(
        self,
        *,
        identity: SessionIdentity,
        access_token: str,
    ) -> PairResult:
        now = self._clock()
        supplied_hash = _secret_hash(access_token)
        async with self._lock:
            self._require_open()
            session = self._require_live_session_locked(identity.session_id, now)
            self._verify_identity(session, identity)
            if not hmac.compare_digest(session.token_hash, supplied_hash):
                raise unauthorized("invalid_access_token")
            if now >= session.token_expires_at:
                raise unauthorized("access_token_expired")
            token = secrets.token_urlsafe(48)
            session.token_hash = _secret_hash(token)
            session.token_expires_at = now + ACCESS_TOKEN_TTL_SECONDS
            session.last_activity_at = now
            session.generation += 1
            return PairResult(
                identity=session.identity,
                access_token=token,
                granted_scopes=session.scopes,
            )

    async def close_session(
        self,
        *,
        identity: SessionIdentity,
        access_token: str,
    ) -> None:
        now = self._clock()
        supplied_hash = _secret_hash(access_token)
        async with self._lock:
            self._require_open()
            session = self._require_live_session_locked(identity.session_id, now)
            self._verify_identity(session, identity)
            if not hmac.compare_digest(session.token_hash, supplied_hash):
                raise unauthorized("invalid_access_token")
            self._sessions.pop(identity.session_id, None)

    async def cleanup(self) -> None:
        async with self._lock:
            self._cleanup_locked(self._clock())

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._launch_grants.clear()
            self._spent_launch_codes.clear()
            self._sessions.clear()

    def _require_open(self) -> None:
        if self._closed:
            raise LocalAppBridgeError("bridge_closed", 503, "Bridge is unavailable")

    def _require_live_session_locked(self, session_id: str, now: float) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise unauthorized("invalid_session")
        if now - session.last_activity_at >= SESSION_IDLE_TTL_SECONDS:
            self._sessions.pop(session_id, None)
            raise unauthorized("session_idle_expired")
        if now - session.created_at >= SESSION_ABSOLUTE_TTL_SECONDS:
            self._sessions.pop(session_id, None)
            raise unauthorized("session_absolute_expired")
        return session

    @staticmethod
    def _verify_identity(session: _Session, identity: SessionIdentity) -> None:
        expected = session.identity
        if not (
            hmac.compare_digest(expected.app_id, identity.app_id)
            and hmac.compare_digest(expected.client_id, identity.client_id)
            and hmac.compare_digest(expected.session_id, identity.session_id)
        ):
            raise forbidden("session_identity_mismatch")

    def _cleanup_locked(self, now: float) -> None:
        for code_hash, grant in tuple(self._launch_grants.items()):
            if now >= grant.expires_at:
                self._launch_grants.pop(code_hash, None)
        self._cleanup_spent_locked(now)
        self._cleanup_sessions_locked(now)

    def _cleanup_spent_locked(self, now: float) -> None:
        for code_hash, expires_at in tuple(self._spent_launch_codes.items()):
            if now >= expires_at:
                self._spent_launch_codes.pop(code_hash, None)

    def _cleanup_sessions_locked(self, now: float) -> None:
        for session_id, session in tuple(self._sessions.items()):
            if (
                now - session.last_activity_at >= SESSION_IDLE_TTL_SECONDS
                or now - session.created_at >= SESSION_ABSOLUTE_TTL_SECONDS
            ):
                self._sessions.pop(session_id, None)

    def _mark_spent_locked(self, code_hash: str, now: float) -> None:
        if len(self._spent_launch_codes) >= MAX_PENDING_LAUNCH_CODES:
            oldest = min(
                self._spent_launch_codes, key=self._spent_launch_codes.__getitem__
            )
            self._spent_launch_codes.pop(oldest, None)
        self._spent_launch_codes[code_hash] = now + LAUNCH_CODE_TTL_SECONDS
