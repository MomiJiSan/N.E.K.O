from __future__ import annotations

from dataclasses import dataclass

import pytest

from plugin.server.local_app_bridge.contracts import SessionIdentity
from plugin.server.local_app_bridge.errors import LocalAppBridgeError
from plugin.server.local_app_bridge.sessions import LocalAppSessionStore

pytestmark = pytest.mark.plugin_unit


@dataclass
class _Clock:
    now: float = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _paired(clock: _Clock) -> tuple[LocalAppSessionStore, str, SessionIdentity]:
    store = LocalAppSessionStore(clock=clock)
    code = await store.issue_launch_code(
        app_id="demo.app",
        client_id="client-1",
        scopes={"demo:read"},
    )
    result = await store.exchange_launch_code(
        launch_code=code,
        app_id="demo.app",
        client_id="client-1",
    )
    return store, result.access_token, result.identity


@pytest.mark.asyncio
async def test_launch_code_is_single_use_and_bound_to_identity() -> None:
    clock = _Clock()
    store = LocalAppSessionStore(clock=clock)
    code = await store.issue_launch_code(
        app_id="demo.app",
        client_id="client-1",
        scopes={"demo:read"},
    )
    with pytest.raises(LocalAppBridgeError, match="permitted") as mismatch:
        await store.exchange_launch_code(
            launch_code=code,
            app_id="demo.app",
            client_id="forged-client",
        )
    assert mismatch.value.code == "launch_identity_mismatch"

    paired = await store.exchange_launch_code(
        launch_code=code,
        app_id="demo.app",
        client_id="client-1",
    )
    assert paired.granted_scopes == frozenset({"demo:read"})
    with pytest.raises(LocalAppBridgeError) as replay:
        await store.exchange_launch_code(
            launch_code=code,
            app_id="demo.app",
            client_id="client-1",
        )
    assert replay.value.code == "launch_code_replayed"


@pytest.mark.asyncio
async def test_launch_code_expires_after_sixty_seconds() -> None:
    clock = _Clock()
    store = LocalAppSessionStore(clock=clock)
    code = await store.issue_launch_code(
        app_id="demo.app", client_id="client-1", scopes={"demo:read"}
    )
    clock.advance(60)
    with pytest.raises(LocalAppBridgeError) as expired:
        await store.exchange_launch_code(
            launch_code=code, app_id="demo.app", client_id="client-1"
        )
    assert expired.value.code == "launch_code_expired"


@pytest.mark.asyncio
async def test_token_scope_and_all_identity_fields_are_verified() -> None:
    clock = _Clock()
    store, token, identity = await _paired(clock)
    lease = await store.authorize(
        identity=identity, scope="demo:read", access_token=token
    )
    await store.confirm_lease(lease)

    with pytest.raises(LocalAppBridgeError) as wrong_token:
        await store.authorize(
            identity=identity, scope="demo:read", access_token="wrong"
        )
    assert wrong_token.value.code == "invalid_access_token"
    with pytest.raises(LocalAppBridgeError) as wrong_scope:
        await store.authorize(identity=identity, scope="demo:write", access_token=token)
    assert wrong_scope.value.code == "scope_forbidden"
    forged = SessionIdentity(
        app_id=identity.app_id, client_id="other", session_id=identity.session_id
    )
    with pytest.raises(LocalAppBridgeError) as wrong_identity:
        await store.authorize(identity=forged, scope="demo:read", access_token=token)
    assert wrong_identity.value.code == "session_identity_mismatch"


@pytest.mark.asyncio
async def test_access_token_rotates_and_invalidates_the_previous_generation() -> None:
    clock = _Clock()
    store, token, identity = await _paired(clock)
    old_lease = await store.authorize(
        identity=identity, scope="demo:read", access_token=token
    )
    clock.advance(899)
    rotated = await store.rotate_access_token(identity=identity, access_token=token)
    assert rotated.access_token != token
    with pytest.raises(LocalAppBridgeError) as old_token:
        await store.authorize(identity=identity, scope="demo:read", access_token=token)
    assert old_token.value.code == "invalid_access_token"
    with pytest.raises(LocalAppBridgeError) as stale_lease:
        await store.confirm_lease(old_lease)
    assert stale_lease.value.code == "session_changed"
    await store.authorize(
        identity=identity, scope="demo:read", access_token=rotated.access_token
    )


@pytest.mark.asyncio
async def test_access_token_expires_at_fifteen_minutes() -> None:
    clock = _Clock()
    store, token, identity = await _paired(clock)
    clock.advance(900)
    with pytest.raises(LocalAppBridgeError) as expired:
        await store.authorize(
            identity=identity,
            scope="demo:read",
            access_token=token,
        )
    assert expired.value.code == "access_token_expired"


@pytest.mark.asyncio
async def test_token_idle_and_absolute_session_expiry_are_enforced() -> None:
    idle_clock = _Clock()
    idle_store, idle_token, idle_identity = await _paired(idle_clock)
    idle_clock.advance(1800)
    with pytest.raises(LocalAppBridgeError) as idle:
        await idle_store.authorize(
            identity=idle_identity, scope="demo:read", access_token=idle_token
        )
    assert idle.value.code == "session_idle_expired"

    absolute_clock = _Clock()
    absolute_store, current_token, absolute_identity = await _paired(absolute_clock)
    for _ in range(35):
        absolute_clock.advance(800)
        rotated = await absolute_store.rotate_access_token(
            identity=absolute_identity,
            access_token=current_token,
        )
        current_token = rotated.access_token
    absolute_clock.advance(800)
    with pytest.raises(LocalAppBridgeError) as absolute:
        await absolute_store.authorize(
            identity=absolute_identity,
            scope="demo:read",
            access_token=current_token,
        )
    assert absolute.value.code == "session_absolute_expired"


@pytest.mark.asyncio
async def test_close_clears_all_credentials_and_is_idempotent() -> None:
    clock = _Clock()
    store, token, identity = await _paired(clock)
    await store.close_session(identity=identity, access_token=token)
    with pytest.raises(LocalAppBridgeError) as closed_session:
        await store.authorize(identity=identity, scope="demo:read", access_token=token)
    assert closed_session.value.code == "invalid_session"
    await store.close()
    await store.close()
    with pytest.raises(LocalAppBridgeError) as closed_store:
        await store.issue_launch_code(
            app_id="demo.app", client_id="client-1", scopes={"demo:read"}
        )
    assert closed_store.value.code == "bridge_closed"


@pytest.mark.asyncio
async def test_session_capacity_rejects_without_consuming_launch_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plugin.server.local_app_bridge import sessions as sessions_module

    monkeypatch.setattr(sessions_module, "MAX_ACTIVE_SESSIONS", 2)
    clock = _Clock()
    store = LocalAppSessionStore(clock=clock)
    active = []
    for index in range(2):
        code = await store.issue_launch_code(
            app_id="demo.app",
            client_id=f"client-{index}",
            scopes={"demo:read"},
        )
        active.append(
            await store.exchange_launch_code(
                launch_code=code,
                app_id="demo.app",
                client_id=f"client-{index}",
            )
        )
    waiting_code = await store.issue_launch_code(
        app_id="demo.app", client_id="waiting", scopes={"demo:read"}
    )
    with pytest.raises(LocalAppBridgeError) as full:
        await store.exchange_launch_code(
            launch_code=waiting_code,
            app_id="demo.app",
            client_id="waiting",
        )
    assert full.value.code == "session_capacity_reached"
    await store.close_session(
        identity=active[0].identity,
        access_token=active[0].access_token,
    )
    paired = await store.exchange_launch_code(
        launch_code=waiting_code,
        app_id="demo.app",
        client_id="waiting",
    )
    assert paired.identity.client_id == "waiting"


@pytest.mark.asyncio
async def test_pending_launch_capacity_is_bounded_and_expiry_frees_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plugin.server.local_app_bridge import sessions as sessions_module

    monkeypatch.setattr(sessions_module, "MAX_PENDING_LAUNCH_CODES", 2)
    clock = _Clock()
    store = LocalAppSessionStore(clock=clock)
    await store.issue_launch_code(
        app_id="demo.app", client_id="one", scopes={"demo:read"}
    )
    await store.issue_launch_code(
        app_id="demo.app", client_id="two", scopes={"demo:read"}
    )
    with pytest.raises(LocalAppBridgeError) as full:
        await store.issue_launch_code(
            app_id="demo.app", client_id="three", scopes={"demo:read"}
        )
    assert full.value.code == "launch_capacity_reached"
    clock.advance(60)
    code = await store.issue_launch_code(
        app_id="demo.app", client_id="three", scopes={"demo:read"}
    )
    assert code
