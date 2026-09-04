from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugin.server.routes import plugins as module


pytestmark = pytest.mark.plugin_unit


@pytest.mark.asyncio
async def test_start_plugin_endpoint_ensures_messaging_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def _ensure_messaging() -> None:
        calls.append("ensure")

    async def _start_plugin(plugin_id: str, *, persist_user_intent: bool = False) -> dict[str, object]:
        calls.append(f"start:{plugin_id}:{persist_user_intent}")
        return {"success": True, "plugin_id": plugin_id}

    monkeypatch.setattr(module, "ensure_plugin_messaging_started", _ensure_messaging, raising=False)
    monkeypatch.setattr(module.lifecycle_service, "start_plugin", _start_plugin)

    result = await module.start_plugin_endpoint("sample_plugin", _="test")

    assert result == {"success": True, "plugin_id": "sample_plugin"}
    assert calls == ["ensure", "start:sample_plugin:True"]


def _route_client() -> TestClient:
    app = FastAPI()
    app.include_router(module.router)
    return TestClient(app)


def test_local_app_launch_route_accepts_only_app_id_from_trusted_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []

    async def _launch(app_id: str):
        launched.append(app_id)
        return type("Result", (), {"app_id": app_id})()

    monkeypatch.setattr(module, "AUTOSTART_CSRF_TOKEN", "test-csrf-token")
    monkeypatch.setattr(module, "launch_registered_local_app", _launch)

    with _route_client() as client:
        token = client.get(
            "/local-app/ui-token",
            headers={"Origin": "http://testserver"},
        )
        response = client.post(
            "/local-app/launch",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": token.json()["token"],
            },
            json={"app_id": "knowledge_dungeon"},
        )

    assert token.status_code == 200
    assert token.headers["Cache-Control"] == "no-store"
    assert response.status_code == 200
    assert response.json() == {"success": True, "app_id": "knowledge_dungeon"}
    assert launched == ["knowledge_dungeon"]


def test_local_app_launch_route_rejects_origin_csrf_and_extra_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _must_not_launch(_app_id: str):
        raise AssertionError("rejected request must not launch")

    monkeypatch.setattr(module, "AUTOSTART_CSRF_TOKEN", "test-csrf-token")
    monkeypatch.setattr(module, "launch_registered_local_app", _must_not_launch)

    with _route_client() as client:
        missing_origin = client.post(
            "/local-app/launch",
            headers={"X-CSRF-Token": "test-csrf-token"},
            json={"app_id": "knowledge_dungeon"},
        )
        bad_csrf = client.post(
            "/local-app/launch",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": "wrong",
            },
            json={"app_id": "knowledge_dungeon"},
        )
        extra_field = client.post(
            "/local-app/launch",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": "test-csrf-token",
            },
            json={
                "app_id": "knowledge_dungeon",
                "executable": "C:/attacker.exe",
            },
        )

    assert missing_origin.status_code == 403
    assert bad_csrf.status_code == 403
    assert extra_field.status_code == 422


def test_local_app_launch_route_returns_safe_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unavailable(_app_id: str):
        raise OSError("C:/Users/private/secret-path/app.exe")

    monkeypatch.setattr(module, "AUTOSTART_CSRF_TOKEN", "test-csrf-token")
    monkeypatch.setattr(module, "launch_registered_local_app", _unavailable)

    with _route_client() as client:
        response = client.post(
            "/local-app/launch",
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": "test-csrf-token",
            },
            json={"app_id": "knowledge_dungeon"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "local_app_unavailable"}}
    assert "secret-path" not in response.text
