from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from main_routers.config_router import local_asr


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(local_asr, "_validate_local_mutation_request", lambda _: None)
    monkeypatch.setattr(local_asr, "_core_allows_independent_asr", AsyncMock(return_value=True))
    app = FastAPI()
    app.include_router(local_asr.router)
    with TestClient(app) as client:
        yield client


def payload():
    return {"request_id": str(uuid4()), "enabled": True, "provider": "local_streaming"}


def test_prepare_returns_operation_not_credentials_or_launch_paths(api, monkeypatch):
    value = {"state": "loading", "ready": False, "operation_id": str(uuid4()),
             "lease_id": str(uuid4()), "token": "secret", "url": "ws://local", "python": "private"}
    prepare = AsyncMock(return_value=value)
    monkeypatch.setattr(local_asr.local_service, "prepare", prepare)
    body = payload()
    response = api.post("/api/config/local-asr/prepare", json=body)
    assert response.status_code == 200
    assert response.json() == {"success": True, **{k: v for k, v in value.items() if k in local_asr._PUBLIC_FIELDS}}
    prepare.assert_awaited_once_with(body["request_id"])


@pytest.mark.parametrize("change", [{"enabled": False}, {"enabled": 1}, {"enabled": "true"},
                                  {"provider": "auto"}, {"model_dir": "untrusted"}])
def test_prepare_requires_explicit_supported_intent_and_rejects_paths(api, monkeypatch, change):
    prepare = AsyncMock()
    monkeypatch.setattr(local_asr.local_service, "prepare", prepare)
    response = api.post("/api/config/local-asr/prepare", json={**payload(), **change})
    assert response.status_code == 422
    prepare.assert_not_awaited()


def test_native_only_core_cannot_prepare(api, monkeypatch):
    monkeypatch.setattr(local_asr, "_core_allows_independent_asr", AsyncMock(return_value=False))
    prepare = AsyncMock()
    monkeypatch.setattr(local_asr.local_service, "prepare", prepare)
    response = api.post("/api/config/local-asr/prepare", json=payload())
    assert response.status_code == 409
    prepare.assert_not_awaited()


def test_local_mutation_guard_precedes_prepare(api, monkeypatch):
    monkeypatch.setattr(local_asr, "_validate_local_mutation_request", lambda _: JSONResponse({}, status_code=403))
    prepare = AsyncMock()
    monkeypatch.setattr(local_asr.local_service, "prepare", prepare)
    assert api.post("/api/config/local-asr/prepare", json=payload()).status_code == 403
    prepare.assert_not_awaited()


def test_status_and_cancel_are_per_opaque_owner(api, monkeypatch):
    identity = str(uuid4())
    status = AsyncMock(return_value={"state": "ready", "ready": True, "token": "private"})
    cancel = AsyncMock()
    monkeypatch.setattr(local_asr.local_service, "status", status)
    monkeypatch.setattr(local_asr.local_service, "cancel_request", cancel)
    assert api.get("/api/config/local-asr/operations/" + identity).json() == {"success": True, "state": "ready", "ready": True}
    assert api.delete("/api/config/local-asr/prepare/" + identity).json() == {"success": True, "released": True}
    status.assert_awaited_once_with(identity)
    cancel.assert_awaited_once_with(identity)


def test_failure_response_does_not_expose_exception_or_invalid_code(api, monkeypatch):
    error = RuntimeError("secret path and token")
    error.code = "secret/path"
    monkeypatch.setattr(local_asr.local_service, "prepare", AsyncMock(side_effect=error))
    response = api.post("/api/config/local-asr/prepare", json=payload())
    assert response.status_code == 409
    assert response.json() == {"success": False, "code": "ASR_LOCAL_PREPARE_FAILED"}


@pytest.mark.asyncio
@pytest.mark.parametrize("core,allowed", [("qwen", True), ("free", False), ("unknown", False)])
async def test_prepare_gate_uses_effective_core_configuration(monkeypatch, core, allowed):
    from types import SimpleNamespace
    manager = SimpleNamespace(aget_model_api_config=AsyncMock(return_value={"api_type": core}))
    monkeypatch.setattr(local_asr, "get_config_manager", lambda: manager)
    assert await local_asr._core_allows_independent_asr() is allowed


def test_router_shutdown_runs_with_an_application_lifespan(monkeypatch):
    from contextlib import asynccontextmanager
    events = []
    @asynccontextmanager
    async def lifespan(app):
        events.append("start")
        yield
        events.append("stop")
    close = AsyncMock()
    monkeypatch.setattr(local_asr.local_service, "close", close)
    app = FastAPI(lifespan=lifespan)
    app.include_router(local_asr.router)
    with TestClient(app):
        assert events == ["start"]
    assert events == ["start", "stop"]
    assert close.await_count >= 1
