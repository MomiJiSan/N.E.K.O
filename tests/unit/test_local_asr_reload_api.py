from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.responses import JSONResponse

from main_routers.config_router import local_asr
from tests.unit import test_local_asr_prepare_api as prepare_tests
from tests.unit.test_local_asr_prepare_api import payload


@pytest.fixture
def api(monkeypatch):
    yield from prepare_tests.api.__wrapped__(monkeypatch)


@pytest.fixture
def control(monkeypatch):
    value = {"operation_id": str(uuid4()), "state": "draining", "active_changed": True,
             "retirement_pending": True, "token": "secret", "path": "private"}
    control = SimpleNamespace(**{name: AsyncMock(return_value=value) for name in
                                ("reload", "status", "action", "cancel_request", "candidates")})
    monkeypatch.setattr(local_asr.reload_control, "get_control", lambda: control)
    return control


def paths():
    identity = str(uuid4())
    return [("get", "/candidates", None), ("post", "/reload", {**payload(), "candidate_id": "version-b"}),
            ("get", "/reloads/" + identity, None), ("delete", "/reloads/" + identity, None),
            ("post", "/reloads/" + identity + "/finalize", {}),
            ("post", "/reloads/" + identity + "/rollback", {}),
            ("delete", "/reload-requests/" + identity, None)]


@pytest.mark.parametrize("method,path,body", paths())
def test_reload_api_all_routes_use_mutation_guard(api, control, monkeypatch, method, path, body):
    monkeypatch.setattr(local_asr, "_validate_local_mutation_request", lambda _: JSONResponse({}, status_code=403))
    response = api.request(method, "/api/config/local-asr" + path, **({"json": body} if body is not None else {}))
    assert response.status_code == 403
    assert not any(getattr(control, name).await_count for name in vars(control))


@pytest.mark.parametrize("method,path,body", paths())
def test_reload_api_forwards_control_and_preserves_partial_success(api, control, method, path, body):
    response = api.request(method, "/api/config/local-asr" + path, **({"json": body} if body is not None else {}))
    assert response.status_code == 200
    value = response.json()
    assert value["active_changed"] is True and value["retirement_pending"] is True
    assert "token" not in value and "path" not in value
    if path == "/reload":
        control.reload.assert_awaited_once_with(body["request_id"], body["candidate_id"])
    elif method == "post":
        control.action.assert_awaited_once_with(path.split("/")[2], path.split("/")[3])


@pytest.mark.parametrize("change", [{"candidate_id": "../b"}, {"candidate_id": "b/c"}, {"candidate_id": ""},
                                   {"model_dir": "C:\\secret"}, {"enabled": 1}, {"provider": "auto"}])
def test_reload_accepts_catalog_id_only(api, control, change):
    response = api.post("/api/config/local-asr/reload", json={**payload(), "candidate_id": "b", **change})
    assert response.status_code == 422
    control.reload.assert_not_awaited()


def test_reload_respects_core_capability(api, control, monkeypatch):
    monkeypatch.setattr(local_asr, "_core_allows_independent_asr", AsyncMock(return_value=False))
    assert api.post("/api/config/local-asr/reload", json={**payload(), "candidate_id": "b"}).status_code == 409
    control.reload.assert_not_awaited()


@pytest.mark.parametrize("action", ["finalize", "rollback"])
def test_actions_forbid_arbitrary_launch_data(api, control, action):
    path = "/api/config/local-asr/reloads/" + str(uuid4()) + "/" + action
    assert api.post(path, json={"model_dir": "untrusted"}).status_code == 422
    control.action.assert_not_awaited()


def test_invalid_operation_and_action_never_reach_service(api, control):
    assert api.get("/api/config/local-asr/reloads/bad").status_code == 422
    assert api.post("/api/config/local-asr/reloads/" + str(uuid4()) + "/restart", json={}).status_code == 422
    control.status.assert_not_awaited()
    control.action.assert_not_awaited()


@pytest.mark.parametrize("method,path,body", paths())
def test_control_errors_are_sanitized(api, control, method, path, body):
    for name in vars(control):
        getattr(control, name).side_effect = RuntimeError("private path and token")
    response = api.request(method, "/api/config/local-asr" + path, **({"json": body} if body is not None else {}))
    assert response.status_code == 409
    assert response.json() == {"success": False, "code": "ASR_LOCAL_PREPARE_FAILED"}
