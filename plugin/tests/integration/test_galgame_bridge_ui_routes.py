from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from plugin.core.state import state
from plugin.server.infrastructure.exceptions import register_exception_handlers
from plugin.server.routes import plugin_ui as plugin_ui_route_module


pytestmark = pytest.mark.plugin_integration


@pytest.fixture
def galgame_bridge_plugin_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "plugins" / "galgame_bridge"


@pytest.fixture
def plugin_ui_test_app() -> FastAPI:
    app = FastAPI(title="plugin-ui-test-app")
    register_exception_handlers(app)
    app.include_router(plugin_ui_route_module.router)
    return app


@pytest.fixture
async def plugin_ui_async_client(plugin_ui_test_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=plugin_ui_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def registered_galgame_bridge_meta(galgame_bridge_plugin_dir: Path) -> Iterator[None]:
    plugins_backup = copy.deepcopy(state.plugins)
    try:
        with state.acquire_plugins_write_lock():
            state.plugins.clear()
            state.plugins["galgame_bridge"] = {
                "id": "galgame_bridge",
                "name": "Galgame Bridge",
                "config_path": str(galgame_bridge_plugin_dir / "plugin.toml"),
                "static_ui_config": {
                    "enabled": True,
                    "directory": str(galgame_bridge_plugin_dir / "static"),
                    "index_file": "index.html",
                    "cache_control": "no-store, no-cache, must-revalidate, max-age=0",
                    "plugin_id": "galgame_bridge",
                },
                "list_actions": [
                    {
                        "id": "open_ui",
                        "kind": "ui",
                        "target": "/plugin/galgame_bridge/ui/",
                        "open_in": "new_tab",
                    }
                ],
            }
        yield
    finally:
        with state.acquire_plugins_write_lock():
            state.plugins.clear()
            state.plugins.update(plugins_backup)


@pytest.mark.asyncio
async def test_galgame_bridge_ui_index_route_serves_static_dashboard(
    plugin_ui_async_client: AsyncClient,
    registered_galgame_bridge_meta,
) -> None:
    response = await plugin_ui_async_client.get("/plugin/galgame_bridge/ui/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert "<title>Galgame Bridge</title>" in response.text
    assert "N.E.K.O Phase 2" in response.text
    assert "Game LLM Agent 的运行状态与推送记录" in response.text


@pytest.mark.asyncio
async def test_galgame_bridge_ui_script_uses_runs_api_only(
    plugin_ui_async_client: AsyncClient,
    registered_galgame_bridge_meta,
) -> None:
    response = await plugin_ui_async_client.get("/plugin/galgame_bridge/ui/main.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "const RUNS_URL = '/runs';" in response.text
    assert "session.json" not in response.text
    assert "events.jsonl" not in response.text
    assert "galgame_explain_line" in response.text
    assert "galgame_summarize_scene" in response.text
    assert "galgame_get_status" in response.text
    assert "galgame_get_snapshot" in response.text
    assert "galgame_get_history" in response.text
    assert "galgame_agent_command" in response.text
    assert "active_data_source" in response.text
    assert "memory_reader_runtime" in response.text


@pytest.mark.asyncio
async def test_galgame_bridge_ui_info_reports_registered_assets(
    plugin_ui_async_client: AsyncClient,
    registered_galgame_bridge_meta,
) -> None:
    response = await plugin_ui_async_client.get("/plugin/galgame_bridge/ui-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["plugin_id"] == "galgame_bridge"
    assert payload["has_ui"] is True
    assert payload["explicitly_registered"] is True
    assert payload["ui_path"] == "/plugin/galgame_bridge/ui/"
    assert payload["static_files_count"] >= 3
    assert "index.html" in payload["static_files"]
    assert "main.js" in payload["static_files"]
    assert "style.css" in payload["static_files"]


@pytest.mark.asyncio
async def test_galgame_bridge_ui_rejects_path_traversal(
    plugin_ui_async_client: AsyncClient,
    registered_galgame_bridge_meta,
) -> None:
    response = await plugin_ui_async_client.get("/plugin/galgame_bridge/ui/%2e%2e/plugin.toml")

    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied: path traversal detected"
