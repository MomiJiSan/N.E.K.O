from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from plugin.server.local_app_bridge import host_installations as module
from plugin.server.local_app_bridge.contracts import LocalAppPolicy
from plugin.server.local_app_bridge.runtime import (
    LocalAppBridgeRuntime,
    LocalAppPluginRegistration,
    LocalAppPluginTarget,
)


pytestmark = pytest.mark.plugin_unit


def _registration(app_id: str = "knowledge_dungeon") -> LocalAppPluginRegistration:
    return LocalAppPluginRegistration(
        policy=LocalAppPolicy(
            app_id=app_id,
            allowed_operations={
                "study_companion:dungeon": frozenset({"knowledge_dungeon.bootstrap"})
            },
        ),
        targets=(
            LocalAppPluginTarget(
                scope="study_companion:dungeon",
                operation="knowledge_dungeon.bootstrap",
                plugin_id="study_companion",
                plugin_operation="knowledge_dungeon.bootstrap",
            ),
        ),
    )


def _write_config(
    path: Path,
    *,
    app_id: str = "knowledge_dungeon",
    executable: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    installation: dict[str, object] = {
        "app_id": app_id,
        "title": "Knowledge Dungeon",
        "executable": executable or sys.executable,
        "args": ["--product-mode"],
    }
    installation.update(extra or {})
    path.write_text(
        json.dumps({"version": 1, "installations": [installation]}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_host_file_configures_only_manifest_authorized_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = LocalAppBridgeRuntime()
    runtime.register_plugin_app(_registration())
    config_path = tmp_path / "local-app-installations.json"
    _write_config(config_path)
    monkeypatch.setenv(module.LOCAL_APP_INSTALLATIONS_FILE_ENV, str(config_path))
    monkeypatch.setattr(module, "get_local_app_bridge_runtime", lambda: runtime)

    issues = await module.configure_local_app_installations_from_host()
    await runtime.start()
    descriptor = runtime.describe_plugin_app(
        "study_companion", fallback_title="Study Companion"
    )

    assert issues == ()
    assert descriptor is not None
    assert descriptor.app_id == "knowledge_dungeon"
    assert descriptor.title == "Knowledge Dungeon"
    assert descriptor.available is True
    await runtime.close()


@pytest.mark.asyncio
async def test_unknown_app_id_fails_closed_without_changing_plugin_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = LocalAppBridgeRuntime()
    runtime.register_plugin_app(_registration())
    config_path = tmp_path / "local-app-installations.json"
    _write_config(config_path, app_id="untrusted_app")
    monkeypatch.setenv(module.LOCAL_APP_INSTALLATIONS_FILE_ENV, str(config_path))
    monkeypatch.setattr(module, "get_local_app_bridge_runtime", lambda: runtime)

    issues = await module.configure_local_app_installations_from_host()
    await runtime.start()
    descriptor = runtime.describe_plugin_app(
        "study_companion", fallback_title="Study Companion"
    )

    assert issues == (
        module.LocalAppInstallationIssue("installations_configuration_invalid"),
    )
    assert descriptor is not None
    assert descriptor.available is False
    await runtime.close()


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            {
                "version": 1,
                "installations": [
                    {
                        "app_id": "knowledge_dungeon",
                        "title": "Knowledge Dungeon",
                        "executable": "relative.exe",
                        "args": [],
                    }
                ],
            },
            "installation_executable_invalid",
        ),
        (
            {"version": 1, "installations": [], "unexpected": True},
            "installations_fields_invalid",
        ),
    ],
)
def test_host_file_rejects_relative_paths_and_extra_fields(
    tmp_path: Path, payload: dict[str, object], expected_code: str
) -> None:
    config_path = tmp_path / "local-app-installations.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(module._InvalidInstallationConfig, match=expected_code):
        module._load_installations(config_path)


@pytest.mark.asyncio
async def test_missing_host_file_configuration_disables_launch_without_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[tuple[object, ...]] = []

    class _Runtime:
        @staticmethod
        def configure_installations(installations: tuple[object, ...]) -> None:
            configured.append(installations)

    monkeypatch.delenv(module.LOCAL_APP_INSTALLATIONS_FILE_ENV, raising=False)
    monkeypatch.setattr(module, "get_local_app_bridge_runtime", _Runtime)

    assert await module.configure_local_app_installations_from_host() == ()
    assert configured == [()]
