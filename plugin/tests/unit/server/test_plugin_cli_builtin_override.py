from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from plugin.neko_plugin_cli.public import build_plugin
from plugin.server.application.plugin_cli.service import PluginCliService
from plugin.server.application.install_source import (
    InstallSourceManager,
    PluginDirectoryScanner,
    set_global_manager,
)
from plugin.server.application.plugins import source_switch, upgrade_support
from plugin.server.application.plugins.source_switch import SourceSwitchError
from plugin.server.domain.errors import ServerDomainError
from plugin.core.state import state


pytestmark = pytest.mark.plugin_unit


def _write_plugin(root: Path, plugin_id: str, version: str) -> Path:
    plugin_dir = root / plugin_id
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.toml").write_text(
        (
            "[plugin]\n"
            f'id = "{plugin_id}"\n'
            f'name = "{plugin_id}"\n'
            f'version = "{version}"\n'
            'type = "plugin"\n'
            f'entry = "plugin.plugins.{plugin_id}:Plugin"\n'
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text("class Plugin: pass\n", encoding="utf-8")
    return plugin_dir


@pytest.mark.asyncio
async def test_cli_rejects_unverified_direct_builtin_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_id = "study_companion"
    builtin_root = tmp_path / "builtin"
    user_root = tmp_path / "installations" / "plugins"
    packages_root = tmp_path / "packages"
    profiles_root = tmp_path / "profiles"
    _write_plugin(builtin_root, plugin_id, "0.1.5")
    packages_root.mkdir(parents=True)
    package = packages_root / f"{plugin_id}.neko-plugin"
    build_plugin(_write_plugin(tmp_path / "source", plugin_id, "0.1.6"), package)

    import plugin.settings as settings

    monkeypatch.setattr(settings, "BUILTIN_PLUGIN_CONFIG_ROOT", builtin_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_CONFIG_ROOT", user_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_PACKAGES_ROOT", packages_root)
    monkeypatch.setattr(settings, "USER_PACKAGE_PROFILES_ROOT", profiles_root)
    monkeypatch.setattr(settings, "PLUGIN_STATE_ROOT", tmp_path / "state")

    service = PluginCliService()
    plan = await service.plan_install(package=str(package))
    assert plan["action"] == "override_builtin"

    with pytest.raises(ServerDomainError) as exc_info:
        await service.install(
            package=str(package),
            confirm_upgrade=True,
            confirmation_token=str(plan["confirmation_token"]),
        )

    assert exc_info.value.code == "PLUGIN_BUILTIN_OVERRIDE_MARKET_REQUIRED"
    assert not (user_root / plugin_id).exists()


@pytest.mark.asyncio
async def test_install_plan_fails_closed_when_exec_and_state_roots_collide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_root = tmp_path / "plugins"
    packages_root = tmp_path / "packages"
    packages_root.mkdir()
    package = packages_root / "demo.neko-plugin"
    build_plugin(_write_plugin(tmp_path / "source", "demo", "1.0.0"), package)

    import plugin.settings as settings

    monkeypatch.setattr(settings, "BUILTIN_PLUGIN_CONFIG_ROOT", tmp_path / "builtin")
    monkeypatch.setattr(settings, "USER_PLUGIN_CONFIG_ROOT", shared_root)
    monkeypatch.setattr(settings, "PLUGIN_STATE_ROOT", shared_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_PACKAGES_ROOT", packages_root)
    monkeypatch.setattr(settings, "USER_PACKAGE_PROFILES_ROOT", tmp_path / "profiles")

    with pytest.raises(ServerDomainError) as exc_info:
        await PluginCliService().plan_install(package=str(package))

    assert exc_info.value.code == "PLUGIN_EXEC_STATE_ROOT_COLLISION"
    assert not shared_root.exists()


def _market_override(*, plugin_id: str, version: str, package_sha256: str) -> dict[str, object]:
    return {
        "channel": "market",
        "mode": "override_builtin",
        "market_detail": {
            "plugin_market_id": plugin_id,
            "version": version,
            "package_url": "https://example.invalid/study_companion.neko-plugin",
            "channel": "stable",
            "package_sha256": package_sha256,
            "payload_hash": None,
            "published_at": "2026-08-24T00:00:00.000000Z",
            "expected_plugin_toml_id": plugin_id,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_market_start", [False, True])
async def test_market_builtin_override_switches_or_restores_without_touching_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_market_start: bool,
) -> None:
    plugin_id = "study_companion"
    builtin_root = tmp_path / "builtin"
    user_root = tmp_path / "installations" / "plugins"
    packages_root = tmp_path / "packages"
    profiles_root = tmp_path / "profiles"
    state_root = tmp_path / "plugins"
    builtin = _write_plugin(builtin_root, plugin_id, "0.1.5")
    packages_root.mkdir(parents=True)
    package = packages_root / f"{plugin_id}.neko-plugin"
    build_plugin(_write_plugin(tmp_path / "source", plugin_id, "0.1.6"), package)
    package_sha256 = hashlib.sha256(package.read_bytes()).hexdigest()
    state_files = {
        state_root / plugin_id / "data" / "study.db": b"sqlite-db",
        state_root / plugin_id / "data" / "study.db-wal": b"sqlite-wal",
        state_root / plugin_id / "data" / "study.db-shm": b"sqlite-shm",
    }
    for path, content in state_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in state_files}

    import plugin.settings as settings
    from plugin.server.application.plugins import lifecycle_service

    monkeypatch.setattr(settings, "BUILTIN_PLUGIN_CONFIG_ROOT", builtin_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_CONFIG_ROOT", user_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_PACKAGES_ROOT", packages_root)
    monkeypatch.setattr(settings, "USER_PACKAGE_PROFILES_ROOT", profiles_root)
    monkeypatch.setattr(settings, "PLUGIN_STATE_ROOT", state_root)

    manager = InstallSourceManager(
        lock_path=tmp_path / "plugins.lock.json",
        builtin_root=builtin_root,
        user_root=user_root,
        scanner=PluginDirectoryScanner(builtin_root, user_root),
    )
    set_global_manager(manager)
    start_calls: list[str] = []

    async def refresh_registry() -> dict[str, object]:
        effective = user_root / plugin_id if (user_root / plugin_id).is_dir() else builtin
        with state.acquire_plugins_write_lock():
            state.plugins[plugin_id] = {
                "config_path": str(effective / "plugin.toml"),
                "status": "stopped",
            }
        return {"ok": True}

    async def is_running(_plugin_id: str) -> bool:
        return True

    async def stop(_plugin_id: str) -> None:
        return None

    async def start(_plugin_id: str) -> None:
        start_calls.append(_plugin_id)
        if fail_market_start and len(start_calls) == 1 and (user_root / plugin_id).exists():
            raise RuntimeError("market start failed")

    monkeypatch.setattr(lifecycle_service.plugin_registry_service, "refresh_registry", refresh_registry)
    monkeypatch.setattr(upgrade_support, "plugin_is_running", is_running)
    monkeypatch.setattr(upgrade_support, "stop_plugin_for_replace", stop)
    monkeypatch.setattr(upgrade_support, "start_plugin_after_replace", lambda pid, strict: start(pid))

    try:
        service = PluginCliService()
        if fail_market_start:
            with pytest.raises(SourceSwitchError) as exc_info:
                await service.install_builtin_override(
                    package=str(package),
                    market_override=_market_override(
                        plugin_id=plugin_id,
                        version="0.1.6",
                        package_sha256=package_sha256,
                    ),
                )
            assert exc_info.value.code == "override_start_failed"
            assert exc_info.value.rollback_code == "override_rollback_completed"
            assert not (user_root / plugin_id).exists()
            assert (builtin / "plugin.toml").is_file()
            assert start_calls == [plugin_id, plugin_id]
        else:
            result = await service.install_builtin_override(
                package=str(package),
                market_override=_market_override(
                    plugin_id=plugin_id,
                    version="0.1.6",
                    package_sha256=package_sha256,
                ),
            )
            assert result["operation"] == "override_builtin"
            assert result["restarted"] is True
            assert (user_root / plugin_id / "plugin.toml").is_file()
            assert (builtin / "plugin.toml").is_file()
            entry = manager.find_active_market_entry(plugin_id)
            assert entry is not None
            assert entry.directory_name == plugin_id
            assert start_calls == [plugin_id]
        after_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in state_files}
        assert after_hashes == before_hashes
        assert not list(user_root.glob(".neko_override_*"))
    finally:
        set_global_manager(None)
        with state.acquire_plugins_write_lock():
            state.plugins.pop(plugin_id, None)


@pytest.mark.asyncio
async def test_upload_and_install_routes_override_mode_to_source_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugin.settings as settings

    packages_root = tmp_path / "packages"
    user_root = tmp_path / "installations" / "plugins"
    profiles_root = tmp_path / "profiles"
    source_package = tmp_path / "download" / "study_companion.neko-plugin"
    source_package.parent.mkdir()
    source_package.write_bytes(b"verified-package")
    package_sha256 = hashlib.sha256(source_package.read_bytes()).hexdigest()
    target_dir = user_root / "study_companion"

    monkeypatch.setattr(settings, "BUILTIN_PLUGIN_CONFIG_ROOT", tmp_path / "builtin")
    monkeypatch.setattr(settings, "USER_PLUGIN_CONFIG_ROOT", user_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_PACKAGES_ROOT", packages_root)
    monkeypatch.setattr(settings, "USER_PACKAGE_PROFILES_ROOT", profiles_root)
    monkeypatch.setattr(settings, "PLUGIN_STATE_ROOT", tmp_path / "state")

    service = PluginCliService()
    calls: list[dict[str, object]] = []

    async def install_override(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "package_path": str(packages_root / source_package.name),
            "package_type": "plugin",
            "package_id": "study_companion",
            "plugins_root": str(user_root),
            "profiles_root": str(profiles_root),
            "installed_plugins": [
                {
                    "source_folder": "study_companion",
                    "target_plugin_id": "study_companion",
                    "target_dir": str(target_dir),
                    "renamed": False,
                }
            ],
            "profile_dir": None,
            "metadata_found": True,
            "payload_hash": "b" * 64,
            "payload_hash_verified": True,
            "conflict_strategy": "fail",
            "installed_plugin_count": 1,
            "operation": "override_builtin",
            "restarted": True,
            "rollback_status": "not_needed",
            "install_source_warning": None,
        }

    monkeypatch.setattr(service, "install_builtin_override", install_override)
    result = await service.upload_and_install(
        filename=source_package.name,
        package_path=str(source_package),
        install_source_override=_market_override(
            plugin_id="study_companion",
            version="0.1.6",
            package_sha256=package_sha256,
        ),
    )

    assert len(calls) == 1
    assert calls[0]["market_override"]["mode"] == "override_builtin"
    assert result["unpack"]["operation"] == "override_builtin"
    assert result["unpack"]["restarted"] is True
    assert result["install"]["channel"] == "market"
