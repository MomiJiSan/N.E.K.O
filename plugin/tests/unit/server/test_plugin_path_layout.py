from __future__ import annotations

from pathlib import Path

import pytest

import plugin.settings as settings
from plugin.server.application.install_source.manager import resolve_lock_path
from plugin.server.application.plugin_cli.paths import PluginCliPathPolicy

pytestmark = pytest.mark.plugin_unit


def test_default_layout_separates_exec_from_state_and_keeps_metadata_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "N.E.K.O" / "plugins"
    monkeypatch.delenv("PLUGIN_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("PACKAGE_PROFILES_ROOT", raising=False)
    monkeypatch.delenv("PLUGIN_PACKAGES_ROOT", raising=False)
    monkeypatch.delenv("NEKO_PLUGIN_INSTALL_LOCK_PATH", raising=False)
    monkeypatch.setattr(settings, "get_plugins_directory", lambda: state_root)

    expected_exec = tmp_path / "N.E.K.O" / ".neko-plugin-installations" / "plugins"
    assert settings.get_plugin_state_root() == state_root.resolve()
    assert settings.get_user_plugin_exec_root() == expected_exec.resolve()
    assert settings.get_user_plugin_config_root() == expected_exec.resolve()
    assert settings.get_user_package_profiles_root() == (
        tmp_path / "N.E.K.O" / ".neko-package-profiles"
    ).resolve()
    assert settings.get_user_plugin_packages_root() == (
        tmp_path / "N.E.K.O" / ".neko-plugin-packages"
    ).resolve()
    assert resolve_lock_path() == (tmp_path / "N.E.K.O" / "plugins.lock.json").resolve()


def test_explicit_legacy_config_root_remains_the_execution_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_exec = tmp_path / "custom-code"
    state_root = tmp_path / "user" / "plugins"
    monkeypatch.setenv("PLUGIN_CONFIG_ROOT", str(custom_exec))
    monkeypatch.delenv("NEKO_PLUGIN_INSTALL_LOCK_PATH", raising=False)
    monkeypatch.setattr(settings, "get_plugins_directory", lambda: state_root)

    assert settings.get_user_plugin_exec_root() == custom_exec.resolve()
    assert settings.get_plugin_state_root() == state_root.resolve()
    assert resolve_lock_path() == (state_root.parent / "plugins.lock.json").resolve()


def test_collision_has_stable_error_code(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    with pytest.raises(settings.PluginExecStateRootCollisionError) as exc_info:
        settings.ensure_plugin_exec_state_roots_separated(
            exec_root=root,
            state_root=root,
        )

    assert exc_info.value.code == settings.PLUGIN_EXEC_STATE_ROOT_COLLISION


def test_path_policy_keeps_config_root_compatibility_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compat_exec = tmp_path / "compat-exec"
    state_root = tmp_path / "state"
    monkeypatch.setattr(settings, "BUILTIN_PLUGIN_CONFIG_ROOT", tmp_path / "builtin")
    monkeypatch.setattr(settings, "USER_PLUGIN_EXEC_ROOT", tmp_path / "unused-new-name")
    monkeypatch.setattr(settings, "USER_PLUGIN_CONFIG_ROOT", compat_exec)
    monkeypatch.setattr(settings, "PLUGIN_STATE_ROOT", state_root)
    monkeypatch.setattr(settings, "USER_PLUGIN_PACKAGES_ROOT", tmp_path / "packages")
    monkeypatch.setattr(settings, "USER_PACKAGE_PROFILES_ROOT", tmp_path / "profiles")

    policy = PluginCliPathPolicy.from_settings()

    assert policy.user_plugins_root == compat_exec.resolve()
    policy.ensure_writable_layout()


def test_path_policy_rejects_state_collision(tmp_path: Path) -> None:
    shared = tmp_path / "plugins"
    policy = PluginCliPathPolicy(
        builtin_plugins_root=tmp_path / "builtin",
        user_plugins_root=shared,
        package_artifacts_root=tmp_path / "packages",
        package_profiles_root=tmp_path / "profiles",
        plugin_state_root=shared,
    )

    with pytest.raises(settings.PluginExecStateRootCollisionError) as exc_info:
        policy.ensure_writable_layout()

    assert exc_info.value.code == settings.PLUGIN_EXEC_STATE_ROOT_COLLISION
