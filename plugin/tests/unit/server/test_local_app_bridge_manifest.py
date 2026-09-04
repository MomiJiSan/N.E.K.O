from __future__ import annotations

from pathlib import Path

import pytest

from plugin.server.local_app_bridge import manifest as module


pytestmark = pytest.mark.plugin_unit


def _write_manifest(
    root: Path,
    plugin_id: str,
    *,
    app_id: str = "knowledge_dungeon",
    operation: str = "knowledge_dungeon.bootstrap",
    plugin_operation: str = "knowledge_dungeon.bootstrap",
    extra: str = "",
) -> Path:
    path = root / plugin_id / "plugin.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "[plugin]",
                f'id = "{plugin_id}"',
                "",
                "[plugin.local_app]",
                f'app_id = "{app_id}"',
                'scope = "study_companion:dungeon"',
                extra,
                "",
                "[plugin.local_app.operations]",
                f'"{operation}" = "{plugin_operation}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _metadata(path: Path, *, enabled: bool = True) -> dict[str, object]:
    return {"config_path": str(path), "runtime_enabled": enabled}


def test_valid_manifest_builds_fixed_host_target(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, "study_companion")
    registrations, issues = module._discover_registrations(
        {"study_companion": _metadata(path)}
    )
    assert issues == ()
    assert len(registrations) == 1
    registration = registrations[0]
    assert registration.policy.app_id == "knowledge_dungeon"
    assert registration.policy.allowed_operations == {
        "study_companion:dungeon": frozenset({"knowledge_dungeon.bootstrap"})
    }
    assert registration.targets[0].plugin_id == "study_companion"
    assert registration.targets[0].plugin_operation == "knowledge_dungeon.bootstrap"


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    [
        ({"extra": 'unexpected = "value"'}, "local_app_fields_invalid"),
        (
            {"operation": "KnowledgeDungeon.Bootstrap"},
            "local_app_operations_invalid",
        ),
        (
            {"plugin_operation": "bootstrap"},
            "local_app_operations_invalid",
        ),
    ],
)
def test_invalid_declaration_only_disables_local_app_capability(
    tmp_path: Path,
    kwargs: dict[str, str],
    expected_code: str,
) -> None:
    path = _write_manifest(tmp_path, "otherwise_valid_plugin", **kwargs)
    registrations, issues = module._discover_registrations(
        {"otherwise_valid_plugin": _metadata(path)}
    )
    assert registrations == ()
    assert issues == (
        module.LocalAppManifestIssue("otherwise_valid_plugin", expected_code),
    )


def test_duplicate_app_id_disables_both_plugins(tmp_path: Path) -> None:
    first = _write_manifest(tmp_path, "first")
    second = _write_manifest(tmp_path, "second")
    registrations, issues = module._discover_registrations(
        {"first": _metadata(first), "second": _metadata(second)}
    )
    assert registrations == ()
    assert issues == (
        module.LocalAppManifestIssue("first", "duplicate_app_id"),
        module.LocalAppManifestIssue("second", "duplicate_app_id"),
    )


def test_disabled_plugin_declaration_is_not_registered(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, "disabled")
    assert module._discover_registrations(
        {"disabled": _metadata(path, enabled=False)}
    ) == ((), ())


@pytest.mark.asyncio
async def test_registry_configuration_is_installed_as_one_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_manifest(tmp_path, "study_companion")
    configured: list[tuple[object, ...]] = []

    class _Runtime:
        def configure_plugin_apps(self, registrations: tuple[object, ...]) -> None:
            configured.append(registrations)

    monkeypatch.setattr(
        module,
        "_registry_snapshot",
        lambda: {"study_companion": _metadata(path)},
    )
    monkeypatch.setattr(module, "get_local_app_bridge_runtime", _Runtime)

    issues = await module.configure_local_app_bridge_from_registry()

    assert issues == ()
    assert len(configured) == 1 and len(configured[0]) == 1
