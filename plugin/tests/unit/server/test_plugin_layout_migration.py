from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from plugin.server.application.plugins import layout_migration as migration_module
from plugin.server.application.plugins.layout_migration import (
    LAYOUT_LEDGER_FILENAME,
    migrate_legacy_plugin_layout,
)
from plugin.settings import PLUGIN_EXEC_STATE_ROOT_COLLISION

pytestmark = pytest.mark.plugin_unit


def _write_plugin(root: Path, plugin_id: str, *, manifest_id: str | None = None) -> Path:
    plugin_dir = root / plugin_id
    plugin_dir.mkdir(parents=True)
    declared_id = manifest_id or plugin_id
    (plugin_dir / "plugin.toml").write_text(
        "\n".join(
            (
                "[plugin]",
                f'id = "{declared_id}"',
                f'entry = "plugin.plugins.{declared_id}:Plugin"',
                'version = "1.0.0"',
                "",
            )
        ),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text("class Plugin:\n    pass\n", encoding="utf-8")
    return plugin_dir


@pytest.mark.asyncio
async def test_migration_is_atomic_idempotent_and_does_not_resurrect(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "user" / "plugins"
    exec_root = tmp_path / "user" / ".neko-plugin-installations" / "plugins"
    source = _write_plugin(state_root, "study_companion")
    database = source / "data" / "study.db"
    database.parent.mkdir()
    database.write_bytes(b"active database bytes")
    wal = source / "data" / "study.db-wal"
    shm = source / "data" / "study.db-shm"
    wal.write_bytes(b"active wal bytes")
    shm.write_bytes(b"active shm bytes")
    (source / "config").mkdir()
    (source / "config" / "settings.json").write_text("{}", encoding="utf-8")
    (source / "cache").mkdir()
    (source / "cache" / "temporary.bin").write_bytes(b"cache")
    state_files = (database, wal, shm)
    before_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in state_files
    }
    (source / "static" / "index.html").parent.mkdir()
    (source / "static" / "index.html").write_text("ok", encoding="utf-8")

    first = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)

    assert first.migrated == ("study_companion",)
    assert not first.blocked
    destination = exec_root / "study_companion"
    assert (destination / "plugin.toml").is_file()
    assert (destination / "static" / "index.html").read_text(encoding="utf-8") == "ok"
    assert not (destination / "data").exists()
    assert not (destination / "config").exists()
    assert not (destination / "cache").exists()
    assert source.is_dir()
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in state_files
    } == before_hashes
    ledger_path = state_root.parent / LAYOUT_LEDGER_FILENAME
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["entries"][0]["plugin_id"] == "study_companion"
    assert ledger["entries"][0]["old_path"] == str(source.resolve())
    assert ledger["entries"][0]["new_path"] == str(destination.resolve())
    assert len(ledger["entries"][0]["manifest_sha256"]) == 64

    second = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)
    assert second.migrated == ()
    assert second.skipped == ("study_companion",)

    shutil.rmtree(destination)
    third = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)
    assert third.migrated == ()
    assert third.skipped == ("study_companion",)
    assert not destination.exists()


@pytest.mark.asyncio
async def test_pure_state_directory_is_not_migrated(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    state_dir = state_root / "state_only"
    for name in ("config", "data", "cache"):
        (state_dir / name).mkdir(parents=True)
    exec_root = tmp_path / "exec" / "plugins"

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)

    assert result.migrated == ()
    assert result.blocked == ()
    assert not (exec_root / "state_only").exists()


@pytest.mark.asyncio
async def test_manifest_id_mismatch_is_blocked(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    _write_plugin(state_root, "directory_name", manifest_id="different_id")
    exec_root = tmp_path / "exec"

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)

    assert result.migrated == ()
    assert [issue.code for issue in result.blocked] == ["PLUGIN_LAYOUT_MIGRATION_ID_MISMATCH"]
    assert not (exec_root / "directory_name").exists()


@pytest.mark.asyncio
async def test_invalid_entry_is_blocked(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    source = _write_plugin(state_root, "broken")
    (source / "plugin.toml").write_text(
        '[plugin]\nid = "broken"\nentry = "plugins.broken.missing:Plugin"\n',
        encoding="utf-8",
    )

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=tmp_path / "exec")

    assert [issue.code for issue in result.blocked] == ["PLUGIN_LAYOUT_MIGRATION_ENTRY_INVALID"]


@pytest.mark.asyncio
async def test_stale_staging_is_cleaned_on_next_start(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    exec_root = tmp_path / "exec"
    profiles_root = tmp_path / "profiles"
    stale_paths = (
        exec_root / ".neko-layout-v1-old-deadbeef.staging",
        exec_root / ".neko_override_staging_old_deadbeef",
        exec_root / ".neko_override_unpack_old_deadbeef",
        profiles_root / ".neko_override_staging_profile_deadbeef",
        profiles_root / ".neko_override_unpack_profile_deadbeef",
    )
    for stale in stale_paths:
        stale.mkdir(parents=True)
        (stale / "partial").write_text("partial", encoding="utf-8")

    result = await migrate_legacy_plugin_layout(
        state_root=state_root,
        exec_root=exec_root,
        profiles_root=profiles_root,
    )

    assert set(result.cleaned_staging) == {str(path.resolve()) for path in stale_paths}
    assert all(not path.exists() for path in stale_paths)


@pytest.mark.asyncio
async def test_exec_state_collision_fails_closed_without_writes(tmp_path: Path) -> None:
    shared_root = tmp_path / "plugins"
    _write_plugin(shared_root, "legacy")

    result = await migrate_legacy_plugin_layout(
        state_root=shared_root,
        exec_root=shared_root,
    )

    assert [issue.code for issue in result.blocked] == [PLUGIN_EXEC_STATE_ROOT_COLLISION]
    assert not (tmp_path / LAYOUT_LEDGER_FILENAME).exists()
    assert not list(shared_root.glob(".neko-layout-v1-*.staging"))


@pytest.mark.asyncio
async def test_linked_legacy_tree_is_blocked_when_links_are_supported(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    source = _write_plugin(state_root, "linked")
    external = tmp_path / "external.txt"
    external.write_text("outside", encoding="utf-8")
    link = source / "outside.txt"
    try:
        os.symlink(external, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable for this test account")

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=tmp_path / "exec")

    assert [issue.code for issue in result.blocked] == ["PLUGIN_LAYOUT_MIGRATION_SYMLINK"]


@pytest.mark.asyncio
async def test_ledger_write_failure_removes_promoted_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root = tmp_path / "plugins"
    _write_plugin(state_root, "rollback_me")
    exec_root = tmp_path / "exec"

    def _fail_ledger_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(migration_module, "_atomic_write_ledger", _fail_ledger_write)

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)

    assert [issue.code for issue in result.blocked] == [
        "PLUGIN_LAYOUT_MIGRATION_LEDGER_WRITE_FAILED"
    ]
    assert not (exec_root / "rollback_me").exists()
    assert not list(exec_root.glob(".neko-layout-v1-*.staging"))


@pytest.mark.asyncio
async def test_invalid_ledger_blocks_migration_without_overwrite(tmp_path: Path) -> None:
    state_root = tmp_path / "plugins"
    _write_plugin(state_root, "legacy")
    ledger = state_root.parent / LAYOUT_LEDGER_FILENAME
    ledger.write_text("not-json", encoding="utf-8")
    exec_root = tmp_path / "exec"

    result = await migrate_legacy_plugin_layout(state_root=state_root, exec_root=exec_root)

    assert [issue.code for issue in result.blocked] == [
        "PLUGIN_LAYOUT_MIGRATION_LEDGER_INVALID"
    ]
    assert ledger.read_text(encoding="utf-8") == "not-json"
    assert not (exec_root / "legacy").exists()
