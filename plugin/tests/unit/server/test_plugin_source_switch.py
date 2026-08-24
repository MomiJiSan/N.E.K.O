from __future__ import annotations

import copy
from pathlib import Path

import pytest

from plugin.core.state import state
from plugin.server.application.plugins.source_switch import (
    SourceSwitchError,
    SourceSwitchRequest,
    switch_builtin_source,
)

pytestmark = pytest.mark.plugin_unit


def _plan(token: str) -> dict[str, object]:
    return {
        "action": "override_builtin",
        "plugin_id": "study_companion",
        "current_source": "builtin",
        "target_source": "market",
        "confirmation_token": token,
    }


@pytest.mark.asyncio
async def test_switch_builtin_source_promotes_user_code_without_touching_state(
    tmp_path: Path,
) -> None:
    exec_root = tmp_path / "exec"
    staging = exec_root / ".study_companion.staging-test"
    target = exec_root / "study_companion"
    staging.mkdir(parents=True)
    (staging / "plugin.toml").write_text("[plugin]\nid='study_companion'\n", encoding="utf-8")
    state_db = tmp_path / "plugins" / "study_companion" / "data" / "study.db"
    state_db.parent.mkdir(parents=True)
    state_db.write_bytes(b"database-before")
    events: list[str] = []
    running = True
    plugins_backup = copy.deepcopy(state.plugins)

    async def refresh() -> None:
        source = target if target.exists() else tmp_path / "builtin" / "study_companion"
        with state.acquire_plugins_write_lock():
            state.plugins["study_companion"] = {
                "config_path": str(source / "plugin.toml"),
                "effective_source": "user" if target.exists() else "builtin",
            }

    async def is_running(_plugin_id: str) -> bool:
        return running

    async def stop(_plugin_id: str) -> None:
        nonlocal running
        running = False
        events.append("stop")

    async def start(_plugin_id: str) -> None:
        nonlocal running
        running = True
        events.append("start")

    try:
        result = await switch_builtin_source(
            SourceSwitchRequest(
                plugin_id="study_companion",
                staged_plugin_dir=staging,
                target_plugin_dir=target,
                confirmation_token="token",
            ),
            rebuild_plan=lambda: _async_value(_plan("token")),
            read_lock_snapshot=lambda: _async_value({"old": True}),
            commit_lock=lambda: _async_value(None),
            restore_lock=lambda _snapshot: _async_value(None),
            clear_user_source=lambda: _async_value(None),
            refresh_registry=refresh,
            is_running=is_running,
            stop=stop,
            start=start,
        )

        assert result.code == "override_completed"
        assert result.effective_source == "market"
        assert events == ["stop", "start"]
        assert (target / "plugin.toml").is_file()
        assert state_db.read_bytes() == b"database-before"
    finally:
        with state.acquire_plugins_write_lock():
            state.plugins.clear()
            state.plugins.update(plugins_backup)


@pytest.mark.asyncio
async def test_switch_start_failure_rolls_back_code_profile_lock_and_builtin_runtime(
    tmp_path: Path,
) -> None:
    exec_root = tmp_path / "exec"
    staging = exec_root / ".study_companion.staging-test"
    target = exec_root / "study_companion"
    staging.mkdir(parents=True)
    (staging / "plugin.toml").write_text("new", encoding="utf-8")
    profiles_root = tmp_path / "profiles"
    staged_profile = profiles_root / ".study_companion.staging-test"
    target_profile = profiles_root / "study_companion"
    staged_profile.mkdir(parents=True)
    (staged_profile / "default.toml").write_text("profile", encoding="utf-8")
    builtin_config = tmp_path / "builtin" / "study_companion" / "plugin.toml"
    builtin_config.parent.mkdir(parents=True)
    builtin_config.write_text("builtin", encoding="utf-8")
    state_db = tmp_path / "plugins" / "study_companion" / "data" / "study.db"
    state_db.parent.mkdir(parents=True)
    state_db.write_bytes(b"stable-db")
    running = True
    lock_value: object = {"source": "builtin"}
    plugins_backup = copy.deepcopy(state.plugins)

    async def refresh() -> None:
        source = target if target.exists() else builtin_config.parent
        with state.acquire_plugins_write_lock():
            state.plugins["study_companion"] = {"config_path": str(source / "plugin.toml")}

    async def is_running(_plugin_id: str) -> bool:
        return running

    async def stop(_plugin_id: str) -> None:
        nonlocal running
        running = False

    async def start(_plugin_id: str) -> None:
        nonlocal running
        if target.exists():
            raise RuntimeError("market failed to start")
        running = True

    async def commit_lock() -> None:
        nonlocal lock_value
        lock_value = {"source": "market"}

    async def restore_lock(snapshot: object) -> None:
        nonlocal lock_value
        lock_value = snapshot

    try:
        with pytest.raises(SourceSwitchError) as exc_info:
            await switch_builtin_source(
                SourceSwitchRequest(
                    plugin_id="study_companion",
                    staged_plugin_dir=staging,
                    target_plugin_dir=target,
                    confirmation_token="token",
                    staged_profile_dir=staged_profile,
                    target_profile_dir=target_profile,
                ),
                rebuild_plan=lambda: _async_value(_plan("token")),
                read_lock_snapshot=lambda: _async_value(lock_value),
                commit_lock=commit_lock,
                restore_lock=restore_lock,
                clear_user_source=lambda: _async_value(None),
                refresh_registry=refresh,
                is_running=is_running,
                stop=stop,
                start=start,
            )

        assert exc_info.value.code == "override_start_failed"
        assert exc_info.value.rollback_code == "override_rollback_completed"
        assert exc_info.value.as_payload() == {
            "code": "override_start_failed",
            "stage": "start_market",
            "error_type": "RuntimeError",
            "rollback_code": "override_rollback_completed",
            "running": False,
            "restored": True,
        }
        assert target.exists() is False
        assert target_profile.exists() is False
        assert lock_value == {"source": "builtin"}
        assert running is True
        assert state_db.read_bytes() == b"stable-db"
        with state.acquire_plugins_read_lock():
            assert state.plugins["study_companion"]["config_path"] == str(builtin_config)
    finally:
        with state.acquire_plugins_write_lock():
            state.plugins.clear()
            state.plugins.update(plugins_backup)


@pytest.mark.asyncio
async def test_switch_rejects_changed_source_before_mutating(tmp_path: Path) -> None:
    exec_root = tmp_path / "exec"
    staging = exec_root / ".study_companion.staging-test"
    target = exec_root / "study_companion"
    staging.mkdir(parents=True)

    plan = _plan("token")
    plan["current_source"] = "market"
    with pytest.raises(SourceSwitchError) as exc_info:
        await switch_builtin_source(
            SourceSwitchRequest(
                plugin_id="study_companion",
                staged_plugin_dir=staging,
                target_plugin_dir=target,
                confirmation_token="token",
            ),
            rebuild_plan=lambda: _async_value(plan),
            read_lock_snapshot=lambda: _async_value(None),
            commit_lock=lambda: _async_value(None),
            restore_lock=lambda _snapshot: _async_value(None),
            clear_user_source=lambda: _async_value(None),
            refresh_registry=lambda: _async_value(None),
            is_running=lambda _plugin_id: _async_value(False),
            stop=lambda _plugin_id: _async_value(None),
            start=lambda _plugin_id: _async_value(None),
        )

    assert exc_info.value.code == "override_source_changed"
    assert staging.is_dir()
    assert target.exists() is False


async def _async_value(value):
    return value
