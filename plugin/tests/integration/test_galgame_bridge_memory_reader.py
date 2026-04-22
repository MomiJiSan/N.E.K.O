from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugin.plugins.galgame_bridge import GalgameBridgePlugin
from plugin.plugins.galgame_bridge.memory_reader import (
    DetectedGameProcess,
    MemoryReaderBridgeWriter,
    MemoryReaderManager,
)
from plugin.sdk.plugin import Ok


pytestmark = pytest.mark.plugin_integration


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


class _Ctx:
    plugin_id = "galgame_bridge"
    metadata = {}
    bus = None

    def __init__(self, plugin_dir: Path, effective_config: dict[str, object]) -> None:
        self.logger = _Logger()
        self.config_path = plugin_dir / "plugin.toml"
        self._effective_config = {
            "plugin": {"store": {"enabled": True}, "database": {"enabled": False}},
            "plugin_state": {"backend": "memory"},
        }
        self._config = effective_config
        self.pushed_messages: list[dict[str, object]] = []

    async def get_own_config(self, timeout: float = 5.0):
        return {"config": self._config}

    async def get_own_base_config(self, timeout: float = 5.0):
        return {"config": self._config}

    async def get_own_profiles_state(self, timeout: float = 5.0):
        return {"profiles": [], "active": None}

    async def get_own_profile_config(self, profile_name: str, timeout: float = 5.0):
        return {"profile_name": profile_name, "config": self._config}

    async def get_own_effective_config(self, profile_name: str | None = None, timeout: float = 5.0):
        return {"config": self._config}

    async def update_own_config(self, updates, timeout: float = 10.0):
        self._config = dict(self._config)
        self._config.update(dict(updates or {}))
        return {"config": self._config}

    async def query_plugins(self, filters, timeout: float = 5.0):
        return {"plugins": []}

    async def trigger_plugin_event(self, **kwargs):
        raise RuntimeError("unexpected trigger_plugin_event in memory reader integration test")

    async def get_system_config(self, timeout: float = 5.0):
        return {}

    async def query_memory(self, bucket_id: str, query: str, timeout: float = 5.0):
        return {"items": []}

    async def run_update(self, **kwargs):
        return {"ok": True}

    async def export_push(self, **kwargs):
        return {"ok": True}

    async def finish(self, **kwargs):
        return {"ok": True}

    def push_message(self, **kwargs):
        self.pushed_messages.append(dict(kwargs))
        return {"ok": True}

    def update_status(self, status):
        return None


class _FakeTextractorHandle:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = list(lines or [])
        self.writes: list[str] = []
        self.returncode: int | None = None
        self.terminated = False

    async def write(self, payload: str) -> None:
        self.writes.append(payload)

    async def readline(self, timeout: float) -> str | None:
        del timeout
        if not self.lines:
            return None
        return self.lines.pop(0)

    def poll(self) -> int | None:
        return self.returncode

    async def terminate(self) -> None:
        self.terminated = True
        if self.returncode is None:
            self.returncode = 0

    async def wait(self, timeout: float) -> int | None:
        del timeout
        return self.returncode


def _session_state(
    *,
    speaker: str = "",
    text: str = "",
    scene_id: str = "scene-a",
    line_id: str = "",
    route_id: str = "",
    ts: str = "2026-04-22T02:00:00Z",
) -> dict[str, object]:
    return {
        "speaker": speaker,
        "text": text,
        "choices": [],
        "scene_id": scene_id,
        "line_id": line_id,
        "route_id": route_id,
        "is_menu_open": False,
        "save_context": {
            "kind": "unknown",
            "slot_id": "",
            "display_name": "",
        },
        "ts": ts,
    }


def _session(
    *,
    game_id: str,
    session_id: str,
    last_seq: int,
    state: dict[str, object],
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "game_id": game_id,
        "game_title": game_id,
        "engine": "renpy",
        "session_id": session_id,
        "started_at": "2026-04-22T02:00:00Z",
        "last_seq": last_seq,
        "locale": "ja-JP",
        "bridge_sdk_version": "1.0.0",
        "state": state,
    }


def _write_session(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_plugin_dirs(tmp_path: Path) -> tuple[Path, Path]:
    plugin_dir = tmp_path / "plugin_cfg"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.toml").write_text("", encoding="utf-8")
    static_dir = plugin_dir / "static"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>ui</title>", encoding="utf-8")
    bridge_root = tmp_path / "bridge_root"
    bridge_root.mkdir()
    return plugin_dir, bridge_root


def _make_effective_config(bridge_root: Path, textractor_path: Path) -> dict[str, object]:
    return {
        "galgame": {
            "bridge_root": str(bridge_root),
            "active_poll_interval_seconds": 0.1,
            "idle_poll_interval_seconds": 0.1,
            "stale_after_seconds": 0.2,
            "history_events_limit": 500,
            "history_lines_limit": 200,
            "history_choices_limit": 50,
            "dedupe_window_limit": 64,
            "warmup_replay_bytes_limit": 65536,
            "warmup_replay_events_limit": 50,
            "default_mode": "companion",
            "push_notifications": True,
        },
        "llm": {
            "llm_call_timeout_seconds": 15,
            "llm_max_in_flight": 2,
            "llm_request_cache_ttl_seconds": 2,
            "target_entry_ref": "",
        },
        "memory_reader": {
            "enabled": True,
            "textractor_path": str(textractor_path),
            "auto_detect": True,
            "poll_interval_seconds": 1,
        },
    }


@pytest.mark.asyncio
async def test_memory_reader_bridge_tick_fallback_and_bridge_sdk_takeover(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    textractor_path = tmp_path / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root, textractor_path))
    plugin = GalgameBridgePlugin(ctx)
    startup = await plugin.startup()
    assert isinstance(startup, Ok)

    clock = {"now": 1710000000.0}
    handle = _FakeTextractorHandle(["[4242:100:0:0] 雪乃：Memory Reader 已接管。"])

    async def _process_factory(path: str):
        del path
        return handle

    plugin._memory_reader_manager = MemoryReaderManager(
        logger=plugin.logger,
        config=plugin._cfg,
        process_factory=_process_factory,
        process_scanner=lambda: [
            DetectedGameProcess(
                pid=4242,
                name="RenPy Demo.exe",
                create_time=1709999999.0,
                engine="renpy",
            )
        ],
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        writer=MemoryReaderBridgeWriter(
            bridge_root=bridge_root,
            time_fn=lambda: clock["now"],
        ),
    )

    await plugin._poll_bridge(force=True)
    memory_reader_status = await plugin.galgame_get_status()
    memory_reader_snapshot = await plugin.galgame_get_snapshot()

    assert isinstance(memory_reader_status, Ok)
    assert isinstance(memory_reader_snapshot, Ok)
    assert memory_reader_status.value["active_data_source"] == "memory_reader"
    assert memory_reader_snapshot.value["snapshot"]["text"] == "Memory Reader 已接管。"
    assert handle.writes == ["attach -P4242\n"]

    game_dir = bridge_root / "demo.bridge"
    game_dir.mkdir(parents=True, exist_ok=True)
    _write_session(
        game_dir / "session.json",
        _session(
            game_id="demo.bridge",
            session_id="sdk-sess",
            last_seq=3,
            state=_session_state(
                speaker="桥接",
                text="Bridge SDK 已抢占。",
                scene_id="sdk-scene",
                line_id="sdk-line",
            ),
        ),
    )
    (game_dir / "events.jsonl").write_text("", encoding="utf-8")

    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)
    bridge_sdk_status = await plugin.galgame_get_status()

    assert isinstance(bridge_sdk_status, Ok)
    assert bridge_sdk_status.value["active_data_source"] == "bridge_sdk"
    assert bridge_sdk_status.value["active_session_id"] == "sdk-sess"
    assert bridge_sdk_status.value["memory_reader_runtime"]["detail"] == "bridge_sdk_available"
    assert handle.terminated is True

    await plugin.shutdown()
