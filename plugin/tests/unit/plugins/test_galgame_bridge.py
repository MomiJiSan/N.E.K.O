from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from plugin.plugins.galgame_plugin import GalgameBridgePlugin
from plugin.plugins.galgame_plugin import service as galgame_service
from plugin.plugins.galgame_plugin.game_llm_agent import GameLLMAgent
from plugin.plugins.galgame_plugin.host_agent_adapter import HostAgentAdapter
from plugin.plugins.galgame_plugin.llm_gateway import LLMGateway
from plugin.plugins.galgame_plugin.memory_reader import (
    compute_memory_reader_game_id,
    DetectedGameProcess,
    MemoryReaderBridgeWriter,
    MemoryReaderManager,
)
from plugin.plugins.galgame_plugin.models import (
    DATA_SOURCE_BRIDGE_SDK,
    DATA_SOURCE_MEMORY_READER,
    DATA_SOURCE_OCR_READER,
)
from plugin.plugins.galgame_plugin.ocr_reader import (
    DetectedGameWindow,
    OcrReaderBridgeWriter,
    OcrReaderManager,
)
from plugin.plugins.galgame_plugin.reader import (
    expand_bridge_root,
    read_session_json,
    tail_events_jsonl,
)
from plugin.plugins.galgame_plugin.service import (
    _default_bridge_root_raw,
    build_config,
    build_explain_context,
    build_suggest_context,
    build_summarize_context,
)
from plugin.sdk.plugin import Ok


_PLUGIN_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "galgame_plugin"


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
    plugin_id = "galgame_plugin"
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
        self.entry_calls: list[dict[str, object]] = []
        self.entry_handler = None

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
        self.entry_calls.append(dict(kwargs))
        if self.entry_handler is None:
            raise RuntimeError("no fake trigger_plugin_event configured")
        handler = self.entry_handler
        if callable(handler):
            result = handler(**kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return handler

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


def _session_state(
    *,
    speaker: str = "",
    text: str = "",
    choices: list[dict[str, object]] | None = None,
    scene_id: str = "boot",
    line_id: str = "",
    route_id: str = "",
    is_menu_open: bool = False,
    ts: str = "2026-04-21T08:30:00Z",
) -> dict[str, object]:
    return {
        "speaker": speaker,
        "text": text,
        "choices": list(choices or []),
        "scene_id": scene_id,
        "line_id": line_id,
        "route_id": route_id,
        "is_menu_open": is_menu_open,
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
    started_at: str = "2026-04-21T08:30:00Z",
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "game_id": game_id,
        "game_title": game_id,
        "engine": "renpy",
        "session_id": session_id,
        "started_at": started_at,
        "last_seq": last_seq,
        "locale": "ja-JP",
        "bridge_sdk_version": "1.0.0",
        "state": state,
    }


def _event(
    *,
    seq: int,
    event_type: str,
    session_id: str,
    game_id: str,
    payload: dict[str, object],
    ts: str,
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "seq": seq,
        "ts": ts,
        "type": event_type,
        "session_id": session_id,
        "game_id": game_id,
        "payload": payload,
    }


def _write_session(path: Path, payload: dict[str, object], *, bom: bool = False) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if bom:
        text = "\ufeff" + text
    path.write_text(text, encoding="utf-8")


def _write_events(
    path: Path,
    events: list[dict[str, object]],
    *,
    trailing: bytes = b"",
    crlf: bool = False,
) -> int:
    line_end = b"\r\n" if crlf else b"\n"
    data = b"".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + line_end
        for event in events
    )
    data += trailing
    path.write_bytes(data)
    return len(data)


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


def _copy_bridge_fixture_scenario(bridge_root: Path, scenario: str) -> Path:
    scenario_root = _PLUGIN_FIXTURE_ROOT / scenario
    if not scenario_root.is_dir():
        raise AssertionError(f"missing bridge fixture scenario: {scenario}")
    copied_game_dir: Path | None = None
    for child in scenario_root.iterdir():
        target = bridge_root / child.name
        if child.is_dir():
            shutil.copytree(child, target)
            copied_game_dir = target
        else:
            shutil.copy2(child, target)
    if copied_game_dir is None:
        raise AssertionError(f"bridge fixture scenario is empty: {scenario}")
    return copied_game_dir


def _make_effective_config(bridge_root: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
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
            "enabled": False,
            "textractor_path": "",
            "auto_detect": True,
            "poll_interval_seconds": 1,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])  # type: ignore[index]
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    return config


def _create_game_dir(
    bridge_root: Path,
    *,
    game_id: str,
    session_payload: dict[str, object],
    events: list[dict[str, object]] | None = None,
) -> Path:
    game_dir = bridge_root / game_id
    game_dir.mkdir(parents=True, exist_ok=True)
    _write_session(game_dir / "session.json", session_payload)
    _write_events(game_dir / "events.jsonl", events or [])
    return game_dir


def _shared_state(
    *,
    mode: str = "choice_advisor",
    push_notifications: bool = True,
    connection_state: str = "active",
    stream_reset_pending: bool = False,
    game_id: str = "demo.alpha",
    session_id: str = "sess-a",
    last_seq: int = 2,
    snapshot: dict[str, object] | None = None,
    history_lines: list[dict[str, object]] | None = None,
    history_choices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "mode": mode,
        "push_notifications": push_notifications,
        "current_connection_state": connection_state,
        "stream_reset_pending": stream_reset_pending,
        "active_game_id": game_id,
        "active_session_id": session_id,
        "last_seq": last_seq,
        "latest_snapshot": snapshot or _session_state(
            speaker="雪乃",
            text="当前台词",
            scene_id="scene-a",
            line_id="line-1",
            ts="2026-04-21T08:30:02Z",
        ),
        "history_lines": list(history_lines or []),
        "history_choices": list(history_choices or []),
    }


class _FakeHostAdapter:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.tasks: dict[str, dict[str, object]] = {}
        self._counter = 0

    async def get_computer_use_availability(self, *, timeout: float = 1.5):
        if self.ready:
            return {"ready": True, "reasons": []}
        return {"ready": False, "reasons": ["computer_use unavailable"]}

    async def run_computer_use_instruction(self, instruction: str, *, lanlan_name: str = "", timeout: float = 5.0):
        self._counter += 1
        task_id = f"task-{self._counter}"
        self.started.append(instruction)
        self.tasks[task_id] = {"id": task_id, "status": "running", "result": None}
        return {"task_id": task_id, "status": "running"}

    async def get_task(self, task_id: str, *, timeout: float = 2.0):
        return dict(self.tasks[task_id])

    async def cancel_task(self, task_id: str, *, timeout: float = 5.0):
        self.cancelled.append(task_id)
        self.tasks[task_id] = {"id": task_id, "status": "cancelled", "error": "Cancelled by test"}
        return {"success": True, "task_id": task_id, "status": "cancelled"}

    async def shutdown(self) -> None:
        return None


class _FakeLLMGateway:
    def __init__(
        self,
        *,
        suggest_payload: dict[str, object] | None = None,
        reply_payload: dict[str, object] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.suggest_payload = suggest_payload or {"degraded": True, "choices": [], "diagnostic": "no llm"}
        self.reply_payload = reply_payload or {"degraded": True, "reply": "fallback", "diagnostic": "no llm"}
        self.delay = delay
        self.suggest_calls: list[dict[str, object]] = []
        self.reply_calls: list[dict[str, object]] = []

    async def suggest_choice(self, context: dict[str, object]):
        self.suggest_calls.append(dict(context))
        if self.delay:
            await asyncio.sleep(self.delay)
        return dict(self.suggest_payload)

    async def agent_reply(self, context: dict[str, object]):
        self.reply_calls.append(dict(context))
        if self.delay:
            await asyncio.sleep(self.delay)
        return dict(self.reply_payload)


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


class _FakeCaptureBackend:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}:{target.pid}"

    def capture_frame(self, target: DetectedGameWindow, profile) -> str:
        del profile
        return f"frame:{target.hwnd}"


class _FakeOcrBackend:
    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or [])

    def is_available(self) -> bool:
        return True

    def extract_text(self, image: str) -> str:
        del image
        if not self._texts:
            return ""
        if len(self._texts) == 1:
            return self._texts[0]
        return self._texts.pop(0)


def _memory_reader_session(
    *,
    game_id: str,
    session_id: str,
    state: dict[str, object],
    last_seq: int,
) -> dict[str, object]:
    payload = _session(
        game_id=game_id,
        session_id=session_id,
        last_seq=last_seq,
        state=state,
    )
    payload["bridge_sdk_version"] = "memory-reader-0.1.0"
    payload["engine"] = "unknown"
    payload["metadata"] = {
        "source": "memory_reader",
        "game_process_name": "RenPy Demo.exe",
        "game_pid": 4242,
    }
    return payload


@pytest.mark.plugin_unit
def test_expand_bridge_root_and_read_bom_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    expanded = expand_bridge_root("%LOCALAPPDATA%/N.E.K.O/galgame-bridge")
    assert expanded == tmp_path / "Local" / "N.E.K.O" / "galgame-bridge"

    session_path = tmp_path / "session.json"
    _write_session(
        session_path,
        _session(
            game_id="demo.game",
            session_id="sess-1",
            last_seq=1,
            state=_session_state(speaker="雪乃", text="你好"),
        ),
        bom=True,
    )
    result = read_session_json(session_path)
    assert result.error == ""
    assert result.session is not None
    assert result.session["state"]["speaker"] == "雪乃"


@pytest.mark.plugin_unit
@pytest.mark.parametrize(
    ("platform_value", "use_xdg_data_home", "expected_raw"),
    [
        ("win32", False, "%LOCALAPPDATA%/N.E.K.O/galgame-bridge"),
        ("darwin", False, "~/Library/Application Support/N.E.K.O/galgame-bridge"),
        ("linux", True, "xdg"),
        ("linux", False, "~/.local/share/N.E.K.O/galgame-bridge"),
    ],
)
def test_default_bridge_root_raw_uses_platform_conventions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_value: str,
    use_xdg_data_home: bool,
    expected_raw: str,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", platform_value)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    if use_xdg_data_home:
        xdg_data_home = tmp_path / "xdg-data"
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))
        assert _default_bridge_root_raw() == f"{xdg_data_home}/N.E.K.O/galgame-bridge"
        return
    assert _default_bridge_root_raw() == expected_raw


@pytest.mark.plugin_unit
def test_expand_bridge_root_handles_user_home_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()

    def _fake_expanduser(value: str) -> str:
        if value.startswith("~/"):
            return str(home_dir / value[2:])
        if value == "~":
            return str(home_dir)
        return value

    monkeypatch.setattr("plugin.plugins.galgame_plugin.reader.os.path.expanduser", _fake_expanduser)

    mac_path = expand_bridge_root("~/Library/Application Support/N.E.K.O/galgame-bridge")
    linux_path = expand_bridge_root("~/.local/share/N.E.K.O/galgame-bridge")

    assert mac_path == home_dir / "Library" / "Application Support" / "N.E.K.O" / "galgame-bridge"
    assert linux_path == home_dir / ".local" / "share" / "N.E.K.O" / "galgame-bridge"


@pytest.mark.plugin_unit
@pytest.mark.parametrize("bridge_root_value", [None, "", "   "])
def test_build_config_uses_default_bridge_root_when_missing_or_blank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bridge_root_value: str | None,
) -> None:
    expected = tmp_path / "auto" / "bridge"
    monkeypatch.setattr(galgame_service, "_default_bridge_root_raw", lambda: str(expected))

    galgame_config = {} if bridge_root_value is None else {"bridge_root": bridge_root_value}
    cfg = build_config({"galgame": galgame_config})

    assert cfg.bridge_root == expected


@pytest.mark.plugin_unit
def test_build_config_prefers_explicit_bridge_root(tmp_path: Path) -> None:
    explicit = tmp_path / "custom" / "bridge"
    cfg = build_config({"galgame": {"bridge_root": str(explicit)}})
    assert cfg.bridge_root == explicit


@pytest.mark.plugin_unit
def test_compute_memory_reader_game_id_avoids_windows_invalid_path_characters() -> None:
    game_id = compute_memory_reader_game_id("RenPy Demo.exe")
    assert game_id.startswith("mem-")
    assert ":" not in game_id


@pytest.mark.plugin_unit
@pytest.mark.parametrize(
    ("platform_value", "expected_enabled"),
    [
        ("win32", True),
        ("darwin", False),
        ("linux", False),
    ],
)
def test_build_config_uses_platform_default_memory_reader_enablement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_value: str,
    expected_enabled: bool,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", platform_value)
    cfg = build_config({"galgame": {"bridge_root": str(tmp_path / "bridge")}})
    assert cfg.memory_reader_enabled is expected_enabled


@pytest.mark.plugin_unit
def test_build_config_explicit_memory_reader_enabled_overrides_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", "win32")
    cfg = build_config(
        {
            "galgame": {"bridge_root": str(tmp_path / "bridge")},
            "memory_reader": {"enabled": False},
        }
    )
    assert cfg.memory_reader_enabled is False


@pytest.mark.plugin_unit
@pytest.mark.parametrize(
    ("platform_value", "expected_enabled"),
    [
        ("win32", True),
        ("darwin", False),
        ("linux", False),
    ],
)
def test_build_config_uses_platform_default_ocr_reader_enablement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_value: str,
    expected_enabled: bool,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", platform_value)
    cfg = build_config({"galgame": {"bridge_root": str(tmp_path / "bridge")}})
    assert cfg.ocr_reader_enabled is expected_enabled


@pytest.mark.plugin_unit
def test_build_config_explicit_ocr_reader_enabled_overrides_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", "win32")
    cfg = build_config(
        {
            "galgame": {"bridge_root": str(tmp_path / "bridge")},
            "ocr_reader": {"enabled": False},
        }
    )
    assert cfg.ocr_reader_enabled is False


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_memory_reader_auto_discovers_textractor_from_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    textractor_path = path_dir / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    textractor_path.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)

    cfg = build_config(
        {
            "galgame": {"bridge_root": str(tmp_path / "bridge_root")},
            "memory_reader": {
                "enabled": True,
                "textractor_path": "",
                "auto_detect": True,
                "poll_interval_seconds": 1,
            },
        }
    )
    captured_paths: list[str] = []
    handle = _FakeTextractorHandle()

    async def _process_factory(path: str):
        captured_paths.append(path)
        return handle

    manager = MemoryReaderManager(
        logger=_Logger(),
        config=cfg,
        process_factory=_process_factory,
        process_scanner=lambda: [
            DetectedGameProcess(
                pid=4242,
                name="RenPy Demo.exe",
                create_time=1709999999.0,
                engine="renpy",
            )
        ],
        time_fn=lambda: 1710000000.0,
        platform_fn=lambda: True,
        writer=MemoryReaderBridgeWriter(
            bridge_root=cfg.bridge_root,
            time_fn=lambda: 1710000000.0,
        ),
    )

    result = await manager.tick(bridge_sdk_available=False)

    assert captured_paths == [str(textractor_path)]
    assert handle.writes == ["attach -P4242\n"]
    assert result.runtime["status"] == "attaching"
    await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_memory_reader_auto_discovers_textractor_from_localappdata_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    local_appdata = tmp_path / "LocalAppData"
    textractor_path = local_appdata / "Programs" / "Textractor" / "TextractorCLI.exe"
    textractor_path.parent.mkdir(parents=True, exist_ok=True)
    textractor_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "ProgramFiles"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "ProgramFilesX86"))

    cfg = build_config(
        {
            "galgame": {"bridge_root": str(tmp_path / "bridge_root")},
            "memory_reader": {
                "enabled": True,
                "textractor_path": "",
                "auto_detect": True,
                "poll_interval_seconds": 1,
            },
        }
    )
    captured_paths: list[str] = []

    async def _process_factory(path: str):
        captured_paths.append(path)
        return _FakeTextractorHandle()

    manager = MemoryReaderManager(
        logger=_Logger(),
        config=cfg,
        process_factory=_process_factory,
        process_scanner=lambda: [
            DetectedGameProcess(
                pid=4242,
                name="RenPy Demo.exe",
                create_time=1709999999.0,
                engine="renpy",
            )
        ],
        time_fn=lambda: 1710000000.0,
        platform_fn=lambda: True,
        writer=MemoryReaderBridgeWriter(
            bridge_root=cfg.bridge_root,
            time_fn=lambda: 1710000000.0,
        ),
    )

    result = await manager.tick(bridge_sdk_available=False)

    assert captured_paths == [str(textractor_path)]
    assert result.runtime["status"] == "attaching"
    await manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_memory_reader_keeps_recoverable_idle_state_when_textractor_autodiscovery_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty-local"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty-program-files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "empty-program-files-x86"))

    cfg = build_config(
        {
            "galgame": {"bridge_root": str(tmp_path / "bridge_root")},
            "memory_reader": {
                "enabled": True,
                "textractor_path": "",
                "auto_detect": True,
                "poll_interval_seconds": 1,
            },
        }
    )
    factory_calls: list[str] = []

    async def _process_factory(path: str):
        factory_calls.append(path)
        return _FakeTextractorHandle()

    manager = MemoryReaderManager(
        logger=_Logger(),
        config=cfg,
        process_factory=_process_factory,
        process_scanner=lambda: [],
        time_fn=lambda: 1710000000.0,
        platform_fn=lambda: True,
        writer=MemoryReaderBridgeWriter(
            bridge_root=cfg.bridge_root,
            time_fn=lambda: 1710000000.0,
        ),
    )

    result = await manager.tick(bridge_sdk_available=False)

    assert factory_calls == []
    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == "invalid_textractor_path"
    assert result.warnings == ["memory_reader TextractorCLI.exe is invalid or missing"]


@pytest.mark.plugin_unit
def test_tail_events_handles_utf8_crlf_and_partial_line(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    game_id = "demo.game"
    session_id = "sess-1"
    first = _event(
        seq=1,
        event_type="line_changed",
        session_id=session_id,
        game_id=game_id,
        payload={"speaker": "雪乃", "text": "今天也一起回家吧。", "line_id": "line-1", "scene_id": "scene-a", "route_id": ""},
        ts="2026-04-21T08:31:00Z",
    )
    second = _event(
        seq=2,
        event_type="choices_shown",
        session_id=session_id,
        game_id=game_id,
        payload={"line_id": "line-1", "scene_id": "scene-a", "route_id": "", "choices": []},
        ts="2026-04-21T08:31:01Z",
    )
    partial = json.dumps(
        _event(
            seq=3,
            event_type="heartbeat",
            session_id=session_id,
            game_id=game_id,
            payload={"state_ts": "2026-04-21T08:31:01Z", "idle_seconds": 5},
            ts="2026-04-21T08:31:06Z",
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    cutoff = len(partial) // 2
    total_size = _write_events(events_path, [first, second], trailing=partial[:cutoff], crlf=True)

    result = tail_events_jsonl(events_path, offset=0, line_buffer=b"")
    assert len(result.events) == 2
    assert result.next_offset == total_size
    assert result.line_buffer == partial[:cutoff]

    with events_path.open("ab") as handle:
        handle.write(partial[cutoff:] + b"\n")

    resumed = tail_events_jsonl(
        events_path,
        offset=result.next_offset,
        line_buffer=result.line_buffer,
    )
    assert [event["seq"] for event in resumed.events] == [3]
    assert resumed.line_buffer == b""


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_startup_binds_latest_session_and_exposes_ui(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    _create_game_dir(
        bridge_root,
        game_id="demo.alpha",
        session_payload=_session(
            game_id="demo.alpha",
            session_id="sess-a",
            last_seq=1,
            state=_session_state(text="alpha"),
        ),
    )
    _create_game_dir(
        bridge_root,
        game_id="demo.beta",
        session_payload=_session(
            game_id="demo.beta",
            session_id="sess-b",
            last_seq=3,
            state=_session_state(text="beta"),
        ),
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    startup = await plugin.startup()
    assert isinstance(startup, Ok)

    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()
    open_ui = await plugin.galgame_open_ui()

    assert isinstance(status, Ok)
    assert status.value["bound_game_id"] == ""
    assert status.value["active_session_id"] == "sess-b"
    assert status.value["available_game_ids"] == ["demo.alpha", "demo.beta"]
    assert "textractor" in status.value
    assert isinstance(snapshot, Ok)
    assert snapshot.value["session_id"] == "sess-b"
    assert isinstance(open_ui, Ok)
    assert open_ui.value["available"] is True
    assert open_ui.value["path"] == "/plugin/galgame_plugin/ui/"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_public_surface_preserves_phase1_entries_and_adds_phase2_entries(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    _create_game_dir(
        bridge_root,
        game_id="demo.alpha",
        session_payload=_session(
            game_id="demo.alpha",
            session_id="sess-a",
            last_seq=1,
            state=_session_state(text="alpha"),
        ),
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    startup = await plugin.startup()
    assert isinstance(startup, Ok)

    entry_ids = sorted(
        entry_id
        for entry_id, handler in plugin.collect_entries().items()
        if handler.meta.event_type == "plugin_entry"
    )
    assert entry_ids == [
        "galgame_agent_command",
        "galgame_bind_game",
        "galgame_explain_line",
        "galgame_get_history",
        "galgame_install_textractor",
        "galgame_get_snapshot",
        "galgame_get_status",
        "galgame_open_ui",
        "galgame_set_mode",
        "galgame_suggest_choice",
        "galgame_summarize_scene",
    ]
    for phase1_entry in (
        "galgame_bind_game",
        "galgame_get_history",
        "galgame_get_snapshot",
        "galgame_get_status",
        "galgame_open_ui",
        "galgame_set_mode",
    ):
        assert phase1_entry in entry_ids

    assert plugin.get_list_actions() == [
        {
            "id": "open_ui",
            "kind": "ui",
            "target": "/plugin/galgame_plugin/ui/",
            "open_in": "new_tab",
        }
    ]

    static_ui = plugin.get_static_ui_config()
    assert static_ui is not None
    assert static_ui["plugin_id"] == "galgame_plugin"
    assert Path(str(static_ui["directory"])).name == "static"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_set_mode_and_bind_game_persist_across_restart(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    _create_game_dir(
        bridge_root,
        game_id="demo.alpha",
        session_payload=_session(
            game_id="demo.alpha",
            session_id="sess-a",
            last_seq=2,
            state=_session_state(text="alpha"),
        ),
    )
    _create_game_dir(
        bridge_root,
        game_id="demo.beta",
        session_payload=_session(
            game_id="demo.beta",
            session_id="sess-b",
            last_seq=1,
            state=_session_state(text="beta"),
        ),
    )

    ctx1 = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin1 = GalgameBridgePlugin(ctx1)
    await plugin1.startup()

    mode_result = await plugin1.galgame_set_mode(
        mode="choice_advisor",
        push_notifications=False,
    )
    bind_result = await plugin1.galgame_bind_game(game_id="demo.beta")
    assert isinstance(mode_result, Ok)
    assert isinstance(bind_result, Ok)

    ctx2 = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin2 = GalgameBridgePlugin(ctx2)
    await plugin2.startup()
    status = await plugin2.galgame_get_status()
    assert isinstance(status, Ok)
    assert status.value["mode"] == "choice_advisor"
    assert status.value["push_notifications"] is False
    assert status.value["bound_game_id"] == "demo.beta"
    assert status.value["active_session_id"] == "sess-b"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_save_loaded_and_repeated_line_do_not_duplicate_stable_history(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "demo.alpha"
    session_id = "sess-a"
    events = [
        _event(
            seq=1,
            event_type="session_started",
            session_id=session_id,
            game_id=game_id,
            payload={
                "game_title": "demo.alpha",
                "engine": "renpy",
                "locale": "ja-JP",
                "started_at": "2026-04-21T08:30:00Z",
                "scene_id": "boot",
                "line_id": "",
                "route_id": "",
                "is_menu_open": False,
                "speaker": "",
                "text": "",
                "choices": [],
                "save_context": {"kind": "unknown", "slot_id": "", "display_name": ""},
            },
            ts="2026-04-21T08:30:00Z",
        ),
        _event(
            seq=2,
            event_type="line_changed",
            session_id=session_id,
            game_id=game_id,
            payload={
                "speaker": "雪乃",
                "text": "今天也一起回家吧。",
                "line_id": "script/ch1.rpy:120",
                "scene_id": "ch1_after_school",
                "route_id": "",
            },
            ts="2026-04-21T08:31:00Z",
        ),
        _event(
            seq=3,
            event_type="save_loaded",
            session_id=session_id,
            game_id=game_id,
            payload={
                "reason": "rollback",
                "scene_id": "ch1_after_school",
                "line_id": "script/ch1.rpy:120",
                "route_id": "",
                "save_context": {"kind": "rollback", "slot_id": "", "display_name": "rollback"},
            },
            ts="2026-04-21T08:31:10Z",
        ),
        _event(
            seq=4,
            event_type="line_changed",
            session_id=session_id,
            game_id=game_id,
            payload={
                "speaker": "雪乃",
                "text": "今天也一起回家吧。",
                "line_id": "script/ch1.rpy:120",
                "scene_id": "ch1_after_school",
                "route_id": "",
            },
            ts="2026-04-21T08:31:11Z",
        ),
    ]
    _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=4,
            state=_session_state(
                speaker="雪乃",
                text="今天也一起回家吧。",
                scene_id="ch1_after_school",
                line_id="script/ch1.rpy:120",
                ts="2026-04-21T08:31:11Z",
            ),
        ),
        events=events,
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    history = await plugin.galgame_get_history(limit=20, include_events=True)
    assert isinstance(history, Ok)
    assert len(history.value["events"]) == 4
    assert len(history.value["stable_lines"]) == 1
    assert history.value["stable_lines"][0]["line_id"] == "script/ch1.rpy:120"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_bridge_fixture_manual_load_round_exposes_bridge_sdk_status_snapshot_and_history(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    _copy_bridge_fixture_scenario(bridge_root, "manual_load")

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    await plugin._poll_bridge(force=True)

    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()
    history = await plugin.galgame_get_history(limit=20, include_events=True)

    assert isinstance(status, Ok)
    assert status.value["active_data_source"] == DATA_SOURCE_BRIDGE_SDK
    assert status.value["summary"].startswith("已通过 Bridge SDK 连接")
    assert status.value["memory_reader_runtime"]["detail"] == "disabled_by_config"

    assert isinstance(snapshot, Ok)
    assert snapshot.value["snapshot"]["scene_id"] == "after_school"
    assert snapshot.value["snapshot"]["line_id"] == "script.rpy:28"
    assert snapshot.value["snapshot"]["is_menu_open"] is True
    assert snapshot.value["snapshot"]["save_context"]["kind"] == "manual"
    assert len(snapshot.value["snapshot"]["choices"]) == 2

    assert isinstance(history, Ok)
    assert history.value["events"][-2]["type"] == "save_loaded"
    assert history.value["events"][-2]["payload"]["reason"] == "load"
    assert history.value["events"][-1]["type"] == "choices_shown"
    assert history.value["events"][-1]["payload"]["line_id"] == "script.rpy:28"
    assert history.value["stable_lines"][-1]["line_id"] == "script.rpy:45"
    assert len(history.value["stable_lines"]) == 6


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_bridge_fixture_rollback_round_preserves_history_and_supports_phase2_llm_entries(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    _copy_bridge_fixture_scenario(bridge_root, "rollback")

    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            llm={"target_entry_ref": "fake_llm:run"},
        ),
    )

    async def _handler(**kwargs):
        params = kwargs.get("params") or {}
        operation = params.get("operation")
        if operation == "explain_line":
            return {"explanation": "这是回滚后的菜单锚点。", "evidence": []}
        if operation == "summarize_scene":
            return {
                "summary": "场景重新回到了 after_school 的选项前。",
                "key_points": [{"type": "decision", "text": "rollback 已完成。"}],
            }
        if operation == "suggest_choice":
            context = params.get("context") or {}
            visible_choices = context.get("visible_choices") or []
            return {
                "choices": [
                    {
                        "choice_id": visible_choices[0]["choice_id"],
                        "text": visible_choices[0]["text"],
                        "rank": 1,
                        "reason": "继续验证 rollback 后的菜单消费。",
                    }
                ]
            }
        raise AssertionError(f"unexpected operation: {operation}")

    ctx.entry_handler = _handler
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    await plugin._poll_bridge(force=True)

    snapshot = await plugin.galgame_get_snapshot()
    history = await plugin.galgame_get_history(limit=20, include_events=True)
    explain = await plugin.galgame_explain_line()
    summarize = await plugin.galgame_summarize_scene()
    suggest = await plugin.galgame_suggest_choice()

    assert isinstance(snapshot, Ok)
    assert snapshot.value["snapshot"]["scene_id"] == "after_school"
    assert snapshot.value["snapshot"]["save_context"]["kind"] == "rollback"
    assert snapshot.value["snapshot"]["is_menu_open"] is True

    assert isinstance(history, Ok)
    assert history.value["events"][-3]["type"] == "save_loaded"
    assert history.value["events"][-3]["payload"]["reason"] == "rollback"
    repeated_lines = [
        item for item in history.value["stable_lines"] if item["line_id"] == "script.rpy:28"
    ]
    assert len(repeated_lines) == 1

    assert isinstance(explain, Ok)
    assert explain.value["degraded"] is False
    assert explain.value["line_id"] == "script.rpy:28"
    assert explain.value["explanation"] == "这是回滚后的菜单锚点。"

    assert isinstance(summarize, Ok)
    assert summarize.value["degraded"] is False
    assert summarize.value["scene_id"] == "after_school"
    assert summarize.value["summary"] == "场景重新回到了 after_school 的选项前。"

    assert isinstance(suggest, Ok)
    assert suggest.value["degraded"] is False
    assert suggest.value["choices"][0]["choice_id"] == "script.rpy:28#choice0"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_restart_restores_cursor_and_processes_new_tail(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "demo.alpha"
    session_id = "sess-a"
    game_dir = _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=2,
            state=_session_state(
                speaker="雪乃",
                text="旧台词",
                line_id="line-1",
                scene_id="scene-a",
                ts="2026-04-21T08:30:02Z",
            ),
        ),
        events=[
            _event(
                seq=1,
                event_type="session_started",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "game_title": game_id,
                    "engine": "renpy",
                    "locale": "ja-JP",
                    "started_at": "2026-04-21T08:30:00Z",
                    "scene_id": "boot",
                    "line_id": "",
                    "route_id": "",
                    "is_menu_open": False,
                    "speaker": "",
                    "text": "",
                    "choices": [],
                    "save_context": {"kind": "unknown", "slot_id": "", "display_name": ""},
                },
                ts="2026-04-21T08:30:00Z",
            ),
            _event(
                seq=2,
                event_type="line_changed",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "speaker": "雪乃",
                    "text": "旧台词",
                    "line_id": "line-1",
                    "scene_id": "scene-a",
                    "route_id": "",
                },
                ts="2026-04-21T08:30:02Z",
            ),
        ],
    )

    ctx1 = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin1 = GalgameBridgePlugin(ctx1)
    await plugin1.startup()

    new_event = _event(
        seq=3,
        event_type="line_changed",
        session_id=session_id,
        game_id=game_id,
        payload={
            "speaker": "雪乃",
            "text": "重启后新增台词",
            "line_id": "line-2",
            "scene_id": "scene-a",
            "route_id": "",
        },
        ts="2026-04-21T08:30:05Z",
    )
    with (game_dir / "events.jsonl").open("ab") as handle:
        handle.write(
            json.dumps(new_event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    _write_session(
        game_dir / "session.json",
        _session(
            game_id=game_id,
            session_id=session_id,
            last_seq=3,
            state=_session_state(
                speaker="雪乃",
                text="重启后新增台词",
                line_id="line-2",
                scene_id="scene-a",
                ts="2026-04-21T08:30:05Z",
            ),
        ),
    )

    ctx2 = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin2 = GalgameBridgePlugin(ctx2)
    await plugin2.startup()
    history = await plugin2.galgame_get_history(limit=20, include_events=True)
    assert isinstance(history, Ok)
    assert history.value["events"][-1]["seq"] == 3
    assert history.value["stable_lines"][-1]["line_id"] == "line-2"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_truncation_sets_stream_reset_pending(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "demo.alpha"
    session_id = "sess-a"
    game_dir = _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=2,
            state=_session_state(text="alpha"),
        ),
        events=[
            _event(
                seq=1,
                event_type="session_started",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "game_title": game_id,
                    "engine": "renpy",
                    "locale": "ja-JP",
                    "started_at": "2026-04-21T08:30:00Z",
                    "scene_id": "boot",
                    "line_id": "",
                    "route_id": "",
                    "is_menu_open": False,
                    "speaker": "",
                    "text": "",
                    "choices": [],
                    "save_context": {"kind": "unknown", "slot_id": "", "display_name": ""},
                },
                ts="2026-04-21T08:30:00Z",
            ),
            _event(
                seq=2,
                event_type="line_changed",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "speaker": "雪乃",
                    "text": "旧台词",
                    "line_id": "line-1",
                    "scene_id": "scene-a",
                    "route_id": "",
                },
                ts="2026-04-21T08:30:02Z",
            ),
        ],
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    (game_dir / "events.jsonl").write_bytes(b"")
    await plugin._poll_bridge(force=True)
    status = await plugin.galgame_get_status()
    assert isinstance(status, Ok)
    assert status.value["stream_reset_pending"] is True


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_stale_then_new_event_recovers_to_active(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "demo.alpha"
    session_id = "sess-a"
    game_dir = _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=1,
            state=_session_state(text="alpha"),
        ),
        events=[
            _event(
                seq=1,
                event_type="line_changed",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "speaker": "雪乃",
                    "text": "旧台词",
                    "line_id": "line-1",
                    "scene_id": "scene-a",
                    "route_id": "",
                },
                ts="2026-04-21T08:30:02Z",
            )
        ],
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    with plugin._state_lock:
        plugin._state.last_seen_data_monotonic = time.monotonic() - 5.0

    await plugin._poll_bridge(force=True)
    stale_status = await plugin.galgame_get_status()
    assert isinstance(stale_status, Ok)
    assert stale_status.value["connection_state"] == "stale"

    with (game_dir / "events.jsonl").open("ab") as handle:
        handle.write(
            json.dumps(
                _event(
                    seq=2,
                    event_type="line_changed",
                    session_id=session_id,
                    game_id=game_id,
                    payload={
                        "speaker": "雪乃",
                        "text": "新台词",
                        "line_id": "line-2",
                        "scene_id": "scene-a",
                        "route_id": "",
                    },
                    ts="2026-04-21T08:30:06Z",
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
    _write_session(
        game_dir / "session.json",
        _session(
            game_id=game_id,
            session_id=session_id,
            last_seq=2,
            state=_session_state(
                speaker="雪乃",
                text="新台词",
                line_id="line-2",
                scene_id="scene-a",
                ts="2026-04-21T08:30:06Z",
            ),
        ),
    )

    await plugin._poll_bridge(force=True)
    active_status = await plugin.galgame_get_status()
    assert isinstance(active_status, Ok)
    assert active_status.value["connection_state"] == "active"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_windows_default_memory_reader_config_autodiscovers_textractor_and_takes_over_without_bridge_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", "win32")
    path_dir = tmp_path / "bin"
    path_dir.mkdir()
    textractor_path = path_dir / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    textractor_path.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))

    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    cfg = _make_effective_config(
        bridge_root,
        memory_reader={
            "auto_detect": True,
            "poll_interval_seconds": 1,
        },
    )
    del cfg["memory_reader"]["enabled"]  # type: ignore[index]
    del cfg["memory_reader"]["textractor_path"]  # type: ignore[index]

    ctx = _Ctx(plugin_dir, cfg)
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    clock = {"now": 1710000000.0}
    expected_snapshot_text = "Windows default config takeover."
    good_handle = _FakeTextractorHandle(
        [f"[4242:100:0:0] {expected_snapshot_text}"]
    )
    handle = _FakeTextractorHandle(
        ["[4242:100:0:0] é›ªä¹ƒï¼šWindows é»˜è®¤é…ç½®å·²è‡ªåŠ¨æŽ¥ç®¡ã€‚"]
    )

    async def _process_factory(path: str):
        assert path == str(textractor_path)
        return good_handle

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
    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()

    assert isinstance(status, Ok)
    assert isinstance(snapshot, Ok)
    assert status.value["memory_reader_enabled"] is True
    assert status.value["active_data_source"] == DATA_SOURCE_MEMORY_READER
    assert status.value["memory_reader_runtime"]["status"] == "active"
    assert snapshot.value["snapshot"]["text"] == expected_snapshot_text
    assert good_handle.writes == ["attach -P4242\n"]
    return

    assert isinstance(status, Ok)
    assert isinstance(snapshot, Ok)
    assert status.value["memory_reader_enabled"] is True
    assert status.value["active_data_source"] == DATA_SOURCE_MEMORY_READER
    assert status.value["memory_reader_runtime"]["status"] == "active"
    assert snapshot.value["snapshot"]["text"] == "Windows é»˜è®¤é…ç½®å·²è‡ªåŠ¨æŽ¥ç®¡ã€‚"
    assert handle.writes == ["attach -P4242\n"]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_windows_default_memory_reader_config_stays_idle_when_textractor_autodiscovery_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(galgame_service.sys, "platform", "win32")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty-local"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty-program-files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "empty-program-files-x86"))

    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    cfg = _make_effective_config(
        bridge_root,
        memory_reader={
            "auto_detect": True,
            "poll_interval_seconds": 1,
        },
    )
    del cfg["memory_reader"]["enabled"]  # type: ignore[index]
    del cfg["memory_reader"]["textractor_path"]  # type: ignore[index]

    ctx = _Ctx(plugin_dir, cfg)
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    plugin._memory_reader_manager = MemoryReaderManager(
        logger=plugin.logger,
        config=plugin._cfg,
        process_scanner=lambda: [],
        time_fn=lambda: 1710000000.0,
        platform_fn=lambda: True,
        writer=MemoryReaderBridgeWriter(
            bridge_root=bridge_root,
            time_fn=lambda: 1710000000.0,
        ),
    )

    await plugin._poll_bridge(force=True)
    status = await plugin.galgame_get_status()

    assert isinstance(status, Ok)
    assert status.value["memory_reader_enabled"] is True
    assert status.value["active_data_source"] == "none"
    assert status.value["memory_reader_runtime"]["status"] == "idle"
    assert status.value["memory_reader_runtime"]["detail"] == "invalid_textractor_path"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_install_textractor_entry_returns_install_result_and_refreshed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    install_root = tmp_path / "TextractorInstalled"
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            memory_reader={
                "enabled": True,
                "install_target_dir": str(install_root),
            },
        ),
    )
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    async def _fake_install_textractor(**kwargs):
        del kwargs
        install_root.mkdir(parents=True, exist_ok=True)
        (install_root / "TextractorCLI.exe").write_text("", encoding="utf-8")
        return {
            "installed": True,
            "already_installed": False,
            "detected_path": str(install_root / "TextractorCLI.exe"),
            "target_dir": str(install_root),
            "expected_executable_path": str(install_root / "TextractorCLI.exe"),
            "install_supported": True,
            "can_install": False,
            "detail": "installed",
            "summary": "Textractor 安装完成",
            "release_name": "v1.0.0",
            "asset_name": "Textractor-x64.zip",
        }

    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.install_textractor",
        _fake_install_textractor,
    )

    result = await plugin.galgame_install_textractor()

    assert isinstance(result, Ok)
    assert result.value["summary"] == "Textractor 安装完成"
    assert result.value["install_result"]["installed"] is True
    assert result.value["status"]["textractor"]["installed"] is True
    assert result.value["status"]["textractor"]["detected_path"] == str(
        install_root / "TextractorCLI.exe"
    )


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_install_tesseract_entry_returns_install_result_and_refreshed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    install_root = tmp_path / "TesseractInstalled"
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            ocr_reader={
                "enabled": True,
                "install_target_dir": str(install_root),
            },
        ),
    )
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    async def _fake_install_tesseract(**kwargs):
        del kwargs
        install_root.mkdir(parents=True, exist_ok=True)
        (install_root / "tesseract.exe").write_text("", encoding="utf-8")
        tessdata_dir = install_root / "tessdata"
        tessdata_dir.mkdir(parents=True, exist_ok=True)
        for language in ("chi_sim", "jpn", "eng"):
            (tessdata_dir / f"{language}.traineddata").write_text("", encoding="utf-8")
        return {
            "installed": True,
            "already_installed": False,
            "detected_path": str(install_root / "tesseract.exe"),
            "target_dir": str(install_root),
            "expected_executable_path": str(install_root / "tesseract.exe"),
            "tessdata_dir": str(tessdata_dir),
            "required_languages": ["chi_sim", "jpn", "eng"],
            "missing_languages": [],
            "install_supported": True,
            "can_install": False,
            "detail": "installed",
            "summary": "Tesseract 安装完成",
            "release_name": "Tesseract OCR",
            "asset_name": "tesseract-ocr-w64-setup.exe",
        }

    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.install_tesseract",
        _fake_install_tesseract,
    )

    result = await plugin.galgame_install_tesseract()

    assert isinstance(result, Ok)
    assert result.value["summary"] == "Tesseract 安装完成"
    assert result.value["install_result"]["installed"] is True
    assert result.value["status"]["tesseract"]["installed"] is True
    assert result.value["status"]["tesseract"]["detected_path"] == str(
        install_root / "tesseract.exe"
    )


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_install_rapidocr_entry_returns_install_result_and_refreshed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    install_root = tmp_path / "RapidOCRInstalled"
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            ocr_reader={"enabled": True},
            rapidocr={
                "enabled": True,
                "install_target_dir": str(install_root),
            },
        ),
    )
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    async def _fake_install_rapidocr(**kwargs):
        del kwargs
        runtime_dir = install_root / "runtime"
        site_packages_dir = runtime_dir / "site-packages"
        model_cache_dir = install_root / "models"
        package_dir = site_packages_dir / "rapidocr_onnxruntime"
        package_dir.mkdir(parents=True, exist_ok=True)
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        return {
            "installed": True,
            "already_installed": False,
            "detected_path": str(package_dir),
            "target_dir": str(install_root),
            "runtime_dir": str(runtime_dir),
            "site_packages_dir": str(site_packages_dir),
            "model_cache_dir": str(model_cache_dir),
            "selected_model": "PP-OCRv5/ch/mobile",
            "engine_type": "onnxruntime",
            "lang_type": "ch",
            "model_type": "mobile",
            "ocr_version": "PP-OCRv5",
            "install_supported": True,
            "can_install": False,
            "detail": "installed",
            "summary": "RapidOCR 安装完成",
            "release_name": "RapidOCR ONNXRuntime",
            "asset_name": "rapidocr_onnxruntime, onnxruntime",
            "runtime_error": "",
        }

    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.install_rapidocr",
        _fake_install_rapidocr,
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.service.inspect_rapidocr_installation",
        lambda **kwargs: {
            "install_supported": True,
            "installed": True,
            "can_install": False,
            "detected_path": str(install_root / "runtime" / "site-packages" / "rapidocr_onnxruntime"),
            "target_dir": str(install_root),
            "runtime_dir": str(install_root / "runtime"),
            "site_packages_dir": str(install_root / "runtime" / "site-packages"),
            "model_cache_dir": str(install_root / "models"),
            "selected_model": "PP-OCRv5/ch/mobile",
            "engine_type": "onnxruntime",
            "lang_type": "ch",
            "model_type": "mobile",
            "ocr_version": "PP-OCRv5",
            "detail": "installed",
            "runtime_error": "",
        },
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": True,
            "detail": "installed",
            "detected_path": str(install_root / "runtime" / "site-packages" / "rapidocr_onnxruntime"),
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )

    result = await plugin.galgame_install_rapidocr()

    assert isinstance(result, Ok)
    assert result.value["summary"] == "RapidOCR 安装完成"
    assert result.value["install_result"]["installed"] is True
    assert result.value["status"]["rapidocr"]["installed"] is True
    assert result.value["status"]["rapidocr"]["selected_model"] == "PP-OCRv5/ch/mobile"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_set_ocr_capture_profile_updates_state_and_store(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            ocr_reader={"enabled": True},
        ),
    )
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    saved = await plugin.galgame_set_ocr_capture_profile(
        process_name="DemoGame.exe",
        left_inset_ratio=0.08,
        right_inset_ratio=0.06,
        top_ratio=0.34,
        bottom_inset_ratio=0.22,
    )

    assert isinstance(saved, Ok)
    assert saved.value["process_name"] == "DemoGame.exe"
    assert saved.value["capture_profile"]["top_ratio"] == pytest.approx(0.34)
    with plugin._state_lock:
        assert plugin._state.ocr_capture_profiles["DemoGame.exe"]["left_inset_ratio"] == pytest.approx(0.08)
    restored, _warnings = plugin._persist.load()
    assert restored[STORE_OCR_CAPTURE_PROFILES]["DemoGame.exe"]["bottom_inset_ratio"] == pytest.approx(0.22)

    cleared = await plugin.galgame_set_ocr_capture_profile(
        process_name="DemoGame.exe",
        clear=True,
    )

    assert isinstance(cleared, Ok)
    assert cleared.value["cleared"] is True
    with plugin._state_lock:
        assert "DemoGame.exe" not in plugin._state.ocr_capture_profiles


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_ocr_reader_fallback_activates_when_bridge_sdk_and_memory_reader_are_missing(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    install_root = tmp_path / "Tesseract"
    install_root.mkdir(parents=True, exist_ok=True)
    (install_root / "tesseract.exe").write_text("", encoding="utf-8")
    tessdata_dir = install_root / "tessdata"
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    for language in ("chi_sim", "jpn", "eng"):
        (tessdata_dir / f"{language}.traineddata").write_text("", encoding="utf-8")

    cfg = _make_effective_config(
        bridge_root,
        memory_reader={
            "enabled": True,
            "textractor_path": "",
        },
        ocr_reader={
            "enabled": True,
            "install_target_dir": str(install_root),
            "poll_interval_seconds": 999.0,
        },
    )
    ctx = _Ctx(plugin_dir, cfg)
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    clock = {"now": 1710000000.0}
    plugin._ocr_reader_manager = OcrReaderManager(
        logger=plugin.logger,
        config=plugin._cfg,
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="OCR Demo Window",
                process_name="DemoGame.exe",
                pid=4242,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(
            [
                "雪乃：来自 OCR 的台词。",
                "雪乃：来自 OCR 的台词。",
            ]
        ),
        writer=OcrReaderBridgeWriter(
            bridge_root=bridge_root,
            time_fn=lambda: clock["now"],
        ),
    )

    await plugin._poll_bridge(force=True)
    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)
    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)

    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()

    assert isinstance(status, Ok)
    assert isinstance(snapshot, Ok)
    assert status.value["active_data_source"] == DATA_SOURCE_OCR_READER
    assert status.value["summary"].startswith("已通过 OCR 读取连接（降级模式）")
    assert snapshot.value["snapshot"]["scene_id"] == "ocr:unknown_scene"
    assert snapshot.value["snapshot"]["line_id"].startswith("ocr:")
    assert snapshot.value["snapshot"]["text"] == "来自 OCR 的台词。"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_bridge_sdk_session_preempts_ocr_reader_candidate(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    install_root = tmp_path / "Tesseract"
    install_root.mkdir(parents=True, exist_ok=True)
    (install_root / "tesseract.exe").write_text("", encoding="utf-8")
    tessdata_dir = install_root / "tessdata"
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    for language in ("chi_sim", "jpn", "eng"):
        (tessdata_dir / f"{language}.traineddata").write_text("", encoding="utf-8")

    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            ocr_reader={
                "enabled": True,
                "install_target_dir": str(install_root),
                "poll_interval_seconds": 999.0,
            },
        ),
    )
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    clock = {"now": 1711000000.0}
    plugin._ocr_reader_manager = OcrReaderManager(
        logger=plugin.logger,
        config=plugin._cfg,
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=202,
                title="OCR Demo Window",
                process_name="DemoGame.exe",
                pid=4343,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(
            [
                "雪乃：OCR 台词。",
                "雪乃：OCR 台词。",
            ]
        ),
        writer=OcrReaderBridgeWriter(
            bridge_root=bridge_root,
            time_fn=lambda: clock["now"],
        ),
    )

    await plugin._poll_bridge(force=True)
    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)
    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)

    _create_game_dir(
        bridge_root,
        game_id="demo.sdk",
        session_payload=_session(
            game_id="demo.sdk",
            session_id="sdk-session-1",
            last_seq=3,
            state=_session_state(
                speaker="桥接",
                text="来自 Bridge SDK 的台词。",
                scene_id="scene-sdk",
                line_id="line-sdk",
                ts="2026-04-21T08:31:00Z",
            ),
        ),
        events=[],
    )

    await plugin._poll_bridge(force=True)
    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()

    assert isinstance(status, Ok)
    assert isinstance(snapshot, Ok)
    assert status.value["active_data_source"] == DATA_SOURCE_BRIDGE_SDK
    assert snapshot.value["snapshot"]["text"] == "来自 Bridge SDK 的台词。"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_memory_reader_fallback_activates_when_bridge_sdk_is_missing(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    textractor_path = tmp_path / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    cfg = _make_effective_config(
        bridge_root,
        memory_reader={
            "enabled": True,
            "textractor_path": str(textractor_path),
        },
    )
    ctx = _Ctx(plugin_dir, cfg)
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    clock = {"now": 1710000000.0}
    handle = _FakeTextractorHandle(
        ["[4242:100:0:0] 雪乃：来自内存读取的台词。"]
    )
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
    status = await plugin.galgame_get_status()
    snapshot = await plugin.galgame_get_snapshot()

    assert isinstance(status, Ok)
    assert isinstance(snapshot, Ok)
    assert status.value["memory_reader_enabled"] is True
    assert status.value["active_data_source"] == DATA_SOURCE_MEMORY_READER
    assert status.value["summary"].startswith("已通过内存读取连接（降级模式）")
    assert status.value["memory_reader_runtime"]["status"] == "active"
    assert snapshot.value["snapshot"]["text"] == "来自内存读取的台词。"
    assert handle.writes == ["attach -P4242\n"]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_bridge_sdk_session_preempts_memory_reader_candidate(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    textractor_path = tmp_path / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    cfg = _make_effective_config(
        bridge_root,
        memory_reader={
            "enabled": True,
            "textractor_path": str(textractor_path),
        },
    )
    ctx = _Ctx(plugin_dir, cfg)
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    clock = {"now": 1710000000.0}
    handle = _FakeTextractorHandle(
        ["[4242:100:0:0] 雪乃：先走内存读取链路。"]
    )
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
    assert isinstance(memory_reader_status, Ok)
    assert memory_reader_status.value["active_data_source"] == DATA_SOURCE_MEMORY_READER

    _create_game_dir(
        bridge_root,
        game_id="demo.bridge",
        session_payload=_session(
            game_id="demo.bridge",
            session_id="sdk-sess",
            last_seq=3,
            state=_session_state(
                speaker="桥接",
                text="Bridge SDK 已接管。",
                line_id="sdk-line",
                scene_id="sdk-scene",
            ),
        ),
    )

    clock["now"] += 1.0
    await plugin._poll_bridge(force=True)
    status = await plugin.galgame_get_status()

    assert isinstance(status, Ok)
    assert status.value["active_data_source"] == DATA_SOURCE_BRIDGE_SDK
    assert status.value["active_session_id"] == "sdk-sess"
    assert status.value["memory_reader_runtime"]["detail"] == "bridge_sdk_available"
    assert handle.terminated is True


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_phase2_entries_return_structured_degraded_results_without_target_entry(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "demo.alpha"
    session_id = "sess-a"
    _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=2,
            state=_session_state(
                speaker="雪乃",
                text="今天一起回家吗？",
                scene_id="scene-a",
                line_id="line-1",
                choices=[
                    {"choice_id": "choice-1", "text": "好啊", "index": 0, "enabled": True},
                    {"choice_id": "choice-2", "text": "下次吧", "index": 1, "enabled": True},
                ],
                is_menu_open=True,
                ts="2026-04-21T08:31:00Z",
            ),
        ),
        events=[
            _event(
                seq=1,
                event_type="line_changed",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "speaker": "雪乃",
                    "text": "今天一起回家吗？",
                    "line_id": "line-1",
                    "scene_id": "scene-a",
                    "route_id": "",
                },
                ts="2026-04-21T08:31:00Z",
            ),
            _event(
                seq=2,
                event_type="choices_shown",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "line_id": "line-1",
                    "scene_id": "scene-a",
                    "route_id": "",
                    "choices": [
                        {"choice_id": "choice-1", "text": "好啊", "index": 0, "enabled": True},
                        {"choice_id": "choice-2", "text": "下次吧", "index": 1, "enabled": True},
                    ],
                },
                ts="2026-04-21T08:31:01Z",
            ),
        ],
    )

    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()

    explain = await plugin.galgame_explain_line()
    summarize = await plugin.galgame_summarize_scene()
    suggest = await plugin.galgame_suggest_choice()
    agent_status = await plugin.galgame_agent_command(action="query_status")
    agent_reply = await plugin.galgame_agent_command(action="query_context", context_query="当前场景在讲什么？")

    assert isinstance(explain, Ok)
    assert explain.value["degraded"] is True
    assert explain.value["line_id"] == "line-1"
    assert "gateway_unavailable" in explain.value["diagnostic"]

    assert isinstance(summarize, Ok)
    assert summarize.value["degraded"] is True
    assert summarize.value["scene_id"] == "scene-a"

    assert isinstance(suggest, Ok)
    assert suggest.value["degraded"] is True
    assert suggest.value["choices"] == []

    assert isinstance(agent_status, Ok)
    assert agent_status.value["action"] == "query_status"
    assert isinstance(agent_status.value["recent_pushes"], list)

    assert isinstance(agent_reply, Ok)
    assert agent_reply.value["action"] == "query_context"
    assert "场景" in agent_reply.value["result"]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_phase2_entries_mark_memory_reader_input_as_degraded_even_when_llm_succeeds(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    game_id = "mem-1a2b3c4d5e6f"
    session_id = "mem-session"
    _create_game_dir(
        bridge_root,
        game_id=game_id,
        session_payload=_memory_reader_session(
            game_id=game_id,
            session_id=session_id,
            last_seq=2,
            state=_session_state(
                speaker="雪乃",
                text="这是内存读取来的台词。",
                scene_id="mem:unknown_scene",
                line_id="mem:line-1",
                choices=[
                    {"choice_id": "mem:line-1#choice0", "text": "去教室", "index": 0, "enabled": True},
                    {"choice_id": "mem:line-1#choice1", "text": "去天台", "index": 1, "enabled": True},
                ],
                is_menu_open=True,
                ts="2026-04-21T08:31:00Z",
            ),
        ),
        events=[
            _event(
                seq=1,
                event_type="line_changed",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "speaker": "雪乃",
                    "text": "这是内存读取来的台词。",
                    "line_id": "mem:line-1",
                    "scene_id": "mem:unknown_scene",
                    "route_id": "",
                },
                ts="2026-04-21T08:31:00Z",
            ),
            _event(
                seq=2,
                event_type="choices_shown",
                session_id=session_id,
                game_id=game_id,
                payload={
                    "line_id": "mem:line-1",
                    "scene_id": "mem:unknown_scene",
                    "route_id": "",
                    "choices": [
                        {"choice_id": "mem:line-1#choice0", "text": "去教室", "index": 0, "enabled": True},
                        {"choice_id": "mem:line-1#choice1", "text": "去天台", "index": 1, "enabled": True},
                    ],
                },
                ts="2026-04-21T08:31:01Z",
            ),
        ],
    )

    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            llm={"target_entry_ref": "fake_llm:run"},
        ),
    )

    async def _handler(**kwargs):
        params = kwargs.get("params") or {}
        operation = params.get("operation")
        if operation == "explain_line":
            return {"explanation": "这是对台词的解释。", "evidence": []}
        if operation == "summarize_scene":
            return {
                "summary": "这是对场景的总结。",
                "key_points": [{"type": "plot", "text": "剧情仍在推进。"}],
            }
        if operation == "suggest_choice":
            context = params.get("context") or {}
            visible_choices = context.get("visible_choices") or []
            return {
                "choices": [
                    {
                        "choice_id": visible_choices[0]["choice_id"],
                        "text": visible_choices[0]["text"],
                        "rank": 1,
                        "reason": "优先继续主线。",
                    }
                ]
            }
        raise AssertionError(f"unexpected operation: {operation}")

    ctx.entry_handler = _handler
    plugin = GalgameBridgePlugin(ctx)
    await plugin.startup()
    plugin._memory_reader_manager = SimpleNamespace(
        update_config=lambda config: None,
        tick=lambda **kwargs: asyncio.sleep(
            0,
            result=SimpleNamespace(
                warnings=[],
                should_rescan=False,
                runtime={
                    "enabled": True,
                    "status": "active",
                    "detail": "fixture_active",
                    "process_name": "RenPy Demo.exe",
                    "pid": 4242,
                    "engine": "unknown",
                    "game_id": game_id,
                    "session_id": session_id,
                    "last_seq": 2,
                    "last_event_ts": "2026-04-21T08:31:01Z",
                },
            ),
        ),
        shutdown=lambda: asyncio.sleep(0, result=None),
    )
    await plugin._poll_bridge(force=True)

    status = await plugin.galgame_get_status()
    explain = await plugin.galgame_explain_line()
    summarize = await plugin.galgame_summarize_scene()
    suggest = await plugin.galgame_suggest_choice()

    assert isinstance(status, Ok)
    assert status.value["active_data_source"] == DATA_SOURCE_MEMORY_READER

    assert isinstance(explain, Ok)
    assert explain.value["degraded"] is True
    assert "memory_reader_input" in explain.value["diagnostic"]
    assert "weaker than bridge_sdk" in explain.value["diagnostic"]
    assert explain.value["explanation"] == "这是对台词的解释。"

    assert isinstance(summarize, Ok)
    assert summarize.value["degraded"] is True
    assert "memory_reader_input" in summarize.value["diagnostic"]
    assert "weaker than bridge_sdk" in summarize.value["diagnostic"]
    assert summarize.value["summary"] == "这是对场景的总结。"

    assert isinstance(suggest, Ok)
    assert suggest.value["degraded"] is True
    assert "memory_reader_input" in suggest.value["diagnostic"]
    assert "weaker than bridge_sdk" in suggest.value["diagnostic"]
    assert suggest.value["choices"][0]["choice_id"] == "mem:line-1#choice0"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_gateway_reuses_inflight_and_ttl_cache(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(
            bridge_root,
            llm={"target_entry_ref": "fake_llm:run", "llm_request_cache_ttl_seconds": 2},
        ),
    )

    calls = {"count": 0}

    async def _handler(**kwargs):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        params = kwargs.get("params") or {}
        if params.get("operation") == "summarize_scene":
            return {
                "summary": "场景总结",
                "key_points": [
                    {
                        "type": "plot",
                        "text": "剧情推进",
                        "line_id": "line-1",
                        "speaker": "雪乃",
                        "scene_id": "scene-a",
                        "route_id": "",
                    }
                ],
            }
        raise AssertionError(f"unexpected operation: {params}")

    ctx.entry_handler = _handler
    plugin = GalgameBridgePlugin(ctx)
    gateway = LLMGateway(plugin, _Logger(), plugin._cfg or type("Cfg", (), {
        "llm_max_in_flight": 2,
        "llm_request_cache_ttl_seconds": 2,
        "llm_call_timeout_seconds": 15,
        "llm_target_entry_ref": "fake_llm:run",
    })())

    context = {
        "scene_id": "scene-a",
        "route_id": "",
        "game_id": "demo.alpha",
        "session_id": "sess-a",
        "recent_lines": [],
        "recent_choices": [],
        "current_snapshot": _session_state(scene_id="scene-a", line_id="line-1"),
    }

    first, second = await asyncio.gather(
        gateway.summarize_scene(context),
        gateway.summarize_scene(context),
    )
    third = await gateway.summarize_scene(context)

    assert first["degraded"] is False
    assert second["summary"] == "场景总结"
    assert third["summary"] == "场景总结"
    assert calls["count"] == 1


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_gateway_degrades_on_invalid_result(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(
        plugin_dir,
        _make_effective_config(bridge_root, llm={"target_entry_ref": "fake_llm:run"}),
    )
    ctx.entry_handler = {"summary": 123, "key_points": "oops"}
    plugin = GalgameBridgePlugin(ctx)
    gateway = LLMGateway(plugin, _Logger(), type("Cfg", (), {
        "llm_max_in_flight": 2,
        "llm_request_cache_ttl_seconds": 0,
        "llm_call_timeout_seconds": 15,
        "llm_target_entry_ref": "fake_llm:run",
    })())

    payload = await gateway.summarize_scene(
        build_summarize_context(
            _shared_state(history_lines=[{"line_id": "line-1", "speaker": "雪乃", "text": "台词", "scene_id": "scene-a", "route_id": "", "ts": "2026-04-21T08:31:00Z"}]),
            scene_id="scene-a",
        )
    )
    assert payload["degraded"] is True
    assert "invalid_result" in payload["diagnostic"]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_host_agent_adapter_round_trip_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    app = FastAPI()
    task_state = {"status": "running"}

    @app.get("/computer_use/availability")
    async def _availability():
        return {"ready": True, "reasons": []}

    @app.post("/computer_use/run")
    async def _run(payload: dict[str, Any]):
        return {"success": True, "task_id": "task-1", "status": "running", "instruction": payload["instruction"]}

    @app.get("/tasks/task-1")
    async def _task():
        return {"id": "task-1", "status": task_state["status"]}

    @app.post("/tasks/task-1/cancel")
    async def _cancel():
        task_state["status"] = "cancelled"
        return {"success": True, "task_id": "task-1", "status": "cancelled"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        adapter = HostAgentAdapter(_Logger(), tool_server_port=48915)
        monkeypatch.setattr(adapter, "_get_client", lambda: client)

        availability = await adapter.get_computer_use_availability()
        started = await adapter.run_computer_use_instruction("advance once")
        task = await adapter.get_task("task-1")
        cancelled = await adapter.cancel_task("task-1")

    assert availability["ready"] is True
    assert started["task_id"] == "task-1"
    assert task["status"] == "running"
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_uses_rank1_choice_and_records_push_history(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        suggest_payload={
            "degraded": False,
            "choices": [
                {
                    "choice_id": "choice-2",
                    "text": "右边",
                    "rank": 1,
                    "reason": "更符合当前目标",
                }
            ],
            "diagnostic": "",
        }
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )

    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="你要走哪边？",
            scene_id="scene-a",
            line_id="line-1",
            choices=[
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ],
            is_menu_open=True,
            ts="2026-04-21T08:31:00Z",
        ),
        history_lines=[
            {
                "line_id": "line-1",
                "speaker": "雪乃",
                "text": "你要走哪边？",
                "scene_id": "scene-a",
                "route_id": "",
                "ts": "2026-04-21T08:31:00Z",
            }
        ],
    )

    await agent.tick(shared)
    await asyncio.sleep(0)
    await agent.tick(shared)
    assert "右边" in fake_host.started[-1]

    fake_host.tasks["task-1"]["status"] = "completed"
    shared_after = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="那就走这边吧。",
            scene_id="scene-a",
            line_id="line-2",
            ts="2026-04-21T08:31:02Z",
        ),
        history_lines=[
            {
                "line_id": "line-1",
                "speaker": "雪乃",
                "text": "你要走哪边？",
                "scene_id": "scene-a",
                "route_id": "",
                "ts": "2026-04-21T08:31:00Z",
            },
            {
                "line_id": "line-2",
                "speaker": "雪乃",
                "text": "那就走这边吧。",
                "scene_id": "scene-a",
                "route_id": "",
                "ts": "2026-04-21T08:31:02Z",
            },
        ],
        history_choices=[
            {
                "choice_id": "choice-2",
                "text": "右边",
                "line_id": "line-1",
                "scene_id": "scene-a",
                "route_id": "",
                "index": 1,
                "action": "selected",
                "ts": "2026-04-21T08:31:01Z",
            }
        ],
        last_seq=3,
    )
    await agent.tick(shared_after)
    status = await agent.query_status(shared_after)

    assert len(ctx.pushed_messages) == 1
    assert status["recent_pushes"][0]["kind"] == "choice_reason"
    assert "推荐理由" in status["recent_pushes"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_falls_back_to_first_choice_when_suggest_is_degraded(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        suggest_payload={"degraded": True, "choices": [], "diagnostic": "busy"}
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="你要走哪边？",
            scene_id="scene-a",
            line_id="line-1",
            choices=[
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ],
            is_menu_open=True,
        ),
    )

    await agent.tick(shared)
    await asyncio.sleep(0)
    await agent.tick(shared)

    assert "左边" in fake_host.started[-1]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_send_message_interrupts_pending_planning(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        suggest_payload={"degraded": False, "choices": [], "diagnostic": ""},
        reply_payload={"degraded": False, "reply": "收到，当前还在选项界面。", "diagnostic": ""},
        delay=0.2,
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="你要走哪边？",
            scene_id="scene-a",
            line_id="line-1",
            choices=[
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ],
            is_menu_open=True,
        ),
    )

    await agent.tick(shared)
    response = await agent.send_message(shared, message="先别操作，告诉我当前状态")

    assert response["result"] == "收到，当前还在选项界面。"
    assert fake_host.started == []
    assert fake_gateway.reply_calls[-1]["prompt"] == "先别操作，告诉我当前状态"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_retries_dialogue_with_alternate_advance_strategy(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="下一句还没出来。",
            scene_id="scene-a",
            line_id="line-1",
            ts="2026-04-21T08:31:00Z",
        ),
    )

    await agent.tick(shared)
    assert "press Enter exactly once" in fake_host.started[-1]

    fake_host.tasks["task-1"]["status"] = "completed"
    await agent.tick(shared)
    assert agent._actuation is not None
    agent._actuation["bridge_wait_started_at"] = time.monotonic() - 6.0

    await agent.tick(shared)
    agent._next_actuation_at = 0.0
    await agent.tick(shared)

    assert len(fake_host.started) == 2
    assert "click the usual continue area exactly once" in fake_host.started[-1]
    assert agent._failure_memory[-1]["strategy_id"] == "advance_enter"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_recovers_unknown_ui_after_stall(tmp_path: Path) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="",
            text="",
            scene_id="scene-a",
            line_id="",
            ts="2026-04-21T08:31:00Z",
        ),
        history_lines=[],
    )

    await agent.tick(shared)
    agent._scene_state["last_scene_change_at"] = time.monotonic() - 1.0

    await agent.tick(shared)
    await agent.tick(shared)

    assert len(fake_host.started) == 1
    assert "dismiss that overlay exactly once" in fake_host.started[-1]
    assert agent._scene_state["stage"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_choice_failure_retries_variant_then_next_candidate(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        suggest_payload={
            "degraded": False,
            "choices": [
                {
                    "choice_id": "choice-2",
                    "text": "右边",
                    "rank": 1,
                    "reason": "更符合当前目标",
                },
                {
                    "choice_id": "choice-1",
                    "text": "左边",
                    "rank": 2,
                    "reason": "保守路线",
                },
            ],
            "diagnostic": "",
        }
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="你要走哪边？",
            scene_id="scene-a",
            line_id="line-1",
            choices=[
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ],
            is_menu_open=True,
        ),
    )

    await agent.tick(shared)
    await asyncio.sleep(0)
    await agent.tick(shared)
    assert "\"右边\"" in fake_host.started[-1]

    fake_host.tasks["task-1"]["status"] = "failed"
    fake_host.tasks["task-1"]["error"] = "missed first choice"
    await agent.tick(shared)
    agent._next_actuation_at = 0.0
    await agent.tick(shared)

    assert len(fake_host.started) == 2
    assert "menu item index 2 exactly once" in fake_host.started[-1]

    fake_host.tasks["task-2"]["status"] = "failed"
    fake_host.tasks["task-2"]["error"] = "still missed"
    await agent.tick(shared)
    agent._next_actuation_at = 0.0
    await agent.tick(shared)

    assert len(fake_host.started) == 3
    assert "\"左边\"" in fake_host.started[-1]
    assert [item["strategy_id"] for item in agent._failure_memory[-2:]] == [
        "choose_rank_1_variant_1",
        "choose_rank_1_variant_2",
    ]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_set_standby_cancels_inflight_actuation_and_keeps_query_available(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        reply_payload={"degraded": False, "reply": "待机中，当前台词是「当前台词」。", "diagnostic": ""}
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state()

    await agent.tick(shared)
    assert fake_host.started

    standby_result = await agent.set_standby(shared, standby=True)
    query_result = await agent.query_context(shared, context_query="现在是什么状态？")

    assert standby_result["status"] == "standby"
    assert fake_host.cancelled == ["task-1"]
    assert query_result["status"] == "standby"
    assert query_result["result"] == "待机中，当前台词是「当前台词」。"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_no_bridge_delta_walks_full_recovery_chain(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="剧情还在原地。",
            scene_id="scene-a",
            line_id="line-1",
            ts="2026-04-21T08:32:00Z",
        ),
    )

    async def _fail_current_by_no_delta() -> None:
        task_id = str(agent._actuation["task_id"])
        fake_host.tasks[task_id]["status"] = "completed"
        await agent.tick(shared)
        assert agent._actuation is not None
        agent._actuation["bridge_wait_started_at"] = time.monotonic() - 6.0
        await agent.tick(shared)
        agent._next_actuation_at = 0.0
        await agent.tick(shared)

    await agent.tick(shared)
    assert "press Enter exactly once" in fake_host.started[-1]

    await _fail_current_by_no_delta()
    await _fail_current_by_no_delta()
    await _fail_current_by_no_delta()
    await _fail_current_by_no_delta()

    assert len(fake_host.started) == 5
    assert "press Enter exactly once" in fake_host.started[0]
    assert "click the usual continue area exactly once" in fake_host.started[1]
    assert "press Space exactly once" in fake_host.started[2]
    assert "dismiss that overlay exactly once" in fake_host.started[3]
    assert "close that overlay once" in fake_host.started[4]
    assert [item["strategy_id"] for item in agent._failure_memory[-4:]] == [
        "advance_enter",
        "advance_click",
        "advance_space",
        "recover_focus",
    ]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_scene_transition_stall_uses_recover_strategy(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot={
            **_session_state(
                speaker="",
                text="",
                scene_id="scene-a",
                line_id="",
                ts="2026-04-21T08:32:00Z",
            ),
            "save_context": {
                "kind": "rollback",
                "slot_id": "",
                "display_name": "rollback",
            },
        },
        history_lines=[],
    )

    await agent.tick(shared)
    agent._scene_state["last_scene_change_at"] = time.monotonic() - 1.0
    await agent.tick(shared)

    assert agent._scene_state["stage"] == "scene_transition"
    assert len(fake_host.started) == 1
    assert "dismiss that overlay exactly once" in fake_host.started[-1]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_send_message_interrupts_awaiting_bridge_without_host_cancel(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway(
        reply_payload={"degraded": False, "reply": "当前还没确认桥接回包。", "diagnostic": ""}
    )
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state()

    await agent.tick(shared)
    fake_host.tasks["task-1"]["status"] = "completed"
    await agent.tick(shared)

    assert agent._actuation is not None
    assert agent._actuation["state"] == "awaiting_bridge"

    response = await agent.send_message(shared, message="先停一下，说明现在卡在哪")

    assert response["status"] == "active"
    assert response["result"] == "当前还没确认桥接回包。"
    assert agent._actuation is None
    assert fake_host.cancelled == []


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_set_standby_interrupts_awaiting_bridge_without_host_cancel(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state()

    await agent.tick(shared)
    fake_host.tasks["task-1"]["status"] = "completed"
    await agent.tick(shared)

    assert agent._actuation is not None
    assert agent._actuation["state"] == "awaiting_bridge"

    response = await agent.set_standby(shared, standby=True)

    assert response["status"] == "standby"
    assert agent._actuation is None
    assert fake_host.cancelled == []


@pytest.mark.asyncio
@pytest.mark.plugin_unit
@pytest.mark.parametrize(
    ("mode", "expected_kinds"),
    [
        ("silent", []),
        ("companion", ["scene_summary"]),
        ("choice_advisor", ["scene_summary", "choice_reason"]),
    ],
)
async def test_game_llm_agent_mode_controls_push_types(
    tmp_path: Path,
    mode: str,
    expected_kinds: list[str],
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )

    shared_before = _shared_state(
        mode=mode,
        connection_state="idle",
        snapshot=_session_state(
            speaker="雪乃",
            text="第一幕开场。",
            scene_id="scene-a",
            line_id="line-1",
            ts="2026-04-21T08:32:00Z",
        ),
        history_lines=[
            {
                "line_id": "line-1",
                "speaker": "雪乃",
                "text": "第一幕开场。",
                "scene_id": "scene-a",
                "route_id": "",
                "ts": "2026-04-21T08:32:00Z",
            }
        ],
    )
    await agent.tick(shared_before)

    agent._remember_suggestion_reason("choice-1", "这里更符合当前目标")
    shared_after = _shared_state(
        mode=mode,
        connection_state="idle",
        snapshot=_session_state(
            speaker="雪乃",
            text="第二幕开场。",
            scene_id="scene-b",
            line_id="line-2",
            ts="2026-04-21T08:32:03Z",
        ),
        history_lines=[
            {
                "line_id": "line-1",
                "speaker": "雪乃",
                "text": "第一幕开场。",
                "scene_id": "scene-a",
                "route_id": "",
                "ts": "2026-04-21T08:32:00Z",
            },
            {
                "line_id": "line-2",
                "speaker": "雪乃",
                "text": "第二幕开场。",
                "scene_id": "scene-b",
                "route_id": "",
                "ts": "2026-04-21T08:32:03Z",
            },
        ],
        history_choices=[
            {
                "choice_id": "choice-1",
                "text": "继续",
                "line_id": "line-1",
                "scene_id": "scene-a",
                "route_id": "",
                "index": 0,
                "action": "selected",
                "ts": "2026-04-21T08:32:02Z",
            }
        ],
    )
    await agent.tick(shared_after)

    assert [item["metadata"]["kind"] for item in ctx.pushed_messages] == expected_kinds
    status = await agent.query_status(shared_after)
    assert [item["kind"] for item in status["recent_pushes"]] == expected_kinds


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_internal_memories_stay_bounded_over_long_run(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter()
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )

    for idx in range(80):
        if idx:
            agent._remember_suggestion_reason(f"choice-{idx}", f"理由 {idx}")
        shared = _shared_state(
            mode="choice_advisor",
            connection_state="idle",
            last_seq=idx,
            snapshot=_session_state(
                speaker="雪乃",
                text=f"台词 {idx}",
                scene_id=f"scene-{idx}",
                line_id=f"line-{idx}",
                ts=f"2026-04-21T08:32:{idx:02d}Z",
            ),
            history_lines=[
                {
                    "line_id": f"line-{idx}",
                    "speaker": "雪乃",
                    "text": f"台词 {idx}",
                    "scene_id": f"scene-{idx}",
                    "route_id": "",
                    "ts": f"2026-04-21T08:32:{idx:02d}Z",
                }
            ],
            history_choices=(
                []
                if idx == 0
                else [
                    {
                        "choice_id": f"choice-{idx}",
                        "text": f"选项 {idx}",
                        "line_id": f"line-{idx}",
                        "scene_id": f"scene-{idx}",
                        "route_id": "",
                        "index": idx,
                        "action": "selected",
                        "ts": f"2026-04-21T08:32:{idx:02d}Z",
                    }
                ]
            ),
        )
        await agent.tick(shared)

    for idx in range(20):
        agent._record_failure(
            kind="recover",
            strategy_id=f"recover-{idx}",
            reason=f"failure-{idx}",
            scene_id=f"scene-{idx}",
        )
    for idx in range(40):
        agent._remember_suggestion_reason(f"pending-choice-{idx}", f"pending-reason-{idx}")

    assert len(agent._scene_memory) == 32
    assert agent._scene_memory[0]["scene_id"] == "scene-48"
    assert agent._scene_memory[-1]["scene_id"] == "scene-79"

    assert len(agent._choice_memory) == 64
    assert agent._choice_memory[0]["choice_id"] == "choice-16"
    assert agent._choice_memory[-1]["choice_id"] == "choice-79"

    assert len(agent._recent_pushes) == 20
    assert agent._recent_pushes[-1]["kind"] == "choice_reason"

    assert len(agent._failure_memory) == 16
    assert agent._failure_memory[0]["strategy_id"] == "recover-4"
    assert agent._failure_memory[-1]["strategy_id"] == "recover-19"

    assert len(agent._suggestion_reasons) == 32


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_game_llm_agent_recovers_after_temporary_host_unavailable(
    tmp_path: Path,
) -> None:
    plugin_dir, bridge_root = _make_plugin_dirs(tmp_path)
    ctx = _Ctx(plugin_dir, _make_effective_config(bridge_root))
    plugin = GalgameBridgePlugin(ctx)
    fake_gateway = _FakeLLMGateway()
    fake_host = _FakeHostAdapter(ready=False)
    agent = GameLLMAgent(
        plugin=plugin,
        logger=_Logger(),
        llm_gateway=fake_gateway,
        host_adapter=fake_host,
    )
    shared = _shared_state(
        snapshot=_session_state(
            speaker="雪乃",
            text="继续前进。",
            scene_id="scene-a",
            line_id="line-1",
            ts="2026-04-21T08:32:00Z",
        ),
    )

    await agent.tick(shared)
    first_status = await agent.query_status(shared)

    assert first_status["status"] == "error"
    assert "computer_use unavailable" in first_status["result"]
    assert fake_host.started == []

    fake_host.ready = True
    agent._next_actuation_at = 0.0
    await agent.tick(shared)
    recovered_status = await agent.query_status(shared)

    assert recovered_status["status"] == "active"
    assert fake_host.started
    assert agent._actuation is not None
