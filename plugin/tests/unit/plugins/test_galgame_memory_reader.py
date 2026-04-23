from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugin.plugins.galgame_plugin.memory_reader import (
    DetectedGameProcess,
    MemoryReaderBridgeWriter,
    MemoryReaderManager,
    _default_process_scanner,
)
from plugin.plugins.galgame_plugin.reader import read_session_json, tail_events_jsonl
from plugin.plugins.galgame_plugin.service import build_config


pytestmark = pytest.mark.plugin_unit


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


def _make_config(
    bridge_root: Path,
    *,
    enabled: bool = True,
    textractor_path: str = "",
    auto_detect: bool = True,
    poll_interval_seconds: float = 1.0,
) -> object:
    return build_config(
        {
            "galgame": {
                "bridge_root": str(bridge_root),
            },
            "memory_reader": {
                "enabled": enabled,
                "textractor_path": textractor_path,
                "auto_detect": auto_detect,
                "poll_interval_seconds": poll_interval_seconds,
            },
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform_value", "auto_detect", "textractor_exists", "expected_detail", "warning_fragment"),
    [
        (False, True, True, "unsupported_platform", "Windows-only"),
        (True, False, True, "manual_pid_unimplemented", "auto_detect=false"),
        (True, True, False, "invalid_textractor_path", "textractor_path"),
    ],
)
async def test_memory_reader_manager_returns_recoverable_warnings_for_unavailable_runtime(
    tmp_path: Path,
    platform_value: bool,
    auto_detect: bool,
    textractor_exists: bool,
    expected_detail: str,
    warning_fragment: str,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    textractor_path = tmp_path / "TextractorCLI.exe"
    if textractor_exists:
        textractor_path.write_text("", encoding="utf-8")

    manager = MemoryReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            textractor_path=str(textractor_path),
            auto_detect=auto_detect,
        ),
        platform_fn=lambda: platform_value,
    )

    result = await manager.tick(bridge_sdk_available=False)

    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == expected_detail
    assert warning_fragment in result.warnings[0]


def test_default_process_scanner_orders_candidates_by_create_time_then_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Proc:
        def __init__(self, info: dict[str, object], modules: list[str]) -> None:
            self.info = info
            self._modules = modules

        def memory_maps(self, grouped: bool = False):
            del grouped
            return [SimpleNamespace(path=module) for module in self._modules]

    fake_psutil = SimpleNamespace(
        process_iter=lambda fields: [
            _Proc(
                {
                    "pid": 30,
                    "name": "OldGame.exe",
                    "cmdline": ["OldGame.exe"],
                    "create_time": 10.0,
                },
                ["UnityPlayer.dll"],
            ),
            _Proc(
                {
                    "pid": 20,
                    "name": "python.exe",
                    "cmdline": ["python.exe", "renpy"],
                    "create_time": 20.0,
                },
                [],
            ),
            _Proc(
                {
                    "pid": 10,
                    "name": "UnityGame.exe",
                    "cmdline": ["UnityGame.exe"],
                    "create_time": 20.0,
                },
                ["UnityPlayer.dll"],
            ),
        ]
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.memory_reader.psutil",
        fake_psutil,
    )

    detected = _default_process_scanner()

    assert [(item.pid, item.engine) for item in detected] == [
        (10, "unity"),
        (20, "renpy"),
        (30, "unity"),
    ]


def test_memory_reader_bridge_writer_emits_stable_bridge_schema_and_choice_ids(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    writer = MemoryReaderBridgeWriter(bridge_root=bridge_root, time_fn=lambda: 1710000000.0)
    process = DetectedGameProcess(
        pid=4242,
        name="RenPy Demo.exe",
        create_time=1710000000.0,
        engine="renpy",
    )

    writer.start_session(process)
    assert writer.emit_line("雪乃：一起回家吧。", ts="2026-04-22T01:00:00Z") is True
    assert writer.emit_choices(["去教室", "去天台"], ts="2026-04-22T01:00:01Z") is True

    session_path = bridge_root / writer.game_id / "session.json"
    events_path = bridge_root / writer.game_id / "events.jsonl"
    session = read_session_json(session_path).session
    assert session is not None
    assert session["protocol_version"] == 1
    assert session["bridge_sdk_version"].startswith("memory-reader-")
    assert session["metadata"]["source"] == "memory_reader"
    assert session["metadata"]["game_process_name"] == "RenPy Demo.exe"
    assert session["metadata"]["game_pid"] == 4242
    assert session["state"]["scene_id"] == "mem:unknown_scene"
    assert session["state"]["line_id"].startswith("mem:")
    assert session["state"]["choices"][0]["choice_id"] == f"{session['state']['line_id']}#choice0"
    assert session["state"]["choices"][1]["choice_id"] == f"{session['state']['line_id']}#choice1"

    events = tail_events_jsonl(events_path, offset=0, line_buffer=b"").events
    assert [event["type"] for event in events] == [
        "session_started",
        "line_changed",
        "choices_shown",
    ]
    assert events[-1]["payload"]["choices"][0]["text"] == "去教室"


@pytest.mark.asyncio
async def test_memory_reader_manager_attaches_consumes_textractor_output_and_emits_heartbeat(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    textractor_path = tmp_path / "TextractorCLI.exe"
    textractor_path.write_text("", encoding="utf-8")
    handle = _FakeTextractorHandle(
        [
            "[4242:100:0:0] 雪乃：今天也一起回家吧。",
            "[4242:100:0:0] 雪乃：今天也一起回家吧。",
        ]
    )
    clock = {"now": 1710000000.0}

    async def _process_factory(path: str):
        del path
        return handle

    manager = MemoryReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            textractor_path=str(textractor_path),
            auto_detect=True,
            poll_interval_seconds=0.5,
        ),
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
    )

    first = await manager.tick(bridge_sdk_available=False)
    assert first.should_rescan is True
    assert first.runtime["status"] == "active"
    assert handle.writes == ["attach -P4242\n"]

    game_id = first.runtime["game_id"]
    events_path = bridge_root / game_id / "events.jsonl"
    first_events = tail_events_jsonl(events_path, offset=0, line_buffer=b"").events
    assert [event["type"] for event in first_events] == ["session_started", "line_changed"]

    clock["now"] += 1.0
    second = await manager.tick(bridge_sdk_available=False)
    assert second.should_rescan is True
    second_events = tail_events_jsonl(events_path, offset=0, line_buffer=b"").events
    assert [event["type"] for event in second_events] == [
        "session_started",
        "line_changed",
        "heartbeat",
    ]

    await manager.shutdown()
    assert handle.terminated is True
