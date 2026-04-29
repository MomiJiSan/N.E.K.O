from __future__ import annotations

from types import SimpleNamespace

import pytest

from plugin.plugins.galgame_plugin import service as galgame_service
from plugin.plugins.galgame_plugin.models import (
    DATA_SOURCE_BRIDGE_SDK,
    DATA_SOURCE_MEMORY_READER,
    DATA_SOURCE_OCR_READER,
    SessionCandidate,
)
from plugin.plugins.galgame_plugin.service import (
    build_explain_context,
    build_summarize_context,
    choose_candidate,
)


pytestmark = pytest.mark.plugin_unit


def _local_state() -> dict[str, object]:
    return {
        "active_game_id": "game.demo",
        "active_session_id": "session-demo",
        "active_data_source": DATA_SOURCE_OCR_READER,
        "latest_snapshot": {
            "speaker": "",
            "text": "OCR 目标窗口：等待截图",
            "line_id": "",
            "scene_id": "ocr:game:scene-0001",
            "route_id": "ocr",
            "choices": [],
            "is_menu_open": False,
        },
        "history_lines": [
            {
                "speaker": "雪乃",
                "text": "今天先回去吧。",
                "line_id": "ocr:line-stable",
                "scene_id": "ocr:game:scene-0001",
                "route_id": "ocr",
                "stability": "stable",
            }
        ],
        "history_observed_lines": [
            {
                "speaker": "",
                "text": "OCR 目标窗口：等待截图",
                "line_id": "ocr:diagnostic",
                "scene_id": "ocr:game:scene-0001",
                "route_id": "ocr",
                "stability": "tentative",
            },
            {
                "speaker": "",
                "text": "她轻声说：走吧。",
                "line_id": "ocr:line-observed",
                "scene_id": "ocr:game:scene-0001",
                "route_id": "ocr",
                "stability": "tentative",
            },
        ],
        "history_choices": [],
    }


def _patch_status_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(galgame_service, "inspect_dxcam_installation", lambda: {})
    monkeypatch.setattr(galgame_service, "inspect_textractor_installation", lambda **kwargs: {})
    monkeypatch.setattr(galgame_service, "inspect_rapidocr_installation", lambda **kwargs: {})
    monkeypatch.setattr(galgame_service, "inspect_tesseract_installation", lambda **kwargs: {})
    monkeypatch.setattr(galgame_service, "_current_process_performance", lambda: {})


def _candidate(
    tmp_path,
    *,
    game_id: str,
    data_source: str,
    text: str = "",
    choices: list[dict[str, object]] | None = None,
    last_seq: int = 1,
) -> SessionCandidate:
    return SessionCandidate(
        game_id=game_id,
        session_path=tmp_path / game_id / "session.json",
        events_path=tmp_path / game_id / "events.jsonl",
        data_source=data_source,
        session={
            "session_id": f"session-{game_id}",
            "started_at": "2026-04-29T00:00:00Z",
            "last_seq": last_seq,
            "state": {
                "text": text,
                "choices": list(choices or []),
                "ts": "2026-04-29T00:00:01Z",
            },
        },
    )


def _status_state(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "current_connection_state": "active",
        "mode": "companion",
        "push_notifications": True,
        "advance_speed": "medium",
        "bound_game_id": "demo",
        "active_game_id": "demo",
        "available_game_ids": ["demo"],
        "active_session_id": "session-1",
        "active_data_source": DATA_SOURCE_OCR_READER,
        "stream_reset_pending": False,
        "last_seq": 1,
        "last_error": {},
        "memory_reader_runtime": {},
        "ocr_reader_runtime": {"status": "running"},
        "ocr_capture_profiles": {},
        "latest_snapshot": {
            "speaker": "雪乃",
            "text": "今天先回去吧。",
            "line_id": "line-1",
            "scene_id": "scene-1",
            "route_id": "ocr",
            "choices": [],
            "is_menu_open": False,
        },
        "history_observed_lines": [],
        "history_lines": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_service_summarize_context_filters_overlay_diagnostics() -> None:
    context = build_summarize_context(_local_state(), scene_id="ocr:game:scene-0001")

    texts = [item["text"] for item in context["recent_lines"]]

    assert "OCR 目标窗口：等待截图" not in texts
    assert "今天先回去吧。" in texts
    assert "她轻声说：走吧。" in texts
    assert context["input_degraded"] is True
    assert "ocr_reader_source" in context["degraded_reasons"]


def test_service_explain_context_uses_history_when_snapshot_is_diagnostic() -> None:
    context = build_explain_context(_local_state(), line_id="ocr:line-stable")

    assert context["speaker"] == "雪乃"
    assert context["text"] == "今天先回去吧。"
    assert context["evidence"]
    assert context["input_degraded"] is True


def test_choose_candidate_auto_prefers_bridge_text_then_memory_text(tmp_path) -> None:
    bridge = _candidate(
        tmp_path,
        game_id="bridge",
        data_source=DATA_SOURCE_BRIDGE_SDK,
        text="stable bridge text",
        last_seq=1,
    )
    memory = _candidate(
        tmp_path,
        game_id="memory",
        data_source=DATA_SOURCE_MEMORY_READER,
        text="memory reader text",
        last_seq=100,
    )

    assert choose_candidate(
        {"bridge": bridge, "memory": memory},
        bound_game_id="",
        current_game_id="",
        keep_current=False,
    ) is bridge

    empty_bridge = _candidate(
        tmp_path,
        game_id="empty-bridge",
        data_source=DATA_SOURCE_BRIDGE_SDK,
        text="",
        last_seq=200,
    )
    assert choose_candidate(
        {"empty-bridge": empty_bridge, "memory": memory},
        bound_game_id="",
        current_game_id="",
        keep_current=False,
    ) is memory


def test_status_payload_snapshot_fast_path_skips_json_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    config = galgame_service.build_config({})
    state = _status_state()
    _patch_status_dependencies(monkeypatch)

    def _unexpected_json_copy(value: object) -> object:
        raise AssertionError(f"json_copy should not be called for snapshot input: {value!r}")

    monkeypatch.setattr(galgame_service, "json_copy", _unexpected_json_copy)

    payload = galgame_service.build_status_payload(
        state,
        config=config,
        state_is_snapshot=True,
    )

    assert payload["effective_current_line"]["text"] == "今天先回去吧。"
    assert payload["ocr_reader_runtime"] == {"status": "running"}
    assert payload["primary_diagnosis"]["title"]


def test_status_payload_primary_diagnosis_reports_minimized_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = galgame_service.build_config({})
    state = _status_state(
        ocr_reader_runtime={
            "status": "running",
            "target_selection_detail": "memory_reader_window_minimized",
            "last_exclude_reason": "excluded_minimized_window",
            "candidate_count": 0,
        },
        latest_snapshot={
            "speaker": "",
            "text": "",
            "line_id": "",
            "scene_id": "scene-1",
            "route_id": "ocr",
            "choices": [],
            "is_menu_open": False,
        },
    )
    _patch_status_dependencies(monkeypatch)

    payload = galgame_service.build_status_payload(
        state,
        config=config,
        state_is_snapshot=True,
    )

    diagnosis = payload["primary_diagnosis"]
    assert diagnosis["severity"] == "warning"
    assert diagnosis["title"] == "游戏窗口已最小化"
    assert [action["id"] for action in diagnosis["actions"]] == [
        "refresh_ocr_windows",
        "select_ocr_window",
    ]


def test_status_payload_primary_diagnosis_reports_capture_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = galgame_service.build_config({})
    state = _status_state(
        ocr_reader_runtime={
            "status": "running",
            "ocr_context_state": "capture_failed",
            "last_capture_error": "PrintWindow timeout",
            "effective_window_key": "pid:100:hwnd:200",
        },
        latest_snapshot={
            "speaker": "",
            "text": "",
            "line_id": "",
            "scene_id": "scene-1",
            "route_id": "ocr",
            "choices": [],
            "is_menu_open": False,
        },
    )
    _patch_status_dependencies(monkeypatch)

    payload = galgame_service.build_status_payload(
        state,
        config=config,
        state_is_snapshot=True,
    )

    diagnosis = payload["primary_diagnosis"]
    assert diagnosis["severity"] == "error"
    assert diagnosis["title"] == "截图或文字识别失败"
    assert diagnosis["message"] == "PrintWindow timeout"
    assert {"recalibrate_ocr", "capture_backend", "debug_details"} <= {
        action["id"] for action in diagnosis["actions"]
    }


def test_status_payload_primary_diagnosis_reports_poll_not_running() -> None:
    diagnosis = galgame_service.build_primary_diagnosis(
        {
            "ocr_reader_runtime": {
                "status": "running",
                "ocr_context_state": "poll_not_running",
            },
            "ocr_context_state": "poll_not_running",
            "ocr_capture_diagnostic": "OCR 轮询未继续执行。",
        }
    )

    assert diagnosis["severity"] == "error"
    assert diagnosis["title"] == "OCR 轮询没有继续运行"
    assert diagnosis["message"] == "OCR 轮询未继续执行。"


def test_status_payload_primary_diagnosis_reports_observed_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = galgame_service.build_config({})
    state = _status_state(
        ocr_reader_runtime={
            "status": "running",
            "effective_window_key": "pid:100:hwnd:200",
            "last_observed_line": {
                "text": "新的候选台词。",
                "line_id": "line-observed",
            },
            "last_stable_line": {
                "text": "旧台词。",
                "line_id": "line-stable",
            },
        },
        latest_snapshot={
            "speaker": "",
            "text": "",
            "line_id": "",
            "scene_id": "scene-1",
            "route_id": "ocr",
            "choices": [],
            "is_menu_open": False,
        },
    )
    _patch_status_dependencies(monkeypatch)

    payload = galgame_service.build_status_payload(
        state,
        config=config,
        state_is_snapshot=True,
    )

    diagnosis = payload["primary_diagnosis"]
    assert diagnosis["severity"] == "info"
    assert diagnosis["title"] == "刚读到新文字"
    assert [action["id"] for action in diagnosis["actions"]] == ["line_details"]
