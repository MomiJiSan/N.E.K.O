from __future__ import annotations

from pathlib import Path

import pytest

from plugin.plugins.galgame_plugin import ocr_reader as galgame_ocr_reader
from plugin.plugins.galgame_plugin.models import (
    DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_TOP_RATIO,
)
from plugin.plugins.galgame_plugin.ocr_reader import (
    DetectedGameWindow,
    OcrReaderBridgeWriter,
    OcrReaderManager,
    _AihongStage,
    _AihongStateMachine,
    _rapidocr_text_from_output,
    _score_ocr_text,
)
from plugin.plugins.galgame_plugin.reader import read_session_json, tail_events_jsonl
from plugin.plugins.galgame_plugin.service import build_config
from plugin.plugins.galgame_plugin.tesseract_support import (
    DEFAULT_TESSERACT_LANGUAGES,
    default_tesseract_install_target_raw,
    inspect_tesseract_installation,
    resolve_tesseract_install_target,
)


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


class _FakeCaptureBackend:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.capture_calls: list[tuple[int, dict[str, float]]] = []

    def is_available(self) -> bool:
        return self.available

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}:{target.pid}"

    def capture_frame(self, target: DetectedGameWindow, profile) -> str:
        self.capture_calls.append((target.hwnd, profile.to_dict()))
        return f"frame:{target.hwnd}:{len(self.capture_calls)}"


class _FakeOcrBackend:
    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or [])
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def extract_text(self, image: str) -> str:
        del image
        self.calls += 1
        if not self._texts:
            return ""
        if len(self._texts) == 1:
            return self._texts[0]
        return self._texts.pop(0)


def _make_config(
    bridge_root: Path,
    *,
    enabled: bool = True,
    backend_selection: str = "auto",
    tesseract_path: str = "",
    install_target_dir: str = "",
    poll_interval_seconds: float = 999.0,
    no_text_takeover_after_seconds: float = 30.0,
    languages: str = DEFAULT_TESSERACT_LANGUAGES,
    rapidocr_enabled: bool = True,
    rapidocr_install_target_dir: str = "",
    trigger_mode: str = "after_advance",
) -> object:
    return build_config(
        {
            "galgame": {
                "bridge_root": str(bridge_root),
            },
            "ocr_reader": {
                "enabled": enabled,
                "backend_selection": backend_selection,
                "tesseract_path": tesseract_path,
                "install_target_dir": install_target_dir,
                "poll_interval_seconds": poll_interval_seconds,
                "no_text_takeover_after_seconds": no_text_takeover_after_seconds,
                "languages": languages,
                "trigger_mode": trigger_mode,
            },
            "rapidocr": {
                "enabled": rapidocr_enabled,
                "install_target_dir": rapidocr_install_target_dir,
                "engine_type": "onnxruntime",
                "lang_type": "ch",
                "model_type": "mobile",
                "ocr_version": "PP-OCRv5",
            },
        }
    )


def _install_fake_tesseract(root: Path, *, languages: str = DEFAULT_TESSERACT_LANGUAGES) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    executable = root / "tesseract.exe"
    executable.write_text("", encoding="utf-8")
    tessdata_dir = root / "tessdata"
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    for language in [item.strip() for item in languages.split("+") if item.strip()]:
        (tessdata_dir / f"{language}.traineddata").write_text("", encoding="utf-8")
    return executable


def _read_events(events_path: Path) -> list[dict[str, object]]:
    result = tail_events_jsonl(events_path, offset=0, line_buffer=b"")
    return result.events


def _window() -> list[DetectedGameWindow]:
    return [
        DetectedGameWindow(
            hwnd=101,
            title="Demo Window",
            process_name="DemoGame.exe",
            pid=4242,
        )
    ]


def _assert_ocr_runtime_group_matches_flat(runtime: dict[str, object]) -> None:
    ocr = runtime["ocr"]
    assert isinstance(ocr, dict)
    assert ocr["backend_kind"] == runtime["backend_kind"]
    assert ocr["backend_detail"] == runtime["backend_detail"]
    assert ocr["backend_path"] == runtime["backend_path"]
    assert ocr["backend_model"] == runtime["backend_model"]
    assert ocr["tesseract_path"] == runtime["tesseract_path"]
    assert ocr["languages"] == runtime["languages"]
    assert ocr["context_state"] == runtime["ocr_context_state"]
    assert ocr["consecutive_no_text_polls"] == runtime["consecutive_no_text_polls"]
    assert ocr["last_observed_at"] == runtime["last_observed_at"]
    assert ocr["last_capture_attempt_at"] == runtime["last_capture_attempt_at"]
    assert ocr["last_capture_completed_at"] == runtime["last_capture_completed_at"]
    assert ocr["last_capture_error"] == runtime["last_capture_error"]
    assert ocr["last_raw_text"] == runtime["last_raw_ocr_text"]
    assert ocr["last_observed_line"] == runtime["last_observed_line"]
    assert ocr["last_stable_line"] == runtime["last_stable_line"]


def _assert_window_runtime_group_matches_flat(runtime: dict[str, object]) -> None:
    window = runtime["window"]
    assert isinstance(window, dict)
    assert window["process_name"] == runtime["process_name"]
    assert window["pid"] == runtime["pid"]
    assert window["title"] == runtime["window_title"]
    assert window["width"] == runtime["width"]
    assert window["height"] == runtime["height"]
    assert window["aspect_ratio"] == runtime["aspect_ratio"]
    assert window["selection_mode"] == runtime["target_selection_mode"]
    assert window["selection_detail"] == runtime["target_selection_detail"]
    assert window["effective_window_key"] == runtime["effective_window_key"]
    assert window["effective_window_title"] == runtime["effective_window_title"]
    assert window["effective_process_name"] == runtime["effective_process_name"]
    assert window["target_is_foreground"] == runtime["target_is_foreground"]
    assert window["manual_target"] == runtime["manual_target"]
    assert window["locked_target"] == runtime["locked_target"]
    assert window["candidate_count"] == runtime["candidate_count"]
    assert window["excluded_candidate_count"] == runtime["excluded_candidate_count"]
    assert window["last_exclude_reason"] == runtime["last_exclude_reason"]
    assert window["foreground_refresh_at"] == runtime["foreground_refresh_at"]
    assert window["foreground_refresh_detail"] == runtime["foreground_refresh_detail"]
    assert window["foreground_hwnd"] == runtime["foreground_hwnd"]
    assert window["target_hwnd"] == runtime["target_hwnd"]


def test_build_config_defaults_ocr_languages_to_chi_sim_jpn_eng(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    cfg = build_config({"galgame": {"bridge_root": str(bridge_root)}})

    assert cfg.ocr_reader_languages == "chi_sim+jpn+eng"
    assert cfg.ocr_reader_backend_selection == "auto"
    assert cfg.ocr_reader_top_ratio == pytest.approx(DEFAULT_OCR_CAPTURE_TOP_RATIO)
    assert cfg.ocr_reader_bottom_inset_ratio == pytest.approx(
        DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO
    )


def test_ocr_reader_manager_initializes_with_rapidocr_warmup_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    warmup_calls = []
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.RapidOcrBackend.warmup_async",
        lambda self, logger=None: warmup_calls.append(logger),
    )

    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            backend_selection="rapidocr",
            rapidocr_enabled=True,
        ),
    )

    assert manager._writer.bridge_root == bridge_root
    assert warmup_calls


def test_rapidocr_runtime_cache_reuses_loaded_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_target_dir = str(tmp_path / "RapidOCR")
    runtime = object()
    load_calls = []

    def fake_load_runtime(**kwargs):
        load_calls.append(kwargs)
        return runtime, {}

    monkeypatch.setattr(galgame_ocr_reader, "load_rapidocr_runtime", fake_load_runtime)
    cache_key = (
        install_target_dir,
        "onnxruntime",
        "ch",
        "mobile",
        "PP-OCRv5",
    )
    galgame_ocr_reader._RAPIDOCR_RUNTIME_CACHE.pop(cache_key, None)
    try:
        first = galgame_ocr_reader.RapidOcrBackend(
            install_target_dir_raw=install_target_dir,
            engine_type="onnxruntime",
            lang_type="ch",
            model_type="mobile",
            ocr_version="PP-OCRv5",
        )
        second = galgame_ocr_reader.RapidOcrBackend(
            install_target_dir_raw=install_target_dir,
            engine_type="onnxruntime",
            lang_type="ch",
            model_type="mobile",
            ocr_version="PP-OCRv5",
        )

        assert first._ensure_runtime() is runtime
        assert second._ensure_runtime() is runtime
        assert len(load_calls) == 1
    finally:
        galgame_ocr_reader._RAPIDOCR_RUNTIME_CACHE.pop(cache_key, None)


def test_score_ocr_text_prefers_cjk_dialogue_over_ascii_gibberish() -> None:
    gibberish = "hs 四                 A y 3 8\n人~ x ai    アニ"
    chinese_dialogue = "她轻声说：今天先回去吧。"

    assert _score_ocr_text(chinese_dialogue) > _score_ocr_text(gibberish)


def test_inspect_tesseract_installation_reports_custom_install_target(tmp_path: Path) -> None:
    install_root = tmp_path / "CustomTesseract"
    executable = _install_fake_tesseract(install_root)

    status = inspect_tesseract_installation(
        configured_path="",
        install_target_dir_raw=str(install_root),
        languages=DEFAULT_TESSERACT_LANGUAGES,
    )

    assert status["installed"] is True
    assert status["detected_path"] == str(executable)
    assert status["target_dir"] == str(install_root)
    assert status["required_languages"] == ["chi_sim", "jpn", "eng"]
    assert status["missing_languages"] == []


def test_tesseract_default_install_target_matches_neko_programs_root() -> None:
    raw_target = default_tesseract_install_target_raw()

    assert raw_target == "%LOCALAPPDATA%/Programs/N.E.K.O/Tesseract-OCR"
    assert resolve_tesseract_install_target("").name == "Tesseract-OCR"
    assert resolve_tesseract_install_target("").parent.name == "N.E.K.O"


@pytest.mark.asyncio
async def test_ocr_reader_manager_reports_missing_tesseract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(bridge_root, enabled=True, rapidocr_enabled=False),
        platform_fn=lambda: True,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_tesseract_installation",
        lambda **kwargs: {
            "installed": False,
            "detail": "missing_tesseract",
            "detected_path": "",
            "required_languages": ["chi_sim", "jpn", "eng"],
            "missing_languages": ["chi_sim", "jpn", "eng"],
        },
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == "missing_tesseract"
    assert "Tesseract is missing" in result.warnings[0]


@pytest.mark.asyncio
async def test_ocr_reader_manager_auto_reports_rapidocr_first_when_all_backends_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(bridge_root, enabled=True, rapidocr_enabled=True),
        platform_fn=lambda: True,
        capture_backend=_FakeCaptureBackend(),
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": False,
            "detail": "broken_runtime",
            "runtime_error": "access denied",
            "detected_path": "C:/RapidOCR/site-packages/rapidocr_onnxruntime",
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_tesseract_installation",
        lambda **kwargs: {
            "installed": False,
            "detail": "missing",
            "detected_path": "",
            "required_languages": ["chi_sim", "jpn", "eng"],
            "missing_languages": ["chi_sim", "jpn", "eng"],
        },
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "idle"
    assert result.runtime["backend_kind"] == "rapidocr"
    assert result.runtime["detail"] == "broken_runtime"
    assert "RapidOCR is unavailable: broken_runtime" in result.warnings[0]
    assert any("Tesseract fallback" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_ocr_reader_manager_reports_missing_languages(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "TesseractMissingLangs"
    install_root.mkdir(parents=True, exist_ok=True)
    (install_root / "tesseract.exe").write_text("", encoding="utf-8")
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            rapidocr_enabled=False,
        ),
        platform_fn=lambda: True,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == "missing_languages"
    assert result.runtime["tesseract_path"] == str(install_root / "tesseract.exe")
    assert result.runtime["languages"] == DEFAULT_TESSERACT_LANGUAGES


@pytest.mark.asyncio
async def test_ocr_reader_manager_yields_bridge_sdk_and_memory_reader_statuses(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    executable = _install_fake_tesseract(install_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    bridge_result = await manager.tick(
        bridge_sdk_available=True,
        memory_reader_runtime={},
    )
    memory_result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={"detail": "receiving_text", "last_seq": 3},
    )

    assert bridge_result.runtime["detail"] == "bridge_sdk_available"
    assert bridge_result.runtime["tesseract_path"] == str(executable)
    assert memory_result.runtime["detail"] == "memory_reader_active"


@pytest.mark.asyncio
async def test_ocr_reader_manager_waits_before_taking_over_after_memory_reader_text(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 1000.0}
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            no_text_takeover_after_seconds=30.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={"detail": "receiving_text", "last_seq": 2},
    )
    clock["now"] += 5.0
    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == "waiting_for_takeover_window"


@pytest.mark.asyncio
async def test_ocr_reader_manager_does_not_treat_memory_reader_heartbeats_as_live_text(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={
            "status": "active",
            "detail": "attached_no_text_yet",
            "last_seq": 29,
        },
    )

    assert result.runtime["status"] == "active"
    assert result.runtime["detail"] == "attached_no_text_yet"
    assert result.runtime["ocr_context_state"] == "no_text"
    _assert_ocr_runtime_group_matches_flat(result.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_prefers_memory_reader_game_window_over_foreground_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    windows = [
        DetectedGameWindow(
            hwnd=202,
            title="插件详情 - N.E.K.O 插件管理 - Google Chrome",
            process_name="chrome.exe",
            pid=1500,
        ),
        DetectedGameWindow(
            hwnd=101,
            title="哀鸿",
            process_name="TheLamentingGeese.exe",
            pid=28828,
        ),
    ]
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader._foreground_window_handle",
        lambda: 202,
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: list(windows),
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={
            "status": "active",
            "detail": "attached_no_text_yet",
            "process_name": "TheLamentingGeese.exe",
            "pid": 28828,
            "last_seq": 29,
        },
    )

    assert result.runtime["status"] == "active"
    assert result.runtime["detail"] == "attached_no_text_yet"
    assert result.runtime["ocr_context_state"] == "no_text"
    assert result.runtime["process_name"] == "TheLamentingGeese.exe"
    assert result.runtime["pid"] == 28828
    _assert_ocr_runtime_group_matches_flat(result.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_locks_auto_target_when_user_focuses_other_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    game_window = DetectedGameWindow(
        hwnd=101,
        title="哀鸿",
        process_name="TheLamentingGeese.exe",
        pid=28828,
    )
    rebound_game_window = DetectedGameWindow(
        hwnd=303,
        title=game_window.title,
        process_name=game_window.process_name,
        pid=38828,
    )
    other_window = DetectedGameWindow(
        hwnd=202,
        title="Other Tool",
        process_name="Other.exe",
        pid=1500,
    )
    foreground = {"hwnd": game_window.hwnd}
    windows = {"items": [game_window, other_window]}
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader._foreground_window_handle",
        lambda: foreground["hwnd"],
    )
    capture_backend = _FakeCaptureBackend()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: list(windows["items"]),
        capture_backend=capture_backend,
        ocr_backend=_FakeOcrBackend(),
    )

    first = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert first.runtime["process_name"] == "TheLamentingGeese.exe"
    assert first.runtime["target_selection_detail"] == "foreground_window"
    assert first.runtime["target_is_foreground"] is True
    assert first.runtime["locked_target"]["process_name"] == "TheLamentingGeese.exe"
    assert capture_backend.capture_calls[-1][0] == game_window.hwnd
    _assert_window_runtime_group_matches_flat(first.runtime)

    foreground["hwnd"] = other_window.hwnd
    windows["items"] = [other_window]
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert second.runtime["status"] == "idle"
    assert second.runtime["detail"] == "waiting_for_valid_window"
    assert second.runtime["target_selection_detail"] == "locked_target_unavailable"
    assert len(capture_backend.capture_calls) == 1
    _assert_window_runtime_group_matches_flat(second.runtime)

    windows["items"] = [other_window, rebound_game_window]
    third = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert third.runtime["status"] == "active"
    assert third.runtime["process_name"] == "TheLamentingGeese.exe"
    assert third.runtime["pid"] == rebound_game_window.pid
    assert third.runtime["target_selection_detail"] == "locked_target_rebound"
    assert third.runtime["target_is_foreground"] is False
    assert capture_backend.capture_calls[-1][0] == rebound_game_window.hwnd
    _assert_window_runtime_group_matches_flat(third.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_applies_builtin_aihong_capture_profile(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="哀鸿",
                process_name="TheLamentingGeese.exe",
                pid=28828,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["capture_profile"]["left_inset_ratio"] == pytest.approx(0.0)
    assert result.runtime["capture_profile"]["right_inset_ratio"] == pytest.approx(0.0)
    assert result.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.60)
    assert result.runtime["capture_profile"]["bottom_inset_ratio"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_ocr_reader_manager_prefers_manual_capture_profile_over_builtin_aihong_profile(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="哀鸿",
                process_name="TheLamentingGeese.exe",
                pid=28828,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )
    manager.update_capture_profiles(
        {
            "TheLamentingGeese.exe": {
                "left_inset_ratio": 0.11,
                "right_inset_ratio": 0.09,
                "top_ratio": 0.41,
                "bottom_inset_ratio": 0.19,
            }
        }
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["capture_profile"]["left_inset_ratio"] == pytest.approx(0.11)
    assert result.runtime["capture_profile"]["right_inset_ratio"] == pytest.approx(0.09)
    assert result.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.41)
    assert result.runtime["capture_profile"]["bottom_inset_ratio"] == pytest.approx(0.19)


@pytest.mark.asyncio
async def test_aihong_menu_stage_accepts_plain_text_choices_after_dialogue_idle_polls(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 3000.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    capture_backend = _FakeCaptureBackend()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="哀鸿",
                process_name="TheLamentingGeese.exe",
                pid=28828,
            )
        ],
        capture_backend=capture_backend,
        ocr_backend=_FakeOcrBackend(
            [
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "往南跑\n躲进巷子里",
                "往南跑\n躲进巷子里",
            ]
        ),
        writer=writer,
    )

    for _ in range(7):
        latest = await manager.tick(
            bridge_sdk_available=False,
            memory_reader_runtime={},
        )
        clock["now"] += 1.0

    events = _read_events(bridge_root / writer.game_id / "events.jsonl")
    session = read_session_json(bridge_root / writer.game_id / "session.json").session

    assert latest.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.0)
    assert events[-1]["type"] == "choices_shown"
    payload = events[-1]["payload"]
    assert [item["text"] for item in payload["choices"]] == ["往南跑", "躲进巷子里"]
    assert session is not None
    assert session["state"]["is_menu_open"] is True
    assert capture_backend.capture_calls[0][1]["top_ratio"] == pytest.approx(0.60)
    assert capture_backend.capture_calls[0][1]["right_inset_ratio"] == pytest.approx(0.0)
    assert capture_backend.capture_calls[-1][1]["top_ratio"] == pytest.approx(0.0)
    assert capture_backend.capture_calls[-1][1]["bottom_inset_ratio"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_aihong_menu_probe_rejects_dialogue_like_multiline_text(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 4000.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    capture_backend = _FakeCaptureBackend()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="哀鸿",
                process_name="TheLamentingGeese.exe",
                pid=28828,
            )
        ],
        capture_backend=capture_backend,
        ocr_backend=_FakeOcrBackend(
            [
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "旁白：城门将破。\n将军：跟我来。",
                "将军：那纸上所书，必是紧要军令。",
                "旁白：城门将破。\n将军：跟我来。",
            ]
        ),
        writer=writer,
    )

    for _ in range(6):
        latest = await manager.tick(
            bridge_sdk_available=False,
            memory_reader_runtime={},
        )
        clock["now"] += 1.0

    events = _read_events(bridge_root / writer.game_id / "events.jsonl")
    session = read_session_json(bridge_root / writer.game_id / "session.json").session

    assert all(event["type"] != "choices_shown" for event in events)
    assert latest.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.60)
    assert session is not None
    assert session["state"]["is_menu_open"] is False
    assert all(
        call[1]["top_ratio"] == pytest.approx(0.60)
        for call in capture_backend.capture_calls
    )


@pytest.mark.asyncio
async def test_aihong_menu_stage_returns_to_dialogue_profile_after_stable_line(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 5000.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    capture_backend = _FakeCaptureBackend()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="哀鸿",
                process_name="TheLamentingGeese.exe",
                pid=28828,
            )
        ],
        capture_backend=capture_backend,
        ocr_backend=_FakeOcrBackend(
            [
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "将军：那纸上所书，必是紧要军令。",
                "往南跑\n躲进巷子里",
                "往南跑\n躲进巷子里",
                "将军：跟我来。",
                "将军：跟我来。",
                "将军：跟我来。",
            ]
        ),
        writer=writer,
    )

    for _ in range(12):
        latest = await manager.tick(
            bridge_sdk_available=False,
            memory_reader_runtime={},
        )
        clock["now"] += 1.0

    events = _read_events(bridge_root / writer.game_id / "events.jsonl")
    session = read_session_json(bridge_root / writer.game_id / "session.json").session

    assert events[-1]["type"] == "line_changed"
    assert latest.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.60)
    assert session is not None
    assert session["state"]["text"] == "跟我来。"
    assert session["state"]["is_menu_open"] is False
    assert any(
        call[1]["top_ratio"] == pytest.approx(0.0)
        for call in capture_backend.capture_calls
    )
    assert latest.runtime["capture_profile"]["top_ratio"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_ocr_reader_manager_starts_capture_and_emits_stable_line(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 1000.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(["雪乃：你好。", "雪乃：你好。"]),
        writer=writer,
    )

    first = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )
    clock["now"] += 1.0
    second = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )
    clock["now"] += 1.0
    third = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    session_path = bridge_root / writer.game_id / "session.json"
    session = read_session_json(session_path).session

    assert first.runtime["status"] == "active"
    assert first.runtime["detail"] == "receiving_text"
    assert first.runtime["ocr_context_state"] == "stable"
    assert first.runtime["consecutive_no_text_polls"] == 0
    assert first.runtime["last_observed_at"]
    assert first.runtime["last_capture_attempt_at"]
    assert first.runtime["last_capture_completed_at"]
    assert first.runtime["last_raw_ocr_text"] == "雪乃：你好。"
    assert first.runtime["last_observed_line"]["text"] == "你好。"
    _assert_ocr_runtime_group_matches_flat(first.runtime)
    assert second.runtime["status"] == "active"
    assert second.runtime["detail"] == "attached_no_text_yet"
    assert second.runtime["ocr_context_state"] == "no_text"
    assert second.runtime["last_stable_line"]["text"] == "你好。"
    _assert_ocr_runtime_group_matches_flat(second.runtime)
    assert third.runtime["status"] == "active"
    assert third.runtime["game_id"].startswith("ocr-")
    assert session is not None
    assert session["metadata"]["source"] == "ocr_reader"
    assert session["bridge_sdk_version"].startswith("ocr-reader-")
    assert session["state"]["scene_id"].startswith(f"ocr:{writer.game_id}:scene-")
    assert str(session["state"]["line_id"]).startswith("ocr:")
    assert session["state"]["text"] == "你好。"


@pytest.mark.asyncio
async def test_ocr_reader_followup_confirm_emits_next_line_in_same_tick(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 1100.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    capture_backend = _FakeCaptureBackend()
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
            trigger_mode="interval",
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=capture_backend,
        ocr_backend=_FakeOcrBackend(
            [
                "\u96ea\u4e43\uff1a\u4f60\u597d\u3002",
                "\u96ea\u4e43\uff1a\u4f60\u597d\u3002",
                "\u96ea\u4e43\uff1a\u4e0b\u4e00\u53e5\u3002",
                "\u96ea\u4e43\uff1a\u4e0b\u4e00\u53e5\u3002",
            ]
        ),
        writer=writer,
    )

    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    clock["now"] += 1.0
    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    clock["now"] += 1.0
    result = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    session = read_session_json(bridge_root / writer.game_id / "session.json").session
    events = _read_events(bridge_root / writer.game_id / "events.jsonl")

    assert result.stable_event_emitted is True
    assert session is not None
    assert session["state"]["text"] == "\u4e0b\u4e00\u53e5\u3002"
    assert events[-1]["type"] == "line_changed"
    assert len(capture_backend.capture_calls) >= 4


@pytest.mark.asyncio
async def test_ocr_reader_manager_reports_capture_diagnostic_after_repeated_no_text(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 1200.0}
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(["", "", ""]),
    )

    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    clock["now"] += 1.0
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    clock["now"] += 1.0
    third = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    clock["now"] += 1.0
    fourth = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert second.runtime["detail"] == "attached_no_text_yet"
    assert second.runtime["consecutive_no_text_polls"] == 2
    assert third.runtime["detail"] == "ocr_capture_diagnostic_required"
    assert third.runtime["consecutive_no_text_polls"] == 3
    assert third.runtime["ocr_context_state"] == "diagnostic_required"
    _assert_ocr_runtime_group_matches_flat(third.runtime)
    assert fourth.runtime["ocr_capture_diagnostic_required"] is True
    assert fourth.runtime["last_capture_stage"]
    assert fourth.runtime["last_capture_profile"]
    _assert_ocr_runtime_group_matches_flat(fourth.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_emits_choices_after_stable_menu_detection(tmp_path: Path) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    clock = {"now": 2000.0}
    writer = OcrReaderBridgeWriter(
        bridge_root=bridge_root,
        time_fn=lambda: clock["now"],
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        time_fn=lambda: clock["now"],
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(
            [
                "雪乃：选一个吧。",
                "雪乃：选一个吧。",
                "1. 去左边\n2. 去右边",
                "1. 去左边\n2. 去右边",
            ]
        ),
        writer=writer,
    )

    for _ in range(5):
        await manager.tick(
            bridge_sdk_available=False,
            memory_reader_runtime={},
        )
        clock["now"] += 1.0

    events = _read_events(bridge_root / writer.game_id / "events.jsonl")
    session = read_session_json(bridge_root / writer.game_id / "session.json").session

    assert events[-1]["type"] == "choices_shown"
    payload = events[-1]["payload"]
    assert len(payload["choices"]) == 2
    assert payload["choices"][0]["choice_id"].startswith(f"{payload['line_id']}#choice0")
    assert session is not None
    assert session["state"]["is_menu_open"] is True
    assert session["state"]["choices"][1]["text"] == "去右边"


def test_rapidocr_text_adapter_groups_lines_and_filters_low_confidence() -> None:
    low_confidence = [
        ([[0, 0], [10, 0], [10, 8], [0, 8]], "A", 0.30),
        ([[12, 0], [20, 0], [20, 8], [12, 8]], "B", 0.40),
    ]
    assert _rapidocr_text_from_output(low_confidence) == ""

    output = [
        ([[20, 10], [32, 10], [32, 24], [20, 24]], "Hello", 0.92),
        ([[2, 10], [16, 10], [16, 24], [2, 24]], "雪乃", 0.97),
        ([[2, 40], [18, 40], [18, 54], [2, 54]], "今天", 0.96),
        ([[20, 40], [36, 40], [36, 54], [20, 54]], "回家", 0.95),
    ]
    assert _rapidocr_text_from_output(output) == "雪乃Hello\n今天回家"


@pytest.mark.asyncio
async def test_ocr_reader_manager_auto_mode_prefers_rapidocr_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": True,
            "detail": "installed",
            "detected_path": "C:/RapidOCR/site-packages/rapidocr_onnxruntime",
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.RapidOcrBackend.extract_text",
        lambda self, image: "雪乃：你好。",
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            rapidocr_install_target_dir=str(tmp_path / "RapidOCR"),
        ),
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
    )

    first = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    third = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert first.runtime["backend_kind"] == "rapidocr"
    assert second.runtime["backend_kind"] == "rapidocr"
    assert third.runtime["backend_kind"] == "rapidocr"
    assert first.runtime["detail"] == "receiving_text"
    assert first.runtime["ocr_context_state"] == "stable"
    _assert_ocr_runtime_group_matches_flat(first.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_auto_mode_falls_back_to_tesseract_when_rapidocr_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": False,
            "detail": "missing",
            "detected_path": "",
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.TesseractOcrBackend.extract_text",
        lambda self, image: "雪乃：你好。",
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            rapidocr_install_target_dir=str(tmp_path / "RapidOCR"),
        ),
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
    )

    first = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert first.runtime["backend_kind"] == "tesseract"
    assert first.runtime["backend_detail"].startswith("auto_fallback_from_rapidocr")
    _assert_ocr_runtime_group_matches_flat(first.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_falls_back_to_tesseract_after_rapidocr_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": True,
            "detail": "installed",
            "detected_path": "C:/RapidOCR/site-packages/rapidocr_onnxruntime",
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )

    def _boom(self, image):
        raise RuntimeError("rapidocr boom")

    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.RapidOcrBackend.extract_text",
        _boom,
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.TesseractOcrBackend.extract_text",
        lambda self, image: "雪乃：你好。",
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            rapidocr_install_target_dir=str(tmp_path / "RapidOCR"),
        ),
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
    )

    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert second.runtime["backend_kind"] == "tesseract"
    assert second.runtime["backend_detail"] == "fallback_after_runtime_error"
    assert any("rapidocr failed" in warning for warning in second.warnings)
    _assert_ocr_runtime_group_matches_flat(second.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_forced_rapidocr_mode_does_not_fallback_to_tesseract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.inspect_rapidocr_installation",
        lambda **kwargs: {
            "installed": True,
            "detail": "installed",
            "detected_path": "C:/RapidOCR/site-packages/rapidocr_onnxruntime",
            "selected_model": "PP-OCRv5/ch/mobile",
        },
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.RapidOcrBackend.extract_text",
        lambda self, image: (_ for _ in ()).throw(RuntimeError("rapidocr boom")),
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader.TesseractOcrBackend.extract_text",
        lambda self, image: "不应该被调用",
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            backend_selection="rapidocr",
            install_target_dir=str(install_root),
            rapidocr_install_target_dir=str(tmp_path / "RapidOCR"),
        ),
        platform_fn=lambda: True,
        window_scanner=_window,
        capture_backend=_FakeCaptureBackend(),
    )

    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    assert second.runtime["backend_kind"] == "rapidocr"
    assert second.runtime["detail"] == "capture_failed"


@pytest.mark.asyncio
async def test_ocr_reader_manager_excludes_neko_self_window_and_waits_for_valid_target(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=202,
                title="Galgame Plugin - N.E.K.O Plugin Manager - Chrome",
                process_name="chrome.exe",
                pid=1500,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "idle"
    assert result.runtime["detail"] == "waiting_for_valid_window"
    assert result.runtime["candidate_count"] == 0
    assert result.runtime["excluded_candidate_count"] == 1
    assert result.runtime["last_exclude_reason"] == "excluded_self_window"
    _assert_window_runtime_group_matches_flat(result.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_prefers_manual_target_and_rebinds_by_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    manual_window = DetectedGameWindow(
        hwnd=777,
        title="Aiyoku no Eustia",
        process_name="Aiyoku.exe",
        pid=4455,
    )
    rebound_window = DetectedGameWindow(
        hwnd=778,
        title=manual_window.title,
        process_name=manual_window.process_name,
        pid=5566,
    )
    other_window = DetectedGameWindow(
        hwnd=100,
        title="Other Game",
        process_name="Other.exe",
        pid=1,
    )
    monkeypatch.setattr(
        "plugin.plugins.galgame_plugin.ocr_reader._foreground_window_handle",
        lambda: other_window.hwnd,
    )
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: [other_window, rebound_window],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(),
    )
    manager.update_window_target(
        {
            "mode": "manual",
            "window_key": manual_window.window_key,
            "process_name": manual_window.process_name,
            "normalized_title": manual_window.normalized_title,
            "pid": manual_window.pid,
            "last_known_hwnd": manual_window.hwnd,
            "selected_at": "2026-04-24T10:00:00Z",
        }
    )

    result = await manager.tick(
        bridge_sdk_available=False,
        memory_reader_runtime={},
    )

    assert result.runtime["status"] == "active"
    assert result.runtime["detail"] == "attached_no_text_yet"
    assert result.runtime["process_name"] == "Aiyoku.exe"
    assert result.runtime["target_selection_mode"] == "manual"
    assert result.runtime["target_selection_detail"] == "manual_target_rebound"
    assert result.runtime["manual_target"]["window_key"] == rebound_window.window_key
    assert result.runtime["manual_target"]["last_known_hwnd"] == 778
    assert result.runtime["manual_target"]["pid"] == rebound_window.pid
    assert manager.current_window_target()["window_key"] == rebound_window.window_key
    assert manager.current_window_target()["pid"] == rebound_window.pid
    _assert_window_runtime_group_matches_flat(result.runtime)


@pytest.mark.asyncio
async def test_ocr_reader_manager_blocks_text_that_looks_like_neko_plugin_ui(
    tmp_path: Path,
) -> None:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    install_root = tmp_path / "Tesseract"
    _install_fake_tesseract(install_root)
    writer = OcrReaderBridgeWriter(bridge_root=bridge_root)
    manager = OcrReaderManager(
        logger=_Logger(),
        config=_make_config(
            bridge_root,
            enabled=True,
            install_target_dir=str(install_root),
            poll_interval_seconds=999.0,
        ),
        platform_fn=lambda: True,
        window_scanner=lambda: [
            DetectedGameWindow(
                hwnd=101,
                title="Real Game Window",
                process_name="DemoGame.exe",
                pid=4242,
            )
        ],
        capture_backend=_FakeCaptureBackend(),
        ocr_backend=_FakeOcrBackend(
            [
                "RapidOCR install queued task",
                "RapidOCR install queued task",
            ]
        ),
        writer=writer,
    )

    await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})
    second = await manager.tick(bridge_sdk_available=False, memory_reader_runtime={})

    session = read_session_json(bridge_root / writer.game_id / "session.json").session

    assert second.runtime["detail"] == "self_ui_guard_blocked"
    assert session is not None


class TestAihongStateMachine:
    def test_default_state_is_dialogue(self) -> None:
        sm = _AihongStateMachine()
        assert sm.is_dialogue
        assert not sm.is_menu
        assert sm.capture_stage == "dialogue_stage"
        assert sm.dialogue_idle_polls == 0
        assert sm.menu_missing_polls == 0

    def test_reset_returns_to_dialogue(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        sm.dialogue_idle_polls = 3
        sm.menu_missing_polls = 1
        sm.menu_ocr_state.repeat_count = 2
        sm.reset()
        assert sm.is_dialogue
        assert sm.dialogue_idle_polls == 0
        assert sm.menu_missing_polls == 0
        assert sm.menu_ocr_state.repeat_count == 0

    def test_dialogue_emitted_choices_transitions_to_menu(self) -> None:
        sm = _AihongStateMachine()
        sm.on_dialogue_consumed(emitted=True, is_menu_choices=True, is_menu_status=False)
        assert sm.is_menu
        assert sm.dialogue_idle_polls == 0
        assert sm.menu_missing_polls == 0

    def test_dialogue_emitted_line_stays_dialogue_and_resets_menu_ocr(self) -> None:
        sm = _AihongStateMachine()
        sm.menu_ocr_state.repeat_count = 2
        sm.on_dialogue_consumed(emitted=True, is_menu_choices=False, is_menu_status=False)
        assert sm.is_dialogue
        assert sm.menu_ocr_state.repeat_count == 0

    def test_dialogue_idle_increments_when_not_menu_like(self) -> None:
        sm = _AihongStateMachine()
        sm.on_dialogue_consumed(emitted=False, is_menu_choices=False, is_menu_status=False)
        assert sm.dialogue_idle_polls == 1
        sm.on_dialogue_consumed(emitted=False, is_menu_choices=False, is_menu_status=False)
        assert sm.dialogue_idle_polls == 2

    def test_dialogue_idle_maxes_for_menu_status(self) -> None:
        sm = _AihongStateMachine()
        sm.dialogue_idle_polls = 5
        sm.on_dialogue_consumed(emitted=False, is_menu_choices=False, is_menu_status=True)
        assert sm.dialogue_idle_polls == 5

    def test_dialogue_idle_maxes_for_menu_choices(self) -> None:
        sm = _AihongStateMachine()
        sm.dialogue_idle_polls = 5
        sm.on_dialogue_consumed(emitted=False, is_menu_choices=True, is_menu_status=False)
        assert sm.dialogue_idle_polls == 5

    def test_should_probe_menu_after_advance_no_menu_not_idle_enough(self) -> None:
        sm = _AihongStateMachine()
        sm.dialogue_idle_polls = 1
        assert not sm.should_probe_menu(
            after_advance_trigger_mode=True,
            looks_like_menu=False,
        )

    def test_should_probe_menu_after_advance_with_menu(self) -> None:
        sm = _AihongStateMachine()
        assert sm.should_probe_menu(
            after_advance_trigger_mode=True,
            looks_like_menu=True,
        )

    def test_should_probe_menu_interval_when_idle_enough(self) -> None:
        sm = _AihongStateMachine()
        sm.dialogue_idle_polls = 2
        assert sm.should_probe_menu(
            after_advance_trigger_mode=False,
            looks_like_menu=False,
        )

    def test_menu_probe_choices_transitions_to_menu(self) -> None:
        sm = _AihongStateMachine()
        sm.on_menu_probe_result(emitted_kind="choices", has_menu_candidate=True)
        assert sm.is_menu
        assert sm.menu_missing_polls == 0

    def test_menu_probe_status_transitions_to_menu(self) -> None:
        sm = _AihongStateMachine()
        sm.on_menu_probe_result(emitted_kind="", has_menu_candidate=True)
        assert sm.is_menu
        assert sm.menu_missing_polls == 0

    def test_menu_probe_nothing_stays_dialogue(self) -> None:
        sm = _AihongStateMachine()
        sm.on_menu_probe_result(emitted_kind="", has_menu_candidate=False)
        assert sm.is_dialogue

    def test_active_menu_choices_stays_menu(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        sm.menu_missing_polls = 1
        reset = sm.on_active_menu_consumed(
            emitted_kind="choices",
            has_menu_candidate=True,
            text="",
        )
        assert not reset
        assert sm.is_menu
        assert sm.menu_missing_polls == 0

    def test_active_menu_missing_real_text_resets_to_dialogue(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        reset = sm.on_active_menu_consumed(
            emitted_kind="",
            has_menu_candidate=False,
            text="这是一段真实的文本",
        )
        assert reset
        assert sm.is_dialogue

    def test_active_menu_missing_noise_under_max_stays_menu(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        reset = sm.on_active_menu_consumed(
            emitted_kind="",
            has_menu_candidate=False,
            text="ab",
        )
        assert not reset
        assert sm.is_menu
        assert sm.menu_missing_polls == 1

    def test_active_menu_missing_noise_at_max_resets_to_dialogue(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        sm.menu_missing_polls = 1
        reset = sm.on_active_menu_consumed(
            emitted_kind="",
            has_menu_candidate=False,
            text="ab",
        )
        assert reset
        assert sm.is_dialogue
        assert sm.menu_missing_polls == 0

    def test_menu_probe_line_transitions_to_dialogue(self) -> None:
        sm = _AihongStateMachine()
        sm.stage = _AihongStage.MENU
        sm.menu_ocr_state.repeat_count = 2
        sm.on_menu_probe_result(emitted_kind="line", has_menu_candidate=False)
        assert sm.is_dialogue
        assert sm.menu_ocr_state.repeat_count == 0
