from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .models import DATA_SOURCE_OCR_READER, GalgameConfig
from .reader import normalize_text
from .tesseract_support import inspect_tesseract_installation, resolve_tesseract_path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

OCR_READER_VERSION = "0.1.0"
OCR_READER_BRIDGE_VERSION = f"ocr-reader-{OCR_READER_VERSION}"
OCR_READER_GAME_ID_PREFIX = "ocr-"
OCR_READER_UNKNOWN_SCENE = "ocr:unknown_scene"
OCR_READER_ROUTE_ID = ""
OCR_READER_DEFAULT_ENGINE = "unknown"

_MENU_PREFIX_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[\.\)\]:：]\s+)(.+\S)\s*$")


def utc_now_iso(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now))


def _ocr_game_id_from_process(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{OCR_READER_GAME_ID_PREFIX}{digest}"


def _coerce_choice_lines(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return []
    choices: list[str] = []
    for line in lines:
        match = _MENU_PREFIX_RE.match(line)
        if match is None:
            return []
        text = match.group(1).strip()
        if not text:
            return []
        choices.append(text)
    return choices


@dataclass(slots=True)
class OcrCaptureProfile:
    left_inset_ratio: float = 0.05
    right_inset_ratio: float = 0.05
    top_ratio: float = 0.3
    bottom_inset_ratio: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return {
            "left_inset_ratio": self.left_inset_ratio,
            "right_inset_ratio": self.right_inset_ratio,
            "top_ratio": self.top_ratio,
            "bottom_inset_ratio": self.bottom_inset_ratio,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OcrCaptureProfile:
        return cls(
            left_inset_ratio=float(value.get("left_inset_ratio", 0.05)),
            right_inset_ratio=float(value.get("right_inset_ratio", 0.05)),
            top_ratio=float(value.get("top_ratio", 0.3)),
            bottom_inset_ratio=float(value.get("bottom_inset_ratio", 0.3)),
        )


@dataclass(slots=True)
class DetectedGameWindow:
    hwnd: int = 0
    title: str = ""
    process_name: str = ""
    pid: int = 0


@dataclass(slots=True)
class OcrReaderRuntime:
    enabled: bool = False
    status: str = "disabled"
    detail: str = ""
    process_name: str = ""
    pid: int = 0
    window_title: str = ""
    game_id: str = ""
    session_id: str = ""
    last_seq: int = 0
    last_event_ts: str = ""
    capture_profile: dict[str, float] = field(default_factory=dict)
    tesseract_path: str = ""
    languages: str = ""
    takeover_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "status": self.status,
            "detail": self.detail,
            "process_name": self.process_name,
            "pid": self.pid,
            "window_title": self.window_title,
            "game_id": self.game_id,
            "session_id": self.session_id,
            "last_seq": self.last_seq,
            "last_event_ts": self.last_event_ts,
            "capture_profile": dict(self.capture_profile),
            "tesseract_path": self.tesseract_path,
            "languages": self.languages,
            "takeover_reason": self.takeover_reason,
        }


@dataclass(slots=True)
class OcrReaderTickResult:
    warnings: list[str] = field(default_factory=list)
    should_rescan: bool = False
    runtime: dict[str, Any] = field(default_factory=dict)


class CaptureBackend(Protocol):
    def is_available(self) -> bool: ...

    def describe_target(self, target: DetectedGameWindow) -> str: ...

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any: ...


class OcrBackend(Protocol):
    def is_available(self) -> bool: ...

    def extract_text(self, image: Any) -> str: ...


class Win32CaptureBackend:
    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def is_available(self) -> bool:
        try:
            import win32gui
            import win32ui
            import win32con
            return True
        except ImportError:
            return False

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any:
        import win32gui
        import win32ui
        import win32con
        from PIL import Image

        hwnd = target.hwnd
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid window dimensions: {width}x{height}")

        hdc = win32gui.GetWindowDC(hwnd)
        if not hdc:
            raise RuntimeError("Failed to get window DC")

        bmp = None
        mem_dc = None
        hdc_mem = None
        try:
            hdc_mem = win32ui.CreateDCFromHandle(hdc)
            mem_dc = hdc_mem.CreateCompatibleDC()

            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(hdc_mem, width, height)
            mem_dc.SelectObject(bmp)

            # Try PrintWindow with PW_RENDERFULLCONTENT (3) for better game capture
            PW_RENDERFULLCONTENT = 3
            success = ctypes.windll.user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
            if not success:
                mem_dc.BitBlt((0, 0), (width, height), hdc_mem, (0, 0), win32con.SRCCOPY)

            bmp_info = bmp.GetInfo()
            bmp_str = bmp.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_str,
                "raw",
                "BGRX",
                0,
                1,
            )
        finally:
            if mem_dc is not None:
                mem_dc.DeleteDC()
            if hdc_mem is not None:
                hdc_mem.DeleteDC()
            if bmp is not None:
                win32gui.DeleteObject(bmp.GetHandle())
            win32gui.ReleaseDC(hwnd, hdc)

        left = int(width * profile.left_inset_ratio)
        right = int(width * (1.0 - profile.right_inset_ratio))
        top = int(height * profile.top_ratio)
        bottom = int(height * (1.0 - profile.bottom_inset_ratio))

        left = max(0, min(left, width))
        right = max(left, min(right, width))
        top = max(0, min(top, height))
        bottom = max(top, min(bottom, height))

        crop_w = right - left
        crop_h = bottom - top
        if crop_w < 10 or crop_h < 10:
            raise RuntimeError(f"Crop region too small: {crop_w}x{crop_h}")

        return image.crop((left, top, right, bottom))


class TesseractOcrBackend:
    def __init__(self, *, tesseract_path: str = "", languages: str = "") -> None:
        self._tesseract_path = tesseract_path
        self._languages = languages

    def is_available(self) -> bool:
        path = resolve_tesseract_path(self._tesseract_path)
        if not path:
            return False
        inspection = inspect_tesseract_installation(
            configured_path=self._tesseract_path,
            install_target_dir_raw="",
            languages=self._languages,
        )
        return bool(inspection.get("installed"))

    def extract_text(self, image: Any) -> str:
        import pytesseract
        path = resolve_tesseract_path(self._tesseract_path)
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
        lang = self._languages
        # PSM 6 = Assume a single uniform block of text (good for VN dialogue boxes)
        config = "--psm 6"
        text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text.strip()


def _default_window_scanner() -> list[DetectedGameWindow]:
    try:
        import win32gui
        import win32process
    except ImportError:
        return []

    results: list[tuple[int, DetectedGameWindow]] = []

    excluded_classes = {
        "Shell_TrayWnd",
        "Windows.UI.Core.CoreWindow",
        "ApplicationFrameWindow",
        "Windows.UI.Composition.DesktopWindowContentBridge",
    }
    excluded_title_substrings = {
        "program manager",
        "settings",
        "microsoft text input application",
        "nvidia overlay",
        "task manager",
        "visual studio code",
        "obs",
    }

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.IsIconic(hwnd):
            return
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width < 400 or height < 300:
            return
        title = win32gui.GetWindowText(hwnd)
        if not title or len(title) < 2:
            return
        class_name = win32gui.GetClassName(hwnd)
        if class_name in excluded_classes:
            return
        lower_title = title.lower()
        if any(ex in lower_title for ex in excluded_title_substrings):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = ""
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                process_name = proc.name()
            except Exception:
                pass
        area = width * height
        results.append((area, DetectedGameWindow(hwnd=hwnd, title=title, process_name=process_name, pid=pid)))

    win32gui.EnumWindows(callback, None)
    results.sort(key=lambda item: -item[0])
    return [item[1] for item in results]


def _is_windows_platform() -> bool:
    return os.name == "nt"


class OcrReaderBridgeWriter:
    def __init__(
        self,
        *,
        bridge_root: Path,
        version: str = OCR_READER_BRIDGE_VERSION,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._bridge_root = bridge_root
        self._version = version
        self._time_fn = time_fn or time.time
        self._game_id = ""
        self._session_id = ""
        self._process_name = ""
        self._pid = 0
        self._window_title = ""
        self._engine = OCR_READER_DEFAULT_ENGINE
        self._started_at = ""
        self._last_seq = 0
        self._last_event_ts = ""
        self._state = self._initial_state("")
        self._text_to_line_id: dict[str, str] = {}
        self._line_id_owner: dict[str, str] = {}

    @property
    def bridge_root(self) -> Path:
        return self._bridge_root

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def last_seq(self) -> int:
        return self._last_seq

    @property
    def last_event_ts(self) -> str:
        return self._last_event_ts

    def start_session(self, window: DetectedGameWindow) -> None:
        started_at = utc_now_iso(self._time_fn())
        self._game_id = _ocr_game_id_from_process(window.process_name or window.title)
        self._session_id = f"ocr-{uuid4()}"
        self._process_name = window.process_name
        self._pid = window.pid
        self._window_title = window.title
        self._engine = OCR_READER_DEFAULT_ENGINE
        self._started_at = started_at
        self._last_seq = 0
        self._last_event_ts = started_at
        self._state = self._initial_state(started_at)
        self._text_to_line_id.clear()
        self._line_id_owner.clear()
        self._bridge_dir().mkdir(parents=True, exist_ok=True)
        self._events_path().write_bytes(b"")
        self._write_session_snapshot()
        self._append_event(
            "session_started",
            {
                "game_title": window.title or window.process_name,
                "engine": self._engine,
                "locale": "",
                "started_at": started_at,
                "scene_id": self._state["scene_id"],
                "line_id": self._state["line_id"],
                "route_id": self._state["route_id"],
                "is_menu_open": self._state["is_menu_open"],
                "speaker": self._state["speaker"],
                "text": self._state["text"],
                "choices": self._state["choices"],
                "save_context": self._state["save_context"],
            },
            ts=started_at,
        )

    def emit_line(self, raw_text: str, *, ts: str) -> bool:
        cleaned = raw_text.strip()
        if not cleaned or not self._session_id:
            return False
        speaker, text = self._split_speaker_text(cleaned)
        if not text:
            return False
        line_id = self._line_id_for_text(text)
        self._state = {
            **self._state,
            "speaker": speaker,
            "text": text,
            "choices": [],
            "scene_id": OCR_READER_UNKNOWN_SCENE,
            "line_id": line_id,
            "route_id": OCR_READER_ROUTE_ID,
            "is_menu_open": False,
            "save_context": self._state.get("save_context", {"kind": "unknown", "slot_id": "", "display_name": ""}),
            "ts": ts,
        }
        self._append_event(
            "line_changed",
            {
                "speaker": speaker,
                "text": text,
                "line_id": line_id,
                "line_id_source": "text_hash",
                "scene_id": self._state["scene_id"],
                "route_id": self._state["route_id"],
            },
            ts=ts,
        )
        return True

    def emit_choices(self, choices: list[str], *, ts: str) -> bool:
        if not choices or not self._session_id:
            return False
        line_id = str(self._state.get("line_id") or "")
        if not line_id:
            return False
        payload_choices = [
            {
                "choice_id": f"{line_id}#choice{index}",
                "text": text,
                "index": index,
                "enabled": True,
            }
            for index, text in enumerate(choices)
        ]
        self._state = {
            **self._state,
            "choices": payload_choices,
            "is_menu_open": True,
            "ts": ts,
        }
        self._append_event(
            "choices_shown",
            {
                "line_id": line_id,
                "scene_id": self._state["scene_id"],
                "route_id": self._state["route_id"],
                "choices": payload_choices,
            },
            ts=ts,
        )
        return True

    def emit_heartbeat(self, *, ts: str) -> bool:
        if not self._session_id:
            return False
        self._append_event(
            "heartbeat",
            {
                "state_ts": str(self._state.get("ts") or ""),
                "idle_seconds": 0,
                "scene_id": self._state["scene_id"],
                "line_id": self._state["line_id"],
                "route_id": self._state["route_id"],
            },
            ts=ts,
            update_snapshot=False,
        )
        return True

    def emit_error(self, message: str, *, ts: str, details: dict[str, Any] | None = None) -> bool:
        if not self._session_id:
            return False
        payload: dict[str, Any] = {
            "message": message,
            "source": DATA_SOURCE_OCR_READER,
            "scene_id": self._state["scene_id"],
            "line_id": self._state["line_id"],
            "route_id": self._state["route_id"],
        }
        if details:
            payload["details"] = dict(details)
        self._append_event("error", payload, ts=ts, update_snapshot=False)
        return True

    def end_session(self, *, ts: str) -> bool:
        if not self._session_id:
            return False
        payload = {
            "scene_id": self._state["scene_id"],
            "line_id": self._state["line_id"],
            "route_id": self._state["route_id"],
        }
        self._append_event("session_ended", payload, ts=ts, update_snapshot=False)
        return True

    def runtime(self) -> OcrReaderRuntime:
        return OcrReaderRuntime(
            enabled=True,
            status="active" if self._session_id else "idle",
            detail="",
            process_name=self._process_name,
            pid=self._pid,
            window_title=self._window_title,
            game_id=self._game_id,
            session_id=self._session_id,
            last_seq=self._last_seq,
            last_event_ts=self._last_event_ts,
        )

    def _initial_state(self, ts: str) -> dict[str, Any]:
        return {
            "speaker": "",
            "text": "",
            "choices": [],
            "scene_id": OCR_READER_UNKNOWN_SCENE,
            "line_id": "",
            "route_id": OCR_READER_ROUTE_ID,
            "is_menu_open": False,
            "save_context": {"kind": "unknown", "slot_id": "", "display_name": ""},
            "ts": ts,
        }

    def _bridge_dir(self) -> Path:
        return self._bridge_root / self._game_id

    def _session_path(self) -> Path:
        return self._bridge_dir() / "session.json"

    def _events_path(self) -> Path:
        return self._bridge_dir() / "events.jsonl"

    def _session_snapshot(self) -> dict[str, Any]:
        return {
            "protocol_version": 1,
            "game_id": self._game_id,
            "game_title": self._window_title or self._process_name,
            "engine": self._engine,
            "session_id": self._session_id,
            "started_at": self._started_at,
            "last_seq": self._last_seq,
            "locale": "",
            "bridge_sdk_version": self._version,
            "metadata": {
                "source": DATA_SOURCE_OCR_READER,
                "game_process_name": self._process_name,
                "game_pid": self._pid,
                "window_title": self._window_title,
            },
            "state": {
                "speaker": str(self._state.get("speaker") or ""),
                "text": str(self._state.get("text") or ""),
                "choices": list(self._state.get("choices", [])),
                "scene_id": str(self._state.get("scene_id") or OCR_READER_UNKNOWN_SCENE),
                "line_id": str(self._state.get("line_id") or ""),
                "route_id": str(self._state.get("route_id") or OCR_READER_ROUTE_ID),
                "is_menu_open": bool(self._state.get("is_menu_open", False)),
                "save_context": dict(self._state.get("save_context", {"kind": "unknown", "slot_id": "", "display_name": ""})),
                "ts": str(self._state.get("ts") or self._started_at),
            },
        }

    def _write_session_snapshot(self) -> None:
        self._bridge_dir().mkdir(parents=True, exist_ok=True)
        tmp_path = self._session_path().with_suffix(".json.tmp")
        payload = json.dumps(
            self._session_snapshot(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with tmp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._session_path())

    def _append_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        ts: str,
        update_snapshot: bool = True,
    ) -> None:
        self._last_seq += 1
        self._last_event_ts = ts
        event = {
            "protocol_version": 1,
            "seq": self._last_seq,
            "ts": ts,
            "type": event_type,
            "session_id": self._session_id,
            "game_id": self._game_id,
            "payload": payload,
        }
        with self._events_path().open("ab") as handle:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            handle.flush()
        if update_snapshot:
            self._write_session_snapshot()
            return
        self._write_session_snapshot()

    def _line_id_for_text(self, text: str) -> str:
        normalized = normalize_text(text)
        cached = self._text_to_line_id.get(normalized)
        if cached is not None:
            return cached
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        widths = list(range(12, len(digest) + 1, 4))
        if widths[-1] != len(digest):
            widths.append(len(digest))
        for width in widths:
            candidate = f"ocr:{digest[:width]}"
            owner = self._line_id_owner.get(candidate)
            if owner in {None, normalized}:
                self._line_id_owner[candidate] = normalized
                self._text_to_line_id[normalized] = candidate
                return candidate
        suffix = 1
        while True:
            candidate = f"ocr:{digest}#{suffix}"
            owner = self._line_id_owner.get(candidate)
            if owner in {None, normalized}:
                self._line_id_owner[candidate] = normalized
                self._text_to_line_id[normalized] = candidate
                return candidate
            suffix += 1

    @staticmethod
    def _split_speaker_text(raw_text: str) -> tuple[str, str]:
        _SPEAKER_QUOTE_RE = re.compile(r"^\s*([^「」:：]{1,40})[「『](.+)[」』]\s*$")
        _SPEAKER_COLON_RE = re.compile(r"^\s*([^:：]{1,40})[:：]\s*(.+\S)\s*$")
        match = _SPEAKER_QUOTE_RE.match(raw_text)
        if match is not None:
            return match.group(1).strip(), match.group(2).strip()
        match = _SPEAKER_COLON_RE.match(raw_text)
        if match is not None:
            return match.group(1).strip(), match.group(2).strip()
        return "", raw_text.strip()


class OcrReaderManager:
    def __init__(
        self,
        *,
        logger,
        config: GalgameConfig,
        time_fn: Callable[[], float] | None = None,
        platform_fn: Callable[[], bool] | None = None,
        window_scanner: Callable[[], list[DetectedGameWindow]] | None = None,
        capture_backend: CaptureBackend | None = None,
        ocr_backend: OcrBackend | None = None,
        writer: OcrReaderBridgeWriter | None = None,
    ) -> None:
        self._logger = logger
        self._config = config
        self._time_fn = time_fn or time.time
        self._platform_fn = platform_fn or _is_windows_platform
        self._window_scanner = window_scanner or _default_window_scanner
        self._capture_backend = capture_backend or Win32CaptureBackend(logger=logger)
        self._ocr_backend = ocr_backend
        self._writer = writer or OcrReaderBridgeWriter(
            bridge_root=config.bridge_root,
            time_fn=self._time_fn,
        )
        self._runtime = OcrReaderRuntime(enabled=config.ocr_reader_enabled)
        self._capture_profiles: dict[str, OcrCaptureProfile] = {}
        self._last_memory_reader_text_at = 0.0
        self._last_heartbeat_at = 0.0
        self._attached_window: DetectedGameWindow | None = None
        self._last_raw_ocr_text = ""
        self._ocr_repeat_count = 0
        self._stable_ocr_text = ""

    def update_config(self, config: GalgameConfig) -> None:
        self._config = config
        self._runtime.enabled = config.ocr_reader_enabled
        if self._writer.bridge_root != config.bridge_root:
            self._writer = OcrReaderBridgeWriter(
                bridge_root=config.bridge_root,
                time_fn=self._time_fn,
            )
        if self._ocr_backend is None:
            self._ocr_backend = TesseractOcrBackend(
                tesseract_path=config.ocr_reader_tesseract_path,
                languages=config.ocr_reader_languages,
            )

    def update_capture_profiles(self, profiles: dict[str, dict[str, float]]) -> None:
        self._capture_profiles = {}
        for process_name, profile_dict in profiles.items():
            try:
                self._capture_profiles[str(process_name)] = OcrCaptureProfile.from_dict(profile_dict)
            except Exception as exc:
                self._logger.warning("ocr_reader failed to parse capture profile for %s: %s", process_name, exc)

    def runtime(self) -> dict[str, Any]:
        return self._runtime.to_dict()

    async def shutdown(self) -> None:
        if self._writer.session_id:
            self._writer.end_session(ts=utc_now_iso(self._time_fn()))
        self._attached_window = None

    async def tick(
        self,
        *,
        bridge_sdk_available: bool,
        memory_reader_runtime: dict[str, Any],
    ) -> OcrReaderTickResult:
        now = self._time_fn()
        result = OcrReaderTickResult(runtime=self._runtime.to_dict())

        if not self._config.ocr_reader_enabled:
            self._runtime = OcrReaderRuntime(enabled=False, status="disabled", detail="disabled_by_config")
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        if not self._platform_fn():
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="unsupported_platform",
            )
            await self._end_session_if_needed(now)
            result.warnings.append("ocr_reader is Windows-only")
            result.runtime = self._runtime.to_dict()
            return result

        tesseract_path = resolve_tesseract_path(self._config.ocr_reader_tesseract_path)
        if not tesseract_path:
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="missing_tesseract",
            )
            await self._end_session_if_needed(now)
            result.warnings.append("ocr_reader Tesseract is missing or not configured")
            result.runtime = self._runtime.to_dict()
            return result

        inspection = inspect_tesseract_installation(
            configured_path=self._config.ocr_reader_tesseract_path,
            install_target_dir_raw=self._config.ocr_reader_install_target_dir,
            languages=self._config.ocr_reader_languages,
        )
        if inspection.get("detail") == "missing_languages":
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="missing_languages",
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
            )
            await self._end_session_if_needed(now)
            missing = inspection.get("missing_languages", [])
            result.warnings.append(f"ocr_reader Tesseract is missing languages: {missing}")
            result.runtime = self._runtime.to_dict()
            return result

        if bridge_sdk_available:
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="bridge_sdk_available",
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
                game_id=self._runtime.game_id,
                session_id=self._runtime.session_id,
                last_seq=self._runtime.last_seq,
                last_event_ts=self._runtime.last_event_ts,
            )
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        memory_reader_has_text = bool(
            memory_reader_runtime.get("last_seq", 0) > 1
            or memory_reader_runtime.get("detail") == "receiving_text"
        )
        if memory_reader_has_text:
            self._last_memory_reader_text_at = now
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="memory_reader_active",
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
                game_id=self._runtime.game_id,
                session_id=self._runtime.session_id,
                last_seq=self._runtime.last_seq,
                last_event_ts=self._runtime.last_event_ts,
            )
            result.runtime = self._runtime.to_dict()
            return result

        if self._last_memory_reader_text_at > 0:
            elapsed = now - self._last_memory_reader_text_at
            threshold = float(self._config.ocr_reader_no_text_takeover_after_seconds)
            if elapsed < threshold:
                self._runtime = OcrReaderRuntime(
                    enabled=True,
                    status="idle",
                    detail="waiting_for_takeover_window",
                    tesseract_path=tesseract_path,
                    languages=self._config.ocr_reader_languages,
                    game_id=self._runtime.game_id,
                    session_id=self._runtime.session_id,
                    last_seq=self._runtime.last_seq,
                    last_event_ts=self._runtime.last_event_ts,
                )
                result.runtime = self._runtime.to_dict()
                return result

        if not self._capture_backend.is_available():
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="candidate",
                detail="capture_backend_unavailable",
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
                takeover_reason="capture_backend_not_available",
                game_id=self._runtime.game_id,
                session_id=self._runtime.session_id,
                last_seq=self._runtime.last_seq,
                last_event_ts=self._runtime.last_event_ts,
            )
            await self._end_session_if_needed(now)
            result.warnings.append("ocr_reader capture backend is not available")
            result.runtime = self._runtime.to_dict()
            return result

        windows = self._window_scanner()
        if not windows:
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="idle",
                detail="waiting_for_capture_target",
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
                game_id=self._runtime.game_id,
                session_id=self._runtime.session_id,
                last_seq=self._runtime.last_seq,
                last_event_ts=self._runtime.last_event_ts,
            )
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        target = windows[0]
        profile = self._capture_profiles.get(target.process_name, OcrCaptureProfile(
            left_inset_ratio=self._config.ocr_reader_left_inset_ratio,
            right_inset_ratio=self._config.ocr_reader_right_inset_ratio,
            top_ratio=self._config.ocr_reader_top_ratio,
            bottom_inset_ratio=self._config.ocr_reader_bottom_inset_ratio,
        ))

        if self._attached_window is None or self._attached_window.pid != target.pid:
            if not self._writer.session_id or self._writer.game_id != _ocr_game_id_from_process(target.process_name or target.title):
                self._writer.start_session(target)
                result.should_rescan = True
            self._attached_window = target
            self._last_heartbeat_at = now
            self._last_raw_ocr_text = ""
            self._ocr_repeat_count = 0
            self._stable_ocr_text = ""
            self._runtime = OcrReaderRuntime(
                enabled=True,
                status="starting",
                detail="starting_capture",
                process_name=target.process_name,
                pid=target.pid,
                window_title=target.title,
                game_id=self._writer.game_id,
                session_id=self._writer.session_id,
                last_seq=self._writer.last_seq,
                last_event_ts=self._writer.last_event_ts,
                capture_profile=profile.to_dict(),
                tesseract_path=tesseract_path,
                languages=self._config.ocr_reader_languages,
            )
            result.runtime = self._runtime.to_dict()
            return result

        if self._attached_window is not None:
            self._attached_window = target

        emitted = False
        try:
            frame = self._capture_backend.capture_frame(target, profile)
            raw_text = self._ocr_backend.extract_text(frame)
            emitted = self._consume_ocr_text(raw_text, now=now)
        except Exception as exc:
            self._logger.warning("ocr_reader capture/OCR failed: %s", exc)
            result.warnings.append(f"ocr_reader capture failed: {exc}")

        status = self._runtime.status
        detail = self._runtime.detail

        if emitted:
            result.should_rescan = True
            self._last_heartbeat_at = now
            status = "active"
            detail = "receiving_text"
        elif self._writer.session_id and now - self._last_heartbeat_at >= float(self._config.ocr_reader_poll_interval_seconds):
            if self._writer.emit_heartbeat(ts=utc_now_iso(now)):
                result.should_rescan = True
                self._last_heartbeat_at = now
            if status == "starting":
                status = "active"
            if detail == "starting_capture":
                detail = "attached_no_text_yet"

        self._runtime = OcrReaderRuntime(
            enabled=True,
            status=status,
            detail=detail,
            process_name=target.process_name,
            pid=target.pid,
            window_title=target.title,
            game_id=self._writer.game_id,
            session_id=self._writer.session_id,
            last_seq=self._writer.last_seq,
            last_event_ts=self._writer.last_event_ts,
            capture_profile=profile.to_dict(),
            tesseract_path=tesseract_path,
            languages=self._config.ocr_reader_languages,
        )
        result.runtime = self._runtime.to_dict()
        return result

    def _consume_ocr_text(self, raw_text: str, *, now: float) -> bool:
        cleaned = normalize_text(raw_text)
        if not cleaned:
            return False

        if cleaned == self._last_raw_ocr_text:
            self._ocr_repeat_count += 1
        else:
            self._ocr_repeat_count = 1
            self._last_raw_ocr_text = cleaned

        if self._ocr_repeat_count < 2:
            return False

        if cleaned == self._stable_ocr_text:
            return False

        self._stable_ocr_text = cleaned

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        choices = _coerce_choice_lines(lines)
        if choices:
            return self._writer.emit_choices(choices, ts=utc_now_iso(now))

        return self._writer.emit_line(raw_text, ts=utc_now_iso(now))

    async def _end_session_if_needed(self, now: float) -> None:
        if self._writer.session_id:
            self._writer.end_session(ts=utc_now_iso(now))
            self._attached_window = None
            self._last_raw_ocr_text = ""
            self._ocr_repeat_count = 0
            self._stable_ocr_text = ""
