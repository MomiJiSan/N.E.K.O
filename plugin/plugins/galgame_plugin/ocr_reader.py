from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import os
import re
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4

from .models import (
    ADVANCE_SPEED_FAST,
    ADVANCE_SPEED_MEDIUM,
    ADVANCE_SPEED_SLOW,
    ADVANCE_SPEEDS,
    DATA_SOURCE_OCR_READER,
    DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_TOP_RATIO,
    GalgameConfig,
    OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUCKET_ASPECT_NEAREST,
    OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUCKET_EXACT,
    OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUILTIN_PRESET,
    OCR_CAPTURE_PROFILE_MATCH_SOURCE_CONFIG_DEFAULT,
    OCR_CAPTURE_PROFILE_MATCH_SOURCE_PROCESS_FALLBACK,
    OCR_CAPTURE_PROFILE_RATIO_KEYS,
    OCR_CAPTURE_PROFILE_STAGE_DEFAULT,
    OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
    OCR_CAPTURE_PROFILE_STAGE_MENU,
    OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY,
    OCR_TRIGGER_MODE_AFTER_ADVANCE,
    build_ocr_capture_profile_bucket_key,
    compute_ocr_window_aspect_ratio,
    parse_ocr_capture_profile_bucket_key,
)
from .rapidocr_support import (
    inspect_rapidocr_installation,
    load_rapidocr_runtime,
)
from .reader import normalize_text
from .tesseract_support import inspect_tesseract_installation, resolve_tesseract_path

_MODULE_LOGGER = logging.getLogger(__name__)

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
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_CHAR_RE = re.compile(r"[\u3040-\u30ff]")
_WINDOW_SPACE_RE = re.compile(r"\s+")
_SELF_WINDOW_TITLE_SUBSTRINGS = (
    "n.e.k.o",
    "plugin manager",
    "插件管理",
    "galgame plugin",
    "phase 2",
)
_SELF_WINDOW_PATH_SUBSTRINGS = (
    "n.e.k.o",
    "galgame_plugin",
)
_OVERLAY_WINDOW_TITLE_SUBSTRINGS = (
    "nvidia overlay",
    "overlay",
    "launcher",
    "task manager",
    "visual studio code",
    "obs",
    "program manager",
    "settings",
    "microsoft text input application",
)
_OVERLAY_PROCESS_NAME_SUBSTRINGS = (
    "nvidia",
    "overlay",
    "launcher",
    "gamebar",
    "obs",
    "code",
    "steamwebhelper",
)
_AUTO_TARGET_DENY_PROCESS_NAMES = {
    "applicationframehost.exe",
    "chrome.exe",
    "cmd.exe",
    "code.exe",
    "explorer.exe",
    "firefox.exe",
    "msedge.exe",
    "notepad.exe",
    "powershell.exe",
    "windowsterminal.exe",
    "winword.exe",
    "wps.exe",
}
_HELPER_CLASS_NAMES = {
    "Shell_TrayWnd",
    "Windows.UI.Core.CoreWindow",
    "ApplicationFrameWindow",
    "Windows.UI.Composition.DesktopWindowContentBridge",
}
_SELF_UI_GUARD_SUBSTRINGS = (
    ".agent",
    ".codex",
    ".codex_tmp",
    ".codex_pytest_tmp",
    "__pycache__",
    "-pycache_",
    "codex_tmp",
    "documents\\code\\n.e.k.o",
    "rapidocr",
    "tesseract",
    "ocr compatibility fallback",
    "install queued task",
    "plugin manager",
    "galgame plugin",
    "n.e.k.o",
    "插件设置",
    "运行控制",
    "模式静默",
    "静默进入待机",
    "进入待机",
    "恢复活跃",
    "推送通知",
    "推进速度",
    "保存设置",
    "ocr 目标窗口",
    "ocr目标窗口",
    "等待 ocr 窗口候选列表",
    "等待ocr窗口候选列表",
    "查看排除窗口",
    "选择识别窗口",
    "截图校准",
    "最近稳定台词",
    "stable 与 observed",
    "当前台词解释",
    "场景总结",
    "游戏 agent",
    "plugin.plugins.galgame_plugin",
    "uv run python",
    "launcher.py",
    "powershell",
    "ps c:",
)
_GAME_OVERLAY_TEXT_GUARD_SUBSTRINGS = (
    "backlog",
    "history",
    "skip",
    "auto",
    "config",
    "system",
    "load",
    "save",
    "menu",
    "回想",
    "历史",
    "履历",
    "快进",
    "跳过",
    "自动",
    "菜单",
    "设置",
    "系统",
    "存档",
    "读档",
)
_AIHONG_PROCESS_NAMES = frozenset({"thelamentinggeese.exe"})
_AIHONG_TITLE_SUBSTRINGS = ("哀鸿", "aihong")
_AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET = {
    "left_inset_ratio": 0.0,
    "right_inset_ratio": 0.0,
    "top_ratio": 0.60,
    "bottom_inset_ratio": 0.05,
}
_AIHONG_MENU_CAPTURE_PROFILE_PRESET = {
    "left_inset_ratio": 0.0,
    "right_inset_ratio": 0.0,
    "top_ratio": 0.0,
    "bottom_inset_ratio": 0.0,
}
_AIHONG_DIALOGUE_STAGE = OCR_CAPTURE_PROFILE_STAGE_DIALOGUE
_AIHONG_MENU_STAGE = OCR_CAPTURE_PROFILE_STAGE_MENU
_AIHONG_MENU_MIN_LINES = 2
_AIHONG_MENU_MAX_LINES = 4
_AIHONG_MENU_MIN_SIGNIFICANT_CHARS = 2
_AIHONG_MENU_MAX_SIGNIFICANT_CHARS = 10
_AIHONG_MENU_STATUS_KEYWORDS = ("银两剩余", "余额", "剩余")
_AIHONG_MENU_AMOUNT_RE = re.compile(r"^\s*\d+\s*两\S{0,3}\s*$")
_AIHONG_MENU_DIALOGUE_MARKERS = (
    ",",
    ".",
    ":",
    ";",
    "?",
    "!",
    "[",
    "]",
    "，",
    "。",
    "：",
    "；",
    "？",
    "！",
    "「",
    "」",
    "【",
    "】",
)
_DIALOGUE_LINE_MARKERS = (":", "：", "「", "」")
_OCR_DIALOGUE_STRONG_PUNCTUATION_RE = re.compile(r"[。！？!?…]|——|「|」|『|』|“|”")
_OCR_DIALOGUE_WEAK_PUNCTUATION_RE = re.compile(r"[，,、：:]")
_OCR_TRAILING_GARBAGE_AFTER_SENTENCE_RE = re.compile(r"([。！？!?…」』”\]］])\s*[号口日曰益]\s*$")
_OCR_TRAILING_GARBAGE_AFTER_BRACKET_RE = re.compile(
    r"([\]］）】」』”])\s*[^。！？!?…，,、：:；;「」『』“”\[\]［］【】（）()]{1,4}\s*$"
)
_OCR_TRAILING_GARBAGE_AFTER_DASH_RE = re.compile(
    r"((?:——|--|—|－|-))\s*[^。！？!?…，,、：:「」『』“”\[\]［］【】（）()]{1,4}\s*$"
)
_OCR_GAME_OVERLAY_KEYWORD_MIN_LINES = 2
_OCR_GAME_OVERLAY_KEYWORD_MAX_SIGNIFICANT_CHARS = 40
_OCR_GAME_OVERLAY_MIN_LINES = 4
_OCR_GAME_OVERLAY_MIN_DIALOGUE_LINES = 2
_OCR_DIALOGUE_MIN_SIGNIFICANT_CHARS = 2
_OCR_DIALOGUE_MAX_SIGNIFICANT_CHARS = 220
_OCR_DIALOGUE_WEAK_PUNCTUATION_MIN_SIGNIFICANT_CHARS = 8
_OCR_NOISE_MAX_NON_CJK_SIGNIFICANT_CHARS = 2
_OCR_STABLE_TEXT_MIN_REPEAT_THRESHOLD = 1
_OCR_STABLE_TEXT_DEFAULT_REPEAT_THRESHOLD = 2
_OCR_LINE_REPEAT_THRESHOLD_FAST = 1
_OCR_LINE_REPEAT_THRESHOLD_MEDIUM = 2
_OCR_LINE_REPEAT_THRESHOLD_SLOW = 3
_OCR_CHOICES_REPEAT_THRESHOLD = 2
_OCR_FOLLOWUP_CONFIRM_REPEAT_COUNT = 1
_OCR_CAPTURE_DIAGNOSTIC_NO_TEXT_POLLS = 3
_AIHONG_MENU_MISSING_MAX_POLLS = 2
_AIHONG_DIALOGUE_IDLE_BEFORE_MENU_PROBE_POLLS = 2
_AIHONG_MENU_STATUS_IDLE_POLLS = 1
_AFTER_ADVANCE_LINE_REPEAT_THRESHOLD = 1
_AFTER_ADVANCE_BACKGROUND_CONFIRM_POLLS = 1
_WINDOW_TITLE_MIN_CHARS = 2
_WINDOW_SINGLE_FALLBACK_CANDIDATE_COUNT = 1
_OCR_FOLLOWUP_CONFIRM_DELAY_SECONDS = 0.18
_CAPTURE_BACKEND_AUTO = "auto"
_CAPTURE_BACKEND_DXCAM = "dxcam"
_CAPTURE_BACKEND_IMAGEGRAB = "imagegrab"
_CAPTURE_BACKEND_PRINTWINDOW = "printwindow"
_STALE_CAPTURE_FRAME_THRESHOLD = 3
_BACKGROUND_HASH_MIN_INTERVAL_SECONDS = 1.0
_BACKGROUND_HASH_BOTTOM_INSET_RATIO = 0.45
_BACKEND_PLAN_CACHE_TTL_SECONDS = 5.0
_BACKGROUND_SCENE_HASH_SIZE = 8
_BACKGROUND_SCENE_CHANGE_DISTANCE = 18
_BACKGROUND_SCENE_CHANGE_CONFIRM_POLLS = 2
_PENDING_VISUAL_SCENE_MAX_AGE_SECONDS = 2.0
_OCR_LINE_ID_MAX_COLLISION_SUFFIX = 10000
_OCR_AUTO_RECALIBRATE_MAX_SECONDS = 15.0
_OCR_AUTO_RECALIBRATE_MAX_OCR_ATTEMPTS = 96
_OCR_AUTO_RECALIBRATE_IMAGE_SIZE_DIMENSIONS = 2
_OCR_RATIO_PERCENT_BASIS = 100
_OCR_RATIO_ROUND_DIGITS = 2
_OCR_RATIO_MIN = 0.0
_OCR_RATIO_MAX = 0.98
_OCR_AUTO_RECALIBRATE_HORIZONTAL_MAX_INSET_RATIO = 0.45
_OCR_AUTO_RECALIBRATE_MAX_TOTAL_HORIZONTAL_INSET_RATIO = 0.95
_OCR_AUTO_RECALIBRATE_AIHONG_HORIZONTAL_PAIRS = (
    (0.0, 0.0),
    (0.02, 0.02),
    (0.05, 0.05),
)
_OCR_AUTO_RECALIBRATE_HORIZONTAL_SHRINK_DELTA = 0.05
_OCR_AUTO_RECALIBRATE_TOP_SCAN_DELTA_START = -0.14
_OCR_AUTO_RECALIBRATE_TOP_SCAN_DELTA_END = 0.08
_OCR_AUTO_RECALIBRATE_TOP_SCAN_STEP = 0.02
_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_DELTA_START = -0.04
_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_DELTA_END = 0.08
_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_STEP = 0.02
_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_DELTA_START = -0.08
_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_DELTA_END = 0.08
_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_STEP = 0.02
_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_DELTA_START = -0.05
_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_DELTA_END = 0.08
_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_STEP = 0.01
_OCR_AUTO_RECALIBRATE_MIN_CROP_HEIGHT_PX = 24
_OCR_AUTO_RECALIBRATE_MIN_CROP_HEIGHT_RATIO = 0.08
_OCR_AUTO_RECALIBRATE_MAX_CROP_HEIGHT_RATIO = 0.45
_OCR_AUTO_RECALIBRATE_MIN_CROP_WIDTH_PX = 10
_OCR_AUTO_RECALIBRATE_MIN_CANDIDATE_SIGNIFICANT_CHARS = 8
_OCR_AUTO_RECALIBRATE_SUMMARY_SAMPLE_CHARS = 24
_OCR_AUTO_RECALIBRATE_PREFERRED_BOTTOM_DELTAS = (0.0, 0.02, -0.02, 0.04)
_OCR_AUTO_RECALIBRATE_AIHONG_PREFERRED_TOP_DELTAS = (0.0, -0.02, 0.02)
_OCR_AUTO_RECALIBRATE_BASE_PREFERRED_TOP_DELTAS = (0.0, -0.02, 0.02)
_OCR_AUTO_RECALIBRATE_REFINE_TOP_DELTAS = (-0.02, 0.0, 0.02)


def utc_now_iso(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now))


def _ocr_game_id_from_process(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{OCR_READER_GAME_ID_PREFIX}{digest}"


def _normalize_window_title(value: str) -> str:
    normalized = _WINDOW_SPACE_RE.sub(" ", str(value or "").strip().lower())
    return normalized


def _build_window_key(*, process_name: str, pid: int, hwnd: int, title: str) -> str:
    payload = f"{process_name.strip().lower()}|{max(0, int(pid))}|{max(0, int(hwnd))}|{_normalize_window_title(title)}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"ocrwin:{digest}"


def _looks_like_self_window_title(title: str) -> bool:
    normalized = _normalize_window_title(title)
    return any(token in normalized for token in _SELF_WINDOW_TITLE_SUBSTRINGS)


def _looks_like_self_window_path(exe_path: str) -> bool:
    lowered = str(exe_path or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in _SELF_WINDOW_PATH_SUBSTRINGS)


def _looks_like_self_ui_text(text: str) -> bool:
    normalized = normalize_text(text).strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _SELF_UI_GUARD_SUBSTRINGS)


def _looks_like_game_overlay_text(text: str) -> bool:
    normalized = normalize_text(text).strip().lower()
    if not normalized:
        return False
    lines = _stripped_ocr_lines(normalized)
    has_overlay_keyword = any(token in normalized for token in _GAME_OVERLAY_TEXT_GUARD_SUBSTRINGS)
    if has_overlay_keyword and (
        len(lines) >= _OCR_GAME_OVERLAY_KEYWORD_MIN_LINES
        or _significant_char_count(normalized) <= _OCR_GAME_OVERLAY_KEYWORD_MAX_SIGNIFICANT_CHARS
    ):
        return True
    dialogue_like_lines = sum(1 for line in lines if _looks_like_dialogue_line(line))
    return (
        len(lines) >= _OCR_GAME_OVERLAY_MIN_LINES
        and dialogue_like_lines >= _OCR_GAME_OVERLAY_MIN_DIALOGUE_LINES
    )


def _coerce_prefixed_choice_lines(lines: list[str]) -> list[str]:
    if len(lines) < _AIHONG_MENU_MIN_LINES:
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


def _looks_like_dialogue_line(text: str) -> bool:
    normalized = normalize_text(text).strip()
    if not normalized:
        return False
    return any(marker in normalized for marker in _DIALOGUE_LINE_MARKERS)


def _looks_like_ocr_dialogue_text(text: str) -> bool:
    normalized = normalize_text(text).replace("\n", " ").strip()
    if not normalized:
        return False
    significant_chars = _significant_char_count(normalized)
    if (
        significant_chars < _OCR_DIALOGUE_MIN_SIGNIFICANT_CHARS
        or significant_chars > _OCR_DIALOGUE_MAX_SIGNIFICANT_CHARS
    ):
        return False
    if _OCR_DIALOGUE_STRONG_PUNCTUATION_RE.search(normalized):
        return True
    if (
        _OCR_DIALOGUE_WEAK_PUNCTUATION_RE.search(normalized)
        and significant_chars >= _OCR_DIALOGUE_WEAK_PUNCTUATION_MIN_SIGNIFICANT_CHARS
    ):
        return True
    return False


def _clean_ocr_dialogue_text(text: str) -> str:
    normalized = normalize_text(text).replace("\n", " ").strip()
    if not normalized:
        return ""
    previous = None
    cleaned = normalized
    while previous != cleaned:
        previous = cleaned
        cleaned = _OCR_TRAILING_GARBAGE_AFTER_SENTENCE_RE.sub(r"\1", cleaned).strip()
        cleaned = _OCR_TRAILING_GARBAGE_AFTER_BRACKET_RE.sub(r"\1", cleaned).strip()
        cleaned = _OCR_TRAILING_GARBAGE_AFTER_DASH_RE.sub(r"\1", cleaned).strip()
    return cleaned


def _coerce_plain_choice_lines(lines: list[str]) -> list[str]:
    if not _AIHONG_MENU_MIN_LINES <= len(lines) <= _AIHONG_MENU_MAX_LINES:
        return []
    choices: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = normalize_text(str(line or "")).replace("\n", " ").strip()
        if not text or _looks_like_dialogue_line(text):
            return []
        if _significant_char_count(text) > _AIHONG_MENU_MAX_SIGNIFICANT_CHARS:
            return []
        if text in seen:
            continue
        seen.add(text)
        choices.append(text)
    if not _AIHONG_MENU_MIN_LINES <= len(choices) <= _AIHONG_MENU_MAX_LINES:
        return []
    return choices


def _coerce_choice_lines(lines: list[str], *, allow_plain_text: bool = False) -> list[str]:
    choices = _coerce_prefixed_choice_lines(lines)
    if choices:
        return choices
    if allow_plain_text:
        return _coerce_plain_choice_lines(lines)
    return []


def _looks_like_aihong_dialogue_text(text: str) -> bool:
    normalized = normalize_text(text).strip()
    if not normalized:
        return False
    return any(marker in normalized for marker in _AIHONG_MENU_DIALOGUE_MARKERS)


def _looks_like_aihong_menu_status_line(text: str) -> bool:
    normalized = normalize_text(str(text or "")).replace("\n", " ").strip()
    if not normalized:
        return False
    if any(keyword in normalized for keyword in _AIHONG_MENU_STATUS_KEYWORDS):
        return True
    return bool(_AIHONG_MENU_AMOUNT_RE.match(normalized))


def _looks_like_aihong_menu_status_only_text(raw_text: str) -> bool:
    lines = _stripped_ocr_lines(raw_text)
    if not lines:
        return False
    return all(_looks_like_aihong_menu_status_line(line) for line in lines)


def _normalize_aihong_choice_box_text(text: str) -> str:
    normalized = normalize_text(str(text or "")).replace("\n", " ").strip()
    if not normalized or _looks_like_aihong_menu_status_line(normalized):
        return ""
    if normalized.endswith("手") and _significant_char_count(normalized) > _AIHONG_MENU_MIN_SIGNIFICANT_CHARS:
        normalized = normalized[:-1].strip()
    return normalized


def _aihong_choice_boxes(
    choices: list[str],
    boxes: list[OcrTextBox],
) -> list[dict[str, float] | None]:
    remaining = list(boxes)
    matched: list[dict[str, float] | None] = []
    for choice in choices:
        choice_text = normalize_text(str(choice or "")).strip()
        found_index = -1
        for index, box in enumerate(remaining):
            if _normalize_aihong_choice_box_text(box.text) == choice_text:
                found_index = index
                break
        if found_index < 0:
            matched.append(None)
            continue
        box = remaining.pop(found_index)
        matched.append(
            {
                "left": float(box.left),
                "top": float(box.top),
                "right": float(box.right),
                "bottom": float(box.bottom),
            }
        )
    return matched


def _extraction_choice_bounds_metadata(extraction: "OcrExtractionResult") -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if extraction.bounds_coordinate_space:
        metadata["bounds_coordinate_space"] = extraction.bounds_coordinate_space
    if extraction.source_size:
        metadata["source_size"] = dict(extraction.source_size)
    if extraction.capture_rect:
        metadata["capture_rect"] = dict(extraction.capture_rect)
    if extraction.window_rect:
        metadata["window_rect"] = dict(extraction.window_rect)
    return metadata


def _coerce_aihong_menu_choices(lines: list[str]) -> list[str]:
    status_lines = 0
    filtered_lines: list[str] = []
    for line in lines:
        text = normalize_text(str(line or "")).replace("\n", " ").strip()
        if not text:
            continue
        if _looks_like_aihong_menu_status_line(text):
            status_lines += 1
            continue
        if text.endswith("手") and _significant_char_count(text) > _AIHONG_MENU_MIN_SIGNIFICANT_CHARS:
            text = text[:-1].strip()
        filtered_lines.append(text)
    choices = _coerce_choice_lines(filtered_lines, allow_plain_text=True)
    if not _AIHONG_MENU_MIN_LINES <= len(choices) <= _AIHONG_MENU_MAX_LINES:
        return []
    normalized_choices: list[str] = []
    for choice in choices:
        text = normalize_text(str(choice or "")).replace("\n", " ").strip()
        if not text or _looks_like_aihong_dialogue_text(text):
            return []
        significant_chars = _significant_char_count(text)
        if not _AIHONG_MENU_MIN_SIGNIFICANT_CHARS <= significant_chars <= _AIHONG_MENU_MAX_SIGNIFICANT_CHARS:
            return []
        normalized_choices.append(text)
    if status_lines and len(normalized_choices) >= _AIHONG_MENU_MIN_LINES:
        return normalized_choices
    return normalized_choices


@dataclass(slots=True)
class OcrCaptureProfile:
    left_inset_ratio: float = DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO
    right_inset_ratio: float = DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO
    top_ratio: float = DEFAULT_OCR_CAPTURE_TOP_RATIO
    bottom_inset_ratio: float = DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO

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
            left_inset_ratio=float(
                value.get("left_inset_ratio", DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO)
            ),
            right_inset_ratio=float(
                value.get("right_inset_ratio", DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO)
            ),
            top_ratio=float(value.get("top_ratio", DEFAULT_OCR_CAPTURE_TOP_RATIO)),
            bottom_inset_ratio=float(
                value.get("bottom_inset_ratio", DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO)
            ),
        )


def _significant_char_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def _looks_like_noise_ocr_text(text: str) -> bool:
    normalized = normalize_text(str(text or "")).strip()
    if not normalized:
        return True
    significant_chars = _significant_char_count(normalized)
    cjk_or_kana_count = len(_CJK_CHAR_RE.findall(normalized)) + len(_KANA_CHAR_RE.findall(normalized))
    if (
        cjk_or_kana_count <= 0
        and significant_chars <= _OCR_NOISE_MAX_NON_CJK_SIGNIFICANT_CHARS
    ):
        return True
    return False


def _perceptual_hash_image(frame: Any, *, size: int = _BACKGROUND_SCENE_HASH_SIZE) -> str:
    if frame is None:
        return ""
    try:
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image)
        image = frame.convert("L").resize((size, size), resampling.BILINEAR)
        pixels = list(image.getdata())
        if not pixels:
            return ""
        average = sum(int(pixel) for pixel in pixels) / len(pixels)
        bits = "".join("1" if int(pixel) >= average else "0" for pixel in pixels)
        return f"{int(bits, 2):016x}"
    except Exception:
        return ""


@dataclass(slots=True)
class OcrTextBox:
    text: str
    left: float
    top: float
    right: float
    bottom: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "text": self.text,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
        }


@dataclass(slots=True)
class DetectedGameWindow:
    hwnd: int = 0
    title: str = ""
    process_name: str = ""
    pid: int = 0
    class_name: str = ""
    exe_path: str = ""
    width: int = 0
    height: int = 0
    area: int = 0
    is_foreground: bool = False
    eligible: bool = True
    exclude_reason: str = ""
    category: str = "eligible_game_window"
    score: float = 0.0

    @property
    def normalized_title(self) -> str:
        return _normalize_window_title(self.title)

    @property
    def window_key(self) -> str:
        return _build_window_key(
            process_name=self.process_name,
            pid=self.pid,
            hwnd=self.hwnd,
            title=self.title,
        )

    @property
    def aspect_ratio(self) -> float:
        return compute_ocr_window_aspect_ratio(self.width, self.height)

    def to_dict(self, *, is_attached: bool = False, is_manual_target: bool = False) -> dict[str, Any]:
        return {
            "window_key": self.window_key,
            "title": self.title,
            "process_name": self.process_name,
            "pid": self.pid,
            "hwnd": self.hwnd,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "eligible": self.eligible,
            "exclude_reason": self.exclude_reason,
            "is_foreground": self.is_foreground,
            "is_attached": is_attached,
            "is_manual_target": is_manual_target,
            "class_name": self.class_name,
            "exe_path": self.exe_path,
            "category": self.category,
        }


def _matches_aihong_target(target: DetectedGameWindow | None) -> bool:
    if target is None:
        return False
    process_name = str(target.process_name or "").strip().lower()
    if process_name in _AIHONG_PROCESS_NAMES:
        return True
    normalized_title = target.normalized_title
    return any(token in normalized_title for token in _AIHONG_TITLE_SUBSTRINGS)


def _builtin_capture_profile_for_target(target: DetectedGameWindow) -> OcrCaptureProfile | None:
    return _builtin_capture_profile_for_target_stage(target, stage=_AIHONG_DIALOGUE_STAGE)


def _builtin_capture_profile_for_target_stage(
    target: DetectedGameWindow,
    *,
    stage: str,
) -> OcrCaptureProfile | None:
    if not _matches_aihong_target(target):
        return None
    if stage == _AIHONG_MENU_STAGE:
        return OcrCaptureProfile.from_dict(_AIHONG_MENU_CAPTURE_PROFILE_PRESET)
    return OcrCaptureProfile.from_dict(_AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET)


@dataclass(slots=True)
class _StableOcrTextState:
    last_raw_text: str = ""
    repeat_count: int = 0
    stable_text: str = ""

    def reset(self) -> None:
        self.last_raw_text = ""
        self.repeat_count = 0
        self.stable_text = ""


@dataclass(slots=True)
class _MenuConsumeResult:
    emitted_kind: str = ""
    has_menu_candidate: bool = False


class _AihongStage(Enum):
    DIALOGUE = OCR_CAPTURE_PROFILE_STAGE_DIALOGUE
    MENU = OCR_CAPTURE_PROFILE_STAGE_MENU


@dataclass(slots=True)
class _AihongStateMachine:
    stage: _AihongStage = _AihongStage.DIALOGUE
    dialogue_idle_polls: int = 0
    menu_missing_polls: int = 0
    menu_ocr_state: _StableOcrTextState = field(default_factory=_StableOcrTextState)

    def reset(self) -> None:
        self.stage = _AihongStage.DIALOGUE
        self.dialogue_idle_polls = 0
        self.menu_missing_polls = 0
        self.menu_ocr_state.reset()

    @property
    def capture_stage(self) -> str:
        return self.stage.value

    @property
    def is_dialogue(self) -> bool:
        return self.stage == _AihongStage.DIALOGUE

    @property
    def is_menu(self) -> bool:
        return self.stage == _AihongStage.MENU

    def on_dialogue_consumed(
        self,
        *,
        emitted: bool,
        is_menu_choices: bool,
        is_menu_status: bool,
    ) -> None:
        if emitted:
            self.dialogue_idle_polls = 0
            self.menu_missing_polls = 0
            if is_menu_choices:
                self.stage = _AihongStage.MENU
            else:
                self.menu_ocr_state.reset()
        else:
            if is_menu_status or is_menu_choices:
                self.dialogue_idle_polls = max(
                    self.dialogue_idle_polls,
                    _AIHONG_MENU_STATUS_IDLE_POLLS,
                )
            else:
                self.dialogue_idle_polls += 1

    def should_probe_menu(
        self,
        *,
        after_advance_trigger_mode: bool,
        looks_like_menu: bool,
    ) -> bool:
        if after_advance_trigger_mode and not looks_like_menu:
            return False
        return looks_like_menu or self.dialogue_idle_polls >= _AIHONG_DIALOGUE_IDLE_BEFORE_MENU_PROBE_POLLS

    def on_menu_probe_result(
        self,
        *,
        emitted_kind: str,
        has_menu_candidate: bool,
    ) -> None:
        if has_menu_candidate:
            self.menu_missing_polls = 0
        if emitted_kind == "line":
            self._transition_to_dialogue()
        elif emitted_kind == "choices":
            self.stage = _AihongStage.MENU
            self.menu_missing_polls = 0
        elif has_menu_candidate:
            self.stage = _AihongStage.MENU

    def on_active_menu_consumed(
        self,
        *,
        emitted_kind: str,
        has_menu_candidate: bool,
        text: str,
    ) -> bool:
        if emitted_kind == "line":
            self._transition_to_dialogue()
            return False
        if has_menu_candidate:
            self.menu_missing_polls = 0
            return False
        self.menu_missing_polls += 1
        should_reset = False
        if text and not _looks_like_noise_ocr_text(text):
            should_reset = True
        elif self.menu_missing_polls >= _AIHONG_MENU_MISSING_MAX_POLLS:
            should_reset = True
        if should_reset:
            self.reset()
        return should_reset

    def _transition_to_dialogue(self) -> None:
        self.stage = _AihongStage.DIALOGUE
        self.dialogue_idle_polls = 0
        self.menu_missing_polls = 0
        self.menu_ocr_state.reset()


def _canonical_choice_candidate_text(choices: list[str]) -> str:
    normalized = [normalize_text(str(choice or "")).strip() for choice in choices]
    return "\n".join(item for item in normalized if item)


def _stripped_ocr_lines(raw_text: str) -> list[str]:
    return [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]


@dataclass(slots=True)
class ParsedOcrCaptureBucket:
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    stages: dict[str, OcrCaptureProfile] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedOcrCaptureProcessConfig:
    stages: dict[str, OcrCaptureProfile] = field(default_factory=dict)
    window_buckets: dict[str, ParsedOcrCaptureBucket] = field(default_factory=dict)


@dataclass(slots=True)
class ResolvedOcrCaptureSelection:
    profile: OcrCaptureProfile = field(default_factory=OcrCaptureProfile)
    match_source: str = OCR_CAPTURE_PROFILE_MATCH_SOURCE_CONFIG_DEFAULT
    bucket_key: str = ""


def _resolve_stage_capture_profile(
    stage_profiles: dict[str, OcrCaptureProfile],
    *,
    stage: str,
) -> OcrCaptureProfile | None:
    return stage_profiles.get(stage) or stage_profiles.get(OCR_CAPTURE_PROFILE_STAGE_DEFAULT)


def _uses_manual_capture_profile(
    profiles: dict[str, ParsedOcrCaptureProcessConfig],
    target: DetectedGameWindow,
) -> bool:
    process_name = str(target.process_name or "").strip().lower()
    if not process_name:
        return False
    return process_name in profiles


def _lookup_capture_profile(
    profiles: dict[str, ParsedOcrCaptureProcessConfig],
    target: DetectedGameWindow,
    *,
    stage: str,
) -> ResolvedOcrCaptureSelection | None:
    process_name = str(target.process_name or "").strip().lower()
    if not process_name:
        return None
    configured = profiles.get(process_name)
    if configured is None:
        return None

    if target.width > 0 and target.height > 0:
        exact_bucket_key = build_ocr_capture_profile_bucket_key(target.width, target.height).lower()
        exact_bucket = configured.window_buckets.get(exact_bucket_key)
        if exact_bucket is not None:
            exact_profile = _resolve_stage_capture_profile(exact_bucket.stages, stage=stage)
            if exact_profile is not None:
                return ResolvedOcrCaptureSelection(
                    profile=exact_profile,
                    match_source=OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUCKET_EXACT,
                    bucket_key=exact_bucket_key,
                )

        target_aspect_ratio = target.aspect_ratio
        if target_aspect_ratio > 0:
            nearest_bucket_key = ""
            nearest_profile: OcrCaptureProfile | None = None
            nearest_size_delta: int | None = None
            nearest_aspect_delta: float | None = None
            for bucket_key, bucket in configured.window_buckets.items():
                profile = _resolve_stage_capture_profile(bucket.stages, stage=stage)
                if profile is None:
                    continue
                aspect_delta = abs(float(bucket.aspect_ratio or 0.0) - target_aspect_ratio)
                if aspect_delta > 0.03:
                    continue
                size_delta = abs(int(bucket.width or 0) - target.width) + abs(
                    int(bucket.height or 0) - target.height
                )
                if (
                    nearest_size_delta is None
                    or size_delta < nearest_size_delta
                    or (
                        size_delta == nearest_size_delta
                        and (
                            nearest_aspect_delta is None
                            or aspect_delta < nearest_aspect_delta
                        )
                    )
                ):
                    nearest_bucket_key = bucket_key
                    nearest_profile = profile
                    nearest_size_delta = size_delta
                    nearest_aspect_delta = aspect_delta
            if nearest_profile is not None:
                return ResolvedOcrCaptureSelection(
                    profile=nearest_profile,
                    match_source=OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUCKET_ASPECT_NEAREST,
                    bucket_key=nearest_bucket_key,
                )

    fallback_profile = _resolve_stage_capture_profile(configured.stages, stage=stage)
    if fallback_profile is not None:
        return ResolvedOcrCaptureSelection(
            profile=fallback_profile,
            match_source=OCR_CAPTURE_PROFILE_MATCH_SOURCE_PROCESS_FALLBACK,
        )
    return None


def _parse_configured_capture_profiles(
    profiles: dict[str, dict[str, Any]],
    logger,
) -> dict[str, ParsedOcrCaptureProcessConfig]:
    parsed_profiles: dict[str, ParsedOcrCaptureProcessConfig] = {}
    for process_name, profile_value in profiles.items():
        normalized_process_name = str(process_name or "").strip().lower()
        if not normalized_process_name or not isinstance(profile_value, dict):
            continue
        stage_profiles: dict[str, OcrCaptureProfile] = {}
        if all(key in profile_value for key in OCR_CAPTURE_PROFILE_RATIO_KEYS):
            try:
                stage_profiles[OCR_CAPTURE_PROFILE_STAGE_DEFAULT] = OcrCaptureProfile.from_dict(
                    profile_value
                )
            except Exception as exc:
                logger.warning(
                    "ocr_reader failed to parse capture profile for %s: %s",
                    normalized_process_name,
                    exc,
                )
                continue
        else:
            for stage_name, stage_profile in profile_value.items():
                normalized_stage_name = str(stage_name or "").strip()
                if normalized_stage_name == OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY:
                    continue
                if not normalized_stage_name or not isinstance(stage_profile, dict):
                    continue
                try:
                    stage_profiles[normalized_stage_name] = OcrCaptureProfile.from_dict(stage_profile)
                except Exception as exc:
                    logger.warning(
                        "ocr_reader failed to parse capture profile for %s/%s: %s",
                        normalized_process_name,
                        normalized_stage_name,
                        exc,
                    )
        window_buckets: dict[str, ParsedOcrCaptureBucket] = {}
        raw_buckets = profile_value.get(OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY)
        if isinstance(raw_buckets, dict):
            for bucket_key, bucket_value in raw_buckets.items():
                normalized_bucket_key = str(bucket_key or "").strip().lower()
                parsed_dimensions = parse_ocr_capture_profile_bucket_key(normalized_bucket_key)
                if parsed_dimensions is None or not isinstance(bucket_value, dict):
                    continue
                try:
                    width = int(bucket_value.get("width") or parsed_dimensions[0])
                    height = int(bucket_value.get("height") or parsed_dimensions[1])
                except (TypeError, ValueError):
                    continue
                if width <= 0 or height <= 0:
                    continue
                try:
                    aspect_ratio = float(
                        bucket_value.get("aspect_ratio")
                        or compute_ocr_window_aspect_ratio(width, height)
                    )
                except (TypeError, ValueError):
                    aspect_ratio = compute_ocr_window_aspect_ratio(width, height)
                raw_stages = bucket_value.get("stages")
                if not isinstance(raw_stages, dict):
                    continue
                bucket_stages: dict[str, OcrCaptureProfile] = {}
                for stage_name, stage_profile in raw_stages.items():
                    normalized_stage_name = str(stage_name or "").strip()
                    if not normalized_stage_name or not isinstance(stage_profile, dict):
                        continue
                    try:
                        bucket_stages[normalized_stage_name] = OcrCaptureProfile.from_dict(
                            stage_profile
                        )
                    except Exception as exc:
                        logger.warning(
                            "ocr_reader failed to parse capture profile for %s/%s/%s: %s",
                            normalized_process_name,
                            normalized_bucket_key,
                            normalized_stage_name,
                            exc,
                        )
                if bucket_stages:
                    canonical_bucket_key = build_ocr_capture_profile_bucket_key(width, height).lower()
                    window_buckets[canonical_bucket_key] = ParsedOcrCaptureBucket(
                        width=width,
                        height=height,
                        aspect_ratio=aspect_ratio,
                        stages=bucket_stages,
                    )
        if stage_profiles or window_buckets:
            parsed_profiles[normalized_process_name] = ParsedOcrCaptureProcessConfig(
                stages=stage_profiles,
                window_buckets=window_buckets,
            )
    return parsed_profiles


@dataclass(slots=True)
class OcrWindowTarget:
    mode: str = "auto"
    window_key: str = ""
    process_name: str = ""
    normalized_title: str = ""
    pid: int = 0
    last_known_hwnd: int = 0
    selected_at: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> OcrWindowTarget:
        raw = value if isinstance(value, dict) else {}
        mode = str(raw.get("mode") or "auto").strip().lower()
        if mode not in {"auto", "manual"}:
            mode = "auto"
        try:
            pid = int(raw.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        try:
            last_known_hwnd = int(raw.get("last_known_hwnd") or 0)
        except (TypeError, ValueError):
            last_known_hwnd = 0
        return cls(
            mode=mode,
            window_key=str(raw.get("window_key") or "").strip(),
            process_name=str(raw.get("process_name") or "").strip(),
            normalized_title=str(raw.get("normalized_title") or "").strip().lower(),
            pid=max(0, pid),
            last_known_hwnd=max(0, last_known_hwnd),
            selected_at=str(raw.get("selected_at") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "window_key": self.window_key,
            "process_name": self.process_name,
            "normalized_title": self.normalized_title,
            "pid": self.pid,
            "last_known_hwnd": self.last_known_hwnd,
            "selected_at": self.selected_at,
        }

    def is_manual(self) -> bool:
        return self.mode == "manual"

    def matches_exact(self, candidate: DetectedGameWindow) -> bool:
        return bool(self.window_key) and self.window_key == candidate.window_key

    def matches_hwnd(self, candidate: DetectedGameWindow) -> bool:
        return bool(self.last_known_hwnd) and self.last_known_hwnd == candidate.hwnd

    def matches_signature(self, candidate: DetectedGameWindow) -> bool:
        has_process_name = bool(self.process_name)
        has_title = bool(self.normalized_title)
        if has_process_name and self.process_name.strip().lower() != candidate.process_name.strip().lower():
            return False
        if has_title and self.normalized_title != candidate.normalized_title:
            return False
        if not has_process_name and not has_title and self.pid > 0:
            return candidate.pid == self.pid
        return bool(self.process_name or self.normalized_title or self.pid)

    def resolved_for(self, candidate: DetectedGameWindow) -> OcrWindowTarget:
        return OcrWindowTarget(
            mode="manual",
            window_key=candidate.window_key,
            process_name=candidate.process_name,
            normalized_title=candidate.normalized_title,
            pid=candidate.pid,
            last_known_hwnd=candidate.hwnd,
            selected_at=self.selected_at,
        )


@dataclass(slots=True)
class OcrRuntimeWindowState:
    process_name: str = ""
    pid: int = 0
    title: str = ""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    selection_mode: str = "auto"
    selection_detail: str = ""
    effective_window_key: str = ""
    effective_window_title: str = ""
    effective_process_name: str = ""
    target_is_foreground: bool = False
    manual_target: dict[str, Any] = field(default_factory=dict)
    locked_target: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    excluded_candidate_count: int = 0
    last_exclude_reason: str = ""
    foreground_refresh_at: str = ""
    foreground_refresh_detail: str = ""
    foreground_hwnd: int = 0
    target_hwnd: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_name": self.process_name,
            "pid": self.pid,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "selection_mode": self.selection_mode,
            "selection_detail": self.selection_detail,
            "effective_window_key": self.effective_window_key,
            "effective_window_title": self.effective_window_title,
            "effective_process_name": self.effective_process_name,
            "target_is_foreground": self.target_is_foreground,
            "manual_target": dict(self.manual_target),
            "locked_target": dict(self.locked_target),
            "candidate_count": self.candidate_count,
            "excluded_candidate_count": self.excluded_candidate_count,
            "last_exclude_reason": self.last_exclude_reason,
            "foreground_refresh_at": self.foreground_refresh_at,
            "foreground_refresh_detail": self.foreground_refresh_detail,
            "foreground_hwnd": self.foreground_hwnd,
            "target_hwnd": self.target_hwnd,
        }


@dataclass(slots=True)
class OcrRuntimeCaptureState:
    stage: str = ""
    profile: dict[str, float] = field(default_factory=dict)
    profile_match_source: str = ""
    profile_bucket_key: str = ""
    last_profile: dict[str, float] = field(default_factory=dict)
    last_stage: str = ""
    backend_kind: str = ""
    backend_detail: str = ""
    last_image_hash: str = ""
    last_source_size: dict[str, float] = field(default_factory=dict)
    last_rect: dict[str, float] = field(default_factory=dict)
    last_window_rect: dict[str, float] = field(default_factory=dict)
    consecutive_same_frames: int = 0
    stale_backend: bool = False
    diagnostic_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "profile": dict(self.profile),
            "profile_match_source": self.profile_match_source,
            "profile_bucket_key": self.profile_bucket_key,
            "last_profile": dict(self.last_profile),
            "last_stage": self.last_stage,
            "backend_kind": self.backend_kind,
            "backend_detail": self.backend_detail,
            "last_image_hash": self.last_image_hash,
            "last_source_size": dict(self.last_source_size),
            "last_rect": dict(self.last_rect),
            "last_window_rect": dict(self.last_window_rect),
            "consecutive_same_frames": self.consecutive_same_frames,
            "stale_backend": self.stale_backend,
            "diagnostic_required": self.diagnostic_required,
        }


@dataclass(slots=True)
class OcrRuntimeOcrState:
    backend_kind: str = ""
    backend_detail: str = ""
    backend_path: str = ""
    backend_model: str = ""
    tesseract_path: str = ""
    languages: str = ""
    context_state: str = ""
    consecutive_no_text_polls: int = 0
    last_observed_at: str = ""
    last_capture_attempt_at: str = ""
    last_capture_completed_at: str = ""
    last_capture_error: str = ""
    last_raw_text: str = ""
    last_observed_line: dict[str, Any] = field(default_factory=dict)
    last_stable_line: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "backend_detail": self.backend_detail,
            "backend_path": self.backend_path,
            "backend_model": self.backend_model,
            "tesseract_path": self.tesseract_path,
            "languages": self.languages,
            "context_state": self.context_state,
            "consecutive_no_text_polls": self.consecutive_no_text_polls,
            "last_observed_at": self.last_observed_at,
            "last_capture_attempt_at": self.last_capture_attempt_at,
            "last_capture_completed_at": self.last_capture_completed_at,
            "last_capture_error": self.last_capture_error,
            "last_raw_text": self.last_raw_text,
            "last_observed_line": dict(self.last_observed_line),
            "last_stable_line": dict(self.last_stable_line),
        }


@dataclass(slots=True)
class OcrRuntimeTimingState:
    last_capture_total_duration_seconds: float = 0.0
    last_capture_frame_duration_seconds: float = 0.0
    last_capture_background_duration_seconds: float = 0.0
    last_capture_image_hash_duration_seconds: float = 0.0
    last_ocr_extract_duration_seconds: float = 0.0
    last_backend_plan_duration_seconds: float = 0.0
    last_window_scan_duration_seconds: float = 0.0
    last_poll_started_at: str = ""
    last_poll_completed_at: str = ""
    last_poll_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_capture_total_duration_seconds": self.last_capture_total_duration_seconds,
            "last_capture_frame_duration_seconds": self.last_capture_frame_duration_seconds,
            "last_capture_background_duration_seconds": self.last_capture_background_duration_seconds,
            "last_capture_image_hash_duration_seconds": self.last_capture_image_hash_duration_seconds,
            "last_ocr_extract_duration_seconds": self.last_ocr_extract_duration_seconds,
            "last_backend_plan_duration_seconds": self.last_backend_plan_duration_seconds,
            "last_window_scan_duration_seconds": self.last_window_scan_duration_seconds,
            "last_poll_started_at": self.last_poll_started_at,
            "last_poll_completed_at": self.last_poll_completed_at,
            "last_poll_duration_seconds": self.last_poll_duration_seconds,
        }


@dataclass(slots=True)
class OcrRuntimeAdvanceState:
    foreground_monitor_running: bool = False
    foreground_last_seq: int = 0
    foreground_consumed_seq: int = 0
    foreground_last_kind: str = ""
    foreground_last_delta: int = 0
    foreground_last_matched: bool = False
    foreground_last_match_reason: str = ""
    last_background_hash_skipped: bool = False
    last_poll_emitted_event: bool = False
    last_tick_skipped: bool = False
    last_tick_skip_reason: str = ""
    pending_visual_scene_count: int = 0
    last_auto_recalibrate_attempts: int = 0
    last_auto_recalibrate_duration_seconds: float = 0.0
    last_auto_recalibrate_limited: bool = False
    last_auto_recalibrate_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "foreground_monitor_running": self.foreground_monitor_running,
            "foreground_last_seq": self.foreground_last_seq,
            "foreground_consumed_seq": self.foreground_consumed_seq,
            "foreground_last_kind": self.foreground_last_kind,
            "foreground_last_delta": self.foreground_last_delta,
            "foreground_last_matched": self.foreground_last_matched,
            "foreground_last_match_reason": self.foreground_last_match_reason,
            "last_background_hash_skipped": self.last_background_hash_skipped,
            "last_poll_emitted_event": self.last_poll_emitted_event,
            "last_tick_skipped": self.last_tick_skipped,
            "last_tick_skip_reason": self.last_tick_skip_reason,
            "pending_visual_scene_count": self.pending_visual_scene_count,
            "last_auto_recalibrate_attempts": self.last_auto_recalibrate_attempts,
            "last_auto_recalibrate_duration_seconds": self.last_auto_recalibrate_duration_seconds,
            "last_auto_recalibrate_limited": self.last_auto_recalibrate_limited,
            "last_auto_recalibrate_error": self.last_auto_recalibrate_error,
        }


@dataclass(slots=True)
class OcrReaderRuntime:
    enabled: bool = False
    status: str = "disabled"
    detail: str = ""
    process_name: str = ""
    pid: int = 0
    window_title: str = ""
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    game_id: str = ""
    session_id: str = ""
    last_seq: int = 0
    last_event_ts: str = ""
    capture_stage: str = ""
    capture_profile: dict[str, float] = field(default_factory=dict)
    capture_profile_match_source: str = ""
    capture_profile_bucket_key: str = ""
    tesseract_path: str = ""
    languages: str = ""
    takeover_reason: str = ""
    backend_kind: str = ""
    backend_detail: str = ""
    backend_path: str = ""
    backend_model: str = ""
    target_selection_mode: str = "auto"
    target_selection_detail: str = ""
    effective_window_key: str = ""
    effective_window_title: str = ""
    effective_process_name: str = ""
    target_is_foreground: bool = False
    manual_target: dict[str, Any] = field(default_factory=dict)
    locked_target: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    excluded_candidate_count: int = 0
    last_exclude_reason: str = ""
    consecutive_no_text_polls: int = 0
    last_observed_at: str = ""
    last_capture_profile: dict[str, float] = field(default_factory=dict)
    last_capture_stage: str = ""
    ocr_capture_diagnostic_required: bool = False
    ocr_context_state: str = ""
    last_capture_attempt_at: str = ""
    last_capture_completed_at: str = ""
    last_capture_error: str = ""
    last_raw_ocr_text: str = ""
    last_observed_line: dict[str, Any] = field(default_factory=dict)
    last_stable_line: dict[str, Any] = field(default_factory=dict)
    capture_backend_kind: str = ""
    capture_backend_detail: str = ""
    last_capture_image_hash: str = ""
    last_capture_source_size: dict[str, float] = field(default_factory=dict)
    last_capture_rect: dict[str, float] = field(default_factory=dict)
    last_capture_window_rect: dict[str, float] = field(default_factory=dict)
    consecutive_same_capture_frames: int = 0
    stale_capture_backend: bool = False
    foreground_refresh_at: str = ""
    foreground_refresh_detail: str = ""
    foreground_hwnd: int = 0
    target_hwnd: int = 0
    foreground_advance_monitor_running: bool = False
    foreground_advance_last_seq: int = 0
    foreground_advance_consumed_seq: int = 0
    foreground_advance_last_kind: str = ""
    foreground_advance_last_delta: int = 0
    foreground_advance_last_matched: bool = False
    foreground_advance_last_match_reason: str = ""
    last_capture_total_duration_seconds: float = 0.0
    last_capture_frame_duration_seconds: float = 0.0
    last_capture_background_duration_seconds: float = 0.0
    last_capture_image_hash_duration_seconds: float = 0.0
    last_ocr_extract_duration_seconds: float = 0.0
    last_backend_plan_duration_seconds: float = 0.0
    last_window_scan_duration_seconds: float = 0.0
    last_capture_background_hash_skipped: bool = False
    last_poll_started_at: str = ""
    last_poll_completed_at: str = ""
    last_poll_duration_seconds: float = 0.0
    last_poll_emitted_event: bool = False
    last_tick_skipped: bool = False
    last_tick_skip_reason: str = ""
    pending_visual_scene_count: int = 0
    last_auto_recalibrate_attempts: int = 0
    last_auto_recalibrate_duration_seconds: float = 0.0
    last_auto_recalibrate_limited: bool = False
    last_auto_recalibrate_error: str = ""

    def window_state(self) -> OcrRuntimeWindowState:
        return OcrRuntimeWindowState(
            process_name=self.process_name,
            pid=self.pid,
            title=self.window_title,
            width=self.width,
            height=self.height,
            aspect_ratio=self.aspect_ratio,
            selection_mode=self.target_selection_mode,
            selection_detail=self.target_selection_detail,
            effective_window_key=self.effective_window_key,
            effective_window_title=self.effective_window_title,
            effective_process_name=self.effective_process_name,
            target_is_foreground=self.target_is_foreground,
            manual_target=dict(self.manual_target),
            locked_target=dict(self.locked_target),
            candidate_count=self.candidate_count,
            excluded_candidate_count=self.excluded_candidate_count,
            last_exclude_reason=self.last_exclude_reason,
            foreground_refresh_at=self.foreground_refresh_at,
            foreground_refresh_detail=self.foreground_refresh_detail,
            foreground_hwnd=self.foreground_hwnd,
            target_hwnd=self.target_hwnd,
        )

    def apply_window_state(self, state: OcrRuntimeWindowState) -> OcrRuntimeWindowState:
        self.process_name = state.process_name
        self.pid = state.pid
        self.window_title = state.title
        self.width = state.width
        self.height = state.height
        self.aspect_ratio = state.aspect_ratio
        self.target_selection_mode = state.selection_mode
        self.target_selection_detail = state.selection_detail
        self.effective_window_key = state.effective_window_key
        self.effective_window_title = state.effective_window_title
        self.effective_process_name = state.effective_process_name
        self.target_is_foreground = state.target_is_foreground
        self.manual_target = dict(state.manual_target)
        self.locked_target = dict(state.locked_target)
        self.candidate_count = state.candidate_count
        self.excluded_candidate_count = state.excluded_candidate_count
        self.last_exclude_reason = state.last_exclude_reason
        self.foreground_refresh_at = state.foreground_refresh_at
        self.foreground_refresh_detail = state.foreground_refresh_detail
        self.foreground_hwnd = state.foreground_hwnd
        self.target_hwnd = state.target_hwnd
        return state

    def update_window_state(self, **changes: Any) -> OcrRuntimeWindowState:
        state = replace(self.window_state(), **changes) if changes else self.window_state()
        return self.apply_window_state(state)

    def capture_state(self) -> OcrRuntimeCaptureState:
        return OcrRuntimeCaptureState(
            stage=self.capture_stage,
            profile=dict(self.capture_profile),
            profile_match_source=self.capture_profile_match_source,
            profile_bucket_key=self.capture_profile_bucket_key,
            last_profile=dict(self.last_capture_profile),
            last_stage=self.last_capture_stage,
            backend_kind=self.capture_backend_kind,
            backend_detail=self.capture_backend_detail,
            last_image_hash=self.last_capture_image_hash,
            last_source_size=dict(self.last_capture_source_size),
            last_rect=dict(self.last_capture_rect),
            last_window_rect=dict(self.last_capture_window_rect),
            consecutive_same_frames=self.consecutive_same_capture_frames,
            stale_backend=self.stale_capture_backend,
            diagnostic_required=self.ocr_capture_diagnostic_required,
        )

    def apply_capture_state(self, state: OcrRuntimeCaptureState) -> OcrRuntimeCaptureState:
        self.capture_stage = state.stage
        self.capture_profile = dict(state.profile)
        self.capture_profile_match_source = state.profile_match_source
        self.capture_profile_bucket_key = state.profile_bucket_key
        self.last_capture_profile = dict(state.last_profile)
        self.last_capture_stage = state.last_stage
        self.capture_backend_kind = state.backend_kind
        self.capture_backend_detail = state.backend_detail
        self.last_capture_image_hash = state.last_image_hash
        self.last_capture_source_size = dict(state.last_source_size)
        self.last_capture_rect = dict(state.last_rect)
        self.last_capture_window_rect = dict(state.last_window_rect)
        self.consecutive_same_capture_frames = state.consecutive_same_frames
        self.stale_capture_backend = state.stale_backend
        self.ocr_capture_diagnostic_required = state.diagnostic_required
        return state

    def update_capture_state(self, **changes: Any) -> OcrRuntimeCaptureState:
        state = replace(self.capture_state(), **changes) if changes else self.capture_state()
        return self.apply_capture_state(state)

    def ocr_state(self) -> OcrRuntimeOcrState:
        return OcrRuntimeOcrState(
            backend_kind=self.backend_kind,
            backend_detail=self.backend_detail,
            backend_path=self.backend_path,
            backend_model=self.backend_model,
            tesseract_path=self.tesseract_path,
            languages=self.languages,
            context_state=self.ocr_context_state,
            consecutive_no_text_polls=self.consecutive_no_text_polls,
            last_observed_at=self.last_observed_at,
            last_capture_attempt_at=self.last_capture_attempt_at,
            last_capture_completed_at=self.last_capture_completed_at,
            last_capture_error=self.last_capture_error,
            last_raw_text=self.last_raw_ocr_text,
            last_observed_line=dict(self.last_observed_line),
            last_stable_line=dict(self.last_stable_line),
        )

    def apply_ocr_state(self, state: OcrRuntimeOcrState) -> OcrRuntimeOcrState:
        self.backend_kind = state.backend_kind
        self.backend_detail = state.backend_detail
        self.backend_path = state.backend_path
        self.backend_model = state.backend_model
        self.tesseract_path = state.tesseract_path
        self.languages = state.languages
        self.ocr_context_state = state.context_state
        self.consecutive_no_text_polls = state.consecutive_no_text_polls
        self.last_observed_at = state.last_observed_at
        self.last_capture_attempt_at = state.last_capture_attempt_at
        self.last_capture_completed_at = state.last_capture_completed_at
        self.last_capture_error = state.last_capture_error
        self.last_raw_ocr_text = state.last_raw_text
        self.last_observed_line = dict(state.last_observed_line)
        self.last_stable_line = dict(state.last_stable_line)
        return state

    def update_ocr_state(self, **changes: Any) -> OcrRuntimeOcrState:
        state = replace(self.ocr_state(), **changes) if changes else self.ocr_state()
        return self.apply_ocr_state(state)

    def timing_state(self) -> OcrRuntimeTimingState:
        return OcrRuntimeTimingState(
            last_capture_total_duration_seconds=self.last_capture_total_duration_seconds,
            last_capture_frame_duration_seconds=self.last_capture_frame_duration_seconds,
            last_capture_background_duration_seconds=self.last_capture_background_duration_seconds,
            last_capture_image_hash_duration_seconds=self.last_capture_image_hash_duration_seconds,
            last_ocr_extract_duration_seconds=self.last_ocr_extract_duration_seconds,
            last_backend_plan_duration_seconds=self.last_backend_plan_duration_seconds,
            last_window_scan_duration_seconds=self.last_window_scan_duration_seconds,
            last_poll_started_at=self.last_poll_started_at,
            last_poll_completed_at=self.last_poll_completed_at,
            last_poll_duration_seconds=self.last_poll_duration_seconds,
        )

    def apply_timing_state(self, state: OcrRuntimeTimingState) -> OcrRuntimeTimingState:
        self.last_capture_total_duration_seconds = state.last_capture_total_duration_seconds
        self.last_capture_frame_duration_seconds = state.last_capture_frame_duration_seconds
        self.last_capture_background_duration_seconds = state.last_capture_background_duration_seconds
        self.last_capture_image_hash_duration_seconds = state.last_capture_image_hash_duration_seconds
        self.last_ocr_extract_duration_seconds = state.last_ocr_extract_duration_seconds
        self.last_backend_plan_duration_seconds = state.last_backend_plan_duration_seconds
        self.last_window_scan_duration_seconds = state.last_window_scan_duration_seconds
        self.last_poll_started_at = state.last_poll_started_at
        self.last_poll_completed_at = state.last_poll_completed_at
        self.last_poll_duration_seconds = state.last_poll_duration_seconds
        return state

    def update_timing_state(self, **changes: Any) -> OcrRuntimeTimingState:
        state = replace(self.timing_state(), **changes) if changes else self.timing_state()
        return self.apply_timing_state(state)

    def advance_state(self) -> OcrRuntimeAdvanceState:
        return OcrRuntimeAdvanceState(
            foreground_monitor_running=self.foreground_advance_monitor_running,
            foreground_last_seq=self.foreground_advance_last_seq,
            foreground_consumed_seq=self.foreground_advance_consumed_seq,
            foreground_last_kind=self.foreground_advance_last_kind,
            foreground_last_delta=self.foreground_advance_last_delta,
            foreground_last_matched=self.foreground_advance_last_matched,
            foreground_last_match_reason=self.foreground_advance_last_match_reason,
            last_background_hash_skipped=self.last_capture_background_hash_skipped,
            last_poll_emitted_event=self.last_poll_emitted_event,
            last_tick_skipped=self.last_tick_skipped,
            last_tick_skip_reason=self.last_tick_skip_reason,
            pending_visual_scene_count=self.pending_visual_scene_count,
            last_auto_recalibrate_attempts=self.last_auto_recalibrate_attempts,
            last_auto_recalibrate_duration_seconds=self.last_auto_recalibrate_duration_seconds,
            last_auto_recalibrate_limited=self.last_auto_recalibrate_limited,
            last_auto_recalibrate_error=self.last_auto_recalibrate_error,
        )

    def apply_advance_state(self, state: OcrRuntimeAdvanceState) -> OcrRuntimeAdvanceState:
        self.foreground_advance_monitor_running = state.foreground_monitor_running
        self.foreground_advance_last_seq = state.foreground_last_seq
        self.foreground_advance_consumed_seq = state.foreground_consumed_seq
        self.foreground_advance_last_kind = state.foreground_last_kind
        self.foreground_advance_last_delta = state.foreground_last_delta
        self.foreground_advance_last_matched = state.foreground_last_matched
        self.foreground_advance_last_match_reason = state.foreground_last_match_reason
        self.last_capture_background_hash_skipped = state.last_background_hash_skipped
        self.last_poll_emitted_event = state.last_poll_emitted_event
        self.last_tick_skipped = state.last_tick_skipped
        self.last_tick_skip_reason = state.last_tick_skip_reason
        self.pending_visual_scene_count = state.pending_visual_scene_count
        self.last_auto_recalibrate_attempts = state.last_auto_recalibrate_attempts
        self.last_auto_recalibrate_duration_seconds = state.last_auto_recalibrate_duration_seconds
        self.last_auto_recalibrate_limited = state.last_auto_recalibrate_limited
        self.last_auto_recalibrate_error = state.last_auto_recalibrate_error
        return state

    def update_advance_state(self, **changes: Any) -> OcrRuntimeAdvanceState:
        state = replace(self.advance_state(), **changes) if changes else self.advance_state()
        return self.apply_advance_state(state)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "enabled": self.enabled,
            "status": self.status,
            "detail": self.detail,
            "process_name": self.process_name,
            "pid": self.pid,
            "window_title": self.window_title,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "game_id": self.game_id,
            "session_id": self.session_id,
            "last_seq": self.last_seq,
            "last_event_ts": self.last_event_ts,
            "capture_stage": self.capture_stage,
            "capture_profile": dict(self.capture_profile),
            "capture_profile_match_source": self.capture_profile_match_source,
            "capture_profile_bucket_key": self.capture_profile_bucket_key,
            "tesseract_path": self.tesseract_path,
            "languages": self.languages,
            "takeover_reason": self.takeover_reason,
            "backend_kind": self.backend_kind,
            "backend_detail": self.backend_detail,
            "backend_path": self.backend_path,
            "backend_model": self.backend_model,
            "target_selection_mode": self.target_selection_mode,
            "target_selection_detail": self.target_selection_detail,
            "effective_window_key": self.effective_window_key,
            "effective_window_title": self.effective_window_title,
            "effective_process_name": self.effective_process_name,
            "target_is_foreground": self.target_is_foreground,
            "manual_target": dict(self.manual_target),
            "locked_target": dict(self.locked_target),
            "candidate_count": self.candidate_count,
            "excluded_candidate_count": self.excluded_candidate_count,
            "last_exclude_reason": self.last_exclude_reason,
            "consecutive_no_text_polls": self.consecutive_no_text_polls,
            "last_observed_at": self.last_observed_at,
            "last_capture_profile": dict(self.last_capture_profile),
            "last_capture_stage": self.last_capture_stage,
            "ocr_capture_diagnostic_required": self.ocr_capture_diagnostic_required,
            "ocr_context_state": self.ocr_context_state,
            "last_capture_attempt_at": self.last_capture_attempt_at,
            "last_capture_completed_at": self.last_capture_completed_at,
            "last_capture_error": self.last_capture_error,
            "last_raw_ocr_text": self.last_raw_ocr_text,
            "last_observed_line": dict(self.last_observed_line),
            "last_stable_line": dict(self.last_stable_line),
            "capture_backend_kind": self.capture_backend_kind,
            "capture_backend_detail": self.capture_backend_detail,
            "last_capture_image_hash": self.last_capture_image_hash,
            "last_capture_source_size": dict(self.last_capture_source_size),
            "last_capture_rect": dict(self.last_capture_rect),
            "last_capture_window_rect": dict(self.last_capture_window_rect),
            "consecutive_same_capture_frames": self.consecutive_same_capture_frames,
            "stale_capture_backend": self.stale_capture_backend,
            "foreground_refresh_at": self.foreground_refresh_at,
            "foreground_refresh_detail": self.foreground_refresh_detail,
            "foreground_hwnd": self.foreground_hwnd,
            "target_hwnd": self.target_hwnd,
            "foreground_advance_monitor_running": self.foreground_advance_monitor_running,
            "foreground_advance_last_seq": self.foreground_advance_last_seq,
            "foreground_advance_consumed_seq": self.foreground_advance_consumed_seq,
            "foreground_advance_last_kind": self.foreground_advance_last_kind,
            "foreground_advance_last_delta": self.foreground_advance_last_delta,
            "foreground_advance_last_matched": self.foreground_advance_last_matched,
            "foreground_advance_last_match_reason": self.foreground_advance_last_match_reason,
            "last_capture_total_duration_seconds": self.last_capture_total_duration_seconds,
            "last_capture_frame_duration_seconds": self.last_capture_frame_duration_seconds,
            "last_capture_background_duration_seconds": self.last_capture_background_duration_seconds,
            "last_capture_image_hash_duration_seconds": self.last_capture_image_hash_duration_seconds,
            "last_ocr_extract_duration_seconds": self.last_ocr_extract_duration_seconds,
            "last_backend_plan_duration_seconds": self.last_backend_plan_duration_seconds,
            "last_window_scan_duration_seconds": self.last_window_scan_duration_seconds,
            "last_capture_background_hash_skipped": self.last_capture_background_hash_skipped,
            "last_poll_started_at": self.last_poll_started_at,
            "last_poll_completed_at": self.last_poll_completed_at,
            "last_poll_duration_seconds": self.last_poll_duration_seconds,
            "last_poll_emitted_event": self.last_poll_emitted_event,
            "last_tick_skipped": self.last_tick_skipped,
            "last_tick_skip_reason": self.last_tick_skip_reason,
            "pending_visual_scene_count": self.pending_visual_scene_count,
            "last_auto_recalibrate_attempts": self.last_auto_recalibrate_attempts,
            "last_auto_recalibrate_duration_seconds": self.last_auto_recalibrate_duration_seconds,
            "last_auto_recalibrate_limited": self.last_auto_recalibrate_limited,
            "last_auto_recalibrate_error": self.last_auto_recalibrate_error,
        }
        payload["window"] = self.window_state().to_dict()
        payload["capture"] = self.capture_state().to_dict()
        payload["ocr"] = self.ocr_state().to_dict()
        payload["timing"] = self.timing_state().to_dict()
        payload["advance"] = self.advance_state().to_dict()
        return payload


@dataclass(slots=True)
class WindowSelectionResult:
    target: DetectedGameWindow | None = None
    selection_mode: str = "auto"
    selection_detail: str = ""
    manual_target: OcrWindowTarget = field(default_factory=OcrWindowTarget)
    selected_by_manual: bool = False
    candidate_count: int = 0
    excluded_candidate_count: int = 0
    last_exclude_reason: str = ""


@dataclass(slots=True)
class OcrReaderTickResult:
    warnings: list[str] = field(default_factory=list)
    should_rescan: bool = False
    runtime: dict[str, Any] = field(default_factory=dict)
    stable_event_emitted: bool = False


@dataclass(slots=True)
class _TickBackendPlanResult:
    plan: SelectedOcrBackendPlan
    duration_seconds: float = 0.0


@dataclass(slots=True)
class _TickWindowSelectionResult:
    selection: WindowSelectionResult
    duration_seconds: float = 0.0


@dataclass(slots=True)
class _TickCaptureMode:
    after_advance_trigger_mode: bool = False
    emit_observed_lines: bool = True
    line_repeat_threshold: int | None = None
    background_confirm_polls: int = _BACKGROUND_SCENE_CHANGE_CONFIRM_POLLS


@dataclass(slots=True)
class _TickPostCaptureStatus:
    status: str
    detail: str
    observed_or_stable_emitted: bool = False


@dataclass(slots=True)
class _TickExtractionBookkeepingResult:
    active_backend: OcrBackendDescriptor
    backend_detail_override: str = ""


@dataclass(slots=True)
class OcrBackendDescriptor:
    kind: str = ""
    backend: OcrBackend | None = None
    path: str = ""
    model: str = ""
    detail: str = ""
    available: bool = False


@dataclass(slots=True)
class _TickFollowupConfirmResult:
    emitted: bool = False
    guard_blocked: bool = False
    now: float = 0.0
    active_backend: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    backend_detail_override: str = ""


@dataclass(slots=True)
class _TickAihongMenuProbeResult:
    emitted: bool = False
    guard_blocked: bool = False
    active_backend: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    backend_detail_override: str = ""
    runtime_profile: OcrCaptureProfile | None = None
    runtime_capture_profile_selection: ResolvedOcrCaptureSelection | None = None


@dataclass(slots=True)
class _TickAihongDialogueStageResult:
    emitted: bool = False
    guard_blocked: bool = False
    now: float = 0.0
    active_backend: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    backend_detail_override: str = ""
    runtime_profile: OcrCaptureProfile | None = None
    runtime_capture_profile_selection: ResolvedOcrCaptureSelection | None = None


@dataclass(slots=True)
class _TickDefaultDialogueStageResult:
    emitted: bool = False
    guard_blocked: bool = False
    now: float = 0.0
    active_backend: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    backend_detail_override: str = ""


@dataclass(slots=True)
class SelectedOcrBackendPlan:
    selection: str = "auto"
    primary: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    fallback: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    rapidocr_inspection: dict[str, Any] = field(default_factory=dict)
    tesseract_inspection: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OcrExtractionResult:
    text: str = ""
    backend: OcrBackendDescriptor = field(default_factory=OcrBackendDescriptor)
    backend_detail: str = ""
    warnings: list[str] = field(default_factory=list)
    boxes: list[OcrTextBox] = field(default_factory=list)
    bounds_coordinate_space: str = ""
    source_size: dict[str, float] = field(default_factory=dict)
    capture_rect: dict[str, float] = field(default_factory=dict)
    window_rect: dict[str, float] = field(default_factory=dict)
    capture_backend_kind: str = ""
    capture_backend_detail: str = ""
    capture_image_hash: str = ""
    background_hash: str = ""
    timing: dict[str, float | bool] = field(default_factory=dict)


class CaptureBackend(Protocol):
    def is_available(self) -> bool: ...

    def describe_target(self, target: DetectedGameWindow) -> str: ...

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any: ...


class OcrBackend(Protocol):
    def is_available(self) -> bool: ...

    def extract_text(self, image: Any) -> str: ...


from .ocr_backends import (
    RapidOcrBackend,
    TesseractOcrBackend,
    _RAPIDOCR_RUNTIME_CACHE,
    _RAPIDOCR_RUNTIME_CACHE_LOCK,
    _rapidocr_text_from_output,
    _score_ocr_text,
)
from .ocr_capture import (
    DxcamCaptureBackend,
    ImageGrabCaptureBackend,
    PrintWindowCaptureBackend,
    Win32CaptureBackend,
)


def _default_window_scanner() -> list[DetectedGameWindow]:
    try:
        import win32gui
        import win32process
    except ImportError:
        return []

    results: list[DetectedGameWindow] = []
    foreground_hwnd = _foreground_window_handle()

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.IsIconic(hwnd):
            return
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        title = win32gui.GetWindowText(hwnd)
        if not title or len(title) < _WINDOW_TITLE_MIN_CHARS:
            return
        class_name = win32gui.GetClassName(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = ""
        exe_path = ""
        if psutil is not None:
            try:
                proc = psutil.Process(pid)
                process_name = proc.name()
                exe_path = proc.exe()
            except Exception as exc:
                _MODULE_LOGGER.debug(
                    "ocr_reader window scanner could not read process info for pid %s: %s",
                    pid,
                    exc,
                    exc_info=True,
                )
        area = width * height
        candidate = DetectedGameWindow(
            hwnd=hwnd,
            title=title,
            process_name=process_name,
            pid=pid,
            class_name=class_name,
            exe_path=exe_path,
            width=max(0, width),
            height=max(0, height),
            area=max(0, area),
            is_foreground=hwnd == foreground_hwnd,
            score=float(max(area, 0)),
        )
        candidate.is_foreground = _foreground_matches_target(foreground_hwnd, candidate)[0]
        results.append(_classify_window_candidate(candidate))

    win32gui.EnumWindows(callback, None)
    results.sort(key=_window_sort_key, reverse=True)
    return results


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _foreground_window_handle() -> int:
    try:
        return int(ctypes.windll.user32.GetForegroundWindow())
    except Exception as exc:
        _MODULE_LOGGER.debug("ocr_reader could not read foreground window handle: %s", exc, exc_info=True)
        return 0


def _window_handle_from_point(x: int, y: int) -> int:
    if os.name != "nt":
        return 0
    try:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        user32 = ctypes.windll.user32
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [POINT]
        return int(user32.WindowFromPoint(POINT(int(x), int(y))) or 0)
    except Exception as exc:
        _MODULE_LOGGER.debug("ocr_reader could not resolve window from point: %s", exc, exc_info=True)
        return 0


def _root_window_handle(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        root = int(ctypes.windll.user32.GetAncestor(int(hwnd), 2))
        return root or int(hwnd)
    except Exception as exc:
        _MODULE_LOGGER.debug("ocr_reader could not resolve root window handle for %s: %s", hwnd, exc, exc_info=True)
        return int(hwnd)


def _window_process_id(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        return int(pid.value or 0)
    except Exception as exc:
        _MODULE_LOGGER.debug("ocr_reader could not resolve process id for hwnd %s: %s", hwnd, exc, exc_info=True)
        return 0


def _window_process_name(pid: int) -> str:
    if not pid or psutil is None:
        return ""
    try:
        return str(psutil.Process(int(pid)).name() or "").strip()
    except Exception as exc:
        _MODULE_LOGGER.debug("ocr_reader could not resolve process name for pid %s: %s", pid, exc, exc_info=True)
        return ""


def _foreground_matches_target(foreground_hwnd: int, target: DetectedGameWindow | None) -> tuple[bool, str]:
    if target is None or not foreground_hwnd:
        return False, "no_foreground_or_target"
    target_hwnd = int(target.hwnd or 0)
    foreground_root_hwnd = _root_window_handle(int(foreground_hwnd))
    target_root_hwnd = _root_window_handle(target_hwnd)
    if target_hwnd and int(foreground_hwnd) == target_hwnd:
        return True, "hwnd"
    if target_root_hwnd and foreground_root_hwnd and foreground_root_hwnd == target_root_hwnd:
        return True, "root_hwnd"
    foreground_pid = _window_process_id(int(foreground_hwnd)) or _window_process_id(foreground_root_hwnd)
    target_pid = int(target.pid or 0)
    if foreground_pid and target_pid and foreground_pid == target_pid:
        return True, "pid"
    target_process = str(target.process_name or "").strip().lower()
    foreground_process = _window_process_name(foreground_pid).strip().lower()
    if foreground_process and target_process and foreground_process == target_process:
        return True, "process"
    return False, "background"


@dataclass(slots=True)
class _MouseWheelEvent:
    seq: int
    ts: float
    delta: int
    foreground_hwnd: int
    point_hwnd: int = 0
    kind: str = "wheel"


class _MouseWheelMonitor:
    _MAX_EVENTS = 96
    _MAX_EVENT_AGE_SECONDS = 15.0

    def __init__(self, *, time_fn: Callable[[], float]) -> None:
        self._time_fn = time_fn
        self._lock = threading.Lock()
        self._events: list[_MouseWheelEvent] = []
        self._seq = 0
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._hook_handle = 0
        self._callback = None
        self._stop = threading.Event()

    def start(self) -> bool:
        if os.name != "nt":
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._thread = None
        self._hook_handle = 0
        self._thread_id = 0
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="galgame-ocr-wheel-monitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def ensure_running(self) -> bool:
        return self.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def last_seq(self) -> int:
        with self._lock:
            return int(self._seq or 0)

    def stop(self) -> None:
        self._stop.set()
        if os.name == "nt" and self._thread_id:
            try:
                ctypes.windll.user32.PostThreadMessageW(
                    int(self._thread_id),
                    0x0012,  # WM_QUIT
                    0,
                    0,
                )
            except Exception:
                pass

    def events_after(self, seq: int) -> list[_MouseWheelEvent]:
        self.ensure_running()
        with self._lock:
            self._prune_locked()
            return [event for event in self._events if event.seq > seq]

    def _record(
        self,
        *,
        delta: int = 0,
        kind: str = "wheel",
        point_hwnd: int = 0,
        foreground_hwnd: int = 0,
    ) -> None:
        now = self._time_fn()
        with self._lock:
            self._seq += 1
            self._events.append(
                _MouseWheelEvent(
                    seq=self._seq,
                    ts=now,
                    delta=int(delta),
                    foreground_hwnd=max(0, int(foreground_hwnd or 0)),
                    point_hwnd=max(0, int(point_hwnd or 0)),
                    kind=str(kind or "wheel"),
                )
            )
            self._prune_locked(now=now)

    def _prune_locked(self, *, now: float | None = None) -> None:
        now = self._time_fn() if now is None else now
        min_ts = now - self._MAX_EVENT_AGE_SECONDS
        self._events = [
            event for event in self._events[-self._MAX_EVENTS :]
            if event.ts >= min_ts
        ]

    def _run(self) -> None:
        try:
            low_level_mouse_proc = getattr(ctypes, "WINFUNCTYPE", None)
            if low_level_mouse_proc is None:
                return
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            self._thread_id = int(kernel32.GetCurrentThreadId())

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            class MSLLHOOKSTRUCT(ctypes.Structure):
                _fields_ = [
                    ("pt", POINT),
                    ("mouseData", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.c_void_p),
                ]

            proc_type = low_level_mouse_proc(
                ctypes.c_longlong,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            hhook_type = getattr(wintypes, "HHOOK", wintypes.HANDLE)
            hinstance_type = getattr(wintypes, "HINSTANCE", wintypes.HANDLE)
            user32.CallNextHookEx.restype = ctypes.c_longlong
            user32.CallNextHookEx.argtypes = [
                hhook_type,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            ]

            def callback(n_code, w_param, l_param):
                message = int(w_param)
                if n_code >= 0 and message in {0x020A, 0x0201, 0x0202}:  # WM_MOUSEWHEEL, WM_LBUTTONDOWN, WM_LBUTTONUP
                    try:
                        payload = ctypes.cast(
                            l_param,
                            ctypes.POINTER(MSLLHOOKSTRUCT),
                        ).contents
                        point_hwnd = _window_handle_from_point(
                            int(payload.pt.x),
                            int(payload.pt.y),
                        )
                        foreground_hwnd = user32.GetForegroundWindow()
                        if message == 0x020A:
                            delta = ctypes.c_short((int(payload.mouseData) >> 16) & 0xFFFF).value
                            if delta:
                                self._record(
                                    delta=delta,
                                    kind="wheel",
                                    point_hwnd=point_hwnd,
                                    foreground_hwnd=foreground_hwnd,
                                )
                        else:
                            self._record(
                                kind="left_click",
                                point_hwnd=point_hwnd,
                                foreground_hwnd=foreground_hwnd,
                            )
                    except Exception:
                        pass
                return user32.CallNextHookEx(
                    self._hook_handle,
                    n_code,
                    w_param,
                    l_param,
                )

            self._callback = proc_type(callback)
            user32.SetWindowsHookExW.restype = hhook_type
            user32.SetWindowsHookExW.argtypes = [
                ctypes.c_int,
                proc_type,
                hinstance_type,
                wintypes.DWORD,
            ]
            self._hook_handle = int(user32.SetWindowsHookExW(14, self._callback, 0, 0))
            if not self._hook_handle:
                return

            msg = wintypes.MSG()
            while not self._stop.is_set():
                result = user32.GetMessageW(ctypes.byref(msg), 0, 0, 0)
                if result <= 0:
                    break
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
        finally:
            if self._hook_handle:
                try:
                    ctypes.windll.user32.UnhookWindowsHookEx(self._hook_handle)
                except Exception:
                    pass
            self._hook_handle = 0
            self._thread_id = 0


def _classify_window_candidate(candidate: DetectedGameWindow) -> DetectedGameWindow:
    normalized_title = candidate.normalized_title
    lowered_process_name = str(candidate.process_name or "").strip().lower()
    lowered_class_name = str(candidate.class_name or "").strip().lower()

    if candidate.area and candidate.area < (400 * 300):
        candidate.eligible = False
        candidate.exclude_reason = "excluded_small_or_hidden_window"
        candidate.category = "excluded_small_or_hidden_window"
        return candidate

    if _looks_like_self_window_title(candidate.title) or _looks_like_self_window_path(candidate.exe_path):
        candidate.eligible = False
        candidate.exclude_reason = "excluded_self_window"
        candidate.category = "excluded_self_window"
        return candidate

    if candidate.class_name in _HELPER_CLASS_NAMES:
        candidate.eligible = False
        candidate.exclude_reason = "excluded_helper_window"
        candidate.category = "excluded_helper_window"
        return candidate

    if any(token in normalized_title for token in _OVERLAY_WINDOW_TITLE_SUBSTRINGS):
        candidate.eligible = False
        candidate.exclude_reason = "excluded_overlay_window"
        candidate.category = "excluded_overlay_window"
        return candidate

    if lowered_process_name and any(
        token in lowered_process_name for token in _OVERLAY_PROCESS_NAME_SUBSTRINGS
    ):
        candidate.eligible = False
        candidate.exclude_reason = "excluded_overlay_window"
        candidate.category = "excluded_overlay_window"
        return candidate

    if lowered_class_name.startswith("chrome_widgetwin") and _looks_like_self_window_title(candidate.title):
        candidate.eligible = False
        candidate.exclude_reason = "excluded_self_window"
        candidate.category = "excluded_self_window"
        return candidate

    candidate.eligible = True
    candidate.exclude_reason = ""
    candidate.category = "eligible_game_window"
    return candidate


def _is_confident_auto_window(candidate: DetectedGameWindow) -> bool:
    if _matches_aihong_target(candidate):
        return True
    process_name = str(candidate.process_name or "").strip().lower()
    class_name = str(candidate.class_name or "").strip().lower()
    if process_name in _AUTO_TARGET_DENY_PROCESS_NAMES:
        return False
    if class_name.startswith("chrome_widgetwin"):
        return False
    return bool(candidate.hwnd and candidate.eligible)


def _window_sort_key(candidate: DetectedGameWindow) -> tuple[int, int, float, str]:
    return (
        1 if candidate.eligible else 0,
        1 if candidate.is_foreground else 0,
        float(candidate.score or 0.0),
        candidate.normalized_title,
    )


from .ocr_bridge import OcrReaderBridgeWriter


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
        self._custom_capture_backend = capture_backend is not None
        self._capture_backend = capture_backend or Win32CaptureBackend(
            logger=logger,
            selection=config.ocr_reader_capture_backend,
        )
        self._ocr_backend = ocr_backend
        self._custom_ocr_backend = ocr_backend is not None
        self._writer = writer or OcrReaderBridgeWriter(
            bridge_root=config.bridge_root,
            time_fn=self._time_fn,
        )
        self._runtime = OcrReaderRuntime(enabled=config.ocr_reader_enabled)
        self._capture_profiles: dict[str, ParsedOcrCaptureProcessConfig] = {}
        self._last_memory_reader_text_at = 0.0
        self._last_heartbeat_at = 0.0
        self._attached_window: DetectedGameWindow | None = None
        self._default_ocr_state = _StableOcrTextState()
        self._aihong_sm = _AihongStateMachine()
        self._manual_target = OcrWindowTarget()
        self._locked_target = OcrWindowTarget()
        self._last_detected_windows: list[DetectedGameWindow] = []
        self._last_eligible_windows: list[DetectedGameWindow] = []
        self._last_excluded_windows: list[DetectedGameWindow] = []
        self._last_selection = WindowSelectionResult(manual_target=self._manual_target)
        self._advance_speed = ADVANCE_SPEED_MEDIUM
        self._consecutive_no_text_polls = 0
        self._last_observed_at = ""
        self._last_capture_attempt_at = ""
        self._last_capture_completed_at = ""
        self._last_capture_error = ""
        self._last_raw_ocr_text = ""
        self._last_observed_line: dict[str, Any] = {}
        self._last_stable_line: dict[str, Any] = {}
        self._last_capture_image_hash = ""
        self._last_capture_source_size: dict[str, float] = {}
        self._last_capture_rect: dict[str, float] = {}
        self._last_capture_window_rect: dict[str, float] = {}
        self._last_capture_timing: dict[str, float | bool] = {}
        self._consecutive_same_capture_frames = 0
        self._stale_capture_backend = False
        self._last_background_hash = ""
        self._last_background_hash_capture_at = 0.0
        self._pending_background_hash = ""
        self._pending_background_change_count = 0
        self._pending_visual_scene_hash = ""
        self._pending_visual_scene_at = 0.0
        self._pending_visual_scene_count = 0
        self._last_auto_recalibrate_attempts = 0
        self._last_auto_recalibrate_duration_seconds = 0.0
        self._last_auto_recalibrate_limited = False
        self._last_auto_recalibrate_error = ""
        self._tick_lock = threading.Lock()
        self._wheel_monitor = _MouseWheelMonitor(time_fn=self._time_fn)
        self._wheel_monitor.start()
        self._last_consumed_wheel_seq = 0
        self._capture_backend_kind = str(getattr(self._capture_backend, "selection", "custom"))
        self._capture_backend_detail = ""
        self._rapidocr_backend_cache_key: tuple[str, str, str, str, str] | None = None
        self._rapidocr_backend_cache: RapidOcrBackend | None = None
        self._backend_plan_cache_key: tuple[str, ...] | None = None
        self._backend_plan_cache_at = 0.0
        self._backend_plan_cache: SelectedOcrBackendPlan | None = None
        self._start_rapidocr_warmup_if_configured()

    def update_config(self, config: GalgameConfig) -> None:
        self._config = config
        self._runtime.enabled = config.ocr_reader_enabled
        backend_plan_key = self._backend_plan_config_key(config)
        if self._backend_plan_cache_key != backend_plan_key:
            self._backend_plan_cache_key = None
            self._backend_plan_cache_at = 0.0
            self._backend_plan_cache = None
        if not self._custom_capture_backend:
            current_selection = str(getattr(self._capture_backend, "selection", "") or "")
            if current_selection != config.ocr_reader_capture_backend:
                self._capture_backend = Win32CaptureBackend(
                    logger=self._logger,
                    selection=config.ocr_reader_capture_backend,
                )
                self._capture_backend_kind = str(
                    getattr(self._capture_backend, "selection", "custom")
                )
                self._capture_backend_detail = ""
        self._start_rapidocr_warmup_if_configured()

    def _rapidocr_cache_key(self) -> tuple[str, str, str, str, str]:
        return (
            str(self._config.rapidocr_install_target_dir or ""),
            str(self._config.rapidocr_engine_type or ""),
            str(self._config.rapidocr_lang_type or ""),
            str(self._config.rapidocr_model_type or ""),
            str(self._config.rapidocr_ocr_version or ""),
        )

    def _rapidocr_backend_for_config(self) -> RapidOcrBackend:
        key = self._rapidocr_cache_key()
        if self._rapidocr_backend_cache_key == key and self._rapidocr_backend_cache is not None:
            return self._rapidocr_backend_cache
        backend = RapidOcrBackend(
            install_target_dir_raw=self._config.rapidocr_install_target_dir,
            engine_type=self._config.rapidocr_engine_type,
            lang_type=self._config.rapidocr_lang_type,
            model_type=self._config.rapidocr_model_type,
            ocr_version=self._config.rapidocr_ocr_version,
        )
        self._rapidocr_backend_cache_key = key
        self._rapidocr_backend_cache = backend
        return backend

    def _start_rapidocr_warmup_if_configured(self) -> None:
        if self._custom_ocr_backend or not bool(self._config.rapidocr_enabled):
            return
        selection = self._configured_backend_selection()
        if selection not in {"auto", "rapidocr"}:
            return
        self._rapidocr_backend_for_config().warmup_async(self._logger)
        if self._writer.bridge_root != self._config.bridge_root:
            self._writer = OcrReaderBridgeWriter(
                bridge_root=self._config.bridge_root,
                time_fn=self._time_fn,
            )

    def _custom_backend_plan(self) -> SelectedOcrBackendPlan:
        return SelectedOcrBackendPlan(
            selection="custom",
            primary=OcrBackendDescriptor(
                kind=str(self._runtime.backend_kind or "custom"),
                backend=self._ocr_backend,
                detail=str(self._runtime.backend_detail or "custom_backend"),
                available=True,
            ),
        )

    def update_advance_speed(self, advance_speed: str) -> None:
        normalized = str(advance_speed or "").strip().lower()
        self._advance_speed = normalized if normalized in ADVANCE_SPEEDS else ADVANCE_SPEED_MEDIUM

    def _line_changed_repeat_threshold(self) -> int:
        if self._advance_speed == ADVANCE_SPEED_FAST:
            return _OCR_LINE_REPEAT_THRESHOLD_FAST
        if self._advance_speed == ADVANCE_SPEED_SLOW:
            return _OCR_LINE_REPEAT_THRESHOLD_SLOW
        return _OCR_LINE_REPEAT_THRESHOLD_MEDIUM

    def _should_emit_observed_lines_for_capture(self, *, after_advance_trigger_mode: bool) -> bool:
        # after_advance 模式下应 emit observed，提供即时反馈
        return True

    def _mark_observed_progress(self, *, now: float) -> None:
        self._consecutive_no_text_polls = 0
        self._last_observed_at = utc_now_iso(now)

    def _mark_no_text_poll(self) -> None:
        self._consecutive_no_text_polls += 1

    def _ocr_capture_diagnostic_required(self) -> bool:
        return self._consecutive_no_text_polls >= _OCR_CAPTURE_DIAGNOSTIC_NO_TEXT_POLLS

    def _record_capture_attempt(self, *, now: float) -> None:
        self._last_capture_attempt_at = utc_now_iso(now)
        self._last_capture_error = ""

    def _record_capture_completed(self, *, now: float, raw_text: str, image_hash: str = "") -> None:
        self._last_capture_completed_at = utc_now_iso(now)
        self._last_raw_ocr_text = str(raw_text or "")
        self._last_capture_error = ""
        if image_hash:
            if image_hash == self._last_capture_image_hash:
                self._consecutive_same_capture_frames += 1
            else:
                self._last_capture_image_hash = image_hash
                self._consecutive_same_capture_frames = 1
            self._stale_capture_backend = (
                self._consecutive_same_capture_frames >= _STALE_CAPTURE_FRAME_THRESHOLD
            )

    def _record_capture_geometry(self, extraction: OcrExtractionResult) -> None:
        self._last_capture_source_size = dict(extraction.source_size or {})
        self._last_capture_rect = dict(extraction.capture_rect or {})
        self._last_capture_window_rect = dict(extraction.window_rect or {})

    def _record_capture_error(self, *, now: float, error: Exception) -> None:
        if not self._last_capture_attempt_at:
            self._last_capture_attempt_at = utc_now_iso(now)
        self._last_capture_error = str(error)

    @staticmethod
    def _capture_image_hash(frame: Any) -> str:
        if frame is None:
            return ""
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return hashlib.blake2b(bytes(frame), digest_size=8).hexdigest()
        if isinstance(frame, str):
            return hashlib.blake2b(frame.encode("utf-8", "ignore"), digest_size=8).hexdigest()
        try:
            if hasattr(frame, "tobytes"):
                size = getattr(frame, "size", "")
                mode = getattr(frame, "mode", "")
                shape = getattr(frame, "shape", "")
                payload = frame.tobytes()
                metadata = f"{size!r}|{mode!r}|{shape!r}".encode("utf-8", "ignore")
                return hashlib.blake2b(metadata + payload, digest_size=8).hexdigest()
        except Exception as exc:
            self._logger.debug("ocr_reader capture hash bytes fallback failed: %s", exc, exc_info=True)
        try:
            background_hash = _perceptual_hash_image(frame)
            if background_hash:
                return f"phash:{background_hash}"
        except Exception as exc:
            self._logger.debug("ocr_reader perceptual capture hash failed: %s", exc, exc_info=True)
            return ""
        return ""

    @staticmethod
    def _background_capture_profile() -> OcrCaptureProfile:
        return OcrCaptureProfile(
            left_inset_ratio=0.0,
            right_inset_ratio=0.0,
            top_ratio=0.0,
            bottom_inset_ratio=_BACKGROUND_HASH_BOTTOM_INSET_RATIO,
        )

    @staticmethod
    def _background_perceptual_hash(frame: Any) -> str:
        return _perceptual_hash_image(frame)

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        if not left or not right:
            return 0
        left = str(left or "").strip()
        right = str(right or "").strip()
        if left.startswith("phash:"):
            left = left.split(":", 1)[1]
        if right.startswith("phash:"):
            right = right.split(":", 1)[1]
        try:
            return (int(left, 16) ^ int(right, 16)).bit_count()
        except Exception:
            return 0

    def _observe_background_hash(
        self,
        background_hash: str,
        *,
        now: float,
        confirm_polls: int = _BACKGROUND_SCENE_CHANGE_CONFIRM_POLLS,
        defer_scene_emit: bool = False,
    ) -> bool:
        if not background_hash:
            return False
        if not self._last_background_hash:
            self._last_background_hash = background_hash
            self._pending_background_hash = ""
            self._pending_background_change_count = 0
            return False
        distance = self._hash_distance(self._last_background_hash, background_hash)
        if distance < _BACKGROUND_SCENE_CHANGE_DISTANCE:
            self._pending_background_hash = ""
            self._pending_background_change_count = 0
            return False
        if background_hash == self._pending_background_hash:
            self._pending_background_change_count += 1
        else:
            self._pending_background_hash = background_hash
            self._pending_background_change_count = 1
        required_confirm_polls = max(1, int(confirm_polls or 1))
        if self._pending_background_change_count < required_confirm_polls:
            return False
        self._last_background_hash = background_hash
        self._pending_background_hash = ""
        self._pending_background_change_count = 0
        self._default_ocr_state.reset()
        self._aihong_sm.menu_ocr_state.reset()
        if defer_scene_emit:
            self._pending_visual_scene_count = 1
            self._pending_visual_scene_hash = background_hash
            self._pending_visual_scene_at = now
            return False
        return bool(
            self._writer.advance_visual_scene(
                ts=utc_now_iso(now),
                background_hash=background_hash,
            )
        )

    def _commit_pending_visual_scene(self, *, now: float) -> bool:
        background_hash = str(self._pending_visual_scene_hash or "")
        if not background_hash:
            return False
        scene_at = float(self._pending_visual_scene_at or now)
        scene_count = max(1, int(self._pending_visual_scene_count or 1))
        self._pending_visual_scene_hash = ""
        self._pending_visual_scene_at = 0.0
        self._pending_visual_scene_count = 0
        committed = False
        for _index in range(scene_count):
            committed = bool(
                self._writer.advance_visual_scene(
                    ts=utc_now_iso(scene_at if scene_at > 0 else now),
                    background_hash=background_hash,
                )
            ) or committed
        return committed

    def _line_payload_from_writer(self, *, stability: str) -> dict[str, Any]:
        state = self._writer.current_state
        if not isinstance(state, dict):
            return {}
        text = str(state.get("text") or "")
        if not text:
            return {}
        return {
            "line_id": str(state.get("line_id") or ""),
            "speaker": str(state.get("speaker") or ""),
            "text": text,
            "scene_id": str(state.get("scene_id") or ""),
            "route_id": str(state.get("route_id") or ""),
            "stability": stability,
            "ts": str(state.get("ts") or ""),
        }

    def _ocr_context_state_for_detail(self, *, status: str, detail: str) -> str:
        detail = str(detail or "")
        if not self._runtime.enabled and not self._config.ocr_reader_enabled:
            return "disabled"
        if detail == "starting_capture":
            return "capture_pending"
        if detail == "capture_failed":
            return "capture_failed"
        if self._stale_capture_backend:
            return "stale_capture_backend"
        if detail == "ocr_capture_diagnostic_required" or self._ocr_capture_diagnostic_required():
            return "diagnostic_required"
        if detail in {"attached_no_text_yet", "self_ui_guard_blocked"}:
            return "no_text"
        state = self._writer.current_state
        stability = str(state.get("stability") or "") if isinstance(state, dict) else ""
        if stability == "choices":
            return "choices"
        if detail == "receiving_text" or stability == "stable":
            return "stable"
        if detail == "receiving_observed_text" or stability == "tentative":
            return "observed"
        if detail in {"backend_unavailable", "capture_backend_unavailable"}:
            return "capture_failed"
        if str(status or "") == "starting":
            return "capture_pending"
        return detail or str(status or "")

    def update_capture_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        self._capture_profiles = _parse_configured_capture_profiles(profiles, self._logger)

    def update_window_target(self, target: dict[str, Any] | None) -> None:
        self._manual_target = OcrWindowTarget.from_dict(target)
        self._locked_target = OcrWindowTarget()
        self._consecutive_no_text_polls = 0
        self._last_selection = WindowSelectionResult(
            selection_mode="manual" if self._manual_target.is_manual() else "auto",
            selection_detail="manual_target_active"
            if self._manual_target.is_manual()
            else "auto_candidate_scan",
            manual_target=self._manual_target,
            candidate_count=len(self._last_eligible_windows),
            excluded_candidate_count=len(self._last_excluded_windows),
            last_exclude_reason=(
                str(self._last_excluded_windows[0].exclude_reason or "")
                if self._last_excluded_windows
                else ""
            ),
        )

    def current_window_target(self) -> dict[str, Any]:
        return self._manual_target.to_dict()

    def refresh_foreground_state(self) -> dict[str, Any]:
        if not self._config.ocr_reader_enabled or not self._platform_fn():
            return self._runtime.to_dict()
        foreground_hwnd = _foreground_window_handle()
        target, detail = self._foreground_refresh_target()
        target_hwnd = int(target.hwnd or 0) if target is not None else 0
        window_changes: dict[str, Any] = {}
        if target is not None:
            is_foreground, foreground_match_reason = _foreground_matches_target(
                foreground_hwnd,
                target,
            )
            window_changes.update(
                target_is_foreground=is_foreground,
                effective_window_key=str(target.window_key or self._runtime.effective_window_key),
                effective_window_title=str(target.title or self._runtime.effective_window_title),
                effective_process_name=str(
                    target.process_name or self._runtime.effective_process_name
                ),
            )
            if not self._runtime.process_name:
                window_changes["process_name"] = str(target.process_name or "")
            if not self._runtime.window_title:
                window_changes["title"] = str(target.title or "")
            if not self._runtime.pid:
                window_changes["pid"] = int(target.pid or 0)
            detail = (
                f"{detail}:foreground_{foreground_match_reason}"
                if is_foreground
                else f"{detail}:background"
            )
        elif self._runtime.effective_window_key or self._runtime.process_name:
            detail = detail or "target_unresolved"
        else:
            detail = "no_target"
        self._runtime.update_window_state(
            **window_changes,
            foreground_refresh_at=utc_now_iso(self._time_fn()),
            foreground_refresh_detail=detail,
            foreground_hwnd=max(0, int(foreground_hwnd or 0)),
            target_hwnd=max(0, int(target_hwnd or 0)),
        )
        return self._runtime.to_dict()

    def consume_foreground_advance_input(self) -> bool:
        self._wheel_monitor.ensure_running()
        self._runtime.update_advance_state(
            foreground_monitor_running=self._wheel_monitor.is_running(),
            foreground_last_seq=self._wheel_monitor.last_seq(),
            foreground_consumed_seq=self._last_consumed_wheel_seq,
        )
        target, _detail = self._foreground_refresh_target()
        if target is None:
            target = self._attached_window
        if target is None and (
            self._runtime.target_hwnd
            or self._runtime.pid
            or self._runtime.effective_process_name
            or self._runtime.process_name
        ):
            target = DetectedGameWindow(
                hwnd=int(self._runtime.target_hwnd or 0),
                title=str(self._runtime.effective_window_title or self._runtime.window_title or ""),
                process_name=str(
                    self._runtime.effective_process_name
                    or self._runtime.process_name
                    or ""
                ),
                pid=int(self._runtime.pid or 0),
                width=int(self._runtime.width or 0),
                height=int(self._runtime.height or 0),
            )
        if target is None:
            return False
        events = self._wheel_monitor.events_after(self._last_consumed_wheel_seq)
        self._runtime.update_advance_state(
            foreground_monitor_running=self._wheel_monitor.is_running(),
            foreground_last_seq=self._wheel_monitor.last_seq(),
        )
        if not events:
            return False
        triggered = False
        max_seq = self._last_consumed_wheel_seq
        last_kind = ""
        last_delta = 0
        last_matched = False
        last_match_reason = ""
        for event in events:
            max_seq = max(max_seq, int(event.seq or 0))
            last_kind = str(event.kind or "")
            last_delta = int(event.delta or 0)
            if event.kind == "wheel" and event.delta >= 0:
                last_match_reason = "ignored_wheel_up"
                continue
            if event.kind not in {"wheel", "left_click"}:
                last_match_reason = "ignored_event_kind"
                continue
            is_target_foreground, foreground_reason = _foreground_matches_target(
                event.foreground_hwnd,
                target,
            )
            is_target_under_pointer, point_reason = _foreground_matches_target(
                event.point_hwnd,
                target,
            )
            if is_target_foreground or is_target_under_pointer:
                triggered = True
                last_matched = True
                last_match_reason = (
                    f"foreground_{foreground_reason}"
                    if is_target_foreground
                    else f"point_{point_reason}"
                )
            else:
                last_match_reason = f"background:{foreground_reason}/{point_reason}"
        self._last_consumed_wheel_seq = max_seq
        self._runtime.update_advance_state(
            foreground_consumed_seq=self._last_consumed_wheel_seq,
            foreground_last_kind=last_kind,
            foreground_last_delta=last_delta,
            foreground_last_matched=last_matched,
            foreground_last_match_reason=last_match_reason,
        )
        return triggered

    def consume_foreground_wheel_down(self) -> bool:
        return self.consume_foreground_advance_input()

    def _foreground_refresh_target(self) -> tuple[DetectedGameWindow | None, str]:
        windows = list(self._last_detected_windows or [])
        for target, detail in (
            (self._manual_target, "manual_target"),
            (self._locked_target, "locked_target"),
        ):
            if not isinstance(target, OcrWindowTarget):
                continue
            if not (
                target.window_key
                or target.last_known_hwnd
                or target.pid
                or target.process_name
                or target.normalized_title
            ):
                continue
            for candidate in windows:
                if target.matches_exact(candidate) or target.matches_hwnd(candidate):
                    return candidate, f"{detail}_exact"
            for candidate in windows:
                if target.matches_signature(candidate):
                    return candidate, f"{detail}_rebound"
        runtime_key = str(self._runtime.effective_window_key or "").strip()
        runtime_process = str(self._runtime.effective_process_name or self._runtime.process_name or "").strip().lower()
        runtime_pid = int(self._runtime.pid or 0)
        if runtime_key:
            for candidate in windows:
                if candidate.window_key == runtime_key:
                    return candidate, "runtime_effective_key"
        if runtime_pid > 0:
            for candidate in windows:
                if candidate.pid == runtime_pid:
                    return candidate, "runtime_pid"
        if runtime_process:
            for candidate in windows:
                if candidate.process_name.strip().lower() == runtime_process:
                    return candidate, "runtime_process"
        return None, "target_unresolved"

    def _has_locked_target(self) -> bool:
        return bool(
            self._locked_target.window_key
            or self._locked_target.last_known_hwnd
            or self._locked_target.pid
            or self._locked_target.process_name
            or self._locked_target.normalized_title
        )

    def _remember_locked_target(self, target: DetectedGameWindow) -> None:
        if self._manual_target.is_manual():
            return
        self._locked_target = OcrWindowTarget(
            mode="auto",
            window_key=target.window_key,
            process_name=target.process_name,
            normalized_title=target.normalized_title,
            pid=target.pid,
            last_known_hwnd=target.hwnd,
            selected_at=utc_now_iso(self._time_fn()),
        )

    def list_windows_snapshot(self, *, include_excluded: bool = False) -> dict[str, Any]:
        eligible_windows, excluded_windows = self._scan_window_inventory()
        payload = {
            "target_selection_mode": self._manual_target.mode,
            "manual_target": self._manual_target.to_dict(),
            "candidate_count": len(eligible_windows),
            "excluded_candidate_count": len(excluded_windows),
            "windows": [
                candidate.to_dict(
                    is_attached=self._matches_attached_window(candidate),
                    is_manual_target=self._manual_target.is_manual()
                    and (
                        self._manual_target.matches_exact(candidate)
                        or self._manual_target.matches_signature(candidate)
                    ),
                )
                for candidate in eligible_windows
            ],
        }
        if include_excluded:
            payload["excluded_windows"] = [
                candidate.to_dict(
                    is_attached=self._matches_attached_window(candidate),
                    is_manual_target=False,
                )
                for candidate in excluded_windows
            ]
        return payload

    def resolve_manual_window_target(self, window_key: str) -> dict[str, Any]:
        normalized_key = str(window_key or "").strip()
        if not normalized_key:
            raise ValueError("window_key is required")
        eligible_windows, excluded_windows = self._scan_window_inventory()
        for candidate in eligible_windows:
            if candidate.window_key == normalized_key:
                return OcrWindowTarget(
                    mode="manual",
                    window_key=candidate.window_key,
                    process_name=candidate.process_name,
                    normalized_title=candidate.normalized_title,
                    pid=candidate.pid,
                    last_known_hwnd=candidate.hwnd,
                    selected_at=utc_now_iso(self._time_fn()),
                ).to_dict()
        for candidate in excluded_windows:
            if candidate.window_key == normalized_key:
                raise ValueError("window_key points to an excluded OCR window")
        raise ValueError("window_key not found among eligible OCR windows")

    def runtime(self) -> dict[str, Any]:
        return self._runtime.to_dict()

    def refresh_runtime_capture_profile_selection(self) -> dict[str, Any]:
        target = self._attached_window
        if target is None:
            return self._runtime.to_dict()

        if target.width <= 0 and self._runtime.width > 0:
            target.width = int(self._runtime.width)
        if target.height <= 0 and self._runtime.height > 0:
            target.height = int(self._runtime.height)
        resolved_aspect_ratio = float(target.aspect_ratio or self._runtime.aspect_ratio)
        if resolved_aspect_ratio <= 0.0 and target.width > 0 and target.height > 0:
            resolved_aspect_ratio = compute_ocr_window_aspect_ratio(target.width, target.height)

        capture_stage = str(self._runtime.capture_stage or "").strip().lower()
        if not capture_stage or capture_stage == OCR_CAPTURE_PROFILE_STAGE_DEFAULT:
            capture_stage = (
                self._aihong_sm.capture_stage
                if self._should_use_aihong_two_stage(target)
                else OCR_CAPTURE_PROFILE_STAGE_DIALOGUE
            )
        capture_profile_selection = self._capture_profile_selection_for_target(
            target,
            stage=capture_stage,
        )
        self._runtime.update_window_state(
            process_name=str(target.process_name or self._runtime.process_name),
            pid=int(target.pid or self._runtime.pid),
            title=str(target.title or self._runtime.window_title),
            width=int(target.width or self._runtime.width),
            height=int(target.height or self._runtime.height),
            aspect_ratio=resolved_aspect_ratio,
        )
        self._runtime.update_capture_state(
            stage=capture_stage,
            profile=capture_profile_selection.profile.to_dict(),
            profile_match_source=capture_profile_selection.match_source,
            profile_bucket_key=capture_profile_selection.bucket_key,
            last_stage=capture_stage,
            last_profile=capture_profile_selection.profile.to_dict(),
            diagnostic_required=self._ocr_capture_diagnostic_required(),
        )
        self._runtime.update_ocr_state(
            consecutive_no_text_polls=max(0, int(self._consecutive_no_text_polls or 0)),
            last_observed_at=str(self._last_observed_at or self._runtime.last_observed_at),
            context_state=self._ocr_context_state_for_detail(
                status=self._runtime.status,
                detail=self._runtime.detail,
            ),
            last_capture_attempt_at=str(
                self._last_capture_attempt_at or self._runtime.last_capture_attempt_at
            ),
            last_capture_completed_at=str(
                self._last_capture_completed_at or self._runtime.last_capture_completed_at
            ),
            last_capture_error=str(self._last_capture_error or self._runtime.last_capture_error),
            last_raw_text=str(self._last_raw_ocr_text or self._runtime.last_raw_ocr_text),
            last_observed_line=dict(self._last_observed_line or self._runtime.last_observed_line),
            last_stable_line=dict(self._last_stable_line or self._runtime.last_stable_line),
        )
        foreground_hwnd = _foreground_window_handle()
        self._runtime.update_window_state(
            effective_window_key=str(target.window_key or self._runtime.effective_window_key),
            effective_window_title=str(target.title or self._runtime.effective_window_title),
            effective_process_name=str(
                target.process_name or self._runtime.effective_process_name
            ),
            target_is_foreground=_foreground_matches_target(
                foreground_hwnd,
                target,
            )[0],
            foreground_hwnd=max(0, int(foreground_hwnd or 0)),
            target_hwnd=max(0, int(target.hwnd or 0)),
        )
        return self._runtime.to_dict()

    @staticmethod
    def _scan_ratio_values(
        current_value: float,
        *,
        delta_start: float,
        delta_end: float,
        step: float,
    ) -> list[float]:
        values: list[float] = []
        seen: set[int] = set()
        basis = _OCR_RATIO_PERCENT_BASIS
        start = int(round((current_value + delta_start) * basis))
        end = int(round((current_value + delta_end) * basis))
        step_value = max(1, int(round(step * basis)))
        for raw in range(start, end + 1, step_value):
            normalized = max(_OCR_RATIO_MIN, min(raw / basis, _OCR_RATIO_MAX))
            key = int(round(normalized * basis))
            if key in seen:
                continue
            seen.add(key)
            values.append(round(normalized, _OCR_RATIO_ROUND_DIGITS))
        return values

    @staticmethod
    def _crop_box_for_profile_size(
        *,
        width: int,
        height: int,
        profile: OcrCaptureProfile,
    ) -> tuple[int, int, int, int]:
        left = int(width * profile.left_inset_ratio)
        right = int(width * (1.0 - profile.right_inset_ratio))
        top = int(height * profile.top_ratio)
        bottom = int(height * (1.0 - profile.bottom_inset_ratio))
        left = max(0, min(left, width))
        right = max(left, min(right, width))
        top = max(0, min(top, height))
        bottom = max(top, min(bottom, height))
        return (left, top, right, bottom)

    def _record_auto_recalibrate_diagnostic(
        self,
        *,
        started_at: float,
        attempts: int,
        limited: bool,
        error: str = "",
    ) -> None:
        self._last_auto_recalibrate_attempts = max(0, int(attempts or 0))
        self._last_auto_recalibrate_duration_seconds = max(
            0.0,
            float(self._time_fn() - started_at),
        )
        self._last_auto_recalibrate_limited = bool(limited)
        self._last_auto_recalibrate_error = str(error or "")

    def auto_recalibrate_dialogue_profile(self) -> dict[str, Any]:
        started_at = self._time_fn()
        self._record_auto_recalibrate_diagnostic(
            started_at=started_at,
            attempts=0,
            limited=False,
            error="",
        )
        if not self._config.ocr_reader_enabled:
            raise ValueError("ocr_reader 未启用，无法自动重校准对白区")
        if not self._platform_fn():
            raise ValueError("当前平台不是 Windows，无法自动重校准对白区")
        if not self._capture_backend.is_available():
            raise ValueError("当前截图后端不可用，无法自动重校准对白区")
        target = self._attached_window
        if target is None:
            raise ValueError("当前没有已附着的 OCR 目标窗口，无法自动重校准对白区")
        process_name = str(target.process_name or "").strip()
        if not process_name:
            raise ValueError("当前 OCR 目标缺少进程名，无法自动重校准对白区")

        full_window_profile = OcrCaptureProfile(
            left_inset_ratio=0.0,
            right_inset_ratio=0.0,
            top_ratio=0.0,
            bottom_inset_ratio=0.0,
        )
        full_image = self._capture_backend.capture_frame(target, full_window_profile)
        image_size = getattr(full_image, "size", None)
        if (
            not isinstance(image_size, tuple)
            or len(image_size) < _OCR_AUTO_RECALIBRATE_IMAGE_SIZE_DIMENSIONS
            or int(image_size[0]) <= 0
            or int(image_size[1]) <= 0
            or not hasattr(full_image, "crop")
        ):
            raise ValueError("当前截图后端不支持自动重校准所需的整窗截图")

        image_width = int(image_size[0])
        image_height = int(image_size[1])
        if target.width <= 0:
            target.width = image_width
        if target.height <= 0:
            target.height = image_height

        base_selection = self._capture_profile_selection_for_target(
            target,
            stage=OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
        )
        base_profile = base_selection.profile
        is_aihong_target = _matches_aihong_target(target)

        def _append_ratio_values(values: list[float], additions: Iterable[float]) -> list[float]:
            merged = list(values)
            seen = {int(round(value * _OCR_RATIO_PERCENT_BASIS)) for value in merged}
            for raw in additions:
                normalized = round(
                    max(_OCR_RATIO_MIN, min(float(raw), _OCR_RATIO_MAX)),
                    _OCR_RATIO_ROUND_DIGITS,
                )
                key = int(round(normalized * _OCR_RATIO_PERCENT_BASIS))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(normalized)
            return sorted(merged)

        horizontal_pairs: list[tuple[float, float]] = []

        def _add_horizontal_pair(left_ratio: float, right_ratio: float) -> None:
            left_ratio = round(
                max(
                    _OCR_RATIO_MIN,
                    min(float(left_ratio), _OCR_AUTO_RECALIBRATE_HORIZONTAL_MAX_INSET_RATIO),
                ),
                _OCR_RATIO_ROUND_DIGITS,
            )
            right_ratio = round(
                max(
                    _OCR_RATIO_MIN,
                    min(float(right_ratio), _OCR_AUTO_RECALIBRATE_HORIZONTAL_MAX_INSET_RATIO),
                ),
                _OCR_RATIO_ROUND_DIGITS,
            )
            if (
                left_ratio + right_ratio
                >= _OCR_AUTO_RECALIBRATE_MAX_TOTAL_HORIZONTAL_INSET_RATIO
            ):
                return
            pair = (left_ratio, right_ratio)
            if pair not in horizontal_pairs:
                horizontal_pairs.append(pair)

        if is_aihong_target:
            for left_ratio, right_ratio in _OCR_AUTO_RECALIBRATE_AIHONG_HORIZONTAL_PAIRS:
                _add_horizontal_pair(left_ratio, right_ratio)
        _add_horizontal_pair(base_profile.left_inset_ratio, base_profile.right_inset_ratio)
        if not is_aihong_target and (
            base_profile.left_inset_ratio > _OCR_RATIO_MIN
            or base_profile.right_inset_ratio > _OCR_RATIO_MIN
        ):
            _add_horizontal_pair(
                max(
                    _OCR_RATIO_MIN,
                    base_profile.left_inset_ratio
                    - _OCR_AUTO_RECALIBRATE_HORIZONTAL_SHRINK_DELTA,
                ),
                max(
                    _OCR_RATIO_MIN,
                    base_profile.right_inset_ratio
                    - _OCR_AUTO_RECALIBRATE_HORIZONTAL_SHRINK_DELTA,
                ),
            )

        top_values = self._scan_ratio_values(
            base_profile.top_ratio,
            delta_start=_OCR_AUTO_RECALIBRATE_TOP_SCAN_DELTA_START,
            delta_end=_OCR_AUTO_RECALIBRATE_TOP_SCAN_DELTA_END,
            step=_OCR_AUTO_RECALIBRATE_TOP_SCAN_STEP,
        )
        bottom_values = self._scan_ratio_values(
            base_profile.bottom_inset_ratio,
            delta_start=_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_DELTA_START,
            delta_end=_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_DELTA_END,
            step=_OCR_AUTO_RECALIBRATE_BOTTOM_SCAN_STEP,
        )
        if is_aihong_target:
            aihong_preset = OcrCaptureProfile.from_dict(_AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET)
            top_values = _append_ratio_values(
                top_values,
                self._scan_ratio_values(
                    aihong_preset.top_ratio,
                    delta_start=_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_DELTA_START,
                    delta_end=_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_DELTA_END,
                    step=_OCR_AUTO_RECALIBRATE_AIHONG_TOP_SCAN_STEP,
                ),
            )
            bottom_values = _append_ratio_values(
                bottom_values,
                self._scan_ratio_values(
                    aihong_preset.bottom_inset_ratio,
                    delta_start=_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_DELTA_START,
                    delta_end=_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_DELTA_END,
                    step=_OCR_AUTO_RECALIBRATE_AIHONG_BOTTOM_SCAN_STEP,
                ),
            )
        backend_plan = None if self._custom_ocr_backend else self._resolve_backend_plan()
        if backend_plan is not None and not backend_plan.primary.available:
            raise ValueError("当前 OCR backend 不可用，无法自动重校准对白区")

        best_candidate: dict[str, Any] | None = None
        current_distance_basis = (
            round(base_profile.top_ratio, _OCR_RATIO_ROUND_DIGITS),
            round(base_profile.bottom_inset_ratio, _OCR_RATIO_ROUND_DIGITS),
        )
        min_height = max(
            _OCR_AUTO_RECALIBRATE_MIN_CROP_HEIGHT_PX,
            int(image_height * _OCR_AUTO_RECALIBRATE_MIN_CROP_HEIGHT_RATIO),
        )
        max_height = max(
            min_height,
            int(image_height * _OCR_AUTO_RECALIBRATE_MAX_CROP_HEIGHT_RATIO),
        )
        visited_pairs: set[tuple[float, float, float, float]] = set()
        ocr_attempts = 0
        scan_exhausted = False

        def _consider_candidate(
            top_ratio: float,
            bottom_inset_ratio: float,
            left_inset_ratio: float,
            right_inset_ratio: float,
        ) -> None:
            nonlocal best_candidate, ocr_attempts, scan_exhausted
            if scan_exhausted:
                return
            key = (
                round(top_ratio, _OCR_RATIO_ROUND_DIGITS),
                round(bottom_inset_ratio, _OCR_RATIO_ROUND_DIGITS),
                round(left_inset_ratio, _OCR_RATIO_ROUND_DIGITS),
                round(right_inset_ratio, _OCR_RATIO_ROUND_DIGITS),
            )
            if key in visited_pairs:
                return
            visited_pairs.add(key)
            if (
                top_ratio + bottom_inset_ratio >= 1.0
                or left_inset_ratio + right_inset_ratio >= 1.0
            ):
                return
            candidate_profile = OcrCaptureProfile(
                left_inset_ratio=left_inset_ratio,
                right_inset_ratio=right_inset_ratio,
                top_ratio=top_ratio,
                bottom_inset_ratio=bottom_inset_ratio,
            )
            left_px, top_px, right_px, bottom_px = self._crop_box_for_profile_size(
                width=image_width,
                height=image_height,
                profile=candidate_profile,
            )
            crop_height = bottom_px - top_px
            if crop_height < min_height or crop_height > max_height:
                return
            if right_px - left_px < _OCR_AUTO_RECALIBRATE_MIN_CROP_WIDTH_PX:
                return
            if (
                self._time_fn() - started_at >= _OCR_AUTO_RECALIBRATE_MAX_SECONDS
                or ocr_attempts >= _OCR_AUTO_RECALIBRATE_MAX_OCR_ATTEMPTS
            ):
                scan_exhausted = True
                return
            ocr_attempts += 1
            extracted = self._extract_text_from_image(
                full_image.crop((left_px, top_px, right_px, bottom_px)),
                plan=backend_plan,
            )
            sample_text = str(extracted.text or "").strip()
            if not sample_text or _looks_like_self_ui_text(sample_text):
                return
            score, cjk_count, significant_chars = _score_ocr_text(sample_text)
            if (
                significant_chars < _OCR_AUTO_RECALIBRATE_MIN_CANDIDATE_SIGNIFICANT_CHARS
                or cjk_count <= 0
            ):
                return
            distance = abs(
                round(top_ratio, _OCR_RATIO_ROUND_DIGITS) - current_distance_basis[0]
            ) + abs(
                round(bottom_inset_ratio, _OCR_RATIO_ROUND_DIGITS)
                - current_distance_basis[1]
            )
            width_ratio = max(_OCR_RATIO_MIN, 1.0 - left_inset_ratio - right_inset_ratio)
            candidate = {
                "profile": candidate_profile,
                "sample_text": sample_text,
                "score": score,
                "cjk_count": cjk_count,
                "significant_chars": significant_chars,
                "distance": distance,
                "width_ratio": width_ratio,
            }
            if best_candidate is None:
                best_candidate = candidate
                return
            if (
                (candidate["score"], candidate["cjk_count"], candidate["significant_chars"])
                > (
                    best_candidate["score"],
                    best_candidate["cjk_count"],
                    best_candidate["significant_chars"],
                )
                or (
                    (
                        candidate["score"],
                        candidate["cjk_count"],
                        candidate["significant_chars"],
                    )
                    == (
                        best_candidate["score"],
                        best_candidate["cjk_count"],
                        best_candidate["significant_chars"],
                    )
                    and (
                        candidate["width_ratio"] > best_candidate["width_ratio"]
                        or (
                            candidate["width_ratio"] == best_candidate["width_ratio"]
                            and candidate["distance"] < best_candidate["distance"]
                        )
                    )
                )
            ):
                best_candidate = candidate

        preferred_bottom_values: list[float] = []
        for delta in _OCR_AUTO_RECALIBRATE_PREFERRED_BOTTOM_DELTAS:
            candidate_value = round(
                base_profile.bottom_inset_ratio + delta,
                _OCR_RATIO_ROUND_DIGITS,
            )
            if candidate_value in bottom_values and candidate_value not in preferred_bottom_values:
                preferred_bottom_values.append(candidate_value)
        if is_aihong_target:
            preset_bottom = round(
                float(
                    OcrCaptureProfile.from_dict(
                        _AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET
                    ).bottom_inset_ratio
                ),
                _OCR_RATIO_ROUND_DIGITS,
            )
            if preset_bottom in bottom_values and preset_bottom not in preferred_bottom_values:
                preferred_bottom_values.append(preset_bottom)
        if not preferred_bottom_values:
            preferred_bottom_values = list(bottom_values)

        preferred_top_values: list[float] = []
        if is_aihong_target:
            preset_top = round(
                float(
                    OcrCaptureProfile.from_dict(
                        _AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET
                    ).top_ratio
                ),
                _OCR_RATIO_ROUND_DIGITS,
            )
            for delta in _OCR_AUTO_RECALIBRATE_AIHONG_PREFERRED_TOP_DELTAS:
                candidate_value = round(preset_top + delta, _OCR_RATIO_ROUND_DIGITS)
                if candidate_value in top_values and candidate_value not in preferred_top_values:
                    preferred_top_values.append(candidate_value)
        for delta in _OCR_AUTO_RECALIBRATE_BASE_PREFERRED_TOP_DELTAS:
            candidate_value = round(
                base_profile.top_ratio + delta,
                _OCR_RATIO_ROUND_DIGITS,
            )
            if candidate_value in top_values and candidate_value not in preferred_top_values:
                preferred_top_values.append(candidate_value)
        if not preferred_top_values:
            preferred_top_values = list(top_values)

        for top_ratio in preferred_top_values:
            for bottom_inset_ratio in preferred_bottom_values:
                for left_inset_ratio, right_inset_ratio in horizontal_pairs:
                    _consider_candidate(
                        top_ratio,
                        bottom_inset_ratio,
                        left_inset_ratio,
                        right_inset_ratio,
                    )

        if best_candidate is not None:
            refine_top_values: list[float] = []
            best_top_ratio = round(
                float(best_candidate["profile"].top_ratio),
                _OCR_RATIO_ROUND_DIGITS,
            )
            for delta in _OCR_AUTO_RECALIBRATE_REFINE_TOP_DELTAS:
                candidate_value = round(best_top_ratio + delta, _OCR_RATIO_ROUND_DIGITS)
                if candidate_value in top_values and candidate_value not in refine_top_values:
                    refine_top_values.append(candidate_value)
            for top_ratio in refine_top_values:
                for bottom_inset_ratio in bottom_values:
                    for left_inset_ratio, right_inset_ratio in horizontal_pairs:
                        _consider_candidate(
                            top_ratio,
                            bottom_inset_ratio,
                            left_inset_ratio,
                            right_inset_ratio,
                        )
        else:
            for top_ratio in top_values:
                for bottom_inset_ratio in bottom_values:
                    for left_inset_ratio, right_inset_ratio in horizontal_pairs:
                        _consider_candidate(
                            top_ratio,
                            bottom_inset_ratio,
                            left_inset_ratio,
                            right_inset_ratio,
                        )

        if best_candidate is None:
            if scan_exhausted:
                self._record_auto_recalibrate_diagnostic(
                    started_at=started_at,
                    attempts=ocr_attempts,
                    limited=True,
                    error="scan_budget_exhausted",
                )
                raise ValueError("自动重校准超时：请先停在稳定对白界面再重试")
            self._record_auto_recalibrate_diagnostic(
                started_at=started_at,
                attempts=ocr_attempts,
                limited=False,
                error="no_dialogue_candidate",
            )
            raise ValueError("自动重校准失败：请先停在稳定对白界面再重试")

        window_width = max(0, int(target.width or image_width))
        window_height = max(0, int(target.height or image_height))
        bucket_key = (
            build_ocr_capture_profile_bucket_key(window_width, window_height).lower()
            if window_width > 0 and window_height > 0
            else ""
        )
        capture_profile = best_candidate["profile"].to_dict()
        sample_text = str(best_candidate["sample_text"] or "")
        self._record_auto_recalibrate_diagnostic(
            started_at=started_at,
            attempts=ocr_attempts,
            limited=scan_exhausted,
            error="",
        )
        return {
            "process_name": process_name,
            "stage": OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
            "save_scope": "window_bucket",
            "bucket_key": bucket_key,
            "window_width": window_width,
            "window_height": window_height,
            "capture_profile": capture_profile,
            "sample_text": sample_text,
            "summary": (
                f"已自动重校准对白区：{process_name}"
                + (f" / {bucket_key}" if bucket_key else "")
                + f" / 示例文本：{sample_text[:_OCR_AUTO_RECALIBRATE_SUMMARY_SAMPLE_CHARS]}"
            ),
        }

    def _reset_default_ocr_state(self) -> None:
        self._default_ocr_state.reset()
        self._consecutive_no_text_polls = 0
        self._last_capture_error = ""
        self._last_raw_ocr_text = ""
        self._last_observed_line = {}
        self._last_stable_line = {}
        self._last_capture_image_hash = ""
        self._last_capture_source_size = {}
        self._last_capture_rect = {}
        self._last_capture_window_rect = {}
        self._last_capture_timing = {}
        self._consecutive_same_capture_frames = 0
        self._stale_capture_backend = False
        self._last_background_hash = ""
        self._last_background_hash_capture_at = 0.0
        self._pending_background_hash = ""
        self._pending_background_change_count = 0
        self._pending_visual_scene_hash = ""
        self._pending_visual_scene_at = 0.0
        self._pending_visual_scene_count = 0

    def _reset_aihong_menu_state(self) -> None:
        self._aihong_sm.reset()

    def _has_manual_capture_profile(self, target: DetectedGameWindow) -> bool:
        return _uses_manual_capture_profile(self._capture_profiles, target)

    def _should_use_aihong_two_stage(self, target: DetectedGameWindow) -> bool:
        return _matches_aihong_target(target)

    @staticmethod
    def _stabilize_text_key(
        text: str,
        *,
        state: _StableOcrTextState,
        repeat_threshold: int = _OCR_STABLE_TEXT_DEFAULT_REPEAT_THRESHOLD,
    ) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return False
        if cleaned == state.last_raw_text:
            state.repeat_count += 1
        else:
            state.repeat_count = 1
            state.last_raw_text = cleaned
        if state.repeat_count < max(_OCR_STABLE_TEXT_MIN_REPEAT_THRESHOLD, int(repeat_threshold)):
            return False
        if cleaned == state.stable_text:
            return False
        state.stable_text = cleaned
        return True

    @staticmethod
    def _dialogue_candidate_from_ocr_text(raw_text: str) -> str:
        cleaned_text = _clean_ocr_dialogue_text(raw_text)
        if (
            _looks_like_noise_ocr_text(cleaned_text)
            or _looks_like_game_overlay_text(cleaned_text)
            or not _looks_like_ocr_dialogue_text(cleaned_text)
        ):
            return ""
        return cleaned_text

    def _emit_line_from_ocr_text(
        self,
        raw_text: str,
        *,
        now: float,
        state: _StableOcrTextState | None = None,
        emit_observed: bool = True,
        repeat_threshold: int | None = None,
    ) -> bool:
        cleaned_text = self._dialogue_candidate_from_ocr_text(raw_text)
        if not cleaned_text:
            return False
        self._commit_pending_visual_scene(now=now)
        self._last_raw_ocr_text = str(raw_text or "")
        if emit_observed and self._writer.emit_line_observed(cleaned_text, ts=utc_now_iso(now)):
            self._last_observed_line = self._line_payload_from_writer(stability="tentative")
        tracker = state or self._default_ocr_state
        if not self._stabilize_text_key(
            cleaned_text,
            state=tracker,
            repeat_threshold=(
                self._line_changed_repeat_threshold()
                if repeat_threshold is None
                else repeat_threshold
            ),
        ):
            return False
        emitted = self._writer.emit_line(cleaned_text, ts=utc_now_iso(now))
        if emitted:
            stable_line = self._line_payload_from_writer(stability="stable")
            self._last_stable_line = stable_line
            self._last_observed_line = stable_line
        return emitted

    def _emit_choices_from_candidates(
        self,
        choices: list[str],
        *,
        now: float,
        state: _StableOcrTextState | None = None,
        choice_bounds: list[dict[str, float] | None] | None = None,
        choice_bounds_metadata: dict[str, Any] | None = None,
    ) -> bool:
        tracker = state or self._default_ocr_state
        if not self._stabilize_text_key(
            _canonical_choice_candidate_text(choices),
            state=tracker,
            repeat_threshold=_OCR_CHOICES_REPEAT_THRESHOLD,
        ):
            return False
        self._commit_pending_visual_scene(now=now)
        return self._writer.emit_choices(
            choices,
            ts=utc_now_iso(now),
            choice_bounds=choice_bounds,
            choice_bounds_metadata=choice_bounds_metadata,
        )

    @staticmethod
    def _should_attempt_followup_confirm(
        raw_text: str,
        *,
        state: _StableOcrTextState,
    ) -> bool:
        cleaned = normalize_text(raw_text).strip()
        if not cleaned:
            return False
        return (
            bool(state.stable_text)
            and
            state.repeat_count == _OCR_FOLLOWUP_CONFIRM_REPEAT_COUNT
            and state.last_raw_text == cleaned
            and state.stable_text != cleaned
        )

    async def _capture_followup_text(
        self,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        backend_plan: SelectedOcrBackendPlan,
    ) -> OcrExtractionResult:
        await asyncio.sleep(_OCR_FOLLOWUP_CONFIRM_DELAY_SECONDS)
        return await asyncio.to_thread(
            self._capture_and_extract_text,
            target,
            profile,
            backend_plan,
        )

    async def _attempt_followup_confirm_for_tick(
        self,
        *,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        backend_plan: SelectedOcrBackendPlan,
        result: OcrReaderTickResult,
        source_text: str,
        state: _StableOcrTextState,
        now: float,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
        allow_choices: bool,
        emit_observed: bool,
        line_repeat_threshold: int | None,
    ) -> _TickFollowupConfirmResult:
        output = _TickFollowupConfirmResult(
            now=now,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        if not self._should_attempt_followup_confirm(source_text, state=state):
            return output

        followup_extraction = await self._capture_followup_text(
            target,
            profile,
            backend_plan,
        )
        bookkeeping = self._record_extraction_for_tick(
            extraction=followup_extraction,
            result=result,
            now=self._time_fn(),
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        output.active_backend = bookkeeping.active_backend
        output.backend_detail_override = bookkeeping.backend_detail_override
        if self._handle_self_ui_guard_for_tick(text=followup_extraction.text, result=result):
            output.guard_blocked = True
            return output

        followup_now = self._time_fn()
        output.emitted = bool(
            self._consume_ocr_text(
                followup_extraction.text,
                now=followup_now,
                state=state,
                allow_choices=allow_choices,
                emit_observed=emit_observed,
                line_repeat_threshold=line_repeat_threshold,
            )
        )
        if output.emitted:
            output.now = followup_now
        return output

    def _consume_aihong_menu_stage_text(
        self,
        raw_text: str,
        *,
        now: float,
        boxes: list[OcrTextBox] | None = None,
        choice_bounds_metadata: dict[str, Any] | None = None,
    ) -> _MenuConsumeResult:
        lines = _stripped_ocr_lines(raw_text)
        choices = _coerce_aihong_menu_choices(lines)
        if choices:
            return _MenuConsumeResult(
                emitted_kind="choices"
                if self._emit_choices_from_candidates(
                    choices,
                    now=now,
                    state=self._aihong_sm.menu_ocr_state,
                    choice_bounds=_aihong_choice_boxes(choices, list(boxes or [])),
                    choice_bounds_metadata=choice_bounds_metadata,
                )
                else "",
                has_menu_candidate=True,
            )
        if _looks_like_aihong_menu_status_only_text(raw_text):
            return _MenuConsumeResult(emitted_kind="", has_menu_candidate=True)
        # Menu-stage capture intentionally scans a much larger region so option
        # OCR can find buttons anywhere on screen. Do not turn that full-screen
        # text into a dialogue line; switch back to dialogue-stage capture and
        # let the narrower profile read the next line.
        return _MenuConsumeResult(emitted_kind="", has_menu_candidate=False)

    def _consume_aihong_active_menu_stage_for_tick(
        self,
        *,
        extraction: OcrExtractionResult,
        now: float,
    ) -> bool:
        menu_result = self._consume_aihong_menu_stage_text(
            extraction.text,
            now=now,
            boxes=extraction.boxes,
            choice_bounds_metadata=_extraction_choice_bounds_metadata(extraction),
        )
        self._aihong_sm.on_active_menu_consumed(
            emitted_kind=menu_result.emitted_kind,
            has_menu_candidate=menu_result.has_menu_candidate,
            text=extraction.text,
        )
        return bool(menu_result.emitted_kind)

    async def _probe_aihong_menu_stage_for_tick(
        self,
        *,
        target: DetectedGameWindow,
        backend_plan: SelectedOcrBackendPlan,
        result: OcrReaderTickResult,
        now: float,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
        after_advance_trigger_mode: bool,
    ) -> _TickAihongMenuProbeResult:
        output = _TickAihongMenuProbeResult(
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        menu_profile_selection = self._capture_profile_selection_for_target(
            target,
            stage=_AIHONG_MENU_STAGE,
        )
        menu_profile = menu_profile_selection.profile
        menu_extraction = await asyncio.to_thread(
            self._capture_and_extract_text,
            target,
            menu_profile,
            backend_plan,
            True,
            not after_advance_trigger_mode,
        )
        bookkeeping = self._record_extraction_for_tick(
            extraction=menu_extraction,
            result=result,
            now=self._time_fn(),
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        output.active_backend = bookkeeping.active_backend
        output.backend_detail_override = bookkeeping.backend_detail_override
        if self._handle_self_ui_guard_for_tick(text=menu_extraction.text, result=result):
            output.guard_blocked = True
            return output

        menu_result = self._consume_aihong_menu_stage_text(
            menu_extraction.text,
            now=now,
            boxes=menu_extraction.boxes,
            choice_bounds_metadata=_extraction_choice_bounds_metadata(menu_extraction),
        )
        if menu_result.has_menu_candidate:
            output.runtime_profile = menu_profile
            output.runtime_capture_profile_selection = menu_profile_selection
        self._aihong_sm.on_menu_probe_result(
            emitted_kind=menu_result.emitted_kind,
            has_menu_candidate=menu_result.has_menu_candidate,
        )
        if menu_result.emitted_kind:
            output.emitted = True
        if menu_result.emitted_kind in ("line", "choices"):
            output.runtime_profile = menu_profile
            output.runtime_capture_profile_selection = menu_profile_selection
        return output

    async def _consume_aihong_dialogue_stage_for_tick(
        self,
        *,
        extraction: OcrExtractionResult,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        backend_plan: SelectedOcrBackendPlan,
        result: OcrReaderTickResult,
        now: float,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
        after_advance_trigger_mode: bool,
        emit_observed_lines: bool,
        line_repeat_threshold: int | None,
    ) -> _TickAihongDialogueStageResult:
        output = _TickAihongDialogueStageResult(
            now=now,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        dialogue_menu_choices = _coerce_aihong_menu_choices(
            _stripped_ocr_lines(extraction.text)
        )
        dialogue_text_is_menu_status = _looks_like_aihong_menu_status_only_text(extraction.text)
        dialogue_emitted = False
        if dialogue_menu_choices:
            dialogue_emitted = bool(
                self._emit_choices_from_candidates(
                    dialogue_menu_choices,
                    now=now,
                    state=self._aihong_sm.menu_ocr_state,
                    choice_bounds=_aihong_choice_boxes(dialogue_menu_choices, extraction.boxes),
                    choice_bounds_metadata=_extraction_choice_bounds_metadata(extraction),
                )
            )
        elif not dialogue_text_is_menu_status:
            dialogue_emitted = bool(
                self._consume_ocr_text(
                    extraction.text,
                    now=now,
                    state=self._default_ocr_state,
                    allow_choices=False,
                    emit_observed=emit_observed_lines,
                    line_repeat_threshold=line_repeat_threshold,
                )
            )

        if (
            not dialogue_emitted
            and not dialogue_text_is_menu_status
            and not dialogue_menu_choices
        ):
            followup_result = await self._attempt_followup_confirm_for_tick(
                target=target,
                profile=profile,
                backend_plan=backend_plan,
                result=result,
                source_text=extraction.text,
                state=self._default_ocr_state,
                now=now,
                active_backend=output.active_backend,
                backend_detail_override=output.backend_detail_override,
                allow_choices=False,
                emit_observed=emit_observed_lines,
                line_repeat_threshold=line_repeat_threshold,
            )
            output.active_backend = followup_result.active_backend
            output.backend_detail_override = followup_result.backend_detail_override
            output.guard_blocked = output.guard_blocked or followup_result.guard_blocked
            dialogue_emitted = bool(followup_result.emitted)
            output.now = followup_result.now

        output.emitted = dialogue_emitted
        self._aihong_sm.on_dialogue_consumed(
            emitted=dialogue_emitted,
            is_menu_choices=bool(dialogue_menu_choices),
            is_menu_status=dialogue_text_is_menu_status,
        )
        if dialogue_emitted:
            return output

        should_probe_menu = self._aihong_sm.should_probe_menu(
            after_advance_trigger_mode=after_advance_trigger_mode,
            looks_like_menu=dialogue_text_is_menu_status or bool(dialogue_menu_choices),
        )
        if not should_probe_menu:
            return output

        menu_probe = await self._probe_aihong_menu_stage_for_tick(
            target=target,
            backend_plan=backend_plan,
            result=result,
            now=output.now,
            active_backend=output.active_backend,
            backend_detail_override=output.backend_detail_override,
            after_advance_trigger_mode=after_advance_trigger_mode,
        )
        output.active_backend = menu_probe.active_backend
        output.backend_detail_override = menu_probe.backend_detail_override
        output.guard_blocked = output.guard_blocked or menu_probe.guard_blocked
        output.emitted = output.emitted or menu_probe.emitted
        output.runtime_profile = menu_probe.runtime_profile
        output.runtime_capture_profile_selection = menu_probe.runtime_capture_profile_selection
        return output

    async def _consume_default_dialogue_stage_for_tick(
        self,
        *,
        extraction: OcrExtractionResult,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        backend_plan: SelectedOcrBackendPlan,
        result: OcrReaderTickResult,
        now: float,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
        after_advance_trigger_mode: bool,
        emit_observed_lines: bool,
        line_repeat_threshold: int | None,
    ) -> _TickDefaultDialogueStageResult:
        output = _TickDefaultDialogueStageResult(
            now=now,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )
        output.emitted = bool(
            self._consume_ocr_text(
                extraction.text,
                now=now,
                emit_observed=emit_observed_lines,
                line_repeat_threshold=line_repeat_threshold,
            )
        )
        if after_advance_trigger_mode or output.emitted:
            return output

        followup_result = await self._attempt_followup_confirm_for_tick(
            target=target,
            profile=profile,
            backend_plan=backend_plan,
            result=result,
            source_text=extraction.text,
            state=self._default_ocr_state,
            now=now,
            active_backend=output.active_backend,
            backend_detail_override=output.backend_detail_override,
            allow_choices=True,
            emit_observed=emit_observed_lines,
            line_repeat_threshold=line_repeat_threshold,
        )
        output.active_backend = followup_result.active_backend
        output.backend_detail_override = followup_result.backend_detail_override
        output.guard_blocked = output.guard_blocked or followup_result.guard_blocked
        output.emitted = bool(followup_result.emitted)
        output.now = followup_result.now
        return output

    def _matches_attached_window(self, candidate: DetectedGameWindow) -> bool:
        if self._attached_window is None:
            return False
        if candidate.hwnd and self._attached_window.hwnd and candidate.hwnd == self._attached_window.hwnd:
            return True
        return bool(candidate.pid and self._attached_window.pid and candidate.pid == self._attached_window.pid)

    def _prepare_window_inventory(
        self,
        windows: list[DetectedGameWindow],
    ) -> tuple[list[DetectedGameWindow], list[DetectedGameWindow]]:
        foreground_hwnd = _foreground_window_handle()
        prepared: list[DetectedGameWindow] = []
        for window in windows:
            candidate = replace(window)
            candidate.process_name = str(candidate.process_name or "").strip()
            candidate.title = str(candidate.title or "")
            candidate.class_name = str(candidate.class_name or "")
            candidate.exe_path = str(candidate.exe_path or "")
            candidate.pid = max(0, int(candidate.pid or 0))
            candidate.hwnd = max(0, int(candidate.hwnd or 0))
            candidate.area = max(0, int(candidate.area or 0))
            foreground_match, _ = _foreground_matches_target(foreground_hwnd, candidate)
            candidate.is_foreground = foreground_match
            candidate.score = float(max(candidate.area, 1))
            candidate = _classify_window_candidate(candidate)
            prepared.append(candidate)
        prepared.sort(key=_window_sort_key, reverse=True)
        eligible_windows = [candidate for candidate in prepared if candidate.eligible]
        excluded_windows = [candidate for candidate in prepared if not candidate.eligible]
        self._last_detected_windows = list(prepared)
        self._last_eligible_windows = list(eligible_windows)
        self._last_excluded_windows = list(excluded_windows)
        return eligible_windows, excluded_windows

    def _scan_window_inventory(self) -> tuple[list[DetectedGameWindow], list[DetectedGameWindow]]:
        if not self._platform_fn():
            self._last_detected_windows = []
            self._last_eligible_windows = []
            self._last_excluded_windows = []
            return [], []
        scanned = list(self._window_scanner() or [])
        return self._prepare_window_inventory(scanned)

    async def shutdown(self) -> None:
        self._wheel_monitor.stop()
        if self._writer.session_id:
            self._writer.end_session(ts=utc_now_iso(self._time_fn()))
        self._attached_window = None

    def _begin_tick_poll(self, *, poll_started_at: float) -> None:
        self._runtime.update_advance_state(
            last_tick_skipped=False,
            last_tick_skip_reason="",
        )
        self._runtime.update_timing_state(last_poll_started_at=utc_now_iso(poll_started_at))

    def _capture_mode_for_tick(self) -> _TickCaptureMode:
        after_advance_trigger_mode = (
            str(self._config.ocr_reader_trigger_mode or "").strip().lower()
            == OCR_TRIGGER_MODE_AFTER_ADVANCE
        )
        return _TickCaptureMode(
            after_advance_trigger_mode=after_advance_trigger_mode,
            emit_observed_lines=self._should_emit_observed_lines_for_capture(
                after_advance_trigger_mode=after_advance_trigger_mode
            ),
            line_repeat_threshold=(
                _AFTER_ADVANCE_LINE_REPEAT_THRESHOLD if after_advance_trigger_mode else None
            ),
            background_confirm_polls=(
                _AFTER_ADVANCE_BACKGROUND_CONFIRM_POLLS
                if after_advance_trigger_mode
                else _BACKGROUND_SCENE_CHANGE_CONFIRM_POLLS
            ),
        )

    async def _resolve_backend_plan_for_tick(self) -> _TickBackendPlanResult:
        started_at = self._time_fn()
        plan = await asyncio.to_thread(self._resolve_backend_plan)
        return _TickBackendPlanResult(
            plan=plan,
            duration_seconds=max(0.0, self._time_fn() - started_at),
        )

    async def _select_target_for_tick(
        self,
        *,
        memory_reader_runtime: dict[str, Any],
    ) -> _TickWindowSelectionResult:
        started_at = self._time_fn()
        scanned_windows = await asyncio.to_thread(self._window_scanner)
        duration_seconds = max(0.0, self._time_fn() - started_at)
        eligible_windows, excluded_windows = self._prepare_window_inventory(scanned_windows)
        selection = self._select_target_window(
            eligible_windows,
            excluded_windows=excluded_windows,
            memory_reader_runtime=memory_reader_runtime,
        )
        self._last_selection = selection
        return _TickWindowSelectionResult(
            selection=selection,
            duration_seconds=duration_seconds,
        )

    def _prepare_attached_target_for_tick(
        self,
        *,
        target: DetectedGameWindow,
        selection: WindowSelectionResult,
        backend_plan: SelectedOcrBackendPlan,
        result: OcrReaderTickResult,
        now: float,
        aihong_two_stage_enabled: bool,
    ) -> float:
        if self._attached_window is None or self._attached_window.pid != target.pid:
            if (
                not self._writer.session_id
                or self._writer.game_id != _ocr_game_id_from_process(target.process_name or target.title)
            ):
                self._writer.start_session(target)
                now = max(now, self._time_fn())
                result.should_rescan = True
            self._attached_window = target
            self._last_heartbeat_at = now
            self._reset_default_ocr_state()
            self._reset_aihong_menu_state()
            startup_profile_stage = (
                self._aihong_sm.capture_stage
                if aihong_two_stage_enabled
                else OCR_CAPTURE_PROFILE_STAGE_DEFAULT
            )
            startup_profile_selection = self._capture_profile_selection_for_target(
                target,
                stage=(
                    self._aihong_sm.capture_stage
                    if aihong_two_stage_enabled
                    else _AIHONG_DIALOGUE_STAGE
                ),
            )
            self._runtime = self._build_runtime(
                status="starting",
                detail="starting_capture",
                plan=backend_plan,
                target=target,
                capture_stage=startup_profile_stage,
                capture_profile=startup_profile_selection.profile.to_dict(),
                capture_profile_selection=startup_profile_selection,
                selection=selection,
                game_id=self._writer.game_id,
                session_id=self._writer.session_id,
                last_seq=self._writer.last_seq,
                last_event_ts=self._writer.last_event_ts,
            )

        if self._attached_window is not None:
            self._attached_window = target
        self._remember_locked_target(target)
        return now

    def _record_extraction_for_tick(
        self,
        *,
        extraction: OcrExtractionResult,
        result: OcrReaderTickResult,
        now: float,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
    ) -> _TickExtractionBookkeepingResult:
        self._last_capture_timing.update(extraction.timing)
        self._record_capture_completed(
            now=now,
            raw_text=extraction.text,
            image_hash=extraction.capture_image_hash,
        )
        self._record_capture_geometry(extraction)
        self._capture_backend_kind = extraction.capture_backend_kind
        self._capture_backend_detail = extraction.capture_backend_detail
        active_backend = extraction.backend if extraction.backend.kind else active_backend
        backend_detail_override = extraction.backend_detail or backend_detail_override
        result.warnings.extend(extraction.warnings)
        return _TickExtractionBookkeepingResult(
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
        )

    def _handle_self_ui_guard_for_tick(
        self,
        *,
        text: str,
        result: OcrReaderTickResult,
    ) -> bool:
        if not text or not _looks_like_self_ui_text(text):
            return False
        result.warnings.append("ocr_reader ignored text that looks like the N.E.K.O plugin UI")
        self._default_ocr_state.reset()
        self._aihong_sm.menu_ocr_state.reset()
        return True

    def _emit_heartbeat_if_due_for_tick(
        self,
        *,
        now: float,
        result: OcrReaderTickResult,
    ) -> bool:
        if not self._writer.session_id:
            return False
        if now - self._last_heartbeat_at < float(self._config.ocr_reader_poll_interval_seconds):
            return False
        if self._writer.emit_heartbeat(ts=utc_now_iso(now)):
            result.should_rescan = True
            self._last_heartbeat_at = now
        return True

    def _resolve_post_capture_status_for_tick(
        self,
        *,
        result: OcrReaderTickResult,
        now: float,
        status: str,
        detail: str,
        emitted: bool,
        guard_blocked: bool,
        capture_error: bool,
        capture_completed: bool,
        capture_attempted: bool,
        after_advance_trigger_mode: bool,
        event_seq_before_capture: int,
    ) -> _TickPostCaptureStatus:
        pending_scene_committed = False
        if (
            after_advance_trigger_mode
            and not emitted
            and self._pending_visual_scene_hash
            and now - float(self._pending_visual_scene_at or now) >= _PENDING_VISUAL_SCENE_MAX_AGE_SECONDS
        ):
            if self._commit_pending_visual_scene(now=now):
                pending_scene_committed = True
                result.should_rescan = True
        observed_or_stable_emitted = (
            int(self._writer.last_seq or 0) > event_seq_before_capture
            and not pending_scene_committed
        )

        if emitted:
            result.stable_event_emitted = True
            result.should_rescan = True
            self._mark_observed_progress(now=now)
            self._last_heartbeat_at = now
            status = "active"
            detail = "receiving_text"
        elif observed_or_stable_emitted:
            result.should_rescan = True
            self._mark_observed_progress(now=now)
            self._last_heartbeat_at = now
            if status == "starting":
                status = "active"
            detail = "receiving_observed_text"
        elif guard_blocked:
            if status == "starting":
                status = "active"
            detail = "self_ui_guard_blocked"
        elif capture_error:
            if status == "starting":
                status = "active"
            detail = "capture_failed"
        elif capture_completed:
            self._mark_no_text_poll()
            self._emit_heartbeat_if_due_for_tick(now=now, result=result)
            if status == "starting":
                status = "active"
            detail = (
                "ocr_capture_diagnostic_required"
                if self._ocr_capture_diagnostic_required()
                else "attached_no_text_yet"
            )
        elif capture_attempted:
            if status == "starting":
                status = "active"
            detail = "capture_failed"
        elif self._emit_heartbeat_if_due_for_tick(now=now, result=result):
            if status == "starting":
                status = "active"
            if detail == "starting_capture":
                detail = "attached_no_text_yet"

        return _TickPostCaptureStatus(
            status=status,
            detail=detail,
            observed_or_stable_emitted=observed_or_stable_emitted,
        )

    def _finalize_tick_runtime(
        self,
        *,
        result: OcrReaderTickResult,
        poll_started_at: float,
        status: str,
        detail: str,
        backend_plan: SelectedOcrBackendPlan,
        active_backend: OcrBackendDescriptor,
        backend_detail_override: str,
        target: DetectedGameWindow,
        selection: WindowSelectionResult,
        aihong_two_stage_enabled: bool,
        runtime_profile: OcrCaptureProfile,
        runtime_capture_profile_selection: ResolvedOcrCaptureSelection,
        emitted: bool,
        observed_or_stable_emitted: bool,
    ) -> OcrReaderTickResult:
        self._runtime = self._build_runtime(
            status=status,
            detail=detail,
            plan=backend_plan,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
            target=target,
            capture_stage=(
                self._aihong_sm.capture_stage if aihong_two_stage_enabled else OCR_CAPTURE_PROFILE_STAGE_DEFAULT
            ),
            capture_profile=runtime_profile.to_dict(),
            capture_profile_selection=runtime_capture_profile_selection,
            selection=selection,
            game_id=self._writer.game_id,
            session_id=self._writer.session_id,
            last_seq=self._writer.last_seq,
            last_event_ts=self._writer.last_event_ts,
        )
        poll_completed_at = self._time_fn()
        self._runtime.update_timing_state(
            last_poll_started_at=utc_now_iso(poll_started_at),
            last_poll_completed_at=utc_now_iso(poll_completed_at),
            last_poll_duration_seconds=max(0.0, poll_completed_at - poll_started_at),
        )
        self._runtime.update_advance_state(
            last_poll_emitted_event=bool(emitted or observed_or_stable_emitted)
        )
        result.runtime = self._runtime.to_dict()
        return result

    async def tick(
        self,
        *,
        bridge_sdk_available: bool,
        memory_reader_runtime: dict[str, Any],
    ) -> OcrReaderTickResult:
        if not self._tick_lock.acquire(blocking=False):
            self._runtime.update_advance_state(
                last_tick_skipped=True,
                last_tick_skip_reason="previous_tick_running",
            )
            result = OcrReaderTickResult(runtime=self._runtime.to_dict())
            result.warnings.append("ocr_reader tick skipped because previous tick is still running")
            return result
        try:
            return await self._tick_unlocked(
                bridge_sdk_available=bridge_sdk_available,
                memory_reader_runtime=memory_reader_runtime,
            )
        finally:
            self._tick_lock.release()

    async def _tick_unlocked(
        self,
        *,
        bridge_sdk_available: bool,
        memory_reader_runtime: dict[str, Any],
    ) -> OcrReaderTickResult:
        now = self._time_fn()
        poll_started_at = now
        backend_plan_duration = 0.0
        window_scan_duration = 0.0
        result = OcrReaderTickResult(runtime=self._runtime.to_dict())
        self._begin_tick_poll(poll_started_at=poll_started_at)

        if not self._config.ocr_reader_enabled:
            self._runtime = OcrReaderRuntime(enabled=False, status="disabled", detail="disabled_by_config")
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        if not self._platform_fn():
            self._runtime = self._build_runtime(
                status="idle",
                detail="unsupported_platform",
                plan=SelectedOcrBackendPlan(),
            )
            await self._end_session_if_needed(now)
            result.warnings.append("ocr_reader is Windows-only")
            result.runtime = self._runtime.to_dict()
            return result

        backend_plan_result = await self._resolve_backend_plan_for_tick()
        backend_plan = backend_plan_result.plan
        backend_plan_duration = backend_plan_result.duration_seconds
        if not backend_plan.primary.available:
            self._runtime = self._build_runtime(
                status="idle",
                detail=self._backend_unavailable_detail(backend_plan),
                plan=backend_plan,
            )
            await self._end_session_if_needed(now)
            result.warnings.extend(self._backend_unavailable_warnings(backend_plan))
            result.runtime = self._runtime.to_dict()
            return result

        if bridge_sdk_available:
            self._runtime = self._build_runtime(
                status="idle",
                detail="bridge_sdk_available",
                plan=backend_plan,
            )
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        memory_reader_has_text = str(memory_reader_runtime.get("detail") or "") == "receiving_text"
        if memory_reader_has_text:
            self._last_memory_reader_text_at = now
            self._runtime = self._build_runtime(
                status="idle",
                detail="memory_reader_active",
                plan=backend_plan,
            )
            result.runtime = self._runtime.to_dict()
            return result

        if self._last_memory_reader_text_at > 0:
            elapsed = now - self._last_memory_reader_text_at
            threshold = float(self._config.ocr_reader_no_text_takeover_after_seconds)
            if elapsed < threshold:
                self._runtime = self._build_runtime(
                    status="idle",
                    detail="waiting_for_takeover_window",
                    plan=backend_plan,
                )
                result.runtime = self._runtime.to_dict()
                return result

        if not self._capture_backend.is_available():
            self._runtime = self._build_runtime(
                status="candidate",
                detail="capture_backend_unavailable",
                plan=backend_plan,
                takeover_reason="capture_backend_not_available",
            )
            await self._end_session_if_needed(now)
            result.warnings.append("ocr_reader capture backend is not available")
            result.runtime = self._runtime.to_dict()
            return result

        window_selection_result = await self._select_target_for_tick(
            memory_reader_runtime=memory_reader_runtime,
        )
        selection = window_selection_result.selection
        window_scan_duration = window_selection_result.duration_seconds
        target = selection.target
        if target is None:
            self._runtime = self._build_runtime(
                status="idle",
                detail="waiting_for_valid_window",
                plan=backend_plan,
                selection=selection,
            )
            await self._end_session_if_needed(now)
            result.runtime = self._runtime.to_dict()
            return result

        aihong_two_stage_enabled = self._should_use_aihong_two_stage(target)
        if not aihong_two_stage_enabled:
            self._reset_aihong_menu_state()
        profile_stage = self._aihong_sm.capture_stage if aihong_two_stage_enabled else _AIHONG_DIALOGUE_STAGE
        capture_profile_selection = self._capture_profile_selection_for_target(
            target,
            stage=profile_stage,
        )
        profile = capture_profile_selection.profile

        now = self._prepare_attached_target_for_tick(
            target=target,
            selection=selection,
            backend_plan=backend_plan,
            result=result,
            now=now,
            aihong_two_stage_enabled=aihong_two_stage_enabled,
        )

        emitted = False
        guard_blocked = False
        active_backend = backend_plan.primary
        backend_detail_override = ""
        runtime_profile = profile
        runtime_capture_profile_selection = capture_profile_selection
        event_seq_before_capture = int(self._writer.last_seq or 0)
        capture_mode = self._capture_mode_for_tick()
        self._last_capture_timing = {
            "backend_plan_duration_seconds": backend_plan_duration,
            "window_scan_duration_seconds": window_scan_duration,
        }
        capture_attempted = False
        capture_completed = False
        capture_error = False
        try:
            capture_attempted = True
            self._record_capture_attempt(now=now)
            extraction = await asyncio.to_thread(
                self._capture_and_extract_text,
                target,
                profile,
                backend_plan,
                True,
                not capture_mode.after_advance_trigger_mode,
            )
            capture_completed = True
            bookkeeping = self._record_extraction_for_tick(
                extraction=extraction,
                result=result,
                now=now,
                active_backend=active_backend,
                backend_detail_override=backend_detail_override,
            )
            if self._observe_background_hash(
                extraction.background_hash,
                now=now,
                confirm_polls=capture_mode.background_confirm_polls,
                defer_scene_emit=capture_mode.after_advance_trigger_mode,
            ):
                result.should_rescan = True
            active_backend = bookkeeping.active_backend
            backend_detail_override = bookkeeping.backend_detail_override
            if self._handle_self_ui_guard_for_tick(text=extraction.text, result=result):
                guard_blocked = True
            else:
                if aihong_two_stage_enabled:
                    if self._aihong_sm.is_menu:
                        emitted = self._consume_aihong_active_menu_stage_for_tick(
                            extraction=extraction,
                            now=now,
                        )
                    else:
                        dialogue_result = await self._consume_aihong_dialogue_stage_for_tick(
                            extraction=extraction,
                            target=target,
                            profile=profile,
                            backend_plan=backend_plan,
                            result=result,
                            now=now,
                            active_backend=active_backend,
                            backend_detail_override=backend_detail_override,
                            after_advance_trigger_mode=capture_mode.after_advance_trigger_mode,
                            emit_observed_lines=capture_mode.emit_observed_lines,
                            line_repeat_threshold=capture_mode.line_repeat_threshold,
                        )
                        emitted = bool(dialogue_result.emitted)
                        guard_blocked = guard_blocked or dialogue_result.guard_blocked
                        now = dialogue_result.now
                        active_backend = dialogue_result.active_backend
                        backend_detail_override = dialogue_result.backend_detail_override
                        if dialogue_result.runtime_profile is not None:
                            runtime_profile = dialogue_result.runtime_profile
                        if dialogue_result.runtime_capture_profile_selection is not None:
                            runtime_capture_profile_selection = (
                                dialogue_result.runtime_capture_profile_selection
                            )
                else:
                    default_dialogue_result = await self._consume_default_dialogue_stage_for_tick(
                        extraction=extraction,
                        target=target,
                        profile=profile,
                        backend_plan=backend_plan,
                        result=result,
                        now=now,
                        active_backend=active_backend,
                        backend_detail_override=backend_detail_override,
                        after_advance_trigger_mode=capture_mode.after_advance_trigger_mode,
                        emit_observed_lines=capture_mode.emit_observed_lines,
                        line_repeat_threshold=capture_mode.line_repeat_threshold,
                    )
                    emitted = bool(default_dialogue_result.emitted)
                    guard_blocked = guard_blocked or default_dialogue_result.guard_blocked
                    now = default_dialogue_result.now
                    active_backend = default_dialogue_result.active_backend
                    backend_detail_override = default_dialogue_result.backend_detail_override
        except Exception as exc:
            self._logger.warning("ocr_reader capture/OCR failed: %s", exc)
            capture_error = True
            self._record_capture_error(now=now, error=exc)
            result.warnings.append(f"ocr_reader capture failed: {exc}")

        post_capture_status = self._resolve_post_capture_status_for_tick(
            result=result,
            now=now,
            status=self._runtime.status,
            detail=self._runtime.detail,
            emitted=emitted,
            guard_blocked=guard_blocked,
            capture_error=capture_error,
            capture_completed=capture_completed,
            capture_attempted=capture_attempted,
            after_advance_trigger_mode=capture_mode.after_advance_trigger_mode,
            event_seq_before_capture=event_seq_before_capture,
        )

        return self._finalize_tick_runtime(
            result=result,
            poll_started_at=poll_started_at,
            status=post_capture_status.status,
            detail=post_capture_status.detail,
            backend_plan=backend_plan,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
            target=target,
            selection=selection,
            aihong_two_stage_enabled=aihong_two_stage_enabled,
            runtime_profile=runtime_profile,
            runtime_capture_profile_selection=runtime_capture_profile_selection,
            emitted=emitted,
            observed_or_stable_emitted=post_capture_status.observed_or_stable_emitted,
        )

    def _configured_backend_selection(self) -> str:
        selection = str(self._config.ocr_reader_backend_selection or "auto").strip().lower()
        if selection in {"auto", "rapidocr", "tesseract"}:
            return selection
        return "auto"

    def _capture_profile_selection_for_target(
        self,
        target: DetectedGameWindow,
        *,
        stage: str = _AIHONG_DIALOGUE_STAGE,
    ) -> ResolvedOcrCaptureSelection:
        configured_profile = _lookup_capture_profile(
            self._capture_profiles,
            target,
            stage=stage,
        )
        if configured_profile is not None:
            return configured_profile

        builtin_profile = _builtin_capture_profile_for_target_stage(target, stage=stage)
        if builtin_profile is not None:
            return ResolvedOcrCaptureSelection(
                profile=builtin_profile,
                match_source=OCR_CAPTURE_PROFILE_MATCH_SOURCE_BUILTIN_PRESET,
            )

        return ResolvedOcrCaptureSelection(
            profile=OcrCaptureProfile(
                left_inset_ratio=self._config.ocr_reader_left_inset_ratio,
                right_inset_ratio=self._config.ocr_reader_right_inset_ratio,
                top_ratio=self._config.ocr_reader_top_ratio,
                bottom_inset_ratio=self._config.ocr_reader_bottom_inset_ratio,
            ),
            match_source=OCR_CAPTURE_PROFILE_MATCH_SOURCE_CONFIG_DEFAULT,
        )

    def _capture_profile_for_target(
        self,
        target: DetectedGameWindow,
        *,
        stage: str = _AIHONG_DIALOGUE_STAGE,
    ) -> OcrCaptureProfile:
        return self._capture_profile_selection_for_target(target, stage=stage).profile

    def _resolved_tesseract_path(self) -> str:
        return resolve_tesseract_path(
            self._config.ocr_reader_tesseract_path,
            install_target_dir_raw=self._config.ocr_reader_install_target_dir,
        )

    def _tesseract_descriptor(self, inspection: dict[str, Any]) -> OcrBackendDescriptor:
        installed = bool(inspection.get("installed"))
        detail = "selected_primary" if installed else self._tesseract_unavailable_detail(inspection)
        return OcrBackendDescriptor(
            kind="tesseract",
            backend=TesseractOcrBackend(
                tesseract_path=self._config.ocr_reader_tesseract_path,
                install_target_dir_raw=self._config.ocr_reader_install_target_dir,
                languages=self._config.ocr_reader_languages,
            ),
            path=str(inspection.get("detected_path") or self._resolved_tesseract_path()),
            model=self._config.ocr_reader_languages,
            detail=detail,
            available=installed,
        )

    def _rapidocr_descriptor(self, inspection: dict[str, Any], *, enabled: bool) -> OcrBackendDescriptor:
        detail = str(inspection.get("detail") or "missing")
        if not enabled:
            detail = "disabled_by_config"
        return OcrBackendDescriptor(
            kind="rapidocr",
            backend=self._rapidocr_backend_for_config(),
            path=str(inspection.get("detected_path") or ""),
            model=str(
                inspection.get("selected_model")
                or f"{self._config.rapidocr_ocr_version}/{self._config.rapidocr_lang_type}/{self._config.rapidocr_model_type}"
            ),
            detail="selected_primary" if enabled and bool(inspection.get("installed")) else detail,
            available=enabled and bool(inspection.get("installed")),
        )

    @staticmethod
    def _backend_plan_config_key(config: GalgameConfig) -> tuple[str, ...]:
        return (
            str(config.ocr_reader_backend_selection or ""),
            str(config.ocr_reader_tesseract_path or ""),
            str(config.ocr_reader_install_target_dir or ""),
            str(config.ocr_reader_languages or ""),
            str(bool(config.rapidocr_enabled)),
            str(config.rapidocr_install_target_dir or ""),
            str(config.rapidocr_engine_type or ""),
            str(config.rapidocr_lang_type or ""),
            str(config.rapidocr_model_type or ""),
            str(config.rapidocr_ocr_version or ""),
        )

    def _resolve_backend_plan(self) -> SelectedOcrBackendPlan:
        now = self._time_fn()
        cache_key = self._backend_plan_config_key(self._config)
        if (
            self._backend_plan_cache_key == cache_key
            and self._backend_plan_cache is not None
            and now - float(self._backend_plan_cache_at or 0.0) < _BACKEND_PLAN_CACHE_TTL_SECONDS
        ):
            return self._backend_plan_cache
        selection = self._configured_backend_selection()
        tesseract_inspection = inspect_tesseract_installation(
            configured_path=self._config.ocr_reader_tesseract_path,
            install_target_dir_raw=self._config.ocr_reader_install_target_dir,
            languages=self._config.ocr_reader_languages,
        )
        rapidocr_inspection = inspect_rapidocr_installation(
            install_target_dir_raw=self._config.rapidocr_install_target_dir,
            engine_type=self._config.rapidocr_engine_type,
            lang_type=self._config.rapidocr_lang_type,
            model_type=self._config.rapidocr_model_type,
            ocr_version=self._config.rapidocr_ocr_version,
        )
        tesseract = self._tesseract_descriptor(tesseract_inspection)
        rapidocr = self._rapidocr_descriptor(
            rapidocr_inspection,
            enabled=bool(self._config.rapidocr_enabled),
        )
        plan = SelectedOcrBackendPlan(
            selection=selection,
            rapidocr_inspection=rapidocr_inspection,
            tesseract_inspection=tesseract_inspection,
        )

        if selection == "rapidocr":
            plan.primary = rapidocr
            self._backend_plan_cache_key = cache_key
            self._backend_plan_cache_at = now
            self._backend_plan_cache = plan
            return plan
        if selection == "tesseract":
            plan.primary = tesseract
            self._backend_plan_cache_key = cache_key
            self._backend_plan_cache_at = now
            self._backend_plan_cache = plan
            return plan
        if rapidocr.available:
            rapidocr.detail = "selected_primary"
            plan.primary = rapidocr
            if tesseract.available:
                tesseract.detail = "compatibility_fallback"
                plan.fallback = tesseract
            self._backend_plan_cache_key = cache_key
            self._backend_plan_cache_at = now
            self._backend_plan_cache = plan
            return plan
        if tesseract.available:
            tesseract.detail = f"auto_fallback_from_rapidocr:{rapidocr.detail}"
            plan.primary = tesseract
            self._backend_plan_cache_key = cache_key
            self._backend_plan_cache_at = now
            self._backend_plan_cache = plan
            return plan
        plan.primary = rapidocr if bool(self._config.rapidocr_enabled) else tesseract
        if bool(self._config.rapidocr_enabled):
            plan.fallback = tesseract
        self._backend_plan_cache_key = cache_key
        self._backend_plan_cache_at = now
        self._backend_plan_cache = plan
        return plan

    @staticmethod
    def _tesseract_unavailable_detail(inspection: dict[str, Any]) -> str:
        if str(inspection.get("detail") or "") == "missing_languages":
            return "missing_languages"
        return "missing_tesseract"

    def _backend_unavailable_detail(self, plan: SelectedOcrBackendPlan) -> str:
        if plan.selection == "rapidocr":
            return plan.primary.detail or "missing"
        if plan.selection == "tesseract":
            return self._tesseract_unavailable_detail(plan.tesseract_inspection)
        if plan.primary.kind == "rapidocr":
            return plan.primary.detail or "missing"
        if str(plan.tesseract_inspection.get("detail") or "") == "missing_languages":
            return "missing_languages"
        return "missing_tesseract"

    def _backend_unavailable_warnings(self, plan: SelectedOcrBackendPlan) -> list[str]:
        warnings: list[str] = []
        if plan.selection == "rapidocr" or plan.primary.kind == "rapidocr":
            warnings.append(f"ocr_reader RapidOCR is unavailable: {plan.primary.detail or 'missing'}")
            if plan.selection == "rapidocr":
                return warnings
            tesseract_detail = str(plan.tesseract_inspection.get("detail") or "")
            if tesseract_detail == "missing_languages":
                missing = plan.tesseract_inspection.get("missing_languages", [])
                warnings.append(f"ocr_reader Tesseract fallback is missing languages: {missing}")
            elif tesseract_detail and tesseract_detail != "installed":
                warnings.append("ocr_reader Tesseract fallback is missing or not configured")
            return warnings
        if str(plan.tesseract_inspection.get("detail") or "") == "missing_languages":
            missing = plan.tesseract_inspection.get("missing_languages", [])
            warnings.append(f"ocr_reader Tesseract is missing languages: {missing}")
        else:
            warnings.append("ocr_reader Tesseract is missing or not configured")
        rapid_detail = str(plan.rapidocr_inspection.get("detail") or "")
        if rapid_detail and rapid_detail != "installed":
            warnings.append(f"ocr_reader RapidOCR status: {rapid_detail}")
        return warnings

    def _build_runtime(
        self,
        *,
        status: str,
        detail: str,
        plan: SelectedOcrBackendPlan,
        active_backend: OcrBackendDescriptor | None = None,
        backend_detail_override: str = "",
        target: DetectedGameWindow | None = None,
        capture_stage: str = "",
        capture_profile: dict[str, float] | None = None,
        capture_profile_selection: ResolvedOcrCaptureSelection | None = None,
        selection: WindowSelectionResult | None = None,
        takeover_reason: str = "",
        game_id: str = "",
        session_id: str = "",
        last_seq: int | None = None,
        last_event_ts: str = "",
    ) -> OcrReaderRuntime:
        backend = active_backend if active_backend and active_backend.kind else plan.primary
        attached_target = target or self._attached_window
        selection_state = selection or self._last_selection
        effective_target = selection_state.target or attached_target
        manual_target = (
            selection_state.manual_target.to_dict()
            if isinstance(selection_state.manual_target, OcrWindowTarget)
            else self._manual_target.to_dict()
        )
        resolved_last_seq = (
            int(last_seq)
            if last_seq is not None
            else int(self._writer.last_seq or self._runtime.last_seq)
        )
        capture_timing = dict(self._last_capture_timing)

        def _timing_float(key: str, fallback: float) -> float:
            if key in capture_timing:
                return float(capture_timing.get(key) or 0.0)
            return float(fallback or 0.0)

        return OcrReaderRuntime(
            enabled=True,
            status=status,
            detail=detail,
            process_name=str((attached_target.process_name if attached_target is not None else self._runtime.process_name) or ""),
            pid=int((attached_target.pid if attached_target is not None else self._runtime.pid) or 0),
            window_title=str((attached_target.title if attached_target is not None else self._runtime.window_title) or ""),
            width=int((attached_target.width if attached_target is not None else self._runtime.width) or 0),
            height=int((attached_target.height if attached_target is not None else self._runtime.height) or 0),
            aspect_ratio=float(
                (
                    attached_target.aspect_ratio
                    if attached_target is not None
                    else self._runtime.aspect_ratio
                )
                or 0.0
            ),
            game_id=str(game_id or self._writer.game_id or self._runtime.game_id),
            session_id=str(session_id or self._writer.session_id or self._runtime.session_id),
            last_seq=resolved_last_seq,
            last_event_ts=str(last_event_ts or self._writer.last_event_ts or self._runtime.last_event_ts),
            capture_stage=str(capture_stage or self._runtime.capture_stage),
            capture_profile=dict(capture_profile or self._runtime.capture_profile),
            capture_profile_match_source=str(
                (
                    capture_profile_selection.match_source
                    if capture_profile_selection is not None
                    else self._runtime.capture_profile_match_source
                )
                or ""
            ),
            capture_profile_bucket_key=str(
                (
                    capture_profile_selection.bucket_key
                    if capture_profile_selection is not None
                    else self._runtime.capture_profile_bucket_key
                )
                or ""
            ),
            tesseract_path=self._resolved_tesseract_path(),
            languages=self._config.ocr_reader_languages,
            takeover_reason=takeover_reason or self._runtime.takeover_reason,
            backend_kind=str(backend.kind or ""),
            backend_detail=str(backend_detail_override or backend.detail or ""),
            backend_path=str(backend.path or ""),
            backend_model=str(backend.model or ""),
            target_selection_mode=str(selection_state.selection_mode or self._manual_target.mode or "auto"),
            target_selection_detail=str(selection_state.selection_detail or self._runtime.target_selection_detail),
            effective_window_key=str(effective_target.window_key if effective_target is not None else ""),
            effective_window_title=str(effective_target.title if effective_target is not None else ""),
            effective_process_name=str(effective_target.process_name if effective_target is not None else ""),
            target_is_foreground=bool(effective_target.is_foreground) if effective_target is not None else False,
            manual_target=manual_target,
            locked_target=self._locked_target.to_dict() if self._has_locked_target() else {},
            candidate_count=max(0, int(selection_state.candidate_count or 0)),
            excluded_candidate_count=max(0, int(selection_state.excluded_candidate_count or 0)),
            last_exclude_reason=str(selection_state.last_exclude_reason or self._runtime.last_exclude_reason),
            consecutive_no_text_polls=max(0, int(self._consecutive_no_text_polls or 0)),
            last_observed_at=str(self._last_observed_at or self._runtime.last_observed_at),
            last_capture_profile=dict(capture_profile or self._runtime.capture_profile),
            last_capture_stage=str(capture_stage or self._runtime.capture_stage),
            ocr_capture_diagnostic_required=self._ocr_capture_diagnostic_required(),
            ocr_context_state=self._ocr_context_state_for_detail(status=status, detail=detail),
            last_capture_attempt_at=str(
                self._last_capture_attempt_at or self._runtime.last_capture_attempt_at
            ),
            last_capture_completed_at=str(
                self._last_capture_completed_at or self._runtime.last_capture_completed_at
            ),
            last_capture_error=str(self._last_capture_error or self._runtime.last_capture_error),
            last_raw_ocr_text=str(self._last_raw_ocr_text or self._runtime.last_raw_ocr_text),
            last_observed_line=dict(self._last_observed_line or self._runtime.last_observed_line),
            last_stable_line=dict(self._last_stable_line or self._runtime.last_stable_line),
            capture_backend_kind=str(
                self._capture_backend_kind
                or self._runtime.capture_backend_kind
                or getattr(self._capture_backend, "last_backend_kind", "")
                or getattr(self._capture_backend, "selection", "")
            ),
            capture_backend_detail=str(
                self._capture_backend_detail
                or self._runtime.capture_backend_detail
                or getattr(self._capture_backend, "last_backend_detail", "")
                or ""
            ),
            last_capture_image_hash=str(
                self._last_capture_image_hash or self._runtime.last_capture_image_hash
            ),
            last_capture_source_size=dict(
                self._last_capture_source_size or self._runtime.last_capture_source_size
            ),
            last_capture_rect=dict(
                self._last_capture_rect or self._runtime.last_capture_rect
            ),
            last_capture_window_rect=dict(
                self._last_capture_window_rect or self._runtime.last_capture_window_rect
            ),
            consecutive_same_capture_frames=max(
                0,
                int(
                    self._consecutive_same_capture_frames
                    or self._runtime.consecutive_same_capture_frames
                    or 0
                ),
            ),
            stale_capture_backend=bool(
                self._stale_capture_backend or self._runtime.stale_capture_backend
            ),
            foreground_advance_monitor_running=self._wheel_monitor.is_running(),
            foreground_advance_last_seq=max(
                int(self._wheel_monitor.last_seq() or 0),
                int(self._runtime.foreground_advance_last_seq or 0),
            ),
            foreground_advance_consumed_seq=int(
                self._runtime.foreground_advance_consumed_seq or self._last_consumed_wheel_seq
            ),
            foreground_advance_last_kind=str(self._runtime.foreground_advance_last_kind or ""),
            foreground_advance_last_delta=int(self._runtime.foreground_advance_last_delta or 0),
            foreground_advance_last_matched=bool(self._runtime.foreground_advance_last_matched),
            foreground_advance_last_match_reason=str(
                self._runtime.foreground_advance_last_match_reason or ""
            ),
            last_capture_total_duration_seconds=float(
                _timing_float(
                    "total_duration_seconds",
                    self._runtime.last_capture_total_duration_seconds,
                )
            ),
            last_capture_frame_duration_seconds=float(
                _timing_float(
                    "capture_frame_duration_seconds",
                    self._runtime.last_capture_frame_duration_seconds,
                )
            ),
            last_capture_background_duration_seconds=float(
                _timing_float(
                    "background_hash_duration_seconds",
                    self._runtime.last_capture_background_duration_seconds,
                )
            ),
            last_capture_image_hash_duration_seconds=float(
                _timing_float(
                    "capture_image_hash_duration_seconds",
                    self._runtime.last_capture_image_hash_duration_seconds,
                )
            ),
            last_ocr_extract_duration_seconds=float(
                _timing_float(
                    "ocr_extract_duration_seconds",
                    self._runtime.last_ocr_extract_duration_seconds,
                )
            ),
            last_backend_plan_duration_seconds=float(
                _timing_float(
                    "backend_plan_duration_seconds",
                    self._runtime.last_backend_plan_duration_seconds,
                )
            ),
            last_window_scan_duration_seconds=float(
                _timing_float(
                    "window_scan_duration_seconds",
                    self._runtime.last_window_scan_duration_seconds,
                )
            ),
            last_capture_background_hash_skipped=(
                bool(capture_timing["background_hash_skipped"])
                if "background_hash_skipped" in capture_timing
                else bool(self._runtime.last_capture_background_hash_skipped)
            ),
            last_tick_skipped=bool(self._runtime.last_tick_skipped),
            last_tick_skip_reason=str(self._runtime.last_tick_skip_reason or ""),
            pending_visual_scene_count=max(0, int(self._pending_visual_scene_count or 0)),
            last_auto_recalibrate_attempts=max(
                0,
                int(self._last_auto_recalibrate_attempts or 0),
            ),
            last_auto_recalibrate_duration_seconds=float(
                self._last_auto_recalibrate_duration_seconds or 0.0
            ),
            last_auto_recalibrate_limited=bool(self._last_auto_recalibrate_limited),
            last_auto_recalibrate_error=str(self._last_auto_recalibrate_error or ""),
        )

    def _extract_text_from_image(
        self,
        image: Any,
        *,
        plan: SelectedOcrBackendPlan | None = None,
    ) -> OcrExtractionResult:
        if plan is not None:
            resolved_plan = plan
        elif self._custom_ocr_backend:
            resolved_plan = self._custom_backend_plan()
        else:
            resolved_plan = self._resolve_backend_plan()
        if self._custom_ocr_backend:
            return OcrExtractionResult(
                text=self._ocr_backend.extract_text(image),
                backend=resolved_plan.primary,
                backend_detail=resolved_plan.primary.detail or "custom_backend",
            )
        descriptors = [resolved_plan.primary]
        if resolved_plan.fallback.available:
            descriptors.append(resolved_plan.fallback)
        warnings: list[str] = []
        last_error: Exception | None = None
        for index, descriptor in enumerate(descriptors):
            if descriptor.backend is None:
                continue
            try:
                extract_with_boxes = getattr(descriptor.backend, "extract_text_with_boxes", None)
                if callable(extract_with_boxes):
                    try:
                        text, boxes = extract_with_boxes(image)
                    except Exception as boxes_exc:
                        extract_text = getattr(descriptor.backend, "extract_text", None)
                        if not callable(extract_text):
                            raise
                        self._logger.debug(
                            "ocr_reader backend %s boxes unavailable, falling back to text-only OCR: %s",
                            descriptor.kind,
                            boxes_exc,
                            exc_info=True,
                        )
                        warnings.append(
                            f"ocr_reader {descriptor.kind} boxes unavailable: {boxes_exc}"
                        )
                        text = extract_text(image)
                        boxes = []
                else:
                    text = descriptor.backend.extract_text(image)
                    boxes = []
                return OcrExtractionResult(
                    text=text,
                    backend=descriptor,
                    backend_detail=(
                        "fallback_after_runtime_error"
                        if index > 0
                        else (descriptor.detail or "selected_primary")
                    ),
                    warnings=warnings,
                    boxes=list(boxes),
                )
            except Exception as exc:
                last_error = exc
                warning = f"ocr_reader {descriptor.kind} failed: {exc}"
                warnings.append(warning)
                self._logger.warning("ocr_reader backend %s failed: %s", descriptor.kind, exc)
        if last_error is not None:
            raise last_error
        return OcrExtractionResult(backend=resolved_plan.primary, warnings=warnings)

    def _capture_and_extract_text(
        self,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        plan: SelectedOcrBackendPlan,
        collect_background_hash: bool = True,
        allow_separate_background_capture: bool = True,
    ) -> OcrExtractionResult:
        started_at = self._time_fn()
        background_hash = self._last_background_hash
        background_duration = 0.0
        background_hash_skipped = True
        capture_started_at = self._time_fn()
        frame = self._capture_backend.capture_frame(target, profile)
        capture_frame_duration = max(0.0, self._time_fn() - capture_started_at)
        frame_info = getattr(frame, "info", {}) if frame is not None else {}
        embedded_background_hash = (
            str(frame_info.get("galgame_source_background_hash") or "")
            if isinstance(frame_info, dict)
            else ""
        )
        if collect_background_hash and embedded_background_hash:
            background_hash = embedded_background_hash
            background_hash_skipped = False
            self._last_background_hash_capture_at = started_at
        elif (
            collect_background_hash
            and allow_separate_background_capture
            and started_at - float(self._last_background_hash_capture_at or 0.0)
            >= _BACKGROUND_HASH_MIN_INTERVAL_SECONDS
        ):
            try:
                background_started_at = self._time_fn()
                background_frame = self._capture_backend.capture_frame(
                    target,
                    self._background_capture_profile(),
                )
                background_hash = self._background_perceptual_hash(background_frame)
                background_duration = max(0.0, self._time_fn() - background_started_at)
                background_hash_skipped = False
                self._last_background_hash_capture_at = started_at
            except Exception as exc:
                self._logger.debug("ocr_reader background scene hash skipped: %s", exc, exc_info=True)
        hash_started_at = self._time_fn()
        capture_hash = self._capture_image_hash(frame)
        capture_hash_duration = max(0.0, self._time_fn() - hash_started_at)
        ocr_started_at = self._time_fn()
        extraction = self._extract_text_from_image(frame, plan=plan)
        ocr_duration = max(0.0, self._time_fn() - ocr_started_at)
        extraction.capture_image_hash = capture_hash
        extraction.background_hash = background_hash
        extraction.timing = {
            "total_duration_seconds": max(0.0, self._time_fn() - started_at),
            "capture_frame_duration_seconds": capture_frame_duration,
            "background_hash_duration_seconds": background_duration,
            "capture_image_hash_duration_seconds": capture_hash_duration,
            "ocr_extract_duration_seconds": ocr_duration,
            "background_hash_skipped": background_hash_skipped,
        }
        if isinstance(frame_info, dict):
            extraction.capture_backend_kind = str(
                frame_info.get("galgame_capture_backend_kind")
                or getattr(self._capture_backend, "last_backend_kind", "")
                or getattr(self._capture_backend, "selection", "")
            )
            extraction.capture_backend_detail = str(
                frame_info.get("galgame_capture_backend_detail")
                or getattr(self._capture_backend, "last_backend_detail", "")
                or ""
            )
            extraction.bounds_coordinate_space = str(
                frame_info.get("galgame_bounds_coordinate_space") or ""
            )
            source_size = frame_info.get("galgame_source_size")
            if isinstance(source_size, dict):
                extraction.source_size = dict(source_size)
            capture_rect = frame_info.get("galgame_capture_rect")
            if isinstance(capture_rect, dict):
                extraction.capture_rect = dict(capture_rect)
            window_rect = frame_info.get("galgame_window_rect")
            if isinstance(window_rect, dict):
                extraction.window_rect = dict(window_rect)
        else:
            extraction.capture_backend_kind = str(
                getattr(self._capture_backend, "last_backend_kind", "")
                or getattr(self._capture_backend, "selection", "")
                or ""
            )
            extraction.capture_backend_detail = str(
                getattr(self._capture_backend, "last_backend_detail", "") or ""
            )
        return extraction

    def _try_select_manual_target(
        self,
        windows: list[DetectedGameWindow],
        selection: WindowSelectionResult,
    ) -> bool:
        if not self._manual_target.is_manual():
            return False
        for candidate in windows:
            if self._manual_target.matches_exact(candidate) or self._manual_target.matches_hwnd(candidate):
                resolved_target = self._manual_target.resolved_for(candidate)
                self._manual_target = resolved_target
                selection.target = candidate
                selection.selection_detail = "manual_target_exact"
                selection.manual_target = resolved_target
                selection.selected_by_manual = True
                return True
        for candidate in windows:
            if self._manual_target.matches_signature(candidate):
                resolved_target = self._manual_target.resolved_for(candidate)
                self._manual_target = resolved_target
                selection.target = candidate
                selection.selection_detail = "manual_target_rebound"
                selection.manual_target = resolved_target
                selection.selected_by_manual = True
                return True
        selection.selection_detail = "manual_target_unavailable_fallback_to_auto"
        return False

    @staticmethod
    def _try_select_memory_reader_target(
        windows: list[DetectedGameWindow],
        selection: WindowSelectionResult,
        memory_reader_runtime: dict[str, Any] | None,
    ) -> bool:
        preferred_pid = int((memory_reader_runtime or {}).get("pid") or 0)
        preferred_process_name = str(
            (memory_reader_runtime or {}).get("process_name") or ""
        ).strip().lower()
        if preferred_pid > 0:
            for candidate in windows:
                if candidate.pid == preferred_pid:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "memory_reader_pid"
                    return True
        if preferred_process_name:
            for candidate in windows:
                if str(candidate.process_name or "").strip().lower() == preferred_process_name:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "memory_reader_process"
                    return True
        return False

    def _try_select_attached_target(
        self,
        windows: list[DetectedGameWindow],
        selection: WindowSelectionResult,
    ) -> bool:
        if self._attached_window is None:
            return False
        for candidate in windows:
            if candidate.hwnd == self._attached_window.hwnd:
                selection.target = candidate
                if selection.selection_mode == "auto":
                    selection.selection_detail = "attached_hwnd"
                return True
        if self._attached_window.pid:
            for candidate in windows:
                if candidate.pid == self._attached_window.pid:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "attached_pid"
                    return True
        return False

    def _select_locked_target_or_stop(
        self,
        windows: list[DetectedGameWindow],
        selection: WindowSelectionResult,
    ) -> bool:
        if not self._has_locked_target():
            return False
        for candidate in windows:
            if self._locked_target.matches_exact(candidate) or self._locked_target.matches_hwnd(candidate):
                selection.target = candidate
                if selection.selection_mode == "auto":
                    selection.selection_detail = "locked_target_exact"
                return True
        for candidate in windows:
            if self._locked_target.matches_signature(candidate):
                selection.target = candidate
                if selection.selection_mode == "auto":
                    selection.selection_detail = "locked_target_rebound"
                return True
        if selection.selection_mode == "auto":
            selection.selection_detail = "locked_target_unavailable"
        return True

    @staticmethod
    def _try_select_foreground_or_single_target(
        windows: list[DetectedGameWindow],
        selection: WindowSelectionResult,
    ) -> bool:
        foreground_hwnd = _foreground_window_handle()
        if foreground_hwnd:
            for candidate in windows:
                if candidate.hwnd == foreground_hwnd:
                    if not _is_confident_auto_window(candidate):
                        if selection.selection_mode == "auto":
                            selection.selection_detail = "foreground_window_needs_manual_confirmation"
                        return True
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "foreground_window"
                    return True
        if len(windows) == _WINDOW_SINGLE_FALLBACK_CANDIDATE_COUNT:
            candidate = windows[0]
            if foreground_hwnd and _is_confident_auto_window(candidate):
                selection.target = candidate
                if selection.selection_mode == "auto":
                    selection.selection_detail = "single_confident_candidate_without_foreground_match"
                return True
        return False

    def _select_target_window(
        self,
        windows: list[DetectedGameWindow],
        *,
        excluded_windows: list[DetectedGameWindow] | None = None,
        memory_reader_runtime: dict[str, Any] | None = None,
    ) -> WindowSelectionResult:
        excluded = list(excluded_windows or [])
        selection = WindowSelectionResult(
            selection_mode="manual" if self._manual_target.is_manual() else "auto",
            selection_detail="manual_target_active"
            if self._manual_target.is_manual()
            else "auto_candidate_scan",
            manual_target=self._manual_target,
            candidate_count=len(windows),
            excluded_candidate_count=len(excluded),
            last_exclude_reason=str(excluded[0].exclude_reason or "") if excluded else "",
        )
        if not windows:
            selection.selection_detail = (
                "manual_target_unavailable_fallback_to_auto"
                if self._manual_target.is_manual()
                else "no_eligible_window"
            )
            return selection

        if self._try_select_manual_target(windows, selection):
            return selection
        if self._try_select_memory_reader_target(windows, selection, memory_reader_runtime):
            return selection
        if self._try_select_attached_target(windows, selection):
            return selection
        if self._select_locked_target_or_stop(windows, selection):
            return selection
        if self._try_select_foreground_or_single_target(windows, selection):
            return selection
        if selection.selection_mode == "auto":
            selection.selection_detail = "auto_detect_needs_manual_fallback"
        return selection

    def _consume_choice_candidates_from_ocr_text(
        self,
        raw_text: str,
        *,
        now: float,
        state: _StableOcrTextState,
        allow_plain_text_choices: bool,
    ) -> bool:
        choices = _coerce_choice_lines(
            _stripped_ocr_lines(raw_text),
            allow_plain_text=allow_plain_text_choices,
        )
        if not choices:
            return False
        return self._emit_choices_from_candidates(choices, now=now, state=state)

    def _consume_ocr_text(
        self,
        raw_text: str,
        *,
        now: float,
        state: _StableOcrTextState | None = None,
        allow_choices: bool = True,
        allow_plain_text_choices: bool = False,
        emit_observed: bool = True,
        line_repeat_threshold: int | None = None,
    ) -> bool:
        tracker = state or self._default_ocr_state
        if allow_choices and self._consume_choice_candidates_from_ocr_text(
            raw_text,
            now=now,
            state=tracker,
            allow_plain_text_choices=allow_plain_text_choices,
        ):
            return True
        return self._emit_line_from_ocr_text(
            raw_text,
            now=now,
            state=tracker,
            emit_observed=emit_observed,
            repeat_threshold=line_repeat_threshold,
        )

    async def _end_session_if_needed(self, now: float) -> None:
        if self._writer.session_id:
            self._writer.end_session(ts=utc_now_iso(now))
            self._attached_window = None
            self._reset_default_ocr_state()
            self._reset_aihong_menu_state()
