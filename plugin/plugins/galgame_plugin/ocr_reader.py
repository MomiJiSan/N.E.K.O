from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, replace
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
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
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
_OCR_FOLLOWUP_CONFIRM_DELAY_SECONDS = 0.18
_CAPTURE_BACKEND_AUTO = "auto"
_CAPTURE_BACKEND_DXCAM = "dxcam"
_CAPTURE_BACKEND_IMAGEGRAB = "imagegrab"
_CAPTURE_BACKEND_PRINTWINDOW = "printwindow"
_STALE_CAPTURE_FRAME_THRESHOLD = 3


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


def _coerce_prefixed_choice_lines(lines: list[str]) -> list[str]:
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


def _looks_like_dialogue_line(text: str) -> bool:
    normalized = normalize_text(text).strip()
    if not normalized:
        return False
    return any(marker in normalized for marker in _DIALOGUE_LINE_MARKERS)


def _coerce_plain_choice_lines(lines: list[str]) -> list[str]:
    if not 2 <= len(lines) <= _AIHONG_MENU_MAX_LINES:
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
    if not 2 <= len(choices) <= _AIHONG_MENU_MAX_LINES:
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
    if not 2 <= len(choices) <= _AIHONG_MENU_MAX_LINES:
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
    if status_lines and len(normalized_choices) >= 2:
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


def _score_ocr_text(text: str) -> tuple[float, int, int]:
    normalized = normalize_text(text)
    if not normalized:
        return (-1.0, 0, 0)
    cjk_count = len(_CJK_CHAR_RE.findall(normalized))
    kana_count = len(_KANA_CHAR_RE.findall(normalized))
    ascii_tokens = _ASCII_TOKEN_RE.findall(normalized)
    isolated_ascii_tokens = sum(1 for token in ascii_tokens if len(token) == 1)
    multi_char_ascii_tokens = sum(1 for token in ascii_tokens if len(token) > 1)
    significant_chars = sum(1 for ch in normalized if not ch.isspace())
    score = (
        (cjk_count * 5.0)
        + (kana_count * 4.0)
        + (multi_char_ascii_tokens * 1.5)
        + (significant_chars * 0.2)
        - (isolated_ascii_tokens * 2.0)
    )
    return (score, cjk_count + kana_count, significant_chars)


def _significant_char_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def _looks_like_noise_ocr_text(text: str) -> bool:
    normalized = normalize_text(str(text or "")).strip()
    if not normalized:
        return True
    significant_chars = _significant_char_count(normalized)
    cjk_or_kana_count = len(_CJK_CHAR_RE.findall(normalized)) + len(_KANA_CHAR_RE.findall(normalized))
    if cjk_or_kana_count <= 0 and significant_chars <= 2:
        return True
    return False


def _prepare_ocr_image(image: Any) -> Any:
    from PIL import Image, ImageFilter, ImageOps

    resampling = getattr(Image, "Resampling", Image)
    prepared = image.convert("L")
    prepared = ImageOps.autocontrast(prepared)
    prepared = prepared.resize(
        (max(prepared.width * 2, 1), max(prepared.height * 2, 1)),
        resampling.LANCZOS,
    )
    prepared = prepared.filter(ImageFilter.MedianFilter(size=3))
    prepared = prepared.filter(ImageFilter.SHARPEN)
    return prepared


def _rapidocr_points(box: Any) -> list[tuple[float, float]]:
    if hasattr(box, "tolist"):
        box = box.tolist()
    if not isinstance(box, (list, tuple)):
        return []
    points: list[tuple[float, float]] = []
    for point in box:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points


def _should_insert_ascii_space(previous_text: str, next_text: str) -> bool:
    if not previous_text or not next_text:
        return False
    left = previous_text[-1]
    right = next_text[0]
    return left.isascii() and right.isascii() and left.isalnum() and right.isalnum()


def _join_ocr_segments(parts: list[str]) -> str:
    rendered = ""
    for part in parts:
        normalized = normalize_text(str(part or "")).replace("\n", " ").strip()
        if not normalized:
            continue
        if not rendered:
            rendered = normalized
            continue
        if _should_insert_ascii_space(rendered, normalized):
            rendered += " "
        rendered += normalized
    return rendered


@dataclass(slots=True)
class _RapidOcrToken:
    text: str
    score: float
    left: float
    right: float
    top: float
    bottom: float
    height: float


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


def _rapidocr_tokens_from_output(raw_output: Any) -> list[_RapidOcrToken]:
    payload = raw_output[0] if isinstance(raw_output, tuple) and raw_output else raw_output
    if not isinstance(payload, list):
        return []
    tokens: list[_RapidOcrToken] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        normalized = normalize_text(str(text or "")).strip()
        if not normalized:
            continue
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value < 0.45:
            continue
        points = _rapidocr_points(box)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        top = min(ys)
        bottom = max(ys)
        tokens.append(
            _RapidOcrToken(
                text=normalized,
                score=score_value,
                left=min(xs),
                right=max(xs),
                top=top,
                bottom=bottom,
                height=max(bottom - top, 1.0),
            )
        )
    return tokens


def _rapidocr_lines_from_output(raw_output: Any) -> list[tuple[str, float, OcrTextBox]]:
    tokens = _rapidocr_tokens_from_output(raw_output)
    if not tokens:
        return []
    tokens.sort(key=lambda token: (token.top, token.left))
    lines: list[list[_RapidOcrToken]] = []
    for token in tokens:
        token_center = (token.top + token.bottom) / 2.0
        placed = False
        for line in lines:
            line_top = min(item.top for item in line)
            line_bottom = max(item.bottom for item in line)
            line_center = (line_top + line_bottom) / 2.0
            threshold = max((line_bottom - line_top) * 0.6, token.height * 0.6, 12.0)
            if abs(token_center - line_center) <= threshold:
                line.append(token)
                placed = True
                break
        if not placed:
            lines.append([token])
    rendered_lines: list[str] = []
    line_results: list[tuple[str, float, OcrTextBox]] = []
    lines.sort(key=lambda line: (min(item.top for item in line), min(item.left for item in line)))
    for line in lines:
        line.sort(key=lambda item: item.left)
        text = _join_ocr_segments([item.text for item in line])
        if not text:
            continue
        rendered_lines.append(text)
        line_results.append(
            (
                text,
                sum(item.score for item in line) / len(line),
                OcrTextBox(
                    text=text,
                    left=min(item.left for item in line),
                    top=min(item.top for item in line),
                    right=max(item.right for item in line),
                    bottom=max(item.bottom for item in line),
                ),
            )
        )
    text = "\n".join(line for line in rendered_lines if line)
    normalized = normalize_text(text)
    if not normalized:
        return []
    scores = [score for _, score, _ in line_results]
    average_score = (sum(scores) / len(scores)) if scores else 0.0
    if _significant_char_count(normalized) < 4 and average_score < 0.55:
        return []
    return line_results


def _rapidocr_text_from_output(raw_output: Any) -> str:
    lines = _rapidocr_lines_from_output(raw_output)
    if not lines:
        return ""
    return "\n".join(text for text, _score, _box in lines)


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
    last_poll_started_at: str = ""
    last_poll_completed_at: str = ""
    last_poll_duration_seconds: float = 0.0
    last_poll_emitted_event: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "last_poll_started_at": self.last_poll_started_at,
            "last_poll_completed_at": self.last_poll_completed_at,
            "last_poll_duration_seconds": self.last_poll_duration_seconds,
            "last_poll_emitted_event": self.last_poll_emitted_event,
        }


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


@dataclass(slots=True)
class OcrBackendDescriptor:
    kind: str = ""
    backend: OcrBackend | None = None
    path: str = ""
    model: str = ""
    detail: str = ""
    available: bool = False


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


class CaptureBackend(Protocol):
    def is_available(self) -> bool: ...

    def describe_target(self, target: DetectedGameWindow) -> str: ...

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any: ...


class OcrBackend(Protocol):
    def is_available(self) -> bool: ...

    def extract_text(self, image: Any) -> str: ...


def _target_window_rect(target: DetectedGameWindow) -> tuple[int, int, int, int]:
    import win32gui

    rect = win32gui.GetWindowRect(target.hwnd)
    width = int(rect[2] - rect[0])
    height = int(rect[3] - rect[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid window dimensions: {width}x{height}")
    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))


def _run_with_thread_dpi_awareness(fn: Callable[[], tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    user32 = getattr(ctypes, "windll", None)
    user32 = getattr(user32, "user32", None) if user32 is not None else None
    set_context = getattr(user32, "SetThreadDpiAwarenessContext", None) if user32 is not None else None
    if not callable(set_context):
        return fn()
    old_context = None
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2. This is thread-local and
        # avoids globally changing the plugin process.
        old_context = set_context(ctypes.c_void_p(-4))
    except Exception:
        old_context = None
    try:
        return fn()
    finally:
        if old_context:
            try:
                set_context(old_context)
            except Exception:
                pass


def _target_client_rect(target: DetectedGameWindow) -> tuple[int, int, int, int]:
    import win32gui

    def _read_rect() -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetClientRect(target.hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(target.hwnd, (left, top))
        screen_right, screen_bottom = win32gui.ClientToScreen(target.hwnd, (right, bottom))
        return (int(screen_left), int(screen_top), int(screen_right), int(screen_bottom))

    rect = _run_with_thread_dpi_awareness(_read_rect)
    width = int(rect[2] - rect[0])
    height = int(rect[3] - rect[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid client dimensions: {width}x{height}")
    return rect


def _require_visible_capture_target(target: DetectedGameWindow, *, backend_kind: str) -> None:
    if not target.hwnd:
        raise RuntimeError(f"{backend_kind}: target_window_not_resolved_for_capture")
    try:
        import win32gui

        if not win32gui.IsWindow(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_invalid_for_capture")
        if not win32gui.IsWindowVisible(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_not_visible_for_capture")
        if win32gui.IsIconic(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_minimized_for_capture")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{backend_kind}: target_window_visibility_check_failed: {exc}") from exc


def _crop_window_image(
    image: Any,
    *,
    window_rect: tuple[int, int, int, int],
    profile: OcrCaptureProfile,
    backend_kind: str,
    backend_detail: str,
) -> Any:
    width = int(window_rect[2] - window_rect[0])
    height = int(window_rect[3] - window_rect[1])
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

    cropped = image.crop((left, top, right, bottom))
    cropped.info["galgame_bounds_coordinate_space"] = "capture"
    cropped.info["galgame_source_size"] = {"width": float(crop_w), "height": float(crop_h)}
    cropped.info["galgame_capture_rect"] = {
        "left": float(window_rect[0] + left),
        "top": float(window_rect[1] + top),
        "right": float(window_rect[0] + right),
        "bottom": float(window_rect[1] + bottom),
    }
    cropped.info["galgame_window_rect"] = {
        "left": float(window_rect[0]),
        "top": float(window_rect[1]),
        "right": float(window_rect[2]),
        "bottom": float(window_rect[3]),
    }
    cropped.info["galgame_capture_backend_kind"] = backend_kind
    cropped.info["galgame_capture_backend_detail"] = backend_detail
    return cropped


class ImageGrabCaptureBackend:
    kind = _CAPTURE_BACKEND_IMAGEGRAB

    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def is_available(self) -> bool:
        try:
            import win32gui
            from PIL import ImageGrab
            return bool(win32gui and ImageGrab)
        except ImportError:
            return False

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any:
        from PIL import ImageGrab

        _require_visible_capture_target(target, backend_kind=self.kind)
        rect = _target_window_rect(target)
        image = ImageGrab.grab(bbox=rect, all_screens=True).convert("RGB")
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected",
        )


class PrintWindowCaptureBackend:
    kind = _CAPTURE_BACKEND_PRINTWINDOW

    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def is_available(self) -> bool:
        try:
            import win32gui
            import win32ui
            import win32con
            return bool(win32gui and win32ui and win32con)
        except ImportError:
            return False

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any:
        rect = _target_window_rect(target)
        image = self._capture_full_window(target.hwnd, rect)
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected_legacy_fallback",
        )

    @staticmethod
    def _capture_full_window(hwnd: int, rect: tuple[int, int, int, int]) -> Any:
        import win32gui
        import win32ui
        import win32con
        from PIL import Image

        width = int(rect[2] - rect[0])
        height = int(rect[3] - rect[1])
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
        return image


class DxcamCaptureBackend:
    kind = _CAPTURE_BACKEND_DXCAM

    def __init__(self, *, logger=None) -> None:
        self._logger = logger
        self._camera = None
        self._last_create_error = ""

    def is_available(self) -> bool:
        try:
            import dxcam
            return bool(dxcam)
        except ImportError:
            return False

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def _camera_instance(self):
        if self._camera is not None:
            return self._camera
        import dxcam

        try:
            self._camera = dxcam.create(output_color="RGB")
        except Exception as exc:
            self._last_create_error = str(exc)
            raise RuntimeError(f"dxcam_create_failed: {exc}") from exc
        if self._camera is None:
            raise RuntimeError("dxcam_create_failed: returned None")
        return self._camera

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any:
        from PIL import Image

        _require_visible_capture_target(target, backend_kind=self.kind)
        rect = _target_client_rect(target)
        camera = self._camera_instance()
        frame = camera.grab(region=rect)
        if frame is None:
            raise RuntimeError("dxcam_grab_returned_none")
        image = Image.fromarray(frame).convert("RGB")
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected_client_rect",
        )


class Win32CaptureBackend:
    def __init__(self, *, logger=None, selection: str = _CAPTURE_BACKEND_AUTO) -> None:
        self._logger = logger
        self.selection = str(selection or _CAPTURE_BACKEND_AUTO).strip().lower()
        self._backends = self._build_backends()
        self.last_backend_kind = ""
        self.last_backend_detail = ""

    def _build_backends(self) -> list[CaptureBackend]:
        imagegrab = ImageGrabCaptureBackend(logger=self._logger)
        printwindow = PrintWindowCaptureBackend(logger=self._logger)
        dxcam = DxcamCaptureBackend(logger=self._logger)
        if self.selection == _CAPTURE_BACKEND_DXCAM:
            return [dxcam]
        if self.selection == _CAPTURE_BACKEND_IMAGEGRAB:
            return [imagegrab]
        if self.selection == _CAPTURE_BACKEND_PRINTWINDOW:
            return [printwindow]
        return [dxcam, imagegrab, printwindow]

    def is_available(self) -> bool:
        if self.selection != _CAPTURE_BACKEND_AUTO:
            return bool(self._backends) and self._backends[0].is_available()
        return any(backend.is_available() for backend in self._backends)

    def describe_target(self, target: DetectedGameWindow) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: DetectedGameWindow, profile: OcrCaptureProfile) -> Any:
        errors: list[str] = []
        for backend in self._backends:
            kind = str(getattr(backend, "kind", backend.__class__.__name__))
            if not backend.is_available():
                errors.append(f"{kind}_unavailable")
                continue
            try:
                frame = backend.capture_frame(target, profile)
                self.last_backend_kind = kind
                frame_info = getattr(frame, "info", None)
                frame_backend_detail = (
                    str(frame_info.get("galgame_capture_backend_detail") or "")
                    if isinstance(frame_info, dict)
                    else ""
                )
                self.last_backend_detail = frame_backend_detail or (
                    "dxcam_unavailable_fallback"
                    if kind != _CAPTURE_BACKEND_DXCAM and "dxcam_unavailable" in errors
                    else "selected"
                )
                if isinstance(frame_info, dict):
                    frame_info["galgame_capture_backend_kind"] = kind
                    frame_info["galgame_capture_backend_detail"] = self.last_backend_detail
                return frame
            except Exception as exc:
                errors.append(f"{kind}_failed:{exc}")
                if any(
                    marker in str(exc)
                    for marker in (
                        "target_window_not_resolved_for_capture",
                        "target_window_invalid_for_capture",
                        "target_window_not_visible_for_capture",
                        "target_window_minimized_for_capture",
                    )
                ):
                    raise
                if self.selection != _CAPTURE_BACKEND_AUTO:
                    raise
                continue
        if self.selection != _CAPTURE_BACKEND_AUTO:
            raise RuntimeError(
                f"{self.selection}: capture_backend_unavailable"
                + (f": {'; '.join(errors)}" if errors else "")
            )
        raise RuntimeError("; ".join(errors) or "capture_backend_unavailable")


class TesseractOcrBackend:
    def __init__(
        self,
        *,
        tesseract_path: str = "",
        install_target_dir_raw: str = "",
        languages: str = "",
    ) -> None:
        self._tesseract_path = tesseract_path
        self._install_target_dir_raw = install_target_dir_raw
        self._languages = languages

    def is_available(self) -> bool:
        path = resolve_tesseract_path(
            self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
        )
        if not path:
            return False
        inspection = inspect_tesseract_installation(
            configured_path=self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
            languages=self._languages,
        )
        return bool(inspection.get("installed"))

    def extract_text(self, image: Any) -> str:
        import pytesseract

        path = resolve_tesseract_path(
            self._tesseract_path,
            install_target_dir_raw=self._install_target_dir_raw,
        )
        if path:
            pytesseract.pytesseract.tesseract_cmd = path
        lang = self._languages
        # PSM 6 assumes a single dialogue block, which matches VN subtitle boxes.
        config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        prepared = _prepare_ocr_image(image)

        best_text = ""
        best_score = (-1.0, 0, 0)
        for candidate in (image, prepared):
            text = pytesseract.image_to_string(candidate, lang=lang, config=config).strip()
            score = _score_ocr_text(text)
            if score > best_score:
                best_text = text
                best_score = score
        return best_text


class RapidOcrBackend:
    def __init__(
        self,
        *,
        install_target_dir_raw: str,
        engine_type: str,
        lang_type: str,
        model_type: str,
        ocr_version: str,
    ) -> None:
        self._install_target_dir_raw = install_target_dir_raw
        self._engine_type = engine_type
        self._lang_type = lang_type
        self._model_type = model_type
        self._ocr_version = ocr_version
        self._runtime = None

    def is_available(self) -> bool:
        inspection = inspect_rapidocr_installation(
            install_target_dir_raw=self._install_target_dir_raw,
            engine_type=self._engine_type,
            lang_type=self._lang_type,
            model_type=self._model_type,
            ocr_version=self._ocr_version,
        )
        return bool(inspection.get("installed"))

    def extract_text(self, image: Any) -> str:
        import numpy as np

        if self._runtime is None:
            self._runtime, _metadata = load_rapidocr_runtime(
                install_target_dir_raw=self._install_target_dir_raw,
                engine_type=self._engine_type,
                lang_type=self._lang_type,
                model_type=self._model_type,
                ocr_version=self._ocr_version,
                force_reload=False,
            )
        prepared = _prepare_ocr_image(image).convert("RGB")
        output = self._runtime(np.asarray(prepared))
        return _rapidocr_text_from_output(output)

    def extract_text_with_boxes(self, image: Any) -> tuple[str, list[OcrTextBox]]:
        import numpy as np

        if self._runtime is None:
            self._runtime, _metadata = load_rapidocr_runtime(
                install_target_dir_raw=self._install_target_dir_raw,
                engine_type=self._engine_type,
                lang_type=self._lang_type,
                model_type=self._model_type,
                ocr_version=self._ocr_version,
                force_reload=False,
            )
        prepared = _prepare_ocr_image(image).convert("RGB")
        output = self._runtime(np.asarray(prepared))
        lines = _rapidocr_lines_from_output(output)
        if not lines:
            return "", []
        scale_x = prepared.width / max(float(getattr(image, "width", prepared.width)), 1.0)
        scale_y = prepared.height / max(float(getattr(image, "height", prepared.height)), 1.0)
        boxes = [
            OcrTextBox(
                text=box.text,
                left=box.left / scale_x,
                top=box.top / scale_y,
                right=box.right / scale_x,
                bottom=box.bottom / scale_y,
            )
            for _text, _score, box in lines
        ]
        return "\n".join(text for text, _score, _box in lines), boxes


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
        if not title or len(title) < 2:
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
            except Exception:
                pass
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
    except Exception:
        return 0


def _root_window_handle(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        root = int(ctypes.windll.user32.GetAncestor(int(hwnd), 2))
        return root or int(hwnd)
    except Exception:
        return int(hwnd)


def _window_process_id(hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        return int(pid.value or 0)
    except Exception:
        return 0


def _window_process_name(pid: int) -> str:
    if not pid or psutil is None:
        return ""
    try:
        return str(psutil.Process(int(pid)).name() or "").strip()
    except Exception:
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
                "stability": self._state.get("stability", ""),
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
            "scene_id": self._current_scene_id(),
            "line_id": line_id,
            "route_id": OCR_READER_ROUTE_ID,
            "is_menu_open": False,
            "save_context": self._state.get("save_context", {"kind": "unknown", "slot_id": "", "display_name": ""}),
            "stability": "stable",
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
                "stability": "stable",
            },
            ts=ts,
        )
        return True

    def emit_line_observed(self, raw_text: str, *, ts: str) -> bool:
        cleaned = raw_text.strip()
        if not cleaned or not self._session_id:
            return False
        speaker, text = self._split_speaker_text(cleaned)
        if not text:
            return False
        current_text = str(self._state.get("text") or "")
        current_speaker = str(self._state.get("speaker") or "")
        current_stability = str(self._state.get("stability") or "")
        if current_text == text and current_speaker == speaker and current_stability in {"tentative", "stable"}:
            return False
        line_id = self._line_id_for_text(text)
        self._state = {
            **self._state,
            "speaker": speaker,
            "text": text,
            "choices": [],
            "scene_id": self._current_scene_id(),
            "line_id": line_id,
            "route_id": OCR_READER_ROUTE_ID,
            "is_menu_open": False,
            "save_context": self._state.get("save_context", {"kind": "unknown", "slot_id": "", "display_name": ""}),
            "stability": "tentative",
            "ts": ts,
        }
        self._append_event(
            "line_observed",
            {
                "speaker": speaker,
                "text": text,
                "line_id": line_id,
                "line_id_source": "text_hash",
                "scene_id": self._state["scene_id"],
                "route_id": self._state["route_id"],
                "stability": "tentative",
            },
            ts=ts,
        )
        return True

    def emit_choices(
        self,
        choices: list[str],
        *,
        ts: str,
        choice_bounds: list[dict[str, float] | None] | None = None,
        choice_bounds_metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not choices or not self._session_id:
            return False
        line_id = str(self._state.get("line_id") or "")
        if not line_id:
            line_id = self._line_id_for_text(_canonical_choice_candidate_text(choices))
        bounds = list(choice_bounds or [])
        bounds_metadata = dict(choice_bounds_metadata or {})
        payload_choices = []
        for index, text in enumerate(choices):
            item = {
                "choice_id": f"{line_id}#choice{index}",
                "text": text,
                "index": index,
                "enabled": True,
            }
            if index < len(bounds) and bounds[index]:
                item["bounds"] = dict(bounds[index] or {})
                for key in (
                    "bounds_coordinate_space",
                    "source_size",
                    "capture_rect",
                    "window_rect",
                ):
                    value = bounds_metadata.get(key)
                    if value:
                        item[key] = dict(value) if isinstance(value, dict) else value
            payload_choices.append(item)
        self._state = {
            **self._state,
            "line_id": line_id,
            "scene_id": self._current_scene_id(),
            "choices": payload_choices,
            "is_menu_open": True,
            "stability": "choices",
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
            "scene_id": self._current_scene_id(),
            "line_id": "",
            "route_id": OCR_READER_ROUTE_ID,
            "is_menu_open": False,
            "save_context": {"kind": "unknown", "slot_id": "", "display_name": ""},
            "stability": "",
            "ts": ts,
        }

    def _current_scene_id(self) -> str:
        state = getattr(self, "_state", {}) or {}
        current = str(state.get("scene_id") or "").strip()
        if current and current != OCR_READER_UNKNOWN_SCENE:
            return current
        return f"ocr:{self._game_id or 'unknown'}:scene-0001"

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
                "stability": str(self._state.get("stability") or ""),
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
        self._aihong_menu_ocr_state = _StableOcrTextState()
        self._aihong_stage = _AIHONG_DIALOGUE_STAGE
        self._aihong_dialogue_idle_polls = 0
        self._aihong_menu_missing_polls = 0
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
        self._consecutive_same_capture_frames = 0
        self._stale_capture_backend = False
        self._capture_backend_kind = str(getattr(self._capture_backend, "selection", "custom"))
        self._capture_backend_detail = ""

    def update_config(self, config: GalgameConfig) -> None:
        self._config = config
        self._runtime.enabled = config.ocr_reader_enabled
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
        if self._writer.bridge_root != config.bridge_root:
            self._writer = OcrReaderBridgeWriter(
                bridge_root=config.bridge_root,
                time_fn=self._time_fn,
            )

    def update_advance_speed(self, advance_speed: str) -> None:
        normalized = str(advance_speed or "").strip().lower()
        self._advance_speed = normalized if normalized in ADVANCE_SPEEDS else ADVANCE_SPEED_MEDIUM

    def _line_changed_repeat_threshold(self) -> int:
        if self._advance_speed == ADVANCE_SPEED_FAST:
            return 1
        if self._advance_speed == ADVANCE_SPEED_SLOW:
            return 3
        return 2

    def _mark_observed_progress(self, *, now: float) -> None:
        self._consecutive_no_text_polls = 0
        self._last_observed_at = utc_now_iso(now)

    def _mark_no_text_poll(self) -> None:
        self._consecutive_no_text_polls += 1

    def _ocr_capture_diagnostic_required(self) -> bool:
        return self._consecutive_no_text_polls >= 3

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
        try:
            if hasattr(frame, "tobytes") and hasattr(frame, "size"):
                size = getattr(frame, "size", "")
                payload = frame.tobytes()
                return hashlib.sha1(repr(size).encode("utf-8") + payload).hexdigest()[:16]
        except Exception:
            return ""
        try:
            return hashlib.sha1(repr(frame).encode("utf-8", "ignore")).hexdigest()[:16]
        except Exception:
            return ""

    def _line_payload_from_writer(self, *, stability: str) -> dict[str, Any]:
        state = getattr(self._writer, "_state", {})
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
        state = getattr(self._writer, "_state", {})
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
        if target is not None:
            is_foreground, foreground_match_reason = _foreground_matches_target(
                foreground_hwnd,
                target,
            )
            self._runtime.target_is_foreground = is_foreground
            self._runtime.effective_window_key = str(target.window_key or self._runtime.effective_window_key)
            self._runtime.effective_window_title = str(target.title or self._runtime.effective_window_title)
            self._runtime.effective_process_name = str(target.process_name or self._runtime.effective_process_name)
            if not self._runtime.process_name:
                self._runtime.process_name = str(target.process_name or "")
            if not self._runtime.window_title:
                self._runtime.window_title = str(target.title or "")
            if not self._runtime.pid:
                self._runtime.pid = int(target.pid or 0)
            detail = (
                f"{detail}:foreground_{foreground_match_reason}"
                if is_foreground
                else f"{detail}:background"
            )
        elif self._runtime.effective_window_key or self._runtime.process_name:
            detail = detail or "target_unresolved"
        else:
            detail = "no_target"
        self._runtime.foreground_refresh_at = utc_now_iso(self._time_fn())
        self._runtime.foreground_refresh_detail = detail
        self._runtime.foreground_hwnd = max(0, int(foreground_hwnd or 0))
        self._runtime.target_hwnd = max(0, int(target_hwnd or 0))
        return self._runtime.to_dict()

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
                self._aihong_stage
                if self._should_use_aihong_two_stage(target)
                else OCR_CAPTURE_PROFILE_STAGE_DIALOGUE
            )
        capture_profile_selection = self._capture_profile_selection_for_target(
            target,
            stage=capture_stage,
        )
        self._runtime.process_name = str(target.process_name or self._runtime.process_name)
        self._runtime.pid = int(target.pid or self._runtime.pid)
        self._runtime.window_title = str(target.title or self._runtime.window_title)
        self._runtime.width = int(target.width or self._runtime.width)
        self._runtime.height = int(target.height or self._runtime.height)
        self._runtime.aspect_ratio = resolved_aspect_ratio
        self._runtime.capture_stage = capture_stage
        self._runtime.capture_profile = capture_profile_selection.profile.to_dict()
        self._runtime.capture_profile_match_source = capture_profile_selection.match_source
        self._runtime.capture_profile_bucket_key = capture_profile_selection.bucket_key
        self._runtime.consecutive_no_text_polls = max(0, int(self._consecutive_no_text_polls or 0))
        self._runtime.last_observed_at = str(self._last_observed_at or self._runtime.last_observed_at)
        self._runtime.last_capture_stage = capture_stage
        self._runtime.last_capture_profile = capture_profile_selection.profile.to_dict()
        self._runtime.ocr_capture_diagnostic_required = self._ocr_capture_diagnostic_required()
        self._runtime.ocr_context_state = self._ocr_context_state_for_detail(
            status=self._runtime.status,
            detail=self._runtime.detail,
        )
        self._runtime.last_capture_attempt_at = str(
            self._last_capture_attempt_at or self._runtime.last_capture_attempt_at
        )
        self._runtime.last_capture_completed_at = str(
            self._last_capture_completed_at or self._runtime.last_capture_completed_at
        )
        self._runtime.last_capture_error = str(
            self._last_capture_error or self._runtime.last_capture_error
        )
        self._runtime.last_raw_ocr_text = str(
            self._last_raw_ocr_text or self._runtime.last_raw_ocr_text
        )
        self._runtime.last_observed_line = dict(
            self._last_observed_line or self._runtime.last_observed_line
        )
        self._runtime.last_stable_line = dict(
            self._last_stable_line or self._runtime.last_stable_line
        )
        self._runtime.effective_window_key = str(target.window_key or self._runtime.effective_window_key)
        self._runtime.effective_window_title = str(target.title or self._runtime.effective_window_title)
        self._runtime.effective_process_name = str(
            target.process_name or self._runtime.effective_process_name
        )
        foreground_hwnd = _foreground_window_handle()
        self._runtime.target_is_foreground = _foreground_matches_target(
            foreground_hwnd,
            target,
        )[0]
        self._runtime.foreground_hwnd = max(0, int(foreground_hwnd or 0))
        self._runtime.target_hwnd = max(0, int(target.hwnd or 0))
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
        basis = 100
        start = int(round((current_value + delta_start) * basis))
        end = int(round((current_value + delta_end) * basis))
        step_value = max(1, int(round(step * basis)))
        for raw in range(start, end + 1, step_value):
            normalized = max(0.0, min(raw / basis, 0.98))
            key = int(round(normalized * basis))
            if key in seen:
                continue
            seen.add(key)
            values.append(round(normalized, 2))
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

    def auto_recalibrate_dialogue_profile(self) -> dict[str, Any]:
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
            or len(image_size) < 2
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
            seen = {int(round(value * 100)) for value in merged}
            for raw in additions:
                normalized = round(max(0.0, min(float(raw), 0.98)), 2)
                key = int(round(normalized * 100))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(normalized)
            return sorted(merged)

        horizontal_pairs: list[tuple[float, float]] = []

        def _add_horizontal_pair(left_ratio: float, right_ratio: float) -> None:
            left_ratio = round(max(0.0, min(float(left_ratio), 0.45)), 2)
            right_ratio = round(max(0.0, min(float(right_ratio), 0.45)), 2)
            if left_ratio + right_ratio >= 0.95:
                return
            pair = (left_ratio, right_ratio)
            if pair not in horizontal_pairs:
                horizontal_pairs.append(pair)

        if is_aihong_target:
            _add_horizontal_pair(0.0, 0.0)
            _add_horizontal_pair(0.02, 0.02)
            _add_horizontal_pair(0.05, 0.05)
        _add_horizontal_pair(base_profile.left_inset_ratio, base_profile.right_inset_ratio)
        if not is_aihong_target and (
            base_profile.left_inset_ratio > 0.0 or base_profile.right_inset_ratio > 0.0
        ):
            _add_horizontal_pair(
                max(0.0, base_profile.left_inset_ratio - 0.05),
                max(0.0, base_profile.right_inset_ratio - 0.05),
            )

        top_values = self._scan_ratio_values(
            base_profile.top_ratio,
            delta_start=-0.14,
            delta_end=0.08,
            step=0.02,
        )
        bottom_values = self._scan_ratio_values(
            base_profile.bottom_inset_ratio,
            delta_start=-0.04,
            delta_end=0.08,
            step=0.02,
        )
        if is_aihong_target:
            aihong_preset = OcrCaptureProfile.from_dict(_AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET)
            top_values = _append_ratio_values(
                top_values,
                self._scan_ratio_values(
                    aihong_preset.top_ratio,
                    delta_start=-0.08,
                    delta_end=0.08,
                    step=0.02,
                ),
            )
            bottom_values = _append_ratio_values(
                bottom_values,
                self._scan_ratio_values(
                    aihong_preset.bottom_inset_ratio,
                    delta_start=-0.05,
                    delta_end=0.08,
                    step=0.01,
                ),
            )
        backend_plan = None if self._custom_ocr_backend else self._resolve_backend_plan()
        if backend_plan is not None and not backend_plan.primary.available:
            raise ValueError("当前 OCR backend 不可用，无法自动重校准对白区")

        best_candidate: dict[str, Any] | None = None
        current_distance_basis = (
            round(base_profile.top_ratio, 2),
            round(base_profile.bottom_inset_ratio, 2),
        )
        min_height = max(24, int(image_height * 0.08))
        max_height = max(min_height, int(image_height * 0.45))
        visited_pairs: set[tuple[float, float, float, float]] = set()

        def _consider_candidate(
            top_ratio: float,
            bottom_inset_ratio: float,
            left_inset_ratio: float,
            right_inset_ratio: float,
        ) -> None:
            nonlocal best_candidate
            key = (
                round(top_ratio, 2),
                round(bottom_inset_ratio, 2),
                round(left_inset_ratio, 2),
                round(right_inset_ratio, 2),
            )
            if key in visited_pairs:
                return
            visited_pairs.add(key)
            if top_ratio + bottom_inset_ratio >= 1.0 or left_inset_ratio + right_inset_ratio >= 1.0:
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
            if right_px - left_px < 10:
                return
            extracted = self._extract_text_from_image(
                full_image.crop((left_px, top_px, right_px, bottom_px)),
                plan=backend_plan,
            )
            sample_text = str(extracted.text or "").strip()
            if not sample_text or _looks_like_self_ui_text(sample_text):
                return
            score, cjk_count, significant_chars = _score_ocr_text(sample_text)
            if significant_chars < 8 or cjk_count <= 0:
                return
            distance = abs(round(top_ratio, 2) - current_distance_basis[0]) + abs(
                round(bottom_inset_ratio, 2) - current_distance_basis[1]
            )
            width_ratio = max(0.0, 1.0 - left_inset_ratio - right_inset_ratio)
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
        for delta in (0.0, 0.02, -0.02, 0.04):
            candidate_value = round(base_profile.bottom_inset_ratio + delta, 2)
            if candidate_value in bottom_values and candidate_value not in preferred_bottom_values:
                preferred_bottom_values.append(candidate_value)
        if not preferred_bottom_values:
            preferred_bottom_values = list(bottom_values)

        for top_ratio in top_values:
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
            best_top_ratio = round(float(best_candidate["profile"].top_ratio), 2)
            for delta in (-0.02, 0.0, 0.02):
                candidate_value = round(best_top_ratio + delta, 2)
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
                + f" / 示例文本：{sample_text[:24]}"
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
        self._consecutive_same_capture_frames = 0
        self._stale_capture_backend = False

    def _reset_aihong_menu_state(self) -> None:
        self._aihong_menu_ocr_state.reset()
        self._aihong_stage = _AIHONG_DIALOGUE_STAGE
        self._aihong_dialogue_idle_polls = 0
        self._aihong_menu_missing_polls = 0

    def _has_manual_capture_profile(self, target: DetectedGameWindow) -> bool:
        return _uses_manual_capture_profile(self._capture_profiles, target)

    def _should_use_aihong_two_stage(self, target: DetectedGameWindow) -> bool:
        return _matches_aihong_target(target)

    @staticmethod
    def _stabilize_text_key(
        text: str,
        *,
        state: _StableOcrTextState,
        repeat_threshold: int = 2,
    ) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return False
        if cleaned == state.last_raw_text:
            state.repeat_count += 1
        else:
            state.repeat_count = 1
            state.last_raw_text = cleaned
        if state.repeat_count < max(1, int(repeat_threshold)):
            return False
        if cleaned == state.stable_text:
            return False
        state.stable_text = cleaned
        return True

    def _emit_line_from_ocr_text(
        self,
        raw_text: str,
        *,
        now: float,
        state: _StableOcrTextState | None = None,
    ) -> bool:
        if _looks_like_noise_ocr_text(raw_text):
            return False
        self._last_raw_ocr_text = str(raw_text or "")
        if self._writer.emit_line_observed(raw_text, ts=utc_now_iso(now)):
            self._last_observed_line = self._line_payload_from_writer(stability="tentative")
        tracker = state or self._default_ocr_state
        if not self._stabilize_text_key(
            raw_text,
            state=tracker,
            repeat_threshold=self._line_changed_repeat_threshold(),
        ):
            return False
        emitted = self._writer.emit_line(raw_text, ts=utc_now_iso(now))
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
            repeat_threshold=2,
        ):
            return False
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
            state.repeat_count == 1
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
                    state=self._aihong_menu_ocr_state,
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
        poll_started_at = now
        result = OcrReaderTickResult(runtime=self._runtime.to_dict())
        self._runtime.last_poll_started_at = utc_now_iso(poll_started_at)

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

        backend_plan = await asyncio.to_thread(self._resolve_backend_plan)
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

        scanned_windows = await asyncio.to_thread(self._window_scanner)
        eligible_windows, excluded_windows = self._prepare_window_inventory(scanned_windows)
        selection = self._select_target_window(
            eligible_windows,
            excluded_windows=excluded_windows,
            memory_reader_runtime=memory_reader_runtime,
        )
        self._last_selection = selection
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
        profile_stage = self._aihong_stage if aihong_two_stage_enabled else _AIHONG_DIALOGUE_STAGE
        capture_profile_selection = self._capture_profile_selection_for_target(
            target,
            stage=profile_stage,
        )
        profile = capture_profile_selection.profile

        if self._attached_window is None or self._attached_window.pid != target.pid:
            if (
                not self._writer.session_id
                or self._writer.game_id != _ocr_game_id_from_process(target.process_name or target.title)
            ):
                self._writer.start_session(target)
                result.should_rescan = True
            self._attached_window = target
            self._last_heartbeat_at = now
            self._reset_default_ocr_state()
            self._reset_aihong_menu_state()
            startup_profile_stage = (
                self._aihong_stage if aihong_two_stage_enabled else OCR_CAPTURE_PROFILE_STAGE_DEFAULT
            )
            startup_profile_selection = self._capture_profile_selection_for_target(
                target,
                stage=(
                    self._aihong_stage
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

        emitted = False
        guard_blocked = False
        active_backend = backend_plan.primary
        backend_detail_override = ""
        runtime_profile = profile
        runtime_capture_profile_selection = capture_profile_selection
        event_seq_before_capture = int(self._writer.last_seq or 0)
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
            )
            capture_completed = True
            self._record_capture_completed(
                now=now,
                raw_text=extraction.text,
                image_hash=extraction.capture_image_hash,
            )
            self._record_capture_geometry(extraction)
            self._capture_backend_kind = extraction.capture_backend_kind
            self._capture_backend_detail = extraction.capture_backend_detail
            active_backend = extraction.backend if extraction.backend.kind else backend_plan.primary
            backend_detail_override = extraction.backend_detail
            result.warnings.extend(extraction.warnings)
            if extraction.text and _looks_like_self_ui_text(extraction.text):
                guard_blocked = True
                result.warnings.append("ocr_reader ignored text that looks like the N.E.K.O plugin UI")
                self._default_ocr_state.reset()
                self._aihong_menu_ocr_state.reset()
            else:
                if aihong_two_stage_enabled:
                    if self._aihong_stage == _AIHONG_MENU_STAGE:
                        menu_result = self._consume_aihong_menu_stage_text(
                            extraction.text,
                            now=now,
                            boxes=extraction.boxes,
                            choice_bounds_metadata=_extraction_choice_bounds_metadata(extraction),
                        )
                        emitted = bool(menu_result.emitted_kind)
                        if menu_result.emitted_kind == "line":
                            self._aihong_stage = _AIHONG_DIALOGUE_STAGE
                            self._aihong_dialogue_idle_polls = 0
                            self._aihong_menu_missing_polls = 0
                            self._aihong_menu_ocr_state.reset()
                        elif menu_result.has_menu_candidate:
                            self._aihong_menu_missing_polls = 0
                        else:
                            self._aihong_menu_missing_polls += 1
                            if (
                                extraction.text
                                and not _looks_like_noise_ocr_text(extraction.text)
                            ):
                                self._reset_aihong_menu_state()
                            elif self._aihong_menu_missing_polls >= 2:
                                self._reset_aihong_menu_state()
                    else:
                        dialogue_menu_choices = _coerce_aihong_menu_choices(
                            _stripped_ocr_lines(extraction.text)
                        )
                        dialogue_text_is_menu_status = _looks_like_aihong_menu_status_only_text(
                            extraction.text
                        )
                        dialogue_emitted = False
                        if dialogue_menu_choices:
                            dialogue_emitted = bool(
                                self._emit_choices_from_candidates(
                                    dialogue_menu_choices,
                                    now=now,
                                    state=self._aihong_menu_ocr_state,
                                    choice_bounds=_aihong_choice_boxes(
                                        dialogue_menu_choices,
                                        extraction.boxes,
                                    ),
                                    choice_bounds_metadata=_extraction_choice_bounds_metadata(
                                        extraction
                                    ),
                                )
                            )
                        elif not dialogue_text_is_menu_status:
                            dialogue_emitted = bool(
                                self._consume_ocr_text(
                                    extraction.text,
                                    now=now,
                                    state=self._default_ocr_state,
                                    allow_choices=False,
                                )
                            )
                        if (
                            not dialogue_emitted
                            and not dialogue_text_is_menu_status
                            and not dialogue_menu_choices
                            and self._should_attempt_followup_confirm(
                                extraction.text,
                                state=self._default_ocr_state,
                            )
                        ):
                            followup_extraction = await self._capture_followup_text(
                                target,
                                profile,
                                backend_plan,
                            )
                            self._record_capture_completed(
                                now=self._time_fn(),
                                raw_text=followup_extraction.text,
                                image_hash=followup_extraction.capture_image_hash,
                            )
                            self._record_capture_geometry(followup_extraction)
                            self._capture_backend_kind = followup_extraction.capture_backend_kind
                            self._capture_backend_detail = followup_extraction.capture_backend_detail
                            active_backend = (
                                followup_extraction.backend
                                if followup_extraction.backend.kind
                                else active_backend
                            )
                            backend_detail_override = (
                                followup_extraction.backend_detail or backend_detail_override
                            )
                            result.warnings.extend(followup_extraction.warnings)
                            if followup_extraction.text and _looks_like_self_ui_text(followup_extraction.text):
                                guard_blocked = True
                                self._default_ocr_state.reset()
                                self._aihong_menu_ocr_state.reset()
                                result.warnings.append(
                                    "ocr_reader ignored text that looks like the N.E.K.O plugin UI"
                                )
                            else:
                                followup_now = self._time_fn()
                                dialogue_emitted = bool(
                                    self._consume_ocr_text(
                                        followup_extraction.text,
                                        now=followup_now,
                                        state=self._default_ocr_state,
                                        allow_choices=False,
                                    )
                                )
                                if dialogue_emitted:
                                    now = followup_now
                        emitted = dialogue_emitted
                        if dialogue_emitted:
                            self._aihong_dialogue_idle_polls = 0
                            self._aihong_menu_missing_polls = 0
                            if dialogue_menu_choices:
                                self._aihong_stage = _AIHONG_MENU_STAGE
                            else:
                                self._aihong_menu_ocr_state.reset()
                        else:
                            if dialogue_text_is_menu_status or dialogue_menu_choices:
                                self._aihong_dialogue_idle_polls = max(
                                    self._aihong_dialogue_idle_polls,
                                    1,
                                )
                            else:
                                self._aihong_dialogue_idle_polls += 1
                            if (
                                dialogue_text_is_menu_status
                                or dialogue_menu_choices
                                or self._aihong_dialogue_idle_polls >= 2
                            ):
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
                                )
                                self._record_capture_completed(
                                    now=self._time_fn(),
                                    raw_text=menu_extraction.text,
                                    image_hash=menu_extraction.capture_image_hash,
                                )
                                self._record_capture_geometry(menu_extraction)
                                self._capture_backend_kind = menu_extraction.capture_backend_kind
                                self._capture_backend_detail = menu_extraction.capture_backend_detail
                                active_backend = (
                                    menu_extraction.backend
                                    if menu_extraction.backend.kind
                                    else active_backend
                                )
                                backend_detail_override = (
                                    menu_extraction.backend_detail or backend_detail_override
                                )
                                result.warnings.extend(menu_extraction.warnings)
                                if menu_extraction.text and _looks_like_self_ui_text(menu_extraction.text):
                                    guard_blocked = True
                                    self._default_ocr_state.reset()
                                    self._aihong_menu_ocr_state.reset()
                                    result.warnings.append(
                                        "ocr_reader ignored text that looks like the N.E.K.O plugin UI"
                                    )
                                else:
                                    menu_result = self._consume_aihong_menu_stage_text(
                                        menu_extraction.text,
                                        now=now,
                                        boxes=menu_extraction.boxes,
                                        choice_bounds_metadata=_extraction_choice_bounds_metadata(
                                            menu_extraction
                                        ),
                                    )
                                    if menu_result.has_menu_candidate:
                                        self._aihong_menu_missing_polls = 0
                                        runtime_profile = menu_profile
                                        runtime_capture_profile_selection = menu_profile_selection
                                    if menu_result.emitted_kind == "line":
                                        emitted = True
                                        self._aihong_stage = _AIHONG_DIALOGUE_STAGE
                                        self._aihong_dialogue_idle_polls = 0
                                        self._aihong_menu_missing_polls = 0
                                        self._aihong_menu_ocr_state.reset()
                                        runtime_profile = menu_profile
                                        runtime_capture_profile_selection = menu_profile_selection
                                    elif menu_result.emitted_kind == "choices":
                                        emitted = True
                                        self._aihong_stage = _AIHONG_MENU_STAGE
                                        self._aihong_menu_missing_polls = 0
                                        runtime_profile = menu_profile
                                        runtime_capture_profile_selection = menu_profile_selection
                                    elif menu_result.has_menu_candidate:
                                        self._aihong_stage = _AIHONG_MENU_STAGE
                else:
                    emitted = bool(self._consume_ocr_text(extraction.text, now=now))
                    if (
                        not emitted
                        and self._should_attempt_followup_confirm(
                            extraction.text,
                            state=self._default_ocr_state,
                        )
                    ):
                        followup_extraction = await self._capture_followup_text(
                            target,
                            profile,
                            backend_plan,
                        )
                        self._record_capture_completed(
                            now=self._time_fn(),
                            raw_text=followup_extraction.text,
                            image_hash=followup_extraction.capture_image_hash,
                        )
                        self._record_capture_geometry(followup_extraction)
                        self._capture_backend_kind = followup_extraction.capture_backend_kind
                        self._capture_backend_detail = followup_extraction.capture_backend_detail
                        active_backend = (
                            followup_extraction.backend
                            if followup_extraction.backend.kind
                            else active_backend
                        )
                        backend_detail_override = (
                            followup_extraction.backend_detail or backend_detail_override
                        )
                        result.warnings.extend(followup_extraction.warnings)
                        if followup_extraction.text and _looks_like_self_ui_text(followup_extraction.text):
                            guard_blocked = True
                            self._default_ocr_state.reset()
                            self._aihong_menu_ocr_state.reset()
                            result.warnings.append(
                                "ocr_reader ignored text that looks like the N.E.K.O plugin UI"
                            )
                        else:
                            followup_now = self._time_fn()
                            emitted = bool(
                                self._consume_ocr_text(followup_extraction.text, now=followup_now)
                            )
                            if emitted:
                                now = followup_now
        except Exception as exc:
            self._logger.warning("ocr_reader capture/OCR failed: %s", exc)
            capture_error = True
            self._record_capture_error(now=now, error=exc)
            result.warnings.append(f"ocr_reader capture failed: {exc}")

        status = self._runtime.status
        detail = self._runtime.detail
        observed_or_stable_emitted = int(self._writer.last_seq or 0) > event_seq_before_capture

        if emitted:
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
            if self._writer.session_id and now - self._last_heartbeat_at >= float(
                self._config.ocr_reader_poll_interval_seconds
            ):
                if self._writer.emit_heartbeat(ts=utc_now_iso(now)):
                    result.should_rescan = True
                    self._last_heartbeat_at = now
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
        elif self._writer.session_id and now - self._last_heartbeat_at >= float(
            self._config.ocr_reader_poll_interval_seconds
        ):
            if self._writer.emit_heartbeat(ts=utc_now_iso(now)):
                result.should_rescan = True
                self._last_heartbeat_at = now
            if status == "starting":
                status = "active"
            if detail == "starting_capture":
                detail = "attached_no_text_yet"

        self._runtime = self._build_runtime(
            status=status,
            detail=detail,
            plan=backend_plan,
            active_backend=active_backend,
            backend_detail_override=backend_detail_override,
            target=target,
            capture_stage=(
                self._aihong_stage if aihong_two_stage_enabled else OCR_CAPTURE_PROFILE_STAGE_DEFAULT
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
        self._runtime.last_poll_started_at = utc_now_iso(poll_started_at)
        self._runtime.last_poll_completed_at = utc_now_iso(poll_completed_at)
        self._runtime.last_poll_duration_seconds = max(0.0, poll_completed_at - poll_started_at)
        self._runtime.last_poll_emitted_event = bool(emitted or observed_or_stable_emitted)
        result.runtime = self._runtime.to_dict()
        return result

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
            backend=RapidOcrBackend(
                install_target_dir_raw=self._config.rapidocr_install_target_dir,
                engine_type=self._config.rapidocr_engine_type,
                lang_type=self._config.rapidocr_lang_type,
                model_type=self._config.rapidocr_model_type,
                ocr_version=self._config.rapidocr_ocr_version,
            ),
            path=str(inspection.get("detected_path") or ""),
            model=str(
                inspection.get("selected_model")
                or f"{self._config.rapidocr_ocr_version}/{self._config.rapidocr_lang_type}/{self._config.rapidocr_model_type}"
            ),
            detail="selected_primary" if enabled and bool(inspection.get("installed")) else detail,
            available=enabled and bool(inspection.get("installed")),
        )

    def _resolve_backend_plan(self) -> SelectedOcrBackendPlan:
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
            return plan
        if selection == "tesseract":
            plan.primary = tesseract
            return plan
        if rapidocr.available:
            rapidocr.detail = "selected_primary"
            plan.primary = rapidocr
            if tesseract.available:
                tesseract.detail = "compatibility_fallback"
                plan.fallback = tesseract
            return plan
        if tesseract.available:
            tesseract.detail = f"auto_fallback_from_rapidocr:{rapidocr.detail}"
            plan.primary = tesseract
            return plan
        if rapidocr.available or bool(self._config.rapidocr_enabled):
            plan.primary = rapidocr
            if tesseract.kind:
                plan.fallback = tesseract
            return plan
        plan.primary = tesseract
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
            resolved_plan = SelectedOcrBackendPlan(
                primary=OcrBackendDescriptor(
                    kind=str(self._runtime.backend_kind or "custom"),
                    backend=self._ocr_backend,
                    detail=str(self._runtime.backend_detail or "custom_backend"),
                    available=True,
                )
            )
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
    ) -> OcrExtractionResult:
        frame = self._capture_backend.capture_frame(target, profile)
        capture_hash = self._capture_image_hash(frame)
        extraction = self._extract_text_from_image(frame, plan=plan)
        extraction.capture_image_hash = capture_hash
        frame_info = getattr(frame, "info", {}) if frame is not None else {}
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

        if self._manual_target.is_manual():
            for candidate in windows:
                if self._manual_target.matches_exact(candidate) or self._manual_target.matches_hwnd(candidate):
                    resolved_target = self._manual_target.resolved_for(candidate)
                    self._manual_target = resolved_target
                    selection.target = candidate
                    selection.selection_detail = "manual_target_exact"
                    selection.manual_target = resolved_target
                    selection.selected_by_manual = True
                    return selection
            for candidate in windows:
                if self._manual_target.matches_signature(candidate):
                    resolved_target = self._manual_target.resolved_for(candidate)
                    self._manual_target = resolved_target
                    selection.target = candidate
                    selection.selection_detail = "manual_target_rebound"
                    selection.manual_target = resolved_target
                    selection.selected_by_manual = True
                    return selection
            selection.selection_detail = "manual_target_unavailable_fallback_to_auto"

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
                    return selection
        if preferred_process_name:
            for candidate in windows:
                if str(candidate.process_name or "").strip().lower() == preferred_process_name:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "memory_reader_process"
                    return selection
        if self._attached_window is not None:
            for candidate in windows:
                if candidate.hwnd == self._attached_window.hwnd:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "attached_hwnd"
                    return selection
            if self._attached_window.pid:
                for candidate in windows:
                    if candidate.pid == self._attached_window.pid:
                        selection.target = candidate
                        if selection.selection_mode == "auto":
                            selection.selection_detail = "attached_pid"
                        return selection
        if self._has_locked_target():
            for candidate in windows:
                if self._locked_target.matches_exact(candidate) or self._locked_target.matches_hwnd(candidate):
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "locked_target_exact"
                    return selection
            for candidate in windows:
                if self._locked_target.matches_signature(candidate):
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "locked_target_rebound"
                    return selection
            if selection.selection_mode == "auto":
                selection.selection_detail = "locked_target_unavailable"
            return selection
        foreground_hwnd = _foreground_window_handle()
        if foreground_hwnd:
            for candidate in windows:
                if candidate.hwnd == foreground_hwnd:
                    if not _is_confident_auto_window(candidate):
                        if selection.selection_mode == "auto":
                            selection.selection_detail = "foreground_window_needs_manual_confirmation"
                        return selection
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "foreground_window"
                    return selection
        if selection.selection_mode == "auto":
            selection.selection_detail = "auto_detect_needs_manual_fallback"
        return selection

    def _consume_ocr_text(
        self,
        raw_text: str,
        *,
        now: float,
        state: _StableOcrTextState | None = None,
        allow_choices: bool = True,
        allow_plain_text_choices: bool = False,
    ) -> bool:
        tracker = state or self._default_ocr_state
        lines = _stripped_ocr_lines(raw_text)
        if allow_choices:
            choices = _coerce_choice_lines(lines, allow_plain_text=allow_plain_text_choices)
            if choices:
                return self._emit_choices_from_candidates(choices, now=now, state=tracker)
        return self._emit_line_from_ocr_text(raw_text, now=now, state=tracker)

    async def _end_session_if_needed(self, now: float) -> None:
        if self._writer.session_id:
            self._writer.end_session(ts=utc_now_iso(now))
            self._attached_window = None
            self._reset_default_ocr_state()
            self._reset_aihong_menu_state()
