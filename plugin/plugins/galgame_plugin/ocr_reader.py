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
from typing import Any, Callable, Protocol
from uuid import uuid4

from .models import (
    DATA_SOURCE_OCR_READER,
    DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_TOP_RATIO,
    GalgameConfig,
    OCR_CAPTURE_PROFILE_RATIO_KEYS,
    OCR_CAPTURE_PROFILE_STAGE_DEFAULT,
    OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
    OCR_CAPTURE_PROFILE_STAGE_MENU,
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
_HELPER_CLASS_NAMES = {
    "Shell_TrayWnd",
    "Windows.UI.Core.CoreWindow",
    "ApplicationFrameWindow",
    "Windows.UI.Composition.DesktopWindowContentBridge",
}
_SELF_UI_GUARD_SUBSTRINGS = (
    "rapidocr",
    "tesseract",
    "ocr compatibility fallback",
    "install queued task",
    "plugin manager",
    "galgame plugin",
    "n.e.k.o",
)
_AIHONG_PROCESS_NAMES = frozenset({"thelamentinggeese.exe"})
_AIHONG_TITLE_SUBSTRINGS = ("哀鸿", "aihong")
_AIHONG_DIALOGUE_CAPTURE_PROFILE_PRESET = {
    "left_inset_ratio": 0.05,
    "right_inset_ratio": 0.24,
    "top_ratio": 0.73,
    "bottom_inset_ratio": 0.10,
}
_AIHONG_MENU_CAPTURE_PROFILE_PRESET = {
    "left_inset_ratio": 0.20,
    "right_inset_ratio": 0.20,
    "top_ratio": 0.40,
    "bottom_inset_ratio": 0.34,
}
_AIHONG_DIALOGUE_STAGE = OCR_CAPTURE_PROFILE_STAGE_DIALOGUE
_AIHONG_MENU_STAGE = OCR_CAPTURE_PROFILE_STAGE_MENU
_AIHONG_MENU_MAX_LINES = 4
_AIHONG_MENU_MIN_SIGNIFICANT_CHARS = 2
_AIHONG_MENU_MAX_SIGNIFICANT_CHARS = 10
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


def _coerce_aihong_menu_choices(lines: list[str]) -> list[str]:
    choices = _coerce_choice_lines(lines, allow_plain_text=True)
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
    top: float
    bottom: float
    height: float


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
                top=top,
                bottom=bottom,
                height=max(bottom - top, 1.0),
            )
        )
    return tokens


def _rapidocr_text_from_output(raw_output: Any) -> str:
    tokens = _rapidocr_tokens_from_output(raw_output)
    if not tokens:
        return ""
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
    scores: list[float] = []
    lines.sort(key=lambda line: (min(item.top for item in line), min(item.left for item in line)))
    for line in lines:
        line.sort(key=lambda item: item.left)
        scores.extend(item.score for item in line)
        rendered_lines.append(_join_ocr_segments([item.text for item in line]))
    text = "\n".join(line for line in rendered_lines if line)
    normalized = normalize_text(text)
    if not normalized:
        return ""
    average_score = (sum(scores) / len(scores)) if scores else 0.0
    if _significant_char_count(normalized) < 4 and average_score < 0.55:
        return ""
    return text


@dataclass(slots=True)
class DetectedGameWindow:
    hwnd: int = 0
    title: str = ""
    process_name: str = ""
    pid: int = 0
    class_name: str = ""
    exe_path: str = ""
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

    def to_dict(self, *, is_attached: bool = False, is_manual_target: bool = False) -> dict[str, Any]:
        return {
            "window_key": self.window_key,
            "title": self.title,
            "process_name": self.process_name,
            "pid": self.pid,
            "hwnd": self.hwnd,
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


def _uses_manual_capture_profile(
    profiles: dict[str, dict[str, OcrCaptureProfile]],
    target: DetectedGameWindow,
) -> bool:
    process_name = str(target.process_name or "").strip().lower()
    if not process_name:
        return False
    return any(str(name or "").strip().lower() == process_name for name in profiles)


def _lookup_capture_profile(
    profiles: dict[str, dict[str, OcrCaptureProfile]],
    target: DetectedGameWindow,
    *,
    stage: str,
) -> OcrCaptureProfile | None:
    process_name = str(target.process_name or "").strip()
    if not process_name:
        return None
    profile_map = profiles.get(process_name)
    if profile_map is not None:
        profile = profile_map.get(stage) or profile_map.get(OCR_CAPTURE_PROFILE_STAGE_DEFAULT)
        if profile is not None:
            return profile
    lowered = process_name.lower()
    for configured_name, configured_profile_map in profiles.items():
        if str(configured_name or "").strip().lower() == lowered:
            return configured_profile_map.get(stage) or configured_profile_map.get(
                OCR_CAPTURE_PROFILE_STAGE_DEFAULT
            )
    return None


def _parse_configured_capture_profiles(
    profiles: dict[str, dict[str, Any]],
    logger,
) -> dict[str, dict[str, OcrCaptureProfile]]:
    parsed_profiles: dict[str, dict[str, OcrCaptureProfile]] = {}
    for process_name, profile_value in profiles.items():
        normalized_process_name = str(process_name or "").strip()
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
        if stage_profiles:
            parsed_profiles[normalized_process_name] = stage_profiles
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
        if self.pid > 0 and candidate.pid != self.pid:
            return False
        if self.process_name and self.process_name.strip().lower() != candidate.process_name.strip().lower():
            return False
        if self.normalized_title and self.normalized_title != candidate.normalized_title:
            return False
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
    game_id: str = ""
    session_id: str = ""
    last_seq: int = 0
    last_event_ts: str = ""
    capture_stage: str = ""
    capture_profile: dict[str, float] = field(default_factory=dict)
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
    manual_target: dict[str, Any] = field(default_factory=dict)
    candidate_count: int = 0
    excluded_candidate_count: int = 0
    last_exclude_reason: str = ""

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
            "capture_stage": self.capture_stage,
            "capture_profile": dict(self.capture_profile),
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
            "manual_target": dict(self.manual_target),
            "candidate_count": self.candidate_count,
            "excluded_candidate_count": self.excluded_candidate_count,
            "last_exclude_reason": self.last_exclude_reason,
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
            area=max(0, area),
            is_foreground=hwnd == foreground_hwnd,
            score=float(max(area, 0)),
        )
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
        self._custom_ocr_backend = ocr_backend is not None
        self._writer = writer or OcrReaderBridgeWriter(
            bridge_root=config.bridge_root,
            time_fn=self._time_fn,
        )
        self._runtime = OcrReaderRuntime(enabled=config.ocr_reader_enabled)
        self._capture_profiles: dict[str, dict[str, OcrCaptureProfile]] = {}
        self._last_memory_reader_text_at = 0.0
        self._last_heartbeat_at = 0.0
        self._attached_window: DetectedGameWindow | None = None
        self._default_ocr_state = _StableOcrTextState()
        self._aihong_menu_ocr_state = _StableOcrTextState()
        self._aihong_stage = _AIHONG_DIALOGUE_STAGE
        self._aihong_dialogue_idle_polls = 0
        self._aihong_menu_missing_polls = 0
        self._manual_target = OcrWindowTarget()
        self._last_detected_windows: list[DetectedGameWindow] = []
        self._last_eligible_windows: list[DetectedGameWindow] = []
        self._last_excluded_windows: list[DetectedGameWindow] = []
        self._last_selection = WindowSelectionResult(manual_target=self._manual_target)

    def update_config(self, config: GalgameConfig) -> None:
        self._config = config
        self._runtime.enabled = config.ocr_reader_enabled
        if self._writer.bridge_root != config.bridge_root:
            self._writer = OcrReaderBridgeWriter(
                bridge_root=config.bridge_root,
                time_fn=self._time_fn,
            )

    def update_capture_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        self._capture_profiles = _parse_configured_capture_profiles(profiles, self._logger)

    def update_window_target(self, target: dict[str, Any] | None) -> None:
        self._manual_target = OcrWindowTarget.from_dict(target)
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

    def _reset_default_ocr_state(self) -> None:
        self._default_ocr_state.reset()

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
    def _stabilize_text_key(text: str, *, state: _StableOcrTextState) -> bool:
        cleaned = normalize_text(text)
        if not cleaned:
            return False
        if cleaned == state.last_raw_text:
            state.repeat_count += 1
        else:
            state.repeat_count = 1
            state.last_raw_text = cleaned
        if state.repeat_count < 2:
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
        tracker = state or self._default_ocr_state
        if not self._stabilize_text_key(raw_text, state=tracker):
            return False
        return self._writer.emit_line(raw_text, ts=utc_now_iso(now))

    def _emit_choices_from_candidates(
        self,
        choices: list[str],
        *,
        now: float,
        state: _StableOcrTextState | None = None,
    ) -> bool:
        tracker = state or self._default_ocr_state
        if not self._stabilize_text_key(_canonical_choice_candidate_text(choices), state=tracker):
            return False
        return self._writer.emit_choices(choices, ts=utc_now_iso(now))

    def _consume_aihong_menu_stage_text(self, raw_text: str, *, now: float) -> _MenuConsumeResult:
        lines = _stripped_ocr_lines(raw_text)
        choices = _coerce_aihong_menu_choices(lines)
        if choices:
            return _MenuConsumeResult(
                emitted_kind="choices"
                if self._emit_choices_from_candidates(
                    choices,
                    now=now,
                    state=self._aihong_menu_ocr_state,
                )
                else "",
                has_menu_candidate=True,
            )
        return _MenuConsumeResult(
            emitted_kind="line" if self._emit_line_from_ocr_text(raw_text, now=now) else "",
            has_menu_candidate=False,
        )

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
            candidate.is_foreground = candidate.hwnd == foreground_hwnd if candidate.hwnd else bool(candidate.is_foreground)
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
        result = OcrReaderTickResult(runtime=self._runtime.to_dict())

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
        profile = self._capture_profile_for_target(target, stage=profile_stage)

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
            startup_profile = self._capture_profile_for_target(
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
                capture_profile=startup_profile.to_dict(),
                selection=selection,
                game_id=self._writer.game_id,
                session_id=self._writer.session_id,
                last_seq=self._writer.last_seq,
                last_event_ts=self._writer.last_event_ts,
            )
            result.runtime = self._runtime.to_dict()
            return result

        if self._attached_window is not None:
            self._attached_window = target

        emitted = False
        guard_blocked = False
        active_backend = backend_plan.primary
        backend_detail_override = ""
        runtime_profile = profile
        try:
            extraction = await asyncio.to_thread(
                self._capture_and_extract_text,
                target,
                profile,
                backend_plan,
            )
            active_backend = extraction.backend if extraction.backend.kind else backend_plan.primary
            backend_detail_override = extraction.backend_detail
            result.warnings.extend(extraction.warnings)
            if extraction.text and not selection.selected_by_manual and _looks_like_self_ui_text(extraction.text):
                guard_blocked = True
                result.warnings.append("ocr_reader ignored text that looks like the N.E.K.O plugin UI")
            else:
                if aihong_two_stage_enabled:
                    if self._aihong_stage == _AIHONG_MENU_STAGE:
                        menu_result = self._consume_aihong_menu_stage_text(extraction.text, now=now)
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
                            if self._aihong_menu_missing_polls >= 2:
                                self._reset_aihong_menu_state()
                    else:
                        dialogue_emitted = bool(
                            self._consume_ocr_text(
                                extraction.text,
                                now=now,
                                state=self._default_ocr_state,
                                allow_choices=False,
                            )
                        )
                        emitted = dialogue_emitted
                        if dialogue_emitted:
                            self._aihong_dialogue_idle_polls = 0
                            self._aihong_menu_missing_polls = 0
                            self._aihong_menu_ocr_state.reset()
                        else:
                            self._aihong_dialogue_idle_polls += 1
                            if self._aihong_dialogue_idle_polls >= 2:
                                menu_profile = self._capture_profile_for_target(
                                    target,
                                    stage=_AIHONG_MENU_STAGE,
                                )
                                menu_extraction = await asyncio.to_thread(
                                    self._capture_and_extract_text,
                                    target,
                                    menu_profile,
                                    backend_plan,
                                )
                                active_backend = (
                                    menu_extraction.backend
                                    if menu_extraction.backend.kind
                                    else active_backend
                                )
                                backend_detail_override = (
                                    menu_extraction.backend_detail or backend_detail_override
                                )
                                result.warnings.extend(menu_extraction.warnings)
                                if (
                                    menu_extraction.text
                                    and not selection.selected_by_manual
                                    and _looks_like_self_ui_text(menu_extraction.text)
                                ):
                                    result.warnings.append(
                                        "ocr_reader ignored text that looks like the N.E.K.O plugin UI"
                                    )
                                else:
                                    menu_result = self._consume_aihong_menu_stage_text(
                                        menu_extraction.text,
                                        now=now,
                                    )
                                    if menu_result.has_menu_candidate:
                                        self._aihong_menu_missing_polls = 0
                                        runtime_profile = menu_profile
                                    if menu_result.emitted_kind == "line":
                                        emitted = True
                                        self._aihong_stage = _AIHONG_DIALOGUE_STAGE
                                        self._aihong_dialogue_idle_polls = 0
                                        self._aihong_menu_missing_polls = 0
                                        self._aihong_menu_ocr_state.reset()
                                        runtime_profile = menu_profile
                                    elif menu_result.emitted_kind == "choices":
                                        emitted = True
                                        self._aihong_stage = _AIHONG_MENU_STAGE
                                        self._aihong_menu_missing_polls = 0
                                        runtime_profile = menu_profile
                else:
                    emitted = bool(self._consume_ocr_text(extraction.text, now=now))
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
        elif guard_blocked:
            if status == "starting":
                status = "active"
            detail = "self_ui_guard_blocked"
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
            selection=selection,
            game_id=self._writer.game_id,
            session_id=self._writer.session_id,
            last_seq=self._writer.last_seq,
            last_event_ts=self._writer.last_event_ts,
        )
        result.runtime = self._runtime.to_dict()
        return result

    def _configured_backend_selection(self) -> str:
        selection = str(self._config.ocr_reader_backend_selection or "auto").strip().lower()
        if selection in {"auto", "rapidocr", "tesseract"}:
            return selection
        return "auto"

    def _capture_profile_for_target(
        self,
        target: DetectedGameWindow,
        *,
        stage: str = _AIHONG_DIALOGUE_STAGE,
    ) -> OcrCaptureProfile:
        configured_profile = _lookup_capture_profile(
            self._capture_profiles,
            target,
            stage=stage,
        )
        if configured_profile is not None:
            return configured_profile

        builtin_profile = _builtin_capture_profile_for_target_stage(target, stage=stage)
        if builtin_profile is not None:
            return builtin_profile

        return OcrCaptureProfile(
            left_inset_ratio=self._config.ocr_reader_left_inset_ratio,
            right_inset_ratio=self._config.ocr_reader_right_inset_ratio,
            top_ratio=self._config.ocr_reader_top_ratio,
            bottom_inset_ratio=self._config.ocr_reader_bottom_inset_ratio,
        )

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
        if str(plan.tesseract_inspection.get("detail") or "") == "missing_languages":
            return "missing_languages"
        return "missing_tesseract"

    def _backend_unavailable_warnings(self, plan: SelectedOcrBackendPlan) -> list[str]:
        warnings: list[str] = []
        if plan.selection == "rapidocr":
            warnings.append(f"ocr_reader RapidOCR is unavailable: {plan.primary.detail or 'missing'}")
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
            game_id=str(game_id or self._writer.game_id or self._runtime.game_id),
            session_id=str(session_id or self._writer.session_id or self._runtime.session_id),
            last_seq=resolved_last_seq,
            last_event_ts=str(last_event_ts or self._writer.last_event_ts or self._runtime.last_event_ts),
            capture_stage=str(capture_stage or self._runtime.capture_stage),
            capture_profile=dict(capture_profile or self._runtime.capture_profile),
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
            manual_target=manual_target,
            candidate_count=max(0, int(selection_state.candidate_count or 0)),
            excluded_candidate_count=max(0, int(selection_state.excluded_candidate_count or 0)),
            last_exclude_reason=str(selection_state.last_exclude_reason or self._runtime.last_exclude_reason),
        )

    def _capture_and_extract_text(
        self,
        target: DetectedGameWindow,
        profile: OcrCaptureProfile,
        plan: SelectedOcrBackendPlan,
    ) -> OcrExtractionResult:
        frame = self._capture_backend.capture_frame(target, profile)
        if self._custom_ocr_backend:
            return OcrExtractionResult(
                text=self._ocr_backend.extract_text(frame),
                backend=plan.primary,
                backend_detail=plan.primary.detail,
            )
        descriptors = [plan.primary]
        if plan.fallback.available:
            descriptors.append(plan.fallback)
        warnings: list[str] = []
        last_error: Exception | None = None
        for index, descriptor in enumerate(descriptors):
            if descriptor.backend is None:
                continue
            try:
                return OcrExtractionResult(
                    text=descriptor.backend.extract_text(frame),
                    backend=descriptor,
                    backend_detail=(
                        "fallback_after_runtime_error"
                        if index > 0
                        else (descriptor.detail or "selected_primary")
                    ),
                    warnings=warnings,
                )
            except Exception as exc:
                last_error = exc
                warning = f"ocr_reader {descriptor.kind} failed: {exc}"
                warnings.append(warning)
                self._logger.warning("ocr_reader backend %s failed: %s", descriptor.kind, exc)
        if last_error is not None:
            raise last_error
        return OcrExtractionResult(backend=plan.primary, warnings=warnings)

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
                    selection.target = candidate
                    selection.selection_detail = "manual_target_exact"
                    selection.manual_target = self._manual_target.resolved_for(candidate)
                    selection.selected_by_manual = True
                    return selection
            for candidate in windows:
                if self._manual_target.matches_signature(candidate):
                    selection.target = candidate
                    selection.selection_detail = "manual_target_rebound"
                    selection.manual_target = self._manual_target.resolved_for(candidate)
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
        foreground_hwnd = _foreground_window_handle()
        if foreground_hwnd:
            for candidate in windows:
                if candidate.hwnd == foreground_hwnd:
                    selection.target = candidate
                    if selection.selection_mode == "auto":
                        selection.selection_detail = "foreground_window"
                    return selection
        selection.target = windows[0]
        if selection.selection_mode == "auto":
            selection.selection_detail = "scored_candidate"
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
