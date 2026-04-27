from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from .models import (
    DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO,
    DEFAULT_OCR_CAPTURE_TOP_RATIO,
    DATA_SOURCE_BRIDGE_SDK,
    DATA_SOURCE_MEMORY_READER,
    DATA_SOURCE_OCR_READER,
    GalgameConfig,
    MODE_CHOICE_ADVISOR,
    MODE_COMPANION,
    MODES,
    MODE_SILENT,
    STATE_ACTIVE,
    STATE_DISCONNECTED,
    STATE_ERROR,
    STATE_IDLE,
    STATE_STALE,
    SessionCandidate,
    json_copy,
    make_error,
    sanitize_choice,
    sanitize_metadata,
    sanitize_save_context,
    sanitize_snapshot_state,
)
from .dxcam_support import inspect_dxcam_installation
from .reader import expand_bridge_root, normalize_text, read_session_json
from .rapidocr_support import (
    DEFAULT_RAPIDOCR_ENGINE_TYPE,
    DEFAULT_RAPIDOCR_LANG_TYPE,
    DEFAULT_RAPIDOCR_MODEL_TYPE,
    DEFAULT_RAPIDOCR_OCR_VERSION,
    inspect_rapidocr_installation,
)
from .tesseract_support import inspect_tesseract_installation
from .textractor_support import (
    DEFAULT_TEXTRACTOR_RELEASE_API_URL,
    inspect_textractor_installation,
)

_OCR_OVERLAY_TEXT_GUARD_SUBSTRINGS = (
    ".agent",
    ".codex",
    ".codex_tmp",
    ".codex_pytest_tmp",
    "__pycache__",
    "-pycache_",
    "codex_tmp",
    "documents\\code\\n.e.k.o",
    "galgame plugin",
    "n.e.k.o",
    "plugin manager",
    "plugin.plugins.galgame_plugin",
    "uv run python",
    "launcher.py",
    "powershell",
    "ps c:",
    "插件设置",
    "ocr 目标窗口",
    "截图校准",
)
_CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_DIALOGUE_PUNCTUATION_RE = re.compile(r"[。！？!?…，,、：:]")
_NON_DIALOGUE_CONTEXT_TOKENS = (
    "agent",
    "capture_failed",
    "context_state=",
    "dxcam:",
    "galgame_",
    "gateway_unavailable",
    "http://",
    "https://",
    "last_error=",
    "ocr_context_unavailable",
    "plugin/",
    "plugin\\",
    "powershell",
    "status=",
    "stability",
    "当前快照",
    "场景 id",
    "场景id",
    "会话 id",
    "会话id",
    "游戏 id",
    "游戏id",
    "菜单是否打开",
    "台词 id",
    "台词id",
    "路线 id",
    "路线id",
    "快照时间",
    "是否过期",
    "退出全屏",
    "收起",
    "全屏",
    "ocr 诊断",
    "recent raw ocr",
    "最近 raw ocr",
)


def _looks_like_ocr_overlay_text(text: object) -> bool:
    normalized = normalize_text(str(text or "")).strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _OCR_OVERLAY_TEXT_GUARD_SUBSTRINGS)


def _significant_char_count(text: object) -> int:
    return sum(1 for ch in str(text or "") if not ch.isspace())


def _looks_like_game_dialogue_context_line(line: dict[str, Any]) -> bool:
    if not isinstance(line, dict) or bool(line.get("is_diagnostic")):
        return False
    text = normalize_text(str(line.get("text") or "")).strip()
    if not text or _looks_like_ocr_overlay_text(text):
        return False
    lowered = text.lower()
    if any(token in lowered for token in _NON_DIALOGUE_CONTEXT_TOKENS):
        return False
    if text.startswith("{") or text.startswith("[") or "{" in text and "}" in text:
        return False
    significant_chars = _significant_char_count(text)
    if significant_chars < 2 or significant_chars > 220:
        return False
    has_cjk_or_kana = bool(_CJK_OR_KANA_RE.search(text))
    has_dialogue_punctuation = bool(_DIALOGUE_PUNCTUATION_RE.search(text))
    has_speaker = bool(str(line.get("speaker") or "").strip())
    return has_cjk_or_kana or has_dialogue_punctuation or has_speaker


def _payload_is_game_dialogue_line(payload_obj: dict[str, Any], *, ts: str = "") -> bool:
    return _looks_like_game_dialogue_context_line(
        _line_history_entry(payload_obj, ts=ts, stability=str(payload_obj.get("stability") or ""))
    )


def _coerce_float(value: object, default: float, *, minimum: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return default
    return parsed


def _coerce_int(value: object, default: int, *, minimum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return default
    return parsed


def _coerce_bool(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _coerce_ocr_backend_selection(value: object, default: str = "auto") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"auto", "rapidocr", "tesseract"}:
        return normalized
    return default


def _coerce_ocr_capture_backend(value: object, default: str = "auto") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"auto", "dxcam", "imagegrab", "printwindow"}:
        return normalized
    return default


def _default_bridge_root_raw() -> str:
    if sys.platform.startswith("win"):
        return "%LOCALAPPDATA%/N.E.K.O/galgame-bridge"
    if sys.platform == "darwin":
        return "~/Library/Application Support/N.E.K.O/galgame-bridge"
    xdg_data_home = str(os.getenv("XDG_DATA_HOME") or "").strip()
    if xdg_data_home:
        return f"{xdg_data_home}/N.E.K.O/galgame-bridge"
    return "~/.local/share/N.E.K.O/galgame-bridge"


def _default_memory_reader_enabled() -> bool:
    return sys.platform.startswith("win")


def _default_ocr_reader_enabled() -> bool:
    return sys.platform.startswith("win")


def build_config(raw_config: dict[str, Any]) -> GalgameConfig:
    galgame = raw_config.get("galgame")
    llm = raw_config.get("llm")
    memory_reader = raw_config.get("memory_reader")

    galgame_obj = galgame if isinstance(galgame, dict) else {}
    llm_obj = llm if isinstance(llm, dict) else {}
    memory_reader_obj = memory_reader if isinstance(memory_reader, dict) else {}
    ocr_reader = raw_config.get("ocr_reader")
    ocr_reader_obj = ocr_reader if isinstance(ocr_reader, dict) else {}
    rapidocr = raw_config.get("rapidocr")
    rapidocr_obj = rapidocr if isinstance(rapidocr, dict) else {}

    default_mode_obj = galgame_obj.get("default_mode")
    default_mode = (
        default_mode_obj
        if isinstance(default_mode_obj, str) and default_mode_obj in MODES
        else MODE_COMPANION
    )
    bridge_root_value = galgame_obj.get("bridge_root")
    bridge_root_raw = str(bridge_root_value).strip() if bridge_root_value is not None else ""
    if not bridge_root_raw:
        bridge_root_raw = _default_bridge_root_raw()

    return GalgameConfig(
        bridge_root=expand_bridge_root(bridge_root_raw),
        active_poll_interval_seconds=_coerce_float(
            galgame_obj.get("active_poll_interval_seconds"), 1.0, minimum=0.1
        ),
        idle_poll_interval_seconds=_coerce_float(
            galgame_obj.get("idle_poll_interval_seconds"), 3.0, minimum=0.1
        ),
        stale_after_seconds=_coerce_float(
            galgame_obj.get("stale_after_seconds"), 15.0, minimum=0.1
        ),
        history_events_limit=_coerce_int(
            galgame_obj.get("history_events_limit"), 500, minimum=1
        ),
        history_lines_limit=_coerce_int(
            galgame_obj.get("history_lines_limit"), 200, minimum=1
        ),
        history_choices_limit=_coerce_int(
            galgame_obj.get("history_choices_limit"), 50, minimum=1
        ),
        dedupe_window_limit=_coerce_int(
            galgame_obj.get("dedupe_window_limit"), 64, minimum=1
        ),
        warmup_replay_bytes_limit=_coerce_int(
            galgame_obj.get("warmup_replay_bytes_limit"), 65536, minimum=1
        ),
        warmup_replay_events_limit=_coerce_int(
            galgame_obj.get("warmup_replay_events_limit"), 50, minimum=1
        ),
        default_mode=default_mode,
        push_notifications=bool(galgame_obj.get("push_notifications", True)),
        llm_call_timeout_seconds=_coerce_float(
            llm_obj.get("llm_call_timeout_seconds"), 15.0, minimum=0.1
        ),
        llm_max_in_flight=_coerce_int(llm_obj.get("llm_max_in_flight"), 2, minimum=1),
        llm_request_cache_ttl_seconds=_coerce_float(
            llm_obj.get("llm_request_cache_ttl_seconds"), 2.0, minimum=0.0
        ),
        llm_target_entry_ref=str(llm_obj.get("target_entry_ref") or "").strip(),
        memory_reader_enabled=_coerce_bool(
            memory_reader_obj.get("enabled"),
            _default_memory_reader_enabled(),
        ),
        memory_reader_textractor_path=str(memory_reader_obj.get("textractor_path") or ""),
        memory_reader_install_release_api_url=str(
            memory_reader_obj.get("install_release_api_url")
            or DEFAULT_TEXTRACTOR_RELEASE_API_URL
        ).strip(),
        memory_reader_install_target_dir=str(
            memory_reader_obj.get("install_target_dir") or ""
        ).strip(),
        memory_reader_install_timeout_seconds=_coerce_float(
            memory_reader_obj.get("install_timeout_seconds"), 60.0, minimum=1.0
        ),
        memory_reader_auto_detect=bool(memory_reader_obj.get("auto_detect", True)),
        memory_reader_hook_codes=list(
            memory_reader_obj.get("hook_codes")
            or []
        ),
        memory_reader_poll_interval_seconds=_coerce_float(
            memory_reader_obj.get("poll_interval_seconds"), 1.0, minimum=0.1
        ),
        ocr_reader_enabled=_coerce_bool(
            ocr_reader_obj.get("enabled"),
            _default_ocr_reader_enabled(),
        ),
        ocr_reader_backend_selection=_coerce_ocr_backend_selection(
            ocr_reader_obj.get("backend_selection"),
            "auto",
        ),
        ocr_reader_capture_backend=_coerce_ocr_capture_backend(
            ocr_reader_obj.get("capture_backend"),
            "auto",
        ),
        ocr_reader_tesseract_path=str(ocr_reader_obj.get("tesseract_path") or ""),
        ocr_reader_install_manifest_url=str(
            ocr_reader_obj.get("install_manifest_url") or ""
        ).strip(),
        ocr_reader_install_target_dir=str(
            ocr_reader_obj.get("install_target_dir") or ""
        ).strip(),
        ocr_reader_install_timeout_seconds=_coerce_float(
            ocr_reader_obj.get("install_timeout_seconds"), 300.0, minimum=1.0
        ),
        ocr_reader_poll_interval_seconds=_coerce_float(
            ocr_reader_obj.get("poll_interval_seconds"), 2.0, minimum=0.1
        ),
        ocr_reader_no_text_takeover_after_seconds=_coerce_float(
            ocr_reader_obj.get("no_text_takeover_after_seconds"), 30.0, minimum=0.0
        ),
        ocr_reader_languages=str(ocr_reader_obj.get("languages") or "chi_sim+jpn+eng"),
        ocr_reader_left_inset_ratio=_coerce_float(
            ocr_reader_obj.get("left_inset_ratio"),
            DEFAULT_OCR_CAPTURE_LEFT_INSET_RATIO,
            minimum=0.0,
        ),
        ocr_reader_right_inset_ratio=_coerce_float(
            ocr_reader_obj.get("right_inset_ratio"),
            DEFAULT_OCR_CAPTURE_RIGHT_INSET_RATIO,
            minimum=0.0,
        ),
        ocr_reader_top_ratio=_coerce_float(
            ocr_reader_obj.get("top_ratio"),
            DEFAULT_OCR_CAPTURE_TOP_RATIO,
            minimum=0.0,
        ),
        ocr_reader_bottom_inset_ratio=_coerce_float(
            ocr_reader_obj.get("bottom_inset_ratio"),
            DEFAULT_OCR_CAPTURE_BOTTOM_INSET_RATIO,
            minimum=0.0,
        ),
        rapidocr_enabled=_coerce_bool(
            rapidocr_obj.get("enabled"),
            _default_ocr_reader_enabled(),
        ),
        rapidocr_install_manifest_url=str(
            rapidocr_obj.get("install_manifest_url") or ""
        ).strip(),
        rapidocr_install_target_dir=str(
            rapidocr_obj.get("install_target_dir") or ""
        ).strip(),
        rapidocr_install_timeout_seconds=_coerce_float(
            rapidocr_obj.get("install_timeout_seconds"), 180.0, minimum=1.0
        ),
        rapidocr_engine_type=str(
            rapidocr_obj.get("engine_type") or DEFAULT_RAPIDOCR_ENGINE_TYPE
        ).strip()
        or DEFAULT_RAPIDOCR_ENGINE_TYPE,
        rapidocr_lang_type=str(
            rapidocr_obj.get("lang_type") or DEFAULT_RAPIDOCR_LANG_TYPE
        ).strip()
        or DEFAULT_RAPIDOCR_LANG_TYPE,
        rapidocr_model_type=str(
            rapidocr_obj.get("model_type") or DEFAULT_RAPIDOCR_MODEL_TYPE
        ).strip()
        or DEFAULT_RAPIDOCR_MODEL_TYPE,
        rapidocr_ocr_version=str(
            rapidocr_obj.get("ocr_version") or DEFAULT_RAPIDOCR_OCR_VERSION
        ).strip()
        or DEFAULT_RAPIDOCR_OCR_VERSION,
    )


def scan_session_candidates(bridge_root: Path) -> tuple[list[str], dict[str, SessionCandidate], list[str]]:
    available_game_ids: list[str] = []
    candidates: dict[str, SessionCandidate] = {}
    warnings: list[str] = []

    if not bridge_root.exists():
        return available_game_ids, candidates, warnings

    for child in sorted(bridge_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        game_id = child.name
        available_game_ids.append(game_id)
        session_path = child / "session.json"
        events_path = child / "events.jsonl"
        session_result = read_session_json(session_path)
        if session_result.error:
            warnings.append(f"{game_id}: {session_result.error}")
        if not session_result.session:
            continue
        session = dict(session_result.session)
        if not session.get("game_id"):
            session["game_id"] = game_id
        data_source = infer_session_data_source(session)
        candidates[game_id] = SessionCandidate(
            game_id=game_id,
            session_path=session_path,
            events_path=events_path,
            session=session,
            data_source=data_source,
        )

    return available_game_ids, candidates, warnings


def infer_session_data_source(session: dict[str, Any]) -> str:
    metadata = session.get("metadata")
    metadata_obj = metadata if isinstance(metadata, dict) else {}
    if str(metadata_obj.get("source") or "") == DATA_SOURCE_MEMORY_READER:
        return DATA_SOURCE_MEMORY_READER
    if str(session.get("bridge_sdk_version") or "").startswith("memory-reader-"):
        return DATA_SOURCE_MEMORY_READER
    if str(session.get("game_id") or "").startswith(("mem:", "mem-")):
        return DATA_SOURCE_MEMORY_READER
    if str(metadata_obj.get("source") or "") == DATA_SOURCE_OCR_READER:
        return DATA_SOURCE_OCR_READER
    if str(session.get("bridge_sdk_version") or "").startswith("ocr-reader-"):
        return DATA_SOURCE_OCR_READER
    if str(session.get("game_id") or "").startswith(("ocr:", "ocr-")):
        return DATA_SOURCE_OCR_READER
    return DATA_SOURCE_BRIDGE_SDK


def filter_memory_reader_candidates(
    available_game_ids: list[str],
    candidates: dict[str, SessionCandidate],
    *,
    runtime: dict[str, Any],
) -> tuple[list[str], dict[str, SessionCandidate]]:
    runtime_status = str(runtime.get("status") or "")
    runtime_game_id = str(runtime.get("game_id") or "")
    memory_reader_live = runtime_status in {"attaching", "active"} and bool(runtime_game_id)
    filtered_candidates: dict[str, SessionCandidate] = {}
    filtered_out: set[str] = set()
    for game_id, candidate in candidates.items():
        if candidate.data_source != DATA_SOURCE_MEMORY_READER:
            filtered_candidates[game_id] = candidate
            continue
        if memory_reader_live and candidate.game_id == runtime_game_id:
            filtered_candidates[game_id] = candidate
            continue
        filtered_out.add(game_id)
    filtered_ids = [game_id for game_id in available_game_ids if game_id not in filtered_out]
    return filtered_ids, filtered_candidates


def filter_ocr_reader_candidates(
    available_game_ids: list[str],
    candidates: dict[str, SessionCandidate],
    *,
    runtime: dict[str, Any],
) -> tuple[list[str], dict[str, SessionCandidate]]:
    runtime_status = str(runtime.get("status") or "")
    runtime_game_id = str(runtime.get("game_id") or "")
    ocr_reader_live = runtime_status in {"starting", "active"} and bool(runtime_game_id)
    filtered_candidates: dict[str, SessionCandidate] = {}
    filtered_out: set[str] = set()
    for game_id, candidate in candidates.items():
        if candidate.data_source != DATA_SOURCE_OCR_READER:
            filtered_candidates[game_id] = candidate
            continue
        if ocr_reader_live and candidate.game_id == runtime_game_id:
            filtered_candidates[game_id] = candidate
            continue
        filtered_out.add(game_id)
    filtered_ids = [game_id for game_id in available_game_ids if game_id not in filtered_out]
    return filtered_ids, filtered_candidates


def choose_candidate(
    candidates: dict[str, SessionCandidate],
    *,
    bound_game_id: str,
    current_game_id: str,
    keep_current: bool,
) -> SessionCandidate | None:
    if bound_game_id:
        return candidates.get(bound_game_id)
    preferred_candidates = [
        item for item in candidates.values() if item.data_source == DATA_SOURCE_BRIDGE_SDK
    ]
    if not preferred_candidates:
        preferred_candidates = [
            item for item in candidates.values() if item.data_source == DATA_SOURCE_OCR_READER
        ]
    if not preferred_candidates:
        preferred_candidates = list(candidates.values())
    if keep_current and current_game_id:
        current = candidates.get(current_game_id)
        if current is not None and current in preferred_candidates:
            return current
    if not preferred_candidates:
        return None
    return max(
        preferred_candidates,
        key=lambda item: (item.sort_key[0], item.sort_key[1], item.sort_key[2], item.game_id),
    )


def build_active_session_meta(candidate: SessionCandidate) -> dict[str, Any]:
    session = candidate.session
    return {
        "data_source": candidate.data_source,
        "game_id": candidate.game_id,
        "session_id": session.get("session_id", ""),
        "started_at": session.get("started_at", ""),
        "last_seq": session.get("last_seq", 0),
        "engine": session.get("engine", ""),
        "game_title": session.get("game_title", ""),
        "locale": session.get("locale", ""),
        "bridge_sdk_version": session.get("bridge_sdk_version", ""),
        "metadata": sanitize_metadata(session.get("metadata")),
        "session_path": str(candidate.session_path),
        "events_path": str(candidate.events_path),
    }


def derive_connection_state(
    *,
    bridge_root: Path,
    plugin_error: str,
    active_session_id: str,
    last_seen_data_monotonic: float,
    now_monotonic: float,
    stale_after_seconds: float,
    stream_reset_pending: bool,
) -> str:
    if plugin_error:
        return STATE_ERROR
    if not bridge_root.exists() or not bridge_root.is_dir():
        return STATE_DISCONNECTED
    if not active_session_id:
        return STATE_IDLE
    if stream_reset_pending:
        return STATE_ACTIVE
    if last_seen_data_monotonic > 0 and now_monotonic - last_seen_data_monotonic > stale_after_seconds:
        return STATE_STALE
    return STATE_ACTIVE


def next_poll_interval_for_state(connection_state: str, *, stream_reset_pending: bool, config: GalgameConfig) -> float:
    if stream_reset_pending or connection_state == STATE_ACTIVE:
        return config.active_poll_interval_seconds
    return config.idle_poll_interval_seconds


def summarize_status(
    *,
    connection_state: str,
    mode: str,
    bound_game_id: str,
    active_session_id: str,
    last_seq: int,
    last_error: dict[str, Any],
    active_data_source: str,
) -> str:
    if active_data_source == DATA_SOURCE_OCR_READER and active_session_id:
        prefix = "已通过 OCR 读取连接（降级模式）"
    elif active_data_source == DATA_SOURCE_MEMORY_READER and active_session_id:
        prefix = "已通过内存读取连接（降级模式）"
    elif active_data_source == DATA_SOURCE_BRIDGE_SDK and active_session_id:
        prefix = "已通过 Bridge SDK 连接"
    else:
        prefix = connection_state
    parts = [prefix, f"state={connection_state}", f"mode={mode}"]
    if bound_game_id:
        parts.append(f"bound={bound_game_id}")
    if active_session_id:
        parts.append(f"session={active_session_id}")
    parts.append(f"last_seq={last_seq}")
    message = last_error.get("message") if isinstance(last_error, dict) else ""
    if isinstance(message, str) and message:
        parts.append(f"warning={message}")
    return " | ".join(parts)


def _append_limited(items: list[dict[str, Any]], item: dict[str, Any], limit: int) -> None:
    items.append(item)
    if len(items) > limit:
        del items[:-limit]


def _line_fingerprint(game_id: str, line_id: str, text: str) -> dict[str, str]:
    return {
        "game_id": game_id,
        "line_id": line_id,
        "normalized_text": normalize_text(text),
    }


def _line_history_entry(payload_obj: dict[str, Any], *, ts: str, stability: str) -> dict[str, Any]:
    return {
        "line_id": str(payload_obj.get("line_id") or ""),
        "speaker": str(payload_obj.get("speaker") or ""),
        "text": str(payload_obj.get("text") or ""),
        "scene_id": str(payload_obj.get("scene_id") or ""),
        "route_id": str(payload_obj.get("route_id") or ""),
        "stability": stability,
        "ts": ts,
    }


def _append_observed_line(
    history_observed_lines: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    limit: int,
) -> None:
    text = normalize_text(str(item.get("text") or ""))
    if not text:
        return
    line_id = str(item.get("line_id") or "")
    scene_id = str(item.get("scene_id") or "")
    for index in range(len(history_observed_lines) - 1, -1, -1):
        existing = history_observed_lines[index]
        same_line = line_id and line_id == str(existing.get("line_id") or "")
        same_text = (
            text == normalize_text(str(existing.get("text") or ""))
            and scene_id == str(existing.get("scene_id") or "")
        )
        if same_line or same_text:
            history_observed_lines[index] = item
            if len(history_observed_lines) > limit:
                del history_observed_lines[:-limit]
            return
    _append_limited(history_observed_lines, item, limit)


def _update_dedupe_window(
    dedupe_window: list[dict[str, str]],
    fingerprint: dict[str, str],
    limit: int,
) -> bool:
    for index, item in enumerate(dedupe_window):
        if item == fingerprint:
            dedupe_window.append(dedupe_window.pop(index))
            if len(dedupe_window) > limit:
                del dedupe_window[:-limit]
            return True
    dedupe_window.append(fingerprint)
    if len(dedupe_window) > limit:
        del dedupe_window[:-limit]
    return False


def summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    payload_obj = payload if isinstance(payload, dict) else {}
    return {
        "seq": int(event.get("seq") or 0),
        "ts": str(event.get("ts") or ""),
        "type": str(event.get("type") or ""),
        "line_id": str(payload_obj.get("line_id") or ""),
        "scene_id": str(payload_obj.get("scene_id") or ""),
        "route_id": str(payload_obj.get("route_id") or ""),
        "payload": json_copy(payload_obj),
    }


def apply_event_to_snapshot(snapshot: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_snapshot = sanitize_snapshot_state(snapshot)
    event_type = str(event.get("type") or "")
    payload = event.get("payload")
    payload_obj = payload if isinstance(payload, dict) else {}
    event_ts = str(event.get("ts") or "")

    if event_type == "session_started":
        next_snapshot["speaker"] = str(payload_obj.get("speaker") or "")
        next_snapshot["text"] = str(payload_obj.get("text") or "")
        next_snapshot["choices"] = [
            sanitize_choice(item) for item in payload_obj.get("choices", [])
        ] if isinstance(payload_obj.get("choices"), list) else []
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or "")
        next_snapshot["line_id"] = str(payload_obj.get("line_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or "")
        next_snapshot["is_menu_open"] = bool(payload_obj.get("is_menu_open", next_snapshot["choices"]))
        next_snapshot["save_context"] = sanitize_save_context(payload_obj.get("save_context"))
        next_snapshot["stability"] = str(payload_obj.get("stability") or "")
        next_snapshot["ts"] = event_ts
        return next_snapshot

    if event_type in {"line_observed", "line_changed"}:
        if not _payload_is_game_dialogue_line(payload_obj, ts=event_ts):
            return next_snapshot
        next_snapshot["speaker"] = str(payload_obj.get("speaker") or "")
        next_snapshot["text"] = str(payload_obj.get("text") or "")
        next_snapshot["choices"] = []
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or next_snapshot.get("scene_id") or "")
        next_snapshot["line_id"] = str(payload_obj.get("line_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or next_snapshot.get("route_id") or "")
        next_snapshot["is_menu_open"] = False
        next_snapshot["stability"] = str(
            payload_obj.get("stability") or ("stable" if event_type == "line_changed" else "tentative")
        )
        next_snapshot["ts"] = event_ts
        return next_snapshot

    if event_type == "choices_shown":
        choices_obj = payload_obj.get("choices")
        next_snapshot["choices"] = (
            [sanitize_choice(item) for item in choices_obj]
            if isinstance(choices_obj, list)
            else []
        )
        next_snapshot["line_id"] = str(payload_obj.get("line_id") or next_snapshot.get("line_id") or "")
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or next_snapshot.get("scene_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or next_snapshot.get("route_id") or "")
        next_snapshot["is_menu_open"] = bool(next_snapshot["choices"])
        next_snapshot["stability"] = "choices" if next_snapshot["choices"] else ""
        next_snapshot["ts"] = event_ts
        return next_snapshot

    if event_type == "choice_selected":
        next_snapshot["choices"] = []
        next_snapshot["is_menu_open"] = False
        next_snapshot["stability"] = ""
        next_snapshot["line_id"] = str(payload_obj.get("line_id") or next_snapshot.get("line_id") or "")
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or next_snapshot.get("scene_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or next_snapshot.get("route_id") or "")
        next_snapshot["ts"] = event_ts
        return next_snapshot

    if event_type == "scene_changed":
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or next_snapshot.get("scene_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or next_snapshot.get("route_id") or "")
        next_snapshot["ts"] = event_ts
        return next_snapshot

    if event_type == "save_loaded":
        next_snapshot["scene_id"] = str(payload_obj.get("scene_id") or next_snapshot.get("scene_id") or "")
        next_snapshot["line_id"] = str(payload_obj.get("line_id") or "")
        next_snapshot["route_id"] = str(payload_obj.get("route_id") or next_snapshot.get("route_id") or "")
        next_snapshot["save_context"] = sanitize_save_context(payload_obj.get("save_context"))
        next_snapshot["choices"] = []
        next_snapshot["is_menu_open"] = False
        next_snapshot["ts"] = event_ts
        return next_snapshot

    return next_snapshot


def apply_event_to_histories(
    *,
    history_events: list[dict[str, Any]],
    history_lines: list[dict[str, Any]],
    history_observed_lines: list[dict[str, Any]] | None = None,
    history_choices: list[dict[str, Any]],
    dedupe_window: list[dict[str, str]],
    event: dict[str, Any],
    config: GalgameConfig,
    game_id: str,
) -> None:
    payload = event.get("payload")
    payload_obj = payload if isinstance(payload, dict) else {}
    event_type = str(event.get("type") or "")
    event_ts = str(event.get("ts") or "")

    _append_limited(history_events, summarize_event(event), config.history_events_limit)

    if event_type == "line_observed":
        if not _payload_is_game_dialogue_line(payload_obj, ts=event_ts):
            return
        if history_observed_lines is not None:
            _append_observed_line(
                history_observed_lines,
                _line_history_entry(payload_obj, ts=event_ts, stability="tentative"),
                limit=config.history_lines_limit,
            )
        return

    if event_type == "line_changed":
        if not _payload_is_game_dialogue_line(payload_obj, ts=event_ts):
            return
        fingerprint = _line_fingerprint(
            game_id,
            str(payload_obj.get("line_id") or ""),
            str(payload_obj.get("text") or ""),
        )
        duplicate = _update_dedupe_window(
            dedupe_window, fingerprint, config.dedupe_window_limit
        )
        if duplicate:
            return
        _append_limited(
            history_lines,
            _line_history_entry(payload_obj, ts=event_ts, stability="stable"),
            config.history_lines_limit,
        )
        if history_observed_lines is not None:
            _append_observed_line(
                history_observed_lines,
                _line_history_entry(payload_obj, ts=event_ts, stability="stable"),
                limit=config.history_lines_limit,
            )
        return

    if event_type == "choices_shown":
        choices_obj = payload_obj.get("choices")
        if not isinstance(choices_obj, list):
            return
        for choice in choices_obj:
            item = sanitize_choice(choice)
            _append_limited(
                history_choices,
                {
                    "choice_id": item["choice_id"],
                    "text": item["text"],
                    "line_id": str(payload_obj.get("line_id") or ""),
                    "scene_id": str(payload_obj.get("scene_id") or ""),
                    "route_id": str(payload_obj.get("route_id") or ""),
                    "index": item["index"],
                    "action": "shown",
                    "ts": event_ts,
                },
                config.history_choices_limit,
            )
        return

    if event_type == "choice_selected":
        _append_limited(
            history_choices,
            {
                "choice_id": str(payload_obj.get("choice_id") or ""),
                "text": str(payload_obj.get("choice_text") or ""),
                "line_id": str(payload_obj.get("line_id") or ""),
                "scene_id": str(payload_obj.get("scene_id") or ""),
                "route_id": str(payload_obj.get("route_id") or ""),
                "index": int(payload_obj.get("choice_index") or 0),
                "action": "selected",
                "ts": event_ts,
            },
            config.history_choices_limit,
        )


def rebuild_histories_from_events(
    *,
    events: Iterable[dict[str, Any]],
    snapshot: dict[str, Any],
    dedupe_window: list[dict[str, str]],
    config: GalgameConfig,
    game_id: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
    dict[str, Any],
]:
    history_events: list[dict[str, Any]] = []
    history_lines: list[dict[str, Any]] = []
    history_observed_lines: list[dict[str, Any]] = []
    history_choices: list[dict[str, Any]] = []
    working_window = [dict(item) for item in dedupe_window]
    working_snapshot = sanitize_snapshot_state(snapshot)

    for event in events:
        apply_event_to_histories(
            history_events=history_events,
            history_lines=history_lines,
            history_observed_lines=history_observed_lines,
            history_choices=history_choices,
            dedupe_window=working_window,
            event=event,
            config=config,
            game_id=game_id,
        )
        working_snapshot = apply_event_to_snapshot(working_snapshot, event)

    return (
        history_events,
        history_lines,
        history_observed_lines,
        history_choices,
        working_window,
        working_snapshot,
    )


def build_status_payload(state, *, config: GalgameConfig) -> dict[str, Any]:
    dxcam = inspect_dxcam_installation()
    textractor = inspect_textractor_installation(
        configured_path=config.memory_reader_textractor_path,
        install_target_dir_raw=config.memory_reader_install_target_dir,
    )
    rapidocr = inspect_rapidocr_installation(
        install_target_dir_raw=config.rapidocr_install_target_dir,
        engine_type=config.rapidocr_engine_type,
        lang_type=config.rapidocr_lang_type,
        model_type=config.rapidocr_model_type,
        ocr_version=config.rapidocr_ocr_version,
    )
    tesseract = inspect_tesseract_installation(
        configured_path=config.ocr_reader_tesseract_path,
        install_target_dir_raw=config.ocr_reader_install_target_dir,
        languages=config.ocr_reader_languages,
    )
    ocr_runtime = json_copy(state.ocr_reader_runtime)
    ocr_capture_diagnostic_required = bool(
        isinstance(ocr_runtime, dict)
        and (
            ocr_runtime.get("ocr_capture_diagnostic_required")
            or str(ocr_runtime.get("ocr_context_state") or "")
            in {"poll_not_running", "capture_failed", "diagnostic_required", "stale_capture_backend"}
            or str(ocr_runtime.get("detail") or "") == "ocr_capture_diagnostic_required"
        )
    )
    ocr_capture_diagnostic = ""
    if ocr_capture_diagnostic_required and isinstance(ocr_runtime, dict):
        ocr_capture_diagnostic = build_ocr_context_diagnostic(
            {
                "ocr_reader_runtime": ocr_runtime,
                "last_error": json_copy(state.last_error),
            }
        )
    local_state = {
        "latest_snapshot": json_copy(state.latest_snapshot),
        "history_observed_lines": json_copy(state.history_observed_lines),
        "history_lines": json_copy(state.history_lines),
        "ocr_reader_runtime": ocr_runtime,
        "last_error": json_copy(state.last_error),
    }
    effective_current_line = resolve_effective_current_line(local_state)
    return {
        "connection_state": state.current_connection_state,
        "mode": state.mode,
        "push_notifications": state.push_notifications,
        "advance_speed": getattr(state, "advance_speed", "medium"),
        "bound_game_id": state.bound_game_id,
        "available_game_ids": list(state.available_game_ids),
        "active_session_id": state.active_session_id,
        "active_data_source": state.active_data_source,
        "stream_reset_pending": state.stream_reset_pending,
        "last_seq": state.last_seq,
        "last_error": json_copy(state.last_error),
        "memory_reader_runtime": json_copy(state.memory_reader_runtime),
        "ocr_reader_runtime": ocr_runtime,
        "effective_current_line": json_copy(effective_current_line or {}),
        "ocr_capture_diagnostic_required": ocr_capture_diagnostic_required,
        "ocr_capture_diagnostic": ocr_capture_diagnostic,
        "ocr_capture_profiles": json_copy(state.ocr_capture_profiles),
        "summary": summarize_status(
            connection_state=state.current_connection_state,
            mode=state.mode,
            bound_game_id=state.bound_game_id or state.active_game_id,
            active_session_id=state.active_session_id,
            last_seq=state.last_seq,
            last_error=state.last_error,
            active_data_source=state.active_data_source,
        ),
        "phase": "phase_1",
        "memory_reader_enabled": config.memory_reader_enabled,
        "ocr_reader_enabled": config.ocr_reader_enabled,
        "ocr_backend_selection": config.ocr_reader_backend_selection,
        "ocr_capture_backend_selection": config.ocr_reader_capture_backend,
        "ocr_reader_poll_interval_seconds": config.ocr_reader_poll_interval_seconds,
        "rapidocr_enabled": config.rapidocr_enabled,
        "dxcam": dxcam,
        "rapidocr": rapidocr,
        "textractor": textractor,
        "tesseract": tesseract,
    }


def build_snapshot_payload(state) -> dict[str, Any]:
    stale = state.current_connection_state == STATE_STALE
    snapshot = sanitize_snapshot_state(state.latest_snapshot)
    if _looks_like_ocr_overlay_text(snapshot.get("text")):
        snapshot["speaker"] = ""
        snapshot["text"] = ""
        snapshot["line_id"] = ""
        snapshot["stability"] = ""
    effective_current_line = resolve_effective_current_line(
        {
            "latest_snapshot": json_copy(snapshot),
            "history_observed_lines": json_copy(state.history_observed_lines),
            "history_lines": json_copy(state.history_lines),
        }
    )
    return {
        "game_id": state.active_game_id,
        "session_id": state.active_session_id,
        "snapshot": json_copy(snapshot),
        "effective_current_line": json_copy(effective_current_line or {}),
        "snapshot_ts": str(snapshot.get("ts") or ""),
        "stale": stale,
    }


def build_history_payload(state, *, limit: int, include_events: bool) -> dict[str, Any]:
    bounded_limit = max(1, limit)
    stable_lines = [
        item for item in state.history_lines
        if _looks_like_game_dialogue_context_line(item if isinstance(item, dict) else {})
    ]
    observed_lines = [
        item for item in state.history_observed_lines
        if _looks_like_game_dialogue_context_line(item if isinstance(item, dict) else {})
    ]
    return {
        "game_id": state.active_game_id,
        "session_id": state.active_session_id,
        "events": json_copy(state.history_events[-bounded_limit:]) if include_events else [],
        "stable_lines": json_copy(stable_lines[-bounded_limit:]),
        "observed_lines": json_copy(observed_lines[-bounded_limit:]),
        "choices": json_copy(state.history_choices[-bounded_limit:]),
    }


def build_snapshot_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    normalized = sanitize_snapshot_state(snapshot)
    choices = tuple(
        (
            str(item.get("choice_id") or ""),
            str(item.get("text") or ""),
            int(item.get("index") or 0),
            bool(item.get("enabled", True)),
        )
        for item in normalized.get("choices", [])
    )
    return (
        normalized.get("speaker", ""),
        normalized.get("text", ""),
        normalized.get("scene_id", ""),
        normalized.get("line_id", ""),
        normalized.get("route_id", ""),
        bool(normalized.get("is_menu_open", False)),
        tuple(normalized.get("save_context", {}).items()),
        choices,
    )


def latest_selected_choice(history_choices: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(history_choices):
        if str(item.get("action") or "") == "selected":
            return dict(item)
    return None


def build_choice_signature(choices: list[dict[str, Any]]) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            str(item.get("choice_id") or ""),
            str(item.get("text") or ""),
            int(item.get("index") or 0),
        )
        for item in choices
    )


def _current_line_entry(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    normalized = sanitize_snapshot_state(snapshot)
    if not normalized.get("line_id") or not normalized.get("text"):
        return None
    if _looks_like_ocr_overlay_text(normalized.get("text")):
        return None
    entry = {
        "line_id": str(normalized.get("line_id") or ""),
        "speaker": str(normalized.get("speaker") or ""),
        "text": str(normalized.get("text") or ""),
        "scene_id": str(normalized.get("scene_id") or ""),
        "route_id": str(normalized.get("route_id") or ""),
        "stability": str(normalized.get("stability") or ""),
        "source": "snapshot",
        "ts": str(normalized.get("ts") or ""),
    }
    if not _looks_like_game_dialogue_context_line(entry):
        return None
    return entry


def resolve_effective_current_line(local_state: dict[str, Any]) -> dict[str, Any] | None:
    snapshot_line = _current_line_entry(local_state.get("latest_snapshot", {}))
    if snapshot_line is not None:
        return snapshot_line
    for source_key, source_label in (
        ("history_observed_lines", "observed"),
        ("history_lines", "stable"),
    ):
        for item in reversed(local_state.get(source_key, [])):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            line_id = str(item.get("line_id") or "")
            if not text or not line_id:
                continue
            result = dict(item)
            result["source"] = source_label
            result["stability"] = str(
                result.get("stability") or ("stable" if source_label == "stable" else "tentative")
            )
            return result
    return None


def build_ocr_context_diagnostic(local_state: dict[str, Any]) -> str:
    runtime = local_state.get("ocr_reader_runtime")
    runtime_obj = runtime if isinstance(runtime, dict) else {}
    parts = ["ocr_context_unavailable"]
    context_state = str(runtime_obj.get("ocr_context_state") or "").strip()
    detail = str(runtime_obj.get("detail") or "").strip()
    status = str(runtime_obj.get("status") or "").strip()
    if context_state:
        parts.append(f"context_state={context_state}")
    if status:
        parts.append(f"status={status}")
    if detail:
        parts.append(f"detail={detail}")
    backend = str(runtime_obj.get("backend_kind") or "").strip()
    if backend:
        parts.append(f"backend={backend}")
    capture_backend = str(runtime_obj.get("capture_backend_kind") or "").strip()
    if capture_backend:
        parts.append(f"capture_backend={capture_backend}")
    capture_detail = str(runtime_obj.get("capture_backend_detail") or "").strip()
    if capture_detail:
        parts.append(f"capture_detail={capture_detail}")
    if runtime_obj.get("stale_capture_backend"):
        parts.append("stale_capture_backend=true")
    same_frames = int(runtime_obj.get("consecutive_same_capture_frames") or 0)
    if same_frames:
        parts.append(f"same_capture_frames={same_frames}")
    image_hash = str(runtime_obj.get("last_capture_image_hash") or "").strip()
    if image_hash:
        parts.append(f"capture_hash={image_hash}")
    error = str(runtime_obj.get("last_capture_error") or "").strip()
    if error:
        parts.append(f"last_capture_error={error}")
    raw_text = str(runtime_obj.get("last_raw_ocr_text") or "").strip()
    if raw_text:
        parts.append(f"last_raw_ocr_text={raw_text[:80]}")
    profile = runtime_obj.get("capture_profile")
    if profile:
        parts.append(f"profile={profile}")
    target = str(
        runtime_obj.get("effective_process_name")
        or runtime_obj.get("process_name")
        or ""
    ).strip()
    if target:
        parts.append(f"target={target}")
    last_error = local_state.get("last_error")
    if isinstance(last_error, dict) and str(last_error.get("message") or ""):
        parts.append(f"last_error={str(last_error.get('message') or '')}")
    return " | ".join(parts)


def _scene_lines(history_lines: list[dict[str, Any]], scene_id: str, *, limit: int) -> list[dict[str, Any]]:
    if scene_id:
        items = [
            dict(item)
            for item in history_lines
            if str(item.get("scene_id") or "") == scene_id
        ]
    else:
        items = [dict(item) for item in history_lines]
    return items[-limit:]


def _scene_selected_choices(
    history_choices: list[dict[str, Any]],
    scene_id: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    items = [
        dict(item)
        for item in history_choices
        if str(item.get("action") or "") == "selected"
        and (not scene_id or str(item.get("scene_id") or "") == scene_id)
    ]
    return items[-limit:]


def _append_unique_line(
    lines: list[dict[str, Any]],
    line: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not line:
        return lines[-limit:]
    normalized = dict(line)
    target_key = _dialogue_line_dedupe_key(normalized)
    exists = any(_dialogue_line_dedupe_key(item) == target_key for item in lines)
    if exists:
        return lines[-limit:]
    merged = list(lines) + [normalized]
    return merged[-limit:]


def _dialogue_line_dedupe_key(item: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
    if text:
        return "::".join(
            [
                str(item.get("scene_id") or "").strip(),
                str(item.get("speaker") or "").strip(),
                text,
            ]
        )
    return str(item.get("line_id") or "").strip()


def _dialogue_context_lines(lines: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in lines:
        if not _looks_like_game_dialogue_context_line(item):
            continue
        normalized = dict(item)
        key = _dialogue_line_dedupe_key(normalized)
        if not key:
            continue
        if key not in deduped:
            order.append(key)
        deduped[key] = normalized
    return [deduped[key] for key in order][-limit:]


def _is_memory_reader_identifier(value: object) -> bool:
    return isinstance(value, str) and value.startswith("mem:")


def _is_ocr_reader_identifier(value: object) -> bool:
    return isinstance(value, str) and value.startswith("ocr:")


def _build_input_degraded_context(
    local_state: dict[str, Any],
    *,
    scene_id: str,
    line_id: str,
    choice_ids: list[str],
) -> tuple[str, bool, list[str]]:
    input_source = str(local_state.get("active_data_source") or DATA_SOURCE_BRIDGE_SDK)
    reasons: list[str] = []
    if input_source == DATA_SOURCE_MEMORY_READER:
        reasons.append("memory_reader_source")
    if input_source == DATA_SOURCE_OCR_READER:
        reasons.append("ocr_reader_source")
    if _is_memory_reader_identifier(scene_id):
        reasons.append("memory_reader_scene")
    if _is_ocr_reader_identifier(scene_id):
        reasons.append("ocr_reader_scene")
    if _is_memory_reader_identifier(line_id):
        reasons.append("memory_reader_line")
    if _is_ocr_reader_identifier(line_id):
        reasons.append("ocr_reader_line")
    if any(_is_memory_reader_identifier(choice_id) for choice_id in choice_ids):
        reasons.append("memory_reader_choice")
    if any(_is_ocr_reader_identifier(choice_id) for choice_id in choice_ids):
        reasons.append("ocr_reader_choice")
    return input_source, bool(reasons), reasons


def _input_degraded_diagnostic(context: dict[str, Any]) -> str:
    reasons = list(context.get("degraded_reasons") or [])
    if not reasons:
        return ""
    input_source = str(context.get("input_source") or "")
    source_label = (
        DATA_SOURCE_OCR_READER
        if input_source == DATA_SOURCE_OCR_READER
        else DATA_SOURCE_MEMORY_READER
    )
    return (
        f"{source_label}_input: input comes from {source_label}, semantic granularity is "
        "weaker than bridge_sdk but the workflow remains usable "
        f"({','.join(reasons)})"
    )


def apply_input_degraded_result(
    payload: dict[str, Any],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    next_payload = dict(payload)
    semantic_degraded = bool(context.get("input_degraded"))
    next_payload["input_source"] = str(context.get("input_source") or DATA_SOURCE_BRIDGE_SDK)
    next_payload["semantic_degraded"] = semantic_degraded
    next_payload["semantic_granularity"] = (
        "weaker_than_bridge_sdk" if semantic_degraded else "bridge_sdk_level"
    )
    next_payload["fallback_used"] = bool(payload.get("degraded"))
    if not semantic_degraded:
        return next_payload
    next_payload["degraded"] = True
    detail = _input_degraded_diagnostic(context)
    if detail:
        next_payload["input_diagnostic"] = detail
    diagnostic = str(next_payload.get("diagnostic") or "")
    if not diagnostic and detail:
        next_payload["diagnostic"] = detail
    return next_payload


def _resolve_target_line(local_state: dict[str, Any], *, line_id: str) -> dict[str, Any] | None:
    snapshot_line = _current_line_entry(local_state.get("latest_snapshot", {}))
    if snapshot_line and str(snapshot_line.get("line_id") or "") == line_id:
        return snapshot_line
    for item in reversed(local_state.get("history_lines", [])):
        if str(item.get("line_id") or "") == line_id:
            return dict(item)
    for item in reversed(local_state.get("history_observed_lines", [])):
        if str(item.get("line_id") or "") == line_id:
            return dict(item)
    return None


def build_local_scene_summary(
    *,
    scene_id: str,
    route_id: str,
    lines: list[dict[str, Any]],
    selected_choices: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> str:
    normalized_snapshot = sanitize_snapshot_state(snapshot)
    if lines:
        recent_parts = []
        for item in lines[-6:]:
            speaker = str(item.get("speaker") or "旁白").strip() or "旁白"
            text = str(item.get("text") or "").strip()
            if text:
                recent_parts.append(f"{speaker}：{text}")
        summary = f"当前场景 {scene_id or '(unknown)'} 的近期上下文是："
        summary += "；".join(recent_parts) if recent_parts else "暂时只有零散台词。"
    elif normalized_snapshot.get("text"):
        summary = (
            f"当前场景 {scene_id or '(unknown)'} 目前停留在"
            f"「{str(normalized_snapshot.get('speaker') or '旁白')}：{str(normalized_snapshot.get('text') or '')}」"
        )
    else:
        summary = f"当前场景 {scene_id or '(unknown)'} 暂时没有足够台词上下文。"
    if route_id:
        summary += f" 路线 {route_id}。"
    if selected_choices:
        summary += f" 已发生 {len(selected_choices)} 次选项确认。"
    return summary


def build_explain_context(local_state: dict[str, Any], *, line_id: str) -> dict[str, Any]:
    snapshot = sanitize_snapshot_state(local_state.get("latest_snapshot", {}))
    effective_line = resolve_effective_current_line(local_state)
    effective_line_id = line_id or str(
        (effective_line or {}).get("line_id") or snapshot.get("line_id") or ""
    )
    if not effective_line_id:
        raise ValueError(build_ocr_context_diagnostic(local_state))

    target_line = (
        dict(effective_line)
        if effective_line is not None
        and str(effective_line.get("line_id") or "") == effective_line_id
        else _resolve_target_line(local_state, line_id=effective_line_id)
    )
    if target_line is None:
        raise ValueError(f"unknown line_id: {effective_line_id}; {build_ocr_context_diagnostic(local_state)}")

    scene_id = str(target_line.get("scene_id") or snapshot.get("scene_id") or "")
    route_id = str(target_line.get("route_id") or snapshot.get("route_id") or "")
    stable_lines = _scene_lines(local_state.get("history_lines", []), scene_id, limit=8)
    observed_lines = _scene_lines(
        local_state.get("history_observed_lines", []),
        scene_id,
        limit=8,
    )
    scene_lines = _append_unique_line([*stable_lines, *observed_lines], target_line, limit=8)
    selected_choices = _scene_selected_choices(
        local_state.get("history_choices", []),
        scene_id,
        limit=6,
    )

    evidence: list[dict[str, Any]] = []
    snapshot_line = _current_line_entry(snapshot)
    if snapshot_line and str(snapshot_line.get("line_id") or "") == effective_line_id:
        evidence.append(
            {
                "type": "current_line",
                "text": str(snapshot_line.get("text") or ""),
                "line_id": effective_line_id,
                "speaker": str(snapshot_line.get("speaker") or ""),
                "scene_id": str(snapshot_line.get("scene_id") or ""),
                "route_id": str(snapshot_line.get("route_id") or ""),
            }
        )
    for item in scene_lines[-4:]:
        if str(item.get("line_id") or "") == effective_line_id:
            continue
        evidence.append(
            {
                "type": "history_line",
                "text": str(item.get("text") or ""),
                "line_id": str(item.get("line_id") or ""),
                "speaker": str(item.get("speaker") or ""),
                "scene_id": str(item.get("scene_id") or ""),
                "route_id": str(item.get("route_id") or ""),
            }
        )
    for choice in selected_choices[-2:]:
        evidence.append(
            {
                "type": "choice",
                "text": str(choice.get("text") or ""),
                "line_id": str(choice.get("line_id") or ""),
                "speaker": "",
                "scene_id": str(choice.get("scene_id") or ""),
                "route_id": str(choice.get("route_id") or ""),
            }
        )
    input_source, input_degraded, degraded_reasons = _build_input_degraded_context(
        local_state,
        scene_id=scene_id,
        line_id=effective_line_id,
        choice_ids=[str(choice.get("choice_id") or "") for choice in selected_choices],
    )

    return {
        "game_id": str(local_state.get("active_game_id") or ""),
        "session_id": str(local_state.get("active_session_id") or ""),
        "scene_id": scene_id,
        "route_id": route_id,
        "line_id": effective_line_id,
        "speaker": str(target_line.get("speaker") or ""),
        "text": str(target_line.get("text") or ""),
        "current_snapshot": snapshot,
        "recent_lines": scene_lines,
        "stable_lines": stable_lines,
        "observed_lines": observed_lines,
        "recent_choices": selected_choices,
        "evidence": evidence,
        "input_source": input_source,
        "input_degraded": input_degraded,
        "degraded_reasons": degraded_reasons,
    }


def build_summarize_context(
    local_state: dict[str, Any],
    *,
    scene_id: str,
) -> dict[str, Any]:
    snapshot = sanitize_snapshot_state(local_state.get("latest_snapshot", {}))
    effective_line = resolve_effective_current_line(local_state)
    effective_scene_id = scene_id or str(
        snapshot.get("scene_id") or (effective_line or {}).get("scene_id") or ""
    )
    route_id = str(snapshot.get("route_id") or (effective_line or {}).get("route_id") or "")
    stable_lines = _scene_lines(local_state.get("history_lines", []), effective_scene_id, limit=20)
    observed_lines = _scene_lines(
        local_state.get("history_observed_lines", []),
        effective_scene_id,
        limit=20,
    )
    stable_lines = _dialogue_context_lines(stable_lines, limit=20)
    observed_lines = _dialogue_context_lines(observed_lines, limit=20)
    scene_lines = _dialogue_context_lines([*stable_lines, *observed_lines], limit=20)
    selected_choices = _scene_selected_choices(
        local_state.get("history_choices", []),
        effective_scene_id,
        limit=12,
    )
    input_source, input_degraded, degraded_reasons = _build_input_degraded_context(
        local_state,
        scene_id=effective_scene_id,
        line_id=str(snapshot.get("line_id") or ""),
        choice_ids=[str(choice.get("choice_id") or "") for choice in selected_choices],
    )
    return {
        "game_id": str(local_state.get("active_game_id") or ""),
        "session_id": str(local_state.get("active_session_id") or ""),
        "scene_id": effective_scene_id,
        "route_id": route_id,
        "current_snapshot": snapshot,
        "recent_lines": scene_lines,
        "stable_lines": stable_lines,
        "observed_lines": observed_lines,
        "recent_choices": selected_choices,
        "scene_summary_seed": build_local_scene_summary(
            scene_id=effective_scene_id,
            route_id=route_id,
            lines=scene_lines,
            selected_choices=selected_choices,
            snapshot=snapshot,
        ),
        "input_source": input_source,
        "input_degraded": input_degraded,
        "degraded_reasons": degraded_reasons,
    }


def build_suggest_context(local_state: dict[str, Any]) -> dict[str, Any]:
    snapshot = sanitize_snapshot_state(local_state.get("latest_snapshot", {}))
    visible_choices = [
        sanitize_choice(item) for item in snapshot.get("choices", [])
    ]
    scene_id = str(snapshot.get("scene_id") or "")
    route_id = str(snapshot.get("route_id") or "")
    stable_lines = _scene_lines(local_state.get("history_lines", []), scene_id, limit=8)
    observed_lines = _scene_lines(
        local_state.get("history_observed_lines", []),
        scene_id,
        limit=8,
    )
    scene_lines = [*stable_lines, *observed_lines][-8:]
    selected_choices = _scene_selected_choices(
        local_state.get("history_choices", []),
        scene_id,
        limit=8,
    )
    input_source, input_degraded, degraded_reasons = _build_input_degraded_context(
        local_state,
        scene_id=scene_id,
        line_id=str(snapshot.get("line_id") or ""),
        choice_ids=[
            str(choice.get("choice_id") or "")
            for choice in [*visible_choices, *selected_choices]
        ],
    )
    return {
        "game_id": str(local_state.get("active_game_id") or ""),
        "session_id": str(local_state.get("active_session_id") or ""),
        "scene_id": scene_id,
        "route_id": route_id,
        "current_snapshot": snapshot,
        "visible_choices": visible_choices,
        "recent_lines": scene_lines,
        "stable_lines": stable_lines,
        "observed_lines": observed_lines,
        "recent_choices": selected_choices,
        "scene_summary": build_local_scene_summary(
            scene_id=scene_id,
            route_id=route_id,
            lines=scene_lines,
            selected_choices=selected_choices,
            snapshot=snapshot,
        ),
        "input_source": input_source,
        "input_degraded": input_degraded,
        "degraded_reasons": degraded_reasons,
    }


def build_explain_degraded_result(
    context: dict[str, Any],
    *,
    diagnostic: str,
) -> dict[str, Any]:
    speaker = str(context.get("speaker") or "").strip()
    text = str(context.get("text") or "").strip()
    scene_id = str(context.get("scene_id") or "").strip()
    route_id = str(context.get("route_id") or "").strip()
    if speaker and text:
        explanation = f"当前改用本地上下文保守说明：{speaker} 说了「{text}」。"
    elif text:
        explanation = f"当前改用本地上下文保守说明：这句台词是「{text}」。"
    else:
        explanation = "当前改用本地上下文保守说明，暂时拿不到更细的解释。"
    if scene_id:
        explanation += f" 场景 {scene_id}。"
    if route_id:
        explanation += f" 路线 {route_id}。"
    return {
        "degraded": True,
        "line_id": str(context.get("line_id") or ""),
        "speaker": str(context.get("speaker") or ""),
        "text": str(context.get("text") or ""),
        "explanation": explanation,
        "evidence": json_copy(context.get("evidence") or []),
        "diagnostic": diagnostic,
    }


def build_summarize_degraded_result(
    context: dict[str, Any],
    *,
    diagnostic: str,
) -> dict[str, Any]:
    summary = str(context.get("scene_summary_seed") or "").strip()
    if not summary:
        summary = build_local_scene_summary(
            scene_id=str(context.get("scene_id") or ""),
            route_id=str(context.get("route_id") or ""),
            lines=list(context.get("recent_lines") or []),
            selected_choices=list(context.get("recent_choices") or []),
            snapshot=context.get("current_snapshot", {}),
        )
    return {
        "degraded": True,
        "scene_id": str(context.get("scene_id") or ""),
        "summary": summary,
        "key_points": [],
        "diagnostic": diagnostic,
    }


def build_suggest_degraded_result(
    context: dict[str, Any],
    *,
    diagnostic: str,
) -> dict[str, Any]:
    return {
        "degraded": True,
        "scene_id": str(context.get("scene_id") or ""),
        "choices": [],
        "diagnostic": diagnostic,
    }


def phase_1_mode_enabled(mode: str, *, allow_choice_advisor: bool = True) -> bool:
    if mode == MODE_COMPANION:
        return True
    if allow_choice_advisor and mode == MODE_CHOICE_ADVISOR:
        return True
    return False


def build_memory_reader_warning() -> dict[str, Any]:
    return make_error(
        "memory_reader.enabled is set, but Textractor integration is intentionally not implemented in Phase 1",
        source="memory_reader",
        kind="warning",
    )


def mode_allows_agent_push(mode: str) -> bool:
    return mode != MODE_SILENT


def mode_allows_choice_push(mode: str) -> bool:
    return mode == MODE_CHOICE_ADVISOR


def mode_allows_agent_actuation(mode: str) -> bool:
    return mode == MODE_CHOICE_ADVISOR
