from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    neko_plugin,
    plugin_entry,
    timer_interval,
)

from .game_llm_agent import GameLLMAgent
from .host_agent_adapter import HostAgentAdapter
from .llm_gateway import LLMGateway
from .memory_reader import MemoryReaderManager
from .ocr_reader import OcrReaderManager
from .dxcam_support import install_dxcam
from .models import (
    ADVANCE_SPEEDS,
    ADVANCE_SPEED_MEDIUM,
    DATA_SOURCE_BRIDGE_SDK,
    DATA_SOURCE_MEMORY_READER,
    DATA_SOURCE_NONE,
    DATA_SOURCE_OCR_READER,
    MODE_COMPANION,
    MODES,
    build_ocr_capture_profile_bucket_key,
    compute_ocr_window_aspect_ratio,
    OCR_CAPTURE_PROFILE_RATIO_KEYS,
    OCR_CAPTURE_PROFILE_SAVE_SCOPES,
    OCR_CAPTURE_PROFILE_SAVE_SCOPE_PROCESS_FALLBACK,
    OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET,
    OCR_CAPTURE_PROFILE_STAGE_DEFAULT,
    OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
    OCR_CAPTURE_PROFILE_STAGES,
    OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY,
    OCR_TRIGGER_MODE_AFTER_ADVANCE,
    OCR_TRIGGER_MODE_INTERVAL,
    OCR_TRIGGER_MODES,
    parse_ocr_capture_profile_bucket_key,
    READER_MODE_AUTO,
    READER_MODE_MEMORY,
    READER_MODE_OCR,
    READER_MODES,
    STATE_ACTIVE,
    STATE_ERROR,
    STORE_BOUND_GAME_ID,
    STORE_ADVANCE_SPEED,
    STORE_DEDUPE_WINDOW,
    STORE_EVENTS_BYTE_OFFSET,
    STORE_EVENTS_FILE_SIZE,
    STORE_LAST_ERROR,
    STORE_LAST_SEQ,
    STORE_MODE,
    STORE_OCR_CAPTURE_PROFILES,
    STORE_OCR_WINDOW_TARGET,
    STORE_PUSH_NOTIFICATIONS,
    STORE_SESSION_ID,
    json_copy,
    make_error,
)
from .reader import tail_events_jsonl, warmup_replay_events
from .service import (
    apply_event_to_histories,
    apply_event_to_snapshot,
    apply_input_degraded_result,
    build_active_session_meta,
    build_config,
    build_explain_degraded_result,
    build_explain_context,
    build_history_payload,
    build_ocr_context_diagnostic,
    build_primary_diagnosis,
    build_snapshot_payload,
    build_status_payload,
    build_suggest_context,
    build_suggest_degraded_result,
    build_summarize_degraded_result,
    build_summarize_context,
    choose_candidate,
    derive_connection_state,
    filter_memory_reader_candidates,
    filter_ocr_reader_candidates,
    mode_allows_agent_actuation,
    next_poll_interval_for_state,
    rebuild_histories_from_events,
    scan_session_candidates,
)
from .rapidocr_support import install_rapidocr
from .state import GalgameSharedState, build_initial_state
from .store import GalgameStore
from .tesseract_support import install_tesseract
from .textractor_support import install_textractor
from .ui_api import build_open_ui_payload


def _format_install_entry_error(label: str, exc: Exception) -> str:
    message = str(exc or "").strip()
    prefix = f"{label} 安装失败"
    if not message:
        return prefix
    if message.startswith(prefix):
        return message
    return f"{prefix}：{message}"


def _log_plugin_noncritical(logger: Any, level: str, message: str, *args: Any) -> None:
    log_fn = getattr(logger, level, None)
    if not callable(log_fn):
        return
    try:
        log_fn(message, *args)
    except Exception:
        return


_PLUGIN_TOML_PATH = Path(__file__).with_name("plugin.toml")
_OCR_BACKEND_SELECTIONS = {"auto", "rapidocr", "tesseract"}
_OCR_CAPTURE_BACKEND_SELECTIONS = {"auto", "dxcam", "imagegrab", "printwindow"}
_BACKGROUND_BRIDGE_POLL_MIN_STALE_SECONDS = 45.0
_OCR_FOREGROUND_REFRESH_TTL_SECONDS = 2.0
_OCR_AFTER_ADVANCE_CAPTURE_DELAY_SECONDS = 0.15
_OCR_AFTER_ADVANCE_SETTLE_POLL_SECONDS = 0.15
_OCR_AFTER_ADVANCE_MAX_SETTLE_SECONDS = 2.0


def _matches_toml_key(stripped_line: str, key: str) -> bool:
    stripped = str(stripped_line or "").strip()
    return (
        stripped.startswith(f"{key}=")
        or stripped.startswith(f"{key} ")
        or stripped.startswith(f"{key}\t")
    )


def _replace_toml_section_value(text: str, *, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    in_section = False
    section_header = f"[{section}]"
    replacement = f'{key} = "{value}"'
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            continue
        if in_section and _matches_toml_key(stripped, key):
            prefix = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{prefix}{replacement}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    raise ValueError(f"plugin.toml missing [{section}].{key}")


def _replace_toml_section_number_value(
    text: str,
    *,
    section: str,
    key: str,
    value: float,
) -> str:
    lines = text.splitlines()
    in_section = False
    section_header = f"[{section}]"
    formatted = f"{float(value):.3f}".rstrip("0").rstrip(".")
    replacement = f"{key} = {formatted}"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == section_header
            continue
        if in_section and _matches_toml_key(stripped, key):
            prefix = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{prefix}{replacement}"
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    raise ValueError(f"plugin.toml missing [{section}].{key}")


def _normalize_ocr_trigger_mode(value: str | None) -> str:
    normalized = str(value or OCR_TRIGGER_MODE_AFTER_ADVANCE).strip().lower()
    if normalized not in OCR_TRIGGER_MODES:
        raise ValueError(f"invalid OCR trigger_mode: {value!r}")
    return normalized


def _normalize_reader_mode(value: str | None) -> str:
    normalized = str(value or READER_MODE_AUTO).strip().lower()
    if normalized not in READER_MODES:
        raise ValueError(f"invalid reader_mode: {value!r}")
    return normalized


def _session_candidate_has_text(candidate: Any) -> bool:
    session = getattr(candidate, "session", {})
    if not isinstance(session, dict):
        return False
    state = session.get("state", {})
    if not isinstance(state, dict):
        return False
    if str(state.get("text") or "").strip():
        return True
    choices = state.get("choices", [])
    return isinstance(choices, list) and bool(choices)


def _pending_data_source_for_reader_mode(
    reader_mode: str,
    *,
    memory_reader_allowed: bool,
    ocr_reader_allowed: bool,
    memory_reader_candidate_available: bool,
) -> str:
    if reader_mode == READER_MODE_MEMORY:
        return DATA_SOURCE_MEMORY_READER
    if reader_mode == READER_MODE_OCR:
        return DATA_SOURCE_OCR_READER
    if reader_mode == READER_MODE_AUTO:
        if memory_reader_candidate_available and memory_reader_allowed:
            return DATA_SOURCE_MEMORY_READER
        if ocr_reader_allowed:
            return DATA_SOURCE_OCR_READER
    return DATA_SOURCE_NONE


def _normalize_ocr_capture_profile_stage(stage: str | None) -> str:
    normalized = str(stage or OCR_CAPTURE_PROFILE_STAGE_DEFAULT).strip().lower()
    if normalized not in OCR_CAPTURE_PROFILE_STAGES:
        raise ValueError(f"invalid OCR capture profile stage: {stage!r}")
    return normalized


def _normalize_ocr_capture_profile_save_scope(save_scope: str | None) -> str:
    normalized = str(save_scope or "").strip().lower()
    if not normalized:
        return ""
    if normalized not in OCR_CAPTURE_PROFILE_SAVE_SCOPES:
        raise ValueError(f"invalid OCR capture profile save_scope: {save_scope!r}")
    return normalized


def _is_ratio_profile_payload(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in OCR_CAPTURE_PROFILE_RATIO_KEYS)


def _capture_profile_entry_to_stage_map(value: object) -> dict[str, dict[str, float]]:
    if _is_ratio_profile_payload(value):
        return {OCR_CAPTURE_PROFILE_STAGE_DEFAULT: json_copy(value)}
    raw = value if isinstance(value, dict) else {}
    stage_map: dict[str, dict[str, float]] = {}
    for stage_name, profile in raw.items():
        normalized_stage_name = str(stage_name or "").strip().lower()
        if (
            not normalized_stage_name
            or normalized_stage_name == OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY
            or not _is_ratio_profile_payload(profile)
        ):
            continue
        stage_map[normalized_stage_name] = json_copy(profile)
    return stage_map


def _capture_profile_bucket_entry_to_stage_map(value: object) -> dict[str, dict[str, float]]:
    raw = value if isinstance(value, dict) else {}
    stage_map: dict[str, dict[str, float]] = {}
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, dict):
        return stage_map
    for stage_name, profile in raw_stages.items():
        normalized_stage_name = str(stage_name or "").strip().lower()
        if not normalized_stage_name or not _is_ratio_profile_payload(profile):
            continue
        stage_map[normalized_stage_name] = json_copy(profile)
    return stage_map


def _capture_profile_entry_to_window_bucket_map(value: object) -> dict[str, dict[str, Any]]:
    raw = value if isinstance(value, dict) else {}
    raw_buckets = raw.get(OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY)
    if not isinstance(raw_buckets, dict):
        return {}
    bucket_map: dict[str, dict[str, Any]] = {}
    for bucket_key, bucket_value in raw_buckets.items():
        normalized_bucket_key = str(bucket_key or "").strip().lower()
        parsed_dimensions = parse_ocr_capture_profile_bucket_key(normalized_bucket_key)
        if not normalized_bucket_key or parsed_dimensions is None or not isinstance(bucket_value, dict):
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
                bucket_value.get("aspect_ratio") or compute_ocr_window_aspect_ratio(width, height)
            )
        except (TypeError, ValueError):
            aspect_ratio = compute_ocr_window_aspect_ratio(width, height)
        stage_map = _capture_profile_bucket_entry_to_stage_map(bucket_value)
        if not stage_map:
            continue
        bucket_map[normalized_bucket_key] = {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "stages": stage_map,
        }
    return bucket_map


def _window_bucket_map_to_capture_profile_payload(
    bucket_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for bucket_key, bucket_value in bucket_map.items():
        normalized_bucket_key = str(bucket_key or "").strip().lower()
        if not normalized_bucket_key or not isinstance(bucket_value, dict):
            continue
        try:
            width = int(bucket_value.get("width") or 0)
            height = int(bucket_value.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        try:
            aspect_ratio = float(
                bucket_value.get("aspect_ratio") or compute_ocr_window_aspect_ratio(width, height)
            )
        except (TypeError, ValueError):
            aspect_ratio = compute_ocr_window_aspect_ratio(width, height)
        stage_map = _capture_profile_bucket_entry_to_stage_map(bucket_value)
        if not stage_map:
            continue
        payload[normalized_bucket_key] = {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "stages": {
                stage_name: json_copy(profile)
                for stage_name, profile in stage_map.items()
            },
        }
    return payload


def _capture_profile_components_to_entry(
    stage_map: dict[str, dict[str, float]],
    window_bucket_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not window_bucket_map and len(stage_map) == 1 and OCR_CAPTURE_PROFILE_STAGE_DEFAULT in stage_map:
        return json_copy(stage_map[OCR_CAPTURE_PROFILE_STAGE_DEFAULT])
    payload = {stage_name: json_copy(profile) for stage_name, profile in stage_map.items()}
    bucket_payload = _window_bucket_map_to_capture_profile_payload(window_bucket_map)
    if bucket_payload:
        payload[OCR_CAPTURE_PROFILE_WINDOW_BUCKETS_KEY] = bucket_payload
    return payload


@neko_plugin
class GalgamePlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._state_lock = threading.Lock()
        self._poll_bridge_lock = asyncio.Lock()
        self._poll_bridge_thread_lock = threading.RLock()
        self._bridge_poll_task_lock = threading.RLock()
        self._textractor_install_lock = threading.Lock()
        self._tesseract_install_lock = threading.Lock()
        self._rapidocr_install_lock = threading.Lock()
        self._dxcam_install_lock = threading.Lock()
        self._cfg = None
        self._state = build_initial_state(
            mode=MODE_COMPANION,
            push_notifications=True,
            advance_speed=ADVANCE_SPEED_MEDIUM,
        )
        self._persist = GalgameStore(self.store, self.logger)
        self._host_agent_adapter: HostAgentAdapter | None = None
        self._llm_gateway: LLMGateway | None = None
        self._game_agent: GameLLMAgent | None = None
        self._memory_reader_manager: MemoryReaderManager | None = None
        self._ocr_reader_manager: OcrReaderManager | None = None
        self._bridge_poll_task: asyncio.Task[None] | Future[None] | None = None
        self._bridge_poll_loop: asyncio.AbstractEventLoop | None = None
        self._bridge_poll_thread: threading.Thread | None = None
        self._bridge_poll_thread_stop = threading.Event()
        self._bridge_poll_started_at = 0.0
        self._bridge_poll_finished_at = 0.0
        self._last_bridge_poll_duration_seconds = 0.0
        self._last_bridge_poll_launch_at = 0.0
        self._bridge_poll_launch_count = 0
        self._last_agent_tick_at = 0.0
        self._pending_ocr_advance_captures = 0
        self._last_ocr_advance_capture_requested_at = 0.0
        self._last_ocr_advance_capture_reason = ""
        self._last_ocr_foreground_refresh_at = 0.0
        self._state_dirty = True
        self._cached_snapshot: dict[str, Any] | None = None

    def request_ocr_after_advance_capture(self, *, reason: str = "agent_advance") -> None:
        if self._cfg is not None and getattr(self._cfg, "reader_mode", READER_MODE_AUTO) == READER_MODE_MEMORY:
            return
        with self._state_lock:
            self._pending_ocr_advance_captures = min(
                self._pending_ocr_advance_captures + 1,
                8,
            )
            self._last_ocr_advance_capture_requested_at = time.monotonic()
            self._last_ocr_advance_capture_reason = str(reason or "agent_advance")
            self._state.next_poll_at_monotonic = 0.0
            self._state_dirty = True
            self._cached_snapshot = None
        self._start_background_bridge_poll()

    def _has_pending_ocr_advance_capture(self) -> bool:
        with self._state_lock:
            return self._pending_ocr_advance_captures > 0

    def _pending_ocr_advance_capture_delay_remaining(self) -> float:
        with self._state_lock:
            if self._pending_ocr_advance_captures <= 0:
                return 0.0
            requested_at = float(self._last_ocr_advance_capture_requested_at or 0.0)
        if requested_at <= 0.0:
            return 0.0
        elapsed = max(0.0, time.monotonic() - requested_at)
        return max(0.0, _OCR_AFTER_ADVANCE_CAPTURE_DELAY_SECONDS - elapsed)

    def _pending_ocr_advance_capture_age(self) -> float:
        with self._state_lock:
            if self._pending_ocr_advance_captures <= 0:
                return 0.0
            requested_at = float(self._last_ocr_advance_capture_requested_at or 0.0)
        if requested_at <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - requested_at)

    def _consume_ocr_advance_capture(self) -> None:
        with self._state_lock:
            if self._pending_ocr_advance_captures > 0:
                self._pending_ocr_advance_captures -= 1

    def _clear_pending_ocr_advance_captures(self) -> None:
        with self._state_lock:
            self._pending_ocr_advance_captures = 0

    def _snapshot_state(self, *, fresh: bool = False) -> dict[str, Any]:
        with self._state_lock:
            if not fresh and not self._state_dirty and self._cached_snapshot is not None:
                return self._cached_snapshot
            state = self._state
            snap = {
                "bound_game_id": state.bound_game_id,
                "available_game_ids": list(state.available_game_ids),
                "mode": state.mode,
                "push_notifications": state.push_notifications,
                "advance_speed": state.advance_speed,
                "active_game_id": state.active_game_id,
                "active_session_id": state.active_session_id,
                "active_session_meta": json_copy(state.active_session_meta),
                "active_data_source": state.active_data_source,
                "latest_snapshot": json_copy(state.latest_snapshot),
                "history_events": json_copy(state.history_events),
                "history_lines": json_copy(state.history_lines),
                "history_observed_lines": json_copy(state.history_observed_lines),
                "history_choices": json_copy(state.history_choices),
                "dedupe_window": json_copy(state.dedupe_window),
                "line_buffer": state.line_buffer,
                "stream_reset_pending": state.stream_reset_pending,
                "last_error": json_copy(state.last_error),
                "next_poll_at_monotonic": state.next_poll_at_monotonic,
                "current_connection_state": state.current_connection_state,
                "events_byte_offset": state.events_byte_offset,
                "events_file_size": state.events_file_size,
                "last_seq": state.last_seq,
                "last_seen_data_monotonic": state.last_seen_data_monotonic,
                "warmup_session_id": state.warmup_session_id,
                "memory_reader_runtime": json_copy(state.memory_reader_runtime),
                "ocr_reader_runtime": json_copy(state.ocr_reader_runtime),
                "ocr_capture_profiles": json_copy(state.ocr_capture_profiles),
                "ocr_window_target": json_copy(state.ocr_window_target),
                "plugin_error": state.plugin_error,
            }
            if not fresh:
                self._cached_snapshot = snap
                self._state_dirty = False
            return snap

    def _mark_state_dirty(self) -> None:
        with self._state_lock:
            self._state_dirty = True
            self._cached_snapshot = None

    @staticmethod
    def _ocr_capture_scope_label(save_scope: str) -> str:
        if save_scope == OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET:
            return "当前窗口分辨率"
        return "进程通用回退"

    @staticmethod
    def _ocr_capture_stage_label(stage: str) -> str:
        labels = {
            OCR_CAPTURE_PROFILE_STAGE_DEFAULT: "通用区域",
            "dialogue_stage": "对白区",
            "menu_stage": "菜单区",
        }
        return labels.get(stage, stage)

    @staticmethod
    def _process_name_matches(left: str, right: str) -> bool:
        return bool(left.strip()) and left.strip().lower() == right.strip().lower()

    def _resolve_ocr_capture_profile_save_context(
        self,
        *,
        process_name: str,
        save_scope: str | None,
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        with self._state_lock:
            runtime = json_copy(self._state.ocr_reader_runtime)
        runtime_process_name = str(runtime.get("process_name") or "").strip()
        runtime_width = max(0, int(runtime.get("width") or 0))
        runtime_height = max(0, int(runtime.get("height") or 0))
        resolved_width = max(0, int(width or runtime_width))
        resolved_height = max(0, int(height or runtime_height))
        normalized_scope = _normalize_ocr_capture_profile_save_scope(save_scope)
        if not normalized_scope:
            normalized_scope = (
                OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET
                if self._process_name_matches(process_name, runtime_process_name)
                and resolved_width > 0
                and resolved_height > 0
                else OCR_CAPTURE_PROFILE_SAVE_SCOPE_PROCESS_FALLBACK
            )
        bucket_key = ""
        aspect_ratio = 0.0
        if normalized_scope == OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET:
            if resolved_width <= 0 or resolved_height <= 0:
                raise ValueError("当前没有可用的 OCR 窗口尺寸，无法保存到当前窗口分辨率")
            bucket_key = build_ocr_capture_profile_bucket_key(resolved_width, resolved_height).lower()
            aspect_ratio = compute_ocr_window_aspect_ratio(resolved_width, resolved_height)
        return {
            "save_scope": normalized_scope,
            "width": resolved_width,
            "height": resolved_height,
            "bucket_key": bucket_key,
            "aspect_ratio": aspect_ratio,
            "runtime": runtime,
        }

    async def _save_ocr_capture_profile_payload(
        self,
        *,
        process_name: str,
        stage: str,
        capture_profile: dict[str, float] | None,
        clear: bool,
        save_scope: str | None,
        width: int = 0,
        height: int = 0,
    ) -> dict[str, Any]:
        normalized_process_name = str(process_name or "").strip()
        if not normalized_process_name:
            raise ValueError("process_name is required")
        normalized_stage = _normalize_ocr_capture_profile_stage(stage)
        context = self._resolve_ocr_capture_profile_save_context(
            process_name=normalized_process_name,
            save_scope=save_scope,
            width=width,
            height=height,
        )
        with self._state_lock:
            profiles = json_copy(self._state.ocr_capture_profiles)
        existing_entry = profiles.get(normalized_process_name)
        process_stage_map = _capture_profile_entry_to_stage_map(existing_entry)
        window_bucket_map = _capture_profile_entry_to_window_bucket_map(existing_entry)
        normalized_profile = json_copy(capture_profile or {})
        resolved_scope = str(context["save_scope"] or OCR_CAPTURE_PROFILE_SAVE_SCOPE_PROCESS_FALLBACK)
        bucket_key = str(context.get("bucket_key") or "")
        if resolved_scope == OCR_CAPTURE_PROFILE_SAVE_SCOPE_PROCESS_FALLBACK:
            target_stage_map = process_stage_map
        else:
            bucket_entry = window_bucket_map.get(bucket_key) or {
                "width": int(context.get("width") or 0),
                "height": int(context.get("height") or 0),
                "aspect_ratio": float(context.get("aspect_ratio") or 0.0),
                "stages": {},
            }
            target_stage_map = _capture_profile_bucket_entry_to_stage_map(bucket_entry)
        if clear:
            target_stage_map.pop(normalized_stage, None)
        else:
            target_stage_map[normalized_stage] = normalized_profile
        if resolved_scope == OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET:
            if target_stage_map:
                window_bucket_map[bucket_key] = {
                    "width": int(context.get("width") or 0),
                    "height": int(context.get("height") or 0),
                    "aspect_ratio": float(context.get("aspect_ratio") or 0.0),
                    "stages": target_stage_map,
                }
            else:
                window_bucket_map.pop(bucket_key, None)
        if not process_stage_map and not window_bucket_map:
            profiles.pop(normalized_process_name, None)
        else:
            profiles[normalized_process_name] = _capture_profile_components_to_entry(
                process_stage_map,
                window_bucket_map,
            )
        self._persist.persist_ocr_capture_profiles(profiles)
        with self._state_lock:
            self._state.ocr_capture_profiles = json_copy(profiles)
            self._state_dirty = True
            self._cached_snapshot = None
        if self._ocr_reader_manager is not None:
            self._ocr_reader_manager.update_capture_profiles(profiles)
            try:
                refreshed_runtime = (
                    self._ocr_reader_manager.refresh_runtime_capture_profile_selection()
                )
            except Exception as exc:
                self.logger.warning(
                    "galgame_plugin failed to refresh OCR runtime after saving capture profile: {}",
                    exc,
                )
            else:
                with self._state_lock:
                    self._state.ocr_reader_runtime = json_copy(refreshed_runtime)
                    self._state_dirty = True
                    self._cached_snapshot = None
        payload = {
            "process_name": normalized_process_name,
            "stage": normalized_stage,
            "capture_profile": normalized_profile if not clear else {},
            "cleared": bool(clear),
            "save_scope": resolved_scope,
            "bucket_key": bucket_key,
            "window_width": int(context.get("width") or 0),
            "window_height": int(context.get("height") or 0),
        }
        scope_label = self._ocr_capture_scope_label(resolved_scope)
        stage_label = self._ocr_capture_stage_label(normalized_stage)
        if clear:
            payload["summary"] = (
                f"OCR 截图校准已清空：{normalized_process_name} / {stage_label} / {scope_label}"
                + (f" / {bucket_key}" if bucket_key else "")
            )
        else:
            payload["summary"] = (
                f"OCR 截图校准已保存：{normalized_process_name} / {stage_label} / {scope_label}"
                + (f" / {bucket_key}" if bucket_key else "")
            )
        payload["status"] = await self._build_status_payload_async()
        return payload

    def _commit_state(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            state = self._state
            changed = False

            def assign(name: str, value: Any) -> None:
                nonlocal changed
                if getattr(state, name) != value:
                    setattr(state, name, value)
                    changed = True

            def assign_json(name: str, value: Any) -> None:
                nonlocal changed
                if getattr(state, name) != value:
                    setattr(state, name, json_copy(value))
                    changed = True

            assign("bound_game_id", str(payload["bound_game_id"]))
            assign("available_game_ids", list(payload["available_game_ids"]))
            # Preferences can be changed through plugin entries while a bridge poll is in
            # flight. Keep the live values instead of restoring the poll's stale snapshot.
            assign("mode", state.mode if state.mode in MODES else str(payload["mode"]))
            assign("push_notifications", bool(state.push_notifications))
            assign("advance_speed", (
                state.advance_speed
                if state.advance_speed in ADVANCE_SPEEDS
                else str(payload.get("advance_speed") or ADVANCE_SPEED_MEDIUM)
            ))
            assign("active_game_id", str(payload["active_game_id"]))
            assign("active_session_id", str(payload["active_session_id"]))
            assign_json("active_session_meta", payload["active_session_meta"])
            assign("active_data_source", str(payload["active_data_source"]))
            assign_json("latest_snapshot", payload["latest_snapshot"])
            assign_json("history_events", payload["history_events"])
            assign_json("history_lines", payload["history_lines"])
            assign_json("history_observed_lines", payload.get("history_observed_lines", []))
            assign_json("history_choices", payload["history_choices"])
            assign_json("dedupe_window", payload["dedupe_window"])
            assign("line_buffer", payload["line_buffer"])
            assign("stream_reset_pending", bool(payload["stream_reset_pending"]))
            assign_json("last_error", payload["last_error"])
            assign("next_poll_at_monotonic", float(payload["next_poll_at_monotonic"]))
            assign("current_connection_state", str(payload["current_connection_state"]))
            assign("events_byte_offset", int(payload["events_byte_offset"]))
            assign("events_file_size", int(payload["events_file_size"]))
            assign("last_seq", int(payload["last_seq"]))
            assign("last_seen_data_monotonic", float(payload["last_seen_data_monotonic"]))
            assign("warmup_session_id", str(payload["warmup_session_id"]))
            assign_json("memory_reader_runtime", payload["memory_reader_runtime"])
            assign_json("ocr_reader_runtime", payload["ocr_reader_runtime"])
            assign_json("ocr_capture_profiles", payload["ocr_capture_profiles"])
            assign_json("ocr_window_target", payload["ocr_window_target"])
            assign("plugin_error", str(payload["plugin_error"]))
            if changed:
                self._state_dirty = True
                self._cached_snapshot = None

    def _record_error(self, error: dict[str, Any]) -> None:
        with self._state_lock:
            self._state.last_error = json_copy(error)
            self._state_dirty = True
            self._cached_snapshot = None

    def _bridge_poll_debug_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._bridge_poll_task_lock:
            with self._state_lock:
                bridge_poll_task = self._bridge_poll_task
                bridge_poll_started_at = float(self._bridge_poll_started_at or 0.0)
                next_poll_at = float(self._state.next_poll_at_monotonic or 0.0)
                pending_ocr_advance_captures = int(self._pending_ocr_advance_captures or 0)
                last_ocr_advance_capture_requested_at = float(
                    self._last_ocr_advance_capture_requested_at or 0.0
                )
                last_ocr_advance_capture_reason = str(
                    self._last_ocr_advance_capture_reason or ""
                )
        poll_running = bridge_poll_task is not None and not bridge_poll_task.done()
        inflight_seconds = (
            max(0.0, now - bridge_poll_started_at)
            if poll_running and bridge_poll_started_at > 0.0
            else 0.0
        )
        next_poll_in_seconds = max(0.0, next_poll_at - now) if next_poll_at > 0.0 else 0.0
        pending_ocr_advance_capture_age_seconds = (
            max(0.0, now - last_ocr_advance_capture_requested_at)
            if pending_ocr_advance_captures > 0
            and last_ocr_advance_capture_requested_at > 0.0
            else 0.0
        )
        return {
            "bridge_poll_running": poll_running,
            "bridge_poll_inflight_seconds": inflight_seconds,
            "last_bridge_poll_duration_seconds": self._last_bridge_poll_duration_seconds,
            "next_bridge_poll_in_seconds": next_poll_in_seconds,
            "last_agent_tick_at": self._last_agent_tick_at,
            "last_bridge_poll_launch_at": self._last_bridge_poll_launch_at,
            "bridge_poll_launch_count": self._bridge_poll_launch_count,
            "pending_ocr_advance_captures": pending_ocr_advance_captures,
            "pending_ocr_advance_capture_age_seconds": pending_ocr_advance_capture_age_seconds,
            "last_ocr_advance_capture_reason": last_ocr_advance_capture_reason,
        }

    def _clear_completed_background_bridge_poll(
        self,
        completed_task: asyncio.Task[None] | Future[None] | None = None,
    ) -> None:
        with self._bridge_poll_task_lock:
            task = self._bridge_poll_task
            if task is None or not task.done():
                return
            if completed_task is not None and task is not completed_task:
                return
            self._bridge_poll_task = None
        if task.cancelled():
            with self._state_lock:
                self._state.next_poll_at_monotonic = 0.0
                self._state_dirty = True
                self._cached_snapshot = None
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            with self._state_lock:
                self._state.next_poll_at_monotonic = 0.0
                self._state_dirty = True
                self._cached_snapshot = None
        except Exception as exc:
            with self._state_lock:
                self._state.next_poll_at_monotonic = 0.0
                self._state_dirty = True
                self._cached_snapshot = None
            self._record_error(
                make_error(
                    f"bridge background poll failed after completion: {exc}",
                    source="bridge_reader",
                    kind="error",
                )
            )

    def _ensure_bridge_poll_loop(self) -> asyncio.AbstractEventLoop | None:
        loop = self._bridge_poll_loop
        thread = self._bridge_poll_thread
        if loop is not None and thread is not None and thread.is_alive() and not loop.is_closed():
            return loop

        ready = threading.Event()
        holder: dict[str, asyncio.AbstractEventLoop] = {}
        self._bridge_poll_thread_stop.clear()

        def _run_loop() -> None:
            worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(worker_loop)
            holder["loop"] = worker_loop
            ready.set()
            try:
                worker_loop.run_forever()
            finally:
                pending = [task for task in asyncio.all_tasks(worker_loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    worker_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                worker_loop.close()

        thread = threading.Thread(
            target=_run_loop,
            name="galgame-bridge-poll",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=2.0):
            self._record_error(
                make_error(
                    "bridge background poll loop failed to start",
                    source="bridge_reader",
                    kind="error",
                )
            )
            return None
        self._bridge_poll_loop = holder.get("loop")
        self._bridge_poll_thread = thread
        return self._bridge_poll_loop

    def _stop_bridge_poll_loop(self) -> None:
        loop = self._bridge_poll_loop
        thread = self._bridge_poll_thread
        self._bridge_poll_loop = None
        self._bridge_poll_thread = None
        self._bridge_poll_thread_stop.set()
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

    def _background_bridge_poll_stale_timeout_seconds(self) -> float:
        if self._cfg is None:
            return _BACKGROUND_BRIDGE_POLL_MIN_STALE_SECONDS
        interval = max(
            float(self._cfg.active_poll_interval_seconds),
            float(self._cfg.idle_poll_interval_seconds),
            float(self._cfg.ocr_reader_poll_interval_seconds),
            1.0,
        )
        return max(_BACKGROUND_BRIDGE_POLL_MIN_STALE_SECONDS, interval * 12.0)

    def _add_bridge_poll_debug_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched.update(self._bridge_poll_debug_payload())
        runtime = dict(enriched.get("ocr_reader_runtime") or {})
        if runtime:
            context_state = str(runtime.get("ocr_context_state") or "")
            poll_running = bool(enriched.get("bridge_poll_running"))
            has_capture_attempt = bool(str(runtime.get("last_capture_attempt_at") or ""))
            if context_state == "capture_pending" and not poll_running and not has_capture_attempt:
                runtime["ocr_context_state"] = "poll_not_running"
                enriched["ocr_reader_runtime"] = runtime
                enriched["ocr_capture_diagnostic_required"] = True
                enriched["ocr_capture_diagnostic"] = (
                    "OCR 轮询未继续执行，尚未完成首次截图；请检查插件 timer、后端重载状态或刷新运行中的插件。"
                )
            enriched["ocr_context_state"] = str(runtime.get("ocr_context_state") or context_state)
        enriched["primary_diagnosis"] = build_primary_diagnosis(enriched)
        return enriched

    def _start_background_bridge_poll(self) -> bool:
        if self._cfg is None:
            return False
        self._clear_completed_background_bridge_poll()
        with self._bridge_poll_task_lock:
            if self._bridge_poll_task is not None:
                if not self._bridge_poll_task.done():
                    inflight_seconds = max(
                        0.0,
                        time.monotonic() - float(self._bridge_poll_started_at or 0.0),
                    )
                    if inflight_seconds >= self._background_bridge_poll_stale_timeout_seconds():
                        self._record_error(
                            make_error(
                                (
                                    "bridge background poll timed out; canceling stale OCR poll "
                                    f"after {inflight_seconds:.1f}s"
                                ),
                                source="bridge_reader",
                                kind="warning",
                            )
                        )
                        self._bridge_poll_task.cancel()
                        with self._state_lock:
                            self._pending_ocr_advance_captures = 0
                            self._state.next_poll_at_monotonic = 0.0
                            self._state_dirty = True
                            self._cached_snapshot = None
                    return False
                self._bridge_poll_task = None
            self._bridge_poll_started_at = time.monotonic()
            self._last_bridge_poll_launch_at = self._bridge_poll_started_at
            self._bridge_poll_launch_count += 1
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is not None and not running_loop.is_closed():
                task = running_loop.create_task(self._run_background_bridge_poll())
                self._bridge_poll_task = task
                task.add_done_callback(
                    lambda completed: self._clear_completed_background_bridge_poll(completed)
                )
                return True
            loop = self._ensure_bridge_poll_loop()
            if loop is None:
                return False
            task = asyncio.run_coroutine_threadsafe(self._run_background_bridge_poll(), loop)
            self._bridge_poll_task = task
            task.add_done_callback(
                lambda completed: self._clear_completed_background_bridge_poll(completed)
            )
            return True

    async def _run_background_bridge_poll(self) -> None:
        started_at = self._bridge_poll_started_at or time.monotonic()
        self._bridge_poll_started_at = started_at
        try:
            while not self._bridge_poll_thread_stop.is_set():
                await self._poll_bridge(force=False)
                if not self._has_pending_ocr_advance_capture():
                    break
                delay = self._pending_ocr_advance_capture_delay_remaining()
                if delay <= 0.0:
                    delay = _OCR_AFTER_ADVANCE_SETTLE_POLL_SECONDS
                await asyncio.sleep(min(delay, _OCR_AFTER_ADVANCE_SETTLE_POLL_SECONDS))
        except Exception as exc:
            with self._state_lock:
                self._state.next_poll_at_monotonic = 0.0
                self._state_dirty = True
                self._cached_snapshot = None
            self._record_error(
                make_error(
                    f"bridge background poll failed: {exc}",
                    source="bridge_reader",
                    kind="error",
                )
            )
        finally:
            finished_at = time.monotonic()
            self._bridge_poll_finished_at = finished_at
            self._last_bridge_poll_duration_seconds = max(0.0, finished_at - started_at)

    async def _cancel_background_bridge_poll(self) -> None:
        with self._bridge_poll_task_lock:
            task = self._bridge_poll_task
            if task is None:
                return
            self._bridge_poll_task = None
        if not task.done():
            task.cancel()
            try:
                await asyncio.wrap_future(task) if isinstance(task, Future) else await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame bridge background poll cancellation failed: {}",
                    exc,
                )
        self._stop_bridge_poll_loop()

    def _persist_preferences(
        self,
        *,
        bound_game_id: str,
        mode: str,
        push_notifications: bool,
        advance_speed: str,
    ) -> None:
        self._persist.persist_preferences(
            bound_game_id=bound_game_id,
            mode=mode,
            push_notifications=push_notifications,
            advance_speed=advance_speed,
        )

    def _persist_ocr_backend_selection(
        self,
        *,
        backend_selection: str | None,
        capture_backend: str | None,
    ) -> None:
        text = _PLUGIN_TOML_PATH.read_text(encoding="utf-8")
        if backend_selection is not None:
            text = _replace_toml_section_value(
                text,
                section="ocr_reader",
                key="backend_selection",
                value=backend_selection,
            )
        if capture_backend is not None:
            text = _replace_toml_section_value(
                text,
                section="ocr_reader",
                key="capture_backend",
                value=capture_backend,
            )
        _PLUGIN_TOML_PATH.write_text(text, encoding="utf-8")

    def _persist_reader_mode(self, *, reader_mode: str) -> None:
        text = _PLUGIN_TOML_PATH.read_text(encoding="utf-8")
        text = _replace_toml_section_value(
            text,
            section="galgame",
            key="reader_mode",
            value=reader_mode,
        )
        _PLUGIN_TOML_PATH.write_text(text, encoding="utf-8")

    def _persist_ocr_timing(
        self,
        *,
        poll_interval_seconds: float,
        trigger_mode: str,
    ) -> None:
        text = _PLUGIN_TOML_PATH.read_text(encoding="utf-8")
        text = _replace_toml_section_number_value(
            text,
            section="ocr_reader",
            key="poll_interval_seconds",
            value=poll_interval_seconds,
        )
        text = _replace_toml_section_value(
            text,
            section="ocr_reader",
            key="trigger_mode",
            value=trigger_mode,
        )
        _PLUGIN_TOML_PATH.write_text(text, encoding="utf-8")

    def _persist_runtime_state(self, payload: dict[str, Any]) -> None:
        self._persist.persist_runtime(
            session_id=str(payload["active_session_id"]),
            events_byte_offset=int(payload["events_byte_offset"]),
            events_file_size=int(payload["events_file_size"]),
            last_seq=int(payload["last_seq"]),
            dedupe_window=list(payload["dedupe_window"]),
            last_error=dict(payload["last_error"]),
        )

    def _set_runtime_from_store(self, restored: dict[str, Any], warnings: list[str]) -> None:
        with self._state_lock:
            self._state = build_initial_state(
                mode=str(restored.get(STORE_MODE, MODE_COMPANION)),
                push_notifications=bool(restored.get(STORE_PUSH_NOTIFICATIONS, True)),
                advance_speed=str(restored.get(STORE_ADVANCE_SPEED, ADVANCE_SPEED_MEDIUM)),
            )
            self._state.bound_game_id = str(restored.get(STORE_BOUND_GAME_ID, ""))
            self._state.active_session_id = str(restored.get(STORE_SESSION_ID, ""))
            self._state.events_byte_offset = int(restored.get(STORE_EVENTS_BYTE_OFFSET, 0))
            self._state.events_file_size = int(restored.get(STORE_EVENTS_FILE_SIZE, 0))
            self._state.last_seq = int(restored.get(STORE_LAST_SEQ, 0))
            self._state.dedupe_window = json_copy(restored.get(STORE_DEDUPE_WINDOW, []))
            self._state.last_error = json_copy(restored.get(STORE_LAST_ERROR, {}))
            self._state.active_data_source = DATA_SOURCE_NONE
            self._state.memory_reader_runtime = {}
            self._state.ocr_reader_runtime = {}
            self._state.ocr_capture_profiles = json_copy(
                restored.get(STORE_OCR_CAPTURE_PROFILES, {})
            )
            self._state.ocr_window_target = json_copy(restored.get(STORE_OCR_WINDOW_TARGET, {}))
            if warnings and not self._state.last_error:
                self._state.last_error = make_error(
                    "; ".join(warnings),
                    source="store",
                    kind="warning",
                )
            self._state_dirty = True
            self._cached_snapshot = None

    def _current_status_payload(self) -> dict[str, Any]:
        if self._cfg is None:
            return self._add_bridge_poll_debug_payload({
                "connection_state": "error",
                "mode": MODE_COMPANION,
                "push_notifications": True,
                "bound_game_id": "",
                "available_game_ids": [],
                "active_session_id": "",
                "active_data_source": DATA_SOURCE_NONE,
                "stream_reset_pending": False,
                "last_seq": 0,
                "last_error": {},
                "summary": "config_not_loaded",
                "phase": "phase_1",
                "memory_reader_enabled": False,
                "memory_reader_runtime": {},
                "ocr_reader_enabled": False,
                "ocr_reader_runtime": {},
                "ocr_capture_profiles": {},
                "dxcam": {
                    "install_supported": False,
                    "installed": False,
                    "can_install": False,
                    "detected_path": "",
                    "package_name": "dxcam",
                    "target_dir": "",
                    "detail": "config_not_loaded",
                    "runtime_error": "",
                },
                "rapidocr_enabled": False,
                "rapidocr": {
                    "install_supported": False,
                    "installed": False,
                    "can_install": False,
                    "detected_path": "",
                    "target_dir": "",
                    "runtime_dir": "",
                    "site_packages_dir": "",
                    "model_cache_dir": "",
                    "selected_model": "",
                    "engine_type": "",
                    "lang_type": "",
                    "model_type": "",
                    "ocr_version": "",
                    "detail": "config_not_loaded",
                    "runtime_error": "",
                },
                "tesseract": {
                    "install_supported": False,
                    "installed": False,
                    "can_install": False,
                    "detected_path": "",
                    "target_dir": "",
                    "expected_executable_path": "",
                    "tessdata_dir": "",
                    "required_languages": [],
                    "missing_languages": [],
                    "detail": "config_not_loaded",
                },
                "textractor": {
                    "install_supported": False,
                    "installed": False,
                    "can_install": False,
                    "detected_path": "",
                    "target_dir": "",
                    "expected_executable_path": "",
                    "detail": "config_not_loaded",
                },
            })
        return self._add_bridge_poll_debug_payload(build_status_payload(self._state, config=self._cfg))

    async def _build_status_payload_async(self) -> dict[str, Any]:
        if self._cfg is None:
            return self._current_status_payload()
        self._refresh_ocr_foreground_state()
        state_snapshot = self._snapshot_state()
        config = self._cfg
        state = SimpleNamespace(**state_snapshot)
        payload = await asyncio.to_thread(
            build_status_payload,
            state,
            config=config,
            state_is_snapshot=True,
        )
        payload = self._add_bridge_poll_debug_payload(payload)
        if self._game_agent is not None:
            try:
                agent_payload = await self._game_agent.peek_status(state_snapshot)
                payload["agent"] = json_copy(agent_payload)
                payload["agent_status"] = str(agent_payload.get("status") or "")
                payload["agent_user_status"] = str(agent_payload.get("agent_user_status") or "")
                payload["agent_pause_kind"] = str(agent_payload.get("agent_pause_kind") or "")
                payload["agent_pause_message"] = str(
                    agent_payload.get("agent_pause_message") or ""
                )
                payload["agent_can_resume_by_button"] = bool(
                    agent_payload.get("agent_can_resume_by_button")
                )
                payload["agent_can_resume_by_focus"] = bool(
                    agent_payload.get("agent_can_resume_by_focus")
                )
                payload["agent_activity"] = str(agent_payload.get("activity") or "")
                payload["agent_reason"] = str(agent_payload.get("reason") or "")
                payload["agent_error"] = str(agent_payload.get("error") or "")
                payload["agent_inbound_queue_size"] = int(
                    agent_payload.get("inbound_queue_size") or 0
                )
                payload["agent_outbound_queue_size"] = int(
                    agent_payload.get("outbound_queue_size") or 0
                )
                payload["agent_last_interruption"] = json_copy(
                    agent_payload.get("last_interruption") or {}
                )
                payload["agent_last_outbound_message"] = json_copy(
                    agent_payload.get("last_outbound_message") or {}
                )
                agent_debug = agent_payload.get("debug")
                agent_diagnostic = (
                    str(
                        (agent_debug or {}).get("target_window_diagnostic")
                        or (agent_debug or {}).get("ocr_capture_diagnostic")
                        or ""
                    )
                    if isinstance(agent_debug, dict)
                    else ""
                )
                payload["agent_diagnostic"] = agent_diagnostic
                payload["agent_diagnostic_required"] = bool(
                    agent_diagnostic
                    or payload["agent_reason"]
                    in {
                        "ocr_context_unavailable",
                        "input_advance_unconfirmed",
                        "target_window_not_foreground",
                        "hard_error",
                    }
                )
            except Exception as exc:
                payload["agent_status"] = "unknown"
                payload["agent_user_status"] = "error"
                payload["agent_pause_kind"] = "none"
                payload["agent_pause_message"] = ""
                payload["agent_can_resume_by_button"] = False
                payload["agent_can_resume_by_focus"] = False
                payload["agent_activity"] = ""
                payload["agent_reason"] = "agent_status_unavailable"
                payload["agent_error"] = str(exc)
                payload["agent_diagnostic"] = f"agent_status_unavailable: {exc}"
                payload["agent_diagnostic_required"] = True
        payload["primary_diagnosis"] = build_primary_diagnosis(payload)
        return payload

    def _resolve_current_run_id(self) -> str:
        return str(getattr(self.ctx, "run_id", "") or "").strip()

    def _resolve_install_progress_callback(self, current_run_id: str):
        async def _progress_update(event: dict[str, Any]) -> None:
            if not current_run_id:
                return
            await self.run_update(
                run_id=current_run_id,
                status="running",
                progress=float(event.get("progress") or 0.0),
                stage=str(event.get("phase") or ""),
                message=str(event.get("message") or ""),
                metrics={
                    "phase": str(event.get("phase") or ""),
                    "downloaded_bytes": int(event.get("downloaded_bytes") or 0),
                    "total_bytes": int(event.get("total_bytes") or 0),
                    "resume_from": int(event.get("resume_from") or 0),
                    "asset_name": str(event.get("asset_name") or ""),
                    "release_name": str(event.get("release_name") or ""),
                },
            )

        return _progress_update

    async def _load_config(self) -> None:
        raw = await self.config.dump(timeout=5.0)
        raw_config = raw if isinstance(raw, dict) else {}
        self._cfg = build_config(raw_config)

    @lifecycle(id="startup")
    async def startup(self, **_):
        try:
            await self._load_config()
        except Exception as exc:
            self._record_error(
                make_error(f"load config failed: {exc}", source="config", kind="error")
            )
            return Err(SdkError(f"failed to load galgame_plugin config: {exc}"))

        try:
            restored, warnings = self._persist.load()
            self._set_runtime_from_store(restored, warnings)
        except Exception as exc:
            self._record_error(
                make_error(f"restore store failed: {exc}", source="store", kind="error")
            )
            return Err(SdkError(f"failed to restore galgame_plugin store: {exc}"))

        self._host_agent_adapter = HostAgentAdapter(self.logger)
        self._llm_gateway = LLMGateway(self, self.logger, self._cfg)
        self._game_agent = GameLLMAgent(
            plugin=self,
            logger=self.logger,
            llm_gateway=self._llm_gateway,
            host_adapter=self._host_agent_adapter,
        )
        self._memory_reader_manager = MemoryReaderManager(
            logger=self.logger,
            config=self._cfg,
        )
        self._ocr_reader_manager = OcrReaderManager(
            logger=self.logger,
            config=self._cfg,
        )
        self._ocr_reader_manager.update_capture_profiles(self._state.ocr_capture_profiles)
        self._ocr_reader_manager.update_window_target(self._state.ocr_window_target)

        self.register_static_ui("static")
        self.set_list_actions(
            [
                {
                    "id": "open_ui",
                    "kind": "ui",
                    "target": f"/plugin/{self.plugin_id}/ui/",
                    "open_in": "new_tab",
                }
            ]
        )

        await self._poll_bridge(force=True)
        return Ok({"status": "ready", "result": await self._build_status_payload_async()})

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        await self._cancel_background_bridge_poll()
        if self._memory_reader_manager is not None:
            try:
                await self._memory_reader_manager.shutdown()
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame memory reader shutdown failed: {}",
                    exc,
                )
        if self._ocr_reader_manager is not None:
            try:
                await self._ocr_reader_manager.shutdown()
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame OCR reader shutdown failed: {}",
                    exc,
                )
        if self._game_agent is not None:
            try:
                await self._game_agent.shutdown()
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame agent shutdown failed: {}",
                    exc,
                )
        if self._llm_gateway is not None:
            try:
                await self._llm_gateway.shutdown()
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame LLM gateway shutdown failed: {}",
                    exc,
                )
        if self._host_agent_adapter is not None:
            try:
                await self._host_agent_adapter.shutdown()
            except Exception as exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame host agent adapter shutdown failed: {}",
                    exc,
                )
        try:
            await self.store.close()
        except Exception as exc:
            _log_plugin_noncritical(
                self.logger,
                "warning",
                "galgame store shutdown failed: {}",
                exc,
            )
        return Ok({"status": "stopped"})

    @timer_interval(id="bridge_tick", seconds=1, auto_start=True)
    async def bridge_tick(self, **_):
        self._clear_completed_background_bridge_poll()
        self._refresh_ocr_foreground_state()
        self._trigger_ocr_for_manual_foreground_advance()
        if self._game_agent is not None:
            self._last_agent_tick_at = time.monotonic()
            try:
                await self._game_agent.tick(self._snapshot_state())
            except Exception as exc:
                self._record_error(
                    make_error(
                        f"game agent tick failed: {exc}",
                        source="game_agent",
                        kind="error",
                    )
                )
        self._start_background_bridge_poll()
        await asyncio.sleep(0)
        return Ok({"status": "tick"})

    def _refresh_ocr_foreground_state(self, *, force: bool = False) -> None:
        if self._cfg is not None and getattr(self._cfg, "reader_mode", READER_MODE_AUTO) == READER_MODE_MEMORY:
            return
        if self._ocr_reader_manager is None:
            return
        refresh = getattr(self._ocr_reader_manager, "refresh_foreground_state", None)
        if not callable(refresh):
            return
        now = time.monotonic()
        if (
            not force
            and self._last_ocr_foreground_refresh_at > 0.0
            and now - self._last_ocr_foreground_refresh_at < _OCR_FOREGROUND_REFRESH_TTL_SECONDS
        ):
            return
        try:
            runtime = refresh()
        except Exception as exc:
            self._record_error(
                make_error(
                    f"ocr_reader foreground refresh failed: {exc}",
                    source="ocr_reader",
                    kind="warning",
                )
            )
            return
        self._last_ocr_foreground_refresh_at = now
        with self._state_lock:
            self._state.ocr_reader_runtime = json_copy(runtime)
            self._state_dirty = True
            self._cached_snapshot = None

    def _trigger_ocr_for_manual_foreground_advance(self) -> None:
        if self._cfg is None or self._ocr_reader_manager is None:
            return
        if getattr(self._cfg, "reader_mode", READER_MODE_AUTO) == READER_MODE_MEMORY:
            return
        if self._cfg.ocr_reader_trigger_mode != OCR_TRIGGER_MODE_AFTER_ADVANCE:
            return
        consume = getattr(self._ocr_reader_manager, "consume_foreground_advance_input", None)
        if not callable(consume):
            return
        try:
            should_capture = bool(consume())
        except Exception as exc:
            self._record_error(
                make_error(
                    f"ocr_reader foreground advance monitor failed: {exc}",
                    source="ocr_reader",
                    kind="warning",
                )
            )
            return
        if should_capture:
            self.request_ocr_after_advance_capture(reason="manual_foreground_advance")

    async def _poll_bridge(self, *, force: bool) -> None:
        if self._cfg is None:
            return

        while not self._poll_bridge_thread_lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            await self._poll_bridge_locked(force=force)
        finally:
            self._poll_bridge_thread_lock.release()

    async def _poll_bridge_locked(self, *, force: bool) -> None:
        if self._cfg is None:
            return

        now_monotonic = time.monotonic()
        local = self._snapshot_state(fresh=True)
        next_poll_at = float(local["next_poll_at_monotonic"])
        max_reasonable_interval = max(
            float(self._cfg.active_poll_interval_seconds),
            float(self._cfg.idle_poll_interval_seconds),
            float(self._cfg.ocr_reader_poll_interval_seconds),
            1.0,
        ) * 5.0
        if not force and next_poll_at > now_monotonic + max_reasonable_interval:
            local["next_poll_at_monotonic"] = 0.0
            next_poll_at = 0.0
        if not force and now_monotonic < next_poll_at:
            return

        warnings: list[str] = []
        raw_available_game_ids: list[str] = []
        raw_candidates: dict[str, Any] = {}
        memory_reader_runtime = json_copy(local.get("memory_reader_runtime") or {})
        ocr_reader_runtime = json_copy(local.get("ocr_reader_runtime") or {})
        reader_mode = _normalize_reader_mode(getattr(self._cfg, "reader_mode", READER_MODE_AUTO))
        memory_reader_allowed = reader_mode in {READER_MODE_AUTO, READER_MODE_MEMORY}
        ocr_reader_allowed = reader_mode in {READER_MODE_AUTO, READER_MODE_OCR}

        try:
            raw_available_game_ids, raw_candidates, scan_warnings = await asyncio.to_thread(
                scan_session_candidates,
                self._cfg.bridge_root,
            )
            warnings.extend(scan_warnings)
        except Exception as exc:
            local["plugin_error"] = f"scan bridge root failed: {exc}"
            local["available_game_ids"] = []
            local["current_connection_state"] = STATE_ERROR
            local["last_error"] = make_error(
                local["plugin_error"], source="bridge_scan", kind="error"
            )
            interval = next_poll_interval_for_state(
                local["current_connection_state"],
                stream_reset_pending=bool(local["stream_reset_pending"]),
                config=self._cfg,
            )
            local["next_poll_at_monotonic"] = now_monotonic + interval
            self._commit_state(local)
            try:
                self._persist_runtime_state(local)
            except Exception as persist_exc:
                _log_plugin_noncritical(
                    self.logger,
                    "warning",
                    "galgame persist runtime state after bridge scan failure failed: {}",
                    persist_exc,
                )
            return

        memory_reader_candidate_available = any(
            candidate.data_source == DATA_SOURCE_MEMORY_READER
            and _session_candidate_has_text(candidate)
            for candidate in raw_candidates.values()
        )
        bridge_sdk_candidate_available = any(
            candidate.data_source == DATA_SOURCE_BRIDGE_SDK
            and _session_candidate_has_text(candidate)
            for candidate in raw_candidates.values()
        )
        ocr_trigger_mode = str(
            getattr(self._cfg, "ocr_reader_trigger_mode", OCR_TRIGGER_MODE_AFTER_ADVANCE)
            or OCR_TRIGGER_MODE_AFTER_ADVANCE
        )
        ocr_context_state = str(ocr_reader_runtime.get("ocr_context_state") or "")
        ocr_bootstrap_capture_needed = (
            ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            and (
                ocr_context_state in {"", "capture_pending", "observed"}
                or (
                    ocr_context_state == "no_text"
                    and int(ocr_reader_runtime.get("consecutive_no_text_polls") or 0) < 3
                )
            )
        )
        pending_ocr_advance_capture = self._has_pending_ocr_advance_capture()
        with self._state_lock:
            pending_ocr_advance_reason = str(self._last_ocr_advance_capture_reason or "")
        pending_manual_foreground_ocr_capture = (
            pending_ocr_advance_capture
            and ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            and pending_ocr_advance_reason in {
                "manual_foreground_advance",
                "foreground_target_activated",
            }
        )
        pending_ocr_delay_remaining = (
            self._pending_ocr_advance_capture_delay_remaining()
            if pending_ocr_advance_capture and not force
            else 0.0
        )
        if (
            self._memory_reader_manager is not None
            and memory_reader_allowed
        ):
            self._memory_reader_manager.update_config(self._cfg)
            try:
                memory_reader_tick = await self._memory_reader_manager.tick(
                    bridge_sdk_available=bridge_sdk_candidate_available,
                )
                warnings.extend(memory_reader_tick.warnings)
                memory_reader_runtime = memory_reader_tick.runtime
                if memory_reader_tick.should_rescan:
                    (
                        raw_available_game_ids,
                        raw_candidates,
                        rescan_warnings,
                    ) = await asyncio.to_thread(scan_session_candidates, self._cfg.bridge_root)
                    warnings.extend(rescan_warnings)
            except Exception as exc:
                warnings.append(f"memory_reader tick failed: {exc}")
        memory_reader_candidate_available = any(
            candidate.data_source == DATA_SOURCE_MEMORY_READER
            and _session_candidate_has_text(candidate)
            for candidate in raw_candidates.values()
        )
        ocr_reader_explicitly_configured = bool(
            str(getattr(self._cfg, "ocr_reader_tesseract_path", "") or "").strip()
            or str(getattr(self._cfg, "ocr_reader_install_target_dir", "") or "").strip()
            or str(getattr(self._cfg, "rapidocr_install_target_dir", "") or "").strip()
        )
        memory_reader_default_is_unavailable = (
            reader_mode == READER_MODE_AUTO
            and memory_reader_allowed
            and bool(getattr(self._cfg, "memory_reader_enabled", False))
            and not memory_reader_candidate_available
            and str(memory_reader_runtime.get("status") or "") in {"idle", "backoff"}
            and str(memory_reader_runtime.get("detail") or "")
            in {"invalid_textractor_path", "no_detected_game_process"}
            and not ocr_reader_explicitly_configured
            and str(local.get("active_data_source") or "") != DATA_SOURCE_OCR_READER
        )
        if memory_reader_default_is_unavailable:
            ocr_reader_allowed = False
        if (
            reader_mode == READER_MODE_AUTO
            and memory_reader_candidate_available
            and not pending_manual_foreground_ocr_capture
        ):
            ocr_reader_allowed = False
            with self._state_lock:
                self._pending_ocr_advance_captures = 0

        ocr_reader_stable_event_emitted = False
        if (
            self._ocr_reader_manager is not None
            and ocr_reader_allowed
            and ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
        ):
            was_foreground = bool(ocr_reader_runtime.get("target_is_foreground"))
            refresh_foreground_state = getattr(
                self._ocr_reader_manager,
                "refresh_foreground_state",
                None,
            )
            if callable(refresh_foreground_state):
                try:
                    refreshed_runtime = await asyncio.to_thread(refresh_foreground_state)
                    if isinstance(refreshed_runtime, dict):
                        ocr_reader_runtime = json_copy(refreshed_runtime)
                        if (
                            not was_foreground
                            and bool(ocr_reader_runtime.get("target_is_foreground"))
                        ):
                            if not self._has_pending_ocr_advance_capture():
                                with self._state_lock:
                                    self._pending_ocr_advance_captures = min(
                                        self._pending_ocr_advance_captures + 1,
                                        8,
                                    )
                                    self._last_ocr_advance_capture_requested_at = time.monotonic()
                                    self._last_ocr_advance_capture_reason = "foreground_target_activated"
                                    self._state.next_poll_at_monotonic = 0.0
                            pending_ocr_advance_capture = True
                            pending_ocr_delay_remaining = 0.0
                except Exception as exc:
                    warnings.append(f"ocr_reader foreground refresh failed: {exc}")
        ocr_active_fallback_capture_needed = (
            ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            and str(ocr_reader_runtime.get("status") or "") == "active"
            and str(local.get("active_data_source") or "") == DATA_SOURCE_OCR_READER
            and not (pending_ocr_advance_capture and pending_ocr_delay_remaining > 0.0)
        )
        ocr_tick_allowed = (
            ocr_reader_allowed
            and (
                ocr_trigger_mode == OCR_TRIGGER_MODE_INTERVAL
                or force
                or ocr_bootstrap_capture_needed
                or (pending_ocr_advance_capture and pending_ocr_delay_remaining <= 0.0)
                or ocr_active_fallback_capture_needed
                or str(ocr_reader_runtime.get("status") or "") not in {"active"}
                or str(local.get("active_data_source") or "") != DATA_SOURCE_OCR_READER
            )
        )

        if self._ocr_reader_manager is not None and ocr_tick_allowed:
            self._ocr_reader_manager.update_config(self._cfg)
            update_advance_speed = getattr(
                self._ocr_reader_manager,
                "update_advance_speed",
                None,
            )
            if callable(update_advance_speed):
                update_advance_speed(str(local.get("advance_speed") or ADVANCE_SPEED_MEDIUM))
            ocr_reader_tick = None
            try:
                ocr_memory_reader_runtime = (
                    {}
                    if pending_manual_foreground_ocr_capture
                    else memory_reader_runtime
                )
                ocr_reader_tick = await self._ocr_reader_manager.tick(
                    bridge_sdk_available=bridge_sdk_candidate_available,
                    memory_reader_runtime=ocr_memory_reader_runtime,
                )
                warnings.extend(ocr_reader_tick.warnings)
                ocr_reader_runtime = ocr_reader_tick.runtime
                if ocr_reader_tick.should_rescan:
                    (
                        raw_available_game_ids,
                        raw_candidates,
                        rescan_warnings,
                    ) = await asyncio.to_thread(scan_session_candidates, self._cfg.bridge_root)
                    warnings.extend(rescan_warnings)
                resolved_window_target = self._ocr_reader_manager.current_window_target()
                if resolved_window_target != json_copy(local.get("ocr_window_target") or {}):
                    local["ocr_window_target"] = json_copy(resolved_window_target)
                    try:
                        self._persist.persist_ocr_window_target(resolved_window_target)
                    except Exception as exc:
                        warnings.append(f"persist OCR window target failed: {exc}")
            except Exception as exc:
                warnings.append(f"ocr_reader tick failed: {exc}")
            finally:
                pending_capture_settled = bool(
                    ocr_reader_tick is not None
                    and getattr(ocr_reader_tick, "stable_event_emitted", False)
                )
                ocr_reader_stable_event_emitted = pending_capture_settled
                ocr_reader_capture_failed = bool(
                    ocr_reader_tick is not None
                    and isinstance(getattr(ocr_reader_tick, "runtime", None), dict)
                    and str(ocr_reader_tick.runtime.get("detail") or "") == "capture_failed"
                )
                pending_capture_expired = (
                    self._pending_ocr_advance_capture_age()
                    >= _OCR_AFTER_ADVANCE_MAX_SETTLE_SECONDS
                )
                if pending_ocr_advance_capture and ocr_reader_capture_failed:
                    self._clear_pending_ocr_advance_captures()
                    pending_ocr_advance_capture = False
                if (
                    pending_ocr_advance_capture
                    and (force or pending_capture_settled or pending_capture_expired)
                ):
                    self._consume_ocr_advance_capture()

        local["memory_reader_runtime"] = memory_reader_runtime
        local["ocr_reader_runtime"] = ocr_reader_runtime
        available_game_ids, candidates = filter_memory_reader_candidates(
            raw_available_game_ids,
            raw_candidates,
            runtime=memory_reader_runtime,
        )
        available_game_ids, candidates = filter_ocr_reader_candidates(
            available_game_ids,
            candidates,
            runtime=ocr_reader_runtime,
        )
        if reader_mode == READER_MODE_MEMORY:
            candidates = {
                game_id: candidate
                for game_id, candidate in candidates.items()
                if candidate.data_source != DATA_SOURCE_OCR_READER
            }
            available_game_ids = [game_id for game_id in available_game_ids if game_id in candidates]
        elif reader_mode == READER_MODE_OCR:
            candidates = {
                game_id: candidate
                for game_id, candidate in candidates.items()
                if candidate.data_source != DATA_SOURCE_MEMORY_READER
            }
            available_game_ids = [game_id for game_id in available_game_ids if game_id in candidates]
        local["available_game_ids"] = available_game_ids
        candidate_reader_mode = reader_mode
        if (
            reader_mode == READER_MODE_AUTO
            and pending_manual_foreground_ocr_capture
            and ocr_reader_stable_event_emitted
            and not bridge_sdk_candidate_available
        ):
            candidate_reader_mode = READER_MODE_OCR

        keep_current = (
            not local["bound_game_id"]
            and local["current_connection_state"] == STATE_ACTIVE
            and bool(local["active_game_id"])
        )
        candidate = choose_candidate(
            candidates,
            bound_game_id=str(local["bound_game_id"]),
            current_game_id=str(local["active_game_id"]),
            keep_current=keep_current,
            reader_mode=candidate_reader_mode,
        )

        if candidate is not None:
            session = candidate.session
            session_id = str(session.get("session_id") or "")
            session_changed = (
                candidate.game_id != local["active_game_id"]
                or session_id != local["active_session_id"]
            )
            restore_cursor = (
                not session_changed
                and local["events_byte_offset"] > 0
                and local["active_session_id"] == session_id
            )
            warmup_needed = session_id != local["warmup_session_id"] or session_changed

            local["active_game_id"] = candidate.game_id
            local["active_session_id"] = session_id
            local["active_session_meta"] = build_active_session_meta(candidate)
            local["active_data_source"] = candidate.data_source
            local["latest_snapshot"] = json_copy(session.get("state", {}))

            if warmup_needed:
                end_offset = int(local["events_byte_offset"]) if restore_cursor else None
                warmup_events = await asyncio.to_thread(
                    warmup_replay_events,
                    candidate.events_path,
                    bytes_limit=self._cfg.warmup_replay_bytes_limit,
                    events_limit=self._cfg.warmup_replay_events_limit,
                    end_offset=end_offset,
                )
                base_dedupe = (
                    list(local["dedupe_window"]) if restore_cursor else []
                )
                (
                    local["history_events"],
                    local["history_lines"],
                    local["history_observed_lines"],
                    local["history_choices"],
                    local["dedupe_window"],
                    local["latest_snapshot"],
                ) = rebuild_histories_from_events(
                    events=warmup_events,
                    snapshot=local["latest_snapshot"],
                    dedupe_window=base_dedupe,
                    config=self._cfg,
                    game_id=candidate.game_id,
                )
                try:
                    file_size = await asyncio.to_thread(
                        lambda: candidate.events_path.stat().st_size
                    )
                except OSError:
                    file_size = 0
                if restore_cursor and int(local["events_byte_offset"]) <= file_size:
                    local["events_file_size"] = file_size
                    local["last_seq"] = int(local["last_seq"])
                else:
                    local["events_byte_offset"] = file_size
                    local["events_file_size"] = file_size
                    local["last_seq"] = max(
                        int(session.get("last_seq") or 0),
                        max((int(event.get("seq") or 0) for event in warmup_events), default=0),
                    )
                local["line_buffer"] = b""
                local["stream_reset_pending"] = False
                local["warmup_session_id"] = session_id
                local["last_seen_data_monotonic"] = now_monotonic

            if int(session.get("last_seq") or 0) > int(local["last_seq"]):
                local["last_seen_data_monotonic"] = now_monotonic

            read_offset = 0 if local["stream_reset_pending"] else int(local["events_byte_offset"])
            read_buffer = b"" if local["stream_reset_pending"] else bytes(local["line_buffer"])
            tail = await asyncio.to_thread(
                tail_events_jsonl,
                candidate.events_path,
                offset=read_offset,
                line_buffer=read_buffer,
            )
            warnings.extend(tail.errors)

            if tail.reset_detected:
                local["stream_reset_pending"] = True
                local["line_buffer"] = b""
                local["events_file_size"] = tail.file_size
            else:
                confirm_reset = False
                if local["stream_reset_pending"] and tail.events:
                    first = tail.events[0]
                    first_seq = int(first.get("seq") or 0)
                    first_session_id = str(first.get("session_id") or "")
                    confirm_reset = first_seq == 1 and (
                        first_session_id != local["active_session_id"]
                        or int(local["last_seq"]) > 0
                    )

                if confirm_reset:
                    local["history_events"] = []
                    local["history_lines"] = []
                    local["history_observed_lines"] = []
                    local["history_choices"] = []
                    local["dedupe_window"] = []
                    local["line_buffer"] = b""
                    local["events_byte_offset"] = 0
                    local["last_seq"] = 0
                    local["stream_reset_pending"] = False

                if not local["stream_reset_pending"]:
                    for event in tail.events:
                        if str(event.get("session_id") or "") != local["active_session_id"]:
                            continue
                        seq = int(event.get("seq") or 0)
                        if seq <= int(local["last_seq"]):
                            continue
                        apply_event_to_histories(
                            history_events=local["history_events"],
                            history_lines=local["history_lines"],
                            history_observed_lines=local["history_observed_lines"],
                            history_choices=local["history_choices"],
                            dedupe_window=local["dedupe_window"],
                            event=event,
                            config=self._cfg,
                            game_id=candidate.game_id,
                        )
                        local["latest_snapshot"] = apply_event_to_snapshot(
                            local["latest_snapshot"], event
                        )
                        local["last_seq"] = seq
                        local["last_seen_data_monotonic"] = now_monotonic

                    local["events_byte_offset"] = tail.next_offset
                    local["events_file_size"] = tail.file_size
                    local["line_buffer"] = tail.line_buffer
        else:
            local["active_data_source"] = _pending_data_source_for_reader_mode(
                reader_mode,
                memory_reader_allowed=memory_reader_allowed,
                ocr_reader_allowed=ocr_reader_allowed,
                memory_reader_candidate_available=memory_reader_candidate_available,
            )
            if not local["bound_game_id"]:
                local["active_game_id"] = ""
                local["active_session_id"] = ""
                local["active_session_meta"] = {}
            local["line_buffer"] = b""

        if warnings:
            local["last_error"] = make_error(
                "; ".join(warnings[:3]),
                source="bridge_reader",
                kind="warning",
            )
        elif (
            isinstance(local.get("last_error"), dict)
            and str(local["last_error"].get("kind") or "") == "warning"
            and not str(local.get("plugin_error") or "")
        ):
            local["last_error"] = {}

        local["current_connection_state"] = derive_connection_state(
            bridge_root=self._cfg.bridge_root,
            plugin_error=str(local["plugin_error"]),
            active_session_id=str(local["active_session_id"]),
            last_seen_data_monotonic=float(local["last_seen_data_monotonic"]),
            now_monotonic=now_monotonic,
            stale_after_seconds=self._cfg.stale_after_seconds,
            stream_reset_pending=bool(local["stream_reset_pending"]),
        )
        interval = next_poll_interval_for_state(
            local["current_connection_state"],
            stream_reset_pending=bool(local["stream_reset_pending"]),
            config=self._cfg,
        )
        if (
            self._cfg.ocr_reader_enabled
            and ocr_trigger_mode == OCR_TRIGGER_MODE_INTERVAL
            and str(ocr_reader_runtime.get("status") or "") in {"starting", "active"}
            and str(local.get("active_data_source") or "") == DATA_SOURCE_OCR_READER
        ):
            interval = min(interval, float(self._cfg.ocr_reader_poll_interval_seconds))
        elif (
            self._cfg.ocr_reader_enabled
            and ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            and str(ocr_reader_runtime.get("status") or "") == "starting"
        ):
            interval = min(interval, float(self._cfg.ocr_reader_poll_interval_seconds))
        elif (
            self._cfg.ocr_reader_enabled
            and ocr_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            and str(ocr_reader_runtime.get("status") or "") == "active"
            and str(local.get("active_data_source") or "") == DATA_SOURCE_OCR_READER
        ):
            interval = min(interval, float(self._cfg.ocr_reader_poll_interval_seconds))
        if self._has_pending_ocr_advance_capture():
            next_pending_delay = self._pending_ocr_advance_capture_delay_remaining()
            interval = min(
                interval,
                next_pending_delay
                if next_pending_delay > 0.0
                else _OCR_AFTER_ADVANCE_SETTLE_POLL_SECONDS,
            )
        local["next_poll_at_monotonic"] = now_monotonic + interval
        self._commit_state(local)

        try:
            self._persist_runtime_state(local)
        except Exception as exc:
            self._record_error(
                make_error(
                    f"persist runtime failed: {exc}",
                    source="store",
                    kind="error",
                )
            )

    @plugin_entry(
        id="galgame_get_status",
        name="获取 galgame 插件状态",
        description="返回当前 bridge 连接状态、绑定游戏、最近错误与模式。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["summary"],
    )
    async def galgame_get_status(self, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        return Ok(await self._build_status_payload_async())

    @plugin_entry(
        id="galgame_install_textractor",
        name="安装 Textractor",
        description="检测并下载安装 TextractorCLI.exe，随后刷新 galgame_plugin 的桥接与读内存状态。",
        input_schema={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False},
            },
        },
        timeout=180.0,
        llm_result_fields=["summary"],
    )
    async def galgame_install_textractor(self, force: bool = False, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if not self._textractor_install_lock.acquire(blocking=False):
            return Err(SdkError("Textractor install is already in progress"))
        current_run_id = self._resolve_current_run_id()
        progress_callback = self._resolve_install_progress_callback(current_run_id)
        try:
            install_result = await install_textractor(
                logger=self.logger,
                configured_path=self._cfg.memory_reader_textractor_path,
                install_target_dir_raw=self._cfg.memory_reader_install_target_dir,
                release_api_url=self._cfg.memory_reader_install_release_api_url,
                timeout_seconds=self._cfg.memory_reader_install_timeout_seconds,
                force=bool(force),
                task_id=current_run_id or None,
                progress_callback=progress_callback,
            )
            await self._poll_bridge(force=True)
            return Ok(
                {
                    "summary": str(install_result.get("summary") or "Textractor 安装完成"),
                    "install_result": install_result,
                    "status": await self._build_status_payload_async(),
                }
            )
        except Exception as exc:
            return Err(SdkError(_format_install_entry_error("Textractor", exc)))
        finally:
            self._textractor_install_lock.release()

    @plugin_entry(
        id="galgame_install_tesseract",
        name="安装 Tesseract",
        description="检测并下载安装本地 Tesseract OCR，随后刷新 galgame_plugin 的 OCR 状态。",
        input_schema={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False},
            },
        },
        timeout=300.0,
        llm_result_fields=["summary"],
    )
    async def galgame_install_tesseract(self, force: bool = False, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if not self._tesseract_install_lock.acquire(blocking=False):
            return Err(SdkError("Tesseract install is already in progress"))
        current_run_id = self._resolve_current_run_id()
        progress_callback = self._resolve_install_progress_callback(current_run_id)
        try:
            install_result = await install_tesseract(
                logger=self.logger,
                configured_path=self._cfg.ocr_reader_tesseract_path,
                install_target_dir_raw=self._cfg.ocr_reader_install_target_dir,
                manifest_url=self._cfg.ocr_reader_install_manifest_url,
                timeout_seconds=self._cfg.ocr_reader_install_timeout_seconds,
                languages=self._cfg.ocr_reader_languages,
                force=bool(force),
                task_id=current_run_id or None,
                progress_callback=progress_callback,
            )
            await self._poll_bridge(force=True)
            return Ok(
                {
                    "summary": str(install_result.get("summary") or "Tesseract 安装完成"),
                    "install_result": install_result,
                    "status": await self._build_status_payload_async(),
                }
            )
        except Exception as exc:
            return Err(SdkError(_format_install_entry_error("Tesseract", exc)))
        finally:
            self._tesseract_install_lock.release()

    @plugin_entry(
        id="galgame_install_rapidocr",
        name="安装 RapidOCR",
        description="检测并下载安装插件隔离的 RapidOCR 运行时，随后刷新 galgame_plugin 的 OCR 状态。",
        input_schema={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False},
            },
        },
        timeout=300.0,
        llm_result_fields=["summary"],
    )
    async def galgame_install_rapidocr(self, force: bool = False, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if not self._rapidocr_install_lock.acquire(blocking=False):
            return Err(SdkError("RapidOCR install is already in progress"))
        current_run_id = self._resolve_current_run_id()
        progress_callback = self._resolve_install_progress_callback(current_run_id)
        try:
            install_result = await install_rapidocr(
                logger=self.logger,
                install_target_dir_raw=self._cfg.rapidocr_install_target_dir,
                manifest_url=self._cfg.rapidocr_install_manifest_url,
                timeout_seconds=self._cfg.rapidocr_install_timeout_seconds,
                engine_type=self._cfg.rapidocr_engine_type,
                lang_type=self._cfg.rapidocr_lang_type,
                model_type=self._cfg.rapidocr_model_type,
                ocr_version=self._cfg.rapidocr_ocr_version,
                force=bool(force),
                task_id=current_run_id or None,
                progress_callback=progress_callback,
            )
            await self._poll_bridge(force=True)
            return Ok(
                {
                    "summary": str(install_result.get("summary") or "RapidOCR 安装完成"),
                    "install_result": install_result,
                    "status": await self._build_status_payload_async(),
                }
            )
        except Exception as exc:
            return Err(SdkError(_format_install_entry_error("RapidOCR", exc)))
        finally:
            self._rapidocr_install_lock.release()

    @plugin_entry(
        id="galgame_install_dxcam",
        name="安装 DXcam",
        description="检测并安装 DXcam 截图依赖，随后刷新 galgame_plugin 的 OCR 截图后端状态。",
        input_schema={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False},
            },
        },
        timeout=180.0,
        llm_result_fields=["summary"],
    )
    async def galgame_install_dxcam(self, force: bool = False, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if not self._dxcam_install_lock.acquire(blocking=False):
            return Err(SdkError("DXcam install is already in progress"))
        current_run_id = self._resolve_current_run_id()
        progress_callback = self._resolve_install_progress_callback(current_run_id)
        try:
            install_result = await install_dxcam(
                logger=self.logger,
                timeout_seconds=self._cfg.ocr_reader_install_timeout_seconds,
                force=bool(force),
                task_id=current_run_id or None,
                progress_callback=progress_callback,
            )
            await self._poll_bridge(force=True)
            return Ok(
                {
                    "summary": str(install_result.get("summary") or "DXcam 安装完成"),
                    "install_result": install_result,
                    "status": await self._build_status_payload_async(),
                }
            )
        except Exception as exc:
            return Err(SdkError(_format_install_entry_error("DXcam", exc)))
        finally:
            self._dxcam_install_lock.release()

    @plugin_entry(
        id="galgame_get_snapshot",
        name="获取 galgame 快照",
        description="返回当前游戏快照和 stale 状态。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["snapshot"],
    )
    async def galgame_get_snapshot(self, **_):
        with self._state_lock:
            payload = build_snapshot_payload(self._state)
        return Ok(payload)

    @plugin_entry(
        id="galgame_get_history",
        name="获取 galgame 历史",
        description="返回最近事件、稳定台词历史和选项历史。",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "minimum": 1},
                "include_events": {"type": "boolean", "default": True},
            },
        },
        llm_result_fields=["stable_lines", "observed_lines", "choices"],
    )
    async def galgame_get_history(self, limit: int = 50, include_events: bool = True, **_):
        with self._state_lock:
            payload = build_history_payload(
                self._state,
                limit=max(1, int(limit)),
                include_events=bool(include_events),
            )
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_mode",
        name="设置 galgame 模式",
        description="设置 silent / companion / choice_advisor 模式，并可选更新通知开关。",
        input_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": sorted(MODES)},
                "push_notifications": {"type": "boolean"},
                "advance_speed": {"type": "string", "enum": sorted(ADVANCE_SPEEDS)},
                "reader_mode": {"type": "string", "enum": sorted(READER_MODES)},
            },
            "required": ["mode"],
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_mode(
        self,
        mode: str,
        push_notifications: bool | None = None,
        advance_speed: str | None = None,
        reader_mode: str | None = None,
        **_,
    ):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if mode not in MODES:
            return Err(SdkError(f"invalid galgame mode: {mode!r}"))
        if advance_speed is not None and advance_speed not in ADVANCE_SPEEDS:
            return Err(SdkError(f"invalid advance speed: {advance_speed!r}"))
        try:
            normalized_reader_mode = _normalize_reader_mode(reader_mode or self._cfg.reader_mode)
        except ValueError as exc:
            return Err(SdkError(str(exc)))

        old_reader_mode = self._cfg.reader_mode
        self._cfg.reader_mode = normalized_reader_mode
        if self._memory_reader_manager is not None:
            self._memory_reader_manager.update_config(self._cfg)
        if self._ocr_reader_manager is not None:
            self._ocr_reader_manager.update_config(self._cfg)
        with self._state_lock:
            self._state.mode = mode
            if push_notifications is not None:
                self._state.push_notifications = bool(push_notifications)
            if advance_speed is not None:
                self._state.advance_speed = advance_speed
            if normalized_reader_mode == READER_MODE_MEMORY:
                self._pending_ocr_advance_captures = 0
            if not self._state.active_session_id:
                self._state.active_data_source = _pending_data_source_for_reader_mode(
                    normalized_reader_mode,
                    memory_reader_allowed=normalized_reader_mode in {READER_MODE_AUTO, READER_MODE_MEMORY},
                    ocr_reader_allowed=normalized_reader_mode in {READER_MODE_AUTO, READER_MODE_OCR},
                    memory_reader_candidate_available=False,
                )
            self._state.next_poll_at_monotonic = 0.0
            self._state_dirty = True
            self._cached_snapshot = None
            payload = {
                "mode": self._state.mode,
                "push_notifications": self._state.push_notifications,
                "advance_speed": self._state.advance_speed,
                "reader_mode": self._cfg.reader_mode,
                "summary": (
                    f"mode={self._state.mode} "
                    f"push_notifications={self._state.push_notifications} "
                    f"advance_speed={self._state.advance_speed} "
                    f"reader_mode={self._cfg.reader_mode}"
                ),
            }
            bound_game_id = self._state.bound_game_id
            persist_push = self._state.push_notifications
            persist_advance_speed = self._state.advance_speed

        try:
            self._persist_preferences(
                bound_game_id=bound_game_id,
                mode=mode,
                push_notifications=persist_push,
                advance_speed=persist_advance_speed,
            )
            self._persist_reader_mode(reader_mode=normalized_reader_mode)
        except Exception as exc:
            self._cfg.reader_mode = old_reader_mode
            if self._memory_reader_manager is not None:
                self._memory_reader_manager.update_config(self._cfg)
            if self._ocr_reader_manager is not None:
                self._ocr_reader_manager.update_config(self._cfg)
            return Err(SdkError(f"persist mode failed: {exc}"))
        self._start_background_bridge_poll()
        if self._game_agent is not None and not mode_allows_agent_actuation(mode):
            try:
                agent_payload = await self._game_agent.apply_mode_change(self._snapshot_state())
                payload["agent"] = json_copy(agent_payload)
            except Exception as exc:
                payload["agent_warning"] = f"apply_mode_change failed: {exc}"
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_ocr_backend",
        name="设置 OCR / 截图后端",
        description="切换 OCR 文本识别后端和截图后端。只影响 OCR 读取，不改变 Agent 点击安全策略。",
        input_schema={
            "type": "object",
            "properties": {
                "backend_selection": {
                    "type": "string",
                    "enum": sorted(_OCR_BACKEND_SELECTIONS),
                },
                "capture_backend": {
                    "type": "string",
                    "enum": sorted(_OCR_CAPTURE_BACKEND_SELECTIONS),
                },
            },
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_ocr_backend(
        self,
        backend_selection: str | None = None,
        capture_backend: str | None = None,
        **_,
    ):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        normalized_backend = str(backend_selection or "").strip().lower() or None
        normalized_capture = str(capture_backend or "").strip().lower() or None
        if normalized_backend is None and normalized_capture is None:
            return Err(SdkError("backend_selection or capture_backend is required"))
        if normalized_backend is not None and normalized_backend not in _OCR_BACKEND_SELECTIONS:
            return Err(SdkError(f"invalid OCR backend: {backend_selection!r}"))
        if normalized_capture is not None and normalized_capture not in _OCR_CAPTURE_BACKEND_SELECTIONS:
            return Err(SdkError(f"invalid OCR capture backend: {capture_backend!r}"))

        old_backend = self._cfg.ocr_reader_backend_selection
        old_capture = self._cfg.ocr_reader_capture_backend
        if normalized_backend is not None:
            self._cfg.ocr_reader_backend_selection = normalized_backend
        if normalized_capture is not None:
            self._cfg.ocr_reader_capture_backend = normalized_capture
        if self._ocr_reader_manager is not None:
            try:
                self._ocr_reader_manager.update_config(self._cfg)
            except Exception as exc:
                if normalized_backend is not None:
                    self._cfg.ocr_reader_backend_selection = old_backend
                if normalized_capture is not None:
                    self._cfg.ocr_reader_capture_backend = old_capture
                return Err(SdkError(f"apply OCR backend failed: {exc}"))

        with self._state_lock:
            self._state.next_poll_at_monotonic = 0.0
            self._state_dirty = True
            self._cached_snapshot = None

        try:
            self._persist_ocr_backend_selection(
                backend_selection=normalized_backend,
                capture_backend=normalized_capture,
            )
        except Exception as exc:
            self._cfg.ocr_reader_backend_selection = old_backend
            self._cfg.ocr_reader_capture_backend = old_capture
            if self._ocr_reader_manager is not None:
                try:
                    self._ocr_reader_manager.update_config(self._cfg)
                except Exception as rollback_exc:
                    _log_plugin_noncritical(
                        self.logger,
                        "warning",
                        "galgame OCR backend rollback update_config failed: {}",
                        rollback_exc,
                    )
            return Err(SdkError(f"persist OCR backend failed: {exc}"))

        self._start_background_bridge_poll()
        payload = {
            "backend_selection": self._cfg.ocr_reader_backend_selection,
            "capture_backend": self._cfg.ocr_reader_capture_backend,
            "summary": (
                f"OCR backend={self._cfg.ocr_reader_backend_selection} "
                f"capture_backend={self._cfg.ocr_reader_capture_backend}"
            ),
        }
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_ocr_timing",
        name="设置 OCR 识别时机",
        description="设置 OCR Reader 触发模式与轮询间隔；DXcam 截图后端会随 OCR 触发。",
        input_schema={
            "type": "object",
            "properties": {
                "poll_interval_seconds": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 10.0,
                },
                "trigger_mode": {
                    "type": "string",
                    "enum": ["interval", "after_advance"],
                    "default": "after_advance",
                },
            },
            "required": ["poll_interval_seconds"],
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_ocr_timing(
        self,
        poll_interval_seconds: float,
        trigger_mode: str | None = None,
        **_,
    ):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        try:
            normalized_interval = float(poll_interval_seconds)
        except (TypeError, ValueError):
            return Err(SdkError("poll_interval_seconds must be a number"))
        if normalized_interval < 0.5 or normalized_interval > 10.0:
            return Err(SdkError("poll_interval_seconds must be between 0.5 and 10.0"))
        try:
            normalized_trigger_mode = _normalize_ocr_trigger_mode(
                trigger_mode or self._cfg.ocr_reader_trigger_mode
            )
        except ValueError as exc:
            return Err(SdkError(str(exc)))

        old_interval = self._cfg.ocr_reader_poll_interval_seconds
        old_trigger_mode = self._cfg.ocr_reader_trigger_mode
        self._cfg.ocr_reader_poll_interval_seconds = normalized_interval
        self._cfg.ocr_reader_trigger_mode = normalized_trigger_mode
        if self._ocr_reader_manager is not None:
            try:
                self._ocr_reader_manager.update_config(self._cfg)
            except Exception as exc:
                self._cfg.ocr_reader_poll_interval_seconds = old_interval
                self._cfg.ocr_reader_trigger_mode = old_trigger_mode
                return Err(SdkError(f"apply OCR timing failed: {exc}"))

        with self._state_lock:
            self._state.next_poll_at_monotonic = 0.0
            self._state_dirty = True
            self._cached_snapshot = None

        try:
            self._persist_ocr_timing(
                poll_interval_seconds=normalized_interval,
                trigger_mode=normalized_trigger_mode,
            )
        except Exception as exc:
            self._cfg.ocr_reader_poll_interval_seconds = old_interval
            self._cfg.ocr_reader_trigger_mode = old_trigger_mode
            if self._ocr_reader_manager is not None:
                try:
                    self._ocr_reader_manager.update_config(self._cfg)
                except Exception as rollback_exc:
                    _log_plugin_noncritical(
                        self.logger,
                        "warning",
                        "galgame OCR timing rollback update_config failed: {}",
                        rollback_exc,
                    )
            return Err(SdkError(f"persist OCR timing failed: {exc}"))

        self._start_background_bridge_poll()
        trigger_mode_label = (
            "点击对白后识别"
            if self._cfg.ocr_reader_trigger_mode == OCR_TRIGGER_MODE_AFTER_ADVANCE
            else "按间隔识别"
        )
        payload = {
            "poll_interval_seconds": self._cfg.ocr_reader_poll_interval_seconds,
            "trigger_mode": self._cfg.ocr_reader_trigger_mode,
            "summary": (
                f"OCR/DXcam {trigger_mode_label}；间隔="
                f"{self._cfg.ocr_reader_poll_interval_seconds:.1f}s"
            ),
        }
        return Ok(payload)

    @plugin_entry(
        id="galgame_bind_game",
        name="绑定 galgame 游戏",
        description="绑定指定 game_id；传空字符串清除手动绑定并恢复自动选择。",
        input_schema={
            "type": "object",
            "properties": {"game_id": {"type": "string", "default": ""}},
            "required": ["game_id"],
        },
        llm_result_fields=["summary"],
    )
    async def galgame_bind_game(self, game_id: str, **_):
        normalized = game_id.strip()
        with self._state_lock:
            available_game_ids = list(self._state.available_game_ids)
        if normalized and normalized not in available_game_ids:
            return Err(SdkError(f"unknown game_id: {normalized!r}"))

        with self._state_lock:
            self._state.bound_game_id = normalized
            self._state_dirty = True
            self._cached_snapshot = None
            bound_game_id = self._state.bound_game_id
            mode = self._state.mode
            push_notifications = self._state.push_notifications
            advance_speed = self._state.advance_speed

        try:
            self._persist_preferences(
                bound_game_id=bound_game_id,
                mode=mode,
                push_notifications=push_notifications,
                advance_speed=advance_speed,
            )
        except Exception as exc:
            return Err(SdkError(f"persist binding failed: {exc}"))

        await self._poll_bridge(force=True)
        with self._state_lock:
            payload = {
                "bound_game_id": self._state.bound_game_id,
                "active_session_id": self._state.active_session_id,
                "summary": f"bound_game_id={self._state.bound_game_id or '(auto)'} active_session_id={self._state.active_session_id}",
            }
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_ocr_capture_profile",
        name="设置 OCR 截图校准",
        description="按进程名保存或清除 OCR Reader 的截图裁剪配置。",
        input_schema={
            "type": "object",
            "properties": {
                "process_name": {"type": "string", "default": ""},
                "stage": {
                    "type": "string",
                    "enum": sorted(OCR_CAPTURE_PROFILE_STAGES),
                    "default": OCR_CAPTURE_PROFILE_STAGE_DEFAULT,
                },
                "save_scope": {
                    "type": "string",
                    "enum": sorted(OCR_CAPTURE_PROFILE_SAVE_SCOPES),
                },
                "left_inset_ratio": {"type": "number", "default": 0.05},
                "right_inset_ratio": {"type": "number", "default": 0.05},
                "top_ratio": {"type": "number", "default": 0.3},
                "bottom_inset_ratio": {"type": "number", "default": 0.3},
                "clear": {"type": "boolean", "default": False},
            },
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_ocr_capture_profile(
        self,
        process_name: str = "",
        stage: str = OCR_CAPTURE_PROFILE_STAGE_DEFAULT,
        left_inset_ratio: float = 0.05,
        right_inset_ratio: float = 0.05,
        top_ratio: float = 0.3,
        bottom_inset_ratio: float = 0.3,
        save_scope: str | None = None,
        clear: bool = False,
        **_,
    ):
        def _parse_ratio(name: str, value: float) -> float:
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a number")
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a number") from exc
            if parsed < 0.0 or parsed >= 1.0:
                raise ValueError(f"{name} must be >= 0.0 and < 1.0")
            return parsed

        with self._state_lock:
            runtime_process_name = str(
                (self._state.ocr_reader_runtime or {}).get("process_name") or ""
            ).strip()
        normalized_process_name = str(process_name or "").strip() or runtime_process_name
        if not normalized_process_name:
            return Err(SdkError("process_name is required"))

        if clear:
            normalized_profile: dict[str, float] | None = None
        else:
            try:
                normalized_profile = {
                    "left_inset_ratio": _parse_ratio("left_inset_ratio", left_inset_ratio),
                    "right_inset_ratio": _parse_ratio("right_inset_ratio", right_inset_ratio),
                    "top_ratio": _parse_ratio("top_ratio", top_ratio),
                    "bottom_inset_ratio": _parse_ratio(
                        "bottom_inset_ratio",
                        bottom_inset_ratio,
                    ),
                }
            except ValueError as exc:
                return Err(SdkError(str(exc)))
            if (
                normalized_profile["left_inset_ratio"]
                + normalized_profile["right_inset_ratio"]
            ) >= 1.0:
                return Err(SdkError("left_inset_ratio + right_inset_ratio must be < 1.0"))
            if (
                normalized_profile["top_ratio"]
                + normalized_profile["bottom_inset_ratio"]
            ) >= 1.0:
                return Err(SdkError("top_ratio + bottom_inset_ratio must be < 1.0"))
        try:
            payload = await self._save_ocr_capture_profile_payload(
                process_name=normalized_process_name,
                stage=stage,
                capture_profile=normalized_profile,
                clear=bool(clear),
                save_scope=save_scope,
            )
        except ValueError as exc:
            return Err(SdkError(str(exc)))
        except Exception as exc:
            return Err(SdkError(f"persist OCR capture profile failed: {exc}"))
        return Ok(payload)

    @plugin_entry(
        id="galgame_auto_recalibrate_ocr_dialogue_profile",
        name="自动重新校准 OCR 对白区",
        description="对当前已附着 OCR 目标窗口自动重校准对白区，并保存到当前窗口分辨率。",
        input_schema={"type": "object", "properties": {}},
        timeout=120.0,
        llm_result_fields=["summary", "sample_text"],
    )
    async def galgame_auto_recalibrate_ocr_dialogue_profile(self, **_):
        if self._ocr_reader_manager is None:
            return Err(SdkError("ocr_reader manager is not initialized"))
        try:
            recalibrated = await asyncio.to_thread(
                self._ocr_reader_manager.auto_recalibrate_dialogue_profile
            )
            payload = await self._save_ocr_capture_profile_payload(
                process_name=str(recalibrated.get("process_name") or ""),
                stage=OCR_CAPTURE_PROFILE_STAGE_DIALOGUE,
                capture_profile=dict(recalibrated.get("capture_profile") or {}),
                clear=False,
                save_scope=OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET,
                width=int(recalibrated.get("window_width") or 0),
                height=int(recalibrated.get("window_height") or 0),
            )
        except ValueError as exc:
            return Err(SdkError(str(exc)))
        except Exception as exc:
            return Err(SdkError(f"auto recalibrate OCR dialogue profile failed: {exc}"))
        payload.update(
            {
                "sample_text": str(recalibrated.get("sample_text") or ""),
                "save_scope": OCR_CAPTURE_PROFILE_SAVE_SCOPE_WINDOW_BUCKET,
                "bucket_key": str(recalibrated.get("bucket_key") or payload.get("bucket_key") or ""),
                "window_width": int(
                    recalibrated.get("window_width") or payload.get("window_width") or 0
                ),
                "window_height": int(
                    recalibrated.get("window_height") or payload.get("window_height") or 0
                ),
                "summary": str(recalibrated.get("summary") or payload.get("summary") or ""),
            }
        )
        return Ok(payload)

    @plugin_entry(
        id="galgame_list_ocr_windows",
        name="列出 OCR 候选窗口",
        description="返回当前 OCR Reader 的可选窗口，可选包含只读排除列表。",
        input_schema={
            "type": "object",
            "properties": {
                "include_excluded": {"type": "boolean", "default": False},
                "force": {"type": "boolean", "default": False},
            },
        },
        llm_result_fields=["summary"],
    )
    async def galgame_list_ocr_windows(
        self,
        include_excluded: bool = False,
        force: bool = False,
        **_,
    ):
        if self._ocr_reader_manager is None:
            return Err(SdkError("ocr_reader manager is not initialized"))
        try:
            payload = await asyncio.to_thread(
                self._ocr_reader_manager.list_windows_snapshot,
                include_excluded=bool(include_excluded),
                force=bool(force),
            )
        except Exception as exc:
            return Err(SdkError(f"list OCR windows failed: {exc}"))
        payload["summary"] = (
            f"eligible={int(payload.get('candidate_count') or 0)} "
            f"excluded={int(payload.get('excluded_candidate_count') or 0)} "
            f"mode={payload.get('target_selection_mode') or 'auto'}"
        )
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_ocr_window_target",
        name="设置 OCR 目标窗口",
        description="锁定或清除 OCR Reader 的手动目标窗口。",
        input_schema={
            "type": "object",
            "properties": {
                "window_key": {"type": "string", "default": ""},
                "clear": {"type": "boolean", "default": False},
            },
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_ocr_window_target(
        self,
        window_key: str = "",
        clear: bool = False,
        **_,
    ):
        if self._ocr_reader_manager is None:
            return Err(SdkError("ocr_reader manager is not initialized"))

        if clear:
            target_payload = {
                "mode": "auto",
                "window_key": "",
                "process_name": "",
                "normalized_title": "",
                "pid": 0,
                "last_known_hwnd": 0,
                "selected_at": "",
            }
            summary = "OCR window target cleared; waiting for manual lock"
        else:
            try:
                target_payload = await asyncio.to_thread(
                    self._ocr_reader_manager.resolve_manual_window_target,
                    window_key,
                )
            except ValueError as exc:
                return Err(SdkError(str(exc)))
            except Exception as exc:
                return Err(SdkError(f"resolve OCR window target failed: {exc}"))
            summary = (
                f"OCR window target locked to {target_payload.get('process_name') or '(unknown)'}"
            )

        try:
            self._persist.persist_ocr_window_target(target_payload)
        except Exception as exc:
            return Err(SdkError(f"persist OCR window target failed: {exc}"))

        with self._state_lock:
            self._state.ocr_window_target = json_copy(target_payload)
            self._state_dirty = True
            self._cached_snapshot = None
        self._ocr_reader_manager.update_window_target(target_payload)
        background_poll_started = self._start_background_bridge_poll()
        return Ok(
            {
                "window_target": json_copy(target_payload),
                "cleared": bool(clear),
                "summary": summary,
                "background_poll_started": background_poll_started,
            }
        )

    @plugin_entry(
        id="galgame_open_ui",
        name="打开 galgame UI",
        description="返回 galgame_plugin 静态 UI 的访问路径。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["message"],
    )
    async def galgame_open_ui(self, **_):
        payload = build_open_ui_payload(
            plugin_id=self.plugin_id,
            available=self.get_static_ui_config() is not None,
        )
        return Ok(payload)

    @plugin_entry(
        id="galgame_explain_line",
        name="解释当前或指定台词",
        description="对当前快照或指定 line_id 对应的台词进行解释。",
        input_schema={
            "type": "object",
            "properties": {"line_id": {"type": "string", "default": ""}},
        },
        timeout=45.0,
        llm_result_fields=["explanation", "diagnostic"],
    )
    async def galgame_explain_line(self, line_id: str = "", **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        try:
            context = build_explain_context(local, line_id=line_id.strip())
        except ValueError as exc:
            context = {
                "line_id": "",
                "speaker": "",
                "text": "",
                "scene_id": "",
                "route_id": "",
                "evidence": [],
            }
            return Ok(
                build_explain_degraded_result(
                    context,
                    diagnostic=str(exc) or build_ocr_context_diagnostic(local),
                )
            )
        payload = apply_input_degraded_result(
            await self._llm_gateway.explain_line(context),
            context=context,
        )
        payload["line_id"] = str(context.get("line_id") or "")
        payload["speaker"] = str(context.get("speaker") or "")
        payload["text"] = str(context.get("text") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_summarize_scene",
        name="总结当前场景",
        description="总结当前场景或指定 scene_id 的最近剧情进展。",
        input_schema={
            "type": "object",
            "properties": {"scene_id": {"type": "string", "default": ""}},
        },
        timeout=45.0,
        llm_result_fields=["summary", "diagnostic"],
    )
    async def galgame_summarize_scene(self, scene_id: str = "", **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        context = build_summarize_context(local, scene_id=scene_id.strip())
        snapshot = context.get("current_snapshot") if isinstance(context.get("current_snapshot"), dict) else {}
        if not list(context.get("recent_lines") or []) and not str(snapshot.get("text") or ""):
            return Ok(
                build_summarize_degraded_result(
                    context,
                    diagnostic=build_ocr_context_diagnostic(local),
                )
            )
        payload = apply_input_degraded_result(
            await self._llm_gateway.summarize_scene(context),
            context=context,
        )
        payload["scene_id"] = str(context.get("scene_id") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_suggest_choice",
        name="建议当前选项",
        description="对当前可见选项给出推荐顺位与理由。",
        input_schema={"type": "object", "properties": {}},
        timeout=45.0,
        llm_result_fields=["choices", "diagnostic"],
    )
    async def galgame_suggest_choice(self, **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        context = build_suggest_context(local)
        if not context["visible_choices"]:
            return Ok(
                apply_input_degraded_result(
                    build_suggest_degraded_result(
                        context,
                        diagnostic="gateway_unavailable: no visible choices",
                    ),
                    context=context,
                )
            )
        payload = apply_input_degraded_result(
            await self._llm_gateway.suggest_choice(context),
            context=context,
        )
        payload["scene_id"] = str(context.get("scene_id") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_agent_command",
        name="向 Game LLM Agent 发送指令",
        description="查询 Agent 状态、上下文、发送消息或控制待机。",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "query_status",
                        "query_context",
                        "send_message",
                        "set_standby",
                        "list_messages",
                        "ack_message",
                    ],
                },
                "message": {"type": "string", "default": ""},
                "context_query": {"type": "string", "default": ""},
                "message_id": {"type": "string", "default": ""},
                "direction": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 50},
                "standby": {"type": "boolean"},
            },
            "required": ["action"],
        },
        timeout=45.0,
        llm_result_fields=["result", "status"],
    )
    async def galgame_agent_command(
        self,
        action: str,
        message: str = "",
        context_query: str = "",
        message_id: str = "",
        direction: str = "",
        limit: int = 50,
        standby: bool | None = None,
        **_,
    ):
        if self._game_agent is None:
            return Err(SdkError("galgame_plugin game agent is not initialized"))
        local = self._snapshot_state()
        if action == "query_status":
            return Ok(await self._game_agent.query_status(local))
        if action == "query_context":
            if not context_query.strip():
                return Err(SdkError("context_query is required for query_context"))
            return Ok(
                await self._game_agent.query_context(
                    local,
                    context_query=context_query.strip(),
                )
            )
        if action == "send_message":
            if not message.strip():
                return Err(SdkError("message is required for send_message"))
            return Ok(
                await self._game_agent.send_message(
                    local,
                    message=message.strip(),
                )
            )
        if action == "set_standby":
            if standby is None:
                return Err(SdkError("standby is required for set_standby"))
            return Ok(await self._game_agent.set_standby(local, standby=bool(standby)))
        if action == "list_messages":
            return Ok(
                await self._game_agent.list_messages(
                    local,
                    direction=direction,
                    limit=int(limit or 50),
                )
            )
        if action == "ack_message":
            if not message_id.strip():
                return Err(SdkError("message_id is required for ack_message"))
            return Ok(
                await self._game_agent.ack_message(
                    local,
                    message_id=message_id.strip(),
                )
            )
        return Err(SdkError(f"unsupported agent action: {action!r}"))


GalgameBridgePlugin = GalgamePlugin
