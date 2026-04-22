from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODE_SILENT = "silent"
MODE_COMPANION = "companion"
MODE_CHOICE_ADVISOR = "choice_advisor"
MODES = frozenset({MODE_SILENT, MODE_COMPANION, MODE_CHOICE_ADVISOR})

DATA_SOURCE_NONE = "none"
DATA_SOURCE_BRIDGE_SDK = "bridge_sdk"
DATA_SOURCE_MEMORY_READER = "memory_reader"
DATA_SOURCES = frozenset(
    {DATA_SOURCE_NONE, DATA_SOURCE_BRIDGE_SDK, DATA_SOURCE_MEMORY_READER}
)

AGENT_STATUS_ACTIVE = "active"
AGENT_STATUS_STANDBY = "standby"
AGENT_STATUS_ERROR = "error"
AGENT_STATUSES = frozenset(
    {AGENT_STATUS_ACTIVE, AGENT_STATUS_STANDBY, AGENT_STATUS_ERROR}
)

STATE_DISCONNECTED = "disconnected"
STATE_IDLE = "idle"
STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ERROR = "error"
CONNECTION_STATES = frozenset(
    {STATE_DISCONNECTED, STATE_IDLE, STATE_ACTIVE, STATE_STALE, STATE_ERROR}
)

STORE_BOUND_GAME_ID = "bound_game_id"
STORE_MODE = "mode"
STORE_PUSH_NOTIFICATIONS = "push_notifications"
STORE_SESSION_ID = "session_id"
STORE_EVENTS_BYTE_OFFSET = "events_byte_offset"
STORE_EVENTS_FILE_SIZE = "events_file_size"
STORE_LAST_SEQ = "last_seq"
STORE_DEDUPE_WINDOW = "dedupe_window"
STORE_LAST_ERROR = "last_error"
STORE_KEYS = (
    STORE_BOUND_GAME_ID,
    STORE_MODE,
    STORE_PUSH_NOTIFICATIONS,
    STORE_SESSION_ID,
    STORE_EVENTS_BYTE_OFFSET,
    STORE_EVENTS_FILE_SIZE,
    STORE_LAST_SEQ,
    STORE_DEDUPE_WINDOW,
    STORE_LAST_ERROR,
)

DEFAULT_SAVE_CONTEXT = {
    "kind": "unknown",
    "slot_id": "",
    "display_name": "",
}


def json_copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _string(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def sanitize_save_context(value: object) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    return {
        "kind": _string(raw.get("kind"), DEFAULT_SAVE_CONTEXT["kind"]),
        "slot_id": _string(raw.get("slot_id"), DEFAULT_SAVE_CONTEXT["slot_id"]),
        "display_name": _string(
            raw.get("display_name"), DEFAULT_SAVE_CONTEXT["display_name"]
        ),
    }


def sanitize_choice(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        "choice_id": _string(raw.get("choice_id")),
        "text": _string(raw.get("text")),
        "index": _int(raw.get("index"), 0),
        "enabled": _bool(raw.get("enabled"), True),
    }


def sanitize_metadata(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {str(key): json_copy(item) for key, item in raw.items()}


def sanitize_snapshot_state(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    choices_obj = raw.get("choices")
    choices = (
        [sanitize_choice(item) for item in choices_obj]
        if isinstance(choices_obj, list)
        else []
    )
    return {
        "speaker": _string(raw.get("speaker")),
        "text": _string(raw.get("text")),
        "choices": choices,
        "scene_id": _string(raw.get("scene_id")),
        "line_id": _string(raw.get("line_id")),
        "route_id": _string(raw.get("route_id")),
        "is_menu_open": _bool(raw.get("is_menu_open"), bool(choices)),
        "save_context": sanitize_save_context(raw.get("save_context")),
        "ts": _string(raw.get("ts")),
    }


def sanitize_session_snapshot(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    return {
        "protocol_version": _int(raw.get("protocol_version"), 1),
        "game_id": _string(raw.get("game_id")),
        "game_title": _string(raw.get("game_title")),
        "engine": _string(raw.get("engine")),
        "session_id": _string(raw.get("session_id")),
        "started_at": _string(raw.get("started_at")),
        "last_seq": max(0, _int(raw.get("last_seq"), 0)),
        "locale": _string(raw.get("locale")),
        "bridge_sdk_version": _string(raw.get("bridge_sdk_version")),
        "metadata": sanitize_metadata(raw.get("metadata")),
        "state": sanitize_snapshot_state(raw.get("state")),
    }


def sanitize_event(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("payload")
    normalized_payload = dict(payload) if isinstance(payload, dict) else {}
    return {
        "protocol_version": _int(value.get("protocol_version"), 1),
        "seq": max(0, _int(value.get("seq"), 0)),
        "ts": _string(value.get("ts")),
        "type": _string(value.get("type")),
        "session_id": _string(value.get("session_id")),
        "game_id": _string(value.get("game_id")),
        "payload": normalized_payload,
    }


def make_error(
    message: str,
    *,
    source: str,
    kind: str = "warning",
    ts: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "source": source,
        "message": message,
        "ts": ts,
    }
    if details:
        payload["details"] = dict(details)
    return payload


@dataclass(slots=True)
class GalgameConfig:
    bridge_root: Path
    active_poll_interval_seconds: float
    idle_poll_interval_seconds: float
    stale_after_seconds: float
    history_events_limit: int
    history_lines_limit: int
    history_choices_limit: int
    dedupe_window_limit: int
    warmup_replay_bytes_limit: int
    warmup_replay_events_limit: int
    default_mode: str
    push_notifications: bool
    llm_call_timeout_seconds: float
    llm_max_in_flight: int
    llm_request_cache_ttl_seconds: float
    llm_target_entry_ref: str
    memory_reader_enabled: bool
    memory_reader_textractor_path: str
    memory_reader_auto_detect: bool
    memory_reader_poll_interval_seconds: float


@dataclass(slots=True)
class SessionCandidate:
    game_id: str
    session_path: Path
    events_path: Path
    session: dict[str, Any]
    data_source: str = DATA_SOURCE_BRIDGE_SDK

    @property
    def session_id(self) -> str:
        return _string(self.session.get("session_id"))

    @property
    def last_seq(self) -> int:
        return max(0, _int(self.session.get("last_seq"), 0))

    @property
    def sort_key(self) -> tuple[int, str, str]:
        state = self.session.get("state")
        state_ts = _string(state.get("ts")) if isinstance(state, dict) else ""
        return (self.last_seq, state_ts, _string(self.session.get("started_at")))
