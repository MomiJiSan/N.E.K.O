from __future__ import annotations

import re
import time
from typing import Any

STORE_KEY = "tft_replay_snapshots"
NEKO_CONTEXT_QUEUE_STORE_KEY = "game_companion_neko_context_queue"
MAX_SNAPSHOTS = 80
MAX_NEKO_CONTEXT_PACKETS = 40
MAX_CONTEXT_TEXT_LENGTH = 280
DATA_URL_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/][^\s'\"}\])]+")


def build_snapshot(payload: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "profile": payload.get("profile") or payload.get("profile_id") or "tft",
        "source": _source_digest(payload),
        "state": payload.get("state") or {},
        "insights": payload.get("insights") or [],
        "note": _redact_context_text(note),
    }


def load_snapshots(store: Any) -> list[dict[str, Any]]:
    try:
        raw = store._read_value(STORE_KEY, [])
    except Exception:
        return []
    return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def append_snapshot(store: Any, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = [*load_snapshots(store), snapshot][-MAX_SNAPSHOTS:]
    try:
        store._write_value(STORE_KEY, snapshots)
    except Exception:
        pass
    return snapshots


def clear_snapshots(store: Any) -> None:
    try:
        store._write_value(STORE_KEY, [])
    except Exception:
        pass


def list_neko_context_queue(store: Any) -> list[dict[str, Any]]:
    try:
        raw = store._read_value(NEKO_CONTEXT_QUEUE_STORE_KEY, [])
    except Exception:
        return []
    return [_sanitize_context_packet(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def enqueue_neko_context_packet(store: Any, packet: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_context_packet(packet)
    queue = [*list_neko_context_queue(store), sanitized][-MAX_NEKO_CONTEXT_PACKETS:]
    try:
        store._write_value(NEKO_CONTEXT_QUEUE_STORE_KEY, queue)
    except Exception:
        pass
    return {
        "queued": True,
        "queue_size": len(queue),
        "packet": sanitized,
    }


def dequeue_neko_context_packet(store: Any) -> dict[str, Any]:
    queue = list_neko_context_queue(store)
    if not queue:
        return {"available": False, "queue_size": 0, "packet": None}
    packet = queue[0]
    remaining = queue[1:]
    try:
        store._write_value(NEKO_CONTEXT_QUEUE_STORE_KEY, remaining)
    except Exception:
        pass
    return {"available": True, "queue_size": len(remaining), "packet": packet}


def build_training_prompt(snapshot: dict[str, Any]) -> dict[str, Any]:
    state = snapshot.get("state") if isinstance(snapshot, dict) else {}
    insights = snapshot.get("insights") if isinstance(snapshot, dict) else []
    return {
        "type": "tft_training_prompt",
        "question": "根据这个局面，先判断当前羁绊、装备倾向和可观察的升星机会。",
        "state": state or {},
        "insights": insights or [],
        "feedback_options": [
            "羁绊判断准确",
            "装备方向需要修正",
            "单位识别需要修正",
            "洞察表达过强",
        ],
    }


def build_neko_context_packet(payload: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    profile = _redact_context_text(payload.get("profile") or payload.get("profile_id") or "tft")
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    tft_state = payload.get("tft_state") if isinstance(payload.get("tft_state"), dict) else {}
    insights = payload.get("insights") if isinstance(payload.get("insights"), list) else []
    vision = payload.get("vision") if isinstance(payload.get("vision"), dict) else {}
    privacy = vision.get("privacy") if isinstance(vision.get("privacy"), dict) else {}
    return {
        "type": "game_companion_neko_context_packet",
        "schema_version": 1,
        "created_at": time.time(),
        "profile": profile,
        "delivery": {
            "mode": "queued_non_interrupting",
            "target": "yui_context",
            "interrupt_allowed": False,
        },
        "yui_boundary": {
            "agent_mode": "assistance_system",
            "no_roleplay_identity": True,
            "game_llm_is_auxiliary": True,
        },
        "memory_policy": "summary_only",
        "privacy": {
            "raw_image_included": False,
            "source_path_included": False,
            "external_model_calls": bool(privacy.get("external_model_calls") is True),
        },
        "state_digest": _state_digest(state, tft_state=tft_state),
        "events": _context_events(insights),
        "summary": _context_summary(state, insights, tft_state=tft_state),
        "ttl_seconds": 120,
        "note": _redact_context_text(note),
    }


def _sanitize_context_packet(packet: Any) -> dict[str, Any]:
    if not isinstance(packet, dict):
        packet = {}
    allowed_keys = {
        "type",
        "schema_version",
        "created_at",
        "profile",
        "delivery",
        "yui_boundary",
        "memory_policy",
        "privacy",
        "state_digest",
        "events",
        "summary",
        "ttl_seconds",
        "note",
    }
    sanitized = {key: packet.get(key) for key in allowed_keys if key in packet}
    sanitized["type"] = "game_companion_neko_context_packet"
    sanitized["schema_version"] = int(sanitized.get("schema_version") or 1)
    sanitized["delivery"] = {
        "mode": "queued_non_interrupting",
        "target": "yui_context",
        "interrupt_allowed": False,
    }
    sanitized["memory_policy"] = "summary_only"
    privacy = sanitized.get("privacy") if isinstance(sanitized.get("privacy"), dict) else {}
    sanitized["privacy"] = {
        "raw_image_included": False,
        "source_path_included": False,
        "external_model_calls": bool(privacy.get("external_model_calls") is True),
    }
    sanitized["state_digest"] = dict(sanitized.get("state_digest")) if isinstance(sanitized.get("state_digest"), dict) else {}
    sanitized["events"] = _sanitize_context_events(sanitized.get("events"))
    sanitized["summary"] = _redact_context_text(sanitized.get("summary") or "No stable game context is available yet.")
    sanitized["note"] = _redact_context_text(sanitized.get("note") or "")
    return sanitized


def _sanitize_context_events(events: Any) -> list[dict[str, Any]]:
    sanitized_events: list[dict[str, Any]] = []
    if not isinstance(events, list):
        return sanitized_events
    for event in events[:5]:
        if not isinstance(event, dict):
            continue
        sanitized_events.append(
            {
                "type": str(event.get("type") or "observation"),
                "title": _redact_context_text(event.get("title") or ""),
                "detail": _redact_context_text(event.get("detail") or ""),
                "confidence": event.get("confidence"),
            }
        )
    return sanitized_events


def _state_digest(state: dict[str, Any], *, tft_state: dict[str, Any] | None = None) -> dict[str, Any]:
    digest = {
        key: state.get(key)
        for key in ("stage", "level", "gold")
        if state.get(key) is not None
    }
    if isinstance(tft_state, dict) and tft_state:
        digest["tft"] = _tft_state_digest(tft_state)
    return digest


def _tft_state_digest(tft_state: dict[str, Any]) -> dict[str, Any]:
    digest: dict[str, Any] = {
        "layout": _redact_context_text(tft_state.get("layout") or "unknown"),
        "readiness": _redact_context_text(tft_state.get("readiness") or "unknown"),
        "summary": _redact_context_text(tft_state.get("summary") or ""),
    }
    shop = tft_state.get("shop") if isinstance(tft_state.get("shop"), dict) else {}
    if shop:
        digest["shop"] = {
            "slot_count": shop.get("slot_count"),
            "occupied_count": shop.get("occupied_count"),
            "partial_count": shop.get("partial_count") or 0,
            "units": _tft_shop_units_digest(shop.get("slots")),
            "slot_issues": _tft_shop_slot_issues_digest(shop.get("slot_issues")),
        }
    augment = tft_state.get("augment") if isinstance(tft_state.get("augment"), dict) else {}
    if augment:
        digest["augment"] = {
            "option_count": augment.get("option_count"),
            "options": _tft_augment_options_digest(augment.get("options")),
        }
    blockers = tft_state.get("blockers") if isinstance(tft_state.get("blockers"), list) else []
    digest["blockers"] = [
        {
            "code": _redact_context_text(item.get("code") or ""),
            **({"slots": [int(slot) for slot in item.get("slots", []) if isinstance(slot, int)]} if isinstance(item.get("slots"), list) else {}),
        }
        for item in blockers[:5]
        if isinstance(item, dict)
    ]
    return digest


def _tft_shop_units_digest(slots: Any) -> list[dict[str, Any]]:
    if not isinstance(slots, list):
        return []
    units: list[dict[str, Any]] = []
    for slot in slots:
        if not isinstance(slot, dict) or slot.get("state") != "occupied":
            continue
        units.append(
            {
                "slot": slot.get("slot"),
                "name": _redact_context_text(slot.get("name") or ""),
                "cost": slot.get("cost"),
                "confidence": slot.get("confidence"),
                "missing_fields": _tft_missing_fields_digest(slot.get("missing_fields")),
            }
        )
    return units[:5]


def _tft_shop_slot_issues_digest(slot_issues: Any) -> list[dict[str, Any]]:
    if not isinstance(slot_issues, list):
        return []
    issues = []
    for issue in slot_issues[:5]:
        if not isinstance(issue, dict):
            continue
        issues.append(
            {
                "slot": issue.get("slot"),
                "state": _redact_context_text(issue.get("state") or ""),
                "missing_fields": _tft_missing_fields_digest(issue.get("missing_fields")),
            }
        )
    return issues


def _tft_missing_fields_digest(fields: Any) -> list[str]:
    if not isinstance(fields, list):
        return []
    allowed = {"name", "cost"}
    return [field for field in (_redact_context_text(item) for item in fields[:4]) if field in allowed]


def _tft_augment_options_digest(options: Any) -> list[dict[str, Any]]:
    if not isinstance(options, list):
        return []
    return [
        {
            "slot": option.get("slot"),
            "title": _redact_context_text(option.get("title") or ""),
            "confidence": option.get("confidence"),
        }
        for option in options[:3]
        if isinstance(option, dict)
    ]


def _context_events(insights: list[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for insight in insights[:5]:
        if not isinstance(insight, dict):
            continue
        events.append(
            {
                "type": str(insight.get("type") or "observation"),
                "title": _redact_context_text(insight.get("title") or ""),
                "detail": _redact_context_text(insight.get("detail") or ""),
                "confidence": insight.get("confidence"),
            }
        )
    return events


def _context_summary(state: dict[str, Any], insights: list[Any], *, tft_state: dict[str, Any] | None = None) -> str:
    parts = []
    if isinstance(tft_state, dict) and tft_state.get("summary"):
        parts.append(_redact_context_text(tft_state.get("summary") or ""))
    stage = state.get("stage") or state.get("round")
    level = state.get("level")
    gold = state.get("gold")
    if stage:
        parts.append(f"stage {stage}")
    if level is not None:
        parts.append(f"level {level}")
    if gold is not None:
        parts.append(f"{gold} gold")
    titles = [
        _redact_context_text(insight.get("title") or "")
        for insight in insights
        if isinstance(insight, dict) and insight.get("title")
    ]
    if titles:
        parts.append("; ".join(titles[:3]))
    return _redact_context_text(" | ".join(parts)) if parts else "No stable game context is available yet."


def _source_digest(payload: Any) -> dict[str, Any] | None:
    source = payload.get("source") if isinstance(payload, dict) else None
    if not isinstance(source, dict):
        return None
    digest = {
        key: source.get(key)
        for key in ("type", "width", "height", "content_hash")
        if source.get(key) is not None
    }
    if "content_hash" not in digest and isinstance(payload, dict):
        vision = payload.get("vision")
        frame = vision.get("frame") if isinstance(vision, dict) else None
        if isinstance(frame, dict) and frame.get("content_hash") is not None:
            digest["content_hash"] = frame["content_hash"]
    return digest


def _redact_context_text(value: Any) -> str:
    text = str(value or "")
    text = DATA_URL_RE.sub("[redacted_image_data]", text)
    text = WINDOWS_PATH_RE.sub("[redacted_path]", text)
    if len(text) > MAX_CONTEXT_TEXT_LENGTH:
        text = f"{text[:MAX_CONTEXT_TEXT_LENGTH]}..."
    return text
