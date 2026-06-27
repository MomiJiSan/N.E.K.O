from __future__ import annotations

import time
from typing import Any

STORE_KEY = "tft_replay_snapshots"
MAX_SNAPSHOTS = 80


def build_snapshot(payload: dict[str, Any], *, note: str = "") -> dict[str, Any]:
    return {
        "created_at": time.time(),
        "profile": payload.get("profile") or payload.get("profile_id") or "tft",
        "source": payload.get("source"),
        "state": payload.get("state") or {},
        "insights": payload.get("insights") or [],
        "note": note,
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
