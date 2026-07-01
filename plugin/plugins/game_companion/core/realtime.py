from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RealtimeInsightSession:
    enabled: bool = False
    profile_id: str = "tft"
    interval_seconds: float = 2.0
    debounce_seconds: float = 1.0
    frame_count: int = 0
    last_frame_at: float | None = None
    last_result: dict[str, Any] | None = None
    stable_result: dict[str, Any] | None = None
    last_signature: str = ""
    last_signature_at: float = 0.0
    last_content_hash: str = ""
    last_frame_changed: bool = False
    repeated_frame_count: int = 0
    warnings: list[dict[str, str]] = field(default_factory=list)

    def configure(
        self,
        *,
        enabled: bool | None = None,
        profile_id: str | None = None,
        interval_seconds: float | None = None,
        debounce_seconds: float | None = None,
    ) -> dict[str, Any]:
        if enabled is not None:
            self.enabled = bool(enabled)
        if profile_id:
            self.profile_id = str(profile_id).strip().lower()
        if interval_seconds is not None:
            self.interval_seconds = max(1.0, min(10.0, float(interval_seconds)))
        if debounce_seconds is not None:
            self.debounce_seconds = max(0.0, min(10.0, float(debounce_seconds)))
        return self.to_dict()

    def ingest(self, result: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        self.frame_count += 1
        self.last_frame_at = now
        self.last_result = result
        previous_signature = self.last_signature
        signature = _result_signature(result)
        content_hash = _result_content_hash(result)
        if content_hash:
            self.last_frame_changed = content_hash != self.last_content_hash
            self.repeated_frame_count = (
                self.repeated_frame_count + 1
                if not self.last_frame_changed and self.last_content_hash
                else 1
            )
            self.last_content_hash = content_hash
        else:
            self.last_frame_changed = signature != previous_signature
            self.repeated_frame_count = (
                self.repeated_frame_count + 1
                if not self.last_frame_changed and previous_signature
                else 1
            )
        if signature != self.last_signature:
            self.last_signature = signature
            self.last_signature_at = now
        if not self.stable_result or now - self.last_signature_at >= self.debounce_seconds:
            self.stable_result = result
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "profile_id": self.profile_id,
            "interval_seconds": self.interval_seconds,
            "debounce_seconds": self.debounce_seconds,
            "frame_count": self.frame_count,
            "last_frame_at": self.last_frame_at,
            "last_result": self.last_result,
            "stable_result": self.stable_result,
            "last_content_hash": self.last_content_hash,
            "last_frame_changed": self.last_frame_changed,
            "repeated_frame_count": self.repeated_frame_count,
            "warnings": list(self.warnings),
        }


def _result_signature(result: dict[str, Any]) -> str:
    state = result.get("state") if isinstance(result, dict) else {}
    insights = result.get("insights") if isinstance(result, dict) else []
    scene = _result_scene(result)
    return repr(
        (
            result.get("profile") or result.get("profile_id") if isinstance(result, dict) else None,
            _result_content_hash(result),
            scene.get("label") if scene else None,
            scene.get("confidence") if scene else None,
            state.get("stage") if isinstance(state, dict) else None,
            state.get("level") if isinstance(state, dict) else None,
            state.get("gold") if isinstance(state, dict) else None,
            tuple((item.get("type"), item.get("title")) for item in insights if isinstance(item, dict)),
        )
    )


def _result_content_hash(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return ""
    vision = result.get("vision")
    if not isinstance(vision, dict):
        return ""
    frame = vision.get("frame")
    if not isinstance(frame, dict):
        return ""
    content_hash = frame.get("content_hash")
    return str(content_hash) if content_hash else ""


def _result_scene(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    vision = result.get("vision")
    if not isinstance(vision, dict):
        return {}
    scene = vision.get("scene")
    return scene if isinstance(scene, dict) else {}
