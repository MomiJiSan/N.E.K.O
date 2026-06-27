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
        signature = _result_signature(result)
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
            "warnings": list(self.warnings),
        }


def _result_signature(result: dict[str, Any]) -> str:
    state = result.get("state") if isinstance(result, dict) else {}
    insights = result.get("insights") if isinstance(result, dict) else []
    return repr(
        (
            state.get("stage") if isinstance(state, dict) else None,
            state.get("level") if isinstance(state, dict) else None,
            state.get("gold") if isinstance(state, dict) else None,
            tuple((item.get("type"), item.get("title")) for item in insights if isinstance(item, dict)),
        )
    )
