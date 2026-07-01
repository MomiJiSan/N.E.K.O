from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VlmFallbackPolicy:
    confidence_threshold: float = 0.5
    model_role: str = "vision"
    max_tokens: int | None = None
    require_desensitization_for_type_d: bool = True
    send_full_frame: bool = False


def build_vlm_fallback_plan(
    vision: dict[str, Any],
    *,
    user_requested: bool = False,
    state_changed: bool = False,
    unknown_ui: bool = False,
    policy: VlmFallbackPolicy | None = None,
) -> dict[str, Any]:
    active_policy = policy or VlmFallbackPolicy()
    reason = _fallback_reason(
        vision,
        user_requested=user_requested,
        state_changed=state_changed,
        unknown_ui=unknown_ui or _has_unknown_ui(vision),
        confidence_threshold=active_policy.confidence_threshold,
    )
    status = "planned" if reason != "not_needed" else "skipped"
    return {
        "status": status,
        "reason": reason,
        "model_role": active_policy.model_role,
        "max_tokens": active_policy.max_tokens or _default_vision_max_tokens(),
        "requires_desensitization": bool(active_policy.require_desensitization_for_type_d),
        "send_full_frame": bool(active_policy.send_full_frame),
        "input_policy": _build_input_policy(active_policy),
        "external_call_executed": False,
        "merge_target": "vision",
        "notes": [
            "This is a local fallback plan only; no vision model call is executed here.",
            "TYPE_D frames must be cropped or desensitized before any future external VLM call.",
        ],
    }


def apply_vlm_fallback_plan(vision: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(vision)
    diagnostics = payload.setdefault("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        payload["diagnostics"] = diagnostics
    diagnostics["vlm_fallback"] = deepcopy(plan)

    analyzers = diagnostics.setdefault("analyzers", {})
    if isinstance(analyzers, dict):
        analyzers["vlm"] = {
            "status": plan.get("status"),
            "reason": plan.get("reason"),
            "external_call_executed": False,
        }

    privacy = payload.setdefault("privacy", {})
    if not isinstance(privacy, dict):
        privacy = {}
        payload["privacy"] = privacy
    privacy["stores_raw_image"] = False
    privacy["external_model_calls"] = False
    if plan.get("requires_desensitization"):
        privacy["requires_desensitization"] = True
    if isinstance(plan.get("input_policy"), dict):
        privacy["vlm_input_policy"] = deepcopy(plan["input_policy"])

    payload["model_calls"] = []
    return payload


def _fallback_reason(
    vision: dict[str, Any],
    *,
    user_requested: bool,
    state_changed: bool,
    unknown_ui: bool,
    confidence_threshold: float,
) -> str:
    if user_requested:
        return "user_requested"
    if state_changed:
        return "state_changed"
    if unknown_ui:
        return "unknown_ui"
    if _vision_confidence(vision) < confidence_threshold:
        return "low_confidence"
    return "not_needed"


def _vision_confidence(vision: dict[str, Any]) -> float:
    confidences: list[float] = []
    for value in (
        vision.get("confidence"),
        (vision.get("scene") or {}).get("confidence") if isinstance(vision.get("scene"), dict) else None,
    ):
        try:
            if value is not None:
                confidences.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(confidences) if confidences else 0.0


def _has_unknown_ui(vision: dict[str, Any]) -> bool:
    ui_items = vision.get("ui") if isinstance(vision, dict) else None
    if not isinstance(ui_items, list):
        return False
    for item in ui_items:
        if not isinstance(item, dict):
            continue
        ui_type = str(item.get("type") or "").lower()
        label = str(item.get("label") or "").lower()
        status = str(item.get("status") or "").lower()
        if ui_type in {"popup_dialog", "dialog", "modal", "overlay"} and label in {"", "unknown", "uncertain"}:
            return True
        if status in {"unknown", "uncertain", "needs_vlm"}:
            return True
    return False


def _build_input_policy(policy: VlmFallbackPolicy) -> dict[str, Any]:
    requires_desensitization = bool(policy.require_desensitization_for_type_d)
    send_full_frame = bool(policy.send_full_frame)
    return {
        "preferred_payload": "desensitized_frame" if send_full_frame else "cropped_regions",
        "send_full_frame": send_full_frame,
        "requires_desensitization": requires_desensitization,
        "allow_regions": [
            "detected_ui_regions",
            "gameplay_view",
        ],
        "redact_regions": [
            "chat_area",
            "player_names",
            "account_identifiers",
            "scoreboard_names",
        ]
        if requires_desensitization
        else [],
        "raw_image_logging": False,
    }


def _default_vision_max_tokens() -> int:
    try:
        from config import VISION_ANALYSIS_MAX_TOKENS

        return int(VISION_ANALYSIS_MAX_TOKENS)
    except Exception:
        return 500
