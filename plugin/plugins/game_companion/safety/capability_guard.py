from __future__ import annotations

from typing import Any

from .models import Capability, DEFAULT_DENIED_CAPABILITIES, GameType, RuntimeMode


HARD_DENIED_ONLINE_CAPABILITIES: frozenset[Capability] = frozenset(DEFAULT_DENIED_CAPABILITIES)


def evaluate_profile_capability(
    profile: Any,
    capability: Capability | str,
    *,
    profile_id: str | None = None,
    runtime_mode: RuntimeMode | str | None = None,
) -> dict[str, Any]:
    requested = Capability.coerce(capability)
    normalized_profile_id = str(profile_id or getattr(profile, "profile_id", "") or "").strip().lower()
    if profile is None:
        return {
            "allowed": False,
            "reason": "unknown_profile",
            "profile_id": normalized_profile_id,
            "capability": requested.value,
        }

    game_type = getattr(profile, "game_type", GameType.TYPE_D)
    if not isinstance(game_type, GameType):
        game_type = GameType(str(game_type))

    resolved_runtime_mode = runtime_mode or getattr(profile, "default_runtime_mode", RuntimeMode.OFFLINE)
    if not isinstance(resolved_runtime_mode, RuntimeMode):
        resolved_runtime_mode = RuntimeMode(str(resolved_runtime_mode))

    gate = getattr(profile, "capability_gate", None)
    profile_capabilities = tuple(getattr(profile, "capabilities", ()) or ())
    allowed_capabilities = {Capability.coerce(item) for item in profile_capabilities}

    online_competitive = game_type == GameType.TYPE_D
    online_mode = game_type == GameType.TYPE_B and resolved_runtime_mode == RuntimeMode.ONLINE
    if requested in HARD_DENIED_ONLINE_CAPABILITIES and (online_competitive or online_mode):
        reason = "capability_denied"
    elif gate is not None and gate.denies(requested):
        reason = "capability_denied"
    elif requested not in allowed_capabilities:
        reason = "capability_not_allowed"
    elif gate is not None and not gate.allows(requested):
        reason = "capability_not_allowed"
    else:
        reason = ""

    return {
        "allowed": not reason,
        "reason": reason,
        "profile_id": normalized_profile_id or getattr(profile, "profile_id", ""),
        "capability": requested.value,
        "game_type": game_type.value,
        "runtime_mode": resolved_runtime_mode.value,
    }


def capability_error_response(decision: dict[str, Any]) -> dict[str, Any]:
    reason = str(decision.get("reason") or "capability_denied")
    capability = str(decision.get("capability") or "")
    profile_id = str(decision.get("profile_id") or "")
    return {
        "success": False,
        "error": {
            "code": reason,
            "message": f"profile {profile_id!r} cannot use capability {capability!r}",
            "profile_id": profile_id,
            "capability": capability,
            "game_type": decision.get("game_type"),
            "runtime_mode": decision.get("runtime_mode"),
        },
    }
