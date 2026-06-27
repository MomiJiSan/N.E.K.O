from __future__ import annotations

from ...core.profile_registry import ProfileMetadata
from ...safety.models import Capability, GameType, RuntimeMode


def profile() -> ProfileMetadata:
    return ProfileMetadata(
        profile_id="generic",
        display_name="Generic Observer",
        game_type=GameType.TYPE_D,
        default_runtime_mode=RuntimeMode.ONLINE,
        description="Read-only fallback profile for unknown games.",
        capabilities=(
            Capability.SCREEN_OBSERVE,
            Capability.OCR,
            Capability.VISION_CLASSIFY,
            Capability.NEKO_CONTEXT,
        ),
    )
