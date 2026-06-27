from __future__ import annotations

from ...core.profile_registry import ProfileMetadata
from ...safety.models import Capability, GameType, RuntimeMode


def profile() -> ProfileMetadata:
    return ProfileMetadata(
        profile_id="galgame",
        display_name="Galgame",
        game_type=GameType.TYPE_A,
        default_runtime_mode=RuntimeMode.OFFLINE,
        description="Future host profile for the existing galgame companion flow.",
        capabilities=(
            Capability.SCREEN_OBSERVE,
            Capability.OCR,
            Capability.VISION_CLASSIFY,
            Capability.NEKO_CONTEXT,
        ),
    )
