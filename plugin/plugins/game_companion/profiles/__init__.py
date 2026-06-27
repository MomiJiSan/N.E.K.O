from __future__ import annotations

from ..core.profile_registry import ProfileMetadata
from .galgame import profile as galgame_profile
from .generic import profile as generic_profile
from .tft import profile as tft_profile


def builtin_profiles() -> tuple[ProfileMetadata, ...]:
    return (
        generic_profile(),
        galgame_profile(),
        tft_profile(),
    )
