from __future__ import annotations

from ...core.profile_registry import ProfileMetadata
from ...safety.models import Capability, GameType, RuntimeMode
from .ocr import OCR_REGION_KEYS, analyze_tft_ocr_regions
from .insights import (
    FORBIDDEN_DIRECTIVE_PHRASES,
    FORBIDDEN_DIRECTIVE_TERMS,
    generate_insights,
    generate_tft_insights,
)
from .recognition import RecognitionResult, recognize_shop_units
from .screen_regions import (
    BBox,
    REGION_DEFINITIONS,
    SHOP_SLOT_KEYS,
    ScreenRegion,
    UnsupportedAspectRatioError,
    ensure_16_9,
    get_screen_regions,
    get_tft_screen_regions,
    grouped_screen_region_bboxes,
    regions_for_resolution,
    screen_region_bbox,
    screen_region_bboxes,
    save_debug_crops,
    shop_slot_bboxes,
)


def profile() -> ProfileMetadata:
    return ProfileMetadata(
        profile_id="tft",
        display_name="Teamfight Tactics",
        game_type=GameType.TYPE_D,
        default_runtime_mode=RuntimeMode.ONLINE,
        description="Read-only TFT board, shop, item, and trait insight profile placeholder.",
        capabilities=(
            Capability.SCREEN_OBSERVE,
            Capability.OCR,
            Capability.VISION_CLASSIFY,
            Capability.NEKO_CONTEXT,
        ),
    )


__all__ = [
    "BBox",
    "FORBIDDEN_DIRECTIVE_PHRASES",
    "FORBIDDEN_DIRECTIVE_TERMS",
    "OCR_REGION_KEYS",
    "REGION_DEFINITIONS",
    "SHOP_SLOT_KEYS",
    "ScreenRegion",
    "UnsupportedAspectRatioError",
    "analyze_tft_ocr_regions",
    "generate_insights",
    "generate_tft_insights",
    "ensure_16_9",
    "get_screen_regions",
    "get_tft_screen_regions",
    "grouped_screen_region_bboxes",
    "profile",
    "RecognitionResult",
    "recognize_shop_units",
    "regions_for_resolution",
    "screen_region_bbox",
    "screen_region_bboxes",
    "save_debug_crops",
    "shop_slot_bboxes",
]
