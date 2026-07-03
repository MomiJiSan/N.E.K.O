from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from PIL import Image

BBox = tuple[int, int, int, int]
RatioBBox = tuple[float, float, float, float]
Recognizer = str

LAYOUT_NORMAL_SHOP = "normal_shop"
LAYOUT_COMBAT = "combat"
LAYOUT_AUGMENT_SELECT = "augment_select"
LAYOUT_SPECIAL = "special"
LAYOUT_STATES = (
    LAYOUT_NORMAL_SHOP,
    LAYOUT_COMBAT,
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_SPECIAL,
)


class UnsupportedAspectRatioError(ValueError):
    """Raised when TFT screen regions are requested for a non-16:9 image."""


@dataclass(frozen=True)
class TFTLayoutProfile:
    key: str
    display_name: str
    description: str
    primary_regions: tuple[str, ...]
    deep_recognition: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "description": self.description,
            "primary_regions": list(self.primary_regions),
            "deep_recognition": self.deep_recognition,
        }


@dataclass(frozen=True)
class ScreenRegion:
    key: str
    display_name: str
    ratio_bbox: RatioBBox
    layout: str = LAYOUT_NORMAL_SHOP
    priority: int = 100
    purpose: str = "observe"
    recognizers: tuple[Recognizer, ...] = ()
    active_layouts: tuple[str, ...] = (LAYOUT_NORMAL_SHOP,)

    def bbox_for(self, width: int, height: int) -> BBox:
        left, top, right, bottom = self.ratio_bbox
        return (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )

    def metadata_for(self, width: int, height: int) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "layout": self.layout,
            "priority": self.priority,
            "purpose": self.purpose,
            "recognizers": list(self.recognizers),
            "active_layouts": list(self.active_layouts),
            "bbox": list(self.bbox_for(width, height)),
            "ratio_bbox": list(self.ratio_bbox),
        }


def _ratio_box(left: int, top: int, right: int, bottom: int) -> RatioBBox:
    return (left / 1920, top / 1080, right / 1920, bottom / 1080)


SHOP_SLOT_KEYS = (
    "shop_slot_1",
    "shop_slot_2",
    "shop_slot_3",
    "shop_slot_4",
    "shop_slot_5",
)

AUGMENT_OPTION_KEYS = (
    "augment_option_1",
    "augment_option_2",
    "augment_option_3",
)

LEGACY_GROUPED_REGION_KEYS = (
    "shop",
    "bench",
    "board",
    "equipment",
    "gold",
    "level",
    "stage",
    "round",
    "augments",
    "traits_panel",
)

EXTENDED_GROUPED_REGION_KEYS = (
    "items_area",
    "level_exp",
    "players_panel",
    "shop_odds",
    "refresh_button",
    "buy_xp_button",
    "notifications",
)


TFT_LAYOUT_PROFILES: dict[str, TFTLayoutProfile] = {
    LAYOUT_NORMAL_SHOP: TFTLayoutProfile(
        key=LAYOUT_NORMAL_SHOP,
        display_name="Normal shop",
        description=(
            "Preparation-phase PC 16:9 layout with the shop open. This is the "
            "MVP layout for stage, economy, shop, bench, traits, items, and board observation."
        ),
        primary_regions=(
            "stage",
            "round",
            "gold",
            "level",
            "level_exp",
            "shop",
            "shop_slots",
            "shop_odds",
            "refresh_button",
            "buy_xp_button",
            "bench",
            "traits_panel",
            "items_area",
            "equipment",
            "board",
            "players_panel",
        ),
    ),
    LAYOUT_COMBAT: TFTLayoutProfile(
        key=LAYOUT_COMBAT,
        display_name="Combat",
        description=(
            "Combat layout. The shop may remain visible, but board units, item holders, "
            "and player state become more important than shop OCR."
        ),
        primary_regions=("stage", "round", "board", "items_area", "equipment", "traits_panel", "players_panel"),
    ),
    LAYOUT_AUGMENT_SELECT: TFTLayoutProfile(
        key=LAYOUT_AUGMENT_SELECT,
        display_name="Augment select",
        description="Augment selection overlay. It should be detected and handled separately from normal shop analysis.",
        primary_regions=("stage", "round", "augments", *AUGMENT_OPTION_KEYS, "notifications"),
    ),
    LAYOUT_SPECIAL: TFTLayoutProfile(
        key=LAYOUT_SPECIAL,
        display_name="Special",
        description="Carousel, PvE, encounter, or set-specific selection screens. First pass only detects these states.",
        primary_regions=("stage", "round", "board", "notifications"),
        deep_recognition=False,
    ),
}


REGION_DEFINITIONS: tuple[ScreenRegion, ...] = (
    ScreenRegion(
        "stage",
        "Stage",
        _ratio_box(720, 18, 860, 68),
        priority=1,
        purpose="round_state",
        recognizers=("ocr",),
        active_layouts=LAYOUT_STATES,
    ),
    ScreenRegion(
        "round",
        "Round",
        _ratio_box(720, 18, 860, 68),
        priority=1,
        purpose="round_state",
        recognizers=("ocr",),
        active_layouts=LAYOUT_STATES,
    ),
    ScreenRegion("gold", "Gold", _ratio_box(920, 842, 1008, 892), priority=2, purpose="economy", recognizers=("ocr",)),
    ScreenRegion("level", "Level", _ratio_box(300, 845, 510, 920), priority=3, purpose="economy", recognizers=("ocr",)),
    ScreenRegion("level_exp", "Level and XP", _ratio_box(300, 845, 510, 920), priority=3, purpose="economy", recognizers=("ocr",)),
    ScreenRegion("shop", "Shop row", _ratio_box(470, 878, 1422, 1046), priority=4, purpose="shop", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_slot_1", "Shop slot 1", _ratio_box(474, 878, 650, 1046), priority=4, purpose="unit_recognition", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_slot_2", "Shop slot 2", _ratio_box(666, 878, 842, 1046), priority=4, purpose="unit_recognition", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_slot_3", "Shop slot 3", _ratio_box(858, 878, 1034, 1046), priority=4, purpose="unit_recognition", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_slot_4", "Shop slot 4", _ratio_box(1050, 878, 1226, 1046), priority=4, purpose="unit_recognition", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_slot_5", "Shop slot 5", _ratio_box(1242, 878, 1418, 1046), priority=4, purpose="unit_recognition", recognizers=("template_hash", "ocr")),
    ScreenRegion("shop_odds", "Shop odds", _ratio_box(465, 845, 860, 895), priority=4, purpose="shop_odds", recognizers=("ocr",)),
    ScreenRegion("refresh_button", "Refresh button", _ratio_box(300, 930, 510, 1000), priority=9, purpose="ui_context", recognizers=("ocr",)),
    ScreenRegion("buy_xp_button", "Buy XP button", _ratio_box(300, 850, 510, 920), priority=9, purpose="ui_context", recognizers=("ocr",)),
    ScreenRegion("bench", "Bench", _ratio_box(340, 610, 1510, 850), priority=5, purpose="bench_units", recognizers=("template_hash",)),
    ScreenRegion(
        "traits_panel",
        "Traits panel",
        _ratio_box(0, 120, 315, 690),
        priority=6,
        purpose="traits",
        recognizers=("ocr", "template_hash"),
        active_layouts=(LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT),
    ),
    ScreenRegion(
        "items_area",
        "Items area",
        _ratio_box(0, 275, 90, 455),
        priority=7,
        purpose="items",
        recognizers=("template_hash",),
        active_layouts=(LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT),
    ),
    ScreenRegion(
        "equipment",
        "Equipment area",
        _ratio_box(0, 275, 90, 455),
        priority=7,
        purpose="items",
        recognizers=("template_hash",),
        active_layouts=(LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT),
    ),
    ScreenRegion(
        "board",
        "Board",
        _ratio_box(360, 185, 1560, 735),
        priority=8,
        purpose="board_units",
        recognizers=("template_hash",),
        active_layouts=(LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_SPECIAL),
    ),
    ScreenRegion(
        "players_panel",
        "Players panel",
        _ratio_box(1600, 160, 1918, 760),
        priority=9,
        purpose="players",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT),
    ),
    ScreenRegion(
        "notifications",
        "Notifications",
        _ratio_box(650, 96, 1270, 220),
        priority=10,
        purpose="state_detection",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_COMBAT, LAYOUT_AUGMENT_SELECT, LAYOUT_SPECIAL),
    ),
    ScreenRegion(
        "augments",
        "Augments",
        _ratio_box(520, 150, 1400, 900),
        layout=LAYOUT_AUGMENT_SELECT,
        priority=1,
        purpose="augment_text",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_AUGMENT_SELECT,),
    ),
    ScreenRegion(
        "augment_option_1",
        "Augment option 1",
        _ratio_box(520, 260, 800, 860),
        layout=LAYOUT_AUGMENT_SELECT,
        priority=1,
        purpose="augment_option_text",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_AUGMENT_SELECT,),
    ),
    ScreenRegion(
        "augment_option_2",
        "Augment option 2",
        _ratio_box(820, 260, 1100, 860),
        layout=LAYOUT_AUGMENT_SELECT,
        priority=1,
        purpose="augment_option_text",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_AUGMENT_SELECT,),
    ),
    ScreenRegion(
        "augment_option_3",
        "Augment option 3",
        _ratio_box(1120, 260, 1400, 860),
        layout=LAYOUT_AUGMENT_SELECT,
        priority=1,
        purpose="augment_option_text",
        recognizers=("ocr",),
        active_layouts=(LAYOUT_AUGMENT_SELECT,),
    ),
)


def ensure_16_9(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise UnsupportedAspectRatioError(
            f"TFT screen regions require a positive 16:9 image size; got {width}x{height}."
        )
    if width * 9 != height * 16:
        raise UnsupportedAspectRatioError(
            "TFT screen regions only support 16:9 screenshots; "
            f"got {width}x{height}."
        )


def screen_region_bboxes(width: int, height: int) -> dict[str, BBox]:
    ensure_16_9(width, height)
    return {region.key: region.bbox_for(width, height) for region in REGION_DEFINITIONS}


def screen_region_metadata(width: int, height: int) -> dict[str, dict[str, Any]]:
    ensure_16_9(width, height)
    return {region.key: region.metadata_for(width, height) for region in REGION_DEFINITIONS}


def screen_region_definitions(*, layout: str | None = None) -> tuple[ScreenRegion, ...]:
    if layout is None:
        return REGION_DEFINITIONS
    normalized = str(layout or "").strip().lower()
    if normalized in TFT_LAYOUT_PROFILES:
        definitions = {region.key: region for region in REGION_DEFINITIONS}
        return tuple(definitions[key] for key in _layout_region_keys(normalized) if key in definitions)
    return tuple(region for region in REGION_DEFINITIONS if region.layout == normalized)


def layout_profile(layout: str = LAYOUT_NORMAL_SHOP) -> TFTLayoutProfile:
    normalized = str(layout or "").strip().lower()
    try:
        return TFT_LAYOUT_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(TFT_LAYOUT_PROFILES))
        raise KeyError(f"unknown TFT layout {layout!r}; available: {available}") from exc


def layout_region_bboxes(width: int, height: int, layout: str = LAYOUT_NORMAL_SHOP) -> dict[str, BBox]:
    ensure_16_9(width, height)
    regions = screen_region_bboxes(width, height)
    return {key: regions[key] for key in _layout_region_keys(layout) if key in regions}


def screen_region_bbox(width: int, height: int, key: str) -> BBox:
    regions = screen_region_bboxes(width, height)
    try:
        return regions[key]
    except KeyError as exc:
        available = ", ".join(sorted(regions))
        raise KeyError(f"unknown TFT screen region {key!r}; available: {available}") from exc


def shop_slot_bboxes(width: int, height: int) -> dict[str, BBox]:
    regions = screen_region_bboxes(width, height)
    return {key: regions[key] for key in SHOP_SLOT_KEYS}


def grouped_screen_region_bboxes(
    width: int,
    height: int,
    *,
    include_extended: bool = True,
) -> dict[str, BBox | Mapping[str, BBox]]:
    regions = screen_region_bboxes(width, height)
    grouped: dict[str, BBox | Mapping[str, BBox]] = {
        "shop_slots": {key: regions[key] for key in SHOP_SLOT_KEYS},
    }
    grouped.update({key: regions[key] for key in LEGACY_GROUPED_REGION_KEYS})
    if include_extended:
        grouped.update({key: regions[key] for key in EXTENDED_GROUPED_REGION_KEYS})
    return grouped


def semantic_screen_region_bboxes(width: int, height: int) -> dict[str, BBox | Mapping[str, BBox]]:
    return grouped_screen_region_bboxes(width, height, include_extended=True)


def grouped_screen_region_metadata(width: int, height: int) -> dict[str, Any]:
    metadata = screen_region_metadata(width, height)
    return {
        "layout_profiles": {key: profile.to_dict() for key, profile in TFT_LAYOUT_PROFILES.items()},
        "regions": metadata,
        "groups": {
            "shop_slots": [metadata[key] for key in SHOP_SLOT_KEYS],
            "normal_shop": _metadata_for_layout(metadata, LAYOUT_NORMAL_SHOP),
            "combat": _metadata_for_layout(metadata, LAYOUT_COMBAT),
            "augment_select": _metadata_for_layout(metadata, LAYOUT_AUGMENT_SELECT),
            "special": _metadata_for_layout(metadata, LAYOUT_SPECIAL),
        },
    }


def save_debug_crops(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    layout: str | None = None,
    grouped: bool = True,
    include_extended: bool = True,
    semantic_filenames: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    with Image.open(Path(image_path).expanduser()) as image:
        image.load()
        width, height = image.size
        normalized_layout = _normalize_layout(layout)
        if normalized_layout:
            regions = layout_region_bboxes(width, height, normalized_layout)
        else:
            regions = (
                grouped_screen_region_bboxes(width, height, include_extended=include_extended)
                if grouped
                else screen_region_bboxes(width, height)
            )
        region_metadata = screen_region_metadata(width, height)
        for key, bbox in _iter_crop_boxes(regions):
            region_meta = region_metadata.get(key, {"layout": LAYOUT_NORMAL_SHOP, "priority": 99, "bbox": list(bbox)})
            crop_meta = {**region_meta, "capture_layout": normalized_layout}
            filename = _debug_crop_filename(key, crop_meta) if semantic_filenames else f"{key}.png"
            crop_path = output_path / filename
            image.crop(bbox).save(crop_path)
            saved[key] = str(crop_path.resolve())
            metadata[key] = {**crop_meta, "crop_path": str(crop_path.resolve())}
    payload = {
        "output_dir": str(output_path.resolve()),
        "capture_layout": normalized_layout,
        "crops": saved,
        "metadata": metadata,
        "layout_profiles": {key: profile.to_dict() for key, profile in TFT_LAYOUT_PROFILES.items()},
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["metadata_path"] = str(metadata_path.resolve())
    return payload


def _iter_crop_boxes(regions: Mapping[str, Any]):
    for key, value in regions.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                yield child_key, child_value
        else:
            yield key, value


def _debug_crop_filename(key: str, metadata: Mapping[str, Any]) -> str:
    layout = str(metadata.get("capture_layout") or metadata.get("layout") or LAYOUT_NORMAL_SHOP)
    priority = int(metadata.get("priority") or 99)
    return f"{layout}__p{priority:02d}__{key}.png"


def _layout_region_keys(layout: str) -> tuple[str, ...]:
    keys: list[str] = []
    normalized = layout_profile(layout).key
    for key in layout_profile(normalized).primary_regions:
        if key == "shop_slots":
            keys.extend(SHOP_SLOT_KEYS)
        else:
            keys.append(key)
    for region in REGION_DEFINITIONS:
        if normalized in region.active_layouts:
            keys.append(region.key)
    return tuple(dict.fromkeys(keys))


def _metadata_for_layout(metadata: Mapping[str, dict[str, Any]], layout: str) -> list[dict[str, Any]]:
    return [metadata[key] for key in _layout_region_keys(layout) if key in metadata]


def _normalize_layout(layout: str | None) -> str | None:
    if layout is None:
        return None
    normalized = str(layout or "").strip().lower()
    return normalized if normalized in TFT_LAYOUT_PROFILES else None


# Compatibility aliases for downstream workers while the analyzer shape settles.
get_screen_regions = screen_region_bboxes
get_tft_screen_regions = screen_region_bboxes
regions_for_resolution = screen_region_bboxes
