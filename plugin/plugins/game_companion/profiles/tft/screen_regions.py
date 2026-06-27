from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping

from PIL import Image

BBox = tuple[int, int, int, int]
RatioBBox = tuple[float, float, float, float]


class UnsupportedAspectRatioError(ValueError):
    """Raised when TFT screen regions are requested for a non-16:9 image."""


@dataclass(frozen=True)
class ScreenRegion:
    key: str
    display_name: str
    ratio_bbox: RatioBBox

    def bbox_for(self, width: int, height: int) -> BBox:
        left, top, right, bottom = self.ratio_bbox
        return (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )


def _ratio_box(left: int, top: int, right: int, bottom: int) -> RatioBBox:
    return (left / 1920, top / 1080, right / 1920, bottom / 1080)


SHOP_SLOT_KEYS = (
    "shop_slot_1",
    "shop_slot_2",
    "shop_slot_3",
    "shop_slot_4",
    "shop_slot_5",
)


REGION_DEFINITIONS: tuple[ScreenRegion, ...] = (
    ScreenRegion("shop", "Shop row", _ratio_box(470, 800, 1422, 1054)),
    ScreenRegion("shop_slot_1", "Shop slot 1", _ratio_box(474, 806, 650, 1048)),
    ScreenRegion("shop_slot_2", "Shop slot 2", _ratio_box(666, 806, 842, 1048)),
    ScreenRegion("shop_slot_3", "Shop slot 3", _ratio_box(858, 806, 1034, 1048)),
    ScreenRegion("shop_slot_4", "Shop slot 4", _ratio_box(1050, 806, 1226, 1048)),
    ScreenRegion("shop_slot_5", "Shop slot 5", _ratio_box(1242, 806, 1418, 1048)),
    ScreenRegion("bench", "Bench", _ratio_box(462, 665, 1458, 784)),
    ScreenRegion("board", "Board", _ratio_box(360, 185, 1560, 735)),
    ScreenRegion("equipment", "Equipment area", _ratio_box(240, 620, 455, 815)),
    ScreenRegion("gold", "Gold", _ratio_box(820, 760, 930, 805)),
    ScreenRegion("level", "Level", _ratio_box(898, 708, 1022, 765)),
    ScreenRegion("stage", "Stage", _ratio_box(870, 18, 1050, 64)),
    ScreenRegion("round", "Round", _ratio_box(870, 18, 1050, 64)),
    ScreenRegion("augments", "Augments", _ratio_box(1565, 105, 1900, 540)),
    ScreenRegion("traits_panel", "Traits panel", _ratio_box(0, 120, 315, 690)),
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


def grouped_screen_region_bboxes(width: int, height: int) -> dict[str, BBox | Mapping[str, BBox]]:
    regions = screen_region_bboxes(width, height)
    return {
        "shop_slots": {key: regions[key] for key in SHOP_SLOT_KEYS},
        "shop": regions["shop"],
        "bench": regions["bench"],
        "board": regions["board"],
        "equipment": regions["equipment"],
        "gold": regions["gold"],
        "level": regions["level"],
        "stage": regions["stage"],
        "round": regions["round"],
        "augments": regions["augments"],
        "traits_panel": regions["traits_panel"],
    }


def save_debug_crops(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    grouped: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    with Image.open(Path(image_path).expanduser()) as image:
        image.load()
        width, height = image.size
        regions = grouped_screen_region_bboxes(width, height) if grouped else screen_region_bboxes(width, height)
        for key, bbox in _iter_crop_boxes(regions):
            crop_path = output_path / f"{key}.png"
            image.crop(bbox).save(crop_path)
            saved[key] = str(crop_path.resolve())
    return {"output_dir": str(output_path.resolve()), "crops": saved}


def _iter_crop_boxes(regions: Mapping[str, Any]):
    for key, value in regions.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                yield child_key, child_value
        else:
            yield key, value


# Compatibility aliases for downstream workers while the analyzer shape settles.
get_screen_regions = screen_region_bboxes
get_tft_screen_regions = screen_region_bboxes
regions_for_resolution = screen_region_bboxes
