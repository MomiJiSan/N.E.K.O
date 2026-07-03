from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .screen_regions import BBox

OCR_REGION_KEYS = (
    "gold",
    "level",
    "level_exp",
    "stage",
    "round",
    "augments",
    "augment_option_1",
    "augment_option_2",
    "augment_option_3",
    "shop_slot_1",
    "shop_slot_2",
    "shop_slot_3",
    "shop_slot_4",
    "shop_slot_5",
)


def analyze_tft_ocr_regions(
    image_path: str | Path,
    regions: Mapping[str, BBox],
) -> dict[str, Any]:
    backend = _create_rapidocr_backend()
    if backend is None:
        return _unavailable("RapidOCR backend is not importable")

    try:
        if not backend.is_available():
            return _unavailable("RapidOCR backend is not available")
    except Exception as exc:
        return _unavailable(f"RapidOCR availability check failed: {exc}")

    try:
        with Image.open(Path(image_path).expanduser()) as image:
            image.load()
            region_results = {
                key: _ocr_region(backend, image, key, regions[key])
                for key in OCR_REGION_KEYS
                if key in regions
            }
    except Exception as exc:
        return {
            "available": True,
            "status": "failed",
            "error": str(exc),
            "regions": {},
            "parsed": {},
        }

    parsed = _parse_basic_state(region_results)
    return {
        "available": True,
        "status": "ready",
        "error": None,
        "regions": region_results,
        "parsed": parsed,
    }


def _create_rapidocr_backend() -> Any | None:
    try:
        from plugin.plugins._shared.rapidocr import (
            DEFAULT_RAPIDOCR_ENGINE_TYPE,
            DEFAULT_RAPIDOCR_LANG_TYPE,
            DEFAULT_RAPIDOCR_MODEL_TYPE,
            DEFAULT_RAPIDOCR_OCR_VERSION,
            RapidOcrBackend,
        )
    except Exception:
        return None

    return RapidOcrBackend(
        install_target_dir_raw="",
        engine_type=DEFAULT_RAPIDOCR_ENGINE_TYPE,
        lang_type=DEFAULT_RAPIDOCR_LANG_TYPE,
        model_type=DEFAULT_RAPIDOCR_MODEL_TYPE,
        ocr_version=DEFAULT_RAPIDOCR_OCR_VERSION,
        plugin_id="game_companion",
    )


def _ocr_region(backend: Any, image: Image.Image, key: str, bbox: BBox) -> dict[str, Any]:
    crop = image.crop(bbox)
    text, boxes = backend.extract_text_with_boxes(crop)
    return {
        "text": text,
        "confidence": _mean_confidence(boxes),
        "boxes": [
            {
                "text": box.text,
                "left": box.left,
                "top": box.top,
                "right": box.right,
                "bottom": box.bottom,
                "score": box.score,
            }
            for box in boxes
        ],
        "bbox": list(bbox),
        "region": key,
    }


def _mean_confidence(boxes: list[Any]) -> float | None:
    if not boxes:
        return None
    return sum(float(box.score) for box in boxes) / len(boxes)


def _parse_basic_state(region_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "gold": _parse_first_int(region_results.get("gold", {}).get("text")),
        "level": _parse_first_int(region_results.get("level", {}).get("text")),
        "level_exp": _parse_level_exp(region_results.get("level_exp", {}).get("text")),
        "stage": _parse_stage(region_results.get("stage", {}).get("text")),
        "round": _parse_stage(region_results.get("round", {}).get("text")),
        "augments": _parse_lines(region_results.get("augments", {}).get("text")),
    }


def _parse_first_int(text: Any) -> int | None:
    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else None


def _parse_stage(text: Any) -> str | None:
    match = re.search(r"(?<!\d)\d+\s*-\s*\d+(?!\d)", str(text or ""))
    return re.sub(r"\s+", "", match.group(0)) if match else None


def _parse_level_exp(text: Any) -> dict[str, int | None] | None:
    raw = str(text or "")
    level = _parse_first_int(raw)
    xp_match = re.search(r"(\d+)\s*/\s*(\d+)", raw)
    if level is None and not xp_match:
        return None
    return {
        "level": level,
        "xp_current": int(xp_match.group(1)) if xp_match else None,
        "xp_required": int(xp_match.group(2)) if xp_match else None,
    }


def _parse_lines(text: Any) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _unavailable(message: str) -> dict[str, Any]:
    return {
        "available": False,
        "status": "unavailable",
        "error": message,
        "regions": {},
        "parsed": {},
    }
