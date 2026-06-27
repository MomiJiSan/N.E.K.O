from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps

from .screen_regions import BBox, SHOP_SLOT_KEYS, shop_slot_bboxes

try:
    import imagehash as _imagehash
except ImportError:  # pragma: no cover - covered by monkeypatch in unit tests.
    _imagehash = None

DATA_DIR = Path(__file__).resolve().parent / "data"
ITEMS_DATA_PATH = DATA_DIR / "items.json"
UNITS_DATA_PATH = DATA_DIR / "units.json"
TRAITS_DATA_PATH = DATA_DIR / "traits.json"
ITEM_ASSETS_DIR = DATA_DIR / "assets" / "items"
UNIT_ASSETS_DIR = DATA_DIR / "assets" / "units"
IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
UNKNOWN_ID = "unknown"


@dataclass(frozen=True)
class RecognitionResult:
    slot_id: str
    kind: str
    detected_id: str
    confidence: float
    bbox: BBox
    status: str = "ok"
    name: str | None = None
    distance: int | None = None
    reason: str | None = None
    template: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "kind": self.kind,
            "id": self.detected_id,
            "name": self.name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "status": self.status,
            "distance": self.distance,
            "reason": self.reason,
            "template": self.template,
        }


@dataclass(frozen=True)
class TemplateRecord:
    template_id: str
    name: str
    path: Path
    perceptual_hash: Any


def recognize_tft_shop_slots(
    image: str | Path | Image.Image,
    *,
    slot_boxes: Mapping[str, BBox] | None = None,
    confidence_threshold: float = 0.86,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Return conservative recognition results for the five TFT shop slots."""

    with _open_rgb_image(image) as screenshot:
        boxes = dict(slot_boxes or shop_slot_bboxes(*screenshot.size))
        slots = [(slot_id, boxes.get(slot_id)) for slot_id in SHOP_SLOT_KEYS]
        templates, diagnostics = _load_template_records(kind="units", data_dir=Path(data_dir))
        results = []
        for slot_id, bbox in slots:
            if bbox is None:
                result = _unknown_result("unit", "missing_slot_bbox")
            else:
                crop = screenshot.crop(bbox)
                result = _recognize_crop(
                    crop,
                    kind="unit",
                    templates=templates,
                    diagnostics=diagnostics,
                    threshold=confidence_threshold,
                )
            results.append({"slot": slot_id, "bbox": bbox, "result": result})

    return {
        "available": diagnostics["imagehash"]["status"] == "ready" and bool(templates),
        "status": "ready" if templates else diagnostics["templates"]["warning"],
        "kind": "shop_units",
        "slots": results,
        "diagnostics": diagnostics,
    }


def recognize_tft_item_icon(
    image: str | Path | Image.Image,
    *,
    confidence_threshold: float = 0.86,
    data_dir: str | Path = DATA_DIR,
) -> dict[str, Any]:
    """Recognize a single TFT item icon crop from template assets."""

    templates, diagnostics = _load_template_records(kind="items", data_dir=Path(data_dir))
    with _open_rgb_image(image) as item_image:
        result = _recognize_crop(
            item_image,
            kind="item",
            templates=templates,
            diagnostics=diagnostics,
            threshold=confidence_threshold,
        )
    return {
        "available": diagnostics["imagehash"]["status"] == "ready" and bool(templates),
        "status": "ready" if templates else diagnostics["templates"]["warning"],
        "kind": "item",
        "result": result,
        "diagnostics": diagnostics,
    }


def load_tft_recognition_catalogs(data_dir: str | Path = DATA_DIR) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(data_dir)
    return {
        "items": _load_catalog(root / "items.json", "items"),
        "units": _load_catalog(root / "units.json", "units"),
        "traits": _load_catalog(root / "traits.json", "traits"),
    }


def recognize_shop_units(
    image_path: str | Path,
    regions: Mapping[str, Any],
    *,
    data_dir: str | Path = DATA_DIR,
    threshold: float = 0.86,
) -> dict[str, Any]:
    shop_slots = regions.get("shop_slots", regions)
    if not isinstance(shop_slots, Mapping):
        return _unknown_response("units", [], "missing_shop_slots")
    ordered_slots = [(key, shop_slots[key]) for key in SHOP_SLOT_KEYS if key in shop_slots]
    return recognize_slots(
        image_path=image_path,
        slots=ordered_slots,
        kind="units",
        data_dir=data_dir,
        threshold=threshold,
    )


def recognize_slots(
    *,
    image_path: str | Path,
    slots: Iterable[tuple[str, BBox]],
    kind: str,
    data_dir: str | Path = DATA_DIR,
    threshold: float = 0.86,
) -> dict[str, Any]:
    normalized_slots = [(slot_id, tuple(bbox)) for slot_id, bbox in slots]
    if not normalized_slots:
        return _unknown_response(kind, [], "no_slots")

    imagehash = _imagehash
    if imagehash is None:
        return _unknown_response(kind, normalized_slots, "imagehash_unavailable")

    templates = _load_template_hashes(kind=kind, data_dir=Path(data_dir), imagehash=imagehash)
    if not templates:
        return _unknown_response(kind, normalized_slots, "no_templates")

    results: list[RecognitionResult] = []
    try:
        with Image.open(Path(image_path).expanduser()) as image:
            image.load()
            for slot_id, bbox in normalized_slots:
                crop_hash = imagehash.phash(_normalize_image(image.crop(bbox)))
                detected_id, confidence = _best_template_match(crop_hash, templates)
                if confidence < threshold:
                    detected_id = UNKNOWN_ID
                    status = "low_confidence"
                else:
                    status = "ok"
                results.append(
                    RecognitionResult(
                        slot_id=slot_id,
                        kind=kind,
                        detected_id=detected_id,
                        confidence=round(confidence, 4),
                        bbox=bbox,
                        status=status,
                    )
                )
    except Exception as exc:
        return _unknown_response(kind, normalized_slots, "image_read_failed", str(exc))

    return {
        "available": True,
        "status": "ready",
        "kind": kind,
        "results": [result.to_dict() for result in results],
        "diagnostics": {"template_count": len(templates), "warning": None},
    }


def _load_template_hashes(*, kind: str, data_dir: Path, imagehash: Any) -> dict[str, Any]:
    asset_dir = data_dir / "assets" / kind
    if not asset_dir.is_dir():
        return {}
    templates: dict[str, Any] = {}
    for path in _iter_template_paths(asset_dir):
        with Image.open(path) as image:
            image.load()
            templates[_template_id_from_path(path)] = imagehash.phash(_normalize_image(image))
    return templates


def _best_template_match(crop_hash: Any, templates: Mapping[str, Any]) -> tuple[str, float]:
    best_id = UNKNOWN_ID
    best_confidence = 0.0
    for template_id, template_hash in templates.items():
        distance = crop_hash - template_hash
        hash_bits = _hash_size(template_hash)
        confidence = max(0.0, 1.0 - (float(distance) / float(hash_bits)))
        if confidence > best_confidence:
            best_id = template_id
            best_confidence = confidence
    return best_id, best_confidence


def _unknown_response(
    kind: str,
    slots: Iterable[tuple[str, BBox]],
    warning: str,
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "status": warning,
        "kind": kind,
        "results": [
            RecognitionResult(
                slot_id=slot_id,
                kind=kind,
                detected_id=UNKNOWN_ID,
                confidence=0.0,
                bbox=tuple(bbox),
                status=warning,
            ).to_dict()
            for slot_id, bbox in slots
        ],
        "diagnostics": {"template_count": 0, "warning": warning, "detail": detail},
    }


def _load_template_records(*, kind: str, data_dir: Path) -> tuple[tuple[TemplateRecord, ...], dict[str, Any]]:
    imagehash = _imagehash
    catalog_key = "units" if kind == "units" else "items"
    catalog_path = data_dir / f"{catalog_key}.json"
    asset_dir = data_dir / "assets" / catalog_key
    catalog = _load_catalog(catalog_path, catalog_key)
    diagnostics: dict[str, Any] = {
        "imagehash": _imagehash_diagnostics(),
        "catalog": {"path": str(catalog_path), "records": len(catalog)},
        "templates": {
            "kind": catalog_key,
            "path": str(asset_dir),
            "loaded": 0,
            "failed": [],
            "warning": None,
        },
    }
    if imagehash is None:
        diagnostics["templates"]["warning"] = "imagehash_unavailable"
        return (), diagnostics
    if not asset_dir.is_dir():
        diagnostics["templates"]["warning"] = "no_templates"
        return (), diagnostics

    templates: list[TemplateRecord] = []
    for path in _iter_template_paths(asset_dir):
        template_id = _template_id_from_path(path)
        metadata = catalog.get(template_id, {})
        try:
            with Image.open(path) as image:
                image.load()
                perceptual_hash = imagehash.phash(_normalize_image(image))
        except Exception as exc:  # pragma: no cover - defensive for user-added assets.
            diagnostics["templates"]["failed"].append({"path": str(path), "error": str(exc)})
            continue
        templates.append(
            TemplateRecord(
                template_id=template_id,
                name=str(metadata.get("name") or template_id),
                path=path,
                perceptual_hash=perceptual_hash,
            )
        )

    diagnostics["templates"]["loaded"] = len(templates)
    if not templates:
        diagnostics["templates"]["warning"] = "no_templates"
    return tuple(templates), diagnostics


def _recognize_crop(
    image: Image.Image,
    *,
    kind: str,
    templates: tuple[TemplateRecord, ...],
    diagnostics: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    if diagnostics["imagehash"]["status"] != "ready":
        return _unknown_result(kind, "imagehash_unavailable")
    if not templates:
        return _unknown_result(kind, "no_templates")

    candidate_hash = _imagehash.phash(_normalize_image(image))
    best_template: TemplateRecord | None = None
    best_distance: int | None = None
    for template in templates:
        distance = int(candidate_hash - template.perceptual_hash)
        if best_distance is None or distance < best_distance:
            best_template = template
            best_distance = distance

    confidence = _confidence_from_distance(best_distance, _hash_size(candidate_hash))
    if best_template is None or confidence < threshold:
        result = _unknown_result(kind, "low_confidence")
        result["confidence"] = confidence
        result["best_match"] = _best_match_payload(best_template, best_distance)
        return result

    return {
        "status": "matched",
        "kind": kind,
        "id": best_template.template_id,
        "name": best_template.name,
        "confidence": confidence,
        "distance": best_distance,
        "template": str(best_template.path),
    }


def _load_catalog(path: Path, records_key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get(records_key, [])
    else:
        records = []

    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            catalog[record_id] = record
    return catalog


def _iter_template_paths(asset_dir: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(asset_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _open_rgb_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(Path(image).expanduser()).convert("RGB")


def _normalize_image(image: Image.Image) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return ImageOps.fit(image.convert("RGB"), (96, 96), method=resampling)


def _template_id_from_path(path: Path) -> str:
    return path.stem.split("__", 1)[0]


def _imagehash_diagnostics() -> dict[str, Any]:
    if _imagehash is None:
        return {"status": "unavailable", "reason": "Python package 'imagehash' is not installed."}
    return {"status": "ready", "algorithm": "phash"}


def _unknown_result(kind: str, reason: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "kind": kind,
        "id": None,
        "name": None,
        "confidence": 0.0,
        "distance": None,
        "reason": reason,
    }


def _confidence_from_distance(distance: int | None, max_distance: int) -> float:
    if distance is None or max_distance <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (distance / max_distance)), 4)


def _hash_size(perceptual_hash: Any) -> int:
    hash_bits = getattr(perceptual_hash, "hash", None)
    if hash_bits is None:
        return 64
    return int(hash_bits.size)


def _best_match_payload(template: TemplateRecord | None, distance: int | None) -> dict[str, Any] | None:
    if template is None:
        return None
    return {
        "id": template.template_id,
        "name": template.name,
        "distance": distance,
        "template": str(template.path),
    }


__all__ = [
    "DATA_DIR",
    "ITEMS_DATA_PATH",
    "TRAITS_DATA_PATH",
    "UNITS_DATA_PATH",
    "ITEM_ASSETS_DIR",
    "UNIT_ASSETS_DIR",
    "RecognitionResult",
    "load_tft_recognition_catalogs",
    "recognize_shop_units",
    "recognize_slots",
    "recognize_tft_item_icon",
    "recognize_tft_shop_slots",
]
