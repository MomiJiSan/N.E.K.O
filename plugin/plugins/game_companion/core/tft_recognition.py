from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
import time
from typing import Any, Protocol

from PIL import Image, ImageStat

from ..profiles.tft.ocr import analyze_tft_ocr_regions
from ..profiles.tft.screen_regions import (
    AUGMENT_OPTION_KEYS,
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_COMBAT,
    LAYOUT_NORMAL_SHOP,
    LAYOUT_SPECIAL,
    SHOP_SLOT_KEYS,
    UnsupportedAspectRatioError,
    layout_region_bboxes,
)

READINESS_LAYOUT_AUGMENT = "augment"
READINESS_LAYOUT_CAROUSEL = "carousel"
READINESS_LAYOUT_PORTAL = "portal"
READINESS_LAYOUT_POPUP = "popup"
READINESS_LAYOUT_LOADING = "loading"
READINESS_LAYOUT_UNKNOWN = "unknown"

READINESS_LAYOUT_ORDER = (
    LAYOUT_NORMAL_SHOP,
    READINESS_LAYOUT_AUGMENT,
    LAYOUT_COMBAT,
    READINESS_LAYOUT_CAROUSEL,
    READINESS_LAYOUT_PORTAL,
    READINESS_LAYOUT_POPUP,
    READINESS_LAYOUT_LOADING,
    READINESS_LAYOUT_UNKNOWN,
)

READINESS_LAYOUT_RULES: dict[str, dict[str, Any]] = {
    LAYOUT_NORMAL_SHOP: {
        "required_checks": ["stage", "gold", "level", "xp", "shop_slots", "shop_names", "shop_costs"],
        "not_applicable": ["augment_options"],
        "description": "Clean planning-phase shop frame with no hover tooltip or popup.",
    },
    READINESS_LAYOUT_AUGMENT: {
        "required_checks": ["augment_options", "augment_titles", "augment_descriptions"],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "gold", "level", "xp"],
        "description": "Augment selection frame; shop recognition is intentionally N/A.",
    },
    LAYOUT_COMBAT: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Combat frame; current readiness only verifies that layout routing does not require shop OCR.",
    },
    READINESS_LAYOUT_CAROUSEL: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Carousel/special selection frame; first pass is layout-only.",
    },
    READINESS_LAYOUT_PORTAL: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Portal/encounter selection frame; first pass is layout-only.",
    },
    READINESS_LAYOUT_POPUP: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Hover tooltip or modal-contaminated frame; excluded from readiness.",
        "excluded_from_readiness": True,
    },
    READINESS_LAYOUT_LOADING: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Loading/transition frame; first pass is layout-only.",
    },
    READINESS_LAYOUT_UNKNOWN: {
        "required_checks": [],
        "not_applicable": ["shop_slots", "shop_names", "shop_costs", "augment_options"],
        "description": "Unknown frame; blocked until the sample is assigned to a supported layout.",
    },
}


class TftOcrAdapter(Protocol):
    def recognize(self, image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        ...


class RapidOcrTftAdapter:
    def recognize(self, image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        return analyze_tft_ocr_regions(image_path, _flat_regions(regions))


def recognize_tft_frame(
    image_path: str | Path,
    *,
    expected_layout: str | None = None,
    ocr_adapter: TftOcrAdapter | None = None,
) -> dict[str, Any]:
    path = Path(image_path).expanduser()
    layout = _normalize_layout(expected_layout)
    warnings: list[dict[str, Any]] = []
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            regions = layout_region_bboxes(width, height, layout)
            shop = _recognize_shop_slots(image, regions) if layout == LAYOUT_NORMAL_SHOP else []
            ocr_regions = _ocr_regions_for_layout(regions, layout, shop)
    except UnsupportedAspectRatioError as exc:
        return _error_result(path, layout, "unsupported_aspect_ratio", str(exc))
    except OSError as exc:
        return _error_result(path, layout, "image_read_failed", str(exc))

    adapter = ocr_adapter or RapidOcrTftAdapter()
    try:
        ocr = adapter.recognize(path, ocr_regions)
    except Exception as exc:
        ocr = {
            "available": False,
            "status": "failed",
            "error": str(exc),
            "regions": {},
            "parsed": {},
        }
    if ocr.get("status") != "ready":
        warnings.append(
            {
                "code": f"ocr_{ocr.get('status') or 'failed'}",
                "message": str(ocr.get("error") or "OCR is not available"),
            }
        )

    stage_text = _region_text(ocr, "stage") or _region_text(ocr, "round")
    stage_confidence = _region_confidence(ocr, "stage") or _region_confidence(ocr, "round")
    stage = _field_from_text("stage", stage_text, stage_confidence, _parse_stage)
    gold = (
        _field_from_text("gold", _region_text(ocr, "gold"), _region_confidence(ocr, "gold"), _parse_last_int)
        if "gold" in regions
        else None
    )
    level_text = _region_text(ocr, "level_exp") or _region_text(ocr, "level")
    level_confidence = _region_confidence(ocr, "level_exp") or _region_confidence(ocr, "level")
    level = _field_from_text("level", level_text, level_confidence, _parse_int) if "level_exp" in regions else None
    xp = _xp_from_text(level_text, level_confidence) if "level_exp" in regions else None
    shop = _enrich_shop_slots_from_ocr(shop, ocr) if layout == LAYOUT_NORMAL_SHOP else []
    if layout == LAYOUT_NORMAL_SHOP:
        _infer_frame_shop_costs(shop)
    augments = _augment_options_from_ocr(ocr) if layout == LAYOUT_AUGMENT_SELECT else []
    warnings.extend(_missing_field_warnings(layout, stage=stage, gold=gold, level=level, xp=xp, augments=augments))
    field_status = _field_statuses(
        layout,
        stage=stage,
        gold=gold,
        level=level,
        xp=xp,
        shop=shop,
        augments=augments,
    )

    result = {
        "type": "tft_recognition_result",
        "schema_version": 1,
        "success": True,
        "image_path": str(path.resolve()) if path.exists() else str(path),
        "layout": layout,
        "stage": stage,
        "gold": gold,
        "level": level,
        "xp": xp,
        "shop": shop,
        "augments": augments,
        "traits": [],
        "items": [],
        "field_status": field_status,
        "ocr": _ocr_summary(ocr),
        "confidence": _overall_confidence(stage, gold, level, xp, shop, augments, warnings),
        "warnings": warnings,
    }
    result["readiness"] = _recognition_readiness(result)
    return result


def build_tft_recognition_report(
    calibration_report_or_path: Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    ocr_adapter: TftOcrAdapter | None = None,
) -> dict[str, Any]:
    calibration_report = _load_calibration_report(calibration_report_or_path)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "recognition_report_v1.json"
    summary_path = output_path / "recognition_summary_v1.json"
    layout_manifest_path = output_path / "layout_manifest_v1.json"
    shop_review_path = output_path / "recognition_shop_review_v1.json"
    shop_labels_path = output_path / "recognition_shop_labels_v1.json"
    augment_review_path = output_path / "recognition_augment_review_v1.json"
    review_crops_dir = output_path / "review_crops"
    existing_shop_labels = _load_human_labels(shop_labels_path)
    existing_augment_labels = _load_human_labels(augment_review_path)
    results = []
    for screenshot in calibration_report.get("screenshots", []):
        if not isinstance(screenshot, Mapping):
            continue
        image_path = screenshot.get("image_path")
        if not image_path:
            continue
        recognition = recognize_tft_frame(
            image_path,
            expected_layout=str(screenshot.get("expected_layout") or ""),
            ocr_adapter=ocr_adapter,
        )
        recognition["readiness"] = _recognition_readiness(recognition, sample=screenshot)
        results.append(
            {
                "index": screenshot.get("index"),
                "label": screenshot.get("label") or screenshot.get("id") or "",
                "image_path": str(image_path),
                "expected_layout": screenshot.get("expected_layout"),
                "recognition": recognition,
            }
        )
    _infer_report_shop_costs(results, existing_shop_labels)
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        if isinstance(recognition, dict):
            recognition["readiness"] = _recognition_readiness(recognition, sample=item)
    shop_review = _shop_review_report(results, shop_review_path, review_crops_dir, existing_shop_labels)
    shop_labels = _shop_labels_report(shop_review, shop_labels_path)
    augment_review = _augment_review_report(results, augment_review_path, review_crops_dir, existing_augment_labels)
    layout_manifest = _layout_manifest_report(results, layout_manifest_path)
    successes = sum(1 for item in results if item["recognition"].get("success"))
    summary = {
        "total": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "layouts": _layout_counts(results),
        "warnings": sum(len(item["recognition"].get("warnings") or []) for item in results),
        "readiness": _readiness_summary(results),
        "metrics": _recognition_metrics(results, shop_labels, augment_review),
    }
    report = {
        "type": "tft_recognition_report",
        "schema_version": 1,
        "report_version": "recognition_report_v1",
        "created_at": time.time(),
        "source_report_path": str(Path(calibration_report_or_path).resolve())
        if isinstance(calibration_report_or_path, (str, Path))
        else "",
        "report_path": str(report_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "shop_review_path": str(shop_review_path.resolve()),
        "shop_labels_path": str(shop_labels_path.resolve()),
        "augment_review_path": str(augment_review_path.resolve()),
        "layout_manifest_path": str(layout_manifest_path.resolve()),
        "summary": summary,
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    layout_manifest_path.write_text(json.dumps(layout_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shop_review_path.write_text(json.dumps(shop_review, ensure_ascii=False, indent=2), encoding="utf-8")
    shop_labels_path.write_text(json.dumps(shop_labels, ensure_ascii=False, indent=2), encoding="utf-8")
    augment_review_path.write_text(json.dumps(augment_review, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _normalize_layout(layout: str | None) -> str:
    value = str(layout or "").strip().lower()
    if value in {LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_AUGMENT_SELECT}:
        return value
    if value == READINESS_LAYOUT_AUGMENT:
        return LAYOUT_AUGMENT_SELECT
    if value in {
        LAYOUT_SPECIAL,
        READINESS_LAYOUT_CAROUSEL,
        READINESS_LAYOUT_PORTAL,
        READINESS_LAYOUT_POPUP,
        READINESS_LAYOUT_LOADING,
        READINESS_LAYOUT_UNKNOWN,
    }:
        return LAYOUT_SPECIAL
    return LAYOUT_NORMAL_SHOP


def _flat_regions(regions: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in regions.items() if isinstance(value, tuple) and len(value) == 4}


def _ocr_regions_for_layout(
    regions: Mapping[str, Any],
    layout: str,
    shop: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ocr_regions = {
        key: value
        for key, value in _flat_regions(regions).items()
        if key not in SHOP_SLOT_KEYS and key != "augments"
    }
    if layout == LAYOUT_NORMAL_SHOP:
        for slot in shop:
            slot_number = slot.get("slot")
            slot_key = f"shop_slot_{slot_number}"
            if slot.get("state") == "occupied" and slot_key in regions:
                ocr_regions[slot_key] = regions[slot_key]
                subregions = _shop_slot_subregions(regions[slot_key])
                ocr_regions[f"{slot_key}_name"] = subregions["slot_name"]
                ocr_regions[f"{slot_key}_cost"] = subregions["slot_cost"]
    if layout == LAYOUT_AUGMENT_SELECT and isinstance(regions.get("augments"), tuple):
        ocr_regions["augments"] = regions["augments"]
        for option_key in AUGMENT_OPTION_KEYS:
            if option_key in regions:
                ocr_regions[option_key] = regions[option_key]
    return ocr_regions


def _recognize_shop_slots(image: Image.Image, regions: Mapping[str, Any]) -> list[dict[str, Any]]:
    slots = []
    for slot_number, slot_key in enumerate(SHOP_SLOT_KEYS, start=1):
        bbox = regions.get(slot_key)
        if not isinstance(bbox, tuple):
            slots.append({"slot": slot_number, "state": "unknown", "confidence": 0.0, "bbox": None})
            continue
        crop_rgb = image.crop(bbox).convert("RGB")
        margin_x = max(1, int(crop_rgb.width * 0.12))
        margin_y = max(1, int(crop_rgb.height * 0.12))
        inner_rgb = crop_rgb.crop((margin_x, margin_y, crop_rgb.width - margin_x, crop_rgb.height - margin_y))
        crop = inner_rgb.convert("L")
        stat = ImageStat.Stat(crop)
        mean = float(stat.mean[0])
        stddev = float(stat.stddev[0])
        color_stat = ImageStat.Stat(inner_rgb)
        channel_std = [float(value) for value in color_stat.stddev]
        colorfulness = sum(channel_std) / len(channel_std)
        occupied_score = max(0.0, min(1.0, max(stddev - 10.0, colorfulness - 10.0) / 42.0))
        if mean < 22.0 and stddev < 18.0 and colorfulness < 18.0:
            state = "empty"
            confidence = 0.88
        elif occupied_score >= 0.35 or mean >= 34.0 or colorfulness >= 25.0:
            state = "occupied"
            confidence = max(0.55, occupied_score)
        else:
            state = "unknown"
            confidence = 0.35
        slots.append(
            {
                "slot": slot_number,
                "state": state,
                "name": None,
                "cost": None,
                "raw_text": "",
                "review_status": _initial_shop_review_status(state),
                "confidence": round(confidence, 4),
                "bbox": list(bbox),
                "diagnostics": {
                    "mean": round(mean, 2),
                    "stddev": round(stddev, 2),
                    "colorfulness": round(colorfulness, 2),
                },
            }
        )
    return slots


def _initial_shop_review_status(state: str) -> str:
    if state == "empty":
        return "empty"
    if state == "occupied":
        return "needs_ocr"
    return "unknown"


def _enrich_shop_slots_from_ocr(
    shop: list[Mapping[str, Any]],
    ocr: Mapping[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for slot in shop:
        item = dict(slot)
        if item.get("state") != "occupied":
            item.setdefault("raw_text", "")
            item.setdefault("name", None)
            item.setdefault("cost", None)
            item["review_status"] = "empty" if item.get("state") == "empty" else "unknown"
            enriched.append(item)
            continue
        slot_key = f"shop_slot_{item.get('slot')}"
        raw_text = _region_text(ocr, slot_key).strip()
        name_key = f"{slot_key}_name"
        cost_key = f"{slot_key}_cost"
        name_text = _region_text(ocr, name_key).strip()
        cost_text = _region_text(ocr, cost_key).strip()
        confidence = _region_confidence(ocr, slot_key)
        name_confidence = _region_confidence(ocr, name_key)
        cost_confidence = _region_confidence(ocr, cost_key)
        parsed = _parse_shop_card_text(raw_text, name_text=name_text, cost_text=cost_text)
        item["raw_text"] = raw_text
        item["name"] = parsed["name"]
        item["cost"] = parsed["cost"]
        item["name_candidate"] = parsed["name"]
        item["cost_candidate"] = parsed["cost"]
        item["ocr_lines"] = _ocr_lines(raw_text)
        item["name_raw_text"] = name_text
        item["cost_raw_text"] = cost_text
        item["name_ocr_lines"] = _ocr_lines(name_text)
        item["cost_ocr_lines"] = _ocr_lines(cost_text)
        item["ocr_confidence"] = _confidence(confidence)
        item["name_confidence"] = _confidence(name_confidence)
        item["cost_confidence"] = _confidence(cost_confidence)
        item["name_candidate_source"] = parsed["name_source"]
        item["cost_candidate_source"] = parsed["cost_source"]
        if raw_text:
            item["review_status"] = "needs_check"
            item["confidence"] = _confidence((float(item.get("confidence", 0.0)) + _confidence(confidence)) / 2.0)
        else:
            item["review_status"] = "ocr_missing"
        enriched.append(item)
    return enriched


def _ocr_lines(text: Any) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _parse_shop_card_text(
    text: Any,
    *,
    name_text: Any = "",
    cost_text: Any = "",
) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw_name = str(name_text or "").strip()
    raw_cost = str(cost_text or "").strip()
    cost = _parse_shop_cost(raw_cost)
    cost_source = "slot_cost" if cost is not None else None
    if cost is None:
        cost = _parse_shop_cost(raw)
        cost_source = "slot_full" if cost is not None else None
    name = _parse_shop_name(raw_name, cost)
    name_source = "slot_name" if name else None
    if name is None:
        name = _parse_shop_name(raw, cost)
        name_source = "slot_full" if name else None
    return {"name": name, "cost": cost, "name_source": name_source, "cost_source": cost_source}


def _parse_shop_cost(text: str) -> int | None:
    candidates = [int(value) for value in re.findall(r"\d+", text) if 1 <= int(value) <= 5]
    return candidates[-1] if candidates else None


def _parse_shop_name(text: str, cost: int | None) -> str | None:
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        candidate = re.sub(r"\d+", "", line).strip(" -:：|")
        if cost is not None and str(cost) == line.strip():
            continue
        if candidate:
            return candidate
    return None


def _infer_frame_shop_costs(shop: list[dict[str, Any]]) -> None:
    costs_by_name: dict[str, set[int]] = {}
    for slot in shop:
        if not isinstance(slot, dict) or slot.get("state") != "occupied":
            continue
        name = _shop_cost_consensus_key(slot)
        cost = slot.get("cost_candidate", slot.get("cost"))
        if name and isinstance(cost, int) and 1 <= cost <= 5:
            costs_by_name.setdefault(name, set()).add(cost)

    consensus = {name: next(iter(costs)) for name, costs in costs_by_name.items() if len(costs) == 1}
    for slot in shop:
        if not isinstance(slot, dict) or slot.get("state") != "occupied":
            continue
        if slot.get("cost_candidate", slot.get("cost")) is not None:
            continue
        name = _shop_cost_consensus_key(slot)
        if not name or name not in consensus:
            continue
        slot["cost_candidate"] = consensus[name]
        slot["cost"] = consensus[name]
        slot["cost_candidate_source"] = "frame_name_cost_consensus"
        slot["cost_inference"] = {
            "method": "frame_name_cost_consensus",
            "matched_name": name,
            "confidence": 0.52,
        }


def _field_from_text(
    name: str,
    text: Any,
    confidence: float | None,
    parser: Any,
) -> dict[str, Any] | None:
    value = parser(text)
    if value is None:
        return None
    return {
        "field": name,
        "value": value,
        "confidence": _confidence(confidence),
        "raw_text": str(text or ""),
    }


def _xp_from_text(text: Any, confidence: float | None) -> dict[str, Any] | None:
    match = re.search(r"(\d+)\s*/\s*(\d+)", str(text or ""))
    if not match:
        return None
    return {
        "current": int(match.group(1)),
        "required": int(match.group(2)),
        "confidence": _confidence(confidence),
        "raw_text": str(text or ""),
    }


def _augment_options_from_ocr(ocr: Mapping[str, Any]) -> list[dict[str, Any]]:
    regions = _ocr_regions_map(ocr)
    option_regions = [
        _augment_option_from_text(
            slot=index,
            text=_region_text(ocr, f"augment_option_{index}"),
            confidence=_region_confidence(ocr, f"augment_option_{index}"),
            bbox=regions.get(f"augment_option_{index}", {}).get("bbox")
            if isinstance(regions.get(f"augment_option_{index}"), Mapping)
            else None,
        )
        for index in range(1, 4)
        if f"augment_option_{index}" in regions
    ]
    if option_regions:
        return option_regions
    text = _region_text(ocr, "augments")
    if not text:
        return []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines and "选择" in lines[0]:
        lines = lines[1:]
    confidence = _confidence(_region_confidence(ocr, "augments"))
    return [
        _augment_option_from_text(slot=index, text=line, confidence=confidence, bbox=None)
        for index, line in enumerate(lines[:3], start=1)
    ]


def _augment_option_from_text(slot: int, text: Any, confidence: float | None, bbox: Any) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    title = lines[0] if lines else ""
    description = " ".join(lines[1:]) if len(lines) > 1 else ""
    return {
        "slot": slot,
        "title": title,
        "title_candidate": title or None,
        "description": description,
        "description_candidate": description or None,
        "raw_text": raw_text,
        "confidence": _confidence(confidence),
        "review_status": "needs_check" if title else "missing",
        "bbox": list(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
    }


def _missing_field_warnings(
    layout: str,
    *,
    stage: Mapping[str, Any] | None,
    gold: Mapping[str, Any] | None,
    level: Mapping[str, Any] | None,
    xp: Mapping[str, Any] | None,
    augments: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = ["stage"]
    if layout == LAYOUT_NORMAL_SHOP:
        fields.extend(["gold", "level", "xp"])
    if layout == LAYOUT_AUGMENT_SELECT:
        fields.append("augments")
    values = {
        "stage": stage,
        "gold": gold,
        "level": level,
        "xp": xp,
        "augments": augments,
    }
    return [
        {
            "code": "field_missing",
            "field": field,
            "message": f"TFT recognition did not produce a usable {field} value.",
        }
        for field in fields
        if not values.get(field)
    ]


def _field_statuses(
    layout: str,
    *,
    stage: Mapping[str, Any] | None,
    gold: Mapping[str, Any] | None,
    level: Mapping[str, Any] | None,
    xp: Mapping[str, Any] | None,
    shop: list[Mapping[str, Any]],
    augments: list[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "stage": _field_status(layout, True, stage),
        "gold": _field_status(layout, layout == LAYOUT_NORMAL_SHOP, gold),
        "level": _field_status(layout, layout == LAYOUT_NORMAL_SHOP, level),
        "xp": _field_status(layout, layout == LAYOUT_NORMAL_SHOP, xp),
        "shop": _shop_status(layout, shop),
        "augments": _list_field_status(layout, layout == LAYOUT_AUGMENT_SELECT, augments),
        "traits": _field_status(layout, False, None),
        "items": _field_status(layout, False, None),
    }


def _field_status(layout: str, applicable: bool, value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not applicable:
        return {"status": "not_applicable", "applicable": False, "layout": layout, "confidence": None}
    if not value:
        return {"status": "missing", "applicable": True, "layout": layout, "confidence": 0.0}
    return {
        "status": "present",
        "applicable": True,
        "layout": layout,
        "confidence": _confidence(value.get("confidence") if isinstance(value, Mapping) else None),
    }


def _list_field_status(layout: str, applicable: bool, values: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not applicable:
        return {"status": "not_applicable", "applicable": False, "layout": layout, "confidence": None}
    if not values:
        return {"status": "missing", "applicable": True, "layout": layout, "count": 0, "confidence": 0.0}
    confidences = [float(item.get("confidence", 0.0)) for item in values]
    return {
        "status": "present",
        "applicable": True,
        "layout": layout,
        "count": len(values),
        "confidence": _confidence(sum(confidences) / len(confidences)),
    }


def _shop_status(layout: str, shop: list[Mapping[str, Any]]) -> dict[str, Any]:
    if layout != LAYOUT_NORMAL_SHOP:
        return {"status": "not_applicable", "applicable": False, "layout": layout, "confidence": None}
    if not shop:
        return {"status": "missing", "applicable": True, "layout": layout, "slots": 0, "confidence": 0.0}
    known = [slot for slot in shop if slot.get("state") in {"empty", "occupied"}]
    confidences = [float(slot.get("confidence", 0.0)) for slot in known]
    status = "present" if len(known) == len(shop) else "partial"
    return {
        "status": status,
        "applicable": True,
        "layout": layout,
        "slots": len(shop),
        "known_slots": len(known),
        "confidence": _confidence(sum(confidences) / len(confidences)) if confidences else 0.0,
    }


def _region_text(ocr: Mapping[str, Any], key: str) -> str:
    regions = _ocr_regions_map(ocr)
    region = regions.get(key) if isinstance(regions, Mapping) else None
    if isinstance(region, Mapping):
        return str(region.get("text") or "")
    parsed = ocr.get("parsed") if isinstance(ocr.get("parsed"), Mapping) else {}
    value = parsed.get(key) if isinstance(parsed, Mapping) else None
    return str(value or "")


def _region_confidence(ocr: Mapping[str, Any], key: str) -> float | None:
    regions = _ocr_regions_map(ocr)
    region = regions.get(key) if isinstance(regions, Mapping) else None
    if isinstance(region, Mapping) and region.get("confidence") is not None:
        return float(region["confidence"])
    return None


def _ocr_regions_map(ocr: Mapping[str, Any]) -> Mapping[str, Any]:
    regions = ocr.get("regions") if isinstance(ocr.get("regions"), Mapping) else {}
    return regions


def _parse_int(text: Any) -> int | None:
    match = re.search(r"\d+", str(text or ""))
    return int(match.group(0)) if match else None


def _parse_last_int(text: Any) -> int | None:
    matches = re.findall(r"\d+", str(text or ""))
    return int(matches[-1]) if matches else None


def _parse_stage(text: Any) -> str | None:
    match = re.search(r"(?<!\d)\d+\s*-\s*\d+(?!\d)", str(text or ""))
    return re.sub(r"\s+", "", match.group(0)) if match else None


def _confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 4)


def _overall_confidence(
    stage: Mapping[str, Any] | None,
    gold: Mapping[str, Any] | None,
    level: Mapping[str, Any] | None,
    xp: Mapping[str, Any] | None,
    shop: list[Mapping[str, Any]],
    augments: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
) -> float:
    values = [
        item.get("confidence")
        for item in [stage, gold, level, xp]
        if isinstance(item, Mapping) and item.get("confidence") is not None
    ]
    values.extend(item.get("confidence") for item in shop if item.get("confidence") is not None)
    values.extend(item.get("confidence") for item in augments if item.get("confidence") is not None)
    if not values:
        return 0.0 if warnings else 0.5
    penalty = 0.15 * len(warnings)
    return round(max(0.0, (sum(float(value) for value in values) / len(values)) - penalty), 4)


def _ocr_summary(ocr: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(ocr.get("available")),
        "status": str(ocr.get("status") or "unknown"),
        "error": ocr.get("error"),
        "regions": ocr.get("regions") if isinstance(ocr.get("regions"), Mapping) else {},
    }


def _readiness_taxonomy() -> dict[str, dict[str, Any]]:
    return {
        layout: {
            "required_checks": list(rule.get("required_checks") or []),
            "not_applicable": list(rule.get("not_applicable") or []),
            "description": rule.get("description") or "",
            "excluded_from_readiness": bool(rule.get("excluded_from_readiness")),
        }
        for layout, rule in READINESS_LAYOUT_RULES.items()
    }


def _recognition_readiness(
    recognition: Mapping[str, Any],
    *,
    sample: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    layout = _readiness_layout(sample=sample, recognition=recognition)
    rule = READINESS_LAYOUT_RULES.get(layout, READINESS_LAYOUT_RULES[READINESS_LAYOUT_UNKNOWN])
    blockers: list[dict[str, Any]] = []
    if layout == READINESS_LAYOUT_POPUP:
        blockers.append(
            _blocking_issue(
                "contaminated_by_hover",
                "layout",
                "Sample is contaminated by a popup, tooltip, hover card, or modal overlay.",
            )
        )
    elif layout == READINESS_LAYOUT_UNKNOWN:
        blockers.append(
            _blocking_issue(
                "layout_unknown",
                "layout",
                "Sample could not be assigned to a supported TFT readiness layout.",
            )
        )
    elif not recognition.get("success"):
        error = recognition.get("error") if isinstance(recognition.get("error"), Mapping) else {}
        blockers.append(
            _blocking_issue(
                _error_blocker_code(str(error.get("code") or "recognition_failed")),
                "recognition",
                str(error.get("message") or "Recognition failed for this frame."),
            )
        )
    else:
        blockers.extend(_required_check_blockers(layout, recognition))

    excluded = bool(rule.get("excluded_from_readiness"))
    if not recognition.get("success") and layout != READINESS_LAYOUT_UNKNOWN:
        status = "failed"
    elif excluded:
        status = "contaminated"
    elif _is_partial_normal_shop_readiness(layout, recognition, blockers):
        status = "partial"
    elif blockers:
        status = "blocked"
    else:
        status = "ready"
    return {
        "layout": layout,
        "readiness": status,
        "status": status,
        "required_checks": list(rule.get("required_checks") or []),
        "not_applicable": list(rule.get("not_applicable") or []),
        "blocking_issues": blockers,
        "excluded_from_readiness": excluded,
    }


def _readiness_layout(
    *,
    sample: Mapping[str, Any] | None,
    recognition: Mapping[str, Any],
) -> str:
    context = _sample_context_text(sample, recognition)
    if _contains_any(context, ("hover", "tooltip", "popup", "dialog", "modal", "overlay")):
        return READINESS_LAYOUT_POPUP
    if _contains_any(context, ("loading", "loadscreen", "loading_screen", "black_screen")):
        return READINESS_LAYOUT_LOADING

    expected_layout = str((sample or {}).get("expected_layout") or "").strip().lower()
    direct = _direct_readiness_layout(expected_layout)
    if direct:
        return direct
    if expected_layout == LAYOUT_SPECIAL:
        return _special_readiness_layout(context)

    inferred = _special_readiness_layout(context)
    if inferred != READINESS_LAYOUT_UNKNOWN:
        return inferred

    recognition_layout = str(recognition.get("layout") or "").strip().lower()
    direct = _direct_readiness_layout(recognition_layout)
    if direct:
        return direct
    if recognition_layout == LAYOUT_SPECIAL:
        return READINESS_LAYOUT_UNKNOWN
    return READINESS_LAYOUT_UNKNOWN


def _sample_context_text(sample: Mapping[str, Any] | None, recognition: Mapping[str, Any]) -> str:
    values = [
        (sample or {}).get("expected_layout"),
        (sample or {}).get("label"),
        (sample or {}).get("id"),
        (sample or {}).get("image_path"),
        recognition.get("layout"),
        recognition.get("image_path"),
    ]
    return " ".join(str(value or "").lower().replace("-", "_") for value in values)


def _direct_readiness_layout(layout: str) -> str | None:
    if layout == LAYOUT_NORMAL_SHOP:
        return LAYOUT_NORMAL_SHOP
    if layout == LAYOUT_COMBAT:
        return LAYOUT_COMBAT
    if layout in {LAYOUT_AUGMENT_SELECT, READINESS_LAYOUT_AUGMENT}:
        return READINESS_LAYOUT_AUGMENT
    if layout in {
        READINESS_LAYOUT_CAROUSEL,
        READINESS_LAYOUT_PORTAL,
        READINESS_LAYOUT_POPUP,
        READINESS_LAYOUT_LOADING,
        READINESS_LAYOUT_UNKNOWN,
    }:
        return layout
    return None


def _special_readiness_layout(context: str) -> str:
    if _contains_any(context, ("carousel", "draft", "shared_draft", "选秀")):
        return READINESS_LAYOUT_CAROUSEL
    if _contains_any(context, ("portal", "region", "encounter", "传送门", "地区")):
        return READINESS_LAYOUT_PORTAL
    return READINESS_LAYOUT_UNKNOWN


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _required_check_blockers(layout: str, recognition: Mapping[str, Any]) -> list[dict[str, Any]]:
    if layout == LAYOUT_NORMAL_SHOP:
        return _normal_shop_blockers(recognition)
    if layout == READINESS_LAYOUT_AUGMENT:
        return _augment_blockers(recognition)
    return []


def _normal_shop_blockers(recognition: Mapping[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for field in ("stage", "gold", "level", "xp"):
        if not recognition.get(field):
            blockers.append(_blocking_issue("ocr_failed", field, f"Required {field} OCR value is missing."))

    shop = [slot for slot in recognition.get("shop", []) if isinstance(slot, Mapping)]
    if not shop:
        blockers.append(_blocking_issue("roi_misaligned", "shop_slots", "No shop slots were produced for a normal shop sample."))
        return blockers

    unknown_slots = [slot for slot in shop if slot.get("state") not in {"empty", "occupied"}]
    if unknown_slots:
        blockers.append(
            _blocking_issue(
                "roi_misaligned",
                "shop_slots",
                "Some shop slots have unknown occupancy.",
                count=len(unknown_slots),
            )
        )

    occupied = [slot for slot in shop if slot.get("state") == "occupied"]
    blockers.extend(_shop_text_blockers(occupied))
    return blockers


def _shop_text_blockers(occupied: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocker_groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for slot in occupied:
        for code, check, message in _shop_slot_text_blockers(slot):
            key = (code, check, message)
            group = blocker_groups.setdefault(key, {"count": 0, "slots": []})
            group["count"] += 1
            slot_number = _safe_slot_number(slot)
            if slot_number is not None:
                group["slots"].append(slot_number)
    return [
        _blocking_issue(code, check, message, count=payload["count"], slots=payload["slots"])
        for (code, check, message), payload in sorted(blocker_groups.items())
    ]


def _is_partial_normal_shop_readiness(
    layout: str,
    recognition: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> bool:
    if layout != LAYOUT_NORMAL_SHOP or not blockers:
        return False
    shop = [slot for slot in recognition.get("shop", []) if isinstance(slot, Mapping)]
    if not shop or any(issue.get("code") == "roi_misaligned" for issue in blockers):
        return False
    occupied = [slot for slot in shop if slot.get("state") == "occupied"]
    if not occupied:
        return False
    has_any_name = any(slot.get("name_candidate") or slot.get("name") for slot in occupied)
    has_any_cost = any(slot.get("cost_candidate", slot.get("cost")) is not None for slot in occupied)
    if not (has_any_name or has_any_cost):
        return False
    shop_blockers = [issue for issue in blockers if str(issue.get("check") or "").startswith("shop_")]
    non_shop_blockers = [issue for issue in blockers if issue not in shop_blockers]
    return bool(shop_blockers) and not non_shop_blockers


def _shop_slot_text_blockers(slot: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    has_any_text = bool(slot.get("raw_text") or slot.get("name_raw_text") or slot.get("cost_raw_text"))
    if not has_any_text:
        return [
            (
                "shop_slot_occupied_but_no_text",
                "shop_slots",
                "Occupied shop slots produced no OCR text.",
            )
        ]
    blockers = []
    if not (slot.get("name_candidate") or slot.get("name")):
        code = "shop_name_ocr_failed" if slot.get("name_raw_text") else "shop_name_crop_empty"
        blockers.append((code, "shop_names", "Occupied shop slots are missing name candidates."))
    if slot.get("cost_candidate", slot.get("cost")) is None:
        code = "shop_cost_parse_failed" if slot.get("cost_raw_text") else "shop_cost_ocr_failed"
        blockers.append((code, "shop_costs", "Occupied shop slots are missing cost candidates."))
    return blockers


def _augment_blockers(recognition: Mapping[str, Any]) -> list[dict[str, Any]]:
    options = [option for option in recognition.get("augments", []) if isinstance(option, Mapping)]
    if not options:
        return [_blocking_issue("ocr_failed", "augment_options", "No augment options were produced for an augment sample.")]
    blockers: list[dict[str, Any]] = []
    missing_titles = [option for option in options if not (option.get("title_candidate") or option.get("title"))]
    missing_descriptions = [
        option for option in options if not (option.get("description_candidate") or option.get("description"))
    ]
    if missing_titles:
        blockers.append(
            _blocking_issue(
                "ocr_failed",
                "augment_titles",
                "Augment options are missing title candidates.",
                count=len(missing_titles),
            )
        )
    if missing_descriptions:
        blockers.append(
            _blocking_issue(
                "ocr_failed",
                "augment_descriptions",
                "Augment options are missing description candidates.",
                count=len(missing_descriptions),
            )
        )
    return blockers


def _safe_slot_number(slot: Mapping[str, Any]) -> int | None:
    try:
        value = int(slot.get("slot"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _blocking_issue(
    code: str,
    check: str,
    message: str,
    *,
    count: int | None = None,
    slots: list[int] | None = None,
) -> dict[str, Any]:
    issue = {"code": code, "check": check, "message": message}
    if count is not None:
        issue["count"] = count
    if slots:
        issue["slots"] = sorted(set(slots))
    return issue


def _error_blocker_code(code: str) -> str:
    if code in {"unsupported_aspect_ratio", "low_resolution"}:
        return "low_resolution"
    if code.startswith("ocr_"):
        return "ocr_failed"
    return code or "recognition_failed"


def _layout_manifest_report(results: list[Mapping[str, Any]], manifest_path: Path) -> dict[str, Any]:
    samples = []
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        readiness = recognition.get("readiness") if isinstance(recognition.get("readiness"), Mapping) else {}
        samples.append(
            {
                "index": item.get("index"),
                "label": item.get("label") or "",
                "image_path": item.get("image_path"),
                "expected_layout": item.get("expected_layout"),
                "recognition_layout": recognition.get("layout"),
                "layout": readiness.get("layout") or READINESS_LAYOUT_UNKNOWN,
                "required_checks": list(readiness.get("required_checks") or []),
                "not_applicable": list(readiness.get("not_applicable") or []),
                "blocking_issues": list(readiness.get("blocking_issues") or []),
                "readiness": readiness.get("readiness") or "blocked",
                "excluded_from_readiness": bool(readiness.get("excluded_from_readiness")),
            }
        )
    return {
        "type": "tft_layout_manifest",
        "schema_version": 1,
        "report_version": "layout_manifest_v1",
        "report_path": str(manifest_path.resolve()),
        "taxonomy": _readiness_taxonomy(),
        "summary": _layout_manifest_summary(samples),
        "samples": samples,
    }


def _layout_manifest_summary(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for layout in READINESS_LAYOUT_ORDER:
        layout_samples = [sample for sample in samples if sample.get("layout") == layout]
        if not layout_samples:
            continue
        summary[layout] = _layout_sample_summary(layout, layout_samples)
    return summary


def _layout_sample_summary(layout: str, samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    blocker_counts: dict[str, int] = {}
    for sample in samples:
        for issue in sample.get("blocking_issues", []):
            if isinstance(issue, Mapping):
                code = str(issue.get("code") or "unknown")
                blocker_counts[code] = blocker_counts.get(code, 0) + 1
    ready = sum(1 for sample in samples if sample.get("readiness") == "ready")
    blocked = sum(1 for sample in samples if sample.get("readiness") == "blocked")
    failed = sum(1 for sample in samples if sample.get("readiness") == "failed")
    contaminated = sum(1 for sample in samples if sample.get("readiness") == "contaminated")
    rule = READINESS_LAYOUT_RULES.get(layout, READINESS_LAYOUT_RULES[READINESS_LAYOUT_UNKNOWN])
    return {
        "samples": len(samples),
        "ready": ready,
        "blocked": blocked,
        "failed": failed,
        "contaminated": contaminated,
        "main_blocker": _main_blocker(blocker_counts),
        "blockers": blocker_counts,
        "required_checks": list(rule.get("required_checks") or []),
        "not_applicable": list(rule.get("not_applicable") or []),
    }


def _layout_counts(results: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        layout = str((item.get("recognition") or {}).get("layout") or "unknown")
        counts[layout] = counts.get(layout, 0) + 1
    return counts


def _infer_report_shop_costs(
    results: list[Mapping[str, Any]],
    existing_labels: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> None:
    existing_labels = existing_labels or {}
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        sample_index = item.get("index")
        for slot in recognition.get("shop", []):
            if not isinstance(slot, dict) or slot.get("state") != "occupied":
                continue
            slot_number = int(slot.get("slot") or 0)
            label = existing_labels.get((str(sample_index), slot_number), {})
            human = _human_label(_label_payload(label))
            if human.get("status") != "verified":
                continue
            if human.get("name") and not slot.get("name_candidate"):
                slot["name_candidate"] = human.get("name")
                slot["name"] = human.get("name")
                slot["name_candidate_source"] = "human_verified_label"
            if slot.get("cost_candidate", slot.get("cost")) is None and human.get("cost") is not None:
                slot["cost_candidate"] = human.get("cost")
                slot["cost"] = human.get("cost")
                slot["cost_candidate_source"] = "human_verified_label"
                slot["cost_inference"] = {
                    "method": "human_verified_label",
                    "confidence": 1.0,
                }

    costs_by_name: dict[str, set[int]] = {}
    for slot in _report_occupied_shop_slots(results):
        name = _shop_cost_consensus_key(slot)
        cost = slot.get("cost_candidate", slot.get("cost"))
        if name and isinstance(cost, int) and 1 <= cost <= 5:
            costs_by_name.setdefault(name, set()).add(cost)

    consensus = {name: next(iter(costs)) for name, costs in costs_by_name.items() if len(costs) == 1}
    for slot in _report_occupied_shop_slots(results):
        if slot.get("cost_candidate", slot.get("cost")) is not None:
            continue
        name = _shop_cost_consensus_key(slot)
        if not name or name not in consensus:
            continue
        slot["cost_candidate"] = consensus[name]
        slot["cost"] = consensus[name]
        slot["cost_candidate_source"] = "report_name_cost_consensus"
        slot["cost_inference"] = {
            "method": "report_name_cost_consensus",
            "matched_name": name,
            "confidence": 0.55,
        }


def _report_occupied_shop_slots(results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    slots = []
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        for slot in recognition.get("shop", []):
            if isinstance(slot, dict) and slot.get("state") == "occupied":
                slots.append(slot)
    return slots


def _shop_cost_consensus_key(slot: Mapping[str, Any]) -> str:
    raw_name = _shop_unit_name_from_raw_text(slot.get("raw_text"))
    return raw_name or str(slot.get("name_candidate") or slot.get("name") or "").strip()


def _shop_unit_name_from_raw_text(text: Any) -> str:
    lines = _ocr_lines(text)
    if not lines:
        return ""
    candidate = re.sub(r"^\s*[1-5]\s*", "", lines[-1])
    candidate = re.sub(r"\s*[1-5]\s*$", "", candidate).strip(" -:锛殀")
    return candidate


def _readiness_summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_layout: dict[str, list[Mapping[str, Any]]] = {}
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        readiness = recognition.get("readiness") if isinstance(recognition.get("readiness"), Mapping) else {}
        layout = str(readiness.get("layout") or READINESS_LAYOUT_UNKNOWN)
        by_layout.setdefault(layout, []).append(recognition)
    return {layout: _layout_readiness(layout, items) for layout, items in by_layout.items()}


def _layout_readiness(layout: str, items: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ["stage", "gold", "level", "xp", "shop", "augments", "traits", "items"]
    field_summaries = {
        field: _field_readiness([item.get("field_status", {}).get(field, {}) for item in items])
        for field in fields
    }
    readiness_items = [item.get("readiness") for item in items if isinstance(item.get("readiness"), Mapping)]
    ready = sum(1 for item in readiness_items if item.get("readiness") == "ready")
    partial = sum(1 for item in readiness_items if item.get("readiness") == "partial")
    blocked = sum(1 for item in readiness_items if item.get("readiness") == "blocked")
    failed = sum(1 for item in readiness_items if item.get("readiness") == "failed")
    contaminated = sum(1 for item in readiness_items if item.get("readiness") == "contaminated")
    excluded = sum(1 for item in readiness_items if item.get("excluded_from_readiness"))
    blocker_counts: dict[str, int] = {}
    for item in readiness_items:
        for issue in item.get("blocking_issues", []):
            if isinstance(issue, Mapping):
                code = str(issue.get("code") or "unknown")
                blocker_counts[code] = blocker_counts.get(code, 0) + 1

    considered = len(items) - excluded
    if contaminated and considered == 0:
        status = "contaminated"
    elif blocked == 0 and failed == 0 and partial == 0:
        status = "ready"
    elif partial or ready:
        status = "partial"
    elif failed and not blocked:
        status = "failed"
    else:
        status = "blocked"
    rule = READINESS_LAYOUT_RULES.get(layout, READINESS_LAYOUT_RULES[READINESS_LAYOUT_UNKNOWN])
    return {
        "status": status,
        "total": len(items),
        "samples": len(items),
        "ready": ready,
        "partial": partial,
        "blocked": blocked,
        "failed": failed,
        "contaminated": contaminated,
        "excluded_from_readiness": excluded,
        "main_blocker": _main_blocker(blocker_counts),
        "blockers": blocker_counts,
        "required_checks": list(rule.get("required_checks") or []),
        "not_applicable": list(rule.get("not_applicable") or []),
        "fields": field_summaries,
    }


def _main_blocker(blocker_counts: Mapping[str, int]) -> str | None:
    if not blocker_counts:
        return None
    return sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _field_readiness(statuses: list[Any]) -> dict[str, Any]:
    normalized = [status for status in statuses if isinstance(status, Mapping)]
    not_applicable = sum(1 for status in normalized if status.get("status") == "not_applicable")
    present = sum(1 for status in normalized if status.get("status") == "present")
    partial = sum(1 for status in normalized if status.get("status") == "partial")
    missing = sum(1 for status in normalized if status.get("status") == "missing")
    applicable = len(normalized) - not_applicable
    if applicable == 0:
        status = "not_applicable"
    elif missing == 0 and partial == 0:
        status = "ready"
    elif present or partial:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "total": len(normalized),
        "applicable": applicable,
        "present": present,
        "partial": partial,
        "missing": missing,
        "not_applicable": not_applicable,
    }


def _recognition_metrics(
    results: list[Mapping[str, Any]],
    shop_labels: Mapping[str, Any] | None = None,
    augment_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recognitions = [item.get("recognition", {}) for item in results if isinstance(item.get("recognition"), Mapping)]
    normal_shop = [item for item in recognitions if item.get("layout") == LAYOUT_NORMAL_SHOP]
    augment_select = [item for item in recognitions if item.get("layout") == LAYOUT_AUGMENT_SELECT]
    shop_slots = [slot for item in normal_shop for slot in item.get("shop", []) if isinstance(slot, Mapping)]
    occupied_slots = [slot for slot in shop_slots if slot.get("state") == "occupied"]
    augment_options = [option for item in augment_select for option in item.get("augments", []) if isinstance(option, Mapping)]
    shop_label_samples = (shop_labels or {}).get("samples") if isinstance(shop_labels, Mapping) else []
    augment_samples = (augment_review or {}).get("samples") if isinstance(augment_review, Mapping) else []
    shop_human_labels = [
        sample.get("human", {})
        for sample in shop_label_samples
        if isinstance(sample, Mapping) and isinstance(sample.get("human"), Mapping)
    ]
    augment_human_labels = [
        option.get("human_label", {})
        for sample in augment_samples
        if isinstance(sample, Mapping)
        for option in sample.get("augments", [])
        if isinstance(option, Mapping) and isinstance(option.get("human_label"), Mapping)
    ]
    return {
        "stage_present_rate": _rate(recognitions, lambda item: bool(item.get("stage"))),
        "gold_present_rate": _rate(normal_shop, lambda item: bool(item.get("gold"))),
        "level_xp_present_rate": _rate(normal_shop, lambda item: bool(item.get("level")) and bool(item.get("xp"))),
        "shop_slot_state_rate": _rate(shop_slots, lambda slot: slot.get("state") in {"empty", "occupied"}),
        "shop_cost_present_rate": _rate(occupied_slots, lambda slot: slot.get("cost_candidate") is not None),
        "shop_name_present_rate": _rate(occupied_slots, lambda slot: bool(slot.get("name_candidate"))),
        "augment_title_present_rate": _rate(augment_options, lambda option: bool(option.get("title_candidate") or option.get("title"))),
        "shop_cost_candidate_rate": _rate(occupied_slots, lambda slot: slot.get("cost_candidate") is not None),
        "shop_name_candidate_rate": _rate(occupied_slots, lambda slot: bool(slot.get("name_candidate"))),
        "shop_cost_verified_rate": _rate(shop_human_labels, lambda label: label.get("status") == "verified" and label.get("cost") is not None),
        "shop_name_verified_rate": _rate(shop_human_labels, lambda label: label.get("status") == "verified" and bool(label.get("name"))),
        "augment_title_candidate_rate": _rate(
            augment_options,
            lambda option: bool(option.get("title_candidate") or option.get("title")),
        ),
        "augment_description_candidate_rate": _rate(
            augment_options,
            lambda option: bool(option.get("description_candidate") or option.get("description")),
        ),
        "augment_title_verified_rate": _rate(
            augment_human_labels,
            lambda label: label.get("status") == "verified" and bool(label.get("title")),
        ),
        "augment_description_verified_rate": _rate(
            augment_human_labels,
            lambda label: label.get("status") == "verified" and bool(label.get("description")),
        ),
        "shop_occupied_slot_count": len(occupied_slots),
        "shop_label_count": len(shop_human_labels),
        "augment_option_count": len(augment_options),
        "augment_label_count": len(augment_human_labels),
    }


def _rate(items: list[Any], predicate: Any) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if predicate(item)) / len(items), 4)


def _shop_review_report(
    results: list[Mapping[str, Any]],
    report_path: Path,
    crops_dir: Path,
    existing_labels: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    samples = []
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        if recognition.get("layout") != LAYOUT_NORMAL_SHOP:
            continue
        sample_crops_dir = crops_dir / f"sample_{item.get('index')}"
        shop = _shop_review_slots(
            image_path=item.get("image_path"),
            sample_index=item.get("index"),
            shop=recognition.get("shop") or [],
            crops_dir=sample_crops_dir,
            existing_labels=existing_labels,
        )
        samples.append(
            {
                "index": item.get("index"),
                "label": item.get("label") or "",
                "image_path": item.get("image_path"),
                "layout": recognition.get("layout"),
                "shop": shop,
            }
        )
    return {
        "type": "tft_shop_review",
        "schema_version": 1,
        "report_version": "recognition_shop_review_v1",
        "report_path": str(report_path.resolve()),
        "samples": samples,
    }


def _shop_review_slots(
    *,
    image_path: Any,
    sample_index: Any,
    shop: list[Mapping[str, Any]],
    crops_dir: Path,
    existing_labels: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    image = _open_review_image(image_path)
    reviewed = []
    for slot in shop:
        item = dict(slot)
        slot_number = int(item.get("slot") or 0)
        label = existing_labels.get((str(sample_index), slot_number), {})
        human_label = _human_label(label.get("human") if isinstance(label, Mapping) else None)
        item["name_candidate"] = item.get("name_candidate") or item.get("name")
        item["cost_candidate"] = item.get("cost_candidate", item.get("cost"))
        item["human_label"] = human_label
        item["crop_paths"] = {}
        if image is not None and isinstance(item.get("bbox"), list):
            subregions = _shop_slot_subregions(tuple(int(value) for value in item["bbox"]))
            item["crop_quality"] = _shop_crop_quality(tuple(int(value) for value in item["bbox"]), subregions)
            if item.get("state") != "occupied":
                subregions = {"slot_full": subregions["slot_full"]}
            item["crop_paths"] = _save_review_crops(image, subregions, crops_dir, f"slot_{slot_number}")
        else:
            item["crop_quality"] = {"status": "roi_misaligned", "issues": ["missing_image_or_bbox"]}
        reviewed.append(item)
    if image is not None:
        image.close()
    return reviewed


def _shop_slot_subregions(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    return {
        "slot_full": bbox,
        "slot_name": (
            left + int(width * 0.14),
            top + int(height * 0.76),
            right - int(width * 0.05),
            bottom - int(height * 0.03),
        ),
        "slot_cost": (
            left,
            top + int(height * 0.76),
            left + int(width * 0.18),
            bottom - int(height * 0.03),
        ),
        "slot_traits": (
            left + int(width * 0.04),
            top + int(height * 0.22),
            left + int(width * 0.58),
            top + int(height * 0.75),
        ),
    }


def _shop_crop_quality(
    bbox: tuple[int, int, int, int],
    subregions: Mapping[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    issues = []
    for key, region in subregions.items():
        if not _bbox_inside(region, bbox):
            issues.append(f"{key}_outside_slot")
    name = subregions["slot_name"]
    cost = subregions["slot_cost"]
    traits = subregions["slot_traits"]
    if (name[2] - name[0]) < width * 0.45 or (name[3] - name[1]) < height * 0.12:
        issues.append("name_crop_too_small")
    if (cost[2] - cost[0]) < width * 0.16 or (cost[3] - cost[1]) < height * 0.12:
        issues.append("cost_crop_too_small")
    if _bboxes_overlap(cost, traits):
        issues.append("traits_overlap_cost")
    if any(issue.endswith("_outside_slot") or issue == "traits_overlap_cost" for issue in issues):
        status = "roi_misaligned"
    elif "name_crop_too_small" in issues:
        status = "name_crop_too_small"
    elif "cost_crop_too_small" in issues:
        status = "cost_crop_too_small"
    else:
        status = "roi_ok"
    return {"status": status, "issues": issues}


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] < inner[2] <= outer[2] and outer[1] <= inner[1] < inner[3] <= outer[3]


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _shop_labels_report(shop_review: Mapping[str, Any], labels_path: Path) -> dict[str, Any]:
    samples = []
    for sample in shop_review.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        for slot in sample.get("shop", []):
            if not isinstance(slot, Mapping) or slot.get("state") != "occupied":
                continue
            samples.append(
                {
                    "index": sample.get("index"),
                    "label": sample.get("label") or "",
                    "slot": slot.get("slot"),
                    "image_path": sample.get("image_path"),
                    "crop_path": (slot.get("crop_paths") or {}).get("slot_full"),
                    "machine": {
                        "name_candidate": slot.get("name_candidate"),
                        "cost_candidate": slot.get("cost_candidate"),
                        "raw_text": slot.get("raw_text") or "",
                        "confidence": slot.get("confidence"),
                    },
                    "human": slot.get("human_label") or _human_label(None),
                }
            )
    return {
        "type": "tft_shop_labels",
        "schema_version": 1,
        "report_version": "recognition_shop_labels_v1",
        "report_path": str(labels_path.resolve()),
        "samples": samples,
    }


def _augment_review_report(
    results: list[Mapping[str, Any]],
    report_path: Path,
    crops_dir: Path,
    existing_labels: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    samples = []
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        if recognition.get("layout") != LAYOUT_AUGMENT_SELECT:
            continue
        image = _open_review_image(item.get("image_path"))
        sample_crops_dir = crops_dir / f"sample_{item.get('index')}" / "augments"
        augments = []
        for option in recognition.get("augments", []):
            if not isinstance(option, Mapping):
                continue
            slot_number = int(option.get("slot") or 0)
            label = existing_labels.get((str(item.get("index")), slot_number), {})
            crop_path = None
            if image is not None and isinstance(option.get("bbox"), list):
                saved = _save_review_crops(
                    image,
                    {"option": tuple(int(value) for value in option["bbox"])},
                    sample_crops_dir,
                    f"augment_{slot_number}",
                )
                crop_path = saved.get("option")
            augments.append(
                {
                    "slot": option.get("slot"),
                    "title_candidate": option.get("title_candidate") or option.get("title"),
                    "description_candidate": option.get("description_candidate") or option.get("description"),
                    "raw_text": option.get("raw_text") or "",
                    "confidence": option.get("confidence"),
                    "review_status": option.get("review_status") or "needs_check",
                    "crop_path": crop_path,
                    "bbox": option.get("bbox"),
                    "human_label": _human_label(_label_payload(label), augment=True),
                }
            )
        if image is not None:
            image.close()
        samples.append(
            {
                "index": item.get("index"),
                "label": item.get("label") or "",
                "image_path": item.get("image_path"),
                "layout": recognition.get("layout"),
                "augments": augments,
            }
        )
    return {
        "type": "tft_augment_review",
        "schema_version": 1,
        "report_version": "recognition_augment_review_v1",
        "report_path": str(report_path.resolve()),
        "samples": samples,
    }


def _open_review_image(image_path: Any) -> Image.Image | None:
    try:
        image = Image.open(Path(str(image_path)).expanduser())
        image.load()
        return image
    except Exception:
        return None


def _save_review_crops(
    image: Image.Image,
    regions: Mapping[str, tuple[int, int, int, int]],
    crops_dir: Path,
    prefix: str,
) -> dict[str, str]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key, bbox in regions.items():
        crop_path = crops_dir / f"{prefix}_{key}.png"
        image.crop(bbox).save(crop_path)
        saved[key] = str(crop_path.resolve())
    return saved


def _load_human_labels(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    labels = {}
    for sample in payload.get("samples", []):
        if not isinstance(sample, Mapping):
            continue
        slot = sample.get("slot")
        if slot is None:
            slot = sample.get("augment_slot")
        if slot is not None:
            labels[(str(sample.get("index")), int(slot))] = sample
        for option in sample.get("augments", []):
            if not isinstance(option, Mapping):
                continue
            option_slot = option.get("slot") if option.get("slot") is not None else option.get("augment_slot")
            if option_slot is not None:
                labels[(str(sample.get("index")), int(option_slot))] = {"index": sample.get("index"), **option}
    return labels


def _label_payload(label: Any) -> Any:
    if not isinstance(label, Mapping):
        return None
    if isinstance(label.get("human"), Mapping):
        return label.get("human")
    if isinstance(label.get("human_label"), Mapping):
        return label.get("human_label")
    return label


def _human_label(label: Any, *, augment: bool = False) -> dict[str, Any]:
    source = dict(label) if isinstance(label, Mapping) else {}
    if augment:
        return {**source, "title": source.get("title"), "description": source.get("description"), "status": source.get("status") or "unreviewed"}
    return {**source, "name": source.get("name"), "cost": source.get("cost"), "status": source.get("status") or "unreviewed"}


def _load_calibration_report(report_or_path: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report_or_path, Mapping):
        return report_or_path
    return json.loads(Path(report_or_path).expanduser().read_text(encoding="utf-8"))


def _error_result(path: Path, layout: str, code: str, message: str) -> dict[str, Any]:
    result = {
        "type": "tft_recognition_result",
        "schema_version": 1,
        "success": False,
        "image_path": str(path),
        "layout": layout,
        "stage": None,
        "gold": None,
        "level": None,
        "xp": None,
        "shop": [],
        "augments": [],
        "traits": [],
        "items": [],
        "field_status": _field_statuses(
            layout,
            stage=None,
            gold=None,
            level=None,
            xp=None,
            shop=[],
            augments=[],
        ),
        "ocr": {"available": False, "status": "skipped", "error": None, "regions": {}},
        "confidence": 0.0,
        "warnings": [{"code": code, "message": message}],
        "error": {"code": code, "message": message},
    }
    result["readiness"] = _recognition_readiness(result)
    return result
