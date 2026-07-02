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
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_COMBAT,
    LAYOUT_NORMAL_SHOP,
    SHOP_SLOT_KEYS,
    UnsupportedAspectRatioError,
    layout_region_bboxes,
)


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
    except UnsupportedAspectRatioError as exc:
        return _error_result(path, layout, "unsupported_aspect_ratio", str(exc))
    except OSError as exc:
        return _error_result(path, layout, "image_read_failed", str(exc))

    adapter = ocr_adapter or RapidOcrTftAdapter()
    try:
        ocr = adapter.recognize(path, regions)
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

    return {
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


def build_tft_recognition_report(
    calibration_report_or_path: Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    ocr_adapter: TftOcrAdapter | None = None,
) -> dict[str, Any]:
    calibration_report = _load_calibration_report(calibration_report_or_path)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
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
        results.append(
            {
                "index": screenshot.get("index"),
                "label": screenshot.get("label") or screenshot.get("id") or "",
                "image_path": str(image_path),
                "expected_layout": screenshot.get("expected_layout"),
                "recognition": recognition,
            }
        )

    successes = sum(1 for item in results if item["recognition"].get("success"))
    summary = {
        "total": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "layouts": _layout_counts(results),
        "warnings": sum(len(item["recognition"].get("warnings") or []) for item in results),
        "readiness": _readiness_summary(results),
    }
    report_path = output_path / "recognition_report_v1.json"
    summary_path = output_path / "recognition_summary_v1.json"
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
        "summary": summary,
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _normalize_layout(layout: str | None) -> str:
    value = str(layout or "").strip().lower()
    if value in {LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_AUGMENT_SELECT}:
        return value
    return LAYOUT_NORMAL_SHOP


def _flat_regions(regions: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in regions.items() if isinstance(value, tuple) and len(value) == 4}


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
    text = _region_text(ocr, "augments")
    if not text:
        return []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines and "选择" in lines[0]:
        lines = lines[1:]
    confidence = _confidence(_region_confidence(ocr, "augments"))
    return [
        {
            "slot": index,
            "title": line,
            "description": "",
            "raw_text": line,
            "confidence": confidence,
        }
        for index, line in enumerate(lines[:3], start=1)
    ]


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
    regions = ocr.get("regions") if isinstance(ocr.get("regions"), Mapping) else {}
    region = regions.get(key) if isinstance(regions, Mapping) else None
    if isinstance(region, Mapping):
        return str(region.get("text") or "")
    parsed = ocr.get("parsed") if isinstance(ocr.get("parsed"), Mapping) else {}
    value = parsed.get(key) if isinstance(parsed, Mapping) else None
    return str(value or "")


def _region_confidence(ocr: Mapping[str, Any], key: str) -> float | None:
    regions = ocr.get("regions") if isinstance(ocr.get("regions"), Mapping) else {}
    region = regions.get(key) if isinstance(regions, Mapping) else None
    if isinstance(region, Mapping) and region.get("confidence") is not None:
        return float(region["confidence"])
    return None


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


def _layout_counts(results: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        layout = str((item.get("recognition") or {}).get("layout") or "unknown")
        counts[layout] = counts.get(layout, 0) + 1
    return counts


def _readiness_summary(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_layout: dict[str, list[Mapping[str, Any]]] = {}
    for item in results:
        recognition = item.get("recognition") if isinstance(item.get("recognition"), Mapping) else {}
        layout = str(recognition.get("layout") or "unknown")
        by_layout.setdefault(layout, []).append(recognition)
    return {layout: _layout_readiness(items) for layout, items in by_layout.items()}


def _layout_readiness(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ["stage", "gold", "level", "xp", "shop", "augments", "traits", "items"]
    field_summaries = {
        field: _field_readiness([item.get("field_status", {}).get(field, {}) for item in items])
        for field in fields
    }
    applicable_fields = [summary for summary in field_summaries.values() if summary["status"] != "not_applicable"]
    if not applicable_fields:
        status = "not_applicable"
    elif all(summary["missing"] == 0 and summary["partial"] == 0 for summary in applicable_fields):
        status = "ready"
    elif any(summary["present"] or summary["partial"] for summary in applicable_fields):
        status = "partial"
    else:
        status = "missing"
    return {"status": status, "total": len(items), "fields": field_summaries}


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


def _load_calibration_report(report_or_path: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report_or_path, Mapping):
        return report_or_path
    return json.loads(Path(report_or_path).expanduser().read_text(encoding="utf-8"))


def _error_result(path: Path, layout: str, code: str, message: str) -> dict[str, Any]:
    return {
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
