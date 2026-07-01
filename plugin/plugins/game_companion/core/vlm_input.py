from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


MAX_CROPPED_REGION_PAYLOADS = 6
REDACTION_FILL = (0, 0, 0)
SENSITIVE_REGION_TERMS = frozenset(
    {
        "account",
        "account_identifiers",
        "chat",
        "chat_area",
        "name",
        "names",
        "player",
        "player_names",
        "scoreboard",
        "scoreboard_names",
        "summoner",
    }
)


def prepare_vlm_input(
    vision: Mapping[str, Any],
    image_path: str | Path,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_plan = dict(plan or _fallback_plan_from_vision(vision))
    if fallback_plan.get("status") != "planned":
        return _empty_preparation("skipped", str(fallback_plan.get("reason") or "not_planned"), fallback_plan)

    input_policy = fallback_plan.get("input_policy") if isinstance(fallback_plan.get("input_policy"), Mapping) else {}
    if input_policy.get("send_full_frame") or fallback_plan.get("send_full_frame"):
        if fallback_plan.get("requires_desensitization") and not _has_redactions(vision):
            return _empty_preparation("blocked", "missing_desensitization_redactions", fallback_plan)
        return _prepare_full_frame(vision, image_path, fallback_plan)
    return _prepare_cropped_regions(vision, image_path, fallback_plan)


def summarize_vlm_input_preparation(preparation: Mapping[str, Any]) -> dict[str, Any]:
    payloads = preparation.get("payloads") if isinstance(preparation.get("payloads"), list) else []
    return {
        "type": "vlm_input_preparation_summary",
        "status": str(preparation.get("status") or "unknown"),
        "reason": str(preparation.get("reason") or ""),
        "payload_kind": preparation.get("payload_kind"),
        "payload_count": len(payloads),
        "privacy": deepcopy(dict(preparation.get("privacy"))) if isinstance(preparation.get("privacy"), Mapping) else {},
        "external_call_executed": bool(preparation.get("external_call_executed") is True),
        "model_calls": [],
    }


def _prepare_cropped_regions(
    vision: Mapping[str, Any],
    image_path: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    regions = _detected_ui_regions(vision)
    if not regions:
        return _empty_preparation("blocked", "no_safe_regions", plan)
    if plan.get("requires_desensitization") and not _has_redactions(vision):
        return _empty_preparation("blocked", "missing_desensitization_redactions", plan)

    payloads = []
    with Image.open(image_path) as image:
        frame = image.convert("RGB")
        for index, region in enumerate(regions[:MAX_CROPPED_REGION_PAYLOADS], start=1):
            bbox = _clip_bbox(region["bbox"], frame.size)
            if bbox is None:
                continue
            crop = frame.crop(tuple(bbox))
            for redaction in _redaction_bboxes_for_crop(vision, crop_bbox=bbox):
                _fill_bbox(crop, redaction)
            payloads.append(
                {
                    "id": f"crop_{index}",
                    "kind": "image_png_base64",
                    "scope": "detected_ui_region",
                    "mime_type": "image/png",
                    "bbox": bbox,
                    "label": region.get("label") or "",
                    "ui_type": region.get("type") or "",
                    "data_base64": _encode_png_base64(crop),
                }
            )

    if not payloads:
        return _empty_preparation("blocked", "no_safe_regions", plan)

    return {
        "type": "vlm_input_preparation",
        "status": "prepared",
        "reason": str(plan.get("reason") or ""),
        "payload_kind": "cropped_regions",
        "payloads": payloads,
        "privacy": {
            "raw_image_logging": False,
            "contains_full_frame": False,
            "redaction_applied": _has_redactions(vision),
            "requires_desensitization": bool(plan.get("requires_desensitization")),
        },
        "external_call_executed": False,
        "model_calls": [],
        "plan": deepcopy(dict(plan)),
    }


def _prepare_full_frame(
    vision: Mapping[str, Any],
    image_path: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    with Image.open(image_path) as image:
        frame = image.convert("RGB")
        for bbox in _redaction_bboxes(vision):
            clipped = _clip_bbox(bbox, frame.size)
            if clipped is not None:
                _fill_bbox(frame, clipped)
        payload = {
            "id": "desensitized_frame",
            "kind": "image_png_base64",
            "scope": "desensitized_frame",
            "mime_type": "image/png",
            "bbox": [0, 0, frame.width, frame.height],
            "data_base64": _encode_png_base64(frame),
        }

    return {
        "type": "vlm_input_preparation",
        "status": "prepared",
        "reason": str(plan.get("reason") or ""),
        "payload_kind": "desensitized_frame",
        "payloads": [payload],
        "privacy": {
            "raw_image_logging": False,
            "contains_full_frame": True,
            "redaction_applied": _has_redactions(vision),
            "requires_desensitization": bool(plan.get("requires_desensitization")),
        },
        "external_call_executed": False,
        "model_calls": [],
        "plan": deepcopy(dict(plan)),
    }


def _fallback_plan_from_vision(vision: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = vision.get("diagnostics") if isinstance(vision.get("diagnostics"), Mapping) else {}
    plan = diagnostics.get("vlm_fallback") if isinstance(diagnostics, Mapping) else {}
    return plan if isinstance(plan, Mapping) else {}


def _empty_preparation(status: str, reason: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "vlm_input_preparation",
        "status": status,
        "reason": reason,
        "payload_kind": None,
        "payloads": [],
        "privacy": {
            "raw_image_logging": False,
            "contains_full_frame": False,
            "redaction_applied": False,
            "requires_desensitization": bool(plan.get("requires_desensitization")),
        },
        "external_call_executed": False,
        "model_calls": [],
        "plan": deepcopy(dict(plan)),
    }


def _detected_ui_regions(vision: Mapping[str, Any]) -> list[dict[str, Any]]:
    regions = []
    ui_items = vision.get("ui") if isinstance(vision.get("ui"), list) else []
    for item in ui_items:
        if not isinstance(item, Mapping):
            continue
        bbox = item.get("bbox")
        if _bbox_values(bbox) is None:
            continue
        if _is_sensitive_region(item):
            continue
        regions.append(
            {
                "bbox": list(_bbox_values(bbox) or []),
                "label": item.get("label"),
                "type": item.get("type"),
            }
        )
    return regions


def _is_sensitive_region(item: Mapping[str, Any]) -> bool:
    label = str(item.get("label") or "").strip().lower()
    item_type = str(item.get("type") or "").strip().lower()
    combined = f"{item_type} {label}"
    return any(term in combined for term in SENSITIVE_REGION_TERMS)


def _redaction_bboxes_for_crop(vision: Mapping[str, Any], *, crop_bbox: list[int]) -> list[list[int]]:
    redactions = []
    crop_left, crop_top, crop_right, crop_bottom = crop_bbox
    for bbox in _redaction_bboxes(vision):
        left = max(crop_left, bbox[0])
        top = max(crop_top, bbox[1])
        right = min(crop_right, bbox[2])
        bottom = min(crop_bottom, bbox[3])
        if right > left and bottom > top:
            redactions.append([left - crop_left, top - crop_top, right - crop_left, bottom - crop_top])
    return redactions


def _redaction_bboxes(vision: Mapping[str, Any]) -> list[list[int]]:
    privacy = vision.get("privacy") if isinstance(vision.get("privacy"), Mapping) else {}
    raw_bboxes = privacy.get("redact_bboxes") if isinstance(privacy, Mapping) else []
    if not isinstance(raw_bboxes, list):
        return []
    bboxes = []
    for bbox in raw_bboxes:
        values = _bbox_values(bbox)
        if values is not None:
            bboxes.append(values)
    return bboxes


def _has_redactions(vision: Mapping[str, Any]) -> bool:
    return bool(_redaction_bboxes(vision))


def _bbox_values(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _clip_bbox(value: Any, image_size: tuple[int, int]) -> list[int] | None:
    bbox = _bbox_values(value)
    if bbox is None:
        return None
    width, height = image_size
    left = min(max(0, bbox[0]), width)
    top = min(max(0, bbox[1]), height)
    right = min(max(0, bbox[2]), width)
    bottom = min(max(0, bbox[3]), height)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _fill_bbox(image: Image.Image, bbox: list[int]) -> None:
    left, top, right, bottom = bbox
    for x in range(left, right):
        for y in range(top, bottom):
            image.putpixel((x, y), REDACTION_FILL)


def _encode_png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
