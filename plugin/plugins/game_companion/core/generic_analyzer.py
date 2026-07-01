from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .image_source import ImageSourceError, read_image_metadata
from .local_vision import analyze_local_vision
from .vlm_fallback import apply_vlm_fallback_plan, build_vlm_fallback_plan
from .vlm_input import prepare_vlm_input, summarize_vlm_input_preparation
from .vision_schema import (
    VisionFrameAnalysis,
    build_frame_metadata,
    error_vision_payload,
    redact_sensitive_text,
    source_with_origin,
)

try:
    from PIL import Image, ImageStat
except ImportError:  # pragma: no cover - Pillow is a test/runtime dependency in normal installs.
    Image = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]


GENERIC_ANALYZER_VERSION = "generic_image_v1"


def analyze_generic_image(
    profile_id: str,
    image_path: str | Path,
    *,
    vlm_requested: bool = False,
    state_changed: bool = False,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_profile_id = str(profile_id or "generic").strip().lower() or "generic"
    try:
        image = read_image_metadata(image_path)
    except ImageSourceError as exc:
        return _error_response(exc.code, normalized_profile_id, exc.message)

    source = source_with_origin(image.to_dict(), source_context)
    quality = inspect_image_quality(image.path, width=image.width, height=image.height)
    local_vision = analyze_local_vision(image.path, profile_id=normalized_profile_id)
    scene = dict(local_vision["scene"])
    scene["source"] = GENERIC_ANALYZER_VERSION
    diagnostics = {
        "warnings": [
            {"code": flag, "message": _quality_message(flag)}
            for flag in quality["flags"]
        ],
        "analyzers": {
            "ocr": {"status": "skipped", "reason": "not_configured"},
            "detector": local_vision["diagnostics"]["detector"],
            "classifier": local_vision["diagnostics"]["classifier"],
            "template_matcher": {"status": "skipped", "reason": "not_configured"},
            "vlm": {"status": "skipped", "reason": "not_requested"},
        },
        "local_vision": local_vision,
        "quality": quality,
    }
    vision = VisionFrameAnalysis(
        profile_id=normalized_profile_id,
        source=source,
        frame=build_frame_metadata(
            image_path=image.path,
            width=image.width,
            height=image.height,
            quality=quality,
        ),
        scene=scene,
        objects=local_vision["objects"],
        ui=local_vision["ui"],
        game_state={},
        diagnostics=diagnostics,
    ).to_dict()
    vlm_plan = build_vlm_fallback_plan(
        vision,
        user_requested=vlm_requested,
        state_changed=state_changed,
    )
    vision = apply_vlm_fallback_plan(vision, vlm_plan)
    if vlm_plan.get("status") == "planned":
        vision["diagnostics"]["vlm_input_preparation"] = summarize_vlm_input_preparation(
            prepare_vlm_input(vision, image.path, plan=vlm_plan)
        )
    return {
        "ok": True,
        "success": True,
        "error": None,
        "profile": normalized_profile_id,
        "profile_id": normalized_profile_id,
        "analyzer": GENERIC_ANALYZER_VERSION,
        "source": source,
        "state": {},
        "regions": {},
        "insights": [],
        "vision": vision,
        "diagnostics": {
            "warnings": diagnostics["warnings"],
            "quality": quality,
            "ocr": diagnostics["analyzers"]["ocr"],
            "recognition": {"status": "skipped", "reason": "not_configured"},
            "analysis": {"scene": vision["scene"]},
        },
    }


def inspect_image_quality(image_path: str | Path, *, width: int, height: int) -> dict[str, Any]:
    flags: list[str] = []
    if width < 320 or height < 180:
        flags.append("low_resolution")
    mean_luma = _mean_luma(image_path)
    if mean_luma is not None:
        if mean_luma <= 8.0:
            flags.append("too_dark")
        elif mean_luma >= 247.0:
            flags.append("too_bright")
    return {
        "status": "ok" if not flags else "needs_review",
        "flags": flags,
        "width": int(width),
        "height": int(height),
        "mean_luma": round(mean_luma, 2) if mean_luma is not None else None,
    }


def _mean_luma(image_path: str | Path) -> float | None:
    if Image is None or ImageStat is None:
        return None
    with Image.open(image_path) as image:
        sample = image.convert("L")
        sample.thumbnail((256, 256))
        return float(ImageStat.Stat(sample).mean[0])


def _quality_message(flag: str) -> str:
    if flag == "low_resolution":
        return "image resolution is low for reliable game UI recognition"
    if flag == "too_dark":
        return "image is too dark for reliable visual recognition"
    if flag == "too_bright":
        return "image is too bright for reliable visual recognition"
    return flag


def _error_response(code: str, profile_id: str, message: str) -> dict[str, Any]:
    error_message = redact_sensitive_text(message)
    return {
        "ok": False,
        "success": False,
        "error": {"code": code, "message": error_message},
        "profile": profile_id,
        "profile_id": profile_id,
        "analyzer": GENERIC_ANALYZER_VERSION,
        "source": None,
        "state": {},
        "regions": {},
        "insights": [],
        "vision": error_vision_payload(profile_id, code, error_message),
        "diagnostics": {"warnings": [], "ocr": {"status": "skipped"}},
    }
