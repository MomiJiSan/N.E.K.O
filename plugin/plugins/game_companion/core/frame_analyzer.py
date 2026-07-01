from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .generic_analyzer import analyze_generic_image, inspect_image_quality
from .image_source import ImageSourceError, image_data_url_to_temp_file, read_image_metadata
from .local_vision import analyze_local_vision
from .profile_registry import ProfileMetadata, ProfileRegistry
from .vlm_fallback import apply_vlm_fallback_plan, build_vlm_fallback_plan
from .vlm_input import prepare_vlm_input, summarize_vlm_input_preparation
from .vision_schema import (
    VisionFrameAnalysis,
    build_frame_metadata,
    error_vision_payload,
    redact_sensitive_text,
    source_with_origin,
)
from ..profiles.tft.insights import generate_insights
from ..profiles.tft.ocr import analyze_tft_ocr_regions
from ..profiles.tft.recognition import recognize_shop_units
from ..profiles.tft.screen_regions import (
    UnsupportedAspectRatioError,
    grouped_screen_region_bboxes,
    layout_profile,
    layout_region_bboxes,
    save_debug_crops,
)
from ..profiles.tft.state_parser import analyze_tft_state, parse_tft_state

SUPPORTED_PROFILE_IDS = frozenset({"generic", "tft"})
ANALYZER_VERSION = "offline_image_v1"


def analyze_offline_image(
    profile_id: str,
    image_path: str | Path,
    *,
    debug_crops_dir: str | Path | None = None,
    debug_crops_layout: str | None = None,
    vlm_requested: bool = False,
    state_changed: bool = False,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_profile_id = _normalize_profile_id(profile_id)
    profile = _get_builtin_profile(normalized_profile_id)
    if profile is None:
        return _error_response("unknown_profile", normalized_profile_id)

    if profile.profile_id not in SUPPORTED_PROFILE_IDS:
        return _error_response("unsupported_profile", profile.profile_id)

    if profile.profile_id == "generic":
        return analyze_generic_image(
            profile.profile_id,
            image_path,
            vlm_requested=vlm_requested,
            state_changed=state_changed,
            source_context=source_context,
        )

    try:
        image = read_image_metadata(image_path)
    except ImageSourceError as exc:
        return _error_response(exc.code, profile.profile_id, exc.message)
    source = source_with_origin(image.to_dict(), source_context)

    warnings: list[dict[str, str]] = []
    regions: dict[str, Any] = {}
    ocr_result: dict[str, Any] = {
        "available": False,
        "status": "skipped",
        "error": None,
        "regions": {},
        "parsed": {},
    }
    recognition_result: dict[str, Any] = {
        "available": False,
        "status": "skipped",
        "kind": "units",
        "results": [],
        "diagnostics": {},
    }
    debug_crops: dict[str, Any] | None = None
    local_vision_result: dict[str, Any] | None = None
    try:
        local_vision_result = analyze_local_vision(image.path, profile_id=profile.profile_id)
        layout_hint = _layout_hint_from_source_context(source_context)
        layout = layout_profile(layout_hint) if layout_hint else None
        raw_regions = (
            layout_region_bboxes(image.width, image.height, layout_hint)
            if layout_hint
            else grouped_screen_region_bboxes(image.width, image.height)
        )
        ocr_result = analyze_tft_ocr_regions(image.path, _flat_ocr_regions(raw_regions))
        if layout is not None and not layout.deep_recognition:
            recognition_result = {
                "available": False,
                "status": "skipped",
                "reason": f"{layout.key}_layout",
                "kind": "units",
                "results": [],
                "diagnostics": {"layout": layout.key},
            }
        else:
            recognition_result = recognize_shop_units(image.path, raw_regions)
        if debug_crops_dir:
            try:
                debug_crops = save_debug_crops(image.path, debug_crops_dir, layout=debug_crops_layout or layout_hint)
            except Exception as exc:
                warnings.append({"code": "debug_crops_failed", "message": str(exc)})
        regions = _json_safe_bboxes(raw_regions)
    except UnsupportedAspectRatioError as exc:
        warnings.append({"code": "unsupported_aspect_ratio", "message": str(exc)})

    state = _empty_tft_state()
    if "layout_hint" not in locals():
        layout_hint = None
    if layout_hint:
        state["layout"] = layout_hint
    _apply_ocr_state(state, ocr_result.get("parsed", {}))
    state["shop_units"] = _shop_units_from_recognition(recognition_result)
    if ocr_result.get("status") in {"unavailable", "failed"}:
        warnings.append(
            {
                "code": f"ocr_{ocr_result['status']}",
                "message": str(ocr_result.get("error") or "OCR is not available"),
            }
        )
    if recognition_result.get("status") not in {"ready", "skipped"}:
        warnings.append(
            {
                "code": f"recognition_{recognition_result.get('status')}",
                "message": str(
                    (recognition_result.get("diagnostics") or {}).get("detail")
                    or (recognition_result.get("diagnostics") or {}).get("warning")
                    or "visual recognition is not available"
                ),
            }
        )
    parsed_state = parse_tft_state(state)
    analyzed_state = analyze_tft_state(parsed_state)
    state_analysis = _serialize_tft_state_analysis(analyzed_state)
    state["active_traits"] = state_analysis["active_traits"]
    insights = generate_insights(state, state_analysis)
    vision = _build_tft_vision_payload(
        profile_id=profile.profile_id,
        source=source,
        image_path=image.path,
        state=state,
        regions=regions,
        insights=insights,
        warnings=warnings,
        ocr_result=ocr_result,
        recognition_result=recognition_result,
        state_analysis=state_analysis,
        local_vision_result=local_vision_result,
        layout_hint=layout_hint,
        vlm_requested=vlm_requested,
        state_changed=state_changed,
    )

    return {
        "ok": True,
        "success": True,
        "error": None,
        "profile": profile.profile_id,
        "profile_id": profile.profile_id,
        "analyzer": ANALYZER_VERSION,
        "source": source,
        "state": state,
        "regions": regions,
        "insights": insights,
        "vision": vision,
        "diagnostics": {
            "warnings": warnings,
            "ocr": ocr_result,
            "recognition": recognition_result,
            "analysis": state_analysis,
            "debug_crops": debug_crops,
        },
    }


def analyze_frame(
    profile_id: str,
    image_path: str | Path,
    *,
    debug_crops_dir: str | Path | None = None,
    debug_crops_layout: str | None = None,
    vlm_requested: bool = False,
    state_changed: bool = False,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return analyze_offline_image(
        profile_id,
        image_path,
        debug_crops_dir=debug_crops_dir,
        debug_crops_layout=debug_crops_layout,
        vlm_requested=vlm_requested,
        state_changed=state_changed,
        source_context=source_context,
    )


def analyze_frame_data_url(
    profile_id: str,
    image_data_url: str,
    *,
    vlm_requested: bool = False,
    state_changed: bool = False,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temp_path: Path | None = None
    try:
        temp_path = image_data_url_to_temp_file(image_data_url)
        payload = analyze_offline_image(
            profile_id,
            temp_path,
            vlm_requested=vlm_requested,
            state_changed=state_changed,
            source_context=source_context,
        )
        if payload.get("source"):
            payload["source"]["type"] = "image_data_url"
            payload["source"]["path"] = None
        vision = payload.get("vision")
        if isinstance(vision, dict) and isinstance(vision.get("source"), dict):
            vision["source"]["type"] = "image_data_url"
            vision["source"]["path"] = None
        return payload
    except ImageSourceError as exc:
        return _error_response(exc.code, _normalize_profile_id(profile_id), exc.message)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _normalize_profile_id(profile_id: str) -> str:
    return str(profile_id or "").strip().lower()


def _get_builtin_profile(profile_id: str) -> ProfileMetadata | None:
    registry = ProfileRegistry()
    for profile in _builtin_profiles():
        registry.register(profile)
    return registry.get(profile_id)


def _builtin_profiles() -> tuple[ProfileMetadata, ...]:
    from ..profiles import builtin_profiles

    return builtin_profiles()


def _empty_tft_state() -> dict[str, Any]:
    return {
        "stage": None,
        "level": None,
        "gold": None,
        "board_units": [],
        "bench_units": [],
        "shop_units": [],
        "items": [],
        "traits": [],
        "round": None,
        "augments": [],
        "layout": None,
    }


def _error_response(code: str, profile_id: str, message: str | None = None) -> dict[str, Any]:
    error_message = redact_sensitive_text(message or _default_error_message(code, profile_id))
    return {
        "ok": False,
        "success": False,
        "error": {
            "code": code,
            "message": error_message,
        },
        "profile": profile_id,
        "profile_id": profile_id,
        "analyzer": ANALYZER_VERSION,
        "source": None,
        "state": {},
        "regions": {},
        "insights": [],
        "vision": error_vision_payload(profile_id, code, error_message),
        "diagnostics": {"warnings": [], "ocr": {"status": "skipped"}},
    }


def _default_error_message(code: str, profile_id: str) -> str:
    if code == "unknown_profile":
        return f"profile is not registered: {profile_id}"
    if code == "unsupported_profile":
        return f"profile is not supported by offline image analysis: {profile_id}"
    if code == "image_not_found":
        return "image file was not found"
    if code == "image_decode_failed":
        return "image file could not be decoded"
    return code


def _json_safe_bboxes(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_safe_bboxes(item) for key, item in value.items()}
    return value


def _flat_ocr_regions(regions: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in regions.items()
        if key in {"gold", "level", "stage", "round", "augments"} and isinstance(value, tuple)
    }


def _layout_hint_from_source_context(source_context: Mapping[str, Any] | None) -> str | None:
    if not isinstance(source_context, Mapping):
        return None
    expected_layout = str(source_context.get("expected_layout") or "").strip().lower()
    if not expected_layout:
        return None
    try:
        return layout_profile(expected_layout).key
    except KeyError:
        return None


def _apply_ocr_state(state: dict[str, Any], parsed: dict[str, Any]) -> None:
    for key in ("gold", "level", "stage", "round", "augments"):
        value = parsed.get(key)
        if value is not None:
            state[key] = value
    if state.get("stage") is None and state.get("round"):
        state["stage"] = state["round"]


def _shop_units_from_recognition(recognition_result: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for result in recognition_result.get("results", []) or []:
        unit_id = str(result.get("id") or "unknown")
        units.append(
            {
                "unit_id": unit_id,
                "star": None,
                "items": [],
                "traits": [],
                "confidence": float(result.get("confidence") or 0.0),
                "slot_id": result.get("slot_id"),
                "status": result.get("status"),
            }
        )
    return units


def _serialize_tft_state_analysis(analyzed_state: Any) -> dict[str, Any]:
    return {
        "state": {
            "stage": analyzed_state.stage,
            "level": analyzed_state.level,
            "gold": analyzed_state.gold,
            "board_units": [_serialize_unit(unit) for unit in analyzed_state.board_units],
            "bench_units": [_serialize_unit(unit) for unit in analyzed_state.bench_units],
            "shop_units": [_serialize_unit(unit) for unit in analyzed_state.shop_units],
            "items": list(analyzed_state.items),
        },
        "active_traits": [
            {
                "trait": trait.trait,
                "trait_id": trait.trait,
                "count": trait.count,
                "active_tier": trait.active_tier,
                "next_tier": trait.next_tier,
                "unit_names": list(trait.unit_names),
            }
            for trait in analyzed_state.active_traits
        ],
        "trait_gaps": [
            {
                "trait": gap.trait,
                "trait_id": gap.trait,
                "current": gap.current_count,
                "current_count": gap.current_count,
                "target": gap.target_count,
                "target_count": gap.target_count,
                "gap": gap.units_needed,
                "needed": gap.units_needed,
                "candidate_unit_names": list(gap.candidate_unit_names),
                "source_locations": list(gap.source_locations),
            }
            for gap in analyzed_state.trait_gaps
        ],
        "item_direction": _serialize_item_direction(analyzed_state.item_biases),
        "item_biases": [
            {
                "bias": bias.bias,
                "score": bias.score,
                "components": list(bias.components),
                "reasons": list(bias.reasons),
            }
            for bias in analyzed_state.item_biases
        ],
        "pairs": [
            {
                "unit": opportunity.champion_name,
                "unit_id": opportunity.champion_name,
                "count": opportunity.owned_copies,
                "copies": opportunity.owned_copies,
                "needed_copies": opportunity.needed_copies,
                "kind": opportunity.kind,
            }
            for opportunity in analyzed_state.upgrade_opportunities
            if opportunity.kind in {"pair", "upgrade"}
        ],
        "upgrade_opportunities": [
            {
                "champion_name": opportunity.champion_name,
                "owned_copies": opportunity.owned_copies,
                "needed_copies": opportunity.needed_copies,
                "target_stars": opportunity.target_stars,
                "kind": opportunity.kind,
                "source_counts": list(opportunity.source_counts),
            }
            for opportunity in analyzed_state.upgrade_opportunities
        ],
    }


def _serialize_unit(unit: Any) -> dict[str, Any]:
    return {
        "name": unit.name,
        "unit_id": unit.name,
        "star": unit.stars,
        "stars": unit.stars,
        "items": list(unit.items),
        "traits": list(unit.traits),
        "location": unit.location,
        "count": unit.count,
    }


def _serialize_item_direction(item_biases: tuple[Any, ...]) -> dict[str, Any]:
    if not item_biases:
        return {"direction": None, "scores": {}}
    return {
        "direction": item_biases[0].bias,
        "scores": {bias.bias: bias.score for bias in item_biases},
    }


def _build_tft_vision_payload(
    *,
    profile_id: str,
    source: dict[str, Any],
    image_path: str | Path,
    state: dict[str, Any],
    regions: dict[str, Any],
    insights: list[dict[str, Any]],
    warnings: list[dict[str, str]],
    ocr_result: dict[str, Any],
    recognition_result: dict[str, Any],
    state_analysis: dict[str, Any],
    local_vision_result: dict[str, Any] | None = None,
    layout_hint: str | None = None,
    vlm_requested: bool = False,
    state_changed: bool = False,
) -> dict[str, Any]:
    confidence = _tft_analysis_confidence(warnings, ocr_result, recognition_result)
    quality = inspect_image_quality(
        image_path,
        width=int(source.get("width") or 0),
        height=int(source.get("height") or 0),
    )
    diagnostics = {
        "warnings": list(warnings),
        "analyzers": {
            "ocr": _analyzer_summary(ocr_result),
            "template_matcher": _analyzer_summary(recognition_result),
            "classifier": _local_analyzer_status(local_vision_result, "classifier"),
            "detector": _local_analyzer_status(local_vision_result, "detector"),
            "vlm": {"status": "skipped", "reason": "not_requested"},
        },
        "analysis": state_analysis,
        "local_vision": local_vision_result or {},
        "quality": quality,
    }
    if layout_hint:
        diagnostics["layout_hint"] = layout_hint
    vision = VisionFrameAnalysis(
        profile_id=profile_id,
        source=source,
        frame=build_frame_metadata(
            image_path=image_path,
            width=int(source.get("width") or 0),
            height=int(source.get("height") or 0),
            quality=quality,
        ),
        scene={"label": f"tft_{layout_hint}" if layout_hint else "tft_unknown", "confidence": confidence, "source": ANALYZER_VERSION},
        ui=_ui_regions_from_bboxes(regions),
        game_state=state,
        insights=insights,
        confidence=confidence,
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
            prepare_vlm_input(vision, image_path, plan=vlm_plan)
        )
    return vision


def _analyzer_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status", "unknown"),
        "available": bool(result.get("available")),
    }


def _local_analyzer_status(result: dict[str, Any] | None, key: str) -> dict[str, Any]:
    diagnostics = result.get("diagnostics") if isinstance(result, dict) else {}
    status = diagnostics.get(key) if isinstance(diagnostics, dict) else None
    if isinstance(status, dict):
        return dict(status)
    return {"status": "skipped", "reason": "not_configured"}


def _tft_analysis_confidence(
    warnings: list[dict[str, str]],
    ocr_result: dict[str, Any],
    recognition_result: dict[str, Any],
) -> float:
    confidence = 0.55
    warning_codes = {warning.get("code") for warning in warnings}
    if "unsupported_aspect_ratio" in warning_codes:
        return 0.2
    if ocr_result.get("status") in {"ready", "skipped"}:
        confidence += 0.1
    if recognition_result.get("status") == "ready":
        confidence += 0.15
    return min(confidence, 0.9)


def _ui_regions_from_bboxes(regions: dict[str, Any]) -> list[dict[str, Any]]:
    ui: list[dict[str, Any]] = []
    for key, value in regions.items():
        if isinstance(value, list) and len(value) == 4:
            ui.append({"type": "region", "label": key, "bbox": value, "confidence": 1.0})
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                if isinstance(child_value, list) and len(child_value) == 4:
                    ui.append(
                        {
                            "type": "region",
                            "label": child_key,
                            "group": key,
                            "bbox": child_value,
                            "confidence": 1.0,
                        }
                    )
    return ui
