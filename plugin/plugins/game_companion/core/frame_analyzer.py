from __future__ import annotations

from pathlib import Path
from typing import Any

from .image_source import ImageSourceError, image_data_url_to_temp_file, read_image_metadata
from .profile_registry import ProfileMetadata, ProfileRegistry
from ..profiles.tft.insights import generate_insights
from ..profiles.tft.ocr import analyze_tft_ocr_regions
from ..profiles.tft.recognition import recognize_shop_units
from ..profiles.tft.screen_regions import (
    UnsupportedAspectRatioError,
    grouped_screen_region_bboxes,
    save_debug_crops,
)
from ..profiles.tft.state_parser import analyze_tft_state, parse_tft_state

SUPPORTED_PROFILE_IDS = frozenset({"tft"})
ANALYZER_VERSION = "offline_image_v1"


def analyze_offline_image(
    profile_id: str,
    image_path: str | Path,
    *,
    debug_crops_dir: str | Path | None = None,
) -> dict[str, Any]:
    normalized_profile_id = _normalize_profile_id(profile_id)
    profile = _get_builtin_profile(normalized_profile_id)
    if profile is None:
        return _error_response("unknown_profile", normalized_profile_id)

    if profile.profile_id not in SUPPORTED_PROFILE_IDS:
        return _error_response("unsupported_profile", profile.profile_id)

    try:
        image = read_image_metadata(image_path)
    except ImageSourceError as exc:
        return _error_response(exc.code, profile.profile_id, exc.message)

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
    try:
        raw_regions = grouped_screen_region_bboxes(image.width, image.height)
        ocr_result = analyze_tft_ocr_regions(image.path, _flat_ocr_regions(raw_regions))
        recognition_result = recognize_shop_units(image.path, raw_regions)
        if debug_crops_dir:
            try:
                debug_crops = save_debug_crops(image.path, debug_crops_dir)
            except Exception as exc:
                warnings.append({"code": "debug_crops_failed", "message": str(exc)})
        regions = _json_safe_bboxes(raw_regions)
    except UnsupportedAspectRatioError as exc:
        warnings.append({"code": "unsupported_aspect_ratio", "message": str(exc)})

    state = _empty_tft_state()
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

    return {
        "ok": True,
        "success": True,
        "error": None,
        "profile": profile.profile_id,
        "profile_id": profile.profile_id,
        "analyzer": ANALYZER_VERSION,
        "source": image.to_dict(),
        "state": state,
        "regions": regions,
        "insights": insights,
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
) -> dict[str, Any]:
    return analyze_offline_image(profile_id, image_path, debug_crops_dir=debug_crops_dir)


def analyze_frame_data_url(profile_id: str, image_data_url: str) -> dict[str, Any]:
    temp_path: Path | None = None
    try:
        temp_path = image_data_url_to_temp_file(image_data_url)
        payload = analyze_offline_image(profile_id, temp_path)
        if payload.get("source"):
            payload["source"]["type"] = "image_data_url"
            payload["source"]["path"] = None
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
    }


def _error_response(code: str, profile_id: str, message: str | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "success": False,
        "error": {
            "code": code,
            "message": message or _default_error_message(code, profile_id),
        },
        "profile": profile_id,
        "profile_id": profile_id,
        "analyzer": ANALYZER_VERSION,
        "source": None,
        "state": {},
        "regions": {},
        "insights": [],
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
