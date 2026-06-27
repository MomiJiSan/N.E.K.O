from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
import json
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageStat

from .frame_analyzer import analyze_frame
from ..profiles.tft.screen_regions import LAYOUT_STATES, grouped_screen_region_metadata

MANIFEST_SCHEMA_VERSION = 1
CROP_ACCEPTANCE_THRESHOLD = 0.9
ANNOTATION_STATUSES = ("unchecked", "pass", "fail", "needs_adjustment")
CRITICAL_CALIBRATION_CHECKS = (
    "stage_round_clean",
    "gold_clean",
    "level_exp_clean",
    "shop_slots_complete",
)
RECOMMENDED_SAMPLE_TAGS = (
    "shop_open",
    "shop_five_units",
    "bench_units",
    "traits_panel_expanded",
    "items_visible",
)
SUPPORTED_SCREENSHOT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
DEFAULT_LOCAL_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / ".local_calibration"

CALIBRATION_CHECKS = (
    ("shop_slots_complete", "All five shop slots include the intended card area."),
    ("gold_clean", "Gold crop is clean enough for OCR."),
    ("level_exp_clean", "Level and XP crop is clean enough for OCR."),
    ("stage_round_clean", "Stage / round crop is centered and readable."),
    ("bench_complete", "Bench crop covers all bench positions without swallowing shop cards."),
    ("traits_panel_aligned", "Traits panel crop is aligned and not shifted into the board."),
    ("items_area_reasonable", "Items/equipment crop captures loose items without overfitting board skin."),
)

CALIBRATION_CHECK_REGIONS = {
    "shop_slots_complete": ("shop_slot_1", "shop_slot_2", "shop_slot_3", "shop_slot_4", "shop_slot_5"),
    "gold_clean": ("gold",),
    "level_exp_clean": ("level", "level_exp"),
    "stage_round_clean": ("stage", "round"),
    "bench_complete": ("bench",),
    "traits_panel_aligned": ("traits_panel",),
    "items_area_reasonable": ("equipment", "items_area"),
}


def init_tft_layout_calibration_workspace(
    root_dir: str | Path | None = None,
    *,
    overwrite_manifest: bool = False,
) -> dict[str, Any]:
    root_path = Path(root_dir).expanduser() if root_dir else DEFAULT_LOCAL_CALIBRATION_DIR
    input_dir = root_path / "input"
    output_dir = root_path / "output"
    reports_dir = root_path / "reports"
    for directory in (input_dir, output_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = root_path / "samples_manifest.json"
    if overwrite_manifest or not manifest_path.exists():
        manifest = build_tft_layout_sample_manifest(input_dir, manifest_path)
    else:
        manifest = _manifest_status(manifest_path)
    readme_path = root_path / "README.md"
    if overwrite_manifest or not readme_path.exists():
        readme_path.write_text(_workspace_readme(input_dir, manifest_path, output_dir), encoding="utf-8")
    return {
        "type": "tft_layout_calibration_workspace",
        "root_dir": str(root_path.resolve()),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "reports_dir": str(reports_dir.resolve()),
        "samples_manifest_path": str(manifest_path.resolve()),
        "readme_path": str(readme_path.resolve()),
        "manifest": manifest,
        "next_steps": [
            "Put 5-10 real TFT screenshots in input_dir.",
            "Run game_companion_prepare_layout_calibration_manifest or edit samples_manifest.json.",
            "Run game_companion_calibrate_layout with samples_manifest_path and output_dir.",
        ],
    }


def capture_tft_layout_calibration_screenshot(
    output_dir: str | Path | None = None,
    *,
    label: str | None = None,
    expected_layout: str | None = None,
    tags: Iterable[str] | None = None,
    image_grabber: Any | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser() if output_dir else DEFAULT_LOCAL_CALIBRATION_DIR / "input"
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_layout = _normalize_expected_layout(expected_layout)
    normalized_tags = _normalize_tags(tags)
    filename = _capture_filename(label=label, expected_layout=normalized_layout, tags=normalized_tags)
    image_path = output_path / filename
    grabber = image_grabber or _grab_primary_screen
    image = grabber()
    if not isinstance(image, Image.Image):
        raise OSError("screen capture did not return a PIL image")
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(image_path)

    metadata = _image_manifest_metadata(image_path)
    diagnostics = _capture_image_diagnostics(image)
    warnings = []
    if not _is_supported_16_9(metadata.get("width"), metadata.get("height")):
        warnings.append(
            {
                "code": "unsupported_aspect_ratio",
                "message": f"TFT calibration expects 16:9 screenshots; got {metadata.get('width')}x{metadata.get('height')}.",
            }
        )
    if diagnostics["looks_black"]:
        warnings.append(
            {
                "code": "possible_black_capture",
                "message": "Captured image is almost black. Switch TFT to borderless/windowed mode and capture again.",
            }
        )
    return {
        "type": "tft_layout_calibration_capture",
        "image_path": str(image_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "expected_layout": normalized_layout,
        "tags": normalized_tags,
        "label": str(label or image_path.stem),
        "metadata": metadata,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "next_steps": [
            "Inspect the captured image before using it for calibration.",
            "Run game_companion_prepare_layout_calibration_manifest after collecting 5-10 samples.",
        ],
    }


def build_tft_layout_sample_manifest(
    input_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_dir).expanduser()
    if not input_path.is_dir():
        raise FileNotFoundError(f"calibration input directory was not found: {input_path}")
    screenshots = [
        path
        for path in sorted(input_path.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SCREENSHOT_EXTENSIONS
    ]
    samples = [_sample_from_path(path, input_path) for path in screenshots]
    warnings = []
    if not samples:
        warnings.append({"code": "no_screenshots_found", "message": "no supported screenshot files were found"})
    manifest: dict[str, Any] = {
        "type": "tft_layout_calibration_samples_manifest",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": time.time(),
        "input_dir": str(input_path.resolve()),
        "base_dir": str(input_path.resolve()),
        "supported_extensions": list(SUPPORTED_SCREENSHOT_EXTENSIONS),
        "layout_states": list(LAYOUT_STATES),
        "recommended_sample_tags": list(RECOMMENDED_SAMPLE_TAGS),
        "warnings": warnings,
        "samples": samples,
        "summary": _manifest_summary(samples),
    }
    if output_path is not None:
        manifest_path = Path(output_path).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest["manifest_path"] = str(manifest_path.resolve())
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_tft_layout_sample_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("invalid calibration samples manifest: expected a JSON object")
    if data.get("type") != "tft_layout_calibration_samples_manifest":
        raise ValueError("invalid calibration samples manifest: unexpected type")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"invalid calibration samples manifest schema version: {data.get('schema_version')!r}")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("invalid calibration samples manifest: samples must be a list")
    if not samples:
        raise ValueError("empty calibration samples manifest")
    base_dir = Path(str(data.get("base_dir") or path.parent)).expanduser()
    resolved_samples = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise ValueError(f"invalid calibration sample #{index}: expected object")
        if sample.get("include") is False:
            continue
        resolved_samples.append(_resolve_manifest_sample(sample, base_dir, index))
    if not resolved_samples:
        raise ValueError("empty calibration samples manifest")
    return resolved_samples


def build_tft_layout_calibration_status(
    *,
    input_dir: str | Path | None = None,
    samples_manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    input_status = _input_dir_status(input_dir) if input_dir else None
    manifest_status = _manifest_status(samples_manifest_path) if samples_manifest_path else None
    report_status = _report_status(report_path) if report_path else None
    return {
        "type": "tft_layout_calibration_status",
        "input_dir": input_status,
        "samples_manifest": manifest_status,
        "report": report_status,
        "next_steps": _status_next_steps(input_status, manifest_status, report_status),
    }


def build_tft_layout_calibration_report(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    profile_id: str = "tft",
    samples: Iterable[Mapping[str, Any]] | None = None,
    samples_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    image_path_list = list(image_paths)
    if samples_manifest_path is not None and (samples is not None or image_path_list):
        raise ValueError("conflicting sample inputs: use only one of image_paths, samples, or samples_manifest_path")
    if samples is None and samples_manifest_path is not None:
        samples = load_tft_layout_sample_manifest(samples_manifest_path)
    sample_specs = _normalize_sample_specs(image_path_list, samples)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    screenshots: list[dict[str, Any]] = []
    for index, sample in enumerate(sample_specs, start=1):
        image_path = sample["image_path"]
        crop_dir = output_path / f"{index:02d}_{_safe_stem(image_path)}"
        try:
            analysis = analyze_frame(
                profile_id=profile_id,
                image_path=image_path,
                debug_crops_dir=crop_dir,
                debug_crops_layout=sample.get("expected_layout"),
            )
            screenshots.append(_screenshot_payload(index, image_path, crop_dir, analysis, sample))
        except OSError as exc:
            screenshots.append(_screenshot_error_payload(index, image_path, crop_dir, "calibration_io_failed", str(exc), sample))

    report: dict[str, Any] = {
        "type": "tft_layout_calibration_report",
        "profile": profile_id,
        "created_at": time.time(),
        "output_dir": str(output_path.resolve()),
        "required_screenshot_count": {"min": 5, "max": 10},
        "layout_states": list(LAYOUT_STATES),
        "recommended_sample_tags": list(RECOMMENDED_SAMPLE_TAGS),
        "annotation_statuses": list(ANNOTATION_STATUSES),
        "manual_checks": [
            {"id": check_id, "label": label, "status": "unchecked"}
            for check_id, label in CALIBRATION_CHECKS
        ],
        "layout_metadata": _layout_metadata_from_successful_screenshot(screenshots),
        "screenshots": screenshots,
        "summary": _summary(screenshots),
    }
    report["annotation_summary"] = summarize_tft_layout_calibration_report(report)
    report_path = output_path / "calibration_report.json"
    html_path = output_path / "index.html"
    report["report_path"] = str(report_path.resolve())
    report["html_path"] = str(html_path.resolve())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    return report


def update_tft_layout_calibration_check(
    report_path: str | Path,
    *,
    screenshot_index: int,
    check_id: str,
    status: str,
    note: str | None = None,
) -> dict[str, Any]:
    result = update_tft_layout_calibration_checks(
        report_path,
        updates=[
            {
                "screenshot_index": screenshot_index,
                "check_id": check_id,
                "status": status,
                "note": note,
            }
        ],
    )
    updated = result["updates"][0]
    return {
        "updated": True,
        "report_path": result["report_path"],
        "html_path": result["html_path"],
        "screenshot_index": screenshot_index,
        "check": updated["check"],
        "annotation_summary": result["annotation_summary"],
    }


def update_tft_layout_calibration_checks(
    report_path: str | Path,
    *,
    updates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(report_path).expanduser()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("invalid calibration report: expected a JSON object")
    if report.get("type") != "tft_layout_calibration_report":
        raise ValueError("invalid calibration report: unexpected type")
    applied_updates = []
    for index, update in enumerate(updates, start=1):
        if not isinstance(update, Mapping):
            raise ValueError(f"invalid calibration check update #{index}: expected object")
        screenshot_index = int(update.get("screenshot_index") or 0)
        check_id = str(update.get("check_id") or "")
        status = str(update.get("status") or "").strip().lower()
        if status not in ANNOTATION_STATUSES:
            raise ValueError(f"invalid calibration check status: {update.get('status')!r}")
        screenshot = _find_report_screenshot(report, screenshot_index)
        check = _find_report_check(screenshot, check_id)
        check["status"] = status
        if "note" in update:
            check["note"] = str(update.get("note") or "")
        applied_updates.append(
            {
                "screenshot_index": screenshot_index,
                "check_id": check_id,
                "check": dict(check),
            }
        )
    if not applied_updates:
        raise ValueError("calibration check updates must not be empty")
    report["annotation_summary"] = summarize_tft_layout_calibration_report(report)
    report["report_path"] = str(path.resolve())
    html_path = Path(str(report.get("html_path") or path.with_name("index.html"))).expanduser()
    report["html_path"] = str(html_path.resolve())
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    return {
        "updated": True,
        "report_path": str(path.resolve()),
        "html_path": str(html_path.resolve()),
        "updates": applied_updates,
        "annotation_summary": report["annotation_summary"],
    }


def _find_report_screenshot(report: Mapping[str, Any], screenshot_index: int) -> dict[str, Any]:
    screenshots = report.get("screenshots")
    if not isinstance(screenshots, list):
        raise ValueError("invalid calibration report: screenshots must be a list")
    for screenshot in screenshots:
        if isinstance(screenshot, dict) and int(screenshot.get("index") or -1) == screenshot_index:
            return screenshot
    raise ValueError(f"calibration screenshot was not found: {screenshot_index}")


def _find_report_check(screenshot: Mapping[str, Any], check_id: str) -> dict[str, Any]:
    normalized = str(check_id or "").strip()
    checks = screenshot.get("manual_checks")
    if not isinstance(checks, list):
        raise ValueError("invalid calibration report: manual_checks must be a list")
    for check in checks:
        if isinstance(check, dict) and check.get("id") == normalized:
            return check
    raise ValueError(f"calibration check was not found: {normalized}")


def _sample_from_path(path: Path, base_dir: Path) -> dict[str, Any]:
    expected_layout = _infer_expected_layout(path.stem)
    tags = _infer_tags(path.stem)
    metadata = _image_manifest_metadata(path)
    relative_path = str(path.resolve().relative_to(base_dir.resolve()))
    return {
        "id": path.stem,
        "image_path": str(path.resolve()),
        "relative_path": relative_path,
        "file_size": path.stat().st_size,
        **metadata,
        "expected_layout": expected_layout,
        "tags": tags,
        "label": path.stem,
        "note": "",
        "include": True,
        "skip_reason": "",
        "needs_manual_label": expected_layout is None or not tags,
    }


def _input_dir_status(input_dir: str | Path | None) -> dict[str, Any]:
    input_path = Path(input_dir or "").expanduser()
    if not input_path.is_dir():
        return {
            "exists": False,
            "path": str(input_path),
            "error": {"code": "input_dir_not_found", "message": f"calibration input directory was not found: {input_path}"},
        }
    manifest = build_tft_layout_sample_manifest(input_path)
    return {
        "exists": True,
        "path": str(input_path.resolve()),
        "summary": manifest["summary"],
        "warnings": manifest.get("warnings", []),
        "sample_count": len(manifest.get("samples", [])),
    }


def _workspace_readme(input_dir: Path, manifest_path: Path, output_dir: Path) -> str:
    return f"""# TFT Layout Calibration Workspace

This directory is local-only and should not be committed.

1. Put 5-10 real TFT screenshots in:
   `{input_dir.resolve()}`
2. Generate or edit:
   `{manifest_path.resolve()}`
3. Make sure samples cover:
   normal_shop, combat, augment_select, special
4. Make sure tags cover:
   shop_open, shop_five_units, bench_units, traits_panel_expanded, items_visible
5. Run layout calibration with output_dir:
   `{output_dir.resolve()}`
6. Open `index.html`, review crops, and update checks with:
   game_companion_update_layout_calibration_check
   or game_companion_update_layout_calibration_checks

Do not proceed to OCR or template recognition until the calibration report says
ready_for_recognition is true.
"""


def _manifest_status(samples_manifest_path: str | Path | None) -> dict[str, Any]:
    path = Path(samples_manifest_path or "").expanduser()
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "valid": False,
            "error": {"code": "samples_manifest_not_found", "message": f"samples manifest was not found: {path}"},
        }
    try:
        samples = load_tft_layout_sample_manifest(path)
    except json.JSONDecodeError as exc:
        return {"exists": True, "path": str(path.resolve()), "valid": False, "error": {"code": "samples_manifest_decode_failed", "message": str(exc)}}
    except ValueError as exc:
        return {"exists": True, "path": str(path.resolve()), "valid": False, "error": {"code": "invalid_samples_manifest", "message": str(exc)}}
    summary = _manifest_summary(samples)
    return {
        "exists": True,
        "path": str(path.resolve()),
        "valid": True,
        "summary": summary,
        "sample_count": len(samples),
    }


def _report_status(report_path: str | Path | None) -> dict[str, Any]:
    path = Path(report_path or "").expanduser()
    if not path.is_file():
        return {
            "exists": False,
            "path": str(path),
            "valid": False,
            "error": {"code": "report_not_found", "message": f"calibration report was not found: {path}"},
        }
    try:
        summary = summarize_tft_layout_calibration_report(path)
    except json.JSONDecodeError as exc:
        return {"exists": True, "path": str(path.resolve()), "valid": False, "error": {"code": "report_decode_failed", "message": str(exc)}}
    except ValueError as exc:
        return {"exists": True, "path": str(path.resolve()), "valid": False, "error": {"code": "invalid_report", "message": str(exc)}}
    return {
        "exists": True,
        "path": str(path.resolve()),
        "valid": True,
        "annotation_summary": summary,
        "ready_for_recognition": bool(summary.get("ready_for_recognition")),
    }


def _status_next_steps(
    input_status: Mapping[str, Any] | None,
    manifest_status: Mapping[str, Any] | None,
    report_status: Mapping[str, Any] | None,
) -> list[str]:
    if report_status and report_status.get("valid"):
        summary = report_status.get("annotation_summary") if isinstance(report_status.get("annotation_summary"), Mapping) else {}
        if summary.get("ready_for_recognition"):
            return ["Layout calibration is ready for phase-5 recognition work."]
        return ["Finish manual checks in calibration_report.json, then run game_companion_summarize_layout_calibration again."]
    if manifest_status and manifest_status.get("valid"):
        summary = manifest_status.get("summary") if isinstance(manifest_status.get("summary"), Mapping) else {}
        file_checks = summary.get("file_checks") if isinstance(summary.get("file_checks"), Mapping) else {}
        if not file_checks.get("all_files_exist", True):
            return ["Fix missing screenshot paths in samples_manifest.json, then check calibration status again."]
        if not file_checks.get("all_images_decodable", True):
            return ["Replace undecodable screenshot files, then check calibration status again."]
        if not summary.get("has_recommended_sample_count"):
            return ["Collect 5-10 TFT screenshots before running layout calibration."]
        if not summary.get("has_layout_state_coverage") or not summary.get("has_recommended_tag_coverage"):
            return ["Edit samples_manifest.json to cover all layout states and recommended tags."]
        return ["Run game_companion_calibrate_layout with samples_manifest_path."]
    if input_status and input_status.get("exists"):
        if not input_status.get("sample_count"):
            return ["Add TFT screenshots to the input directory, then prepare the manifest again."]
        return ["Review or write samples_manifest.json, then run game_companion_calibrate_layout."]
    return ["Provide input_dir, samples_manifest_path, or report_path for calibration status."]


def _image_manifest_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError as exc:
        return {
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "metadata_error": str(exc),
        }
    return {
        "width": width,
        "height": height,
        "aspect_ratio": f"{width}:{height}",
        "metadata_error": None,
    }


def _grab_primary_screen() -> Image.Image:
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise OSError("Pillow ImageGrab is not available in this runtime") from exc
    try:
        return ImageGrab.grab(all_screens=False)
    except TypeError:
        return ImageGrab.grab()


def _capture_filename(
    *,
    label: str | None,
    expected_layout: str | None,
    tags: Iterable[str],
) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1) * 1000)
    parts = ["tft", expected_layout or "sample"]
    safe_label = _safe_token(label)
    if safe_label:
        parts.append(safe_label)
    for tag in tags:
        safe_tag = _safe_token(tag)
        if safe_tag and safe_tag not in parts:
            parts.append(safe_tag)
    parts.append(f"{timestamp}_{millis:03d}")
    return f"{'_'.join(parts)}.png"


def _safe_token(value: Any) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value or ""))
    return "_".join(token for token in normalized.split("_") if token)


def _capture_image_diagnostics(image: Image.Image) -> dict[str, Any]:
    probe = image.convert("L").resize((64, 36))
    stat = ImageStat.Stat(probe)
    extrema = probe.getextrema()
    mean_luma = float(stat.mean[0])
    return {
        "width": image.width,
        "height": image.height,
        "mean_luma": round(mean_luma, 2),
        "luma_extrema": [int(extrema[0]), int(extrema[1])],
        "looks_black": mean_luma < 3.0 or extrema[1] < 8,
    }


def _is_supported_16_9(width: Any, height: Any) -> bool:
    try:
        width_int = int(width)
        height_int = int(height)
    except (TypeError, ValueError):
        return False
    return width_int > 0 and height_int > 0 and width_int * 9 == height_int * 16


def _resolve_manifest_sample(sample: Mapping[str, Any], base_dir: Path, index: int) -> dict[str, Any]:
    image_path_value = sample.get("image_path")
    relative_path_value = sample.get("relative_path")
    if image_path_value:
        image_path = Path(str(image_path_value)).expanduser()
        if not image_path.is_absolute():
            image_path = base_dir / image_path
    elif relative_path_value:
        image_path = base_dir / str(relative_path_value)
    else:
        raise ValueError(f"invalid calibration sample #{index}: image_path or relative_path is required")
    if image_path.suffix.lower() not in SUPPORTED_SCREENSHOT_EXTENSIONS:
        raise ValueError(f"unsupported calibration sample extension for {image_path}")
    expected_layout = _normalize_expected_layout(sample.get("expected_layout") or sample.get("layout"))
    return {
        "id": str(sample.get("id") or image_path.stem),
        "image_path": image_path.expanduser(),
        "relative_path": str(relative_path_value or image_path.name),
        "file_size": sample.get("file_size"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "aspect_ratio": sample.get("aspect_ratio"),
        "expected_layout": expected_layout,
        "tags": _normalize_tags(sample.get("tags")),
        "label": str(sample.get("label") or image_path.stem),
        "note": str(sample.get("note") or ""),
        "include": True,
        "skip_reason": str(sample.get("skip_reason") or ""),
        "needs_manual_label": expected_layout is None or not _normalize_tags(sample.get("tags")),
    }


def _infer_expected_layout(name: str) -> str | None:
    tokens = _name_tokens(name)
    joined = "_".join(tokens)
    if "augment_select" in joined or "augment" in tokens or "hex" in tokens or "rune" in tokens:
        return "augment_select"
    if "combat" in tokens or "fight" in tokens or "battle" in tokens:
        return "combat"
    if "special" in tokens or "carousel" in tokens or "pve" in tokens or "reward" in tokens or "encounter" in tokens:
        return "special"
    if "normal_shop" in joined or "shop" in tokens or "prepare" in tokens or "planning" in tokens:
        return "normal_shop"
    return None


def _infer_tags(name: str) -> list[str]:
    tokens = _name_tokens(name)
    joined = "_".join(tokens)
    tags = []
    if "shop" in tokens:
        tags.append("shop_open")
    if "five" in tokens or "5" in tokens or "fullshop" in tokens or "shop5" in joined:
        tags.append("shop_five_units")
    if "bench" in tokens:
        tags.append("bench_units")
    if ("traits" in tokens or "trait" in tokens or "synergy" in tokens) and ("expanded" in tokens or "open" in tokens):
        tags.append("traits_panel_expanded")
    if "items" in tokens or "item" in tokens or "equipment" in tokens:
        tags.append("items_visible")
    return tags


def _name_tokens(name: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in name)
    return [token for token in normalized.split("_") if token]


def _manifest_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    screenshots = [
        {
            "expected_layout": sample.get("expected_layout"),
            "tags": sample.get("tags") if isinstance(sample.get("tags"), list) else [],
        }
        for sample in samples
    ]
    coverage = _coverage_summary(screenshots)
    file_checks = _sample_file_checks(samples)
    return {
        "total": len(samples),
        "coverage": coverage,
        "file_checks": file_checks,
        "has_recommended_sample_count": 5 <= len(samples) <= 10,
        "has_layout_state_coverage": not coverage["missing_layouts"],
        "has_recommended_tag_coverage": not coverage["missing_tags"],
        "needs_manual_labels": [
            sample["image_path"]
            for sample in samples
            if not sample.get("expected_layout") or not sample.get("tags")
        ],
    }


def _sample_file_checks(samples: list[dict[str, Any]]) -> dict[str, Any]:
    missing_files = []
    decode_errors = []
    for sample in samples:
        image_path = Path(str(sample.get("image_path") or "")).expanduser()
        if not image_path.is_file():
            missing_files.append(str(image_path))
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except OSError as exc:
            decode_errors.append({"image_path": str(image_path), "message": str(exc)})
    return {
        "all_files_exist": not missing_files,
        "all_images_decodable": not decode_errors,
        "missing_files": missing_files,
        "decode_errors": decode_errors,
    }


def _normalize_sample_specs(
    image_paths: Iterable[str | Path],
    samples: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if samples is None:
        return [
            {
                "image_path": Path(path).expanduser(),
                "relative_path": "",
                "expected_layout": None,
                "tags": [],
                "label": "",
                "note": "",
                "include": True,
                "needs_manual_label": True,
            }
            for path in image_paths
        ]
    specs: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, Mapping):
            raise ValueError(f"invalid calibration sample #{index}: expected object")
        if sample.get("include") is False:
            continue
        image_path = sample.get("image_path") or sample.get("path")
        if not image_path:
            raise ValueError(f"invalid calibration sample #{index}: image_path is required")
        if Path(str(image_path)).suffix.lower() not in SUPPORTED_SCREENSHOT_EXTENSIONS:
            raise ValueError(f"unsupported calibration sample extension for {image_path}")
        expected_layout = _normalize_expected_layout(sample.get("expected_layout") or sample.get("layout"))
        tags = _normalize_tags(sample.get("tags"))
        specs.append(
            {
                "id": str(sample.get("id") or Path(str(image_path)).stem),
                "image_path": Path(str(image_path)).expanduser(),
                "relative_path": str(sample.get("relative_path") or ""),
                "file_size": sample.get("file_size"),
                "width": sample.get("width"),
                "height": sample.get("height"),
                "aspect_ratio": sample.get("aspect_ratio"),
                "expected_layout": expected_layout,
                "tags": tags,
                "label": str(sample.get("label") or ""),
                "note": str(sample.get("note") or ""),
                "include": True,
                "skip_reason": str(sample.get("skip_reason") or ""),
                "needs_manual_label": expected_layout is None or not tags,
            }
        )
    if not specs:
        raise ValueError("empty calibration samples")
    return specs


def _normalize_expected_layout(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in LAYOUT_STATES else None


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = [value]
    elif isinstance(value, Iterable):
        raw_tags = list(value)
    else:
        raw_tags = []
    tags = []
    for tag in raw_tags:
        normalized = str(tag).strip().lower()
        if normalized:
            tags.append(normalized)
    return list(dict.fromkeys(tags))


def summarize_tft_layout_calibration_report(report_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    report = _load_report(report_or_path)
    screenshots = report.get("screenshots") if isinstance(report.get("screenshots"), list) else []
    per_check = {
        check_id: {
            "id": check_id,
            "label": label,
            "regions": list(CALIBRATION_CHECK_REGIONS.get(check_id, ())),
            "statuses": {status: 0 for status in ANNOTATION_STATUSES},
            "notes": [],
        }
        for check_id, label in CALIBRATION_CHECKS
    }
    unknown_statuses: dict[str, int] = {}
    failed_regions: dict[str, list[str]] = {}

    for screenshot in screenshots:
        if not isinstance(screenshot, Mapping):
            continue
        label = _screenshot_label(screenshot)
        for check in _manual_checks_for_screenshot(screenshot):
            check_id = str(check.get("id") or "").strip()
            if check_id not in per_check:
                continue
            status = str(check.get("status") or "unchecked").strip().lower()
            if status not in ANNOTATION_STATUSES:
                unknown_statuses[status] = unknown_statuses.get(status, 0) + 1
                status = "unchecked"
            per_check[check_id]["statuses"][status] += 1
            note = str(check.get("note") or "").strip()
            if note:
                per_check[check_id]["notes"].append({"screenshot": label, "note": note})
            if status in {"fail", "needs_adjustment"}:
                for region in CALIBRATION_CHECK_REGIONS.get(check_id, ()):
                    failed_regions.setdefault(region, []).append(check_id)

    check_summaries = list(per_check.values())
    total_checks = sum(sum(item["statuses"].values()) for item in check_summaries)
    pass_count = sum(item["statuses"]["pass"] for item in check_summaries)
    fail_count = sum(item["statuses"]["fail"] for item in check_summaries)
    adjustment_count = sum(item["statuses"]["needs_adjustment"] for item in check_summaries)
    unchecked_count = sum(item["statuses"]["unchecked"] for item in check_summaries)
    successful_screenshots = sum(1 for screenshot in screenshots if isinstance(screenshot, Mapping) and screenshot.get("success"))
    calibration_ready_screenshots = sum(
        1 for screenshot in screenshots if isinstance(screenshot, Mapping) and screenshot.get("calibration_ready")
    )
    coverage = _coverage_summary([screenshot for screenshot in screenshots if isinstance(screenshot, dict)])
    has_recommended_sample_count = 5 <= len(screenshots) <= 10
    has_layout_state_coverage = not coverage["missing_layouts"]
    has_recommended_tag_coverage = not coverage["missing_tags"]
    crop_acceptance = _crop_acceptance_summary(
        screenshots=[screenshot for screenshot in screenshots if isinstance(screenshot, Mapping)],
        total_checks=total_checks,
        pass_count=pass_count,
        fail_count=fail_count,
        adjustment_count=adjustment_count,
        unchecked_count=unchecked_count,
    )
    layout_acceptance = _layout_acceptance_summary([screenshot for screenshot in screenshots if isinstance(screenshot, Mapping)])
    critical_acceptance = _critical_acceptance_summary([screenshot for screenshot in screenshots if isinstance(screenshot, Mapping)])
    ready_for_region_tuning = bool(fail_count or adjustment_count)
    ready_for_recognition = bool(
        screenshots
        and has_recommended_sample_count
        and calibration_ready_screenshots == len(screenshots)
        and has_layout_state_coverage
        and has_recommended_tag_coverage
        and total_checks > 0
        and crop_acceptance["meets_acceptance"]
        and layout_acceptance["meets_acceptance"]
        and critical_acceptance["meets_acceptance"]
    )
    return {
        "type": "tft_layout_calibration_annotation_summary",
        "screenshots": len(screenshots),
        "successful_screenshots": successful_screenshots,
        "calibration_ready_screenshots": calibration_ready_screenshots,
        "has_recommended_sample_count": has_recommended_sample_count,
        "coverage": coverage,
        "has_layout_state_coverage": has_layout_state_coverage,
        "has_recommended_tag_coverage": has_recommended_tag_coverage,
        "total_checks": total_checks,
        "status_counts": {
            "pass": pass_count,
            "fail": fail_count,
            "needs_adjustment": adjustment_count,
            "unchecked": unchecked_count,
        },
        "crop_acceptance": crop_acceptance,
        "layout_acceptance": layout_acceptance,
        "critical_acceptance": critical_acceptance,
        "unknown_statuses": unknown_statuses,
        "per_check": check_summaries,
        "failed_regions": [
            {"region": region, "checks": sorted(set(checks)), "count": len(checks)}
            for region, checks in sorted(failed_regions.items())
        ],
        "ready_for_region_tuning": ready_for_region_tuning,
        "ready_for_recognition": ready_for_recognition,
    }


def _layout_acceptance_summary(screenshots: list[Mapping[str, Any]]) -> dict[str, Any]:
    layouts: dict[str, dict[str, Any]] = {
        layout: {"total_checks": 0, "passes": 0, "unchecked": 0, "undocumented_misses": []}
        for layout in LAYOUT_STATES
    }
    unknown = {"total_checks": 0, "passes": 0, "unchecked": 0, "undocumented_misses": []}
    for screenshot in screenshots:
        layout = screenshot.get("expected_layout")
        bucket = layouts.get(str(layout), unknown)
        label = _screenshot_label(screenshot)
        for check in _manual_checks_for_screenshot(screenshot):
            status = str(check.get("status") or "unchecked").strip().lower()
            bucket["total_checks"] += 1
            if status == "pass":
                bucket["passes"] += 1
            elif status == "unchecked":
                bucket["unchecked"] += 1
            elif status in {"fail", "needs_adjustment"} and not str(check.get("note") or "").strip():
                bucket["undocumented_misses"].append(
                    {"screenshot": label, "check_id": str(check.get("id") or ""), "status": status}
                )
    layout_results = {
        layout: _acceptance_bucket(layouts[layout])
        for layout in LAYOUT_STATES
    }
    missing_or_failed = [
        layout
        for layout, result in layout_results.items()
        if result["total_checks"] == 0 or not result["meets_acceptance"]
    ]
    return {
        "threshold": CROP_ACCEPTANCE_THRESHOLD,
        "layouts": layout_results,
        "unknown": _acceptance_bucket(unknown),
        "missing_or_failed_layouts": missing_or_failed,
        "meets_acceptance": not missing_or_failed,
    }


def _critical_acceptance_summary(screenshots: list[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {
        check_id: {"passes": 0, "misses": [], "unchecked": 0}
        for check_id in CRITICAL_CALIBRATION_CHECKS
    }
    for screenshot in screenshots:
        label = _screenshot_label(screenshot)
        for check in _manual_checks_for_screenshot(screenshot):
            check_id = str(check.get("id") or "")
            if check_id not in statuses:
                continue
            status = str(check.get("status") or "unchecked").strip().lower()
            if status == "pass":
                statuses[check_id]["passes"] += 1
            elif status == "unchecked":
                statuses[check_id]["unchecked"] += 1
            elif status in {"fail", "needs_adjustment"}:
                statuses[check_id]["misses"].append(
                    {"screenshot": label, "status": status, "note": str(check.get("note") or "").strip()}
                )
    blocking_checks = [
        check_id
        for check_id, result in statuses.items()
        if result["unchecked"] or result["misses"] or result["passes"] == 0
    ]
    return {
        "critical_checks": list(CRITICAL_CALIBRATION_CHECKS),
        "checks": statuses,
        "blocking_checks": blocking_checks,
        "meets_acceptance": not blocking_checks,
    }


def _acceptance_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    total_checks = int(bucket.get("total_checks") or 0)
    passes = int(bucket.get("passes") or 0)
    unchecked = int(bucket.get("unchecked") or 0)
    pass_rate = passes / total_checks if total_checks else 0.0
    undocumented_misses = list(bucket.get("undocumented_misses") or [])
    return {
        "total_checks": total_checks,
        "passes": passes,
        "unchecked": unchecked,
        "pass_rate": pass_rate,
        "pass_percent": round(pass_rate * 100, 2),
        "misses_documented": not undocumented_misses,
        "undocumented_misses": undocumented_misses,
        "meets_acceptance": bool(
            total_checks > 0
            and unchecked == 0
            and pass_rate >= CROP_ACCEPTANCE_THRESHOLD
            and not undocumented_misses
        ),
    }


def _crop_acceptance_summary(
    *,
    screenshots: list[Mapping[str, Any]],
    total_checks: int,
    pass_count: int,
    fail_count: int,
    adjustment_count: int,
    unchecked_count: int,
) -> dict[str, Any]:
    pass_rate = pass_count / total_checks if total_checks else 0.0
    undocumented_misses = _undocumented_misses(screenshots)
    misses = fail_count + adjustment_count
    return {
        "threshold": CROP_ACCEPTANCE_THRESHOLD,
        "pass_rate": pass_rate,
        "pass_percent": round(pass_rate * 100, 2),
        "passes": pass_count,
        "misses": misses,
        "unchecked": unchecked_count,
        "review_complete": total_checks > 0 and unchecked_count == 0,
        "misses_documented": not undocumented_misses,
        "undocumented_misses": undocumented_misses,
        "meets_acceptance": bool(
            total_checks > 0
            and unchecked_count == 0
            and pass_rate >= CROP_ACCEPTANCE_THRESHOLD
            and not undocumented_misses
        ),
    }


def _undocumented_misses(screenshots: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    misses: list[dict[str, str]] = []
    for screenshot in screenshots:
        label = _screenshot_label(screenshot)
        for check in _manual_checks_for_screenshot(screenshot):
            status = str(check.get("status") or "unchecked").strip().lower()
            if status not in {"fail", "needs_adjustment"}:
                continue
            note = str(check.get("note") or "").strip()
            if note:
                continue
            misses.append(
                {
                    "screenshot": label,
                    "check_id": str(check.get("id") or ""),
                    "status": status,
                }
            )
    return misses


def _load_report(report_or_path: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report_or_path, Mapping):
        return report_or_path
    path = Path(report_or_path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("invalid calibration report: expected a JSON object")
    return data


def _manual_checks_for_screenshot(screenshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    checks = screenshot.get("manual_checks")
    if not isinstance(checks, list):
        return []
    return [check for check in checks if isinstance(check, Mapping)]


def _screenshot_label(screenshot: Mapping[str, Any]) -> str:
    index = screenshot.get("index")
    image_path = screenshot.get("image_path")
    return f"#{index} {image_path}" if image_path else f"#{index}"


def _screenshot_payload(
    index: int,
    image_path: Path,
    crop_dir: Path,
    analysis: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    debug_crops = analysis.get("diagnostics", {}).get("debug_crops") if isinstance(analysis.get("diagnostics"), Mapping) else None
    source = analysis.get("source") if isinstance(analysis.get("source"), Mapping) else {}
    warnings = analysis.get("diagnostics", {}).get("warnings") if isinstance(analysis.get("diagnostics"), Mapping) else []
    readiness = _calibration_readiness(analysis, debug_crops, warnings)
    return {
        "index": index,
        "id": sample.get("id") or "",
        "label": sample.get("label") or "",
        "relative_path": sample.get("relative_path") or "",
        "expected_layout": sample.get("expected_layout"),
        "tags": list(sample.get("tags") or []),
        "note": sample.get("note") or "",
        "file_size": sample.get("file_size"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "aspect_ratio": sample.get("aspect_ratio"),
        "needs_manual_label": bool(sample.get("needs_manual_label")),
        "image_path": str(image_path.resolve()) if image_path.exists() else str(image_path),
        "crop_dir": str(crop_dir.resolve()),
        "success": bool(analysis.get("success")),
        "calibration_ready": readiness["ready"],
        "calibration_error": readiness["error"],
        "error": analysis.get("error"),
        "warnings": warnings if isinstance(warnings, list) else [],
        "source": source,
        "regions": analysis.get("regions") or {},
        "debug_crops": debug_crops or {},
        "manual_checks": [
            {"id": check_id, "label": label, "status": "unchecked", "note": ""}
            for check_id, label in CALIBRATION_CHECKS
        ],
    }


def _screenshot_error_payload(
    index: int,
    image_path: Path,
    crop_dir: Path,
    code: str,
    message: str,
    sample: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample = sample or {}
    return {
        "index": index,
        "id": sample.get("id") or "",
        "label": sample.get("label") or "",
        "relative_path": sample.get("relative_path") or "",
        "expected_layout": sample.get("expected_layout"),
        "tags": list(sample.get("tags") or []),
        "note": sample.get("note") or "",
        "file_size": sample.get("file_size"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "aspect_ratio": sample.get("aspect_ratio"),
        "needs_manual_label": bool(sample.get("needs_manual_label")),
        "image_path": str(image_path.resolve()) if image_path.exists() else str(image_path),
        "crop_dir": str(crop_dir.resolve()),
        "success": False,
        "calibration_ready": False,
        "calibration_error": {"code": code, "message": message},
        "error": {"code": code, "message": message},
        "warnings": [],
        "source": {},
        "regions": {},
        "debug_crops": {},
        "manual_checks": [
            {"id": check_id, "label": label, "status": "unchecked", "note": ""}
            for check_id, label in CALIBRATION_CHECKS
        ],
    }


def _calibration_readiness(
    analysis: Mapping[str, Any],
    debug_crops: Any,
    warnings: Any,
) -> dict[str, Any]:
    if not analysis.get("success"):
        error = analysis.get("error") if isinstance(analysis.get("error"), Mapping) else {}
        return {
            "ready": False,
            "error": {
                "code": str(error.get("code") or "analysis_failed"),
                "message": str(error.get("message") or "frame analysis failed"),
            },
        }
    warning_list = warnings if isinstance(warnings, list) else []
    unsupported = _find_warning(warning_list, "unsupported_aspect_ratio")
    if unsupported:
        return {"ready": False, "error": unsupported}
    crop_failed = _find_warning(warning_list, "debug_crops_failed")
    if crop_failed:
        return {"ready": False, "error": crop_failed}
    if not isinstance(debug_crops, Mapping) or not debug_crops.get("crops"):
        return {
            "ready": False,
            "error": {
                "code": "debug_crops_missing",
                "message": "debug crops were not generated for this screenshot",
            },
        }
    return {"ready": True, "error": None}


def _find_warning(warnings: list[Any], code: str) -> dict[str, str] | None:
    for warning in warnings:
        if isinstance(warning, Mapping) and warning.get("code") == code:
            return {
                "code": str(warning.get("code") or code),
                "message": str(warning.get("message") or code),
            }
    return None


def _layout_metadata_from_successful_screenshot(screenshots: list[dict[str, Any]]) -> dict[str, Any]:
    for screenshot in screenshots:
        source = screenshot.get("source") if isinstance(screenshot.get("source"), Mapping) else {}
        width = source.get("width")
        height = source.get("height")
        if screenshot.get("calibration_ready") and isinstance(width, int) and isinstance(height, int):
            return grouped_screen_region_metadata(width, height)
    return {}


def _summary(screenshots: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(1 for screenshot in screenshots if screenshot.get("success"))
    ready = sum(1 for screenshot in screenshots if screenshot.get("calibration_ready"))
    coverage = _coverage_summary(screenshots)
    return {
        "total": len(screenshots),
        "successes": successes,
        "failures": len(screenshots) - successes,
        "calibration_ready": ready,
        "calibration_blocked": len(screenshots) - ready,
        "ready_for_manual_review": ready > 0,
        "has_recommended_sample_count": 5 <= len(screenshots) <= 10,
        "coverage": coverage,
        "has_layout_state_coverage": not coverage["missing_layouts"],
        "has_recommended_tag_coverage": not coverage["missing_tags"],
    }


def _coverage_summary(screenshots: list[dict[str, Any]]) -> dict[str, Any]:
    layout_counts = {layout: 0 for layout in LAYOUT_STATES}
    tag_counts = {tag: 0 for tag in RECOMMENDED_SAMPLE_TAGS}
    unknown_layouts = 0
    untagged = 0
    for screenshot in screenshots:
        expected_layout = screenshot.get("expected_layout")
        if expected_layout in layout_counts:
            layout_counts[expected_layout] += 1
        else:
            unknown_layouts += 1
        tags = screenshot.get("tags") if isinstance(screenshot.get("tags"), list) else []
        if not tags:
            untagged += 1
        for tag in tags:
            if tag in tag_counts:
                tag_counts[tag] += 1
    return {
        "layout_counts": layout_counts,
        "missing_layouts": [layout for layout, count in layout_counts.items() if count == 0],
        "unknown_layouts": unknown_layouts,
        "tag_counts": tag_counts,
        "missing_tags": [tag for tag, count in tag_counts.items() if count == 0],
        "untagged": untagged,
    }


def _safe_stem(path: Path) -> str:
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem)
    return stem[:64] or "screenshot"


def _render_html(report: Mapping[str, Any]) -> str:
    rows = []
    for screenshot in report.get("screenshots", []):
        if not isinstance(screenshot, Mapping):
            continue
        rows.append(_render_screenshot_section(screenshot))
    checks = "\n".join(
        f"<li><code>{escape(str(check['id']))}</code>: {escape(str(check['label']))}</li>"
        for check in report.get("manual_checks", [])
        if isinstance(check, Mapping)
    )
    statuses = ", ".join(f"<code>{escape(status)}</code>" for status in ANNOTATION_STATUSES)
    summary = _render_annotation_summary(report.get("annotation_summary"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>TFT Layout Calibration</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 system-ui, sans-serif; background: #111318; color: #eef2f7; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    section {{ border: 1px solid #303844; border-radius: 8px; padding: 16px; margin: 16px 0; background: #1a1f28; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 16px; }}
    th, td {{ border: 1px solid #303844; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #202735; color: #dce5f3; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid #303844; border-radius: 6px; padding: 8px; background: #12161d; }}
    img {{ width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #07090c; }}
    figcaption {{ margin-top: 6px; color: #b5bfce; word-break: break-word; }}
    code {{ color: #7dd3bd; }}
    .hint, .muted {{ color: #b5bfce; }}
    .bad {{ color: #ff9aa8; }}
  </style>
</head>
<body>
  <h1>TFT Layout Calibration</h1>
  <p>Output: <code>{escape(str(report.get("output_dir", "")))}</code></p>
  <p class="hint">Edit <code>calibration_report.json</code> screenshot manual checks with one of: {statuses}. Then run <code>game_companion_summarize_layout_calibration</code>.</p>
  {summary}
  <h2>Manual Checks</h2>
  <ul>{checks}</ul>
  {''.join(rows)}
</body>
</html>
"""


def _render_annotation_summary(summary: Any) -> str:
    if not isinstance(summary, Mapping):
        return ""
    counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), Mapping) else {}
    crop_acceptance = summary.get("crop_acceptance") if isinstance(summary.get("crop_acceptance"), Mapping) else {}
    layout_acceptance = summary.get("layout_acceptance") if isinstance(summary.get("layout_acceptance"), Mapping) else {}
    critical_acceptance = summary.get("critical_acceptance") if isinstance(summary.get("critical_acceptance"), Mapping) else {}
    cells = "".join(
        f"<td><code>{escape(status)}</code>: {escape(str(counts.get(status, 0)))}</td>"
        for status in ANNOTATION_STATUSES
    )
    layout_counts = coverage.get("layout_counts") if isinstance(coverage.get("layout_counts"), Mapping) else {}
    tag_counts = coverage.get("tag_counts") if isinstance(coverage.get("tag_counts"), Mapping) else {}
    layout_cells = "".join(
        f"<td><code>{escape(layout)}</code>: {escape(str(layout_counts.get(layout, 0)))}</td>"
        for layout in LAYOUT_STATES
    )
    tag_cells = "".join(
        f"<td><code>{escape(tag)}</code>: {escape(str(tag_counts.get(tag, 0)))}</td>"
        for tag in RECOMMENDED_SAMPLE_TAGS
    )
    failed_regions = summary.get("failed_regions") if isinstance(summary.get("failed_regions"), list) else []
    failed_text = ", ".join(
        str(item.get("region"))
        for item in failed_regions
        if isinstance(item, Mapping) and item.get("region")
    )
    missing_layouts = ", ".join(str(item) for item in coverage.get("missing_layouts", [])) if coverage else ""
    missing_tags = ", ".join(str(item) for item in coverage.get("missing_tags", [])) if coverage else ""
    failed_layouts = ", ".join(str(item) for item in layout_acceptance.get("missing_or_failed_layouts", [])) if layout_acceptance else ""
    critical_blockers = ", ".join(str(item) for item in critical_acceptance.get("blocking_checks", [])) if critical_acceptance else ""
    return f"""<section>
  <h2>Annotation Summary</h2>
  <table><tr>{cells}</tr></table>
  <h3>Sample Coverage</h3>
  <table><tr>{layout_cells}</tr><tr>{tag_cells}</tr></table>
  <p class="muted">Missing layouts: {escape(missing_layouts or "none")}; missing tags: {escape(missing_tags or "none")}</p>
  <p class="muted">Crop acceptance: <code>{escape(str(crop_acceptance.get("pass_percent", 0)))}</code>% pass, threshold <code>{escape(str(crop_acceptance.get("threshold", CROP_ACCEPTANCE_THRESHOLD)))}</code>, complete <code>{escape(str(crop_acceptance.get("review_complete")))}</code>, documented <code>{escape(str(crop_acceptance.get("misses_documented")))}</code></p>
  <p class="muted">Layout acceptance blockers: {escape(failed_layouts or "none")}; critical blockers: {escape(critical_blockers or "none")}</p>
  <p class="muted">Ready for region tuning: <code>{escape(str(summary.get("ready_for_region_tuning")))}</code>; ready for recognition: <code>{escape(str(summary.get("ready_for_recognition")))}</code></p>
  <p class="muted">Failed regions: {escape(failed_text or "none")}</p>
</section>
"""


def _render_screenshot_checks(screenshot: Mapping[str, Any]) -> str:
    checks = _manual_checks_for_screenshot(screenshot)
    if not checks:
        return ""
    rows = []
    for check in checks:
        check_id = str(check.get("id") or "")
        rows.append(
            f"<tr data-check-id=\"{escape(check_id)}\">"
            f"<td><code>{escape(check_id)}</code></td>"
            f"<td>{escape(str(check.get('label') or ''))}</td>"
            f"<td><code>{escape(str(check.get('status') or 'unchecked'))}</code></td>"
            f"<td>{escape(str(check.get('note') or ''))}</td>"
            "</tr>"
        )
    return (
        "<h3>Screenshot Checks</h3>"
        "<table><thead><tr><th>Check</th><th>Label</th><th>Status</th><th>Note</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_screenshot_section(screenshot: Mapping[str, Any]) -> str:
    title = f"#{screenshot.get('index')} {screenshot.get('image_path')}"
    sample_meta = (
        f"<p class=\"muted\">Expected layout: <code>{escape(str(screenshot.get('expected_layout') or 'unknown'))}</code>; "
        f"tags: <code>{escape(', '.join(str(tag) for tag in screenshot.get('tags', [])) or 'none')}</code></p>"
    )
    debug_crops = screenshot.get("debug_crops") if isinstance(screenshot.get("debug_crops"), Mapping) else {}
    metadata = debug_crops.get("metadata") if isinstance(debug_crops.get("metadata"), Mapping) else {}
    figures = []
    for key, meta in sorted(metadata.items(), key=_metadata_sort_key):
        if not isinstance(meta, Mapping):
            continue
        crop_path = str(meta.get("crop_path") or "")
        figures.append(
            "<figure>"
            f"<img src=\"{escape(Path(crop_path).as_uri() if crop_path else '')}\" alt=\"{escape(str(key))}\">"
            f"<figcaption><code>{escape(str(key))}</code><br>"
            f"{escape(str(meta.get('layout', '')))} / p{escape(str(meta.get('priority', '')))} / "
            f"{escape(str(meta.get('purpose', '')))}</figcaption>"
            "</figure>"
        )
    error = screenshot.get("error")
    error_html = f"<p class=\"bad\">{escape(str(error))}</p>" if error else ""
    checks_html = _render_screenshot_checks(screenshot)
    return f"""<section>
  <h2>{escape(title)}</h2>
  {sample_meta}
  {error_html}
  {checks_html}
  <div class="grid">{''.join(figures)}</div>
</section>
"""


def _metadata_sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
    key, meta = item
    if not isinstance(meta, Mapping):
        return (99, str(key))
    try:
        priority = int(meta.get("priority", 99))
    except (TypeError, ValueError):
        priority = 99
    return (priority, str(key))
