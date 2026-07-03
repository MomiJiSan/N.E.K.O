from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

from .calibration import DEFAULT_LOCAL_CALIBRATION_DIR
from .tft_runtime import build_tft_video_state_report

TFT_NORMAL_SHOP_SMOKE_REPORT_VERSION = "tft_normal_shop_smoke_v1"
TFT_SMOKE_SHOP50_FRAMES = tuple(range(16445, 16445 + 50 * 5, 5))
TFT_SMOKE_MIXED33_FRAMES: dict[str, tuple[int, ...]] = {
    "augment_select": tuple(range(8222, 8222 + 11 * 10, 10)),
    "normal_shop": tuple(range(16445, 16445 + 11 * 10, 10)),
    "combat": tuple(range(57557, 57557 + 11 * 10, 10)),
}
TFT_SMOKE_OVERLAY12_FRAMES = tuple(range(8222, 8222 + 12 * 10, 10))

RuntimeReportBuilder = Callable[..., dict[str, Any]]


def build_tft_normal_shop_smoke_report(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    report_builder: RuntimeReportBuilder = build_tft_video_state_report,
) -> dict[str, Any]:
    video = Path(video_path).expanduser()
    output_path = Path(output_dir).expanduser() if output_dir else _default_smoke_output_dir(video)
    output_path.mkdir(parents=True, exist_ok=True)

    normal_report = report_builder(
        video,
        output_dir=output_path / "normal_shop_50",
        max_frames=len(TFT_SMOKE_SHOP50_FRAMES),
        frame_indices=list(TFT_SMOKE_SHOP50_FRAMES),
        expected_layout="normal_shop",
    )
    mixed_report = report_builder(
        video,
        output_dir=output_path / "mixed_33",
        max_frames=len(_mixed_frame_indices()),
        frame_indices=_mixed_frame_indices(),
        frame_layouts=_mixed_frame_layouts(),
    )
    overlay_report = report_builder(
        video,
        output_dir=output_path / "overlay_12",
        max_frames=len(TFT_SMOKE_OVERLAY12_FRAMES),
        frame_indices=list(TFT_SMOKE_OVERLAY12_FRAMES),
        frame_layouts={index: "augment_select" for index in TFT_SMOKE_OVERLAY12_FRAMES},
    )

    normal = _normal_shop_summary(normal_report)
    mixed = _mixed_summary(mixed_report)
    overlay = _overlay_summary(overlay_report)
    failures = _threshold_failures(normal=normal, mixed=mixed, overlay=overlay)
    report_path = output_path / f"{TFT_NORMAL_SHOP_SMOKE_REPORT_VERSION}.json"
    report = {
        "type": "tft_normal_shop_smoke_report",
        "schema_version": 1,
        "report_version": TFT_NORMAL_SHOP_SMOKE_REPORT_VERSION,
        "success": True,
        "video_path": str(video.resolve()),
        "output_dir": str(output_path.resolve()),
        "report_path": str(report_path.resolve()),
        "normal_shop": normal,
        "mixed": mixed,
        "overlay": overlay,
        "pass": not failures,
        "failures": failures,
        "run_reports": {
            "normal_shop": _runtime_report_refs(normal_report),
            "mixed": _runtime_report_refs(mixed_report),
            "overlay": _runtime_report_refs(overlay_report),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _default_smoke_output_dir(video: Path) -> Path:
    return DEFAULT_LOCAL_CALIBRATION_DIR / f"{video.stem}_normal_shop_smoke_v1"


def _mixed_frame_indices() -> list[int]:
    return [frame for frames in TFT_SMOKE_MIXED33_FRAMES.values() for frame in frames]


def _mixed_frame_layouts() -> dict[int, str]:
    return {frame: layout for layout, frames in TFT_SMOKE_MIXED33_FRAMES.items() for frame in frames}


def _normal_shop_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "frame_count": int(report.get("frame_count") or summary.get("total_frames") or 0),
        "ready_rate": _float(summary.get("normal_shop_ready_rate")),
        "cost_coverage": _float(summary.get("cost_coverage")),
        "name_coverage": _float(summary.get("name_coverage")),
        "fallback_ratio": _float(summary.get("fallback_ratio")),
        "main_blockers": dict(summary.get("main_blockers") or {}),
        "shop_cost_source_counts": dict(summary.get("shop_cost_source_counts") or {}),
        "shop_name_source_counts": dict(summary.get("shop_name_source_counts") or {}),
        "fallback_cost_count": int(summary.get("fallback_cost_count") or 0),
        "ocr_cost_count": int(summary.get("ocr_cost_count") or 0),
        "unknown_count": int(summary.get("unknown_count") or 0),
        "contaminated_count": int(summary.get("contaminated_count") or 0),
    }


def _mixed_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "frame_count": int(report.get("frame_count") or summary.get("total_frames") or 0),
        "augment_ready_count": int(summary.get("augment_ready_count") or 0),
        "normal_shop_ready_count": int(summary.get("normal_shop_ready_count") or 0),
        "combat_ready_count": int(summary.get("combat_ready_count") or 0),
        "non_shop_source_slots": _non_shop_source_slots(report),
        "fallback_cost_count": int(summary.get("fallback_cost_count") or 0),
        "ocr_cost_count": int(summary.get("ocr_cost_count") or 0),
        "unknown_count": int(summary.get("unknown_count") or 0),
        "contaminated_count": int(summary.get("contaminated_count") or 0),
    }


def _overlay_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(report)
    return {
        "frame_count": int(report.get("frame_count") or summary.get("total_frames") or 0),
        "contaminated_count": int(summary.get("contaminated_count") or 0),
        "shop_payloads": _shop_payload_count(report),
        "fallback_cost_count": int(summary.get("fallback_cost_count") or 0),
        "unknown_count": int(summary.get("unknown_count") or 0),
        "main_blockers": dict(summary.get("main_blockers") or {}),
    }


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _runtime_report_refs(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary_path": str(report.get("summary_path") or ""),
        "jsonl_path": str(report.get("jsonl_path") or ""),
        "contact_sheet_path": str(report.get("contact_sheet_path") or ""),
    }


def _non_shop_source_slots(report: Mapping[str, Any]) -> int:
    count = 0
    for record in _jsonl_records(report):
        state = _state(record)
        if state.get("layout") == "normal_shop":
            continue
        for slot in _shop_slots(state):
            if slot.get("name_source") or slot.get("cost_source"):
                count += 1
    return count


def _shop_payload_count(report: Mapping[str, Any]) -> int:
    return sum(1 for record in _jsonl_records(report) if isinstance(_state(record).get("shop"), Mapping))


def _jsonl_records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    path_value = report.get("jsonl_path")
    if not path_value:
        return []
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _state(record: Mapping[str, Any]) -> dict[str, Any]:
    state = record.get("state")
    return dict(state) if isinstance(state, Mapping) else {}


def _shop_slots(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    shop = state.get("shop")
    if not isinstance(shop, Mapping):
        return []
    return [dict(slot) for slot in shop.get("slots", []) if isinstance(slot, Mapping)]


def _threshold_failures(
    *,
    normal: Mapping[str, Any],
    mixed: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = [
        ("normal_shop.ready_rate", ">= 0.95", normal.get("ready_rate"), 0.95, _gte),
        ("normal_shop.cost_coverage", ">= 0.9", normal.get("cost_coverage"), 0.9, _gte),
        ("normal_shop.name_coverage", ">= 0.8", normal.get("name_coverage"), 0.8, _gte),
        ("mixed.non_shop_source_slots", "== 0", mixed.get("non_shop_source_slots"), 0, _eq),
        ("overlay.shop_payloads", "== 0", overlay.get("shop_payloads"), 0, _eq),
        ("overlay.fallback_cost_count", "== 0", overlay.get("fallback_cost_count"), 0, _eq),
    ]
    failures = []
    for check, expected, actual, threshold, predicate in checks:
        normalized = _float(actual) if isinstance(threshold, float) else int(actual or 0)
        if not predicate(normalized, threshold):
            failures.append({"check": check, "expected": expected, "actual": normalized})
    return failures


def _gte(actual: float, expected: float) -> bool:
    return actual >= expected


def _eq(actual: int, expected: int) -> bool:
    return actual == expected


def _float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0
