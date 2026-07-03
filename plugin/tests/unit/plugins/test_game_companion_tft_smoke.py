from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugin.plugins.game_companion.core.tft_smoke import build_tft_normal_shop_smoke_report


def test_build_tft_normal_shop_smoke_report_aggregates_three_fixed_runs(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake")
    calls: list[dict[str, Any]] = []

    def report_builder(video_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        calls.append({"video_path": video_path, **kwargs})
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        if output_dir.name == "normal_shop_50":
            return _fake_runtime_report(
                output_dir,
                summary={
                    "total_frames": 50,
                    "normal_shop_ready_rate": 1.0,
                    "cost_coverage": 1.0,
                    "name_coverage": 1.0,
                    "fallback_ratio": 0.25,
                    "main_blockers": {},
                    "shop_cost_source_counts": {"runtime_local_calibration_name_cost": 50, "slot_cost": 150},
                    "shop_name_source_counts": {"slot_name": 200},
                    "fallback_cost_count": 50,
                    "ocr_cost_count": 150,
                    "unknown_count": 0,
                    "contaminated_count": 0,
                },
                records=[_record("normal_shop", "ready")],
            )
        if output_dir.name == "mixed_33":
            return _fake_runtime_report(
                output_dir,
                summary={
                    "total_frames": 33,
                    "augment_ready_count": 11,
                    "normal_shop_ready_count": 11,
                    "combat_ready_count": 11,
                    "fallback_cost_count": 11,
                    "ocr_cost_count": 33,
                    "unknown_count": 0,
                    "contaminated_count": 0,
                },
                records=[
                    _record("augment", "ready"),
                    _record("normal_shop", "ready", shop_slots=[{"state": "occupied", "cost_source": "slot_cost"}]),
                    _record("combat", "ready"),
                ],
            )
        return _fake_runtime_report(
            output_dir,
            summary={
                "total_frames": 12,
                "contaminated_count": 12,
                "fallback_cost_count": 0,
                "unknown_count": 0,
                "main_blockers": {"contaminated_by_hover": 12},
            },
            records=[_record("popup", "contaminated")],
        )

    report = build_tft_normal_shop_smoke_report(video, output_dir=tmp_path / "smoke", report_builder=report_builder)

    assert report["type"] == "tft_normal_shop_smoke_report"
    assert report["report_version"] == "tft_normal_shop_smoke_v1"
    assert report["pass"] is True
    assert report["failures"] == []
    assert report["normal_shop"]["frame_count"] == 50
    assert report["normal_shop"]["ready_rate"] == 1.0
    assert report["normal_shop"]["cost_coverage"] == 1.0
    assert report["normal_shop"]["fallback_ratio"] == 0.25
    assert report["mixed"]["non_shop_source_slots"] == 0
    assert report["mixed"]["augment_ready_count"] == 11
    assert report["mixed"]["combat_ready_count"] == 11
    assert report["overlay"]["contaminated_count"] == 12
    assert report["overlay"]["shop_payloads"] == 0
    assert Path(report["report_path"]).is_file()
    assert [Path(call["output_dir"]).name for call in calls] == ["normal_shop_50", "mixed_33", "overlay_12"]
    assert len(calls[0]["frame_indices"]) == 50
    assert calls[0]["expected_layout"] == "normal_shop"
    assert len(calls[1]["frame_indices"]) == 33
    assert set(calls[1]["frame_layouts"].values()) == {"augment_select", "normal_shop", "combat"}
    assert len(calls[2]["frame_indices"]) == 12
    assert set(calls[2]["frame_layouts"].values()) == {"augment_select"}


def test_build_tft_normal_shop_smoke_report_reports_threshold_failures(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake")

    def report_builder(_video_path: str | Path, **kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        if output_dir.name == "normal_shop_50":
            return _fake_runtime_report(
                output_dir,
                summary={
                    "total_frames": 50,
                    "normal_shop_ready_rate": 0.92,
                    "cost_coverage": 0.72,
                    "name_coverage": 0.65,
                    "fallback_ratio": 0.8,
                    "main_blockers": {"shop_cost_ocr_failed": 14},
                },
                records=[_record("normal_shop", "partial")],
            )
        if output_dir.name == "mixed_33":
            return _fake_runtime_report(
                output_dir,
                summary={"total_frames": 33, "augment_ready_count": 11, "normal_shop_ready_count": 11, "combat_ready_count": 11},
                records=[_record("combat", "ready", shop_slots=[{"state": "occupied", "cost_source": "slot_cost"}])],
            )
        return _fake_runtime_report(
            output_dir,
            summary={"total_frames": 12, "contaminated_count": 12, "fallback_cost_count": 1},
            records=[_record("popup", "contaminated", shop_slots=[{"state": "occupied"}])],
        )

    report = build_tft_normal_shop_smoke_report(video, output_dir=tmp_path / "smoke", report_builder=report_builder)

    assert report["pass"] is False
    assert {
        (failure["check"], failure["expected"], failure["actual"])
        for failure in report["failures"]
    } == {
        ("normal_shop.ready_rate", ">= 0.95", 0.92),
        ("normal_shop.cost_coverage", ">= 0.9", 0.72),
        ("normal_shop.name_coverage", ">= 0.8", 0.65),
        ("mixed.non_shop_source_slots", "== 0", 1),
        ("overlay.shop_payloads", "== 0", 1),
        ("overlay.fallback_cost_count", "== 0", 1),
    }


def _fake_runtime_report(output_dir: Path, *, summary: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl_path = output_dir / "tft_state_v1.jsonl"
    summary_path = output_dir / "tft_state_summary_v1.json"
    contact_sheet_path = output_dir / "contact_sheet.png"
    jsonl_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    contact_sheet_path.write_bytes(b"fake png")
    return {
        "type": "tft_runtime_state_video_report",
        "report_version": "tft_state_v1",
        "frame_count": summary.get("total_frames", len(records)),
        "summary": summary,
        "jsonl_path": str(jsonl_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "contact_sheet_path": str(contact_sheet_path.resolve()),
    }


def _record(layout: str, readiness: str, *, shop_slots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    shop = {"slots": shop_slots or []} if shop_slots is not None else None
    return {"state": {"layout": layout, "readiness": readiness, "shop": shop}}
