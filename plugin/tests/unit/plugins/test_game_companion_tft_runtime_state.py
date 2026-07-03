from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from plugin.plugins.game_companion.core import tft_runtime
from plugin.plugins.game_companion.core.tft_runtime import build_tft_video_state_report
from plugin.plugins.game_companion.core.tft_state import build_tft_state


def test_build_tft_state_maps_normal_shop_to_runtime_json() -> None:
    recognition = {
        "success": True,
        "image_path": "C:/frames/shop.png",
        "layout": "normal_shop",
        "confidence": 0.87,
        "readiness": {"status": "ready", "blockers": []},
        "warnings": [],
        "stage": {"value": "3-2", "confidence": 0.91},
        "gold": {"value": 48, "confidence": 0.9},
        "level": {"value": 5, "confidence": 0.86},
        "shop": [
            {"slot": 1, "state": "empty", "confidence": 0.88},
            {
                "slot": 2,
                "state": "occupied",
                "name": "Lux",
                "cost": 3,
                "confidence": 0.92,
                "name_confidence": 0.66,
                "cost_confidence": 0.68,
            },
        ],
        "augments": [],
    }

    state = build_tft_state(recognition, timestamp=123.5, source_context={"type": "video_frame", "frame_index": 30})

    assert state == {
        "type": "tft_frame_state",
        "schema_version": 1,
        "game": "tft",
        "layout": "normal_shop",
        "readiness": "ready",
        "confidence": 0.87,
        "source_frame": "C:/frames/shop.png",
        "timestamp": 123.5,
        "source_context": {"type": "video_frame", "frame_index": 30},
        "shop": {
            "slots": [
                {
                    "slot": 1,
                    "state": "empty",
                    "name": None,
                    "cost": None,
                    "confidence": 0.88,
                    "name_confidence": 0.0,
                    "cost_confidence": 0.0,
                    "missing_fields": [],
                },
                {
                    "slot": 2,
                    "state": "occupied",
                    "name": "Lux",
                    "cost": 3,
                    "confidence": 0.92,
                    "name_confidence": 0.66,
                    "cost_confidence": 0.68,
                    "missing_fields": [],
                },
            ],
            "occupied_count": 1,
            "slot_count": 2,
            "partial_count": 0,
            "slot_issues": [],
        },
        "augment": None,
        "combat": None,
        "blockers": [],
        "warnings": [],
        "quality": {
            "hover_contaminated": False,
            "ocr_ready": True,
            "blocked": False,
        },
        "summary": "当前是商店界面，2 个商店栏位可识别，1 个有棋子。",
    }


def test_build_tft_state_keeps_augment_and_combat_layout_specific() -> None:
    augment = build_tft_state(
        {
            "success": True,
            "layout": "augment_select",
            "confidence": 0.8,
            "readiness": {"status": "ready", "blockers": []},
            "warnings": [],
            "augments": [
                {"slot": 1, "title": "Combat", "description": "Gain power", "confidence": 0.7},
                {"slot": 2, "title": "Economy", "description": "", "confidence": 0.72},
            ],
            "shop": [],
        }
    )
    combat = build_tft_state(
        {
            "success": True,
            "layout": "combat",
            "confidence": 0.74,
            "readiness": {"status": "ready", "blockers": []},
            "warnings": [],
            "shop": [],
            "augments": [],
        }
    )

    assert augment["layout"] == "augment"
    assert augment["readiness"] == "ready"
    assert augment["shop"] is None
    assert augment["augment"]["option_count"] == 2
    assert augment["quality"]["ocr_ready"] is True
    assert combat["layout"] == "combat"
    assert combat["readiness"] == "ready"
    assert combat["shop"] is None
    assert combat["augment"] is None
    assert combat["combat"] == {"status": "observed", "details": []}


def test_build_tft_state_maps_popup_and_unknown_to_non_ready_states() -> None:
    popup = build_tft_state(
        {
            "success": True,
            "layout": "special",
            "confidence": 0.4,
            "readiness": {
                "status": "contaminated",
                "main_blocker": "contaminated_by_hover",
                "blockers": [{"code": "contaminated_by_hover", "field": "layout"}],
            },
            "warnings": [{"code": "hover", "message": "tooltip"}],
            "shop": [],
            "augments": [],
        }
    )
    unknown = build_tft_state(
        {
            "success": True,
            "layout": "unknown",
            "confidence": 0.2,
            "readiness": {
                "status": "blocked",
                "main_blocker": "unknown_layout",
                "blockers": [{"code": "unknown_layout", "field": "layout"}],
            },
            "warnings": [],
            "shop": [],
            "augments": [],
        }
    )

    assert popup["layout"] == "popup"
    assert popup["readiness"] == "contaminated"
    assert popup["quality"]["hover_contaminated"] is True
    assert popup["blockers"][0]["code"] == "contaminated_by_hover"
    assert unknown["layout"] == "unknown"
    assert unknown["readiness"] == "blocked"
    assert unknown["quality"]["blocked"] is True


def test_build_tft_state_maps_recognition_error_to_blocked_state() -> None:
    state = build_tft_state(
        {
            "success": False,
            "image_path": "missing.png",
            "layout": "normal_shop",
            "error": {"code": "image_read_failed", "message": "cannot open"},
        }
    )

    assert state["game"] == "tft"
    assert state["readiness"] == "blocked"
    assert state["shop"] is None
    assert state["quality"]["ocr_ready"] is False
    assert state["blockers"] == [{"code": "image_read_failed", "message": "cannot open"}]
    assert state["summary"] == "当前截图无法识别：image_read_failed。"


def test_build_tft_state_uses_recognition_blocking_issues() -> None:
    state = build_tft_state(
        {
            "success": True,
            "layout": "normal_shop",
            "confidence": 0.4,
            "readiness": {
                "status": "blocked",
                "blocking_issues": [
                    {"code": "ocr_failed", "check": "stage", "message": "stage missing"},
                    {"code": "shop_cost_ocr_failed", "check": "shop_costs", "count": 5},
                ],
            },
            "warnings": [],
            "shop": [{"slot": 1, "state": "occupied", "name": "Lux", "cost": None, "confidence": 0.7}],
            "augments": [],
        }
    )

    assert state["readiness"] == "blocked"
    assert [item["code"] for item in state["blockers"]] == ["ocr_failed", "shop_cost_ocr_failed"]
    assert state["summary"] == "当前画面暂不可用，主要阻塞原因是 ocr_failed。"


def test_build_tft_state_summarizes_partial_shop_slot_issues() -> None:
    state = build_tft_state(
        {
            "success": True,
            "layout": "normal_shop",
            "confidence": 0.72,
            "readiness": {
                "status": "partial",
                "blocking_issues": [
                    {
                        "code": "shop_cost_ocr_failed",
                        "check": "shop_costs",
                        "count": 1,
                        "slots": [2],
                    }
                ],
            },
            "warnings": [],
            "shop": [
                {"slot": 1, "state": "occupied", "name": "Lux", "cost": 3, "confidence": 0.8},
                {"slot": 2, "state": "occupied", "name": "Ahri", "cost": None, "confidence": 0.7},
            ],
            "augments": [],
        }
    )

    assert state["readiness"] == "partial"
    assert state["shop"]["partial_count"] == 1
    assert state["shop"]["slot_issues"] == [{"slot": 2, "state": "occupied", "missing_fields": ["cost"]}]
    assert state["shop"]["slots"][1]["missing_fields"] == ["cost"]
    assert "商店可部分识别" in state["summary"]
    assert "slot 2 缺费用" in state["summary"]


def test_build_tft_video_state_report_writes_jsonl_summary_and_contact_sheet(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake video placeholder")
    output_dir = tmp_path / "runtime_state_v1"

    def frame_reader(_video_path: Path, *, frame_indices: list[int], max_frames: int) -> list[dict[str, object]]:
        assert frame_indices == [0, 60, 120]
        assert max_frames == 3
        return [
            {"frame_index": 0, "timestamp_seconds": 0.0, "image": Image.new("RGB", (1920, 1080), "navy")},
            {"frame_index": 60, "timestamp_seconds": 2.0, "image": Image.new("RGB", (1920, 1080), "black")},
            {"frame_index": 120, "timestamp_seconds": 4.0, "image": Image.new("RGB", (1920, 1080), "purple")},
        ]

    def recognizer(image_path: str | Path, *, expected_layout: str | None = None) -> dict[str, object]:
        layout = expected_layout or "normal_shop"
        return {
            "success": True,
            "image_path": str(image_path),
            "layout": layout,
            "confidence": 0.8,
            "readiness": {"status": "ready", "blockers": []},
            "warnings": [],
            "shop": [
                {"slot": 1, "state": "occupied", "name": "Lux", "cost": 3, "confidence": 0.7},
            ]
            if layout == "normal_shop"
            else [],
            "augments": [{"slot": 1, "title": "Combat", "description": "Gain power", "confidence": 0.7}]
            if layout == "augment_select"
            else [],
        }

    report = build_tft_video_state_report(
        video,
        output_dir=output_dir,
        max_frames=3,
        frame_indices=[0, 60, 120],
        frame_layouts={0: "normal_shop", 60: "augment_select", 120: "combat"},
        frame_reader=frame_reader,
        recognizer=recognizer,
    )

    jsonl_path = Path(report["jsonl_path"])
    summary_path = Path(report["summary_path"])
    contact_sheet_path = Path(report["contact_sheet_path"])
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert report["type"] == "tft_runtime_state_video_report"
    assert report["frame_count"] == 3
    assert len(records) == 3
    assert records[0]["state"]["layout"] == "normal_shop"
    assert records[1]["state"]["layout"] == "augment"
    assert records[2]["state"]["layout"] == "combat"
    assert summary["total_frames"] == 3
    assert summary["layout_counts"] == {"normal_shop": 1, "augment": 1, "combat": 1}
    assert summary["readiness_counts"] == {"ready": 3}
    assert summary["normal_shop_ready_count"] == 1
    assert summary["augment_ready_count"] == 1
    assert summary["combat_ready_count"] == 1
    assert summary["contaminated_count"] == 0
    assert summary["unknown_count"] == 0
    assert summary["main_blockers"] == {}
    assert contact_sheet_path.is_file()
    assert all(Path(frame["image_path"]).is_file() for frame in report["frames"])


def test_build_tft_video_state_report_applies_verified_shop_labels(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake video placeholder")
    labels_path = tmp_path / "recognition_shop_labels_v1.json"
    labels_path.write_text(
        json.dumps(
            {
                "type": "tft_shop_labels",
                "samples": [
                    {
                        "index": 6,
                        "image_path": "C:/captures/02_normal_shop_normal_shop_stage_2_f016445.png",
                        "slot": 2,
                        "human": {"name": "Ahri", "cost": 3, "status": "verified"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def frame_reader(_video_path: Path, *, frame_indices: list[int], max_frames: int) -> list[dict[str, object]]:
        return [{"frame_index": 16445, "timestamp_seconds": 274.0, "image": Image.new("RGB", (1920, 1080), "navy")}]

    def recognizer(image_path: str | Path, *, expected_layout: str | None = None) -> dict[str, object]:
        return {
            "success": True,
            "image_path": str(image_path),
            "layout": "normal_shop",
            "confidence": 0.8,
            "readiness": {
                "status": "partial",
                "blocking_issues": [
                    {
                        "code": "shop_cost_ocr_failed",
                        "check": "shop_costs",
                        "count": 1,
                        "slots": [2],
                    }
                ],
            },
            "warnings": [],
            "shop": [
                {"slot": 1, "state": "empty", "name": None, "cost": None, "confidence": 0.8},
                {"slot": 2, "state": "occupied", "name": "Ahri", "cost": None, "confidence": 0.8},
                {"slot": 3, "state": "occupied", "name": "Lux", "cost": 2, "confidence": 0.8},
                {"slot": 4, "state": "occupied", "name": "Jinx", "cost": 4, "confidence": 0.8},
                {"slot": 5, "state": "occupied", "name": "Vi", "cost": 1, "confidence": 0.8},
            ],
            "augments": [],
        }

    report = build_tft_video_state_report(
        video,
        output_dir=tmp_path / "runtime_state_v1",
        max_frames=1,
        frame_indices=[16445],
        frame_layouts={16445: "normal_shop"},
        shop_labels_path=labels_path,
        frame_reader=frame_reader,
        recognizer=recognizer,
    )

    record = json.loads(Path(report["jsonl_path"]).read_text(encoding="utf-8").splitlines()[0])
    occupied = [slot for slot in record["state"]["shop"]["slots"] if slot["state"] == "occupied"]
    cost_candidates = [slot for slot in occupied if slot["cost"] is not None]

    assert record["state"]["readiness"] == "ready"
    assert record["state"]["shop"]["slots"][1]["cost"] == 3
    assert record["raw_recognition"]["shop"][1]["cost_candidate_source"] == "human_verified_label"
    assert len(cost_candidates) / len(occupied) >= 0.8
    assert report["summary"]["normal_shop_ready_count"] == 1


def test_build_tft_video_state_report_applies_temporal_shop_cost_consensus(tmp_path: Path) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake video placeholder")

    def frame_reader(_video_path: Path, *, frame_indices: list[int], max_frames: int) -> list[dict[str, object]]:
        return [
            {"frame_index": 16445, "timestamp_seconds": 274.0, "image": Image.new("RGB", (1920, 1080), "navy")},
            {"frame_index": 16465, "timestamp_seconds": 274.3, "image": Image.new("RGB", (1920, 1080), "navy")},
        ]

    def recognizer(image_path: str | Path, *, expected_layout: str | None = None) -> dict[str, object]:
        frame_index = 16465 if "016465" in str(image_path) else 16445
        return {
            "success": True,
            "image_path": str(image_path),
            "layout": "normal_shop",
            "confidence": 0.8,
            "readiness": {
                "status": "partial" if frame_index == 16445 else "ready",
                "blocking_issues": [
                    {
                        "code": "shop_cost_ocr_failed",
                        "check": "shop_costs",
                        "count": 1,
                        "slots": [2],
                    }
                ]
                if frame_index == 16445
                else [],
            },
            "warnings": [],
            "shop": [
                {"slot": 1, "state": "empty", "name": None, "cost": None, "confidence": 0.8},
                {
                    "slot": 2,
                    "state": "occupied",
                    "name": "Ahri",
                    "cost": None if frame_index == 16445 else 3,
                    "confidence": 0.8,
                },
                {"slot": 3, "state": "occupied", "name": "Lux", "cost": 2, "confidence": 0.8},
                {"slot": 4, "state": "occupied", "name": "Jinx", "cost": 4, "confidence": 0.8},
                {"slot": 5, "state": "occupied", "name": "Vi", "cost": 1, "confidence": 0.8},
            ],
            "augments": [],
        }

    report = build_tft_video_state_report(
        video,
        output_dir=tmp_path / "runtime_state_v1",
        max_frames=2,
        frame_indices=[16445, 16465],
        frame_layouts={16445: "normal_shop", 16465: "normal_shop"},
        frame_reader=frame_reader,
        recognizer=recognizer,
    )

    records = [json.loads(line) for line in Path(report["jsonl_path"]).read_text(encoding="utf-8").splitlines()]
    first = records[0]
    occupied = [slot for slot in first["state"]["shop"]["slots"] if slot["state"] == "occupied"]
    cost_candidates = [slot for slot in occupied if slot["cost"] is not None]

    assert first["state"]["readiness"] == "ready"
    assert first["state"]["shop"]["slots"][1]["cost"] == 3
    assert first["raw_recognition"]["shop"][1]["cost_candidate_source"] == "runtime_temporal_slot_name_cost_consensus"
    assert len(cost_candidates) / len(occupied) >= 0.8
    assert report["summary"]["normal_shop_ready_count"] == 2


def test_build_tft_video_state_report_uses_local_calibration_cost_hints_without_label_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake video placeholder")
    calibration_root = tmp_path / "calibration"
    labels_path = calibration_root / "run" / "recognition_shop_labels_v1.json"
    labels_path.parent.mkdir(parents=True)
    labels_path.write_text(
        json.dumps(
            {
                "type": "tft_shop_labels",
                "samples": [
                    {
                        "slot": 2,
                        "human": {"name": "厄加特", "cost": 3, "status": "verified"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tft_runtime, "DEFAULT_LOCAL_CALIBRATION_DIR", calibration_root)

    def frame_reader(_video_path: Path, *, frame_indices: list[int], max_frames: int) -> list[dict[str, object]]:
        return [
            {"frame_index": 16445, "timestamp_seconds": 274.0, "image": Image.new("RGB", (1920, 1080), "navy")},
            {"frame_index": 16465, "timestamp_seconds": 274.3, "image": Image.new("RGB", (1920, 1080), "navy")},
        ]

    def recognizer(image_path: str | Path, *, expected_layout: str | None = None) -> dict[str, object]:
        return {
            "success": True,
            "image_path": str(image_path),
            "layout": "normal_shop",
            "confidence": 0.8,
            "readiness": {
                "status": "partial",
                "blocking_issues": [
                    {
                        "code": "shop_cost_ocr_failed",
                        "check": "shop_costs",
                        "count": 1,
                        "slots": [2],
                    }
                ],
            },
            "warnings": [],
            "shop": [
                {"slot": 1, "state": "empty", "name": None, "cost": None, "confidence": 0.8},
                {"slot": 2, "state": "occupied", "name": "厄加特", "cost": None, "confidence": 0.8},
                {"slot": 3, "state": "occupied", "name": "Lux", "cost": 2, "confidence": 0.8},
                {"slot": 4, "state": "occupied", "name": "Jinx", "cost": 4, "confidence": 0.8},
                {"slot": 5, "state": "occupied", "name": "Vi", "cost": 1, "confidence": 0.8},
            ],
            "augments": [],
        }

    report = build_tft_video_state_report(
        video,
        output_dir=tmp_path / "runtime_state_v1",
        max_frames=2,
        frame_indices=[16445, 16465],
        frame_layouts={16445: "normal_shop", 16465: "normal_shop"},
        frame_reader=frame_reader,
        recognizer=recognizer,
    )

    records = [json.loads(line) for line in Path(report["jsonl_path"]).read_text(encoding="utf-8").splitlines()]

    assert [record["state"]["readiness"] for record in records] == ["ready", "ready"]
    assert records[0]["state"]["shop"]["slots"][1]["cost"] == 3
    assert records[0]["raw_recognition"]["shop"][1]["cost_candidate_source"] == "runtime_local_calibration_name_cost"
    assert report["summary"]["normal_shop_ready_count"] == 2
