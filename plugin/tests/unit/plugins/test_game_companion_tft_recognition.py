from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from plugin.plugins.game_companion.core.tft_recognition import (
    _parse_stage as _parse_recognition_stage,
    _human_label,
    _shop_slot_subregions,
    build_tft_recognition_report,
    recognize_tft_frame,
)
from plugin.plugins.game_companion.profiles.tft.ocr import _parse_stage as _parse_ocr_stage
from plugin.plugins.game_companion.profiles.tft.screen_regions import (
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_COMBAT,
    LAYOUT_NORMAL_SHOP,
    LAYOUT_SPECIAL,
    layout_region_bboxes,
)


class _FakeOcrAdapter:
    def __init__(self, *, status: str = "ready") -> None:
        self.status = status

    def recognize(self, _image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        if self.status != "ready":
            return {
                "available": False,
                "status": self.status,
                "error": "fake OCR unavailable",
                "regions": {},
                "parsed": {},
            }
        results = {}
        if "stage" in regions:
            results["stage"] = {"text": "3-2", "confidence": 0.94, "bbox": list(regions["stage"])}
        if "gold" in regions:
            results["gold"] = {"text": "48", "confidence": 0.91, "bbox": list(regions["gold"])}
        if "level_exp" in regions:
            results["level_exp"] = {"text": "5级 12/20", "confidence": 0.86, "bbox": list(regions["level_exp"])}
        if "augments" in regions:
            results["augments"] = {
                "text": "选择一件\n战力强化\n经济强化\n装备强化",
                "confidence": 0.81,
                "bbox": list(regions["augments"]),
            }
        for index in range(1, 4):
            key = f"augment_option_{index}"
            if key in regions:
                results[key] = {
                    "text": f"Augment {index}\nDescription {index}\n\u6218\u529b\u5f3a\u5316",
                    "confidence": 0.7 + index * 0.03,
                    "bbox": list(regions[key]),
                }
        for index in range(1, 6):
            key = f"shop_slot_{index}"
            if key in regions:
                results[key] = {
                    "text": f"Lux {min(index + 1, 5)}",
                    "confidence": 0.62,
                    "bbox": list(regions[key]),
                }
            name_key = f"{key}_name"
            if name_key in regions:
                results[name_key] = {
                    "text": "Lux",
                    "confidence": 0.66,
                    "bbox": list(regions[name_key]),
                }
            cost_key = f"{key}_cost"
            if cost_key in regions:
                results[cost_key] = {
                    "text": str(min(index + 1, 5)),
                    "confidence": 0.68,
                    "bbox": list(regions[cost_key]),
                }
        return {"available": True, "status": "ready", "error": None, "regions": results, "parsed": {}}


class _NoisyGoldOcrAdapter:
    def recognize(self, _image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        return {
            "available": True,
            "status": "ready",
            "error": None,
            "regions": {
                "round": {"text": "3-2", "confidence": 0.9, "bbox": list(regions["round"])},
                "gold": {"text": "9 48", "confidence": 0.8, "bbox": list(regions["gold"])},
                "level_exp": {"text": "5级 12/20", "confidence": 0.86, "bbox": list(regions["level_exp"])},
            },
            "parsed": {},
        }


class _MissingStageOcrAdapter:
    def recognize(self, _image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        return {
            "available": True,
            "status": "ready",
            "error": None,
            "regions": {
                "gold": {"text": "48", "confidence": 0.9, "bbox": list(regions["gold"])},
                "level_exp": {"text": "5级 12/20", "confidence": 0.86, "bbox": list(regions["level_exp"])},
            },
            "parsed": {},
        }


class _FailingOcrAdapter:
    def recognize(self, _image_path: str | Path, _regions: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("OCR exploded")


class _RecordingShopOcrAdapter(_FakeOcrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.region_keys: set[str] = set()

    def recognize(self, image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        self.region_keys = set(regions)
        return super().recognize(image_path, regions)


class _ShopCostParseFailedOcrAdapter(_FakeOcrAdapter):
    def recognize(self, _image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        result = super().recognize(_image_path, regions)
        for key in list(result["regions"]):
            if key.endswith("_cost"):
                result["regions"][key]["text"] = "coin"
            elif key.startswith("shop_slot_") and key.rsplit("_", 1)[-1].isdigit():
                result["regions"][key]["text"] = "Lux"
        return result


class _PathSensitiveShopCostOcrAdapter(_FakeOcrAdapter):
    def recognize(self, image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        result = super().recognize(image_path, regions)
        names = {1: "UnitOne", 2: "UnitTwo", 3: "UnitThree", 4: "UnitFour", 5: "UnitFive"}
        for index in range(1, 6):
            cost = min(index + 1, 5)
            slot_key = f"shop_slot_{index}"
            if slot_key in result["regions"]:
                result["regions"][slot_key]["text"] = f"{names[index]} {cost}"
            name_key = f"{slot_key}_name"
            if name_key in result["regions"]:
                result["regions"][name_key]["text"] = names[index]
            cost_key = f"{slot_key}_cost"
            if cost_key in result["regions"]:
                result["regions"][cost_key]["text"] = str(cost)
        if "missing_cost" not in Path(image_path).stem:
            return result
        for key in list(result["regions"]):
            if key == "shop_slot_2" or key == "shop_slot_2_cost":
                result["regions"][key]["text"] = "UnitTwo"
        return result


class _SingleMissingCostOcrAdapter(_PathSensitiveShopCostOcrAdapter):
    def recognize(self, image_path: str | Path, regions: dict[str, Any]) -> dict[str, Any]:
        result = super().recognize(image_path, regions)
        for key in ("shop_slot_2", "shop_slot_2_cost"):
            if key in result["regions"]:
                result["regions"][key]["text"] = "UnitTwo" if key == "shop_slot_2" else ""
        return result


def _synthetic_tft_image(path: Path, layout: str = LAYOUT_NORMAL_SHOP) -> None:
    image = Image.new("RGB", (1920, 1080), color=(42, 72, 94))
    draw = ImageDraw.Draw(image)
    regions = layout_region_bboxes(1920, 1080, layout)
    for key in ("stage", "gold", "level_exp"):
        if key in regions:
            draw.rectangle(regions[key], fill=(20, 30, 30))
    if "shop_slot_1" in regions:
        draw.rectangle(regions["shop_slot_1"], fill=(5, 8, 10))
    if "shop_slot_2" in regions:
        draw.rectangle(regions["shop_slot_2"], fill=(45, 90, 155))
        draw.rectangle((680, 900, 820, 1030), fill=(150, 210, 80))
    if "augments" in regions:
        draw.rectangle(regions["augments"], fill=(72, 34, 140))
    image.save(path)


def test_tft_recognition_outputs_structured_normal_shop_state(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_FakeOcrAdapter(),
    )

    assert result["type"] == "tft_recognition_result"
    assert result["layout"] == LAYOUT_NORMAL_SHOP
    assert result["stage"]["value"] == "3-2"
    assert result["gold"]["value"] == 48
    assert result["level"]["value"] == 5
    assert result["xp"] == {"current": 12, "required": 20, "confidence": 0.86, "raw_text": "5级 12/20"}
    assert len(result["shop"]) == 5
    assert result["shop"][0]["state"] == "empty"
    assert result["shop"][1]["state"] == "occupied"
    assert result["shop"][0]["raw_text"] == ""
    assert result["shop"][0]["review_status"] == "empty"
    assert result["shop"][1]["raw_text"] == "Lux 3"
    assert result["shop"][1]["name"] == "Lux"
    assert result["shop"][1]["cost"] == 3
    assert result["shop"][1]["name_candidate"] == "Lux"
    assert result["shop"][1]["cost_candidate"] == 3
    assert result["shop"][1]["ocr_lines"] == ["Lux 3"]
    assert result["shop"][1]["name_raw_text"] == "Lux"
    assert result["shop"][1]["cost_raw_text"] == "3"
    assert result["shop"][1]["name_candidate_source"] == "slot_name"
    assert result["shop"][1]["cost_candidate_source"] == "slot_cost"
    assert result["shop"][1]["review_status"] == "needs_check"
    assert result["augments"] == []
    assert result["field_status"]["shop"]["status"] == "present"
    assert result["warnings"] == []
    assert 0.0 <= result["confidence"] <= 1.0


def test_tft_recognition_uses_round_fallback_and_last_gold_number(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_NoisyGoldOcrAdapter(),
    )

    assert result["stage"]["value"] == "3-2"
    assert result["gold"]["value"] == 48


def test_tft_stage_parser_accepts_cjk_prefix_noise() -> None:
    assert _parse_recognition_stage("\u603b3-2") == "3-2"
    assert _parse_ocr_stage("\u5fd84-2") == "4-2"


def test_tft_shop_slot_subregions_keep_traits_outside_cost_area() -> None:
    subregions = _shop_slot_subregions((100, 200, 300, 500))

    assert subregions["slot_full"] == (100, 200, 300, 500)
    assert subregions["slot_cost"] == (100, 428, 136, 491)
    assert subregions["slot_name"] == (128, 428, 290, 491)
    assert subregions["slot_traits"][3] <= subregions["slot_cost"][1]


def test_tft_human_label_preserves_extra_review_metadata() -> None:
    assert _human_label({"name": "Lux", "cost": 3, "status": "verified", "notes": "checked"}) == {
        "name": "Lux",
        "cost": 3,
        "status": "verified",
        "notes": "checked",
    }
    assert _human_label({"title": "Augment", "description": "Desc", "status": "verified", "tags": ["good"]}, augment=True) == {
        "title": "Augment",
        "description": "Desc",
        "status": "verified",
        "tags": ["good"],
    }


def test_tft_recognition_skips_shop_for_combat_layout(tmp_path: Path) -> None:
    screenshot = tmp_path / "combat.png"
    _synthetic_tft_image(screenshot, layout=LAYOUT_COMBAT)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_COMBAT,
        ocr_adapter=_FakeOcrAdapter(status="unavailable"),
    )

    assert result["layout"] == LAYOUT_COMBAT
    assert result["shop"] == []
    assert result["gold"] is None
    assert result["level"] is None
    assert result["field_status"]["shop"]["status"] == "not_applicable"
    assert result["field_status"]["gold"]["status"] == "not_applicable"
    assert result["warnings"][0]["code"] == "ocr_unavailable"


def test_tft_recognition_ocr_exception_becomes_warning(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_FailingOcrAdapter(),
    )

    assert result["success"] is True
    assert result["stage"] is None
    assert result["shop"]
    assert result["shop"][1]["review_status"] == "ocr_missing"
    assert result["shop"][1]["raw_text"] == ""
    assert result["shop"][1]["name"] is None
    assert result["shop"][1]["cost"] is None
    assert result["warnings"][0]["code"] == "ocr_failed"


def test_tft_recognition_warns_when_expected_fields_are_missing(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_MissingStageOcrAdapter(),
    )

    assert result["success"] is True
    assert result["stage"] is None
    assert any(warning["code"] == "field_missing" and warning["field"] == "stage" for warning in result["warnings"])


def test_tft_recognition_reports_detailed_shop_cost_blocker(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_ShopCostParseFailedOcrAdapter(),
    )

    issue_codes = {issue["code"] for issue in result["readiness"]["blocking_issues"]}
    assert "shop_cost_parse_failed" in issue_codes
    assert "ocr_failed" not in issue_codes


def test_tft_recognition_marks_single_missing_shop_cost_partial_with_slot_blocker(tmp_path: Path) -> None:
    screenshot = tmp_path / "missing_cost.png"
    _synthetic_tft_image(screenshot)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=_SingleMissingCostOcrAdapter(),
    )

    assert result["readiness"]["status"] == "partial"
    issues = result["readiness"]["blocking_issues"]
    cost_issue = next(issue for issue in issues if issue["code"] == "shop_cost_ocr_failed")
    assert cost_issue["slots"] == [2]
    assert cost_issue["count"] == 1


def test_tft_recognition_reads_augment_text_without_shop_noise(tmp_path: Path) -> None:
    screenshot = tmp_path / "augment.png"
    _synthetic_tft_image(screenshot, layout=LAYOUT_AUGMENT_SELECT)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_AUGMENT_SELECT,
        ocr_adapter=_FakeOcrAdapter(),
    )

    assert result["layout"] == LAYOUT_AUGMENT_SELECT
    assert result["shop"] == []
    assert result["gold"] is None
    assert result["augments"][0]["slot"] == 1
    assert result["field_status"]["shop"]["status"] == "not_applicable"
    assert result["field_status"]["augments"]["status"] == "present"
    assert "战力强化" in result["augments"][0]["raw_text"]


def test_tft_recognition_only_sends_occupied_shop_slots_to_ocr(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    _synthetic_tft_image(screenshot)
    adapter = _RecordingShopOcrAdapter()

    recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_NORMAL_SHOP,
        ocr_adapter=adapter,
    )

    assert "shop_slot_1" not in adapter.region_keys
    assert "shop_slot_2" in adapter.region_keys
    assert "shop_slot_2_name" in adapter.region_keys
    assert "shop_slot_2_cost" in adapter.region_keys


def test_tft_recognition_splits_augment_options(tmp_path: Path) -> None:
    screenshot = tmp_path / "augment.png"
    _synthetic_tft_image(screenshot, layout=LAYOUT_AUGMENT_SELECT)

    result = recognize_tft_frame(
        screenshot,
        expected_layout=LAYOUT_AUGMENT_SELECT,
        ocr_adapter=_FakeOcrAdapter(),
    )

    assert len(result["augments"]) == 3
    assert result["augments"][0]["slot"] == 1
    assert result["augments"][0]["title"] == "Augment 1"
    assert result["augments"][0]["description"] == "Description 1 战力强化"
    assert result["augments"][0]["review_status"] == "needs_check"


def test_tft_recognition_report_batches_calibration_screenshots(tmp_path: Path) -> None:
    normal = tmp_path / "normal.png"
    combat = tmp_path / "combat.png"
    augment = tmp_path / "augment.png"
    _synthetic_tft_image(normal)
    _synthetic_tft_image(combat, layout=LAYOUT_COMBAT)
    _synthetic_tft_image(augment, layout=LAYOUT_AUGMENT_SELECT)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(normal), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "shop"},
            {"index": 2, "image_path": str(combat), "expected_layout": LAYOUT_COMBAT, "label": "combat"},
            {"index": 3, "image_path": str(augment), "expected_layout": LAYOUT_AUGMENT_SELECT, "label": "augment"},
        ],
    }
    output_dir = tmp_path / "recognition"
    output_dir.mkdir()
    labels_path = output_dir / "recognition_shop_labels_v1.json"
    labels_path.write_text(
        """{
  "type": "tft_shop_labels",
  "schema_version": 1,
  "report_version": "recognition_shop_labels_v1",
  "samples": [
    {
      "index": 1,
      "slot": 2,
      "human": {"name": "Verified Lux", "cost": 3, "status": "verified"}
    }
  ]
}""",
        encoding="utf-8",
    )
    augment_review_path = output_dir / "recognition_augment_review_v1.json"
    augment_review_path.write_text(
        """{
  "type": "tft_augment_review",
  "schema_version": 1,
  "report_version": "recognition_augment_review_v1",
  "samples": [
    {
      "index": 3,
      "augments": [
        {
          "slot": 1,
          "human_label": {
            "title": "Verified Augment",
            "description": "Verified description",
            "status": "verified",
            "notes": "kept"
          }
        }
      ]
    }
  ]
}""",
        encoding="utf-8",
    )

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=output_dir,
        ocr_adapter=_FakeOcrAdapter(),
    )

    assert report["type"] == "tft_recognition_report"
    assert report["summary"]["total"] == 3
    assert report["summary"]["successes"] == 3
    assert report["summary"]["metrics"]["stage_present_rate"] == 1.0
    assert report["summary"]["metrics"]["shop_slot_state_rate"] == 1.0
    assert report["summary"]["metrics"]["shop_cost_candidate_rate"] > 0.0
    assert report["summary"]["metrics"]["shop_name_candidate_rate"] > 0.0
    assert report["summary"]["metrics"]["shop_cost_verified_rate"] > 0.0
    assert report["summary"]["metrics"]["shop_name_verified_rate"] > 0.0
    assert report["summary"]["metrics"]["augment_title_candidate_rate"] == 1.0
    assert report["summary"]["metrics"]["augment_description_candidate_rate"] == 1.0
    assert report["summary"]["metrics"]["augment_title_verified_rate"] > 0.0
    assert report["summary"]["metrics"]["augment_description_verified_rate"] > 0.0
    assert report["summary"]["metrics"]["shop_occupied_slot_count"] > 0
    assert report["summary"]["metrics"]["shop_label_count"] > 0
    assert report["summary"]["metrics"]["augment_option_count"] == 3
    assert report["summary"]["metrics"]["augment_label_count"] == 3
    assert report["summary"]["readiness"][LAYOUT_NORMAL_SHOP]["status"] == "ready"
    assert report["summary"]["readiness"][LAYOUT_COMBAT]["status"] == "ready"
    assert report["summary"]["readiness"][LAYOUT_COMBAT]["fields"]["shop"]["status"] == "not_applicable"
    assert report["summary"]["readiness"]["augment"]["status"] == "ready"
    assert report["summary"]["readiness"]["augment"]["not_applicable"] == [
        "shop_slots",
        "shop_names",
        "shop_costs",
        "gold",
        "level",
        "xp",
    ]
    assert report["results"][2]["recognition"]["readiness"]["layout"] == "augment"
    assert report["results"][2]["recognition"]["readiness"]["required_checks"] == [
        "augment_options",
        "augment_titles",
        "augment_descriptions",
    ]
    assert report["results"][0]["recognition"]["shop"][1]["state"] == "occupied"
    assert Path(report["report_path"]).is_file()
    assert Path(report["summary_path"]).is_file()
    assert Path(report["layout_manifest_path"]).is_file()
    assert Path(report["shop_review_path"]).is_file()
    assert Path(report["shop_labels_path"]).is_file()
    assert Path(report["augment_review_path"]).is_file()
    layout_manifest = json.loads(Path(report["layout_manifest_path"]).read_text(encoding="utf-8"))
    assert layout_manifest["type"] == "tft_layout_manifest"
    assert [sample["layout"] for sample in layout_manifest["samples"]] == ["normal_shop", "combat", "augment"]
    assert layout_manifest["samples"][1]["not_applicable"] == [
        "shop_slots",
        "shop_names",
        "shop_costs",
        "augment_options",
    ]
    assert layout_manifest["summary"]["combat"]["ready"] == 1
    shop_review = json.loads(Path(report["shop_review_path"]).read_text(encoding="utf-8"))
    first_slot = shop_review["samples"][0]["shop"][0]
    occupied_slot = shop_review["samples"][0]["shop"][1]
    assert first_slot["state"] == "empty"
    assert set(first_slot["crop_paths"]) == {"slot_full"}
    assert occupied_slot["name_candidate"] == "Lux"
    assert occupied_slot["cost_candidate"] == 3
    assert occupied_slot["human_label"] == {"name": "Verified Lux", "cost": 3, "status": "verified"}
    assert set(occupied_slot["crop_paths"]) == {"slot_full", "slot_name", "slot_cost", "slot_traits"}
    for crop_path in occupied_slot["crop_paths"].values():
        assert Path(crop_path).is_file()
    shop_labels = json.loads(Path(report["shop_labels_path"]).read_text(encoding="utf-8"))
    assert shop_labels["type"] == "tft_shop_labels"
    assert shop_labels["samples"][0]["human"] == {"name": "Verified Lux", "cost": 3, "status": "verified"}
    augment_review = json.loads(Path(report["augment_review_path"]).read_text(encoding="utf-8"))
    assert augment_review["type"] == "tft_augment_review"
    assert len(augment_review["samples"]) == 1
    assert len(augment_review["samples"][0]["augments"]) == 3
    assert augment_review["samples"][0]["augments"][0]["title_candidate"] == "Augment 1"
    assert Path(augment_review["samples"][0]["augments"][0]["crop_path"]).is_file()
    assert augment_review["samples"][0]["augments"][0]["human_label"] == {
        "title": "Verified Augment",
        "description": "Verified description",
        "status": "verified",
        "notes": "kept",
    }


def test_tft_recognition_report_marks_popup_contaminated_and_portal_layout_only(tmp_path: Path) -> None:
    popup = tmp_path / "popup_tooltip.png"
    portal = tmp_path / "portal_vote.png"
    _synthetic_tft_image(popup, layout=LAYOUT_SPECIAL)
    _synthetic_tft_image(portal, layout=LAYOUT_SPECIAL)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(popup), "expected_layout": "popup", "label": "hover tooltip"},
            {"index": 2, "image_path": str(portal), "expected_layout": "portal", "label": "portal region vote"},
        ],
    }

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=tmp_path / "recognition",
        ocr_adapter=_FakeOcrAdapter(),
    )

    assert report["summary"]["readiness"]["popup"]["status"] == "contaminated"
    assert report["summary"]["readiness"]["popup"]["main_blocker"] == "contaminated_by_hover"
    assert report["summary"]["readiness"]["portal"]["status"] == "ready"
    assert report["summary"]["readiness"]["portal"]["required_checks"] == []
    assert report["results"][0]["recognition"]["layout"] == LAYOUT_SPECIAL
    assert report["results"][0]["recognition"]["readiness"]["excluded_from_readiness"] is True
    assert report["results"][0]["recognition"]["readiness"]["blocking_issues"][0]["code"] == "contaminated_by_hover"
    assert report["results"][1]["recognition"]["readiness"]["layout"] == "portal"

    layout_manifest = json.loads(Path(report["layout_manifest_path"]).read_text(encoding="utf-8"))
    assert [sample["layout"] for sample in layout_manifest["samples"]] == ["popup", "portal"]
    assert layout_manifest["summary"]["popup"]["contaminated"] == 1
    assert layout_manifest["summary"]["portal"]["ready"] == 1


def test_tft_recognition_report_distinguishes_unknown_and_failed_samples(tmp_path: Path) -> None:
    unknown = tmp_path / "mystery.png"
    missing = tmp_path / "missing.png"
    _synthetic_tft_image(unknown, layout=LAYOUT_SPECIAL)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(unknown), "expected_layout": "unknown", "label": "unclassified"},
            {"index": 2, "image_path": str(missing), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "missing image"},
        ],
    }

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=tmp_path / "recognition",
        ocr_adapter=_FakeOcrAdapter(),
    )

    unknown_readiness = report["results"][0]["recognition"]["readiness"]
    failed_readiness = report["results"][1]["recognition"]["readiness"]
    assert unknown_readiness["layout"] == "unknown"
    assert unknown_readiness["readiness"] == "blocked"
    assert unknown_readiness["blocking_issues"][0]["code"] == "layout_unknown"
    assert failed_readiness["layout"] == "normal_shop"
    assert failed_readiness["readiness"] == "failed"
    assert failed_readiness["blocking_issues"][0]["code"] == "image_read_failed"

    layout_manifest = json.loads(Path(report["layout_manifest_path"]).read_text(encoding="utf-8"))
    assert layout_manifest["summary"]["unknown"]["main_blocker"] == "layout_unknown"
    assert layout_manifest["summary"]["normal_shop"]["failed"] == 1


def test_tft_recognition_report_fills_missing_shop_cost_from_batch_consensus(tmp_path: Path) -> None:
    known = tmp_path / "known_cost.png"
    missing = tmp_path / "missing_cost.png"
    _synthetic_tft_image(known)
    _synthetic_tft_image(missing)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(known), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "known"},
            {"index": 2, "image_path": str(missing), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "missing"},
        ],
    }

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=tmp_path / "recognition",
        ocr_adapter=_PathSensitiveShopCostOcrAdapter(),
    )

    inferred_slot = report["results"][1]["recognition"]["shop"][1]
    assert inferred_slot["cost_candidate"] == 3
    assert inferred_slot["cost_candidate_source"] == "report_name_cost_consensus"
    assert inferred_slot["cost_inference"]["matched_name"] == "UnitTwo"
    assert report["summary"]["readiness"][LAYOUT_NORMAL_SHOP]["status"] == "ready"


def test_tft_recognition_report_counts_partial_normal_shop_samples(tmp_path: Path) -> None:
    screenshot = tmp_path / "missing_cost.png"
    _synthetic_tft_image(screenshot)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(screenshot), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "missing"},
        ],
    }

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=tmp_path / "recognition",
        ocr_adapter=_SingleMissingCostOcrAdapter(),
    )

    readiness = report["summary"]["readiness"][LAYOUT_NORMAL_SHOP]
    assert readiness["status"] == "partial"
    assert readiness["partial"] == 1
    assert readiness["blocked"] == 0
    assert readiness["main_blocker"] == "shop_cost_ocr_failed"


def test_tft_recognition_report_uses_verified_shop_label_for_missing_cost(tmp_path: Path) -> None:
    screenshot = tmp_path / "missing_cost.png"
    _synthetic_tft_image(screenshot)
    output_dir = tmp_path / "recognition"
    output_dir.mkdir()
    (output_dir / "recognition_shop_labels_v1.json").write_text(
        """{
  "type": "tft_shop_labels",
  "schema_version": 1,
  "report_version": "recognition_shop_labels_v1",
  "samples": [
    {
      "index": 1,
      "slot": 2,
      "human": {"name": "UnitTwo", "cost": 4, "status": "verified"}
    }
  ]
}""",
        encoding="utf-8",
    )
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {"index": 1, "image_path": str(screenshot), "expected_layout": LAYOUT_NORMAL_SHOP, "label": "missing"},
        ],
    }

    report = build_tft_recognition_report(
        calibration_report,
        output_dir=output_dir,
        ocr_adapter=_PathSensitiveShopCostOcrAdapter(),
    )

    inferred_slot = report["results"][0]["recognition"]["shop"][1]
    assert inferred_slot["cost_candidate"] == 4
    assert inferred_slot["cost_candidate_source"] == "human_verified_label"
