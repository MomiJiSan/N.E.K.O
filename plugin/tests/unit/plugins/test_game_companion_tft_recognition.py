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
                    "text": f"Lux {index + 1}",
                    "confidence": 0.62,
                    "bbox": list(regions[key]),
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
    assert subregions["slot_cost"] == (100, 416, 156, 500)
    assert subregions["slot_traits"][0] >= subregions["slot_cost"][2]


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
    assert report["results"][0]["recognition"]["shop"][1]["state"] == "occupied"
    assert Path(report["report_path"]).is_file()
    assert Path(report["summary_path"]).is_file()
    assert Path(report["shop_review_path"]).is_file()
    assert Path(report["shop_labels_path"]).is_file()
    assert Path(report["augment_review_path"]).is_file()
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
