from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
import types

from PIL import Image

from plugin.plugins.game_companion.core.calibration import (
    build_tft_layout_calibration_report,
    build_tft_layout_calibration_status,
    build_tft_layout_sample_manifest,
    capture_tft_layout_calibration_screenshot,
    extract_tft_layout_calibration_video_frames,
    init_tft_layout_calibration_workspace,
    load_tft_layout_sample_manifest,
    summarize_tft_layout_calibration_report,
    update_tft_layout_calibration_check,
    update_tft_layout_calibration_checks,
)
from plugin.plugins.game_companion.profiles.tft.screen_regions import (
    LAYOUT_AUGMENT_SELECT,
    LAYOUT_COMBAT,
    LAYOUT_NORMAL_SHOP,
    LAYOUT_SPECIAL,
    LAYOUT_STATES,
    SHOP_SLOT_KEYS,
    TFT_LAYOUT_PROFILES,
    grouped_screen_region_metadata,
    layout_profile,
    layout_region_bboxes,
    save_debug_crops,
    screen_region_definitions,
    screen_region_metadata,
)


def test_tft_layout_profiles_define_reconnaissance_states() -> None:
    assert LAYOUT_STATES == (
        LAYOUT_NORMAL_SHOP,
        LAYOUT_COMBAT,
        LAYOUT_AUGMENT_SELECT,
        LAYOUT_SPECIAL,
    )
    assert set(TFT_LAYOUT_PROFILES) == set(LAYOUT_STATES)
    assert layout_profile(LAYOUT_NORMAL_SHOP).deep_recognition is True
    assert layout_profile(LAYOUT_SPECIAL).deep_recognition is False
    assert "shop_slots" in layout_profile(LAYOUT_NORMAL_SHOP).primary_regions
    assert "augments" in layout_profile(LAYOUT_AUGMENT_SELECT).primary_regions


def test_screen_regions_expose_semantic_metadata_for_16_9_layout() -> None:
    metadata = screen_region_metadata(1920, 1080)

    assert metadata["shop_slot_1"]["layout"] == LAYOUT_NORMAL_SHOP
    assert metadata["shop_slot_1"]["active_layouts"] == [LAYOUT_NORMAL_SHOP]
    assert metadata["shop_slot_1"]["purpose"] == "unit_recognition"
    assert metadata["shop_slot_1"]["recognizers"] == ["template_hash", "ocr"]
    assert metadata["stage"]["active_layouts"] == list(LAYOUT_STATES)
    assert LAYOUT_COMBAT in metadata["board"]["active_layouts"]
    assert LAYOUT_SPECIAL in metadata["board"]["active_layouts"]
    assert metadata["gold"]["priority"] < metadata["shop_slot_1"]["priority"]
    assert metadata["augments"]["layout"] == LAYOUT_AUGMENT_SELECT
    assert metadata["augments"]["active_layouts"] == [LAYOUT_AUGMENT_SELECT]
    assert metadata["players_panel"]["purpose"] == "players"
    assert metadata["level_exp"]["purpose"] == "economy"


def test_layout_region_bboxes_filter_by_primary_layout() -> None:
    normal_regions = layout_region_bboxes(1920, 1080, LAYOUT_NORMAL_SHOP)
    augment_regions = layout_region_bboxes(1920, 1080, LAYOUT_AUGMENT_SELECT)

    assert all(
        key in normal_regions
        for key in ("stage", "gold", "level_exp", "shop", "shop_odds", "refresh_button", "buy_xp_button", "board")
    )
    assert "equipment" in normal_regions
    assert all(key in normal_regions for key in SHOP_SLOT_KEYS)
    assert "augments" not in normal_regions
    assert set(augment_regions) == {"stage", "round", "augments", "notifications"}
    assert [region.key for region in screen_region_definitions(layout=LAYOUT_AUGMENT_SELECT)] == [
        "stage",
        "round",
        "augments",
        "notifications",
    ]


def test_grouped_region_metadata_exposes_semantic_groups() -> None:
    grouped = grouped_screen_region_metadata(1920, 1080)

    assert set(grouped["layout_profiles"]) == set(LAYOUT_STATES)
    assert len(grouped["groups"]["shop_slots"]) == 5
    assert any(region["key"] == "gold" for region in grouped["groups"]["normal_shop"])
    assert any(region["key"] == "equipment" for region in grouped["groups"]["normal_shop"])
    assert any(region["key"] == "board" for region in grouped["groups"]["combat"])
    assert any(region["key"] == "equipment" for region in grouped["groups"]["combat"])
    assert any(region["key"] == "augments" for region in grouped["groups"]["augment_select"])


def test_debug_crops_include_layout_metadata_and_readable_filenames(tmp_path: Path) -> None:
    image_path = tmp_path / "tft.png"
    crops_dir = tmp_path / "crops"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(image_path)

    payload = save_debug_crops(image_path, crops_dir)

    shop_crop = Path(payload["crops"]["shop_slot_1"])
    assert shop_crop.is_file()
    assert shop_crop.name.startswith("normal_shop__p04__shop_slot_1")
    assert payload["metadata"]["shop_slot_1"]["purpose"] == "unit_recognition"
    assert payload["metadata"]["shop_slot_1"]["crop_path"] == str(shop_crop.resolve())
    assert payload["metadata"]["augments"]["layout"] == LAYOUT_AUGMENT_SELECT
    assert payload["layout_profiles"][LAYOUT_NORMAL_SHOP]["primary_regions"][0] == "stage"
    assert Path(payload["metadata_path"]).is_file()


def test_debug_crops_can_filter_to_expected_layout(tmp_path: Path) -> None:
    image_path = tmp_path / "tft.png"
    crops_dir = tmp_path / "crops"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(image_path)

    payload = save_debug_crops(image_path, crops_dir, layout=LAYOUT_COMBAT)

    assert payload["capture_layout"] == LAYOUT_COMBAT
    assert "board" in payload["crops"]
    assert "equipment" in payload["crops"]
    assert "shop_slot_1" not in payload["crops"]
    assert "augments" not in payload["crops"]
    assert Path(payload["crops"]["board"]).name.startswith("combat__p08__board")
    assert payload["metadata"]["board"]["capture_layout"] == LAYOUT_COMBAT


def test_layout_calibration_report_batches_screenshots_and_writes_review_files(tmp_path: Path) -> None:
    screenshot_a = tmp_path / "normal_shop.png"
    screenshot_b = tmp_path / "combat.png"
    output_dir = tmp_path / "calibration"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot_a)
    Image.new("RGB", (1920, 1080), color=(40, 30, 20)).save(screenshot_b)

    report = build_tft_layout_calibration_report([screenshot_a, screenshot_b], output_dir)

    assert report["type"] == "tft_layout_calibration_report"
    assert report["summary"]["total"] == 2
    assert report["summary"]["successes"] == 2
    assert report["summary"]["calibration_ready"] == 2
    assert report["summary"]["calibration_blocked"] == 0
    assert report["summary"]["ready_for_manual_review"] is True
    assert report["summary"]["has_recommended_sample_count"] is False
    assert Path(report["report_path"]).is_file()
    assert Path(report["html_path"]).is_file()
    html = Path(report["html_path"]).read_text(encoding="utf-8")
    assert "game_companion_summarize_layout_calibration" in html
    assert 'data-check-id="shop_slots_complete"' in html
    assert {check["id"] for check in report["manual_checks"]} >= {
        "shop_slots_complete",
        "gold_clean",
        "items_area_reasonable",
    }
    first = report["screenshots"][0]
    assert first["success"] is True
    assert first["calibration_ready"] is True
    assert Path(first["debug_crops"]["crops"]["shop_slot_1"]).is_file()
    assert first["debug_crops"]["metadata"]["shop_slot_1"]["purpose"] == "unit_recognition"


def test_layout_calibration_marks_non_16_9_screenshot_not_ready(tmp_path: Path) -> None:
    screenshot = tmp_path / "not_16_9.png"
    output_dir = tmp_path / "calibration"
    Image.new("RGB", (1024, 768), color=(20, 30, 40)).save(screenshot)

    report = build_tft_layout_calibration_report([screenshot], output_dir)

    assert report["summary"]["successes"] == 1
    assert report["summary"]["calibration_ready"] == 0
    assert report["summary"]["calibration_blocked"] == 1
    assert report["summary"]["ready_for_manual_review"] is False
    first = report["screenshots"][0]
    assert first["success"] is True
    assert first["calibration_ready"] is False
    assert first["calibration_error"]["code"] == "unsupported_aspect_ratio"
    assert first["debug_crops"] == {}
    assert report["layout_metadata"] == {}


def test_layout_calibration_samples_track_layout_and_tag_coverage(tmp_path: Path) -> None:
    samples = []
    for index, layout in enumerate(
        [LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_AUGMENT_SELECT, LAYOUT_SPECIAL, LAYOUT_NORMAL_SHOP],
        start=1,
    ):
        screenshot = tmp_path / f"{index}_{layout}.png"
        Image.new("RGB", (1920, 1080), color=(index * 20, 30, 40)).save(screenshot)
        tags = []
        if index == 1:
            tags = ["shop_open", "shop_five_units", "bench_units"]
        elif index == 2:
            tags = ["items_visible"]
        elif index == 3:
            tags = ["traits_panel_expanded"]
        samples.append(
            {
                "image_path": str(screenshot),
                "expected_layout": layout,
                "tags": tags,
                "label": f"sample-{index}",
            }
        )

    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples=samples)

    coverage = report["summary"]["coverage"]
    assert report["summary"]["has_recommended_sample_count"] is True
    assert report["summary"]["has_layout_state_coverage"] is True
    assert report["summary"]["has_recommended_tag_coverage"] is True
    assert coverage["layout_counts"][LAYOUT_NORMAL_SHOP] == 2
    assert coverage["missing_layouts"] == []
    assert coverage["missing_tags"] == []
    assert report["screenshots"][0]["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert report["screenshots"][0]["label"] == "sample-1"
    html = Path(report["html_path"]).read_text(encoding="utf-8")
    assert "Sample Coverage" in html
    assert "Expected layout" in html

    for screenshot in report["screenshots"]:
        for check in screenshot["manual_checks"]:
            check["status"] = "pass"
    summary = summarize_tft_layout_calibration_report(report)

    assert summary["has_layout_state_coverage"] is True
    assert summary["has_recommended_tag_coverage"] is True
    assert summary["ready_for_recognition"] is True


def test_layout_calibration_first_pass_roi_does_not_require_special_sample(tmp_path: Path) -> None:
    samples = []
    layouts = [
        LAYOUT_NORMAL_SHOP,
        LAYOUT_NORMAL_SHOP,
        LAYOUT_COMBAT,
        LAYOUT_COMBAT,
        LAYOUT_AUGMENT_SELECT,
    ]
    tags_by_index = {
        1: ["shop_open", "shop_five_units", "bench_units"],
        3: ["items_visible"],
        5: ["traits_panel_expanded"],
    }
    for index, layout in enumerate(layouts, start=1):
        screenshot = tmp_path / f"{index}_{layout}.png"
        Image.new("RGB", (1920, 1080), color=(index * 20, 30, 40)).save(screenshot)
        samples.append(
            {
                "image_path": str(screenshot),
                "expected_layout": layout,
                "tags": tags_by_index.get(index, []),
            }
        )
    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples=samples)

    for screenshot in report["screenshots"]:
        for check in screenshot["manual_checks"]:
            check["status"] = "pass"
    summary = summarize_tft_layout_calibration_report(report)

    assert summary["coverage"]["missing_layouts"] == [LAYOUT_SPECIAL]
    assert summary["coverage"]["missing_first_pass_layouts"] == []
    assert summary["has_layout_state_coverage"] is False
    assert summary["has_first_pass_layout_coverage"] is True
    assert summary["ready_for_first_pass_roi"] is True
    assert summary["ready_for_recognition"] is True


def test_layout_calibration_crop_acceptance_allows_documented_90_percent_pass_rate(tmp_path: Path) -> None:
    samples = []
    layouts = [
        LAYOUT_NORMAL_SHOP,
        LAYOUT_NORMAL_SHOP,
        LAYOUT_COMBAT,
        LAYOUT_COMBAT,
        LAYOUT_AUGMENT_SELECT,
        LAYOUT_AUGMENT_SELECT,
        LAYOUT_SPECIAL,
        LAYOUT_SPECIAL,
    ]
    tags_by_index = {
        1: ["shop_open", "shop_five_units", "bench_units"],
        3: ["items_visible"],
        5: ["traits_panel_expanded"],
    }
    for index, layout in enumerate(layouts, start=1):
        screenshot = tmp_path / f"{index}_{layout}.png"
        Image.new("RGB", (1920, 1080), color=(index * 20, 30, 40)).save(screenshot)
        samples.append(
            {
                "image_path": str(screenshot),
                "expected_layout": layout,
                "tags": tags_by_index.get(index, []),
            }
    )
    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples=samples)

    for screenshot in report["screenshots"]:
        for check in screenshot["manual_checks"]:
            check["status"] = "pass"
    misses = (
        (0, "bench_complete"),
        (2, "traits_panel_aligned"),
        (4, "items_area_reasonable"),
    )
    for screenshot_index, check_id in misses:
        check = next(item for item in report["screenshots"][screenshot_index]["manual_checks"] if item["id"] == check_id)
        check["status"] = "needs_adjustment"
        check["note"] = "Documented crop padding follow-up."

    summary = summarize_tft_layout_calibration_report(report)

    assert summary["crop_acceptance"]["misses"] == 3
    assert summary["crop_acceptance"]["pass_rate"] >= 0.9
    assert summary["crop_acceptance"]["review_complete"] is True
    assert summary["crop_acceptance"]["misses_documented"] is True
    assert summary["crop_acceptance"]["meets_acceptance"] is True
    assert summary["layout_acceptance"]["meets_acceptance"] is True
    assert summary["critical_acceptance"]["meets_acceptance"] is True
    assert summary["ready_for_recognition"] is True


def test_layout_calibration_crop_acceptance_blocks_unchecked_or_undocumented_misses(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot)
    report = build_tft_layout_calibration_report([screenshot], tmp_path / "calibration")

    unchecked_summary = summarize_tft_layout_calibration_report(report)
    assert unchecked_summary["crop_acceptance"]["review_complete"] is False
    assert unchecked_summary["crop_acceptance"]["meets_acceptance"] is False

    for check in report["screenshots"][0]["manual_checks"]:
        check["status"] = "pass"
    report["screenshots"][0]["manual_checks"][0]["status"] = "fail"
    undocumented_summary = summarize_tft_layout_calibration_report(report)

    assert undocumented_summary["crop_acceptance"]["review_complete"] is True
    assert undocumented_summary["crop_acceptance"]["misses_documented"] is False
    assert undocumented_summary["crop_acceptance"]["undocumented_misses"][0]["check_id"] == "shop_slots_complete"
    assert undocumented_summary["crop_acceptance"]["meets_acceptance"] is False


def test_layout_calibration_blocks_critical_check_misses_even_when_documented(tmp_path: Path) -> None:
    samples = []
    layouts = [LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_AUGMENT_SELECT, LAYOUT_SPECIAL, LAYOUT_NORMAL_SHOP]
    tags_by_index = {
        1: ["shop_open", "shop_five_units", "bench_units"],
        2: ["items_visible"],
        3: ["traits_panel_expanded"],
    }
    for index, layout in enumerate(layouts, start=1):
        screenshot = tmp_path / f"{index}_{layout}.png"
        Image.new("RGB", (1920, 1080), color=(index * 20, 30, 40)).save(screenshot)
        samples.append(
            {
                "image_path": str(screenshot),
                "expected_layout": layout,
                "tags": tags_by_index.get(index, []),
            }
        )
    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples=samples)
    for screenshot in report["screenshots"]:
        for check in screenshot["manual_checks"]:
            check["status"] = "pass"
    critical = next(item for item in report["screenshots"][0]["manual_checks"] if item["id"] == "gold_clean")
    critical["status"] = "needs_adjustment"
    critical["note"] = "Gold crop needs a tighter bbox."

    summary = summarize_tft_layout_calibration_report(report)

    assert summary["crop_acceptance"]["meets_acceptance"] is True
    assert summary["critical_acceptance"]["blocking_checks"] == ["gold_clean"]
    assert summary["critical_acceptance"]["meets_acceptance"] is False
    assert summary["ready_for_recognition"] is False


def test_layout_calibration_manifest_scans_directory_and_loads_samples(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    names = [
        "01_normal_shop_shop5_bench_traits_expanded_items.png",
        "02_combat_items.jpg",
        "03_augment_select.png",
        "04_special_carousel.webp",
        "ignore.txt",
    ]
    for name in names:
        path = input_dir / name
        if path.suffix == ".txt":
            path.write_text("ignore", encoding="utf-8")
        else:
            Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(path)

    manifest_path = tmp_path / "samples_manifest.json"
    manifest = build_tft_layout_sample_manifest(input_dir, manifest_path)

    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["schema_version"] == 1
    assert manifest["base_dir"] == str(input_dir.resolve())
    assert manifest["summary"]["total"] == 4
    assert manifest["samples"][0]["id"] == "01_normal_shop_shop5_bench_traits_expanded_items"
    assert manifest["samples"][0]["relative_path"] == "01_normal_shop_shop5_bench_traits_expanded_items.png"
    assert manifest["samples"][0]["file_size"] > 0
    assert manifest["samples"][0]["width"] == 1920
    assert manifest["samples"][0]["height"] == 1080
    assert manifest["samples"][0]["needs_manual_label"] is False
    assert manifest["samples"][0]["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert "shop_five_units" in manifest["samples"][0]["tags"]
    assert "bench_units" in manifest["samples"][0]["tags"]
    assert "traits_panel_expanded" in manifest["samples"][0]["tags"]
    assert "items_visible" in manifest["samples"][0]["tags"]
    assert manifest["summary"]["coverage"]["layout_counts"][LAYOUT_AUGMENT_SELECT] == 1

    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples_manifest_path=manifest_path)

    assert report["summary"]["total"] == 4
    assert report["screenshots"][0]["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert report["screenshots"][0]["label"].startswith("01_normal_shop")
    assert report["screenshots"][0]["relative_path"] == "01_normal_shop_shop5_bench_traits_expanded_items.png"


def test_layout_calibration_manifest_marks_empty_and_unknown_samples(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    unknown_dir = tmp_path / "unknown"
    empty_dir.mkdir()
    unknown_dir.mkdir()
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(unknown_dir / "mystery.png")

    empty = build_tft_layout_sample_manifest(empty_dir)
    unknown = build_tft_layout_sample_manifest(unknown_dir)

    assert empty["summary"]["total"] == 0
    assert empty["warnings"][0]["code"] == "no_screenshots_found"
    assert unknown["samples"][0]["expected_layout"] is None
    assert unknown["samples"][0]["tags"] == []
    assert unknown["samples"][0]["needs_manual_label"] is True


def test_layout_calibration_status_reports_manifest_file_issues(tmp_path: Path) -> None:
    manifest_path = tmp_path / "samples_manifest.json"
    bad_image = tmp_path / "bad.png"
    bad_image.write_text("not an image", encoding="utf-8")
    missing_image = tmp_path / "missing.png"
    manifest_path.write_text(
        json.dumps(
            {
                "type": "tft_layout_calibration_samples_manifest",
                "schema_version": 1,
                "base_dir": str(tmp_path),
                "samples": [
                    {"image_path": str(bad_image), "expected_layout": LAYOUT_NORMAL_SHOP, "tags": ["shop_open"]},
                    {"image_path": str(missing_image), "expected_layout": LAYOUT_COMBAT, "tags": ["items_visible"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    status = build_tft_layout_calibration_status(samples_manifest_path=manifest_path)
    checks = status["samples_manifest"]["summary"]["file_checks"]

    assert status["samples_manifest"]["valid"] is True
    assert checks["all_files_exist"] is False
    assert checks["all_images_decodable"] is False
    assert checks["missing_files"] == [str(missing_image)]
    assert checks["decode_errors"][0]["image_path"] == str(bad_image)
    assert status["next_steps"] == ["Fix missing screenshot paths in samples_manifest.json, then check calibration status again."]


def test_layout_calibration_workspace_init_creates_local_structure(tmp_path: Path) -> None:
    workspace = init_tft_layout_calibration_workspace(tmp_path / "workspace")

    assert Path(workspace["input_dir"]).is_dir()
    assert Path(workspace["output_dir"]).is_dir()
    assert Path(workspace["reports_dir"]).is_dir()
    assert Path(workspace["samples_manifest_path"]).is_file()
    assert Path(workspace["readme_path"]).is_file()
    assert workspace["manifest"]["summary"]["total"] == 0
    assert "ready_for_recognition" in Path(workspace["readme_path"]).read_text(encoding="utf-8")


def test_layout_calibration_capture_saves_sample_with_diagnostics(tmp_path: Path) -> None:
    capture = capture_tft_layout_calibration_screenshot(
        tmp_path,
        label="shop open",
        expected_layout=LAYOUT_NORMAL_SHOP,
        tags=["shop_open", "shop_five_units"],
        image_grabber=lambda: Image.new("RGB", (1920, 1080), color=(80, 90, 100)),
    )

    image_path = Path(capture["image_path"])

    assert capture["type"] == "tft_layout_calibration_capture"
    assert image_path.is_file()
    assert image_path.name.startswith("tft_normal_shop_shop_open_shop_five_units_")
    assert capture["metadata"]["width"] == 1920
    assert capture["metadata"]["height"] == 1080
    assert capture["diagnostics"]["looks_black"] is False
    assert capture["warnings"] == []
    assert capture["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert capture["tags"] == ["shop_open", "shop_five_units"]


def test_layout_calibration_capture_warns_on_black_or_non_16_9_sample(tmp_path: Path) -> None:
    capture = capture_tft_layout_calibration_screenshot(
        tmp_path,
        image_grabber=lambda: Image.new("RGB", (1024, 768), color=(0, 0, 0)),
    )

    assert {warning["code"] for warning in capture["warnings"]} == {
        "unsupported_aspect_ratio",
        "possible_black_capture",
    }
    assert capture["diagnostics"]["looks_black"] is True


def test_layout_calibration_video_extraction_writes_frames_and_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "match.mp4"
    output_dir = tmp_path / "frames"
    manifest_path = tmp_path / "samples_manifest.json"
    video_path.write_bytes(b"fake video placeholder")

    def fake_reader(_video_path: Path, **_: object):
        return [
            {"frame_index": 10, "timestamp_seconds": 1.25, "image": Image.new("RGB", (1920, 1080), color=(20, 30, 40))},
            {"frame_index": 200, "timestamp_seconds": 25.0, "image": Image.new("RGB", (1920, 1080), color=(40, 30, 20))},
        ]

    payload = extract_tft_layout_calibration_video_frames(
        video_path,
        output_dir=output_dir,
        samples_manifest_path=manifest_path,
        expected_layout=LAYOUT_NORMAL_SHOP,
        tags=["shop_open", "shop_five_units"],
        label="shop open",
        frame_reader=fake_reader,
    )

    assert payload["type"] == "tft_layout_calibration_video_frames"
    assert payload["video_path"] == str(video_path.resolve())
    assert payload["output_dir"] == str(output_dir.resolve())
    assert payload["frame_count"] == 2
    assert Path(payload["frames"][0]["image_path"]).is_file()
    assert payload["frames"][0]["frame_index"] == 10
    assert payload["frames"][0]["timestamp_seconds"] == 1.25
    assert payload["frames"][0]["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert payload["frames"][0]["tags"] == ["shop_open", "shop_five_units"]
    assert payload["frames"][0]["source"] == {
        "type": "video_frame",
        "profile_id": "tft",
        "video_path": str(video_path.resolve()),
        "ordinal": 1,
        "frame_index": 10,
        "timestamp_seconds": 1.25,
        "expected_layout": LAYOUT_NORMAL_SHOP,
        "tags": ["shop_open", "shop_five_units"],
        "label": "shop open",
    }
    assert payload["frames"][0]["warnings"] == []
    assert payload["manifest"]["manifest_path"] == str(manifest_path.resolve())
    assert payload["manifest"]["summary"]["total"] == 2
    assert payload["manifest"]["samples"][0]["expected_layout"] == LAYOUT_NORMAL_SHOP
    assert "shop_five_units" in payload["manifest"]["samples"][0]["tags"]
    assert payload["manifest"]["samples"][0]["source"]["type"] == "video_frame"
    assert payload["manifest"]["samples"][0]["source"]["frame_index"] == 10
    assert payload["manifest"]["samples"][0]["source"]["timestamp_seconds"] == 1.25
    loaded_samples = load_tft_layout_sample_manifest(manifest_path)
    assert loaded_samples[0]["source"] == payload["manifest"]["samples"][0]["source"]
    report = build_tft_layout_calibration_report([], tmp_path / "calibration", samples_manifest_path=manifest_path)
    assert report["screenshots"][0]["sample_source"] == payload["manifest"]["samples"][0]["source"]
    assert report["screenshots"][0]["source"]["origin"] == {
        **payload["manifest"]["samples"][0]["source"],
        "video_path": "[redacted_path]",
    }
    assert "game_companion_calibrate_layout" in payload["next_steps"][-1]


def test_layout_calibration_video_extraction_reports_frame_quality(tmp_path: Path) -> None:
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"fake video placeholder")

    def fake_reader(_video_path: Path, **_: object):
        return [{"frame_index": 1, "timestamp_seconds": 0.1, "image": Image.new("RGB", (1024, 768), color=(0, 0, 0))}]

    payload = extract_tft_layout_calibration_video_frames(video_path, output_dir=tmp_path / "frames", frame_reader=fake_reader)

    warning_codes = {warning["code"] for warning in payload["frames"][0]["warnings"]}
    assert warning_codes == {"unsupported_aspect_ratio", "possible_black_frame"}
    assert payload["manifest"]["samples"][0]["needs_manual_label"] is True


def test_layout_calibration_video_extraction_uses_pyav_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"fake video placeholder")

    class _FakeFrame:
        def __init__(self, index: int) -> None:
            self.pts = index * 10

        def to_image(self) -> Image.Image:
            return Image.new("RGB", (1920, 1080), color=(20 + self.pts, 30, 40))

    class _FakeStream:
        type = "video"
        frames = 2
        time_base = Fraction(1, 100)

    class _FakeStreams:
        video = [_FakeStream()]

        def __iter__(self):
            return iter(self.video)

    class _FakeContainer:
        streams = _FakeStreams()

        def decode(self, video: int = 0):
            assert video == 0
            yield _FakeFrame(0)
            yield _FakeFrame(1)

        def close(self) -> None:
            pass

    fake_av = types.ModuleType("av")
    fake_av.open = lambda _path: _FakeContainer()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    payload = extract_tft_layout_calibration_video_frames(
        video_path,
        output_dir=tmp_path / "frames",
        max_frames=2,
    )

    assert payload["frame_count"] == 2
    assert payload["frames"][0]["frame_index"] == 0
    assert payload["frames"][0]["timestamp_seconds"] == 0.0
    assert payload["manifest"]["samples"][0]["source"]["timestamp_seconds"] == 0.0
    assert payload["frames"][1]["frame_index"] == 1
    assert payload["frames"][1]["timestamp_seconds"] == 0.1
    assert payload["manifest"]["summary"]["total"] == 2


def test_layout_calibration_pyav_reader_seeks_to_explicit_frame_indices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "match.mp4"
    video_path.write_bytes(b"fake video placeholder")
    seeks: list[int] = []
    decoded_windows: list[int] = []

    class _FakeFrame:
        def __init__(self, index: int) -> None:
            self.pts = index

        def to_image(self) -> Image.Image:
            return Image.new("RGB", (1920, 1080), color=(20, 30, 40))

    class _FakeStream:
        type = "video"
        frames = 100_000
        time_base = Fraction(1, 60)

    class _FakeStreams:
        video = [_FakeStream()]

        def __iter__(self):
            return iter(self.video)

    class _FakeContainer:
        streams = _FakeStreams()

        def __init__(self) -> None:
            self.current = 0

        def seek(self, offset: int, *, stream: object, any_frame: bool = False, backward: bool = True) -> None:
            assert stream is self.streams.video[0]
            assert backward is True
            seeks.append(offset)
            self.current = int(offset)

        def decode(self, video: int = 0):
            assert video == 0
            start = self.current
            decoded_windows.append(start)
            for index in range(start, start + 3):
                yield _FakeFrame(index)

        def close(self) -> None:
            pass

    fake_av = types.ModuleType("av")
    fake_av.open = lambda _path: _FakeContainer()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    payload = extract_tft_layout_calibration_video_frames(
        video_path,
        output_dir=tmp_path / "frames",
        frame_indices=[0, 50_000, 90_000],
        max_frames=3,
    )

    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 50_000, 90_000]
    assert decoded_windows == [0, 50_000, 90_000]
    assert seeks == [0, 50_000, 90_000]


def test_layout_calibration_video_extraction_rejects_missing_or_unsupported_video(tmp_path: Path) -> None:
    try:
        extract_tft_layout_calibration_video_frames(tmp_path / "missing.mp4", output_dir=tmp_path / "frames")
    except FileNotFoundError as exc:
        assert "calibration video was not found" in str(exc)
    else:
        raise AssertionError("missing video should fail")

    unsupported = tmp_path / "match.txt"
    unsupported.write_text("not a video", encoding="utf-8")

    try:
        extract_tft_layout_calibration_video_frames(unsupported, output_dir=tmp_path / "frames")
    except ValueError as exc:
        assert "unsupported calibration video extension" in str(exc)
    else:
        raise AssertionError("unsupported video extension should fail")


def test_layout_calibration_status_reports_next_steps(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    input_dir = tmp_path / "input"
    empty_dir.mkdir()
    input_dir.mkdir()
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(input_dir / "normal_shop_shop5_bench.png")

    empty_status = build_tft_layout_calibration_status(input_dir=empty_dir)
    input_status = build_tft_layout_calibration_status(input_dir=input_dir)
    missing_status = build_tft_layout_calibration_status(samples_manifest_path=tmp_path / "missing_manifest.json")

    assert empty_status["input_dir"]["exists"] is True
    assert empty_status["input_dir"]["sample_count"] == 0
    assert empty_status["next_steps"] == ["Add TFT screenshots to the input directory, then prepare the manifest again."]
    assert input_status["input_dir"]["sample_count"] == 1
    assert input_status["next_steps"] == ["Review or write samples_manifest.json, then run game_companion_calibrate_layout."]
    assert missing_status["samples_manifest"]["valid"] is False
    assert missing_status["samples_manifest"]["error"]["code"] == "samples_manifest_not_found"


def test_layout_calibration_status_reports_video_decoder_preflight() -> None:
    status = build_tft_layout_calibration_status()
    decoders = status["video_decoders"]

    assert set(decoders) == {"available", "preferred", "opencv", "pyav"}
    assert isinstance(decoders["available"], bool)
    assert decoders["preferred"] in {"opencv", "pyav", None}
    assert isinstance(decoders["opencv"]["available"], bool)
    assert isinstance(decoders["pyav"]["available"], bool)
    if decoders["opencv"]["available"]:
        assert decoders["preferred"] == "opencv"
    elif decoders["pyav"]["available"]:
        assert decoders["preferred"] == "pyav"
    else:
        assert decoders["preferred"] is None


def test_layout_calibration_manifest_rejects_invalid_samples(tmp_path: Path) -> None:
    empty_manifest = tmp_path / "empty_manifest.json"
    invalid_sample_manifest = tmp_path / "invalid_sample_manifest.json"
    unsupported_manifest = tmp_path / "unsupported_manifest.json"
    empty_manifest.write_text(
        json.dumps({"type": "tft_layout_calibration_samples_manifest", "schema_version": 1, "samples": []}),
        encoding="utf-8",
    )
    invalid_sample_manifest.write_text(
        json.dumps({"type": "tft_layout_calibration_samples_manifest", "schema_version": 1, "samples": [{}]}),
        encoding="utf-8",
    )
    unsupported_manifest.write_text(
        json.dumps(
            {
                "type": "tft_layout_calibration_samples_manifest",
                "schema_version": 1,
                "samples": [{"image_path": str(tmp_path / "not_image.txt")}],
            }
        ),
        encoding="utf-8",
    )

    for manifest_path in (empty_manifest, invalid_sample_manifest, unsupported_manifest):
        try:
            build_tft_layout_calibration_report([], tmp_path / manifest_path.stem, samples_manifest_path=manifest_path)
        except ValueError as exc:
            assert str(exc)
        else:
            raise AssertionError(f"expected invalid manifest to fail: {manifest_path}")


def test_layout_calibration_summary_tracks_manual_check_statuses(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    output_dir = tmp_path / "calibration"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot)

    report = build_tft_layout_calibration_report([screenshot], output_dir)
    checks = report["screenshots"][0]["manual_checks"]
    checks[0]["status"] = "pass"
    checks[1]["status"] = "needs_adjustment"
    checks[1]["note"] = "Gold crop includes part of the buy XP button."
    checks[2]["status"] = "fail"
    Path(report["report_path"]).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    summary = summarize_tft_layout_calibration_report(report)
    path_summary = summarize_tft_layout_calibration_report(report["report_path"])

    assert summary["type"] == "tft_layout_calibration_annotation_summary"
    assert summary["status_counts"]["pass"] == 1
    assert summary["status_counts"]["needs_adjustment"] == 1
    assert summary["status_counts"]["fail"] == 1
    assert summary["ready_for_region_tuning"] is True
    assert summary["ready_for_recognition"] is False
    assert any(item["region"] == "gold" for item in summary["failed_regions"])
    gold_check = next(item for item in summary["per_check"] if item["id"] == "gold_clean")
    assert gold_check["notes"][0]["note"] == "Gold crop includes part of the buy XP button."
    assert path_summary["status_counts"]["pass"] == 1
    assert path_summary["status_counts"]["needs_adjustment"] == 1
    assert path_summary["status_counts"]["fail"] == 1


def test_layout_calibration_update_check_refreshes_report_and_html(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    output_dir = tmp_path / "calibration"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot)
    report = build_tft_layout_calibration_report([screenshot], output_dir)

    updated = update_tft_layout_calibration_check(
        report["report_path"],
        screenshot_index=1,
        check_id="gold_clean",
        status="pass",
        note="Gold crop is centered.",
    )
    saved_report = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    html = Path(report["html_path"]).read_text(encoding="utf-8")

    assert updated["updated"] is True
    assert updated["check"]["status"] == "pass"
    assert updated["check"]["note"] == "Gold crop is centered."
    assert updated["annotation_summary"]["status_counts"]["pass"] == 1
    assert saved_report["screenshots"][0]["manual_checks"][1]["status"] == "pass"
    assert saved_report["annotation_summary"]["status_counts"]["pass"] == 1
    assert "Gold crop is centered." in html


def test_layout_calibration_batch_update_checks_refreshes_once(tmp_path: Path) -> None:
    screenshot = tmp_path / "normal_shop.png"
    output_dir = tmp_path / "calibration"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot)
    report = build_tft_layout_calibration_report([screenshot], output_dir)

    updated = update_tft_layout_calibration_checks(
        report["report_path"],
        updates=[
            {
                "screenshot_index": 1,
                "check_id": "gold_clean",
                "status": "pass",
                "note": "Gold crop is centered.",
            },
            {
                "screenshot_index": 1,
                "check_id": "level_exp_clean",
                "status": "needs_adjustment",
                "note": "XP text needs extra right padding.",
            },
        ],
    )
    saved_report = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert updated["updated"] is True
    assert len(updated["updates"]) == 2
    assert updated["annotation_summary"]["status_counts"]["pass"] == 1
    assert updated["annotation_summary"]["status_counts"]["needs_adjustment"] == 1
    assert saved_report["annotation_summary"]["status_counts"]["needs_adjustment"] == 1
