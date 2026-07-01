from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from plugin.plugins.game_companion.profiles.tft import recognition
from plugin.plugins.game_companion.profiles.tft.screen_regions import SHOP_SLOT_KEYS, shop_slot_bboxes


def _recognition_data_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    unit_assets = data_dir / "assets" / "units"
    item_assets = data_dir / "assets" / "items"
    unit_assets.mkdir(parents=True)
    item_assets.mkdir(parents=True)
    units_path = data_dir / "units.json"
    items_path = data_dir / "items.json"
    traits_path = data_dir / "traits.json"
    units_path.write_text('{"schema_version": 1, "units": []}', encoding="utf-8")
    items_path.write_text('{"schema_version": 1, "items": []}', encoding="utf-8")
    traits_path.write_text('{"schema_version": 1, "traits": []}', encoding="utf-8")
    return data_dir, unit_assets, item_assets


def _template_image(size: tuple[int, int] = (176, 242)) -> Image.Image:
    image = Image.new("RGB", size, color=(24, 36, 54))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, size[0] - 18, size[1] - 18), outline=(220, 198, 78), width=8)
    draw.ellipse((44, 38, size[0] - 44, size[1] - 82), fill=(65, 144, 210))
    draw.rectangle((58, size[1] - 78, size[0] - 58, size[1] - 34), fill=(162, 80, 190))
    return image


def test_shop_slot_recognition_returns_unknowns_without_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, _unit_assets, _item_assets = _recognition_data_dir(tmp_path)
    screenshot = Image.new("RGB", (1920, 1080), color=(10, 20, 30))

    payload = recognition.recognize_tft_shop_slots(screenshot, data_dir=data_dir)

    assert payload["kind"] == "shop_units"
    assert len(payload["slots"]) == 5
    assert [slot["slot"] for slot in payload["slots"]] == list(SHOP_SLOT_KEYS)
    assert payload["diagnostics"]["templates"]["loaded"] == 0
    for slot in payload["slots"]:
        assert slot["result"]["status"] == "unknown"
        assert slot["result"]["id"] is None
        assert slot["result"]["reason"] in {"no_templates", "imagehash_unavailable"}


def test_imagehash_unavailable_is_reported_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, _unit_assets, _item_assets = _recognition_data_dir(tmp_path)
    monkeypatch.setattr(recognition, "_imagehash", None)

    payload = recognition.recognize_tft_item_icon(
        Image.new("RGB", (64, 64), color=(80, 120, 160)),
        data_dir=data_dir,
    )

    assert payload["result"]["status"] == "unknown"
    assert payload["result"]["reason"] == "imagehash_unavailable"
    assert payload["diagnostics"]["imagehash"]["status"] == "unavailable"


def test_shop_slot_recognition_matches_template_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imagehash = pytest.importorskip("imagehash")
    data_dir, unit_assets, _item_assets = _recognition_data_dir(tmp_path)
    monkeypatch.setattr(recognition, "_imagehash", imagehash)
    (data_dir / "units.json").write_text(
        '{"schema_version": 1, "units": [{"id": "demo_unit", "name": "Demo Unit"}]}',
        encoding="utf-8",
    )
    slot_boxes = shop_slot_bboxes(1920, 1080)
    first_slot = slot_boxes["shop_slot_1"]
    template = _template_image(size=(first_slot[2] - first_slot[0], first_slot[3] - first_slot[1]))
    (unit_assets / "demo_unit.png").parent.mkdir(parents=True, exist_ok=True)
    template.save(unit_assets / "demo_unit.png")
    screenshot = Image.new("RGB", (1920, 1080), color=(2, 8, 14))
    for bbox in slot_boxes.values():
        screenshot.paste(template.resize((bbox[2] - bbox[0], bbox[3] - bbox[1])), box=bbox[:2])

    payload = recognition.recognize_tft_shop_slots(screenshot, data_dir=data_dir)

    assert payload["diagnostics"]["imagehash"]["status"] == "ready"
    assert payload["diagnostics"]["templates"]["loaded"] == 1
    for slot in payload["slots"]:
        assert slot["result"]["status"] == "matched"
        assert slot["result"]["id"] == "demo_unit"
        assert slot["result"]["name"] == "Demo Unit"
        assert slot["result"]["confidence"] >= 0.84


def test_low_confidence_shop_slot_does_not_guess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imagehash = pytest.importorskip("imagehash")
    data_dir, unit_assets, _item_assets = _recognition_data_dir(tmp_path)
    monkeypatch.setattr(recognition, "_imagehash", imagehash)
    (data_dir / "units.json").write_text(
        '{"schema_version": 1, "units": [{"id": "demo_unit", "name": "Demo Unit"}]}',
        encoding="utf-8",
    )
    _template_image().save(unit_assets / "demo_unit.png")
    screenshot = Image.new("RGB", (1920, 1080), color=(250, 250, 250))

    payload = recognition.recognize_tft_shop_slots(
        screenshot,
        confidence_threshold=1.01,
        data_dir=data_dir,
    )

    for slot in payload["slots"]:
        assert slot["result"]["status"] == "unknown"
        assert slot["result"]["id"] is None
        assert slot["result"]["reason"] == "low_confidence"
        assert slot["result"]["best_match"]["id"] == "demo_unit"
