from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from plugin.plugins.game_companion.profiles.tft import ocr as tft_ocr


@dataclass
class _FakeOcrBox:
    text: str
    left: float = 0.0
    top: float = 0.0
    right: float = 10.0
    bottom: float = 10.0
    score: float = 0.9


class _FakeRapidOcrBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def is_available(self) -> bool:
        return True

    def extract_text_with_boxes(self, image: Image.Image) -> tuple[str, list[_FakeOcrBox]]:
        self.calls.append(image.size)
        return "3", [_FakeOcrBox("3")]


def test_analyze_tft_ocr_regions_includes_shop_slot_subregions(tmp_path: Path, monkeypatch) -> None:
    screenshot = tmp_path / "shop.png"
    Image.new("RGB", (300, 200), "black").save(screenshot)
    backend = _FakeRapidOcrBackend()
    monkeypatch.setattr(tft_ocr, "_create_rapidocr_backend", lambda: backend)

    result = tft_ocr.analyze_tft_ocr_regions(
        screenshot,
        {
            "shop_slot_2": (10, 20, 110, 160),
            "shop_slot_2_name": (30, 120, 100, 150),
            "shop_slot_2_cost": (10, 120, 28, 150),
        },
    )

    assert result["status"] == "ready"
    assert set(result["regions"]) == {"shop_slot_2", "shop_slot_2_name", "shop_slot_2_cost"}
    assert result["regions"]["shop_slot_2_cost"]["text"] == "3"
    assert len(backend.calls) == 3
