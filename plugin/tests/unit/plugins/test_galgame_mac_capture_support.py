from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from plugin.plugins.galgame_plugin import mac_capture_support
from plugin.plugins.galgame_plugin.ocr_reader import DetectedGameWindow, OcrCaptureProfile


def test_mac_native_window_capture_backend_is_unavailable_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "exists", lambda self: True)

    backend = mac_capture_support.MacNativeWindowCaptureBackend(logger=None)

    assert backend.is_available() is False


def test_mac_native_window_capture_backend_is_unavailable_without_screencapture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "exists", lambda self: False)

    backend = mac_capture_support.MacNativeWindowCaptureBackend(logger=None)

    assert backend.is_available() is False


def test_mac_native_window_capture_backend_applies_profile_crop_and_stamps_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    del monkeypatch
    image_path = tmp_path / "capture.png"
    Image.new("RGB", (640, 360), "white").save(image_path)

    def _fake_capture_window(*, window_id: int, output_path: Path) -> None:
        del window_id
        output_path.write_bytes(image_path.read_bytes())

    backend = mac_capture_support.MacNativeWindowCaptureBackend(
        logger=None,
        selection="smart",
        capture_window=_fake_capture_window,
    )

    frame = backend.capture_frame(
        DetectedGameWindow(
            hwnd=901,
            title="Episode 1",
            process_name="SteamGame",
            pid=4242,
            width=1280,
            height=720,
            is_foreground=True,
        ),
        OcrCaptureProfile(
            left_inset_ratio=0.1,
            right_inset_ratio=0.2,
            top_ratio=0.25,
            bottom_inset_ratio=0.25,
        ),
    )

    assert frame.size == (448, 180)
    assert frame.info["galgame_bounds_coordinate_space"] == "capture"
    assert frame.info["galgame_source_size"] == {"width": 448.0, "height": 180.0}
    assert frame.info["galgame_capture_rect"] == {
        "left": 64.0,
        "top": 90.0,
        "right": 512.0,
        "bottom": 270.0,
    }
    assert frame.info["galgame_window_rect"] == {
        "left": 0.0,
        "top": 0.0,
        "right": 640.0,
        "bottom": 360.0,
    }
    assert frame.info["galgame_capture_backend_kind"] == "imagegrab"
    assert frame.info["galgame_capture_backend_detail"] == "selected"
