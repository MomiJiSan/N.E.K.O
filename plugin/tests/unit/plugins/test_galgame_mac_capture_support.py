from __future__ import annotations

from pathlib import Path

from PIL import Image

from plugin.plugins.galgame_plugin import mac_capture_support
from plugin.plugins.galgame_plugin.ocr_reader import DetectedGameWindow, OcrCaptureProfile


def test_mac_native_window_capture_backend_stamps_capture_metadata(
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
        OcrCaptureProfile(),
    )

    assert frame.size == (640, 360)
    assert frame.info["galgame_capture_backend_kind"] == "imagegrab"
    assert frame.info["galgame_capture_backend_detail"] == "selected"
