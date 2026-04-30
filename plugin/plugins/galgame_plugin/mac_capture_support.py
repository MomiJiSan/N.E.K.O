from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image

_SCREENCAPTURE_PATH = Path("/usr/sbin/screencapture")


def _crop_window_image(image: Any, profile) -> Any:
    width, height = image.size
    left = int(width * float(getattr(profile, "left_inset_ratio", 0.0) or 0.0))
    right = int(width * (1.0 - float(getattr(profile, "right_inset_ratio", 0.0) or 0.0)))
    top = int(height * float(getattr(profile, "top_ratio", 0.0) or 0.0))
    bottom = int(height * (1.0 - float(getattr(profile, "bottom_inset_ratio", 0.0) or 0.0)))

    left = max(0, min(left, width))
    right = max(left, min(right, width))
    top = max(0, min(top, height))
    bottom = max(top, min(bottom, height))

    cropped = image.crop((left, top, right, bottom))
    cropped.info["galgame_bounds_coordinate_space"] = "capture"
    cropped.info["galgame_source_size"] = {
        "width": float(cropped.size[0]),
        "height": float(cropped.size[1]),
    }
    cropped.info["galgame_capture_rect"] = {
        "left": float(left),
        "top": float(top),
        "right": float(right),
        "bottom": float(bottom),
    }
    cropped.info["galgame_window_rect"] = {
        "left": 0.0,
        "top": 0.0,
        "right": float(width),
        "bottom": float(height),
    }
    return cropped


class MacNativeWindowCaptureBackend:
    kind = "imagegrab"

    def __init__(
        self,
        *,
        logger=None,
        selection: str = "smart",
        capture_window: Callable[..., None] | None = None,
    ) -> None:
        self._logger = logger
        self.selection = str(selection or "smart").strip().lower()
        self._capture_window = capture_window or self._run_screencapture

    def is_available(self) -> bool:
        return sys.platform == "darwin" and _SCREENCAPTURE_PATH.exists()

    def describe_target(self, target) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target, profile) -> Any:
        window_id = int(getattr(target, "hwnd", 0) or 0)
        if window_id <= 0:
            raise RuntimeError("target_window_not_resolved_for_capture")
        with tempfile.TemporaryDirectory(prefix="neko-galgame-mac-") as tmpdir:
            output_path = Path(tmpdir) / "window.png"
            self._capture_window(window_id=window_id, output_path=output_path)
            with Image.open(output_path) as source:
                image = source.convert("RGB")
                image.load()
        cropped = _crop_window_image(image, profile)
        cropped.info["galgame_capture_backend_kind"] = self.kind
        cropped.info["galgame_capture_backend_detail"] = "selected"
        return cropped

    @staticmethod
    def _run_screencapture(*, window_id: int, output_path: Path) -> None:
        subprocess.run(
            [
                "/usr/sbin/screencapture",
                "-x",
                "-o",
                "-l",
                str(window_id),
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
