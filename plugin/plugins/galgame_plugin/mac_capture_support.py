from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image


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
        return True

    def describe_target(self, target) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target, profile) -> Any:
        del profile
        window_id = int(getattr(target, "hwnd", 0) or 0)
        if window_id <= 0:
            raise RuntimeError("target_window_not_resolved_for_capture")
        with tempfile.TemporaryDirectory(prefix="neko-galgame-mac-") as tmpdir:
            output_path = Path(tmpdir) / "window.png"
            self._capture_window(window_id=window_id, output_path=output_path)
            with Image.open(output_path) as source:
                image = source.convert("RGB")
                image.load()
        image.info["galgame_capture_backend_kind"] = self.kind
        image.info["galgame_capture_backend_detail"] = "selected"
        return image

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
