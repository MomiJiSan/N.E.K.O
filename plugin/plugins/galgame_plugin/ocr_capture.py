from __future__ import annotations

import ctypes
import time
from typing import Any, Protocol


CAPTURE_BACKEND_AUTO = "auto"
CAPTURE_BACKEND_DXCAM = "dxcam"
CAPTURE_BACKEND_IMAGEGRAB = "imagegrab"
CAPTURE_BACKEND_PRINTWINDOW = "printwindow"

_BACKGROUND_HASH_BOTTOM_INSET_RATIO = 0.45
_BACKGROUND_SCENE_HASH_SIZE = 8
_DXCAM_GRAB_RETRY_ATTEMPTS = 2
_DXCAM_GRAB_RETRY_DELAY_SECONDS = 0.05


class CaptureBackend(Protocol):
    def is_available(self) -> bool: ...

    def describe_target(self, target: Any) -> str: ...

    def capture_frame(self, target: Any, profile: Any) -> Any: ...


def _perceptual_hash_image(frame: Any, *, size: int = _BACKGROUND_SCENE_HASH_SIZE) -> str:
    if frame is None:
        return ""
    try:
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image)
        image = frame.convert("L").resize((size, size), resampling.BILINEAR)
        pixels = list(image.getdata())
        if not pixels:
            return ""
        average = sum(int(pixel) for pixel in pixels) / len(pixels)
        bits = "".join("1" if int(pixel) >= average else "0" for pixel in pixels)
        return f"{int(bits, 2):016x}"
    except Exception:
        return ""


def _target_window_rect(target: Any) -> tuple[int, int, int, int]:
    import win32gui

    rect = win32gui.GetWindowRect(target.hwnd)
    width = int(rect[2] - rect[0])
    height = int(rect[3] - rect[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid window dimensions: {width}x{height}")
    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))


def _run_with_thread_dpi_awareness(fn) -> tuple[int, int, int, int]:
    user32 = getattr(ctypes, "windll", None)
    user32 = getattr(user32, "user32", None) if user32 is not None else None
    set_context = getattr(user32, "SetThreadDpiAwarenessContext", None) if user32 is not None else None
    if not callable(set_context):
        return fn()
    old_context = None
    try:
        old_context = set_context(ctypes.c_void_p(-4))
    except Exception:
        old_context = None
    try:
        return fn()
    finally:
        if old_context:
            try:
                set_context(old_context)
            except Exception:
                pass


def _target_client_rect(target: Any) -> tuple[int, int, int, int]:
    import win32gui

    def _read_rect() -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetClientRect(target.hwnd)
        screen_left, screen_top = win32gui.ClientToScreen(target.hwnd, (left, top))
        screen_right, screen_bottom = win32gui.ClientToScreen(target.hwnd, (right, bottom))
        return (int(screen_left), int(screen_top), int(screen_right), int(screen_bottom))

    rect = _run_with_thread_dpi_awareness(_read_rect)
    width = int(rect[2] - rect[0])
    height = int(rect[3] - rect[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid client dimensions: {width}x{height}")
    return rect


def _require_visible_capture_target(target: Any, *, backend_kind: str) -> None:
    if not target.hwnd:
        raise RuntimeError(f"{backend_kind}: target_window_not_resolved_for_capture")
    try:
        import win32gui

        if not win32gui.IsWindow(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_invalid_for_capture")
        if not win32gui.IsWindowVisible(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_not_visible_for_capture")
        if win32gui.IsIconic(target.hwnd):
            raise RuntimeError(f"{backend_kind}: target_window_minimized_for_capture")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{backend_kind}: target_window_visibility_check_failed: {exc}") from exc


def _crop_window_image(
    image: Any,
    *,
    window_rect: tuple[int, int, int, int],
    profile: Any,
    backend_kind: str,
    backend_detail: str,
) -> Any:
    width = int(window_rect[2] - window_rect[0])
    height = int(window_rect[3] - window_rect[1])
    left = int(width * profile.left_inset_ratio)
    right = int(width * (1.0 - profile.right_inset_ratio))
    top = int(height * profile.top_ratio)
    bottom = int(height * (1.0 - profile.bottom_inset_ratio))

    left = max(0, min(left, width))
    right = max(left, min(right, width))
    top = max(0, min(top, height))
    bottom = max(top, min(bottom, height))

    crop_w = right - left
    crop_h = bottom - top
    if crop_w < 10 or crop_h < 10:
        raise RuntimeError(f"Crop region too small: {crop_w}x{crop_h}")

    background_bottom = max(
        0,
        min(int(height * (1.0 - _BACKGROUND_HASH_BOTTOM_INSET_RATIO)), height),
    )
    source_background_hash = ""
    if background_bottom >= 10:
        source_background_hash = _perceptual_hash_image(
            image.crop((0, 0, width, background_bottom))
        )

    cropped = image.crop((left, top, right, bottom))
    cropped.info["galgame_bounds_coordinate_space"] = "capture"
    cropped.info["galgame_source_size"] = {"width": float(crop_w), "height": float(crop_h)}
    cropped.info["galgame_source_background_hash"] = source_background_hash
    cropped.info["galgame_capture_rect"] = {
        "left": float(window_rect[0] + left),
        "top": float(window_rect[1] + top),
        "right": float(window_rect[0] + right),
        "bottom": float(window_rect[1] + bottom),
    }
    cropped.info["galgame_window_rect"] = {
        "left": float(window_rect[0]),
        "top": float(window_rect[1]),
        "right": float(window_rect[2]),
        "bottom": float(window_rect[3]),
    }
    cropped.info["galgame_capture_backend_kind"] = backend_kind
    cropped.info["galgame_capture_backend_detail"] = backend_detail
    return cropped


class ImageGrabCaptureBackend:
    kind = CAPTURE_BACKEND_IMAGEGRAB

    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def is_available(self) -> bool:
        try:
            import win32gui
            from PIL import ImageGrab

            return bool(win32gui and ImageGrab)
        except ImportError:
            return False

    def describe_target(self, target: Any) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: Any, profile: Any) -> Any:
        from PIL import ImageGrab

        _require_visible_capture_target(target, backend_kind=self.kind)
        rect = _target_window_rect(target)
        image = ImageGrab.grab(bbox=rect, all_screens=True).convert("RGB")
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected",
        )


class PrintWindowCaptureBackend:
    kind = CAPTURE_BACKEND_PRINTWINDOW

    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def is_available(self) -> bool:
        try:
            import win32con
            import win32gui
            import win32ui

            return bool(win32gui and win32ui and win32con)
        except ImportError:
            return False

    def describe_target(self, target: Any) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: Any, profile: Any) -> Any:
        _require_visible_capture_target(target, backend_kind=self.kind)
        rect = _target_window_rect(target)
        image = self._capture_full_window(target.hwnd, rect)
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected_legacy_fallback",
        )

    @staticmethod
    def _capture_full_window(hwnd: int, rect: tuple[int, int, int, int]) -> Any:
        import win32con
        import win32gui
        import win32ui
        from PIL import Image

        width = int(rect[2] - rect[0])
        height = int(rect[3] - rect[1])
        hdc = win32gui.GetWindowDC(hwnd)
        if not hdc:
            raise RuntimeError("Failed to get window DC")

        bmp = None
        mem_dc = None
        hdc_mem = None
        try:
            hdc_mem = win32ui.CreateDCFromHandle(hdc)
            mem_dc = hdc_mem.CreateCompatibleDC()

            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(hdc_mem, width, height)
            mem_dc.SelectObject(bmp)

            pw_renderfullcontent = 3
            success = ctypes.windll.user32.PrintWindow(
                hwnd,
                mem_dc.GetSafeHdc(),
                pw_renderfullcontent,
            )
            if not success:
                mem_dc.BitBlt((0, 0), (width, height), hdc_mem, (0, 0), win32con.SRCCOPY)

            bmp_info = bmp.GetInfo()
            bmp_str = bmp.GetBitmapBits(True)
            image = Image.frombuffer(
                "RGB",
                (bmp_info["bmWidth"], bmp_info["bmHeight"]),
                bmp_str,
                "raw",
                "BGRX",
                0,
                1,
            )
        finally:
            if mem_dc is not None:
                mem_dc.DeleteDC()
            if hdc_mem is not None:
                hdc_mem.DeleteDC()
            if bmp is not None:
                win32gui.DeleteObject(bmp.GetHandle())
            win32gui.ReleaseDC(hwnd, hdc)
        return image


class DxcamCaptureBackend:
    kind = CAPTURE_BACKEND_DXCAM

    def __init__(self, *, logger=None) -> None:
        self._logger = logger
        self._camera = None
        self._last_create_error = ""

    def is_available(self) -> bool:
        try:
            import dxcam

            return bool(dxcam)
        except ImportError:
            return False

    def describe_target(self, target: Any) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def _camera_instance(self):
        if self._camera is not None:
            return self._camera
        import dxcam

        try:
            self._camera = dxcam.create(output_color="RGB")
        except Exception as exc:
            self._last_create_error = str(exc)
            raise RuntimeError(f"dxcam_create_failed: {exc}") from exc
        if self._camera is None:
            raise RuntimeError("dxcam_create_failed: returned None")
        return self._camera

    def _reset_camera(self) -> None:
        camera = self._camera
        self._camera = None
        stop = getattr(camera, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                if self._logger is not None:
                    self._logger.debug("ocr_reader dxcam camera stop failed during reset: %s", exc, exc_info=True)

    def capture_frame(self, target: Any, profile: Any) -> Any:
        from PIL import Image

        _require_visible_capture_target(target, backend_kind=self.kind)
        rect = _target_client_rect(target)
        frame = None
        for attempt in range(_DXCAM_GRAB_RETRY_ATTEMPTS + 1):
            camera = self._camera_instance()
            frame = camera.grab(region=rect)
            if frame is not None:
                break
            self._reset_camera()
            if attempt < _DXCAM_GRAB_RETRY_ATTEMPTS:
                time.sleep(_DXCAM_GRAB_RETRY_DELAY_SECONDS)
        if frame is None:
            raise RuntimeError(
                f"dxcam_grab_returned_none_after_{_DXCAM_GRAB_RETRY_ATTEMPTS + 1}_attempts"
            )
        image = Image.fromarray(frame).convert("RGB")
        return _crop_window_image(
            image,
            window_rect=rect,
            profile=profile,
            backend_kind=self.kind,
            backend_detail="selected_client_rect",
        )


class Win32CaptureBackend:
    def __init__(self, *, logger=None, selection: str = CAPTURE_BACKEND_AUTO) -> None:
        self._logger = logger
        self.selection = str(selection or CAPTURE_BACKEND_AUTO).strip().lower()
        self._backends = self._build_backends()
        self.last_backend_kind = ""
        self.last_backend_detail = ""

    def _build_backends(self) -> list[CaptureBackend]:
        imagegrab = ImageGrabCaptureBackend(logger=self._logger)
        printwindow = PrintWindowCaptureBackend(logger=self._logger)
        dxcam = DxcamCaptureBackend(logger=self._logger)
        if self.selection == CAPTURE_BACKEND_DXCAM:
            return [dxcam, imagegrab, printwindow]
        if self.selection == CAPTURE_BACKEND_IMAGEGRAB:
            return [imagegrab]
        if self.selection == CAPTURE_BACKEND_PRINTWINDOW:
            return [printwindow]
        return [dxcam, imagegrab, printwindow]

    def is_available(self) -> bool:
        if self.selection not in {CAPTURE_BACKEND_AUTO, CAPTURE_BACKEND_DXCAM}:
            return bool(self._backends) and self._backends[0].is_available()
        return any(backend.is_available() for backend in self._backends)

    def describe_target(self, target: Any) -> str:
        return f"{target.process_name}({target.pid}) {target.title}"

    def capture_frame(self, target: Any, profile: Any) -> Any:
        errors: list[str] = []
        for backend in self._backends:
            kind = str(getattr(backend, "kind", backend.__class__.__name__))
            if not backend.is_available():
                errors.append(f"{kind}_unavailable")
                continue
            try:
                frame = backend.capture_frame(target, profile)
                self.last_backend_kind = kind
                frame_info = getattr(frame, "info", None)
                frame_backend_detail = (
                    str(frame_info.get("galgame_capture_backend_detail") or "")
                    if isinstance(frame_info, dict)
                    else ""
                )
                self.last_backend_detail = frame_backend_detail or (
                    "dxcam_unavailable_fallback"
                    if kind != CAPTURE_BACKEND_DXCAM and "dxcam_unavailable" in errors
                    else "dxcam_failed_fallback"
                    if kind != CAPTURE_BACKEND_DXCAM
                    and any(error.startswith("dxcam_failed:") for error in errors)
                    else "selected"
                )
                if isinstance(frame_info, dict):
                    frame_info["galgame_capture_backend_kind"] = kind
                    frame_info["galgame_capture_backend_detail"] = self.last_backend_detail
                return frame
            except Exception as exc:
                errors.append(f"{kind}_failed:{exc}")
                if self._logger is not None:
                    self._logger.debug(
                        "ocr_reader capture backend %s failed while selection=%s: %s",
                        kind,
                        self.selection,
                        exc,
                        exc_info=True,
                    )
                if any(
                    marker in str(exc)
                    for marker in (
                        "target_window_not_resolved_for_capture",
                        "target_window_invalid_for_capture",
                        "target_window_not_visible_for_capture",
                        "target_window_minimized_for_capture",
                    )
                ):
                    raise
                if self.selection != CAPTURE_BACKEND_AUTO:
                    raise
                continue
        if self.selection != CAPTURE_BACKEND_AUTO:
            raise RuntimeError(
                f"{self.selection}: capture_backend_unavailable"
                + (f": {'; '.join(errors)}" if errors else "")
            )
        raise RuntimeError("; ".join(errors) or "capture_backend_unavailable")
