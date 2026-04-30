from __future__ import annotations

from typing import Any


def _window_records() -> list[dict[str, Any]]:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGNullWindowID,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionOnScreenOnly,
    )

    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    return list(CGWindowListCopyWindowInfo(options, kCGNullWindowID) or [])


def _frontmost_pid() -> int:
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return int(app.processIdentifier()) if app is not None else 0


def list_native_candidate_windows() -> list[dict[str, Any]]:
    frontmost_pid = _frontmost_pid()
    results: list[dict[str, Any]] = []
    for record in _window_records():
        bounds = dict(record.get("kCGWindowBounds") or {})
        width = int(bounds.get("Width") or 0)
        height = int(bounds.get("Height") or 0)
        pid = int(record.get("kCGWindowOwnerPID") or 0)
        process_name = str(record.get("kCGWindowOwnerName") or "").strip()
        title = str(record.get("kCGWindowName") or "").strip()
        if pid <= 0 or width <= 0 or height <= 0 or not process_name:
            continue
        results.append(
            {
                "hwnd": int(record.get("kCGWindowNumber") or 0),
                "pid": pid,
                "process_name": process_name,
                "title": title,
                "width": width,
                "height": height,
                "area": max(0, width * height),
                "is_foreground": pid == frontmost_pid,
                "is_minimized": False,
                "class_name": "CGWindow",
                "exe_path": "",
            }
        )
    return results
