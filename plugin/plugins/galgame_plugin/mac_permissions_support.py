from __future__ import annotations

import importlib
from typing import Any


def _preflight_screen_capture_access() -> bool:
    quartz = importlib.import_module("Quartz")

    return bool(quartz.CGPreflightScreenCaptureAccess())


def _is_process_trusted() -> bool:
    application_services = importlib.import_module("ApplicationServices")

    return bool(application_services.AXIsProcessTrusted())


def _status_payload(*, granted: bool, detail: str) -> dict[str, Any]:
    return {
        "granted": bool(granted),
        "detail": str(detail or ""),
    }


def _screen_recording_status() -> dict[str, Any]:
    try:
        granted = _preflight_screen_capture_access()
    except Exception:
        return _status_payload(
            granted=False,
            detail="screen_recording_inspection_unavailable",
        )
    if granted:
        return _status_payload(granted=True, detail="granted")
    return _status_payload(
        granted=False,
        detail="screen_recording_permission_denied",
    )


def _accessibility_status() -> dict[str, Any]:
    try:
        granted = _is_process_trusted()
    except Exception:
        return _status_payload(
            granted=False,
            detail="accessibility_inspection_unavailable",
        )
    if granted:
        return _status_payload(granted=True, detail="granted")
    return _status_payload(
        granted=False,
        detail="accessibility_permission_denied",
    )


def inspect_macos_permissions() -> dict[str, Any]:
    return {
        "screen_recording": _screen_recording_status(),
        "accessibility": _accessibility_status(),
    }
