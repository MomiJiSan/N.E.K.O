from __future__ import annotations

from plugin.plugins.galgame_plugin import mac_permissions_support as mac_permissions


def test_inspect_macos_permissions_reports_denied_screen_recording(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mac_permissions,
        "_preflight_screen_capture_access",
        lambda: False,
    )
    monkeypatch.setattr(
        mac_permissions,
        "_is_process_trusted",
        lambda: True,
    )

    snapshot = mac_permissions.inspect_macos_permissions()

    assert snapshot["screen_recording"] == {
        "granted": False,
        "detail": "screen_recording_permission_denied",
    }
    assert snapshot["accessibility"] == {
        "granted": True,
        "detail": "granted",
    }


def test_inspect_macos_permissions_reports_unavailable_accessibility_probe(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mac_permissions,
        "_preflight_screen_capture_access",
        lambda: True,
    )

    def _raise_runtime_error() -> bool:
        raise RuntimeError("ApplicationServices unavailable")

    monkeypatch.setattr(
        mac_permissions,
        "_is_process_trusted",
        _raise_runtime_error,
    )

    snapshot = mac_permissions.inspect_macos_permissions()

    assert snapshot["screen_recording"]["detail"] == "granted"
    assert snapshot["accessibility"] == {
        "granted": False,
        "detail": "accessibility_inspection_unavailable",
    }
