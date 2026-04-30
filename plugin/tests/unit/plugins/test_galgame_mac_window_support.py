from __future__ import annotations

from plugin.plugins.galgame_plugin import mac_window_support


def test_list_native_candidate_windows_normalizes_frontmost_window(monkeypatch) -> None:
    monkeypatch.setattr(
        mac_window_support,
        "_window_records",
        lambda: [
            {
                "kCGWindowNumber": 901,
                "kCGWindowOwnerPID": 4242,
                "kCGWindowOwnerName": "SteamGame",
                "kCGWindowName": "Episode 1",
                "kCGWindowBounds": {"X": 10, "Y": 20, "Width": 1280, "Height": 720},
            }
        ],
    )
    monkeypatch.setattr(mac_window_support, "_frontmost_pid", lambda: 4242)

    windows = mac_window_support.list_native_candidate_windows()

    assert windows == [
        {
            "hwnd": 901,
            "pid": 4242,
            "process_name": "SteamGame",
            "title": "Episode 1",
            "width": 1280,
            "height": 720,
            "area": 921600,
            "is_foreground": True,
            "is_minimized": False,
            "class_name": "CGWindow",
            "exe_path": "",
        }
    ]


def test_list_native_candidate_windows_uses_non_foreground_fallback_when_frontmost_lookup_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mac_window_support,
        "_window_records",
        lambda: [
            {
                "kCGWindowNumber": 901,
                "kCGWindowOwnerPID": 4242,
                "kCGWindowOwnerName": "SteamGame",
                "kCGWindowName": "Episode 1",
                "kCGWindowBounds": {"Width": 1280, "Height": 720},
            }
        ],
    )

    def _raise_frontmost_pid() -> int:
        raise RuntimeError("AppKit unavailable")

    monkeypatch.setattr(mac_window_support, "_frontmost_pid", _raise_frontmost_pid)

    windows = mac_window_support.list_native_candidate_windows()

    assert windows[0]["is_foreground"] is False


def test_list_native_candidate_windows_returns_empty_when_window_inventory_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mac_window_support, "_frontmost_pid", lambda: 4242)

    def _raise_window_records():
        raise RuntimeError("Quartz unavailable")

    monkeypatch.setattr(mac_window_support, "_window_records", _raise_window_records)

    windows = mac_window_support.list_native_candidate_windows()

    assert windows == []
