from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

from plugin.plugins.galgame_plugin import mac_permissions_support as mac_permissions


def test_inspect_macos_permissions_reports_unavailable_when_quartz_import_fails(
    monkeypatch,
) -> None:
    imported_modules: list[str] = []

    def _fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "Quartz":
            return SimpleNamespace(CGPreflightScreenCaptureAccess=lambda: True)
        if name == "ApplicationServices":
            return SimpleNamespace(AXIsProcessTrusted=lambda: True)
        return _real_import(name, globals, locals, fromlist, level)

    def _fake_import_module(name: str, package: str | None = None):
        imported_modules.append(name)
        if name == "Quartz":
            raise ModuleNotFoundError("No module named 'Quartz'")
        if name == "ApplicationServices":
            return SimpleNamespace(AXIsProcessTrusted=lambda: True)
        return _real_import_module(name, package)

    _real_import = builtins.__import__
    _real_import_module = importlib.import_module
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(mac_permissions.importlib, "import_module", _fake_import_module)

    snapshot = mac_permissions.inspect_macos_permissions()

    assert snapshot["screen_recording"] == {
        "granted": False,
        "detail": "screen_recording_inspection_unavailable",
    }
    assert snapshot["accessibility"] == {
        "granted": True,
        "detail": "granted",
    }
    assert imported_modules == ["Quartz", "ApplicationServices"]


def test_inspect_macos_permissions_reports_unavailable_when_application_services_import_fails(
    monkeypatch,
) -> None:
    imported_modules: list[str] = []

    def _fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "Quartz":
            return SimpleNamespace(CGPreflightScreenCaptureAccess=lambda: True)
        if name == "ApplicationServices":
            return SimpleNamespace(AXIsProcessTrusted=lambda: True)
        return _real_import(name, globals, locals, fromlist, level)

    def _fake_import_module(name: str, package: str | None = None):
        imported_modules.append(name)
        if name == "Quartz":
            return SimpleNamespace(CGPreflightScreenCaptureAccess=lambda: True)
        if name == "ApplicationServices":
            raise ModuleNotFoundError("No module named 'ApplicationServices'")
        return _real_import_module(name, package)

    _real_import = builtins.__import__
    _real_import_module = importlib.import_module
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(mac_permissions.importlib, "import_module", _fake_import_module)

    snapshot = mac_permissions.inspect_macos_permissions()

    assert snapshot["screen_recording"] == {
        "granted": True,
        "detail": "granted",
    }
    assert snapshot["accessibility"] == {
        "granted": False,
        "detail": "accessibility_inspection_unavailable",
    }
    assert imported_modules == ["Quartz", "ApplicationServices"]


def test_module_import_is_safe_without_framework_modules(monkeypatch) -> None:
    real_import_module = importlib.import_module
    imported_modules: list[str] = []

    def _fake_import_module(name: str, package: str | None = None):
        imported_modules.append(name)
        if name in {"Quartz", "ApplicationServices"}:
            raise AssertionError("framework imports must stay lazy")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    reloaded = importlib.reload(mac_permissions)

    assert reloaded is mac_permissions
    assert imported_modules == []
