from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

from PIL import Image

from plugin.plugins.game_companion import GameCompanionPlugin
from plugin.plugins.game_companion.core.profile_registry import ProfileRegistry
from plugin.plugins.game_companion.core.realtime import RealtimeInsightSession
from plugin.plugins.game_companion.profiles import builtin_profiles


class _Store:
    def __init__(self) -> None:
        self.values = {}

    def _read_value(self, key, default=None):
        return self.values.get(key, default)

    def _write_value(self, key, value):
        self.values[key] = value


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class _Ctx:
    plugin_id = "game_companion"
    metadata = {"id": "game_companion", "name": "Game Companion", "version": "0.1.0"}
    logger = _Logger()
    config_path = str(Path("plugin/plugins/game_companion/plugin.toml").resolve())
    bus = None
    _effective_config = {"store": {"enabled": False}}

    async def get_own_config(self, timeout=5.0):
        return {}

    async def get_own_base_config(self, timeout=5.0):
        return {}

    async def get_own_profiles_state(self, timeout=5.0):
        return {}

    async def get_own_profile_config(self, profile_name, timeout=5.0):
        return {}

    async def get_own_effective_config(self, profile_name=None, timeout=5.0):
        return self._effective_config

    async def update_own_config(self, updates, timeout=10.0):
        return {}

    async def upsert_own_profile_config(self, profile_name, config, make_active=False, timeout=10.0):
        return {}

    async def delete_own_profile_config(self, profile_name, timeout=10.0):
        return {}

    async def set_own_active_profile(self, profile_name, timeout=10.0):
        return {}

    async def query_plugins(self, filters, timeout=5.0):
        return []

    async def trigger_plugin_event(self, **kwargs):
        return None

    async def get_system_config(self, timeout=5.0):
        return {}

    async def query_memory(self, bucket_id, query, timeout=5.0):
        return []

    async def run_update(self, **kwargs):
        return None

    async def export_push(self, **kwargs):
        return None

    async def finish(self, **kwargs):
        return kwargs.get("data")

    def push_message(self, **kwargs):
        return None

    def update_status(self, status):
        pass


def _plugin() -> GameCompanionPlugin:
    plugin = object.__new__(GameCompanionPlugin)
    plugin._profiles = ProfileRegistry()
    for profile in builtin_profiles():
        plugin._profiles.register(profile)
    plugin._active_profile_id = "generic"
    plugin._realtime = RealtimeInsightSession()
    plugin._last_auto_snapshot_key = ""
    plugin.store = _Store()
    return plugin


def _payload(value):
    return value.value if hasattr(value, "value") else value


def test_plugin_manifest_entry_imports_and_collects_expected_entries() -> None:
    manifest = tomllib.loads(Path("plugin/plugins/game_companion/plugin.toml").read_text(encoding="utf-8"))
    entry = manifest["plugin"]["entry"]
    module_name, class_name = entry.split(":", 1)

    plugin_cls = getattr(importlib.import_module(module_name), class_name)
    plugin = plugin_cls(_Ctx())
    entries = plugin.collect_entries(wrap_with_hooks=False)

    assert plugin_cls is GameCompanionPlugin
    assert {
        "game_companion_status",
        "game_companion_list_profiles",
        "game_companion_select_profile",
        "game_companion_analyze_frame",
        "game_companion_ingest_frame",
        "game_companion_realtime_status",
        "game_companion_realtime_configure",
        "game_companion_save_review_snapshot",
        "game_companion_list_review_snapshots",
        "game_companion_clear_review_snapshots",
        "game_companion_training_prompt",
        "startup",
        "shutdown",
    }.issubset(entries)


def test_game_companion_entrypoints_run_offline_tft_frame_with_debug_crops(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    crops_dir = tmp_path / "crops"
    Image.new("RGB", (1920, 1080), color=(12, 24, 36)).save(screenshot)

    selected = _payload(plugin.select_profile("tft"))
    analyzed = _payload(
        plugin.analyze_frame_entry(
            profile_id="tft",
            image_path=str(screenshot),
            debug_crops_dir=str(crops_dir),
        )
    )

    assert selected == {"selected": True, "active_profile": "tft"}
    assert analyzed["success"] is True
    assert analyzed["profile"] == "tft"
    assert analyzed["source"]["type"] == "image_path"
    assert analyzed["state"]["shop_units"]
    debug_crops = analyzed["diagnostics"]["debug_crops"]
    assert debug_crops["output_dir"] == str(crops_dir.resolve())
    assert Path(debug_crops["crops"]["shop_slot_1"]).is_file()


def test_game_companion_realtime_and_review_entries_roundtrip(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(8, 16, 24)).save(screenshot)

    configured = _payload(plugin.realtime_configure(enabled=True, profile_id="tft", interval_seconds=2))
    ingested = _payload(plugin.ingest_frame(profile_id="tft", image_path=str(screenshot)))
    status = _payload(plugin.realtime_status())
    saved = _payload(plugin.save_review_snapshot(note="entrypoint smoke"))
    listed = _payload(plugin.list_review_snapshots())
    prompt = _payload(plugin.training_prompt())
    cleared = _payload(plugin.clear_review_snapshots())

    assert configured["configured"] is True
    assert ingested["result"]["success"] is True
    assert ingested["realtime"]["frame_count"] == 1
    assert status["stable_result"]["profile"] == "tft"
    assert saved["saved"] is True
    assert listed["snapshots"][0]["note"] == "entrypoint smoke"
    assert prompt["available"] is True
    assert prompt["prompt"]["type"] == "tft_training_prompt"
    assert cleared == {"cleared": True, "snapshots": []}
