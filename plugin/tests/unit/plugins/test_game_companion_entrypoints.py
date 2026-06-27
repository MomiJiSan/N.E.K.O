from __future__ import annotations

import importlib
import json
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
        "game_companion_init_layout_calibration_workspace",
        "game_companion_capture_layout_calibration_screenshot",
        "game_companion_prepare_layout_calibration_manifest",
        "game_companion_layout_calibration_status",
        "game_companion_calibrate_layout",
        "game_companion_summarize_layout_calibration",
        "game_companion_update_layout_calibration_check",
        "game_companion_update_layout_calibration_checks",
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


def test_game_companion_init_layout_calibration_workspace_entry(tmp_path: Path) -> None:
    plugin = _plugin()
    workspace = _payload(plugin.init_layout_calibration_workspace_entry(root_dir=str(tmp_path / "workspace")))

    assert workspace["type"] == "tft_layout_calibration_workspace"
    assert Path(workspace["input_dir"]).is_dir()
    assert Path(workspace["samples_manifest_path"]).is_file()
    assert Path(workspace["readme_path"]).is_file()


def test_game_companion_layout_calibration_entry_generates_report(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    report = _payload(plugin.calibrate_layout_entry(image_paths=[str(screenshot)], output_dir=str(output_dir)))

    assert report["type"] == "tft_layout_calibration_report"
    assert report["summary"]["total"] == 1
    assert report["summary"]["successes"] == 1
    assert Path(report["report_path"]).is_file()
    assert Path(report["html_path"]).is_file()
    assert Path(report["screenshots"][0]["debug_crops"]["crops"]["gold"]).name.startswith("normal_shop__p02__gold")


def test_game_companion_prepare_layout_calibration_manifest_entry(tmp_path: Path) -> None:
    plugin = _plugin()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    screenshot = input_dir / "normal_shop_shop5_bench.png"
    manifest_path = tmp_path / "samples_manifest.json"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    manifest = _payload(
        plugin.prepare_layout_calibration_manifest_entry(
            input_dir=str(input_dir),
            output_path=str(manifest_path),
        )
    )

    assert manifest["type"] == "tft_layout_calibration_samples_manifest"
    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["samples"][0]["expected_layout"] == "normal_shop"
    assert "shop_five_units" in manifest["samples"][0]["tags"]

    report = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration"),
            samples_manifest_path=str(manifest_path),
        )
    )
    assert report["summary"]["coverage"]["layout_counts"]["normal_shop"] == 1


def test_game_companion_prepare_layout_calibration_manifest_entry_errors(tmp_path: Path) -> None:
    plugin = _plugin()

    result = _payload(plugin.prepare_layout_calibration_manifest_entry(input_dir=str(tmp_path / "missing")))

    assert result["success"] is False
    assert result["error"]["code"] == "input_dir_not_found"


def test_game_companion_layout_calibration_status_entry_inspects_inputs(tmp_path: Path) -> None:
    plugin = _plugin()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    screenshot = input_dir / "normal_shop_shop5_bench.png"
    manifest_path = tmp_path / "samples_manifest.json"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    input_status = _payload(plugin.layout_calibration_status_entry(input_dir=str(input_dir)))
    manifest = _payload(plugin.prepare_layout_calibration_manifest_entry(str(input_dir), str(manifest_path)))
    manifest_status = _payload(plugin.layout_calibration_status_entry(samples_manifest_path=str(manifest_path)))
    report = _payload(plugin.calibrate_layout_entry(output_dir=str(output_dir), samples_manifest_path=str(manifest_path)))
    report_status = _payload(plugin.layout_calibration_status_entry(report_path=report["report_path"]))

    assert input_status["input_dir"]["exists"] is True
    assert input_status["input_dir"]["sample_count"] == 1
    assert manifest["type"] == "tft_layout_calibration_samples_manifest"
    assert manifest_status["samples_manifest"]["valid"] is True
    assert manifest_status["samples_manifest"]["sample_count"] == 1
    assert report_status["report"]["valid"] is True
    assert report_status["report"]["ready_for_recognition"] is False
    assert report_status["next_steps"]


def test_game_companion_layout_calibration_entry_accepts_structured_samples(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "normal_shop.png"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    report = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(output_dir),
            samples=[
                {
                    "image_path": str(screenshot),
                    "expected_layout": "normal_shop",
                    "tags": ["shop_open", "shop_five_units"],
                    "label": "normal shop smoke",
                }
            ],
        )
    )

    assert report["summary"]["coverage"]["layout_counts"]["normal_shop"] == 1
    assert report["summary"]["coverage"]["tag_counts"]["shop_open"] == 1
    assert report["screenshots"][0]["label"] == "normal shop smoke"


def test_game_companion_layout_calibration_entry_requires_images_or_samples(tmp_path: Path) -> None:
    plugin = _plugin()

    result = _payload(plugin.calibrate_layout_entry(output_dir=str(tmp_path / "layout_calibration")))

    assert result["success"] is False
    assert result["error"]["code"] == "missing_images"


def test_game_companion_layout_calibration_entry_returns_stable_manifest_errors(tmp_path: Path) -> None:
    plugin = _plugin()
    bad_json = tmp_path / "bad_samples_manifest.json"
    invalid_manifest = tmp_path / "invalid_samples_manifest.json"
    bad_json.write_text("{not json", encoding="utf-8")
    invalid_manifest.write_text("{}", encoding="utf-8")

    missing = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration_1"),
            samples_manifest_path=str(tmp_path / "missing.json"),
        )
    )
    decoded = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration_2"),
            samples_manifest_path=str(bad_json),
        )
    )
    invalid = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration_3"),
            samples_manifest_path=str(invalid_manifest),
        )
    )

    assert missing["success"] is False
    assert missing["error"]["code"] == "samples_manifest_not_found"
    assert decoded["success"] is False
    assert decoded["error"]["code"] == "samples_manifest_decode_failed"
    assert invalid["success"] is False
    assert invalid["error"]["code"] == "invalid_samples_manifest"


def test_game_companion_layout_calibration_entry_rejects_conflicting_sample_inputs(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "normal_shop.png"
    manifest_path = tmp_path / "samples_manifest.json"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)
    manifest_path.write_text(
        json.dumps(
            {
                "type": "tft_layout_calibration_samples_manifest",
                "schema_version": 1,
                "samples": [{"image_path": str(screenshot)}],
            }
        ),
        encoding="utf-8",
    )

    result = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration"),
            image_paths=[str(screenshot)],
            samples_manifest_path=str(manifest_path),
        )
    )

    assert result["success"] is False
    assert result["error"]["code"] == "invalid_samples_manifest"
    assert "conflicting sample inputs" in result["error"]["message"]


def test_game_companion_layout_calibration_entry_returns_stable_write_error(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    output_file = tmp_path / "not_a_directory"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)
    output_file.write_text("occupied", encoding="utf-8")

    result = _payload(plugin.calibrate_layout_entry(image_paths=[str(screenshot)], output_dir=str(output_file)))

    assert result["success"] is False
    assert result["error"]["code"] == "output_write_failed"


def test_game_companion_layout_calibration_summary_entry_reads_report(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    report = _payload(plugin.calibrate_layout_entry(image_paths=[str(screenshot)], output_dir=str(output_dir)))
    report["screenshots"][0]["manual_checks"][0]["status"] = "pass"
    report["screenshots"][0]["manual_checks"][1]["status"] = "fail"

    summary = _payload(plugin.summarize_layout_calibration_entry(report=report))

    assert summary["status_counts"]["pass"] == 1
    assert summary["status_counts"]["fail"] == 1
    assert summary["ready_for_region_tuning"] is True

    missing = _payload(plugin.summarize_layout_calibration_entry())
    assert missing["success"] is False
    assert missing["error"]["code"] == "missing_report"


def test_game_companion_layout_calibration_update_check_entry_refreshes_report(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)
    report = _payload(plugin.calibrate_layout_entry(image_paths=[str(screenshot)], output_dir=str(output_dir)))

    updated = _payload(
        plugin.update_layout_calibration_check_entry(
            report_path=report["report_path"],
            screenshot_index=1,
            check_id="gold_clean",
            status="pass",
            note="Gold crop is centered.",
        )
    )
    invalid = _payload(
        plugin.update_layout_calibration_check_entry(
            report_path=report["report_path"],
            screenshot_index=1,
            check_id="gold_clean",
            status="maybe",
        )
    )

    assert updated["updated"] is True
    assert updated["annotation_summary"]["status_counts"]["pass"] == 1
    assert invalid["success"] is False
    assert invalid["error"]["code"] == "invalid_report"


def test_game_companion_layout_calibration_batch_update_checks_entry(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    output_dir = tmp_path / "layout_calibration"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)
    report = _payload(plugin.calibrate_layout_entry(image_paths=[str(screenshot)], output_dir=str(output_dir)))

    updated = _payload(
        plugin.update_layout_calibration_checks_entry(
            report_path=report["report_path"],
            updates=[
                {"screenshot_index": 1, "check_id": "gold_clean", "status": "pass", "note": "Gold crop is centered."},
                {"screenshot_index": 1, "check_id": "level_exp_clean", "status": "pass", "note": "Level crop is centered."},
            ],
        )
    )

    assert updated["updated"] is True
    assert len(updated["updates"]) == 2
    assert updated["annotation_summary"]["status_counts"]["pass"] == 2


def test_game_companion_layout_calibration_summary_entry_returns_stable_report_errors(tmp_path: Path) -> None:
    plugin = _plugin()
    missing_path = tmp_path / "missing_report.json"
    bad_json = tmp_path / "bad_report.json"
    invalid_report = tmp_path / "invalid_report.json"
    bad_json.write_text("{not json", encoding="utf-8")
    invalid_report.write_text("[]", encoding="utf-8")

    missing = _payload(plugin.summarize_layout_calibration_entry(report_path=str(missing_path)))
    decoded = _payload(plugin.summarize_layout_calibration_entry(report_path=str(bad_json)))
    invalid = _payload(plugin.summarize_layout_calibration_entry(report_path=str(invalid_report)))

    assert missing["success"] is False
    assert missing["error"]["code"] == "report_not_found"
    assert decoded["success"] is False
    assert decoded["error"]["code"] == "report_decode_failed"
    assert invalid["success"] is False
    assert invalid["error"]["code"] == "invalid_report"


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
