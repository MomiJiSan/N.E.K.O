from __future__ import annotations

from fractions import Fraction
import importlib
import json
from pathlib import Path
import sys
import tomllib
import types

from PIL import Image

import plugin.plugins.game_companion as game_companion_module
from plugin.plugins.game_companion import GameCompanionPlugin
from plugin.plugins.game_companion.core.local_vision import get_default_local_vision_backend, reset_default_local_vision_backend
from plugin.plugins.game_companion.core.profile_registry import ProfileRegistry
from plugin.plugins.game_companion.core.profile_registry import ProfileMetadata
from plugin.plugins.game_companion.core.realtime import RealtimeInsightSession
from plugin.plugins.game_companion.profiles import builtin_profiles
from plugin.plugins.game_companion.safety import Capability, CapabilityGate, GameType, RuntimeMode


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
    plugin.logger = _Logger()
    return plugin


def _payload(value):
    return value.value if hasattr(value, "value") else value


def test_plugin_manifest_entry_imports_and_collects_expected_entries() -> None:
    manifest = tomllib.loads(Path("plugin/plugins/game_companion/plugin.toml").read_text(encoding="utf-8"))
    entry = manifest["plugin"]["entry"]
    local_vision = manifest["game_companion"]["local_vision"]
    module_name, class_name = entry.split(":", 1)

    plugin_cls = getattr(importlib.import_module(module_name), class_name)
    plugin = plugin_cls(_Ctx())
    entries = plugin.collect_entries(wrap_with_hooks=False)

    assert plugin_cls is GameCompanionPlugin
    assert local_vision["classifier_enabled"] is False
    assert local_vision["classifier_model_path"] == ""
    assert local_vision["classifier_labels_path"] == ""
    assert local_vision["classifier_input_size"] == [224, 224]
    assert {
        "game_companion_status",
        "game_companion_list_profiles",
        "game_companion_select_profile",
        "game_companion_analyze_frame",
        "game_companion_recognize_tft_frame",
        "game_companion_build_tft_recognition_report",
        "game_companion_build_tft_video_state_report",
        "game_companion_tft_normal_shop_smoke",
        "game_companion_init_layout_calibration_workspace",
        "game_companion_capture_layout_calibration_screenshot",
        "game_companion_extract_layout_calibration_video_frames",
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
        "game_companion_neko_context",
        "game_companion_list_neko_context_queue",
        "game_companion_dequeue_neko_context",
        "startup",
        "shutdown",
    }.issubset(entries)


def test_configure_local_vision_backend_registers_onnx_classifier(tmp_path: Path) -> None:
    class _Input:
        name = "pixels"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def run(self, _output_names, _feeds):
            return [[[0.1, 1.9]]]

    model_path = tmp_path / "screen_classifier.onnx"
    labels_path = tmp_path / "labels.json"
    screenshot = tmp_path / "frame.png"
    model_path.write_bytes(b"fake")
    labels_path.write_text('{"labels": ["loading", "shop"]}', encoding="utf-8")
    Image.new("RGB", (640, 360), color=(80, 90, 100)).save(screenshot)
    plugin = _plugin()

    try:
        status = plugin._configure_local_vision_backend(
            {
                "local_vision": {
                    "classifier_enabled": True,
                    "classifier_model_path": str(model_path),
                    "classifier_labels_path": str(labels_path),
                    "classifier_model_name": "test-startup-classifier",
                    "classifier_input_size": [32, 32],
                }
            },
            session_factory=lambda _path: _Session(),
        )
        payload = _payload(plugin.analyze_frame_entry(profile_id="generic", image_path=str(screenshot)))
    finally:
        reset_default_local_vision_backend()

    assert status == {"classifier": "registered", "detector": "not_configured"}
    assert payload["vision"]["scene"]["label"] == "shop"
    assert payload["vision"]["scene"]["model_name"] == "test-startup-classifier"
    assert payload["vision"]["diagnostics"]["analyzers"]["classifier"]["status"] == "ready"


def test_configure_local_vision_backend_resets_when_disabled() -> None:
    plugin = _plugin()
    reset_default_local_vision_backend()

    status = plugin._configure_local_vision_backend({"local_vision": {"classifier_enabled": False}})

    assert status == {"classifier": "disabled", "detector": "not_configured"}
    assert get_default_local_vision_backend() is None


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


def test_game_companion_recognize_tft_frame_entry_returns_structured_state(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft_shop.png"
    Image.new("RGB", (1920, 1080), color=(12, 24, 36)).save(screenshot)

    recognized = _payload(
        plugin.recognize_tft_frame_entry(
            image_path=str(screenshot),
            expected_layout="normal_shop",
        )
    )
    combat = _payload(
        plugin.recognize_tft_frame_entry(
            image_path=str(screenshot),
            expected_layout="combat",
        )
    )
    missing = _payload(plugin.recognize_tft_frame_entry(image_path=str(tmp_path / "missing.png")))

    assert recognized["type"] == "tft_recognition_result"
    assert recognized["success"] is True
    assert recognized["layout"] == "normal_shop"
    assert recognized["state"]["type"] == "tft_frame_state"
    assert recognized["state"]["game"] == "tft"
    assert recognized["state"]["layout"] == "normal_shop"
    assert recognized["state"]["shop"]["slot_count"] == 5
    assert len(recognized["shop"]) == 5
    assert combat["layout"] == "combat"
    assert combat["state"]["layout"] == "combat"
    assert combat["state"]["shop"] is None
    assert combat["state"]["combat"] == {"status": "observed", "details": []}
    assert combat["shop"] == []
    assert missing["success"] is False
    assert missing["error"]["code"] == "image_read_failed"
    assert missing["state"]["readiness"] == "blocked"


def test_game_companion_build_tft_recognition_report_entry_writes_report(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft_shop.png"
    Image.new("RGB", (1920, 1080), color=(12, 24, 36)).save(screenshot)
    calibration_report = {
        "type": "tft_layout_calibration_report",
        "screenshots": [
            {
                "index": 1,
                "image_path": str(screenshot),
                "expected_layout": "normal_shop",
                "label": "shop",
            }
        ],
    }
    report_path = tmp_path / "calibration_report.json"
    report_path.write_text(json.dumps(calibration_report), encoding="utf-8")

    report = _payload(
        plugin.build_tft_recognition_report_entry(
            calibration_report_path=str(report_path),
            output_dir=str(tmp_path / "recognition"),
        )
    )

    assert report["type"] == "tft_recognition_report"
    assert report["report_version"] == "recognition_report_v1"
    assert report["summary"]["total"] == 1
    assert Path(report["report_path"]).is_file()
    assert Path(report["summary_path"]).is_file()


def test_game_companion_build_tft_video_state_report_entry_reports_missing_video(tmp_path: Path) -> None:
    plugin = _plugin()

    result = _payload(
        plugin.build_tft_video_state_report_entry(
            video_path=str(tmp_path / "missing.mp4"),
            output_dir=str(tmp_path / "runtime_state_v1"),
        )
    )

    assert result["success"] is False
    assert result["error"]["code"] == "video_not_found"


def test_game_companion_tft_normal_shop_smoke_entry_writes_report(tmp_path: Path, monkeypatch) -> None:
    plugin = _plugin()
    plugin._active_profile_id = "tft"
    video = tmp_path / "match.mp4"
    video.write_bytes(b"fake")

    def fake_smoke_report(video_path: str, *, output_dir: str | None = None):
        report_path = Path(output_dir or tmp_path) / "tft_normal_shop_smoke_v1.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "type": "tft_normal_shop_smoke_report",
            "report_version": "tft_normal_shop_smoke_v1",
            "success": True,
            "video_path": str(Path(video_path).resolve()),
            "report_path": str(report_path.resolve()),
            "pass": True,
            "failures": [],
            "normal_shop": {"ready_rate": 1.0},
            "mixed": {"non_shop_source_slots": 0},
            "overlay": {"shop_payloads": 0},
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(game_companion_module, "build_tft_normal_shop_smoke_report", fake_smoke_report)

    result = _payload(
        plugin.tft_normal_shop_smoke_entry(
            video_path=str(video),
            output_dir=str(tmp_path / "smoke"),
        )
    )

    assert result["success"] is True
    assert result["pass"] is True
    assert result["report_version"] == "tft_normal_shop_smoke_v1"
    assert Path(result["report_path"]).is_file()


def test_game_companion_tft_recognition_entries_require_vision_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="recognition_no_vision",
            display_name="Recognition No Vision",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    plugin._active_profile_id = "recognition_no_vision"
    screenshot = tmp_path / "tft_shop.png"
    report_path = tmp_path / "calibration_report.json"
    Image.new("RGB", (1920, 1080), color=(12, 24, 36)).save(screenshot)
    report_path.write_text(json.dumps({"type": "tft_layout_calibration_report", "screenshots": []}), encoding="utf-8")

    recognized = _payload(plugin.recognize_tft_frame_entry(image_path=str(screenshot)))
    report = _payload(
        plugin.build_tft_recognition_report_entry(
            calibration_report_path=str(report_path),
            output_dir=str(tmp_path / "recognition"),
        )
    )
    video_report = _payload(plugin.build_tft_video_state_report_entry(video_path=str(tmp_path / "missing.mp4")))
    smoke_report = _payload(plugin.tft_normal_shop_smoke_entry(video_path=str(tmp_path / "missing.mp4")))

    assert recognized["success"] is False
    assert recognized["error"]["code"] == "capability_not_allowed"
    assert recognized["error"]["capability"] == "vision_classify"
    assert report["success"] is False
    assert report["error"]["code"] == "capability_not_allowed"
    assert report["error"]["capability"] == "vision_classify"
    assert video_report["success"] is False
    assert video_report["error"]["code"] == "capability_not_allowed"
    assert video_report["error"]["capability"] == "vision_classify"
    assert smoke_report["success"] is False
    assert smoke_report["error"]["code"] == "capability_not_allowed"
    assert smoke_report["error"]["capability"] == "vision_classify"


def test_game_companion_entrypoint_analyzes_generic_frame(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "generic.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    analyzed = _payload(plugin.analyze_frame_entry(profile_id="generic", image_path=str(screenshot)))

    assert analyzed["success"] is True
    assert analyzed["profile"] == "generic"
    assert analyzed["state"] == {}
    assert analyzed["vision"]["schema_version"] == 1
    assert analyzed["vision"]["scene"]["label"] == "unknown"


def test_game_companion_entrypoint_forwards_source_context(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "generic_video_frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    source_context = {
        "type": "video_frame",
        "profile_id": "generic",
        "video_path": str((tmp_path / "match.mp4").resolve()),
        "ordinal": 2,
        "frame_index": 90,
        "timestamp_seconds": 9.0,
    }

    analyzed = _payload(
        plugin.analyze_frame_entry(
            profile_id="generic",
            image_path=str(screenshot),
            source_context=source_context,
        )
    )

    assert analyzed["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}
    assert analyzed["vision"]["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}


def test_game_companion_entrypoint_plans_vlm_fallback_without_external_call(tmp_path: Path) -> None:
    plugin = _plugin()
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(12, 24, 36)).save(screenshot)

    analyzed = _payload(plugin.analyze_frame_entry(profile_id="tft", image_path=str(screenshot), vlm_requested=True))

    assert analyzed["success"] is True
    assert analyzed["vision"]["diagnostics"]["vlm_fallback"]["status"] == "planned"
    assert analyzed["vision"]["diagnostics"]["vlm_fallback"]["reason"] == "user_requested"
    assert analyzed["vision"]["privacy"]["external_model_calls"] is False
    assert analyzed["vision"]["model_calls"] == []


def test_game_companion_runtime_guard_reports_denied_type_d_capability() -> None:
    plugin = _plugin()

    decision = plugin._require_capability("tft", Capability.INPUT_CONTROL)

    assert decision["success"] is False
    assert decision["error"]["code"] == "capability_denied"
    assert decision["error"]["capability"] == "input_control"
    assert decision["error"]["profile_id"] == "tft"
    assert decision["error"]["game_type"] == "online_competitive"
    assert decision["error"]["runtime_mode"] == "online"


def test_type_d_hard_denies_unsafe_capability_even_if_profile_gate_allows() -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="unsafe_tft",
            display_name="Unsafe TFT",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capability_gate=CapabilityGate(
                allowed=(Capability.SCREEN_OBSERVE, Capability.INPUT_CONTROL),
                denied=(),
            ),
            capabilities=(Capability.SCREEN_OBSERVE, Capability.INPUT_CONTROL),
        )
    )

    decision = plugin._require_capability("unsafe_tft", Capability.INPUT_CONTROL)

    assert decision["success"] is False
    assert decision["error"]["code"] == "capability_denied"
    assert decision["error"]["capability"] == "input_control"


def test_game_companion_analyze_frame_entry_rejects_profile_without_vision_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="watch_only",
            display_name="Watch Only",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    screenshot = tmp_path / "watch_only.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    analyzed = _payload(plugin.analyze_frame_entry(profile_id="watch_only", image_path=str(screenshot)))

    assert analyzed["success"] is False
    assert analyzed["error"]["code"] == "capability_not_allowed"
    assert analyzed["error"]["capability"] == "vision_classify"


def test_game_companion_neko_context_entry_respects_active_profile_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="no_neko_context",
            display_name="No NEKO Context",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE, Capability.VISION_CLASSIFY),
        )
    )
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    plugin._active_profile_id = "no_neko_context"
    plugin._realtime.last_result = _payload(plugin.analyze_frame_entry(profile_id="generic", image_path=str(screenshot)))

    context = _payload(plugin.neko_context())

    assert context["success"] is False
    assert context["error"]["code"] == "capability_not_allowed"
    assert context["error"]["capability"] == "neko_context"


def test_game_companion_ingest_frame_respects_profile_vision_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="ingest_watch_only",
            display_name="Ingest Watch Only",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    screenshot = tmp_path / "ingest.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    ingested = _payload(plugin.ingest_frame(profile_id="ingest_watch_only", image_path=str(screenshot)))

    assert ingested["result"]["success"] is False
    assert ingested["result"]["error"]["code"] == "capability_not_allowed"
    assert ingested["result"]["error"]["capability"] == "vision_classify"
    assert ingested["realtime"]["frame_count"] == 0
    assert ingested["auto_snapshot"] == {"saved": False, "reason": "capability_denied"}


def test_capture_layout_calibration_screenshot_entry_requires_screen_observe() -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="blind_profile",
            display_name="Blind Profile",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.VISION_CLASSIFY,),
        )
    )
    plugin._active_profile_id = "blind_profile"

    captured = _payload(plugin.capture_layout_calibration_screenshot_entry())

    assert captured["success"] is False
    assert captured["error"]["code"] == "capability_not_allowed"
    assert captured["error"]["capability"] == "screen_observe"


def test_calibrate_layout_entry_requires_vision_capability_before_analyzing(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="no_vision_tft",
            display_name="No Vision TFT",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(11, 22, 33)).save(screenshot)

    result = _payload(
        plugin.calibrate_layout_entry(
            output_dir=str(tmp_path / "layout_calibration"),
            image_paths=[str(screenshot)],
            profile_id="no_vision_tft",
        )
    )

    assert result["success"] is False
    assert result["error"]["code"] == "capability_not_allowed"
    assert result["error"]["capability"] == "vision_classify"


def test_realtime_configure_rejects_enabled_profile_without_vision_capability() -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="realtime_watch_only",
            display_name="Realtime Watch Only",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )

    configured = _payload(plugin.realtime_configure(enabled=True, profile_id="realtime_watch_only"))

    assert configured["configured"] is False
    assert configured["error"]["code"] == "capability_not_allowed"
    assert configured["error"]["capability"] == "vision_classify"


def test_neko_context_checks_supplied_analysis_profile_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="payload_no_neko",
            display_name="Payload No NEKO",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE, Capability.VISION_CLASSIFY),
        )
    )
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    analysis = _payload(plugin.analyze_frame_entry(profile_id="generic", image_path=str(screenshot)))
    analysis["profile"] = "payload_no_neko"

    context = _payload(plugin.neko_context(analysis=analysis))

    assert context["success"] is False
    assert context["error"]["code"] == "capability_not_allowed"
    assert context["error"]["profile_id"] == "payload_no_neko"
    assert context["error"]["capability"] == "neko_context"


def test_neko_context_queue_entries_require_neko_context_capability() -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="queue_no_neko",
            display_name="Queue No NEKO",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE, Capability.VISION_CLASSIFY),
        )
    )
    plugin._active_profile_id = "queue_no_neko"

    listed = _payload(plugin.list_neko_context_queue())
    dequeued = _payload(plugin.dequeue_neko_context())

    assert listed["success"] is False
    assert listed["error"]["code"] == "capability_not_allowed"
    assert dequeued["success"] is False
    assert dequeued["error"]["code"] == "capability_not_allowed"


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


def test_game_companion_extract_layout_calibration_video_frames_entry_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plugin = _plugin()
    video_path = tmp_path / "match.mp4"
    manifest_path = tmp_path / "samples_manifest.json"
    video_path.write_bytes(b"fake video placeholder")

    class _FakeFrame:
        pts = 0

        def to_image(self) -> Image.Image:
            return Image.new("RGB", (1920, 1080), color=(20, 30, 40))

    class _FakeStream:
        type = "video"
        frames = 1
        time_base = Fraction(1, 100)

    class _FakeStreams:
        video = [_FakeStream()]

        def __iter__(self):
            return iter(self.video)

    class _FakeContainer:
        streams = _FakeStreams()

        def decode(self, video: int = 0):
            assert video == 0
            yield _FakeFrame()

        def close(self) -> None:
            pass

    fake_av = types.ModuleType("av")
    fake_av.open = lambda _path: _FakeContainer()
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    result = _payload(
        plugin.extract_layout_calibration_video_frames_entry(
            video_path=str(video_path),
            output_dir=str(tmp_path / "frames"),
            samples_manifest_path=str(manifest_path),
            expected_layout="normal_shop",
            tags=["shop_open"],
            max_frames=1,
        )
    )

    assert result["type"] == "tft_layout_calibration_video_frames"
    assert result["frame_count"] == 1
    assert Path(result["frames"][0]["image_path"]).is_file()
    assert result["manifest"]["manifest_path"] == str(manifest_path.resolve())
    assert result["manifest"]["samples"][0]["expected_layout"] == "normal_shop"


def test_calibration_video_and_manifest_entries_require_vision_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="calibration_no_vision",
            display_name="Calibration No Vision",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    plugin._active_profile_id = "calibration_no_vision"
    video_path = tmp_path / "match.mp4"
    input_dir = tmp_path / "input"
    video_path.write_bytes(b"fake video placeholder")
    input_dir.mkdir()

    extracted = _payload(
        plugin.extract_layout_calibration_video_frames_entry(
            video_path=str(video_path),
            output_dir=str(tmp_path / "frames"),
        )
    )
    prepared = _payload(plugin.prepare_layout_calibration_manifest_entry(input_dir=str(input_dir)))

    assert extracted["success"] is False
    assert extracted["error"]["code"] == "capability_not_allowed"
    assert extracted["error"]["capability"] == "vision_classify"
    assert prepared["success"] is False
    assert prepared["error"]["code"] == "capability_not_allowed"
    assert prepared["error"]["capability"] == "vision_classify"


def test_calibration_report_entries_require_vision_capability(tmp_path: Path) -> None:
    plugin = _plugin()
    plugin._profiles.register(
        ProfileMetadata(
            profile_id="report_no_vision",
            display_name="Report No Vision",
            game_type=GameType.TYPE_D,
            default_runtime_mode=RuntimeMode.ONLINE,
            capabilities=(Capability.SCREEN_OBSERVE,),
        )
    )
    plugin._active_profile_id = "report_no_vision"
    report_path = tmp_path / "calibration_report.json"

    summarized = _payload(plugin.summarize_layout_calibration_entry(report_path=str(report_path)))
    updated = _payload(
        plugin.update_layout_calibration_check_entry(
            report_path=str(report_path),
            screenshot_index=0,
            check_id="gold",
            status="pass",
        )
    )
    batch_updated = _payload(
        plugin.update_layout_calibration_checks_entry(
            report_path=str(report_path),
            updates=[{"screenshot_index": 0, "check_id": "gold", "status": "pass"}],
        )
    )

    for result in (summarized, updated, batch_updated):
        assert result["success"] is False
        assert result["error"]["code"] == "capability_not_allowed"
        assert result["error"]["capability"] == "vision_classify"


def test_game_companion_extract_layout_calibration_video_frames_entry_errors(tmp_path: Path) -> None:
    plugin = _plugin()

    result = _payload(
        plugin.extract_layout_calibration_video_frames_entry(
            video_path=str(tmp_path / "missing.mp4"),
            output_dir=str(tmp_path / "frames"),
        )
    )

    assert result["success"] is False
    assert result["error"]["code"] == "video_not_found"


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
    source_context = {
        "type": "video_frame",
        "profile_id": "tft",
        "video_path": str((tmp_path / "match.mp4").resolve()),
        "ordinal": 1,
        "frame_index": 30,
        "timestamp_seconds": 3.0,
    }

    configured = _payload(plugin.realtime_configure(enabled=True, profile_id="tft", interval_seconds=2))
    ingested = _payload(plugin.ingest_frame(profile_id="tft", image_path=str(screenshot), source_context=source_context))
    status = _payload(plugin.realtime_status())
    saved = _payload(plugin.save_review_snapshot(note="entrypoint smoke"))
    listed = _payload(plugin.list_review_snapshots())
    prompt = _payload(plugin.training_prompt())
    context = _payload(plugin.neko_context(note="entrypoint context", enqueue=True))
    queued = _payload(plugin.list_neko_context_queue())
    dequeued = _payload(plugin.dequeue_neko_context())
    cleared = _payload(plugin.clear_review_snapshots())

    assert configured["configured"] is True
    assert ingested["ok"] is True
    assert ingested["state"]["type"] == "tft_frame_state"
    assert ingested["state"]["game"] == "tft"
    assert ingested["state"]["source_context"]["frame_index"] == 30
    assert ingested["result"]["tft_state"]["type"] == "tft_frame_state"
    assert ingested["raw_recognition"]["type"] == "tft_recognition_result"
    assert ingested["result"]["success"] is True
    assert ingested["result"]["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}
    assert ingested["result"]["vision"]["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}
    assert ingested["realtime"]["frame_count"] == 1
    assert status["stable_result"]["profile"] == "tft"
    assert saved["saved"] is True
    assert listed["snapshots"][0]["note"] == "entrypoint smoke"
    assert prompt["available"] is True
    assert prompt["prompt"]["type"] == "tft_training_prompt"
    assert context["available"] is True
    assert context["packet"]["type"] == "game_companion_neko_context_packet"
    assert context["packet"]["delivery"]["mode"] == "queued_non_interrupting"
    assert context["packet"]["note"] == "entrypoint context"
    assert context["packet"]["state_digest"]["tft"]["layout"] == ingested["state"]["layout"]
    assert context["queued"]["queue_size"] == 1
    assert queued["queue_size"] == 1
    assert queued["packets"][0]["note"] == "entrypoint context"
    assert dequeued["available"] is True
    assert dequeued["packet"]["note"] == "entrypoint context"
    assert dequeued["queue_size"] == 0
    assert cleared == {"cleared": True, "snapshots": []}


def test_game_companion_neko_context_entry_without_analysis() -> None:
    plugin = _plugin()

    context = _payload(plugin.neko_context())

    assert context == {"available": False, "error": "no_analysis_available"}
