from __future__ import annotations

import json
from typing import Any

from plugin.sdk.plugin import (
    NekoPluginBase,
    Ok,
    lifecycle,
    neko_plugin,
    plugin_entry,
)

from .core.calibration import (
    build_tft_layout_calibration_report,
    build_tft_layout_calibration_status,
    build_tft_layout_sample_manifest,
    capture_tft_layout_calibration_screenshot,
    init_tft_layout_calibration_workspace,
    summarize_tft_layout_calibration_report,
    update_tft_layout_calibration_check,
    update_tft_layout_calibration_checks,
)
from .core.frame_analyzer import analyze_frame, analyze_frame_data_url
from .core.profile_registry import ProfileRegistry
from .core.realtime import RealtimeInsightSession
from .core.replay import (
    append_snapshot,
    build_snapshot,
    build_training_prompt,
    clear_snapshots,
    load_snapshots,
)
from .profiles import builtin_profiles


@neko_plugin
class GameCompanionPlugin(NekoPluginBase):
    """Shared host for game-specific companion profiles."""

    def __init__(self, ctx: Any):
        super().__init__(ctx)
        self.logger = ctx.logger
        self._profiles = ProfileRegistry()
        for profile in builtin_profiles():
            self._profiles.register(profile)
        self._active_profile_id = "generic"
        self._realtime = RealtimeInsightSession()
        self._last_auto_snapshot_key = ""

    @lifecycle(id="startup")
    async def startup(self, **_: Any):
        cfg = await self._load_config()
        requested = str(cfg.get("default_profile") or "generic").strip() or "generic"
        if self._profiles.has(requested):
            self._active_profile_id = requested
        else:
            self.logger.warning(
                "game_companion default_profile={} is unknown; falling back to generic",
                requested,
            )
            self._active_profile_id = "generic"
        self.register_static_ui("static", cache_control="no-store")
        return Ok(self._status_payload())

    @lifecycle(id="shutdown")
    async def shutdown(self, **_: Any):
        return Ok({"status": "stopped"})

    @lifecycle(id="config_change")
    async def config_change(self, **_: Any):
        cfg = await self._load_config()
        requested = str(cfg.get("default_profile") or self._active_profile_id).strip()
        if requested and self._profiles.has(requested):
            self._active_profile_id = requested
        return Ok(self._status_payload())

    async def _load_config(self) -> dict[str, Any]:
        try:
            data = await self.config.dump(timeout=5.0)
        except Exception:
            self.logger.debug("game_companion config dump failed", exc_info=True)
            return {}
        section = data.get("game_companion", {}) if isinstance(data, dict) else {}
        return section if isinstance(section, dict) else {}

    def _ensure_runtime_state(self) -> None:
        if not hasattr(self, "_realtime"):
            self._realtime = RealtimeInsightSession()
        if not hasattr(self, "_last_auto_snapshot_key"):
            self._last_auto_snapshot_key = ""

    def _status_payload(self) -> dict[str, Any]:
        self._ensure_runtime_state()
        return {
            "status": "ready",
            "active_profile": self._active_profile_id,
            "profiles": [profile.to_dict() for profile in self._profiles.list()],
            "realtime": self._realtime.to_dict(),
        }

    @plugin_entry(
        id="game_companion_status",
        name="Game Companion status",
        description="Return the active profile and available game companion profiles.",
    )
    def status(self, **_: Any):
        return Ok(self._status_payload())

    @plugin_entry(
        id="game_companion_list_profiles",
        name="List game profiles",
        description="Return all built-in game companion profiles.",
    )
    def list_profiles(self, **_: Any):
        return Ok({"profiles": [profile.to_dict() for profile in self._profiles.list()]})

    @plugin_entry(
        id="game_companion_select_profile",
        name="Select game profile",
        description="Select an available game companion profile for this session.",
        input_schema={
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "string",
                    "description": "Profile id, such as generic, galgame, or tft.",
                }
            },
            "required": ["profile_id"],
        },
    )
    def select_profile(self, profile_id: str, **_: Any):
        normalized = str(profile_id or "").strip().lower()
        if not self._profiles.has(normalized):
            return Ok(
                {
                    "selected": False,
                    "active_profile": self._active_profile_id,
                    "error": "unknown_profile",
                    "available_profiles": [profile.profile_id for profile in self._profiles.list()],
                }
            )
        self._active_profile_id = normalized
        return Ok({"selected": True, "active_profile": self._active_profile_id})

    @plugin_entry(
        id="game_companion_analyze_frame",
        name="Analyze game frame",
        description="Analyze a local image file with the selected game companion profile.",
        input_schema={
            "type": "object",
            "properties": {
                "profile_id": {
                    "type": "string",
                    "description": "Profile id. Phase 1 supports tft.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Absolute or workspace-visible path to a local screenshot.",
                },
                "debug_crops_dir": {
                    "type": "string",
                    "description": "Optional directory where TFT region crops should be saved for calibration.",
                },
            },
            "required": ["profile_id", "image_path"],
        },
    )
    def analyze_frame_entry(self, profile_id: str, image_path: str, debug_crops_dir: str | None = None, **_: Any):
        return Ok(analyze_frame(profile_id=profile_id, image_path=image_path, debug_crops_dir=debug_crops_dir))

    @plugin_entry(
        id="game_companion_init_layout_calibration_workspace",
        name="Initialize TFT layout calibration workspace",
        description="Create local-only folders for TFT screenshots, manifests, crops, and reports.",
        input_schema={
            "type": "object",
            "properties": {
                "root_dir": {
                    "type": "string",
                    "description": "Optional local calibration workspace root. Defaults to plugin/plugins/game_companion/.local_calibration.",
                },
                "overwrite_manifest": {
                    "type": "boolean",
                    "description": "Regenerate samples_manifest.json and README.md if they already exist.",
                },
            },
        },
    )
    def init_layout_calibration_workspace_entry(
        self,
        root_dir: str | None = None,
        overwrite_manifest: bool = False,
        **_: Any,
    ):
        try:
            return Ok(init_tft_layout_calibration_workspace(root_dir, overwrite_manifest=bool(overwrite_manifest)))
        except OSError as exc:
            return Ok(_entry_error("workspace_init_failed", str(exc)))

    @plugin_entry(
        id="game_companion_capture_layout_calibration_screenshot",
        name="Capture TFT layout calibration screenshot",
        description="Capture the current primary screen into the local TFT layout calibration input directory.",
        input_schema={
            "type": "object",
            "properties": {
                "output_dir": {
                    "type": "string",
                    "description": "Optional screenshot output directory. Defaults to plugin/plugins/game_companion/.local_calibration/input.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label included in the saved screenshot filename and manifest metadata.",
                },
                "expected_layout": {
                    "type": "string",
                    "description": "Optional expected layout: normal_shop, combat, augment_select, or special.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional calibration tags such as shop_open, shop_five_units, or bench_units.",
                },
            },
        },
    )
    def capture_layout_calibration_screenshot_entry(
        self,
        output_dir: str | None = None,
        label: str | None = None,
        expected_layout: str | None = None,
        tags: list[str] | None = None,
        **_: Any,
    ):
        try:
            return Ok(
                capture_tft_layout_calibration_screenshot(
                    output_dir,
                    label=label,
                    expected_layout=expected_layout,
                    tags=tags or [],
                )
            )
        except OSError as exc:
            return Ok(_entry_error("screen_capture_failed", str(exc)))

    @plugin_entry(
        id="game_companion_prepare_layout_calibration_manifest",
        name="Prepare TFT layout calibration manifest",
        description="Scan a local screenshot directory and write an editable TFT calibration samples manifest.",
        input_schema={
            "type": "object",
            "properties": {
                "input_dir": {
                    "type": "string",
                    "description": "Local directory containing TFT screenshots.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path for samples_manifest.json.",
                },
            },
            "required": ["input_dir"],
        },
    )
    def prepare_layout_calibration_manifest_entry(
        self,
        input_dir: str,
        output_path: str | None = None,
        **_: Any,
    ):
        try:
            return Ok(build_tft_layout_sample_manifest(input_dir, output_path))
        except FileNotFoundError as exc:
            return Ok(_entry_error("input_dir_not_found", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("manifest_write_failed", str(exc)))

    @plugin_entry(
        id="game_companion_layout_calibration_status",
        name="TFT layout calibration status",
        description="Inspect TFT calibration input, samples manifest, or report readiness without generating crops.",
        input_schema={
            "type": "object",
            "properties": {
                "input_dir": {
                    "type": "string",
                    "description": "Optional screenshot directory to inspect.",
                },
                "samples_manifest_path": {
                    "type": "string",
                    "description": "Optional samples_manifest.json path to inspect.",
                },
                "report_path": {
                    "type": "string",
                    "description": "Optional calibration_report.json path to inspect.",
                },
            },
        },
    )
    def layout_calibration_status_entry(
        self,
        input_dir: str | None = None,
        samples_manifest_path: str | None = None,
        report_path: str | None = None,
        **_: Any,
    ):
        try:
            return Ok(
                build_tft_layout_calibration_status(
                    input_dir=input_dir,
                    samples_manifest_path=samples_manifest_path,
                    report_path=report_path,
                )
            )
        except OSError as exc:
            return Ok(_entry_error("calibration_status_failed", str(exc)))

    @plugin_entry(
        id="game_companion_calibrate_layout",
        name="Calibrate TFT layout",
        description="Batch-analyze TFT screenshots and generate debug crops plus a manual layout calibration report.",
        input_schema={
            "type": "object",
            "properties": {
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Local TFT screenshot paths. Recommended: 5-10 real screenshots.",
                },
                "samples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string"},
                            "expected_layout": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "label": {"type": "string"},
                            "note": {"type": "string"},
                        },
                    },
                    "description": "Optional structured screenshot samples with expected_layout and calibration tags.",
                },
                "samples_manifest_path": {
                    "type": "string",
                    "description": "Optional path to samples_manifest.json generated by game_companion_prepare_layout_calibration_manifest.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Local directory for crops, calibration_report.json, and index.html.",
                },
                "profile_id": {
                    "type": "string",
                    "description": "Profile id. Phase 4 calibration supports tft.",
                },
            },
            "required": ["output_dir"],
        },
    )
    def calibrate_layout_entry(
        self,
        output_dir: str,
        image_paths: list[str] | None = None,
        samples: list[dict[str, Any]] | None = None,
        samples_manifest_path: str | None = None,
        profile_id: str = "tft",
        **_: Any,
    ):
        normalized = str(profile_id or "tft").strip().lower()
        if normalized != "tft":
            return Ok(
                {
                    "success": False,
                    "error": {"code": "unsupported_profile", "message": "layout calibration currently supports tft only"},
                }
            )
        if not image_paths and not samples and not samples_manifest_path:
            return Ok(_entry_error("missing_images", "image_paths, samples, or samples_manifest_path is required"))
        try:
            return Ok(
                build_tft_layout_calibration_report(
                    image_paths or [],
                    output_dir,
                    profile_id=normalized,
                    samples=samples,
                    samples_manifest_path=samples_manifest_path,
                )
            )
        except FileNotFoundError as exc:
            return Ok(_entry_error("samples_manifest_not_found", str(exc)))
        except json.JSONDecodeError as exc:
            return Ok(_entry_error("samples_manifest_decode_failed", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("invalid_samples_manifest", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("output_write_failed", str(exc)))

    @plugin_entry(
        id="game_companion_summarize_layout_calibration",
        name="Summarize TFT layout calibration",
        description="Summarize edited TFT calibration_report.json manual check statuses.",
        input_schema={
            "type": "object",
            "properties": {
                "report_path": {
                    "type": "string",
                    "description": "Path to calibration_report.json after manual check statuses are edited.",
                },
                "report": {
                    "type": "object",
                    "description": "Optional in-memory calibration report payload.",
                },
            },
        },
    )
    def summarize_layout_calibration_entry(
        self,
        report_path: str | None = None,
        report: dict[str, Any] | None = None,
        **_: Any,
    ):
        try:
            if isinstance(report, dict):
                return Ok(summarize_tft_layout_calibration_report(report))
            if report_path:
                return Ok(summarize_tft_layout_calibration_report(report_path))
        except FileNotFoundError as exc:
            return Ok(_entry_error("report_not_found", str(exc)))
        except json.JSONDecodeError as exc:
            return Ok(_entry_error("report_decode_failed", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("report_read_failed", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("invalid_report", str(exc)))
        return Ok(
            {
                "success": False,
                "error": {
                    "code": "missing_report",
                    "message": "report_path or report is required",
                },
            }
        )

    @plugin_entry(
        id="game_companion_update_layout_calibration_check",
        name="Update TFT layout calibration check",
        description="Update one manual check in calibration_report.json and refresh its summary and HTML report.",
        input_schema={
            "type": "object",
            "properties": {
                "report_path": {"type": "string"},
                "screenshot_index": {"type": "integer"},
                "check_id": {"type": "string"},
                "status": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["report_path", "screenshot_index", "check_id", "status"],
        },
    )
    def update_layout_calibration_check_entry(
        self,
        report_path: str,
        screenshot_index: int,
        check_id: str,
        status: str,
        note: str | None = None,
        **_: Any,
    ):
        try:
            return Ok(
                update_tft_layout_calibration_check(
                    report_path,
                    screenshot_index=screenshot_index,
                    check_id=check_id,
                    status=status,
                    note=note,
                )
            )
        except FileNotFoundError as exc:
            return Ok(_entry_error("report_not_found", str(exc)))
        except json.JSONDecodeError as exc:
            return Ok(_entry_error("report_decode_failed", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("report_write_failed", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("invalid_report", str(exc)))

    @plugin_entry(
        id="game_companion_update_layout_calibration_checks",
        name="Update TFT layout calibration checks",
        description="Batch-update manual checks in calibration_report.json and refresh its summary and HTML report.",
        input_schema={
            "type": "object",
            "properties": {
                "report_path": {"type": "string"},
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "screenshot_index": {"type": "integer"},
                            "check_id": {"type": "string"},
                            "status": {"type": "string"},
                            "note": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["report_path", "updates"],
        },
    )
    def update_layout_calibration_checks_entry(
        self,
        report_path: str,
        updates: list[dict[str, Any]],
        **_: Any,
    ):
        try:
            return Ok(update_tft_layout_calibration_checks(report_path, updates=updates))
        except FileNotFoundError as exc:
            return Ok(_entry_error("report_not_found", str(exc)))
        except json.JSONDecodeError as exc:
            return Ok(_entry_error("report_decode_failed", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("report_write_failed", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("invalid_report", str(exc)))

    @plugin_entry(
        id="game_companion_ingest_frame",
        name="Ingest TFT frame",
        description="Analyze one TFT frame from image_path or image_data_url and update realtime insight state.",
        input_schema={
            "type": "object",
            "properties": {
                "profile_id": {"type": "string", "description": "Profile id. Phase 7 supports tft."},
                "image_path": {"type": "string", "description": "Optional local screenshot path."},
                "image_data_url": {"type": "string", "description": "Optional PNG/JPEG data URL from Electron capture."},
                "debug_crops_dir": {"type": "string", "description": "Optional crop output directory for image_path ingestion."},
            },
            "required": ["profile_id"],
        },
    )
    def ingest_frame(
        self,
        profile_id: str,
        image_path: str | None = None,
        image_data_url: str | None = None,
        debug_crops_dir: str | None = None,
        **_: Any,
    ):
        self._ensure_runtime_state()
        normalized = str(profile_id or self._realtime.profile_id or "tft").strip().lower()
        if image_data_url:
            result = analyze_frame_data_url(profile_id=normalized, image_data_url=image_data_url)
        elif image_path:
            result = analyze_frame(profile_id=normalized, image_path=image_path, debug_crops_dir=debug_crops_dir)
        else:
            result = {
                "success": False,
                "profile": normalized,
                "error": {"code": "missing_image_source", "message": "image_path or image_data_url is required"},
                "state": {},
                "regions": {},
                "insights": [],
                "diagnostics": {"warnings": []},
            }
        realtime = self._realtime.ingest(result)
        auto_snapshot = self._maybe_auto_save_snapshot(result)
        return Ok({"result": result, "realtime": realtime, "auto_snapshot": auto_snapshot})

    @plugin_entry(
        id="game_companion_realtime_status",
        name="TFT realtime insight status",
        description="Return the current realtime TFT insight session state.",
    )
    def realtime_status(self, **_: Any):
        self._ensure_runtime_state()
        return Ok(self._realtime.to_dict())

    @plugin_entry(
        id="game_companion_realtime_configure",
        name="Configure TFT realtime insights",
        description="Enable or disable UI-driven realtime TFT insight ingestion.",
        input_schema={
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "profile_id": {"type": "string"},
                "interval_seconds": {"type": "number"},
                "debounce_seconds": {"type": "number"},
            },
        },
    )
    def realtime_configure(
        self,
        enabled: bool | None = None,
        profile_id: str | None = None,
        interval_seconds: float | None = None,
        debounce_seconds: float | None = None,
        **_: Any,
    ):
        self._ensure_runtime_state()
        if profile_id is not None and not self._profiles.has(profile_id):
            return Ok({"configured": False, "error": "unknown_profile", "realtime": self._realtime.to_dict()})
        realtime = self._realtime.configure(
            enabled=enabled,
            profile_id=profile_id,
            interval_seconds=interval_seconds,
            debounce_seconds=debounce_seconds,
        )
        return Ok({"configured": True, "realtime": realtime})

    @plugin_entry(
        id="game_companion_save_review_snapshot",
        name="Save TFT review snapshot",
        description="Save the latest or provided TFT analysis for later review.",
        input_schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "object"},
                "note": {"type": "string"},
            },
        },
    )
    def save_review_snapshot(self, analysis: dict[str, Any] | None = None, note: str = "", **_: Any):
        self._ensure_runtime_state()
        source = analysis or self._realtime.stable_result or self._realtime.last_result
        if not isinstance(source, dict):
            return Ok({"saved": False, "error": "no_analysis_available", "snapshots": load_snapshots(self.store)})
        snapshot = build_snapshot(source, note=str(note or ""))
        snapshots = append_snapshot(self.store, snapshot)
        return Ok({"saved": True, "snapshot": snapshot, "count": len(snapshots)})

    @plugin_entry(
        id="game_companion_list_review_snapshots",
        name="List TFT review snapshots",
        description="List saved TFT review snapshots.",
    )
    def list_review_snapshots(self, **_: Any):
        return Ok({"snapshots": load_snapshots(self.store)})

    @plugin_entry(
        id="game_companion_clear_review_snapshots",
        name="Clear TFT review snapshots",
        description="Clear saved TFT review snapshots.",
    )
    def clear_review_snapshots(self, **_: Any):
        clear_snapshots(self.store)
        return Ok({"cleared": True, "snapshots": []})

    @plugin_entry(
        id="game_companion_training_prompt",
        name="Build TFT training prompt",
        description="Build a review/training prompt from a saved or latest TFT snapshot.",
        input_schema={
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
            },
        },
    )
    def training_prompt(self, index: int | None = None, **_: Any):
        self._ensure_runtime_state()
        snapshots = load_snapshots(self.store)
        snapshot: dict[str, Any] | None = None
        if snapshots:
            selected = -1 if index is None else int(index)
            try:
                snapshot = snapshots[selected]
            except IndexError:
                snapshot = None
        if snapshot is None and isinstance(self._realtime.stable_result, dict):
            snapshot = build_snapshot(self._realtime.stable_result)
        if snapshot is None:
            return Ok({"available": False, "error": "no_snapshot_available"})
        return Ok({"available": True, "prompt": build_training_prompt(snapshot)})

    def _maybe_auto_save_snapshot(self, result: dict[str, Any]) -> dict[str, Any]:
        if not result.get("success"):
            return {"saved": False, "reason": "analysis_failed"}
        state = result.get("state") if isinstance(result.get("state"), dict) else {}
        stage = str(state.get("stage") or state.get("round") or "").strip()
        if not stage:
            return {"saved": False, "reason": "stage_unknown"}
        key = f"{result.get('profile') or 'tft'}:{stage}"
        if key == self._last_auto_snapshot_key:
            return {"saved": False, "reason": "same_stage"}
        self._last_auto_snapshot_key = key
        snapshot = build_snapshot(result, note=f"auto:{stage}")
        snapshots = append_snapshot(self.store, snapshot)
        return {"saved": True, "reason": "stage_changed", "snapshot": snapshot, "count": len(snapshots)}


def _entry_error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}
