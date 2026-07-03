from __future__ import annotations

import json
from pathlib import Path
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
    extract_tft_layout_calibration_video_frames,
    init_tft_layout_calibration_workspace,
    summarize_tft_layout_calibration_report,
    update_tft_layout_calibration_check,
    update_tft_layout_calibration_checks,
)
from .core.frame_analyzer import analyze_frame, analyze_frame_data_url
from .core.local_vision import LocalVisionBackend, reset_default_local_vision_backend, set_default_local_vision_backend
from .core.onnx_local_vision import create_onnx_classifier_backend, load_onnx_classifier_config
from .core.profile_registry import ProfileRegistry
from .core.realtime import RealtimeInsightSession
from .core.replay import (
    append_snapshot,
    build_neko_context_packet,
    build_snapshot,
    build_training_prompt,
    clear_snapshots,
    dequeue_neko_context_packet,
    enqueue_neko_context_packet,
    list_neko_context_queue,
    load_snapshots,
)
from .core.tft_recognition import build_tft_recognition_report, recognize_tft_frame
from .core.tft_runtime import build_tft_video_state_report
from .core.tft_smoke import build_tft_normal_shop_smoke_report
from .core.tft_state import build_tft_state
from .profiles import builtin_profiles
from .safety import Capability, capability_error_response, evaluate_profile_capability


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
        self._configure_local_vision_backend(cfg)
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
        reset_default_local_vision_backend()
        return Ok({"status": "stopped"})

    @lifecycle(id="config_change")
    async def config_change(self, **_: Any):
        cfg = await self._load_config()
        self._configure_local_vision_backend(cfg)
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

    def _configure_local_vision_backend(self, cfg: dict[str, Any], *, session_factory: Any | None = None) -> dict[str, str]:
        local_vision_cfg = cfg.get("local_vision") if isinstance(cfg.get("local_vision"), dict) else {}
        classifier_cfg = {
            "enabled": bool(local_vision_cfg.get("classifier_enabled")),
            "model_path": local_vision_cfg.get("classifier_model_path"),
            "labels_path": local_vision_cfg.get("classifier_labels_path"),
            "model_name": local_vision_cfg.get("classifier_model_name"),
            "input_size": local_vision_cfg.get("classifier_input_size"),
            "normalize_imagenet": local_vision_cfg.get("classifier_normalize_imagenet", True),
        }
        config = load_onnx_classifier_config(classifier_cfg, base_dir=self._plugin_base_dir())
        if config is None:
            reset_default_local_vision_backend()
            return {
                "classifier": "disabled" if not classifier_cfg["enabled"] else "not_configured",
                "detector": "not_configured",
            }
        classifier = create_onnx_classifier_backend(config, session_factory=session_factory)
        set_default_local_vision_backend(LocalVisionBackend(classifier=classifier))
        return {"classifier": "registered", "detector": "not_configured"}

    def _plugin_base_dir(self) -> str:
        config_path = getattr(getattr(self, "ctx", None), "config_path", None)
        if config_path:
            return str(Path(str(config_path)).expanduser().resolve().parent)
        return str(Path(__file__).resolve().parent)

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

    def _require_capability(self, profile_id: str, capability: Capability | str) -> dict[str, Any] | None:
        normalized = str(profile_id or self._active_profile_id or "generic").strip().lower()
        profile = self._profiles.get(normalized)
        decision = evaluate_profile_capability(profile, capability, profile_id=normalized)
        if decision["allowed"]:
            return None
        return capability_error_response(decision)

    def _profile_id_for_payload(self, payload: dict[str, Any] | None = None, fallback: str | None = None) -> str:
        if isinstance(payload, dict):
            candidate = str(payload.get("profile") or payload.get("profile_id") or "").strip().lower()
            if candidate:
                return candidate
        return str(fallback or self._active_profile_id or "generic").strip().lower()

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
                    "description": "Profile id. Supports generic and tft offline analysis.",
                },
                "image_path": {
                    "type": "string",
                    "description": "Absolute or workspace-visible path to a local screenshot.",
                },
                "debug_crops_dir": {
                    "type": "string",
                    "description": "Optional directory where TFT region crops should be saved for calibration.",
                },
                "vlm_requested": {
                    "type": "boolean",
                    "description": "Plan a vision-model fallback because the user explicitly asked for semantic screen understanding.",
                },
                "source_context": {
                    "type": "object",
                    "description": "Optional provenance metadata, for example a video_frame origin with frame_index and timestamp_seconds.",
                },
            },
            "required": ["profile_id", "image_path"],
        },
    )
    def analyze_frame_entry(
        self,
        profile_id: str,
        image_path: str,
        debug_crops_dir: str | None = None,
        vlm_requested: bool = False,
        source_context: dict[str, Any] | None = None,
        **_: Any,
    ):
        capability_error = self._require_capability(profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        return Ok(
            analyze_frame(
                profile_id=profile_id,
                image_path=image_path,
                debug_crops_dir=debug_crops_dir,
                vlm_requested=bool(vlm_requested),
                source_context=source_context,
            )
        )

    @plugin_entry(
        id="game_companion_recognize_tft_frame",
        name="Recognize TFT frame",
        description="Return a focused TFT structured recognition result for one local screenshot.",
        input_schema={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute or workspace-visible path to a local TFT screenshot.",
                },
                "expected_layout": {
                    "type": "string",
                    "description": "Optional TFT layout hint: normal_shop, combat, or augment_select.",
                },
            },
            "required": ["image_path"],
        },
    )
    def recognize_tft_frame_entry(
        self,
        image_path: str,
        expected_layout: str | None = None,
        **_: Any,
    ):
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        recognition = recognize_tft_frame(image_path, expected_layout=expected_layout)
        recognition["state"] = build_tft_state(recognition)
        return Ok(recognition)

    @plugin_entry(
        id="game_companion_build_tft_recognition_report",
        name="Build TFT recognition report",
        description="Run focused TFT recognition over screenshots from a calibration report.",
        input_schema={
            "type": "object",
            "properties": {
                "calibration_report_path": {
                    "type": "string",
                    "description": "Path to a TFT layout calibration_report.json.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where recognition_report_v1.json and recognition_summary_v1.json are written.",
                },
            },
            "required": ["calibration_report_path", "output_dir"],
        },
    )
    def build_tft_recognition_report_entry(
        self,
        calibration_report_path: str,
        output_dir: str,
        **_: Any,
    ):
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        try:
            return Ok(build_tft_recognition_report(calibration_report_path, output_dir=output_dir))
        except FileNotFoundError as exc:
            return Ok(_entry_error("report_not_found", str(exc)))
        except json.JSONDecodeError as exc:
            return Ok(_entry_error("report_decode_failed", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("report_write_failed", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("invalid_report", str(exc)))

    @plugin_entry(
        id="game_companion_build_tft_video_state_report",
        name="Build TFT video state report",
        description="Sample a local TFT recording and write runtime TFTState JSONL, summary, and contact sheet.",
        input_schema={
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Local path to a TFT recording such as mp4, mkv, mov, webm, avi, or m4v.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where runtime_state_v1 artifacts are written.",
                },
                "sample_interval_seconds": {
                    "type": "number",
                    "description": "Target frame sampling interval in seconds. Defaults to 2.0.",
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum frames to sample. Clamped by the runtime reporter.",
                },
                "frame_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional exact zero-based frame indices to sample.",
                },
                "frame_layouts": {
                    "type": "object",
                    "description": "Optional mapping of frame index to layout hint for mixed-layout video verification.",
                },
                "shop_labels_path": {
                    "type": "string",
                    "description": "Optional recognition_shop_labels_v1.json used as local verified shop cost/name fallback.",
                },
                "expected_layout": {
                    "type": "string",
                    "description": "Optional layout hint applied to sampled frames: normal_shop, combat, or augment_select.",
                },
            },
            "required": ["video_path"],
        },
    )
    def build_tft_video_state_report_entry(
        self,
        video_path: str,
        output_dir: str | None = None,
        sample_interval_seconds: float = 2.0,
        max_frames: int = 60,
        expected_layout: str | None = None,
        frame_indices: list[int] | None = None,
        frame_layouts: dict[str, str] | None = None,
        shop_labels_path: str | None = None,
        **_: Any,
    ):
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        try:
            return Ok(
                build_tft_video_state_report(
                    video_path,
                    output_dir=output_dir,
                    sample_interval_seconds=sample_interval_seconds,
                    max_frames=max_frames,
                    expected_layout=expected_layout,
                    frame_indices=frame_indices,
                    frame_layouts=frame_layouts,
                    shop_labels_path=shop_labels_path,
                )
            )
        except FileNotFoundError as exc:
            return Ok(_entry_error("video_not_found", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("video_state_report_invalid", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("video_state_report_failed", str(exc)))

    @plugin_entry(
        id="game_companion_tft_normal_shop_smoke",
        name="Run TFT normal shop smoke",
        description="Run the fixed TFT normal_shop smoke suite over a local recording and write a unified pass/fail report.",
        input_schema={
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Local path to the TFT recording used for the fixed normal_shop smoke suite.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory where tft_normal_shop_smoke_v1.json and child runtime reports are written.",
                },
            },
            "required": ["video_path"],
        },
    )
    def tft_normal_shop_smoke_entry(
        self,
        video_path: str,
        output_dir: str | None = None,
        **_: Any,
    ):
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        try:
            return Ok(build_tft_normal_shop_smoke_report(video_path, output_dir=output_dir))
        except FileNotFoundError as exc:
            return Ok(_entry_error("video_not_found", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("tft_smoke_invalid", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("tft_smoke_failed", str(exc)))

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
        capability_error = self._require_capability(self._active_profile_id, Capability.SCREEN_OBSERVE)
        if capability_error:
            return Ok(capability_error)
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
        id="game_companion_extract_layout_calibration_video_frames",
        name="Extract TFT calibration frames from video",
        description="Extract local TFT recording frames into the layout calibration input directory.",
        input_schema={
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Local path to a TFT recording such as mp4, mkv, mov, webm, avi, or m4v.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory. Defaults to plugin/plugins/game_companion/.local_calibration/input.",
                },
                "samples_manifest_path": {
                    "type": "string",
                    "description": "Optional path for the generated samples_manifest.json.",
                },
                "frame_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional exact zero-based frame indices to extract.",
                },
                "max_frames": {
                    "type": "integer",
                    "description": "Maximum frames to sample when frame_indices is omitted. Clamped to 1-60.",
                },
                "expected_layout": {
                    "type": "string",
                    "description": "Optional expected layout applied to extracted frames.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional calibration tags applied to extracted frames.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label included in extracted frame filenames.",
                },
            },
            "required": ["video_path"],
        },
    )
    def extract_layout_calibration_video_frames_entry(
        self,
        video_path: str,
        output_dir: str | None = None,
        samples_manifest_path: str | None = None,
        frame_indices: list[int] | None = None,
        max_frames: int = 8,
        expected_layout: str | None = None,
        tags: list[str] | None = None,
        label: str | None = None,
        **_: Any,
    ):
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
        try:
            return Ok(
                extract_tft_layout_calibration_video_frames(
                    video_path,
                    output_dir=output_dir,
                    samples_manifest_path=samples_manifest_path,
                    frame_indices=frame_indices,
                    max_frames=max_frames,
                    expected_layout=expected_layout,
                    tags=tags or [],
                    label=label,
                )
            )
        except FileNotFoundError as exc:
            return Ok(_entry_error("video_not_found", str(exc)))
        except ValueError as exc:
            return Ok(_entry_error("video_frame_extract_invalid", str(exc)))
        except OSError as exc:
            return Ok(_entry_error("video_frame_extract_failed", str(exc)))

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
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
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
        capability_error = self._require_capability(normalized, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
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
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
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
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
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
        capability_error = self._require_capability(self._active_profile_id, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok(capability_error)
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
        name="Ingest game frame",
        description="Analyze one game frame from image_path or image_data_url and update realtime insight state.",
        input_schema={
            "type": "object",
            "properties": {
                "profile_id": {"type": "string", "description": "Profile id. Supports generic and tft."},
                "image_path": {"type": "string", "description": "Optional local screenshot path."},
                "image_data_url": {"type": "string", "description": "Optional PNG/JPEG data URL from Electron capture."},
                "debug_crops_dir": {"type": "string", "description": "Optional crop output directory for image_path ingestion."},
                "vlm_requested": {
                    "type": "boolean",
                    "description": "Plan a vision-model fallback for this ingested frame without executing an external call.",
                },
                "source_context": {
                    "type": "object",
                    "description": "Optional provenance metadata, for example a video_frame origin with frame_index and timestamp_seconds.",
                },
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
        vlm_requested: bool = False,
        source_context: dict[str, Any] | None = None,
        **_: Any,
    ):
        self._ensure_runtime_state()
        normalized = str(profile_id or self._realtime.profile_id or "tft").strip().lower()
        capability_error = self._require_capability(normalized, Capability.VISION_CLASSIFY)
        if capability_error:
            return Ok({"result": capability_error, "realtime": self._realtime.to_dict(), "auto_snapshot": {"saved": False, "reason": "capability_denied"}})
        if image_data_url:
            result = analyze_frame_data_url(
                profile_id=normalized,
                image_data_url=image_data_url,
                vlm_requested=bool(vlm_requested),
                source_context=source_context,
            )
        elif image_path:
            result = analyze_frame(
                profile_id=normalized,
                image_path=image_path,
                debug_crops_dir=debug_crops_dir,
                vlm_requested=bool(vlm_requested),
                source_context=source_context,
            )
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
        raw_recognition = (
            recognize_tft_frame(image_path, expected_layout=_expected_tft_layout(source_context))
            if normalized == "tft" and image_path
            else None
        )
        state = (
            build_tft_state(
                raw_recognition,
                timestamp=_source_timestamp(source_context),
                source_context=_redacted_source_context(source_context),
            )
            if raw_recognition is not None
            else None
        )
        if raw_recognition is not None:
            raw_recognition["state"] = state
        if state is not None and isinstance(result, dict):
            result["tft_state"] = state
        realtime = self._realtime.ingest(result)
        auto_snapshot = self._maybe_auto_save_snapshot(result)
        payload = {"ok": bool(result.get("success")), "result": result, "realtime": realtime, "auto_snapshot": auto_snapshot}
        if state is not None:
            payload["state"] = state
        if raw_recognition is not None:
            payload["raw_recognition"] = raw_recognition
        return Ok(payload)

    @plugin_entry(
        id="game_companion_realtime_status",
        name="Game realtime insight status",
        description="Return the current realtime game insight session state.",
    )
    def realtime_status(self, **_: Any):
        self._ensure_runtime_state()
        return Ok(self._realtime.to_dict())

    @plugin_entry(
        id="game_companion_realtime_configure",
        name="Configure game realtime insights",
        description="Enable or disable UI-driven realtime game insight ingestion.",
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
        requested_profile = str(profile_id or self._realtime.profile_id or self._active_profile_id or "tft").strip().lower()
        if enabled is True:
            capability_error = self._require_capability(requested_profile, Capability.VISION_CLASSIFY)
            if capability_error:
                return Ok({"configured": False, "error": capability_error["error"], "realtime": self._realtime.to_dict()})
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

    @plugin_entry(
        id="game_companion_neko_context",
        name="Build NEKO game context",
        description="Build a queued, non-interrupting summary-only context packet for Yui/NEKO.",
        input_schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "object", "description": "Optional analysis payload. Defaults to latest stable frame."},
                "note": {"type": "string", "description": "Optional local note for the context packet."},
                "enqueue": {"type": "boolean", "description": "Store the sanitized context packet in the plugin-local Yui queue."},
            },
        },
    )
    def neko_context(self, analysis: dict[str, Any] | None = None, note: str = "", enqueue: bool = False, **_: Any):
        self._ensure_runtime_state()
        source = analysis or self._realtime.stable_result or self._realtime.last_result
        if not isinstance(source, dict):
            return Ok({"available": False, "error": "no_analysis_available"})
        guard_profile_id = self._profile_id_for_payload(analysis) if isinstance(analysis, dict) else self._active_profile_id
        capability_error = self._require_capability(guard_profile_id, Capability.NEKO_CONTEXT)
        if capability_error:
            return Ok(capability_error)
        packet = build_neko_context_packet(source, note=str(note or ""))
        result = {"available": True, "packet": packet}
        if enqueue:
            result["queued"] = enqueue_neko_context_packet(self.store, packet)
        return Ok(result)

    @plugin_entry(
        id="game_companion_list_neko_context_queue",
        name="List NEKO game context queue",
        description="List sanitized queued game context packets for Yui/NEKO without exposing raw screenshots.",
    )
    def list_neko_context_queue(self, **_: Any):
        capability_error = self._require_capability(self._active_profile_id, Capability.NEKO_CONTEXT)
        if capability_error:
            return Ok(capability_error)
        packets = list_neko_context_queue(self.store)
        return Ok({"queue_size": len(packets), "packets": packets})

    @plugin_entry(
        id="game_companion_dequeue_neko_context",
        name="Dequeue NEKO game context",
        description="Pop the oldest sanitized queued game context packet for Yui/NEKO.",
    )
    def dequeue_neko_context(self, **_: Any):
        capability_error = self._require_capability(self._active_profile_id, Capability.NEKO_CONTEXT)
        if capability_error:
            return Ok(capability_error)
        return Ok(dequeue_neko_context_packet(self.store))

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


def _expected_tft_layout(source_context: dict[str, Any] | None) -> str | None:
    if not isinstance(source_context, dict):
        return None
    for key in ("expected_layout", "layout", "layout_hint"):
        value = source_context.get(key)
        if value:
            return str(value)
    return None


def _source_timestamp(source_context: dict[str, Any] | None) -> float | None:
    if not isinstance(source_context, dict):
        return None
    value = source_context.get("timestamp_seconds")
    if value is None:
        value = source_context.get("timestamp")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _redacted_source_context(source_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source_context, dict):
        return {}
    redacted = dict(source_context)
    if redacted.get("video_path"):
        redacted["video_path"] = "[redacted_path]"
    if redacted.get("image_path"):
        redacted["image_path"] = "[redacted_path]"
    return redacted
