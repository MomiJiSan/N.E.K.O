from __future__ import annotations

from typing import Any

from plugin.sdk.plugin import (
    NekoPluginBase,
    Ok,
    lifecycle,
    neko_plugin,
    plugin_entry,
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
