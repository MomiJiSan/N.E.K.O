from __future__ import annotations

import threading
import time
from typing import Any

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    neko_plugin,
    plugin_entry,
    timer_interval,
)

from .game_llm_agent import GameLLMAgent
from .host_agent_adapter import HostAgentAdapter
from .llm_gateway import LLMGateway
from .memory_reader import MemoryReaderManager
from .ocr_reader import OcrReaderManager
from .models import (
    DATA_SOURCE_NONE,
    MODE_COMPANION,
    MODES,
    STATE_ACTIVE,
    STATE_ERROR,
    STORE_BOUND_GAME_ID,
    STORE_DEDUPE_WINDOW,
    STORE_EVENTS_BYTE_OFFSET,
    STORE_EVENTS_FILE_SIZE,
    STORE_LAST_ERROR,
    STORE_LAST_SEQ,
    STORE_MODE,
    STORE_OCR_CAPTURE_PROFILES,
    STORE_PUSH_NOTIFICATIONS,
    STORE_SESSION_ID,
    json_copy,
    make_error,
)
from .reader import tail_events_jsonl, warmup_replay_events
from .service import (
    apply_event_to_histories,
    apply_event_to_snapshot,
    apply_input_degraded_result,
    build_active_session_meta,
    build_config,
    build_explain_context,
    build_history_payload,
    build_snapshot_payload,
    build_status_payload,
    build_suggest_context,
    build_suggest_degraded_result,
    build_summarize_context,
    choose_candidate,
    derive_connection_state,
    filter_memory_reader_candidates,
    filter_ocr_reader_candidates,
    next_poll_interval_for_state,
    rebuild_histories_from_events,
    scan_session_candidates,
)
from .state import GalgameSharedState, build_initial_state
from .store import GalgameStore
from .textractor_support import install_textractor
from .ui_api import build_open_ui_payload


@neko_plugin
class GalgamePlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._state_lock = threading.Lock()
        self._textractor_install_lock = threading.Lock()
        self._cfg = None
        self._state = build_initial_state(mode=MODE_COMPANION, push_notifications=True)
        self._persist = GalgameStore(self.store, self.logger)
        self._host_agent_adapter: HostAgentAdapter | None = None
        self._llm_gateway: LLMGateway | None = None
        self._game_agent: GameLLMAgent | None = None
        self._memory_reader_manager: MemoryReaderManager | None = None
        self._ocr_reader_manager: OcrReaderManager | None = None

    def _snapshot_state(self) -> dict[str, Any]:
        with self._state_lock:
            state = self._state
            return {
                "bound_game_id": state.bound_game_id,
                "available_game_ids": list(state.available_game_ids),
                "mode": state.mode,
                "push_notifications": state.push_notifications,
                "active_game_id": state.active_game_id,
                "active_session_id": state.active_session_id,
                "active_session_meta": json_copy(state.active_session_meta),
                "active_data_source": state.active_data_source,
                "latest_snapshot": json_copy(state.latest_snapshot),
                "history_events": json_copy(state.history_events),
                "history_lines": json_copy(state.history_lines),
                "history_choices": json_copy(state.history_choices),
                "dedupe_window": json_copy(state.dedupe_window),
                "line_buffer": state.line_buffer,
                "stream_reset_pending": state.stream_reset_pending,
                "last_error": json_copy(state.last_error),
                "next_poll_at_monotonic": state.next_poll_at_monotonic,
                "current_connection_state": state.current_connection_state,
                "events_byte_offset": state.events_byte_offset,
                "events_file_size": state.events_file_size,
                "last_seq": state.last_seq,
                "last_seen_data_monotonic": state.last_seen_data_monotonic,
                "warmup_session_id": state.warmup_session_id,
                "memory_reader_runtime": json_copy(state.memory_reader_runtime),
                "ocr_reader_runtime": json_copy(state.ocr_reader_runtime),
                "ocr_capture_profiles": dict(state.ocr_capture_profiles),
                "plugin_error": state.plugin_error,
            }

    def _commit_state(self, payload: dict[str, Any]) -> None:
        with self._state_lock:
            state = self._state
            state.bound_game_id = str(payload["bound_game_id"])
            state.available_game_ids = list(payload["available_game_ids"])
            state.mode = str(payload["mode"])
            state.push_notifications = bool(payload["push_notifications"])
            state.active_game_id = str(payload["active_game_id"])
            state.active_session_id = str(payload["active_session_id"])
            state.active_session_meta = json_copy(payload["active_session_meta"])
            state.active_data_source = str(payload["active_data_source"])
            state.latest_snapshot = json_copy(payload["latest_snapshot"])
            state.history_events = json_copy(payload["history_events"])
            state.history_lines = json_copy(payload["history_lines"])
            state.history_choices = json_copy(payload["history_choices"])
            state.dedupe_window = json_copy(payload["dedupe_window"])
            state.line_buffer = payload["line_buffer"]
            state.stream_reset_pending = bool(payload["stream_reset_pending"])
            state.last_error = json_copy(payload["last_error"])
            state.next_poll_at_monotonic = float(payload["next_poll_at_monotonic"])
            state.current_connection_state = str(payload["current_connection_state"])
            state.events_byte_offset = int(payload["events_byte_offset"])
            state.events_file_size = int(payload["events_file_size"])
            state.last_seq = int(payload["last_seq"])
            state.last_seen_data_monotonic = float(payload["last_seen_data_monotonic"])
            state.warmup_session_id = str(payload["warmup_session_id"])
            state.memory_reader_runtime = json_copy(payload["memory_reader_runtime"])
            state.ocr_reader_runtime = json_copy(payload["ocr_reader_runtime"])
            state.ocr_capture_profiles = dict(payload["ocr_capture_profiles"])
            state.plugin_error = str(payload["plugin_error"])

    def _record_error(self, error: dict[str, Any]) -> None:
        with self._state_lock:
            self._state.last_error = json_copy(error)

    def _persist_preferences(self, *, bound_game_id: str, mode: str, push_notifications: bool) -> None:
        self._persist.persist_preferences(
            bound_game_id=bound_game_id,
            mode=mode,
            push_notifications=push_notifications,
        )

    def _persist_runtime_state(self, payload: dict[str, Any]) -> None:
        self._persist.persist_runtime(
            session_id=str(payload["active_session_id"]),
            events_byte_offset=int(payload["events_byte_offset"]),
            events_file_size=int(payload["events_file_size"]),
            last_seq=int(payload["last_seq"]),
            dedupe_window=list(payload["dedupe_window"]),
            last_error=dict(payload["last_error"]),
        )

    def _set_runtime_from_store(self, restored: dict[str, Any], warnings: list[str]) -> None:
        with self._state_lock:
            self._state = build_initial_state(
                mode=str(restored.get(STORE_MODE, MODE_COMPANION)),
                push_notifications=bool(restored.get(STORE_PUSH_NOTIFICATIONS, True)),
            )
            self._state.bound_game_id = str(restored.get(STORE_BOUND_GAME_ID, ""))
            self._state.active_session_id = str(restored.get(STORE_SESSION_ID, ""))
            self._state.events_byte_offset = int(restored.get(STORE_EVENTS_BYTE_OFFSET, 0))
            self._state.events_file_size = int(restored.get(STORE_EVENTS_FILE_SIZE, 0))
            self._state.last_seq = int(restored.get(STORE_LAST_SEQ, 0))
            self._state.dedupe_window = json_copy(restored.get(STORE_DEDUPE_WINDOW, []))
            self._state.last_error = json_copy(restored.get(STORE_LAST_ERROR, {}))
            self._state.active_data_source = DATA_SOURCE_NONE
            self._state.memory_reader_runtime = {}
            self._state.ocr_reader_runtime = {}
            self._state.ocr_capture_profiles = restored.get(STORE_OCR_CAPTURE_PROFILES, {})
            if warnings and not self._state.last_error:
                self._state.last_error = make_error(
                    "; ".join(warnings),
                    source="store",
                    kind="warning",
                )

    def _current_status_payload(self) -> dict[str, Any]:
        if self._cfg is None:
            return {
                "connection_state": "error",
                "mode": MODE_COMPANION,
                "push_notifications": True,
                "bound_game_id": "",
                "available_game_ids": [],
                "active_session_id": "",
                "active_data_source": DATA_SOURCE_NONE,
                "stream_reset_pending": False,
                "last_seq": 0,
                "last_error": {},
                "summary": "config_not_loaded",
                "phase": "phase_1",
                "memory_reader_enabled": False,
                "memory_reader_runtime": {},
                "ocr_reader_enabled": False,
                "ocr_reader_runtime": {},
                "textractor": {
                    "install_supported": False,
                    "installed": False,
                    "can_install": False,
                    "detected_path": "",
                    "target_dir": "",
                    "expected_executable_path": "",
                    "detail": "config_not_loaded",
                },
            }
        return build_status_payload(self._state, config=self._cfg)

    async def _load_config(self) -> None:
        raw = await self.config.dump(timeout=5.0)
        raw_config = raw if isinstance(raw, dict) else {}
        self._cfg = build_config(raw_config)

    @lifecycle(id="startup")
    async def startup(self, **_):
        try:
            await self._load_config()
        except Exception as exc:
            self._record_error(
                make_error(f"load config failed: {exc}", source="config", kind="error")
            )
            return Err(SdkError(f"failed to load galgame_plugin config: {exc}"))

        try:
            restored, warnings = self._persist.load()
            self._set_runtime_from_store(restored, warnings)
        except Exception as exc:
            self._record_error(
                make_error(f"restore store failed: {exc}", source="store", kind="error")
            )
            return Err(SdkError(f"failed to restore galgame_plugin store: {exc}"))

        self._host_agent_adapter = HostAgentAdapter(self.logger)
        self._llm_gateway = LLMGateway(self, self.logger, self._cfg)
        self._game_agent = GameLLMAgent(
            plugin=self,
            logger=self.logger,
            llm_gateway=self._llm_gateway,
            host_adapter=self._host_agent_adapter,
        )
        self._memory_reader_manager = MemoryReaderManager(
            logger=self.logger,
            config=self._cfg,
        )
        self._ocr_reader_manager = OcrReaderManager(
            logger=self.logger,
            config=self._cfg,
        )
        self._ocr_reader_manager.update_capture_profiles(self._state.ocr_capture_profiles)

        self.register_static_ui("static")
        self.set_list_actions(
            [
                {
                    "id": "open_ui",
                    "kind": "ui",
                    "target": f"/plugin/{self.plugin_id}/ui/",
                    "open_in": "new_tab",
                }
            ]
        )

        await self._poll_bridge(force=True)
        return Ok({"status": "ready", "result": self._current_status_payload()})

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        if self._memory_reader_manager is not None:
            try:
                await self._memory_reader_manager.shutdown()
            except Exception:
                pass
        if self._ocr_reader_manager is not None:
            try:
                await self._ocr_reader_manager.shutdown()
            except Exception:
                pass
        if self._game_agent is not None:
            try:
                await self._game_agent.shutdown()
            except Exception:
                pass
        if self._llm_gateway is not None:
            try:
                await self._llm_gateway.shutdown()
            except Exception:
                pass
        if self._host_agent_adapter is not None:
            try:
                await self._host_agent_adapter.shutdown()
            except Exception:
                pass
        try:
            await self.store.close()
        except Exception:
            pass
        return Ok({"status": "stopped"})

    @timer_interval(id="bridge_tick", seconds=1, auto_start=True)
    async def bridge_tick(self, **_):
        await self._poll_bridge(force=False)
        if self._game_agent is not None:
            try:
                await self._game_agent.tick(self._snapshot_state())
            except Exception as exc:
                self._record_error(
                    make_error(
                        f"game agent tick failed: {exc}",
                        source="game_agent",
                        kind="error",
                    )
                )
        return Ok({"status": "tick"})

    async def _poll_bridge(self, *, force: bool) -> None:
        if self._cfg is None:
            return

        now_monotonic = time.monotonic()
        local = self._snapshot_state()
        if not force and now_monotonic < float(local["next_poll_at_monotonic"]):
            return

        warnings: list[str] = []
        raw_available_game_ids: list[str] = []
        raw_candidates: dict[str, Any] = {}
        memory_reader_runtime = json_copy(local.get("memory_reader_runtime") or {})
        ocr_reader_runtime = json_copy(local.get("ocr_reader_runtime") or {})

        try:
            raw_available_game_ids, raw_candidates, scan_warnings = scan_session_candidates(
                self._cfg.bridge_root
            )
            warnings.extend(scan_warnings)
        except Exception as exc:
            local["plugin_error"] = f"scan bridge root failed: {exc}"
            local["available_game_ids"] = []
            local["current_connection_state"] = STATE_ERROR
            local["last_error"] = make_error(
                local["plugin_error"], source="bridge_scan", kind="error"
            )
            interval = next_poll_interval_for_state(
                local["current_connection_state"],
                stream_reset_pending=bool(local["stream_reset_pending"]),
                config=self._cfg,
            )
            local["next_poll_at_monotonic"] = now_monotonic + interval
            self._commit_state(local)
            try:
                self._persist_runtime_state(local)
            except Exception:
                pass
            return

        bridge_sdk_available = any(
            candidate.data_source == "bridge_sdk"
            for candidate in raw_candidates.values()
        )

        if self._memory_reader_manager is not None:
            self._memory_reader_manager.update_config(self._cfg)
            try:
                memory_reader_tick = await self._memory_reader_manager.tick(
                    bridge_sdk_available=bridge_sdk_available,
                )
                warnings.extend(memory_reader_tick.warnings)
                memory_reader_runtime = memory_reader_tick.runtime
                if memory_reader_tick.should_rescan:
                    (
                        raw_available_game_ids,
                        raw_candidates,
                        rescan_warnings,
                    ) = scan_session_candidates(self._cfg.bridge_root)
                    warnings.extend(rescan_warnings)
            except Exception as exc:
                warnings.append(f"memory_reader tick failed: {exc}")

        if self._ocr_reader_manager is not None:
            self._ocr_reader_manager.update_config(self._cfg)
            try:
                ocr_reader_tick = await self._ocr_reader_manager.tick(
                    bridge_sdk_available=bridge_sdk_available,
                    memory_reader_runtime=memory_reader_runtime,
                )
                warnings.extend(ocr_reader_tick.warnings)
                ocr_reader_runtime = ocr_reader_tick.runtime
                if ocr_reader_tick.should_rescan:
                    (
                        raw_available_game_ids,
                        raw_candidates,
                        rescan_warnings,
                    ) = scan_session_candidates(self._cfg.bridge_root)
                    warnings.extend(rescan_warnings)
            except Exception as exc:
                warnings.append(f"ocr_reader tick failed: {exc}")

        local["memory_reader_runtime"] = memory_reader_runtime
        local["ocr_reader_runtime"] = ocr_reader_runtime
        available_game_ids, candidates = filter_memory_reader_candidates(
            raw_available_game_ids,
            raw_candidates,
            runtime=memory_reader_runtime,
        )
        available_game_ids, candidates = filter_ocr_reader_candidates(
            available_game_ids,
            candidates,
            runtime=ocr_reader_runtime,
        )
        local["available_game_ids"] = available_game_ids

        keep_current = (
            not local["bound_game_id"]
            and local["current_connection_state"] == STATE_ACTIVE
            and bool(local["active_game_id"])
        )
        candidate = choose_candidate(
            candidates,
            bound_game_id=str(local["bound_game_id"]),
            current_game_id=str(local["active_game_id"]),
            keep_current=keep_current,
        )

        if candidate is not None:
            session = candidate.session
            session_id = str(session.get("session_id") or "")
            session_changed = (
                candidate.game_id != local["active_game_id"]
                or session_id != local["active_session_id"]
            )
            restore_cursor = (
                not session_changed
                and local["events_byte_offset"] > 0
                and local["active_session_id"] == session_id
            )
            warmup_needed = session_id != local["warmup_session_id"] or session_changed

            local["active_game_id"] = candidate.game_id
            local["active_session_id"] = session_id
            local["active_session_meta"] = build_active_session_meta(candidate)
            local["active_data_source"] = candidate.data_source
            local["latest_snapshot"] = json_copy(session.get("state", {}))

            if warmup_needed:
                end_offset = int(local["events_byte_offset"]) if restore_cursor else None
                warmup_events = warmup_replay_events(
                    candidate.events_path,
                    bytes_limit=self._cfg.warmup_replay_bytes_limit,
                    events_limit=self._cfg.warmup_replay_events_limit,
                    end_offset=end_offset,
                )
                base_dedupe = (
                    list(local["dedupe_window"]) if restore_cursor else []
                )
                (
                    local["history_events"],
                    local["history_lines"],
                    local["history_choices"],
                    local["dedupe_window"],
                    local["latest_snapshot"],
                ) = rebuild_histories_from_events(
                    events=warmup_events,
                    snapshot=local["latest_snapshot"],
                    dedupe_window=base_dedupe,
                    config=self._cfg,
                    game_id=candidate.game_id,
                )
                try:
                    file_size = candidate.events_path.stat().st_size
                except OSError:
                    file_size = 0
                if restore_cursor and int(local["events_byte_offset"]) <= file_size:
                    local["events_file_size"] = file_size
                    local["last_seq"] = int(local["last_seq"])
                else:
                    local["events_byte_offset"] = file_size
                    local["events_file_size"] = file_size
                    local["last_seq"] = max(
                        int(session.get("last_seq") or 0),
                        max((int(event.get("seq") or 0) for event in warmup_events), default=0),
                    )
                local["line_buffer"] = b""
                local["stream_reset_pending"] = False
                local["warmup_session_id"] = session_id
                local["last_seen_data_monotonic"] = now_monotonic

            if int(session.get("last_seq") or 0) > int(local["last_seq"]):
                local["last_seen_data_monotonic"] = now_monotonic

            read_offset = 0 if local["stream_reset_pending"] else int(local["events_byte_offset"])
            read_buffer = b"" if local["stream_reset_pending"] else bytes(local["line_buffer"])
            tail = tail_events_jsonl(
                candidate.events_path,
                offset=read_offset,
                line_buffer=read_buffer,
            )
            warnings.extend(tail.errors)

            if tail.reset_detected:
                local["stream_reset_pending"] = True
                local["line_buffer"] = b""
                local["events_file_size"] = tail.file_size
            else:
                confirm_reset = False
                if local["stream_reset_pending"] and tail.events:
                    first = tail.events[0]
                    first_seq = int(first.get("seq") or 0)
                    first_session_id = str(first.get("session_id") or "")
                    confirm_reset = first_seq == 1 and (
                        first_session_id != local["active_session_id"]
                        or int(local["last_seq"]) > 0
                    )

                if confirm_reset:
                    local["history_events"] = []
                    local["history_lines"] = []
                    local["history_choices"] = []
                    local["dedupe_window"] = []
                    local["line_buffer"] = b""
                    local["events_byte_offset"] = 0
                    local["last_seq"] = 0
                    local["stream_reset_pending"] = False

                if not local["stream_reset_pending"]:
                    for event in tail.events:
                        if str(event.get("session_id") or "") != local["active_session_id"]:
                            continue
                        seq = int(event.get("seq") or 0)
                        if seq <= int(local["last_seq"]):
                            continue
                        apply_event_to_histories(
                            history_events=local["history_events"],
                            history_lines=local["history_lines"],
                            history_choices=local["history_choices"],
                            dedupe_window=local["dedupe_window"],
                            event=event,
                            config=self._cfg,
                            game_id=candidate.game_id,
                        )
                        local["latest_snapshot"] = apply_event_to_snapshot(
                            local["latest_snapshot"], event
                        )
                        local["last_seq"] = seq
                        local["last_seen_data_monotonic"] = now_monotonic

                    local["events_byte_offset"] = tail.next_offset
                    local["events_file_size"] = tail.file_size
                    local["line_buffer"] = tail.line_buffer
        else:
            local["active_data_source"] = DATA_SOURCE_NONE
            if not local["bound_game_id"]:
                local["active_game_id"] = ""
                local["active_session_id"] = ""
                local["active_session_meta"] = {}
            local["line_buffer"] = b""

        if warnings:
            local["last_error"] = make_error(
                "; ".join(warnings[:3]),
                source="bridge_reader",
                kind="warning",
            )
        elif (
            isinstance(local.get("last_error"), dict)
            and str(local["last_error"].get("kind") or "") == "warning"
            and not str(local.get("plugin_error") or "")
        ):
            local["last_error"] = {}

        local["current_connection_state"] = derive_connection_state(
            bridge_root=self._cfg.bridge_root,
            plugin_error=str(local["plugin_error"]),
            active_session_id=str(local["active_session_id"]),
            last_seen_data_monotonic=float(local["last_seen_data_monotonic"]),
            now_monotonic=now_monotonic,
            stale_after_seconds=self._cfg.stale_after_seconds,
            stream_reset_pending=bool(local["stream_reset_pending"]),
        )
        interval = next_poll_interval_for_state(
            local["current_connection_state"],
            stream_reset_pending=bool(local["stream_reset_pending"]),
            config=self._cfg,
        )
        local["next_poll_at_monotonic"] = now_monotonic + interval
        self._commit_state(local)

        try:
            self._persist_runtime_state(local)
        except Exception as exc:
            self._record_error(
                make_error(
                    f"persist runtime failed: {exc}",
                    source="store",
                    kind="error",
                )
            )

    @plugin_entry(
        id="galgame_get_status",
        name="获取 galgame 插件状态",
        description="返回当前 bridge 连接状态、绑定游戏、最近错误与模式。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["summary"],
    )
    async def galgame_get_status(self, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        with self._state_lock:
            payload = build_status_payload(self._state, config=self._cfg)
        return Ok(payload)

    @plugin_entry(
        id="galgame_install_textractor",
        name="安装 Textractor",
        description="检测并下载安装 TextractorCLI.exe，随后刷新 galgame_plugin 的桥接与读内存状态。",
        input_schema={
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False},
            },
        },
        timeout=180.0,
        llm_result_fields=["summary"],
    )
    async def galgame_install_textractor(self, force: bool = False, **_):
        if self._cfg is None:
            return Err(SdkError("galgame_plugin is not configured"))
        if not self._textractor_install_lock.acquire(blocking=False):
            return Err(SdkError("Textractor install is already in progress"))
        current_run_id = str(getattr(self.ctx, "run_id", "") or "").strip()

        async def _progress_update(event: dict[str, Any]) -> None:
            if not current_run_id:
                return
            await self.run_update(
                run_id=current_run_id,
                status="running",
                progress=float(event.get("progress") or 0.0),
                stage=str(event.get("phase") or ""),
                message=str(event.get("message") or ""),
                metrics={
                    "phase": str(event.get("phase") or ""),
                    "downloaded_bytes": int(event.get("downloaded_bytes") or 0),
                    "total_bytes": int(event.get("total_bytes") or 0),
                    "resume_from": int(event.get("resume_from") or 0),
                    "asset_name": str(event.get("asset_name") or ""),
                    "release_name": str(event.get("release_name") or ""),
                },
            )
        try:
            install_result = await install_textractor(
                logger=self.logger,
                configured_path=self._cfg.memory_reader_textractor_path,
                install_target_dir_raw=self._cfg.memory_reader_install_target_dir,
                release_api_url=self._cfg.memory_reader_install_release_api_url,
                timeout_seconds=self._cfg.memory_reader_install_timeout_seconds,
                force=bool(force),
                task_id=current_run_id or None,
                progress_callback=_progress_update,
            )
            await self._poll_bridge(force=True)
            return Ok(
                {
                    "summary": str(install_result.get("summary") or "Textractor 安装完成"),
                    "install_result": install_result,
                    "status": self._current_status_payload(),
                }
            )
        except Exception as exc:
            return Err(SdkError(f"Textractor install failed: {exc}"))
        finally:
            self._textractor_install_lock.release()

    @plugin_entry(
        id="galgame_get_snapshot",
        name="获取 galgame 快照",
        description="返回当前游戏快照和 stale 状态。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["snapshot"],
    )
    async def galgame_get_snapshot(self, **_):
        with self._state_lock:
            payload = build_snapshot_payload(self._state)
        return Ok(payload)

    @plugin_entry(
        id="galgame_get_history",
        name="获取 galgame 历史",
        description="返回最近事件、稳定台词历史和选项历史。",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 50, "minimum": 1},
                "include_events": {"type": "boolean", "default": True},
            },
        },
        llm_result_fields=["stable_lines", "choices"],
    )
    async def galgame_get_history(self, limit: int = 50, include_events: bool = True, **_):
        with self._state_lock:
            payload = build_history_payload(
                self._state,
                limit=max(1, int(limit)),
                include_events=bool(include_events),
            )
        return Ok(payload)

    @plugin_entry(
        id="galgame_set_mode",
        name="设置 galgame 模式",
        description="设置 silent / companion / choice_advisor 模式，并可选更新通知开关。",
        input_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": sorted(MODES)},
                "push_notifications": {"type": "boolean"},
            },
            "required": ["mode"],
        },
        llm_result_fields=["summary"],
    )
    async def galgame_set_mode(self, mode: str, push_notifications: bool | None = None, **_):
        if mode not in MODES:
            return Err(SdkError(f"invalid galgame mode: {mode!r}"))

        with self._state_lock:
            self._state.mode = mode
            if push_notifications is not None:
                self._state.push_notifications = bool(push_notifications)
            payload = {
                "mode": self._state.mode,
                "push_notifications": self._state.push_notifications,
                "summary": f"mode={self._state.mode} push_notifications={self._state.push_notifications}",
            }
            bound_game_id = self._state.bound_game_id
            persist_push = self._state.push_notifications

        try:
            self._persist_preferences(
                bound_game_id=bound_game_id,
                mode=mode,
                push_notifications=persist_push,
            )
        except Exception as exc:
            return Err(SdkError(f"persist mode failed: {exc}"))
        return Ok(payload)

    @plugin_entry(
        id="galgame_bind_game",
        name="绑定 galgame 游戏",
        description="绑定指定 game_id；传空字符串清除手动绑定并恢复自动选择。",
        input_schema={
            "type": "object",
            "properties": {"game_id": {"type": "string", "default": ""}},
            "required": ["game_id"],
        },
        llm_result_fields=["summary"],
    )
    async def galgame_bind_game(self, game_id: str, **_):
        normalized = game_id.strip()
        with self._state_lock:
            available_game_ids = list(self._state.available_game_ids)
        if normalized and normalized not in available_game_ids:
            return Err(SdkError(f"unknown game_id: {normalized!r}"))

        with self._state_lock:
            self._state.bound_game_id = normalized
            bound_game_id = self._state.bound_game_id
            mode = self._state.mode
            push_notifications = self._state.push_notifications

        try:
            self._persist_preferences(
                bound_game_id=bound_game_id,
                mode=mode,
                push_notifications=push_notifications,
            )
        except Exception as exc:
            return Err(SdkError(f"persist binding failed: {exc}"))

        await self._poll_bridge(force=True)
        with self._state_lock:
            payload = {
                "bound_game_id": self._state.bound_game_id,
                "active_session_id": self._state.active_session_id,
                "summary": f"bound_game_id={self._state.bound_game_id or '(auto)'} active_session_id={self._state.active_session_id}",
            }
        return Ok(payload)

    @plugin_entry(
        id="galgame_open_ui",
        name="打开 galgame UI",
        description="返回 galgame_plugin 静态 UI 的访问路径。",
        input_schema={"type": "object", "properties": {}},
        llm_result_fields=["message"],
    )
    async def galgame_open_ui(self, **_):
        payload = build_open_ui_payload(
            plugin_id=self.plugin_id,
            available=self.get_static_ui_config() is not None,
        )
        return Ok(payload)

    @plugin_entry(
        id="galgame_explain_line",
        name="解释当前或指定台词",
        description="对当前快照或指定 line_id 对应的台词进行解释。",
        input_schema={
            "type": "object",
            "properties": {"line_id": {"type": "string", "default": ""}},
        },
        timeout=45.0,
        llm_result_fields=["explanation", "diagnostic"],
    )
    async def galgame_explain_line(self, line_id: str = "", **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        try:
            context = build_explain_context(local, line_id=line_id.strip())
        except ValueError as exc:
            return Err(SdkError(str(exc)))
        payload = apply_input_degraded_result(
            await self._llm_gateway.explain_line(context),
            context=context,
        )
        payload["line_id"] = str(context.get("line_id") or "")
        payload["speaker"] = str(context.get("speaker") or "")
        payload["text"] = str(context.get("text") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_summarize_scene",
        name="总结当前场景",
        description="总结当前场景或指定 scene_id 的最近剧情进展。",
        input_schema={
            "type": "object",
            "properties": {"scene_id": {"type": "string", "default": ""}},
        },
        timeout=45.0,
        llm_result_fields=["summary", "diagnostic"],
    )
    async def galgame_summarize_scene(self, scene_id: str = "", **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        context = build_summarize_context(local, scene_id=scene_id.strip())
        payload = apply_input_degraded_result(
            await self._llm_gateway.summarize_scene(context),
            context=context,
        )
        payload["scene_id"] = str(context.get("scene_id") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_suggest_choice",
        name="建议当前选项",
        description="对当前可见选项给出推荐顺位与理由。",
        input_schema={"type": "object", "properties": {}},
        timeout=45.0,
        llm_result_fields=["choices", "diagnostic"],
    )
    async def galgame_suggest_choice(self, **_):
        if self._llm_gateway is None:
            return Err(SdkError("galgame_plugin llm_gateway is not initialized"))
        local = self._snapshot_state()
        context = build_suggest_context(local)
        if not context["visible_choices"]:
            return Ok(
                apply_input_degraded_result(
                    build_suggest_degraded_result(
                        context,
                        diagnostic="gateway_unavailable: no visible choices",
                    ),
                    context=context,
                )
            )
        payload = apply_input_degraded_result(
            await self._llm_gateway.suggest_choice(context),
            context=context,
        )
        payload["scene_id"] = str(context.get("scene_id") or "")
        return Ok(payload)

    @plugin_entry(
        id="galgame_agent_command",
        name="向 Game LLM Agent 发送指令",
        description="查询 Agent 状态、上下文、发送消息或控制待机。",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "query_status",
                        "query_context",
                        "send_message",
                        "set_standby",
                    ],
                },
                "message": {"type": "string", "default": ""},
                "context_query": {"type": "string", "default": ""},
                "standby": {"type": "boolean"},
            },
            "required": ["action"],
        },
        timeout=45.0,
        llm_result_fields=["result", "status"],
    )
    async def galgame_agent_command(
        self,
        action: str,
        message: str = "",
        context_query: str = "",
        standby: bool | None = None,
        **_,
    ):
        if self._game_agent is None:
            return Err(SdkError("galgame_plugin game agent is not initialized"))
        local = self._snapshot_state()
        if action == "query_status":
            return Ok(await self._game_agent.query_status(local))
        if action == "query_context":
            if not context_query.strip():
                return Err(SdkError("context_query is required for query_context"))
            return Ok(
                await self._game_agent.query_context(
                    local,
                    context_query=context_query.strip(),
                )
            )
        if action == "send_message":
            if not message.strip():
                return Err(SdkError("message is required for send_message"))
            return Ok(
                await self._game_agent.send_message(
                    local,
                    message=message.strip(),
                )
            )
        if action == "set_standby":
            if standby is None:
                return Err(SdkError("standby is required for set_standby"))
            return Ok(await self._game_agent.set_standby(local, standby=bool(standby)))
        return Err(SdkError(f"unsupported agent action: {action!r}"))


GalgameBridgePlugin = GalgamePlugin
