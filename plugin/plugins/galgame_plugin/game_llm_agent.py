from __future__ import annotations

import asyncio
import time
from typing import Any

from .host_agent_adapter import HostAgentAdapter, HostAgentError
from .models import (
    AGENT_STATUS_ACTIVE,
    AGENT_STATUS_ERROR,
    AGENT_STATUS_STANDBY,
    DATA_SOURCE_BRIDGE_SDK,
    json_copy,
    sanitize_snapshot_state,
)
from .service import (
    build_choice_signature,
    build_local_scene_summary,
    build_snapshot_signature,
    build_suggest_context,
    build_summarize_context,
    latest_selected_choice,
    mode_allows_agent_push,
    mode_allows_choice_push,
)


class GameLLMAgent:
    _DIALOGUE_ADVANCE_VARIANTS = (
        {
            "id": "advance_enter",
            "instruction": (
                "Focus the visual novel window. If a dialogue line is visible and no menu choices "
                "are open, press Enter exactly once. Stop immediately after the single input."
            ),
        },
        {
            "id": "advance_click",
            "instruction": (
                "Focus the visual novel window. If a dialogue line or continue prompt is visible "
                "and no menu choices are open, click the usual continue area exactly once, then stop."
            ),
        },
        {
            "id": "advance_space",
            "instruction": (
                "Focus the visual novel window. If a dialogue line is waiting to advance and no "
                "menu choices are open, press Space exactly once. If Space is clearly not appropriate, "
                "click the continue area once instead, then stop."
            ),
        },
    )
    _RECOVER_UI_VARIANTS = (
        {
            "id": "recover_focus",
            "instruction": (
                "Bring the visual novel window to the foreground. If a backlog, history, auto, skip, "
                "or system overlay is open above the game, dismiss that overlay exactly once and stop. "
                "Do not select branch choices."
            ),
        },
        {
            "id": "recover_overlay",
            "instruction": (
                "Focus the visual novel window. If the game appears blocked by a transient overlay or "
                "menu, close that overlay once using the most normal dismiss action, then stop without "
                "advancing dialogue or selecting choices."
            ),
        },
    )

    def __init__(
        self,
        *,
        plugin,
        logger,
        llm_gateway,
        host_adapter: HostAgentAdapter,
    ) -> None:
        self._plugin = plugin
        self._logger = logger
        self._llm_gateway = llm_gateway
        self._host_adapter = host_adapter
        self._op_lock = asyncio.Lock()
        self._explicit_standby = False
        self._hard_error = ""
        self._hard_error_retryable = False
        self._planning_task: asyncio.Task[dict[str, Any]] | None = None
        self._planning_choice_signature: tuple[tuple[str, str, int], ...] = ()
        self._planning_candidates: list[dict[str, Any]] = []
        self._planning_started_at = 0.0
        self._actuation: dict[str, Any] | None = None
        self._pending_strategy: dict[str, Any] | None = None
        self._next_actuation_at = 0.0
        self._scene_memory: list[dict[str, Any]] = []
        self._choice_memory: list[dict[str, Any]] = []
        self._recent_pushes: list[dict[str, Any]] = []
        self._failure_memory: list[dict[str, Any]] = []
        self._suggestion_reasons: dict[str, str] = {}
        self._observed_session_id = ""
        self._observed_scene_id = ""
        self._observed_choice_marker = ""
        self._scene_state = self._build_empty_scene_state()
        self._last_status = AGENT_STATUS_STANDBY

    async def shutdown(self) -> None:
        async with self._op_lock:
            await self._interrupt_current()

    async def tick(self, shared: dict[str, Any]) -> None:
        async with self._op_lock:
            await self._observe(shared)
            now = time.monotonic()
            self._update_scene_state(shared, now)
            self._recover_retryable_error_if_ready(now)

            if self._actuation is not None:
                await self._progress_actuation(shared, now)
                self._last_status = self._compute_status(shared)
                return

            if self._planning_task is not None:
                await self._progress_planning(shared, now)
                self._last_status = self._compute_status(shared)
                return

            if self._compute_status(shared) != AGENT_STATUS_ACTIVE:
                self._last_status = self._compute_status(shared)
                return

            if now < self._next_actuation_at:
                self._last_status = self._compute_status(shared)
                return

            strategy = self._take_pending_strategy()
            if strategy is not None:
                await self._start_actuation_from_strategy(shared, strategy=strategy, now=now)
                self._last_status = self._compute_status(shared)
                return

            snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
            visible_choices = list(snapshot.get("choices", []))
            if self._scene_state["stage"] == "choice_menu" and visible_choices:
                context = build_suggest_context(shared)
                self._planning_choice_signature = build_choice_signature(context["visible_choices"])
                self._planning_candidates = []
                self._planning_started_at = now
                self._planning_task = asyncio.create_task(self._llm_gateway.suggest_choice(context))
                self._last_status = self._compute_status(shared)
                return

            strategy = self._build_scene_strategy(shared, now=now)
            if strategy is not None:
                await self._start_actuation_from_strategy(shared, strategy=strategy, now=now)
            self._last_status = self._compute_status(shared)

    async def query_status(self, shared: dict[str, Any]) -> dict[str, Any]:
        async with self._op_lock:
            interrupted = await self._interrupt_for_status_query()
            await self._observe(shared)
            self._update_scene_state(shared, time.monotonic())
            status = self._compute_status(shared)
            result = self._build_status_result(
                shared,
                status=status,
                interrupted=interrupted,
            )
            self._last_status = status
            return {
                "action": "query_status",
                "result": result,
                "status": status,
                "recent_pushes": json_copy(self._recent_pushes[-20:]),
            }

    async def _interrupt_for_status_query(self) -> bool:
        if self._planning_task is None:
            return False
        self._planning_task.cancel()
        await asyncio.gather(self._planning_task, return_exceptions=True)
        self._planning_task = None
        self._planning_candidates = []
        # Status queries should preempt LLM planning, but they should not tear down
        # an already running host actuation or a retry that is about to resume.
        self._next_actuation_at = time.monotonic() + 0.2
        return True

    async def query_context(self, shared: dict[str, Any], *, context_query: str) -> dict[str, Any]:
        async with self._op_lock:
            await self._interrupt_current()
            await self._observe(shared)
            payload = await self._llm_gateway.agent_reply(
                self._build_agent_reply_context(shared, prompt=context_query)
            )
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "query_context",
                "result": str(payload.get("reply") or ""),
                "status": status,
            }

    async def send_message(self, shared: dict[str, Any], *, message: str) -> dict[str, Any]:
        async with self._op_lock:
            await self._interrupt_current()
            await self._observe(shared)
            payload = await self._llm_gateway.agent_reply(
                self._build_agent_reply_context(shared, prompt=message)
            )
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "send_message",
                "result": str(payload.get("reply") or ""),
                "status": status,
            }

    async def set_standby(self, shared: dict[str, Any], *, standby: bool) -> dict[str, Any]:
        async with self._op_lock:
            self._explicit_standby = bool(standby)
            if self._explicit_standby:
                await self._interrupt_current()
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "set_standby",
                "result": "agent entered standby" if standby else "agent resumed",
                "status": status,
            }

    def _compute_status(self, shared: dict[str, Any]) -> str:
        if self._explicit_standby:
            return AGENT_STATUS_STANDBY
        if not self._is_actionable(shared):
            return AGENT_STATUS_STANDBY
        if self._hard_error:
            return AGENT_STATUS_ERROR
        return AGENT_STATUS_ACTIVE

    @staticmethod
    def _is_actionable(shared: dict[str, Any]) -> bool:
        connection_state = str(shared.get("current_connection_state") or "")
        if connection_state != "active":
            return False
        if not str(shared.get("active_session_id") or ""):
            return False
        if bool(shared.get("stream_reset_pending")):
            return False
        snapshot = shared.get("latest_snapshot")
        return isinstance(snapshot, dict) and bool(snapshot)

    async def _observe(self, shared: dict[str, Any]) -> None:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        session_id = str(shared.get("active_session_id") or "")
        if session_id != self._observed_session_id:
            self._scene_memory.clear()
            self._choice_memory.clear()
            self._recent_pushes.clear()
            self._failure_memory.clear()
            self._suggestion_reasons.clear()
            self._clear_hard_error()
            self._observed_choice_marker = ""
            self._observed_scene_id = str(snapshot.get("scene_id") or "")
            self._observed_session_id = session_id
            self._planning_candidates = []
            self._pending_strategy = None
            self._scene_state = self._build_empty_scene_state()
            return

        current_scene_id = str(snapshot.get("scene_id") or "")
        current_route_id = str(snapshot.get("route_id") or "")
        if current_scene_id and current_scene_id != self._observed_scene_id:
            context = build_summarize_context(shared, scene_id=current_scene_id)
            summary = build_local_scene_summary(
                scene_id=current_scene_id,
                route_id=current_route_id,
                lines=context["recent_lines"],
                selected_choices=context["recent_choices"],
                snapshot=snapshot,
            )
            self._append_bounded(
                self._scene_memory,
                {
                    "scene_id": current_scene_id,
                    "route_id": current_route_id,
                    "summary": summary,
                    "ts": str(snapshot.get("ts") or ""),
                },
                limit=32,
            )
            if self._observed_scene_id and self._should_push_scene(shared):
                self._push_agent_message(
                    shared,
                    kind="scene_summary",
                    content=summary,
                    scene_id=current_scene_id,
                    route_id=current_route_id,
                )
            self._observed_scene_id = current_scene_id

        selected = latest_selected_choice(shared.get("history_choices", []))
        if selected is not None:
            marker = (
                f"{str(selected.get('ts') or '')}:"
                f"{str(selected.get('choice_id') or '')}:"
                f"{str(selected.get('scene_id') or '')}"
            )
            if marker and marker != self._observed_choice_marker:
                choice_id = str(selected.get("choice_id") or "")
                choice_text = str(selected.get("text") or "")
                self._append_bounded(
                    self._choice_memory,
                    {
                        "choice_id": choice_id,
                        "text": choice_text,
                        "scene_id": str(selected.get("scene_id") or ""),
                        "route_id": str(selected.get("route_id") or ""),
                        "ts": str(selected.get("ts") or ""),
                    },
                    limit=64,
                )
                reason = self._suggestion_reasons.pop(choice_id, "")
                self._suggestion_reasons.clear()
                if self._should_push_choice(shared) and reason:
                    self._push_agent_message(
                        shared,
                        kind="choice_reason",
                        content=f"已选择「{choice_text}」。推荐理由：{reason}",
                        scene_id=str(selected.get("scene_id") or ""),
                        route_id=str(selected.get("route_id") or ""),
                    )
                self._observed_choice_marker = marker

    def _should_push_scene(self, shared: dict[str, Any]) -> bool:
        return bool(shared.get("push_notifications")) and mode_allows_agent_push(
            str(shared.get("mode") or "")
        )

    def _should_push_choice(self, shared: dict[str, Any]) -> bool:
        return bool(shared.get("push_notifications")) and mode_allows_choice_push(
            str(shared.get("mode") or "")
        )

    def _push_agent_message(
        self,
        shared: dict[str, Any],
        *,
        kind: str,
        content: str,
        scene_id: str,
        route_id: str,
    ) -> None:
        if not content:
            return
        ts = str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._plugin.push_message(
            source=str(getattr(self._plugin, "plugin_id", "") or "galgame_plugin"),
            message_type="proactive_notification",
            description=f"Galgame Agent · {kind}",
            priority=6,
            content=content,
            metadata={
                "kind": kind,
                "scene_id": scene_id,
                "route_id": route_id,
                "ts": ts,
            },
        )
        self._append_bounded(
            self._recent_pushes,
            {
                "ts": ts,
                "kind": kind,
                "content": content,
                "scene_id": scene_id,
                "route_id": route_id,
            },
            limit=20,
        )

    async def _interrupt_current(self) -> None:
        if self._planning_task is not None:
            self._planning_task.cancel()
            await asyncio.gather(self._planning_task, return_exceptions=True)
            self._planning_task = None
            self._planning_candidates = []
        if self._actuation is not None:
            task_id = str(self._actuation.get("task_id") or "")
            if task_id and str(self._actuation.get("state") or "") == "running_host":
                try:
                    await self._host_adapter.cancel_task(task_id)
                except Exception:
                    pass
            self._actuation = None
        self._pending_strategy = None
        self._next_actuation_at = time.monotonic() + 0.2

    async def _progress_planning(self, shared: dict[str, Any], now: float) -> None:
        task = self._planning_task
        if task is None:
            return
        if not task.done():
            return

        self._planning_task = None
        try:
            suggestion = task.result()
        except asyncio.CancelledError:
            self._next_actuation_at = now + 0.2
            return
        except Exception as exc:
            self._logger.warning("galgame choice planning failed: {}", exc)
            suggestion = {"degraded": True, "choices": [], "diagnostic": str(exc)}

        current_choices = list((shared.get("latest_snapshot") or {}).get("choices") or [])
        if build_choice_signature(current_choices) != self._planning_choice_signature:
            self._next_actuation_at = now + 0.2
            return

        candidates = self._build_choice_candidates(current_choices, suggestion)
        self._planning_candidates = json_copy(candidates)
        if not candidates:
            self._next_actuation_at = now + 0.2
            return

        await self._start_actuation_from_strategy(
            shared,
            strategy=self._build_choice_strategy(
                shared,
                candidate_choices=candidates,
                candidate_index=0,
                instruction_variant=0,
            ),
            now=now,
        )

    async def _start_actuation_from_strategy(
        self,
        shared: dict[str, Any],
        *,
        strategy: dict[str, Any],
        now: float,
    ) -> None:
        await self._start_actuation(
            shared,
            kind=str(strategy.get("kind") or ""),
            instruction=str(strategy.get("instruction") or ""),
            suggestion_reason=str(strategy.get("suggestion_reason") or ""),
            now=now,
            choice_id=str(strategy.get("choice_id") or ""),
            strategy_family=str(strategy.get("strategy_family") or ""),
            strategy_id=str(strategy.get("strategy_id") or ""),
            instruction_variant=int(strategy.get("instruction_variant") or 0),
            candidate_choices=list(strategy.get("candidate_choices") or []),
            candidate_index=int(strategy.get("candidate_index") or 0),
            retry_reason=str(strategy.get("retry_reason") or ""),
        )

    async def _start_actuation(
        self,
        shared: dict[str, Any],
        *,
        kind: str,
        instruction: str,
        suggestion_reason: str,
        now: float,
        choice_id: str = "",
        strategy_family: str = "",
        strategy_id: str = "",
        instruction_variant: int = 0,
        candidate_choices: list[dict[str, Any]] | None = None,
        candidate_index: int = 0,
        retry_reason: str = "",
    ) -> None:
        try:
            availability = await self._host_adapter.get_computer_use_availability()
        except HostAgentError as exc:
            self._set_hard_error(str(exc), retryable=True)
            self._next_actuation_at = now + 1.0
            return
        if not bool(availability.get("ready")):
            reasons = availability.get("reasons")
            detail = reasons[0] if isinstance(reasons, list) and reasons else "computer_use unavailable"
            self._set_hard_error(str(detail), retryable=True)
            self._next_actuation_at = now + 1.0
            return

        try:
            started = await self._host_adapter.run_computer_use_instruction(instruction)
        except HostAgentError as exc:
            self._set_hard_error(str(exc), retryable=True)
            self._next_actuation_at = now + 1.0
            return

        task_id = str(started.get("task_id") or "")
        if not task_id:
            self._set_hard_error(f"invalid task response: {started}", retryable=False)
            self._next_actuation_at = now + 1.0
            return

        if choice_id and suggestion_reason:
            self._remember_suggestion_reason(choice_id, suggestion_reason)

        self._clear_hard_error()
        self._actuation = {
            "kind": kind,
            "task_id": task_id,
            "state": "running_host",
            "strategy_family": strategy_family,
            "strategy_id": strategy_id,
            "instruction_variant": instruction_variant,
            "started_at": now,
            "bridge_wait_started_at": 0.0,
            "baseline_last_seq": int(shared.get("last_seq") or 0),
            "baseline_signature": build_snapshot_signature(shared.get("latest_snapshot", {})),
            "baseline_stage": str(self._scene_state.get("stage") or ""),
            "baseline_scene_id": str((shared.get("latest_snapshot") or {}).get("scene_id") or ""),
            "baseline_choice_signature": build_choice_signature(
                list((shared.get("latest_snapshot") or {}).get("choices") or [])
            ),
            "choice_id": choice_id,
            "candidate_choices": json_copy(candidate_choices or []),
            "candidate_index": candidate_index,
            "retry_reason": retry_reason,
        }

    async def _progress_actuation(self, shared: dict[str, Any], now: float) -> None:
        actuation = self._actuation
        if actuation is None:
            return

        if str(actuation.get("state") or "") == "running_host":
            try:
                task = await self._host_adapter.get_task(str(actuation.get("task_id") or ""))
            except HostAgentError as exc:
                self._set_hard_error(str(exc), retryable=True)
                self._actuation = None
                self._next_actuation_at = now + 1.0
                return

            status = str(task.get("status") or "")
            if status in {"queued", "running"}:
                return
            if status == "completed":
                actuation["state"] = "awaiting_bridge"
                actuation["bridge_wait_started_at"] = now
                return

            reason = str(task.get("error") or f"actuation task ended with status={status}")
            retry = self._build_retry_strategy(shared, actuation=actuation, failure_reason=reason)
            self._record_failure(
                kind=str(actuation.get("kind") or ""),
                strategy_id=str(actuation.get("strategy_id") or ""),
                reason=reason,
                scene_id=str((shared.get("latest_snapshot") or {}).get("scene_id") or ""),
            )
            self._actuation = None
            if retry is not None:
                self._clear_hard_error()
                self._pending_strategy = retry
                self._next_actuation_at = now + 0.2
                return
            self._set_hard_error(reason, retryable=False)
            self._next_actuation_at = now + 1.0
            return

        current_signature = build_snapshot_signature(shared.get("latest_snapshot", {}))
        baseline_signature = actuation.get("baseline_signature")
        if (
            current_signature != baseline_signature
            and int(shared.get("last_seq") or 0) >= int(actuation.get("baseline_last_seq") or 0)
        ):
            self._clear_hard_error()
            self._actuation = None
            self._pending_strategy = None
            self._next_actuation_at = now + 0.2
            return

        if now - float(actuation.get("bridge_wait_started_at") or now) > 5.0:
            reason = "bridge state did not change after actuation"
            retry = self._build_retry_strategy(shared, actuation=actuation, failure_reason=reason)
            self._record_failure(
                kind=str(actuation.get("kind") or ""),
                strategy_id=str(actuation.get("strategy_id") or ""),
                reason=reason,
                scene_id=str((shared.get("latest_snapshot") or {}).get("scene_id") or ""),
            )
            self._actuation = None
            if retry is not None:
                self._clear_hard_error()
                self._pending_strategy = retry
                self._next_actuation_at = now + 0.2
                return
            self._set_hard_error(reason, retryable=False)
            self._next_actuation_at = now + 1.0

    def _build_scene_strategy(self, shared: dict[str, Any], *, now: float) -> dict[str, Any] | None:
        stage = str(self._scene_state.get("stage") or "unknown")
        if stage == "scene_transition":
            if now - float(self._scene_state.get("last_scene_change_at") or 0.0) < 0.6:
                return None
            if int(self._scene_state.get("stage_ticks") or 0) < 2:
                return None
            return self._build_recover_strategy(
                shared,
                retry_index=0,
                reason="scene transition appears stuck",
            )
        if stage == "dialogue":
            return self._build_dialogue_strategy(shared, retry_index=0, reason="")
        if stage == "unknown":
            if int(self._scene_state.get("stage_ticks") or 0) < 2:
                return None
            return self._build_recover_strategy(
                shared,
                retry_index=0,
                reason="dialogue state is unclear, try recovering the UI first",
            )
        return None

    def _build_dialogue_strategy(
        self,
        shared: dict[str, Any],
        *,
        retry_index: int,
        reason: str,
    ) -> dict[str, Any] | None:
        if retry_index >= len(self._DIALOGUE_ADVANCE_VARIANTS):
            return None
        variant = self._DIALOGUE_ADVANCE_VARIANTS[retry_index]
        return {
            "kind": "advance",
            "strategy_family": "dialogue",
            "strategy_id": str(variant["id"]),
            "instruction": str(variant["instruction"]),
            "instruction_variant": retry_index,
            "candidate_choices": [],
            "candidate_index": 0,
            "retry_reason": reason,
            "choice_id": "",
            "suggestion_reason": "",
        }

    def _build_recover_strategy(
        self,
        shared: dict[str, Any],
        *,
        retry_index: int,
        reason: str,
    ) -> dict[str, Any] | None:
        if retry_index >= len(self._RECOVER_UI_VARIANTS):
            return None
        variant = self._RECOVER_UI_VARIANTS[retry_index]
        return {
            "kind": "recover",
            "strategy_family": "recover",
            "strategy_id": str(variant["id"]),
            "instruction": str(variant["instruction"]),
            "instruction_variant": retry_index,
            "candidate_choices": [],
            "candidate_index": 0,
            "retry_reason": reason,
            "choice_id": "",
            "suggestion_reason": "",
        }

    def _build_choice_strategy(
        self,
        shared: dict[str, Any],
        *,
        candidate_choices: list[dict[str, Any]],
        candidate_index: int,
        instruction_variant: int,
    ) -> dict[str, Any] | None:
        if not candidate_choices or candidate_index >= len(candidate_choices):
            return None
        if instruction_variant >= 2:
            return None
        candidate = dict(candidate_choices[candidate_index])
        choice_text = str(candidate.get("text") or "")
        choice_index = int(candidate.get("index") or 0) + 1
        if instruction_variant == 0:
            instruction = (
                "A visual novel menu is currently open. Select the option with exact text "
                f"\"{choice_text}\". If exact text matching is unreliable, select visible "
                f"menu item index {choice_index}. After one selection attempt, stop."
            )
        else:
            instruction = (
                "A visual novel menu is currently open. Select visible menu item index "
                f"{choice_index} exactly once. Before clicking, verify the item text matches "
                f"\"{choice_text}\" as closely as possible. After one selection attempt, stop."
            )
        return {
            "kind": "choose",
            "strategy_family": "choice",
            "strategy_id": f"choose_rank_{candidate_index + 1}_variant_{instruction_variant + 1}",
            "instruction": instruction,
            "instruction_variant": instruction_variant,
            "candidate_choices": json_copy(candidate_choices),
            "candidate_index": candidate_index,
            "retry_reason": "",
            "choice_id": str(candidate.get("choice_id") or ""),
            "suggestion_reason": str(candidate.get("reason") or ""),
        }

    def _build_choice_candidates(
        self,
        current_choices: list[dict[str, Any]],
        suggestion: dict[str, Any],
    ) -> list[dict[str, Any]]:
        choices_by_id = {
            str(item.get("choice_id") or ""): dict(item)
            for item in current_choices
            if str(item.get("choice_id") or "")
        }
        candidates: list[dict[str, Any]] = []
        if not bool(suggestion.get("degraded")) and suggestion.get("choices"):
            for item in suggestion["choices"]:
                choice_id = str(item.get("choice_id") or "")
                current = choices_by_id.get(choice_id)
                if current is None:
                    continue
                candidates.append(
                    {
                        **current,
                        "rank": int(item.get("rank") or len(candidates) + 1),
                        "reason": str(item.get("reason") or ""),
                    }
                )
        if not candidates:
            for current in current_choices:
                candidates.append(
                    {
                        **dict(current),
                        "rank": len(candidates) + 1,
                        "reason": "",
                    }
                )
        candidates.sort(
            key=lambda item: (
                int(item.get("rank") or 0),
                int(item.get("index") or 0),
                str(item.get("choice_id") or ""),
            )
        )
        for item in candidates:
            item.pop("rank", None)
        return candidates

    def _build_retry_strategy(
        self,
        shared: dict[str, Any],
        *,
        actuation: dict[str, Any],
        failure_reason: str,
    ) -> dict[str, Any] | None:
        kind = str(actuation.get("kind") or "")
        instruction_variant = int(actuation.get("instruction_variant") or 0)
        if kind == "advance":
            retry = self._build_dialogue_strategy(
                shared,
                retry_index=instruction_variant + 1,
                reason=failure_reason,
            )
            if retry is not None:
                return retry
            return self._build_recover_strategy(shared, retry_index=0, reason=failure_reason)

        if kind == "recover":
            return self._build_recover_strategy(
                shared,
                retry_index=instruction_variant + 1,
                reason=failure_reason,
            )

        if kind == "choose":
            candidate_choices = list(actuation.get("candidate_choices") or [])
            candidate_index = int(actuation.get("candidate_index") or 0)
            retry = self._build_choice_strategy(
                shared,
                candidate_choices=candidate_choices,
                candidate_index=candidate_index,
                instruction_variant=instruction_variant + 1,
            )
            if retry is not None:
                return retry
            retry = self._build_choice_strategy(
                shared,
                candidate_choices=candidate_choices,
                candidate_index=candidate_index + 1,
                instruction_variant=0,
            )
            if retry is not None:
                return retry
            return self._build_recover_strategy(shared, retry_index=0, reason=failure_reason)

        return None

    def _take_pending_strategy(self) -> dict[str, Any] | None:
        if self._pending_strategy is None:
            return None
        strategy = dict(self._pending_strategy)
        self._pending_strategy = None
        return strategy

    def _update_scene_state(self, shared: dict[str, Any], now: float) -> None:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        signature = build_snapshot_signature(snapshot)
        scene_id = str(snapshot.get("scene_id") or "")
        route_id = str(snapshot.get("route_id") or "")
        scene_changed = (
            scene_id != str(self._scene_state.get("scene_id") or "")
            or route_id != str(self._scene_state.get("route_id") or "")
        )
        signature_changed = signature != self._scene_state.get("signature")
        next_stage = self._classify_scene_stage(snapshot, now=now, scene_changed=scene_changed)

        if scene_changed:
            summary_context = build_summarize_context(shared, scene_id=scene_id)
            summary_seed = build_local_scene_summary(
                scene_id=scene_id,
                route_id=route_id,
                lines=summary_context["recent_lines"],
                selected_choices=summary_context["recent_choices"],
                snapshot=snapshot,
            )
            self._scene_state = {
                "scene_id": scene_id,
                "route_id": route_id,
                "signature": signature,
                "stage": next_stage,
                "stage_ticks": 1,
                "same_signature_ticks": 0,
                "last_progress_at": now,
                "last_scene_change_at": now,
                "summary_seed": summary_seed,
            }
            return

        if signature_changed:
            self._scene_state["signature"] = signature
            self._scene_state["same_signature_ticks"] = 0
            self._scene_state["last_progress_at"] = now
        else:
            self._scene_state["same_signature_ticks"] = int(
                self._scene_state.get("same_signature_ticks") or 0
            ) + 1

        previous_stage = str(self._scene_state.get("stage") or "")
        if next_stage != previous_stage:
            self._scene_state["stage"] = next_stage
            self._scene_state["stage_ticks"] = 1
        else:
            self._scene_state["stage_ticks"] = int(self._scene_state.get("stage_ticks") or 0) + 1

        self._scene_state["scene_id"] = scene_id
        self._scene_state["route_id"] = route_id

    def _classify_scene_stage(
        self,
        snapshot: dict[str, Any],
        *,
        now: float,
        scene_changed: bool,
    ) -> str:
        choices = list(snapshot.get("choices", []))
        if bool(snapshot.get("is_menu_open")) and choices:
            return "choice_menu"
        if snapshot.get("text") or snapshot.get("line_id"):
            return "dialogue"
        save_kind = str((snapshot.get("save_context") or {}).get("kind") or "")
        if scene_changed or save_kind in {"load", "rollback"}:
            return "scene_transition"
        if now - float(self._scene_state.get("last_scene_change_at") or 0.0) < 0.6:
            return "scene_transition"
        return "unknown"

    @staticmethod
    def _build_empty_scene_state() -> dict[str, Any]:
        return {
            "scene_id": "",
            "route_id": "",
            "signature": (),
            "stage": "unknown",
            "stage_ticks": 0,
            "same_signature_ticks": 0,
            "last_progress_at": 0.0,
            "last_scene_change_at": 0.0,
            "summary_seed": "",
        }

    def _record_failure(self, *, kind: str, strategy_id: str, reason: str, scene_id: str) -> None:
        self._append_bounded(
            self._failure_memory,
            {
                "kind": kind,
                "strategy_id": strategy_id,
                "reason": reason,
                "scene_id": scene_id,
                "ts": str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            },
            limit=16,
        )

    def _set_hard_error(self, message: str, *, retryable: bool) -> None:
        self._hard_error = message
        self._hard_error_retryable = retryable

    def _clear_hard_error(self) -> None:
        self._hard_error = ""
        self._hard_error_retryable = False

    def _recover_retryable_error_if_ready(self, now: float) -> None:
        if not self._hard_error or not self._hard_error_retryable:
            return
        if now < self._next_actuation_at:
            return
        self._clear_hard_error()

    def _remember_suggestion_reason(self, choice_id: str, reason: str, *, limit: int = 32) -> None:
        if not choice_id or not reason:
            return
        self._suggestion_reasons.pop(choice_id, None)
        self._suggestion_reasons[choice_id] = reason
        while len(self._suggestion_reasons) > limit:
            oldest_key = next(iter(self._suggestion_reasons))
            self._suggestion_reasons.pop(oldest_key, None)

    def _build_agent_reply_context(self, shared: dict[str, Any], *, prompt: str) -> dict[str, Any]:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        latest_line = ""
        if snapshot.get("text"):
            latest_line = f"{str(snapshot.get('speaker') or '旁白')}：{str(snapshot.get('text') or '')}"
        return {
            "prompt": prompt,
            "game_id": str(shared.get("active_game_id") or ""),
            "session_id": str(shared.get("active_session_id") or ""),
            "scene_id": str(snapshot.get("scene_id") or ""),
            "route_id": str(snapshot.get("route_id") or ""),
            "current_snapshot": snapshot,
            "latest_line": latest_line,
            "recent_lines": json_copy(list(shared.get("history_lines", []))[-8:]),
            "recent_choices": json_copy(list(shared.get("history_choices", []))[-8:]),
            "scene_memory": json_copy(self._scene_memory[-8:]),
            "choice_memory": json_copy(self._choice_memory[-8:]),
            "failure_memory": json_copy(self._failure_memory[-8:]),
            "scene_strategy": json_copy(self._scene_state),
            "status": self._compute_status(shared),
        }

    def _build_status_result(
        self,
        shared: dict[str, Any],
        *,
        status: str,
        interrupted: bool,
    ) -> str:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        parts = [
            f"status={status}",
            f"session={str(shared.get('active_session_id') or '') or 'none'}",
            f"scene={str(snapshot.get('scene_id') or '') or 'none'}",
            f"route={str(snapshot.get('route_id') or '') or 'none'}",
            f"line={str(snapshot.get('line_id') or '') or 'none'}",
            f"stage={str(self._scene_state.get('stage') or 'unknown')}",
            f"activity={self._current_activity_label()}",
        ]
        if interrupted:
            parts.append("interrupted=yes")
        if self._hard_error:
            parts.append(f"error={self._hard_error}")
        elif self._explicit_standby:
            parts.append("reason=explicit_standby")
        elif not self._is_actionable(shared):
            parts.append("reason=bridge_inactive")
        return " ".join(parts)

    def _current_activity_label(self) -> str:
        if self._planning_task is not None:
            return "planning"
        if self._actuation is not None:
            kind = str(self._actuation.get("kind") or "unknown")
            state = str(self._actuation.get("state") or "running")
            return f"{kind}:{state}"
        if self._pending_strategy is not None:
            return "retry_pending"
        return "idle"

    async def query_status(self, shared: dict[str, Any]) -> dict[str, Any]:
        async with self._op_lock:
            interrupted = await self._interrupt_for_status_query()
            await self._observe(shared)
            self._update_scene_state(shared, time.monotonic())
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "query_status",
                **self._build_status_payload(
                    shared,
                    status=status,
                    interrupted=interrupted,
                ),
            }

    async def query_context(self, shared: dict[str, Any], *, context_query: str) -> dict[str, Any]:
        async with self._op_lock:
            await self._interrupt_current()
            await self._observe(shared)
            payload = await self._llm_gateway.agent_reply(
                self._build_agent_reply_context(shared, prompt=context_query)
            )
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "query_context",
                "result": str(payload.get("reply") or ""),
                "status": status,
                "degraded": bool(payload.get("degraded")),
                "diagnostic": str(payload.get("diagnostic") or ""),
                "input_source": self._current_input_source(shared),
            }

    async def send_message(self, shared: dict[str, Any], *, message: str) -> dict[str, Any]:
        async with self._op_lock:
            await self._interrupt_current()
            await self._observe(shared)
            payload = await self._llm_gateway.agent_reply(
                self._build_agent_reply_context(shared, prompt=message)
            )
            status = self._compute_status(shared)
            self._last_status = status
            return {
                "action": "send_message",
                "result": str(payload.get("reply") or ""),
                "status": status,
                "degraded": bool(payload.get("degraded")),
                "diagnostic": str(payload.get("diagnostic") or ""),
                "input_source": self._current_input_source(shared),
            }

    async def _observe(self, shared: dict[str, Any]) -> None:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        session_id = str(shared.get("active_session_id") or "")
        if session_id != self._observed_session_id:
            self._scene_memory.clear()
            self._choice_memory.clear()
            self._recent_pushes.clear()
            self._failure_memory.clear()
            self._suggestion_reasons.clear()
            self._clear_hard_error()
            self._observed_choice_marker = ""
            self._observed_scene_id = str(snapshot.get("scene_id") or "")
            self._observed_session_id = session_id
            self._planning_candidates = []
            self._pending_strategy = None
            self._scene_state = self._build_empty_scene_state()
            return

        current_scene_id = str(snapshot.get("scene_id") or "")
        current_route_id = str(snapshot.get("route_id") or "")
        if current_scene_id and current_scene_id != self._observed_scene_id:
            context = build_summarize_context(shared, scene_id=current_scene_id)
            summary = build_local_scene_summary(
                scene_id=current_scene_id,
                route_id=current_route_id,
                lines=context["recent_lines"],
                selected_choices=context["recent_choices"],
                snapshot=snapshot,
            )
            self._append_bounded(
                self._scene_memory,
                {
                    "scene_id": current_scene_id,
                    "route_id": current_route_id,
                    "summary": summary,
                    "ts": str(snapshot.get("ts") or ""),
                },
                limit=32,
            )
            if self._observed_scene_id and self._should_push_scene(shared):
                self._push_agent_message(
                    shared,
                    kind="scene_summary",
                    content=summary,
                    scene_id=current_scene_id,
                    route_id=current_route_id,
                )
            self._observed_scene_id = current_scene_id

        selected = latest_selected_choice(shared.get("history_choices", []))
        if selected is not None:
            marker = (
                f"{str(selected.get('ts') or '')}:"
                f"{str(selected.get('choice_id') or '')}:"
                f"{str(selected.get('scene_id') or '')}"
            )
            if marker and marker != self._observed_choice_marker:
                choice_id = str(selected.get("choice_id") or "")
                choice_text = str(selected.get("text") or "")
                self._append_bounded(
                    self._choice_memory,
                    {
                        "choice_id": choice_id,
                        "text": choice_text,
                        "scene_id": str(selected.get("scene_id") or ""),
                        "route_id": str(selected.get("route_id") or ""),
                        "ts": str(selected.get("ts") or ""),
                    },
                    limit=64,
                )
                reason = self._suggestion_reasons.pop(choice_id, "")
                self._suggestion_reasons.clear()
                if self._should_push_choice(shared) and reason:
                    self._push_agent_message(
                        shared,
                        kind="choice_reason",
                        content=(
                            f"\u5df2\u9009\u62e9\u300c{choice_text}\u300d\u3002"
                            f"\u63a8\u8350\u7406\u7531\uff1a{reason}"
                        ),
                        scene_id=str(selected.get("scene_id") or ""),
                        route_id=str(selected.get("route_id") or ""),
                    )
                self._observed_choice_marker = marker

    def _push_agent_message(
        self,
        shared: dict[str, Any],
        *,
        kind: str,
        content: str,
        scene_id: str,
        route_id: str,
    ) -> None:
        if not content:
            return
        ts = str(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._plugin.push_message(
            source=str(getattr(self._plugin, "plugin_id", "") or "galgame_plugin"),
            message_type="proactive_notification",
            description=f"Galgame Agent | {kind}",
            priority=6,
            content=content,
            metadata={
                "kind": kind,
                "scene_id": scene_id,
                "route_id": route_id,
                "ts": ts,
            },
        )
        self._append_bounded(
            self._recent_pushes,
            {
                "ts": ts,
                "kind": kind,
                "content": content,
                "scene_id": scene_id,
                "route_id": route_id,
            },
            limit=20,
        )

    def _build_agent_reply_context(self, shared: dict[str, Any], *, prompt: str) -> dict[str, Any]:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        latest_line = ""
        if snapshot.get("text"):
            speaker = str(snapshot.get("speaker") or "Narration")
            latest_line = (
                f"{speaker}: "
                f"{str(snapshot.get('text') or '')}"
            )
        return {
            "prompt": prompt,
            "game_id": str(shared.get("active_game_id") or ""),
            "session_id": str(shared.get("active_session_id") or ""),
            "scene_id": str(snapshot.get("scene_id") or ""),
            "route_id": str(snapshot.get("route_id") or ""),
            "current_snapshot": snapshot,
            "latest_line": latest_line,
            "recent_lines": json_copy(list(shared.get("history_lines", []))[-8:]),
            "recent_choices": json_copy(list(shared.get("history_choices", []))[-8:]),
            "scene_memory": json_copy(self._scene_memory[-8:]),
            "choice_memory": json_copy(self._choice_memory[-8:]),
            "failure_memory": json_copy(self._failure_memory[-8:]),
            "scene_strategy": json_copy(self._scene_state),
            "status": self._compute_status(shared),
            "mode": str(shared.get("mode") or ""),
            "input_source": self._current_input_source(shared),
            "push_policy": self._current_push_policy(shared),
            "standby_requested": self._explicit_standby,
        }

    def _build_status_payload(
        self,
        shared: dict[str, Any],
        *,
        status: str,
        interrupted: bool,
    ) -> dict[str, Any]:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        recent_pushes = json_copy(self._recent_pushes[-20:])
        return {
            "result": self._build_status_result(
                shared,
                status=status,
                interrupted=interrupted,
            ),
            "status": status,
            "activity": self._current_activity_label(),
            "reason": self._current_status_reason(shared),
            "error": self._hard_error,
            "session_id": str(shared.get("active_session_id") or ""),
            "scene_id": str(snapshot.get("scene_id") or ""),
            "route_id": str(snapshot.get("route_id") or ""),
            "line_id": str(snapshot.get("line_id") or ""),
            "scene_stage": str(self._scene_state.get("stage") or "unknown"),
            "input_source": self._current_input_source(shared),
            "mode": str(shared.get("mode") or ""),
            "push_notifications": bool(shared.get("push_notifications")),
            "push_policy": self._current_push_policy(shared),
            "actionable": self._is_actionable(shared),
            "standby_requested": self._explicit_standby,
            "interrupted": interrupted,
            "memory_counts": {
                "scene_memory": len(self._scene_memory),
                "choice_memory": len(self._choice_memory),
                "failure_memory": len(self._failure_memory),
                "recent_pushes": len(self._recent_pushes),
            },
            "recent_pushes": recent_pushes,
            "last_push": json_copy(recent_pushes[-1]) if recent_pushes else None,
        }

    def _build_status_result(
        self,
        shared: dict[str, Any],
        *,
        status: str,
        interrupted: bool,
    ) -> str:
        snapshot = sanitize_snapshot_state(shared.get("latest_snapshot", {}))
        parts = [
            f"status={status}",
            f"session={str(shared.get('active_session_id') or '') or 'none'}",
            f"scene={str(snapshot.get('scene_id') or '') or 'none'}",
            f"route={str(snapshot.get('route_id') or '') or 'none'}",
            f"line={str(snapshot.get('line_id') or '') or 'none'}",
            f"stage={str(self._scene_state.get('stage') or 'unknown')}",
            f"activity={self._current_activity_label()}",
            f"input_source={self._current_input_source(shared)}",
            f"push_policy={self._current_push_policy(shared)}",
            f"reason={self._current_status_reason(shared)}",
        ]
        if interrupted:
            parts.append("interrupted=yes")
        if self._hard_error:
            parts.append(f"error={self._hard_error}")
        return " ".join(parts)

    @staticmethod
    def _current_input_source(shared: dict[str, Any]) -> str:
        return str(shared.get("active_data_source") or DATA_SOURCE_BRIDGE_SDK)

    def _current_status_reason(self, shared: dict[str, Any]) -> str:
        if self._hard_error:
            return "hard_error"
        if self._explicit_standby:
            return "explicit_standby"
        if not self._is_actionable(shared):
            return "bridge_inactive"
        if self._planning_task is not None:
            return "planning_choice"
        if self._actuation is not None:
            return (
                f"actuating_{str(self._actuation.get('kind') or 'unknown')}_"
                f"{str(self._actuation.get('state') or 'running')}"
            )
        if self._pending_strategy is not None:
            return "retry_pending"
        return "background_loop_ready"

    def _current_push_policy(self, shared: dict[str, Any]) -> str:
        if not bool(shared.get("push_notifications")):
            return "disabled"
        mode = str(shared.get("mode") or "")
        if mode_allows_choice_push(mode):
            return "selective_scene_and_choice"
        if mode_allows_agent_push(mode):
            return "selective_scene_only"
        return "disabled"

    @staticmethod
    def _append_bounded(items: list[dict[str, Any]], item: dict[str, Any], *, limit: int) -> None:
        items.append(dict(item))
        if len(items) > limit:
            del items[:-limit]
