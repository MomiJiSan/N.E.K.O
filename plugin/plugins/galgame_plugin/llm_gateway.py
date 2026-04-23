from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from plugin.sdk.shared.models import Err

from .llm_backend import GalgameLLMBackend
from .models import json_copy
from .service import (
    build_explain_degraded_result,
    build_local_scene_summary,
    build_suggest_degraded_result,
    build_summarize_degraded_result,
)

_EXPLAIN_EVIDENCE_TYPES = frozenset({"current_line", "history_line", "choice"})
_KEY_POINT_TYPES = frozenset({"plot", "emotion", "decision", "reveal", "objective"})


class LLMGateway:
    def __init__(self, plugin, logger, config, *, backend: GalgameLLMBackend | None = None) -> None:
        self._plugin = plugin
        self._logger = logger
        self._config = config
        self._backend = backend or GalgameLLMBackend(logger)
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._active_calls = 0

    def update_config(self, config) -> None:
        self._config = config

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(self._inflight.values())
            self._inflight.clear()
            self._cache.clear()
            self._active_calls = 0
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._backend.shutdown()

    async def explain_line(self, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_cached(
            operation="explain_line",
            context=context,
            validate=self._validate_explain_result,
            degraded=lambda diagnostic: build_explain_degraded_result(
                context,
                diagnostic=diagnostic,
            ),
        )

    async def summarize_scene(self, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_cached(
            operation="summarize_scene",
            context=context,
            validate=self._validate_summarize_result,
            degraded=lambda diagnostic: build_summarize_degraded_result(
                context,
                diagnostic=diagnostic,
            ),
        )

    async def suggest_choice(self, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_cached(
            operation="suggest_choice",
            context=context,
            validate=lambda raw: self._validate_suggest_result(raw, context=context),
            degraded=lambda diagnostic: build_suggest_degraded_result(
                context,
                diagnostic=diagnostic,
            ),
        )

    async def agent_reply(self, context: dict[str, Any]) -> dict[str, Any]:
        return await self._invoke_cached(
            operation="agent_reply",
            context=context,
            validate=self._validate_agent_reply_result,
            degraded=lambda diagnostic: self._build_agent_reply_fallback(
                context,
                diagnostic=diagnostic,
            ),
        )

    async def _invoke_cached(
        self,
        *,
        operation: str,
        context: dict[str, Any],
        validate: Callable[[dict[str, Any]], dict[str, Any]],
        degraded: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        fingerprint = f"{operation}:{repr(context)}"
        now = time.monotonic()
        wait_task: asyncio.Task[dict[str, Any]] | None = None

        async with self._lock:
            cached = self._cache.get(fingerprint)
            if cached is not None and cached[0] > now:
                return json_copy(cached[1])

            in_flight = self._inflight.get(fingerprint)
            if in_flight is not None:
                wait_task = in_flight
            else:
                if self._active_calls >= int(self._config.llm_max_in_flight):
                    return degraded("busy: throttled by llm_max_in_flight")

                self._active_calls += 1
                wait_task = asyncio.create_task(
                    self._perform_call(
                        fingerprint=fingerprint,
                        operation=operation,
                        context=context,
                        validate=validate,
                        degraded=degraded,
                    )
                )
                self._inflight[fingerprint] = wait_task

        return json_copy(await wait_task)

    async def _perform_call(
        self,
        *,
        fingerprint: str,
        operation: str,
        context: dict[str, Any],
        validate: Callable[[dict[str, Any]], dict[str, Any]],
        degraded: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            result = await self._call_target(
                operation=operation,
                context=context,
                validate=validate,
                degraded=degraded,
            )
            ttl = max(0.0, float(self._config.llm_request_cache_ttl_seconds))
            async with self._lock:
                if ttl > 0:
                    self._cache[fingerprint] = (time.monotonic() + ttl, json_copy(result))
            return result
        finally:
            async with self._lock:
                self._inflight.pop(fingerprint, None)
                self._active_calls = max(0, self._active_calls - 1)

    async def _call_target(
        self,
        *,
        operation: str,
        context: dict[str, Any],
        validate: Callable[[dict[str, Any]], dict[str, Any]],
        degraded: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        target_entry_ref = str(self._config.llm_target_entry_ref or "").strip()
        if not target_entry_ref:
            return await self._call_internal_backend(
                operation=operation,
                context=context,
                validate=validate,
                degraded=degraded,
            )

        try:
            response = await asyncio.wait_for(
                self._plugin.plugins.call_entry(
                    target_entry_ref,
                    params={"operation": operation, "context": context},
                    timeout=float(self._config.llm_call_timeout_seconds),
                ),
                timeout=float(self._config.llm_call_timeout_seconds) + 0.5,
            )
        except asyncio.TimeoutError:
            return degraded("timeout: llm target entry timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return degraded(f"internal_error: {exc}")

        if isinstance(response, Err):
            return degraded(self._normalize_plugin_error(response.error))
        if not isinstance(response.value, dict):
            return degraded("invalid_result: target entry returned non-object payload")

        try:
            return validate(dict(response.value))
        except Exception as exc:
            return degraded(f"invalid_result: {exc}")

    async def _call_internal_backend(
        self,
        *,
        operation: str,
        context: dict[str, Any],
        validate: Callable[[dict[str, Any]], dict[str, Any]],
        degraded: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            response = await asyncio.wait_for(
                self._backend.invoke(
                    operation=operation,
                    context=context,
                ),
                timeout=float(self._config.llm_call_timeout_seconds) + 0.5,
            )
        except asyncio.TimeoutError:
            return degraded("timeout: internal llm backend timed out")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return degraded(f"internal_error: {exc}")

        if not isinstance(response, dict):
            return degraded("invalid_result: internal llm backend returned non-object payload")

        try:
            return validate(dict(response))
        except Exception as exc:
            return degraded(f"invalid_result: {exc}")

    @staticmethod
    def _normalize_plugin_error(error: object) -> str:
        message = str(error or "plugin call failed")
        lowered = message.lower()
        if "timeout" in lowered:
            return f"timeout: {message}"
        if "not found" in lowered or "invalid entry" in lowered:
            return f"gateway_unavailable: {message}"
        return f"internal_error: {message}"

    @staticmethod
    def _validate_explain_result(raw: dict[str, Any]) -> dict[str, Any]:
        explanation = str(raw.get("explanation") or "").strip()
        evidence_obj = raw.get("evidence")
        if not explanation:
            raise ValueError("missing explanation")
        if not isinstance(evidence_obj, list):
            raise ValueError("evidence must be array")

        evidence: list[dict[str, Any]] = []
        for item in evidence_obj:
            if not isinstance(item, dict):
                raise ValueError("evidence item must be object")
            evidence_type = str(item.get("type") or "")
            text = str(item.get("text") or "")
            if evidence_type not in _EXPLAIN_EVIDENCE_TYPES or not text:
                raise ValueError("invalid evidence item")
            evidence.append(
                {
                    "type": evidence_type,
                    "text": text,
                    "line_id": str(item.get("line_id") or ""),
                    "speaker": str(item.get("speaker") or ""),
                    "scene_id": str(item.get("scene_id") or ""),
                    "route_id": str(item.get("route_id") or ""),
                }
            )

        return {
            "degraded": False,
            "explanation": explanation,
            "evidence": evidence,
            "diagnostic": "",
        }

    @staticmethod
    def _validate_summarize_result(raw: dict[str, Any]) -> dict[str, Any]:
        summary = str(raw.get("summary") or "").strip()
        key_points_obj = raw.get("key_points")
        if not summary:
            raise ValueError("missing summary")
        if not isinstance(key_points_obj, list):
            raise ValueError("key_points must be array")

        key_points: list[dict[str, Any]] = []
        for item in key_points_obj:
            if not isinstance(item, dict):
                raise ValueError("key_points item must be object")
            item_type = str(item.get("type") or "")
            text = str(item.get("text") or "")
            if item_type not in _KEY_POINT_TYPES or not text:
                raise ValueError("invalid key point item")
            key_points.append(
                {
                    "type": item_type,
                    "text": text,
                    "line_id": str(item.get("line_id") or ""),
                    "speaker": str(item.get("speaker") or ""),
                    "scene_id": str(item.get("scene_id") or ""),
                    "route_id": str(item.get("route_id") or ""),
                }
            )

        return {
            "degraded": False,
            "summary": summary,
            "key_points": key_points,
            "diagnostic": "",
        }

    @staticmethod
    def _validate_suggest_result(raw: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        choices_obj = raw.get("choices")
        visible = {
            str(item.get("choice_id") or ""): dict(item)
            for item in context.get("visible_choices", [])
            if str(item.get("choice_id") or "")
        }
        if not isinstance(choices_obj, list):
            raise ValueError("choices must be array")

        normalized: list[dict[str, Any]] = []
        seen_choice_ids: set[str] = set()
        seen_ranks: set[int] = set()

        for item in choices_obj:
            if not isinstance(item, dict):
                raise ValueError("choice item must be object")
            choice_id = str(item.get("choice_id") or "")
            if not choice_id or choice_id not in visible:
                raise ValueError(f"unknown choice_id: {choice_id}")
            rank = int(item.get("rank") or 0)
            if rank < 1:
                raise ValueError("rank must be >= 1")
            if choice_id in seen_choice_ids or rank in seen_ranks:
                raise ValueError("duplicate choice rank or id")
            seen_choice_ids.add(choice_id)
            seen_ranks.add(rank)
            fallback_text = str(visible[choice_id].get("text") or "")
            text = str(item.get("text") or fallback_text).strip()
            reason = str(item.get("reason") or "").strip()
            if not text or not reason:
                raise ValueError("choice text/reason missing")
            normalized.append(
                {
                    "choice_id": choice_id,
                    "text": text,
                    "rank": rank,
                    "reason": reason,
                }
            )

        normalized.sort(key=lambda item: item["rank"])
        return {
            "degraded": False,
            "choices": normalized,
            "diagnostic": "",
        }

    @staticmethod
    def _validate_agent_reply_result(raw: dict[str, Any]) -> dict[str, Any]:
        reply = str(raw.get("reply") or raw.get("result") or "").strip()
        if not reply:
            raise ValueError("missing reply")
        return {
            "degraded": False,
            "reply": reply,
            "diagnostic": "",
        }

    @staticmethod
    def _build_agent_reply_fallback(
        context: dict[str, Any],
        *,
        diagnostic: str,
    ) -> dict[str, Any]:
        scene_id = str(context.get("scene_id") or "")
        route_id = str(context.get("route_id") or "")
        latest_line = str(context.get("latest_line") or "")
        recent_lines = context.get("recent_lines")
        selected_choices = context.get("recent_choices")
        summary = build_local_scene_summary(
            scene_id=scene_id,
            route_id=route_id,
            lines=list(recent_lines) if isinstance(recent_lines, list) else [],
            selected_choices=list(selected_choices) if isinstance(selected_choices, list) else [],
            snapshot=context.get("current_snapshot", {}),
        )
        if latest_line:
            reply = f"{summary} 当前台词：{latest_line}"
        else:
            reply = summary
        if str(context.get("prompt") or "").strip():
            reply = f"收到请求「{str(context.get('prompt') or '').strip()}」。{reply}"
        return {
            "degraded": True,
            "reply": reply,
            "diagnostic": diagnostic,
        }
