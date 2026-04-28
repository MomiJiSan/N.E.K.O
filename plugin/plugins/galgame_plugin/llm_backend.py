from __future__ import annotations

import asyncio
import re
from typing import Any

from plugin.sdk.plugin import SdkError

from .llm_prompts import build_prompt_messages
from utils.config_manager import get_config_manager
from utils.file_utils import robust_json_loads
from utils.llm_client import ChatOpenAI, create_chat_llm
from utils.token_tracker import set_call_type

_ALLOWED_OPERATIONS = frozenset(
    {"explain_line", "summarize_scene", "suggest_choice", "agent_reply"}
)
_EXPLAIN_EVIDENCE_TYPES = frozenset({"current_line", "history_line", "choice"})
_KEY_POINT_TYPES = frozenset({"plot", "emotion", "decision", "reveal", "objective"})
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)
_JSON_CORRECTION_MAX_ATTEMPTS = 1
_JSON_CORRECTION_BAD_OUTPUT_MAX_CHARS = 12000
_JSON_CORRECTION_ERROR_MAX_CHARS = 600


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        return _CODE_FENCE_RE.sub("", text).strip()
    return text


def _bounded_prompt_text(value: object, *, max_chars: int) -> str:
    text = _as_str(value, str(value))
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


class GalgameLLMBackend:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._llm_cache: dict[tuple[Any, ...], ChatOpenAI] = {}
        self._llm_cache_lock = asyncio.Lock()

    async def shutdown(self) -> None:
        async with self._llm_cache_lock:
            llms = list(self._llm_cache.values())
            self._llm_cache.clear()
        for llm in llms:
            try:
                await llm.aclose()
            except Exception as exc:
                try:
                    self._logger.warning("galgame LLM client close failed: {}", exc)
                except Exception:
                    pass

    async def invoke(
        self,
        *,
        operation: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if operation not in _ALLOWED_OPERATIONS:
            raise SdkError(f"unsupported operation: {operation!r}")
        if not isinstance(context, dict):
            raise SdkError("context must be an object")
        return await self._invoke_operation(operation, dict(context))

    async def _invoke_operation(
        self,
        operation: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self._build_messages(operation, context)
        raw_text = await self._invoke_json_with_correction(
            operation=operation,
            messages=messages,
        )
        parsed = self._parse_json_object(raw_text)
        return self._normalize_result(operation, parsed, context)

    async def _invoke_json_with_correction(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
    ) -> str:
        raw_text = await self._call_model(
            operation=operation,
            messages=messages,
        )
        last_error: SdkError | None = None
        for attempt in range(_JSON_CORRECTION_MAX_ATTEMPTS + 1):
            try:
                self._parse_json_object(raw_text)
                return raw_text
            except SdkError as exc:
                last_error = exc
                if attempt >= _JSON_CORRECTION_MAX_ATTEMPTS:
                    break

            correction_messages = self._build_json_correction_messages(
                operation=operation,
                messages=messages,
                bad_output=raw_text,
                parse_error=last_error,
                attempt=attempt + 1,
                max_attempts=_JSON_CORRECTION_MAX_ATTEMPTS,
            )
            raw_text = await self._call_model(
                operation=operation,
                messages=correction_messages,
            )

        raise SdkError(
            "llm result is not valid json object after "
            f"{_JSON_CORRECTION_MAX_ATTEMPTS} correction attempt(s): {last_error}"
        )

    def _build_json_correction_messages(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
        bad_output: object,
        parse_error: object,
        attempt: int,
        max_attempts: int,
    ) -> list[dict[str, str]]:
        if operation not in _ALLOWED_OPERATIONS:
            raise SdkError(f"unsupported operation: {operation!r}")
        bounded_bad_output = _bounded_prompt_text(
            bad_output,
            max_chars=_JSON_CORRECTION_BAD_OUTPUT_MAX_CHARS,
        )
        bounded_error = _bounded_prompt_text(
            parse_error,
            max_chars=_JSON_CORRECTION_ERROR_MAX_CHARS,
        )
        correction_messages = list(messages)
        correction_messages.append({"role": "assistant", "content": bounded_bad_output})
        correction_messages.append(
            {
                "role": "user",
                "content": (
                    f"JSON 修正请求 {attempt}/{max_attempts}，operation={operation}。\n"
                    f"解析错误：{bounded_error}\n"
                    "你上一条回复不是合法 JSON 对象。请只返回一个合法 JSON 对象，"
                    "不要带 Markdown、解释或额外文本。"
                ),
            }
        )
        return correction_messages

    async def _call_model(
        self,
        *,
        operation: str,
        messages: list[dict[str, str]],
    ) -> str:
        model_role = "agent" if operation == "agent_reply" else "summary"
        api_config = get_config_manager().get_model_api_config(model_role)
        base_url = _as_str(api_config.get("base_url")).strip()
        model = _as_str(api_config.get("model")).strip()
        api_key = _as_str(api_config.get("api_key")).strip()
        if not base_url or not model:
            raise SdkError(f"missing configured {model_role} model")

        temperature = 0.2 if operation == "agent_reply" else 0.0
        max_completion_tokens = 900 if operation == "agent_reply" else 1200
        cache_key = (
            model_role,
            base_url,
            api_key,
            model,
            temperature,
            max_completion_tokens,
        )
        llm = await self._get_or_create_llm(
            cache_key=cache_key,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        set_call_type("agent" if model_role == "agent" else "summary")
        ainvoke = getattr(llm, "ainvoke", None)
        if callable(ainvoke):
            response = await ainvoke(messages)
        else:
            response = await asyncio.to_thread(llm.invoke, messages)
        return _as_str(getattr(response, "content", ""), str(response))

    async def _get_or_create_llm(
        self,
        *,
        cache_key: tuple[Any, ...],
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_completion_tokens: int,
    ) -> ChatOpenAI:
        async with self._llm_cache_lock:
            cached = self._llm_cache.get(cache_key)
            if cached is not None:
                return cached
            llm = create_chat_llm(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                timeout=30.0,
            )
            self._llm_cache[cache_key] = llm
            return llm

    def _build_messages(
        self,
        operation: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        return build_prompt_messages(operation, context)

    def _parse_json_object(self, raw_text: str) -> dict[str, Any]:
        text = _strip_code_fences(raw_text)
        try:
            parsed = robust_json_loads(text)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise SdkError("llm result is not valid json object")
            try:
                parsed = robust_json_loads(match.group(0))
            except Exception as exc:
                raise SdkError(f"llm result is not valid json object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SdkError("llm result must be a json object")
        return dict(parsed)

    def _normalize_result(
        self,
        operation: str,
        raw: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if operation == "explain_line":
            return self._normalize_explain(raw)
        if operation == "summarize_scene":
            return self._normalize_summarize(raw)
        if operation == "suggest_choice":
            return self._normalize_suggest(raw, context)
        return self._normalize_agent_reply(raw)

    def _normalize_explain(self, raw: dict[str, Any]) -> dict[str, Any]:
        explanation = _as_str(raw.get("explanation")).strip()
        if not explanation:
            raise SdkError("missing explanation")
        evidence_items = raw.get("evidence")
        if not isinstance(evidence_items, list):
            raise SdkError("evidence must be array")

        evidence: list[dict[str, Any]] = []
        for item in evidence_items:
            current = _as_dict(item)
            evidence_type = _as_str(current.get("type")).strip()
            text = _as_str(current.get("text")).strip()
            if evidence_type not in _EXPLAIN_EVIDENCE_TYPES or not text:
                continue
            evidence.append(
                {
                    "type": evidence_type,
                    "text": text,
                    "line_id": _as_str(current.get("line_id")),
                    "speaker": _as_str(current.get("speaker")),
                    "scene_id": _as_str(current.get("scene_id")),
                    "route_id": _as_str(current.get("route_id")),
                }
            )
        return {"explanation": explanation, "evidence": evidence}

    def _normalize_summarize(self, raw: dict[str, Any]) -> dict[str, Any]:
        summary = _as_str(raw.get("summary")).strip()
        if not summary:
            raise SdkError("missing summary")
        key_points_obj = raw.get("key_points")
        if not isinstance(key_points_obj, list):
            raise SdkError("key_points must be array")

        key_points: list[dict[str, Any]] = []
        for item in key_points_obj:
            current = _as_dict(item)
            item_type = _as_str(current.get("type")).strip()
            text = _as_str(current.get("text")).strip()
            if item_type not in _KEY_POINT_TYPES or not text:
                continue
            key_points.append(
                {
                    "type": item_type,
                    "text": text,
                    "line_id": _as_str(current.get("line_id")),
                    "speaker": _as_str(current.get("speaker")),
                    "scene_id": _as_str(current.get("scene_id")),
                    "route_id": _as_str(current.get("route_id")),
                }
            )
        return {"summary": summary, "key_points": key_points}

    def _normalize_suggest(
        self,
        raw: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        visible_choices = {
            _as_str(item.get("choice_id")).strip(): dict(item)
            for item in context.get("visible_choices", [])
            if isinstance(item, dict) and _as_str(item.get("choice_id")).strip()
        }
        raw_choices = raw.get("choices")
        if not isinstance(raw_choices, list):
            raise SdkError("choices must be array")

        preliminary: list[tuple[int, int, dict[str, Any]]] = []
        for index, item in enumerate(raw_choices):
            current = _as_dict(item)
            choice_id = _as_str(current.get("choice_id")).strip()
            if choice_id not in visible_choices:
                continue
            reason = _as_str(current.get("reason")).strip()
            if not reason:
                continue
            text = _as_str(current.get("text")).strip() or _as_str(
                visible_choices[choice_id].get("text")
            ).strip()
            if not text:
                continue
            try:
                rank = int(current.get("rank") or index + 1)
            except (TypeError, ValueError):
                rank = index + 1
            preliminary.append(
                (
                    max(1, rank),
                    index,
                    {
                        "choice_id": choice_id,
                        "text": text,
                        "reason": reason,
                    },
                )
            )

        preliminary.sort(key=lambda item: (item[0], item[1]))
        choices: list[dict[str, Any]] = []
        seen_choice_ids: set[str] = set()
        for _, _, item in preliminary:
            choice_id = _as_str(item.get("choice_id"))
            if choice_id in seen_choice_ids:
                continue
            seen_choice_ids.add(choice_id)
            choices.append(
                {
                    "choice_id": choice_id,
                    "text": _as_str(item.get("text")),
                    "rank": len(choices) + 1,
                    "reason": _as_str(item.get("reason")),
                }
            )

        if visible_choices and not choices:
            raise SdkError("model returned no valid visible choice suggestions")
        return {"choices": choices}

    def _normalize_agent_reply(self, raw: dict[str, Any]) -> dict[str, Any]:
        reply = _as_str(raw.get("reply")).strip() or _as_str(raw.get("result")).strip()
        if not reply:
            raise SdkError("missing reply")
        return {"reply": reply}
