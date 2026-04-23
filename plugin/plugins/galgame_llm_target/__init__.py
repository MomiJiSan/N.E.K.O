from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from plugin.sdk.plugin import (
    Err,
    NekoPluginBase,
    Ok,
    SdkError,
    lifecycle,
    neko_plugin,
    plugin_entry,
)

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


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _strip_code_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        return _CODE_FENCE_RE.sub("", text).strip()
    return text


@neko_plugin
class GalgameLLMTargetPlugin(NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.file_logger = self.enable_file_logging(log_level="INFO")
        self.logger = self.file_logger
        self._llm_cache: dict[tuple[Any, ...], ChatOpenAI] = {}
        self._llm_cache_lock = asyncio.Lock()

    @lifecycle(id="startup")
    async def startup(self, **_):
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    async def shutdown(self, **_):
        async with self._llm_cache_lock:
            llms = list(self._llm_cache.values())
            self._llm_cache.clear()
        for llm in llms:
            try:
                await llm.aclose()
            except Exception:
                pass
        return Ok({"status": "stopped"})

    @plugin_entry(
        id="galgame_llm_target_invoke",
        name="Galgame LLM Target Invoke",
        description="Internal single-entry LLM target for galgame_bridge Phase 2.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(_ALLOWED_OPERATIONS),
                },
                "context": {"type": "object"},
            },
            "required": ["operation", "context"],
        },
        timeout=45.0,
        llm_result_fields=["reply", "summary", "explanation"],
    )
    async def galgame_llm_target_invoke(
        self,
        operation: str,
        context: dict[str, Any],
        **_,
    ):
        if operation not in _ALLOWED_OPERATIONS:
            return Err(SdkError(f"unsupported operation: {operation!r}"))
        if not isinstance(context, dict):
            return Err(SdkError("context must be an object"))
        try:
            payload = await self._invoke_operation(operation, dict(context))
        except SdkError as exc:
            return Err(exc)
        except Exception as exc:
            return Err(SdkError(f"galgame_llm_target invoke failed: {exc}"))
        return Ok(payload)

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
        try:
            self._parse_json_object(raw_text)
            return raw_text
        except SdkError:
            correction_messages = list(messages)
            correction_messages.append({"role": "assistant", "content": raw_text})
            correction_messages.append(
                {
                    "role": "user",
                    "content": (
                        "你上一条回复不是合法 JSON。请只返回一个合法 JSON 对象，"
                        "不要带 Markdown、解释或额外文本。"
                    ),
                }
            )
            return await self._call_model(
                operation=operation,
                messages=correction_messages,
            )

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
        response = await llm.ainvoke(messages)
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
        if operation == "explain_line":
            return self._build_explain_messages(context)
        if operation == "summarize_scene":
            return self._build_summarize_messages(context)
        if operation == "suggest_choice":
            return self._build_suggest_messages(context)
        return self._build_agent_reply_messages(context)

    def _build_explain_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        example = {
            "explanation": "这句台词表现了角色的犹豫和试探。",
            "evidence": [
                {
                    "type": "current_line",
                    "text": "今天一起回家吗？",
                    "line_id": "line-1",
                    "speaker": "雪乃",
                    "scene_id": "scene-a",
                    "route_id": "",
                }
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是 N.E.K.O 的 galgame 分析后端，是游戏辅助系统，不扮演角色。"
                    "只能依据给定 context 分析，不得虚构 line_id、scene_id 或剧情事实。"
                    "必须只返回一个合法 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "任务：解释当前或指定台词。\n"
                    "要求：\n"
                    "1. explanation 用 1-3 句说明语气、潜台词或剧情作用。\n"
                    "2. evidence 只能引用 context 中已有的线索。\n"
                    "3. evidence.type 只能是 current_line / history_line / choice。\n"
                    "4. 输出必须匹配这个 JSON 结构：\n"
                    f"{_json_dump(example)}\n\n"
                    "context:\n"
                    f"{_json_dump(context)}"
                ),
            },
        ]

    def _build_summarize_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        example = {
            "summary": "这一段剧情在放学后的对话中推进了角色关系。",
            "key_points": [
                {
                    "type": "plot",
                    "text": "主角被邀请一起回家。",
                    "line_id": "line-1",
                    "speaker": "雪乃",
                    "scene_id": "scene-a",
                    "route_id": "",
                }
            ],
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是 N.E.K.O 的 galgame 场景总结后端，是游戏辅助系统，不扮演角色。"
                    "只能依据给定 context 总结，不得补写不存在的剧情。"
                    "必须只返回一个合法 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "任务：总结当前场景。\n"
                    "要求：\n"
                    "1. summary 用 1-3 句概括当前场景的剧情推进。\n"
                    "2. key_points.type 只能是 plot / emotion / decision / reveal / objective。\n"
                    "3. key_points 只允许引用 context 中能支持的事实。\n"
                    "4. 输出必须匹配这个 JSON 结构：\n"
                    f"{_json_dump(example)}\n\n"
                    "context:\n"
                    f"{_json_dump(context)}"
                ),
            },
        ]

    def _build_suggest_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        example = {
            "choices": [
                {
                    "choice_id": "choice-1",
                    "text": "好啊",
                    "rank": 1,
                    "reason": "更符合当前关系升温的剧情方向。",
                },
                {
                    "choice_id": "choice-2",
                    "text": "下次吧",
                    "rank": 2,
                    "reason": "会让关系推进放缓。",
                },
            ]
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是 N.E.K.O 的 galgame 选项建议后端，是游戏辅助系统，不扮演角色。"
                    "只能在给定 visible_choices 中排序，不得发明新的 choice_id。"
                    "必须只返回一个合法 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "任务：对当前可见选项给出推荐顺位。\n"
                    "要求：\n"
                    "1. 只能返回 context.visible_choices 中出现的 choice_id。\n"
                    "2. rank 从 1 开始，越小越推荐。\n"
                    "3. reason 简洁说明推荐依据。\n"
                    "4. 输出必须匹配这个 JSON 结构：\n"
                    f"{_json_dump(example)}\n\n"
                    "context:\n"
                    f"{_json_dump(context)}"
                ),
            },
        ]

    def _build_agent_reply_messages(self, context: dict[str, Any]) -> list[dict[str, str]]:
        example = {"reply": "当前在放学后的对话场景，雪乃正在试探主角是否愿意一起回家。"}
        return [
            {
                "role": "system",
                "content": (
                    "你是 N.E.K.O 的 galgame Game LLM 辅助系统，不扮演角色，不使用复杂人格。"
                    "你的目标是帮助猫娘理解游戏状态。"
                    "回答应简洁、直接、基于给定 context，不暴露内部私有记忆结构。"
                    "必须只返回一个合法 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "任务：根据给定游戏上下文回答 query_context 或 send_message。\n"
                    "要求：\n"
                    "1. reply 用自然语言给出 best-effort 回答。\n"
                    "2. 若上下文不足，明确说明信息有限，但仍尽量总结当前已知状态。\n"
                    "3. 输出必须匹配这个 JSON 结构：\n"
                    f"{_json_dump(example)}\n\n"
                    "context:\n"
                    f"{_json_dump(context)}"
                ),
            },
        ]

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
