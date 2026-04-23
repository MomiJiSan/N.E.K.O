from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from plugin.plugins.galgame_plugin.llm_backend import GalgameLLMBackend
from plugin.sdk.plugin import SdkError


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def exception(self, *args, **kwargs):
        return None


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_backend_explain_line_parses_fenced_json() -> None:
    backend = GalgameLLMBackend(_Logger())
    captured: list[str] = []

    async def _fake_call_model(*, operation: str, messages):
        captured.append(operation)
        return """```json
        {
          "explanation": "这句台词是在试探对方的态度。",
          "evidence": [
            {
              "type": "current_line",
              "text": "今天一起回家吗？",
              "line_id": "line-1",
              "speaker": "雪乃",
              "scene_id": "scene-a",
              "route_id": ""
            }
          ]
        }
        ```"""

    backend._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await backend.invoke(
        operation="explain_line",
        context={
            "line_id": "line-1",
            "speaker": "雪乃",
            "text": "今天一起回家吗？",
            "evidence": [],
        },
    )

    assert captured == ["explain_line"]
    assert result["explanation"] == "这句台词是在试探对方的态度。"
    assert result["evidence"][0]["type"] == "current_line"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_backend_suggest_choice_filters_invalid_items_and_renumbers() -> None:
    backend = GalgameLLMBackend(_Logger())

    async def _fake_call_model(*, operation: str, messages):
        return """{
          "choices": [
            {"choice_id": "ghost", "text": "不存在", "rank": 1, "reason": "无效"},
            {"choice_id": "choice-2", "text": "右边", "rank": 2, "reason": "更符合当前目标"},
            {"choice_id": "choice-2", "text": "右边", "rank": 3, "reason": "重复"},
            {"choice_id": "choice-1", "text": "左边", "rank": 4, "reason": "备选"}
          ]
        }"""

    backend._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await backend.invoke(
        operation="suggest_choice",
        context={
            "visible_choices": [
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ]
        },
    )

    assert result["choices"] == [
        {
            "choice_id": "choice-2",
            "text": "右边",
            "rank": 1,
            "reason": "更符合当前目标",
        },
        {
            "choice_id": "choice-1",
            "text": "左边",
            "rank": 2,
            "reason": "备选",
        },
    ]


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_backend_agent_reply_retries_once_when_first_reply_is_not_json() -> None:
    backend = GalgameLLMBackend(_Logger())
    calls = {"count": 0}

    async def _fake_call_model(*, operation: str, messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return "这不是 JSON"
        return '{"reply": "当前在放学后的对话场景。"}'

    backend._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await backend.invoke(
        operation="agent_reply",
        context={"prompt": "现在在讲什么？", "scene_id": "scene-a"},
    )

    assert calls["count"] == 2
    assert result["reply"] == "当前在放学后的对话场景。"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_llm_backend_rejects_unknown_operation() -> None:
    backend = GalgameLLMBackend(_Logger())
    with pytest.raises(SdkError, match="unsupported operation"):
        await backend.invoke(operation="unknown", context={})


@pytest.mark.plugin_unit
def test_galgame_plugin_toml_defaults_to_internal_backend() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "plugins"
        / "galgame_plugin"
        / "plugin.toml"
    )
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    assert payload["llm"]["target_entry_ref"] == ""
