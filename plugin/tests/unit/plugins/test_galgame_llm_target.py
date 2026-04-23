from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

import pytest

from plugin.plugins.galgame_llm_target import GalgameLLMTargetPlugin
from plugin.sdk.plugin import Err, Ok


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


class _Ctx:
    plugin_id = "galgame_llm_target"
    metadata = {}
    bus = None

    def __init__(self, plugin_dir: Path) -> None:
        self.logger = _Logger()
        self.config_path = plugin_dir / "plugin.toml"
        self._effective_config = {
            "plugin": {"store": {"enabled": False}, "database": {"enabled": False}},
            "plugin_state": {"backend": "memory"},
        }

    async def get_own_config(self, timeout: float = 5.0):
        return {"config": self._effective_config}

    async def get_own_base_config(self, timeout: float = 5.0):
        return {"config": self._effective_config}

    async def get_own_profiles_state(self, timeout: float = 5.0):
        return {"profiles": [], "active": None}

    async def get_own_profile_config(self, profile_name: str, timeout: float = 5.0):
        return {"profile_name": profile_name, "config": self._effective_config}

    async def get_own_effective_config(
        self,
        profile_name: str | None = None,
        timeout: float = 5.0,
    ):
        return {"config": self._effective_config}

    async def update_own_config(self, updates, timeout: float = 10.0):
        merged = dict(self._effective_config)
        merged.update(dict(updates or {}))
        self._effective_config = merged
        return {"config": self._effective_config}

    def update_status(self, status):
        return None


def _make_plugin_dir() -> Path:
    plugin_dir = Path(tempfile.mkdtemp())
    (plugin_dir / "plugin.toml").write_text("", encoding="utf-8")
    return plugin_dir


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_target_plugin_explain_line_parses_fenced_json() -> None:
    plugin = GalgameLLMTargetPlugin(_Ctx(_make_plugin_dir()))
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

    plugin._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await plugin.galgame_llm_target_invoke(
        operation="explain_line",
        context={
            "line_id": "line-1",
            "speaker": "雪乃",
            "text": "今天一起回家吗？",
            "evidence": [],
        },
    )

    assert isinstance(result, Ok)
    assert captured == ["explain_line"]
    assert result.value["explanation"] == "这句台词是在试探对方的态度。"
    assert result.value["evidence"][0]["type"] == "current_line"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_target_plugin_suggest_choice_filters_invalid_items_and_renumbers() -> None:
    plugin = GalgameLLMTargetPlugin(_Ctx(_make_plugin_dir()))

    async def _fake_call_model(*, operation: str, messages):
        return """{
          "choices": [
            {"choice_id": "ghost", "text": "不存在", "rank": 1, "reason": "无效"},
            {"choice_id": "choice-2", "text": "右边", "rank": 2, "reason": "更符合当前目标"},
            {"choice_id": "choice-2", "text": "右边", "rank": 3, "reason": "重复"},
            {"choice_id": "choice-1", "text": "左边", "rank": 4, "reason": "备选"}
          ]
        }"""

    plugin._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await plugin.galgame_llm_target_invoke(
        operation="suggest_choice",
        context={
            "visible_choices": [
                {"choice_id": "choice-1", "text": "左边", "index": 0, "enabled": True},
                {"choice_id": "choice-2", "text": "右边", "index": 1, "enabled": True},
            ]
        },
    )

    assert isinstance(result, Ok)
    assert result.value["choices"] == [
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
async def test_target_plugin_agent_reply_retries_once_when_first_reply_is_not_json() -> None:
    plugin = GalgameLLMTargetPlugin(_Ctx(_make_plugin_dir()))
    calls = {"count": 0}

    async def _fake_call_model(*, operation: str, messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return "这不是 JSON"
        return '{"reply": "当前在放学后的对话场景。"}'

    plugin._call_model = _fake_call_model  # type: ignore[method-assign]
    result = await plugin.galgame_llm_target_invoke(
        operation="agent_reply",
        context={"prompt": "现在在讲什么？", "scene_id": "scene-a"},
    )

    assert isinstance(result, Ok)
    assert calls["count"] == 2
    assert result.value["reply"] == "当前在放学后的对话场景。"


@pytest.mark.asyncio
@pytest.mark.plugin_unit
async def test_target_plugin_rejects_unknown_operation() -> None:
    plugin = GalgameLLMTargetPlugin(_Ctx(_make_plugin_dir()))
    result = await plugin.galgame_llm_target_invoke(operation="unknown", context={})
    assert isinstance(result, Err)
    assert "unsupported operation" in str(result.error)


@pytest.mark.plugin_unit
def test_galgame_bridge_plugin_toml_defaults_to_real_target_entry() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "plugins"
        / "galgame_bridge"
        / "plugin.toml"
    )
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    assert payload["llm"]["target_entry_ref"] == (
        "galgame_llm_target:galgame_llm_target_invoke"
    )
