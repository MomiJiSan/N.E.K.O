from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from plugin.plugins.galgame_plugin import (
    _replace_toml_section_number_value,
    _replace_toml_section_value,
    _update_plugin_toml,
)
from plugin.plugins.galgame_plugin.install_tasks import install_task_state_path
from plugin.plugins.galgame_plugin.models import GalgameConfig, json_copy


pytestmark = pytest.mark.plugin_unit


def test_replace_toml_section_value_matches_exact_key_only() -> None:
    text = (
        "[galgame.ocr_reader]\n"
        'backend_selection_extra = "keep"\n'
        'backend_selection = "auto"\n'
    )

    updated = _replace_toml_section_value(
        text,
        section="galgame.ocr_reader",
        key="backend_selection",
        value="rapidocr",
    )

    assert 'backend_selection_extra = "keep"' in updated
    assert 'backend_selection = "rapidocr"' in updated


def test_replace_toml_section_number_value_matches_exact_key_only() -> None:
    text = (
        "[galgame.ocr_reader]\n"
        "poll_interval_seconds_extra = 9\n"
        "poll_interval_seconds = 2\n"
    )

    updated = _replace_toml_section_number_value(
        text,
        section="galgame.ocr_reader",
        key="poll_interval_seconds",
        value=1.0,
    )

    assert "poll_interval_seconds_extra = 9" in updated
    assert "poll_interval_seconds = 1" in updated


def test_update_plugin_toml_writes_atomically(tmp_path: Path) -> None:
    config_path = tmp_path / "plugin.toml"
    config_path.write_text(
        "[galgame]\n"
        'reader_mode = "auto"\n'
        "\n"
        "[ocr_reader]\n"
        "poll_interval_seconds = 2\n",
        encoding="utf-8",
    )

    def _update(document: dict[str, object]) -> None:
        galgame = document["galgame"]
        assert isinstance(galgame, dict)
        galgame["reader_mode"] = "ocr"

    _update_plugin_toml(_update, path=config_path)

    updated = config_path.read_text(encoding="utf-8")
    assert 'reader_mode = "ocr"' in updated
    assert list(tmp_path.glob(".plugin.toml.*.tmp")) == []


def test_install_task_state_path_rejects_path_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    with pytest.raises(ValueError, match="invalid task_id"):
        install_task_state_path("../outside", kind="textractor")

    with pytest.raises(ValueError, match="invalid task_id"):
        install_task_state_path(r"..\outside", kind="textractor")


def test_json_copy_fast_path_preserves_copy_semantics() -> None:
    shallow = {"status": "active", "count": 1, "enabled": True}
    shallow_copy = json_copy(shallow)

    assert shallow_copy == shallow
    assert shallow_copy is not shallow

    nested = {"items": [{"text": "hello"}]}
    nested_copy = json_copy(nested)
    nested_copy["items"][0]["text"] = "changed"

    assert nested["items"][0]["text"] == "hello"


def test_galgame_config_groups_fields_and_keeps_flat_compatibility(tmp_path: Path) -> None:
    cfg = GalgameConfig(
        bridge_root=tmp_path / "bridge",
        llm_target_entry_ref="entry-1",
        ocr_reader_enabled=True,
        ocr_reader_backend_selection="tesseract",
        ocr_reader_screen_templates=[{"id": "demo", "stage": "title_stage"}],
        ocr_reader_screen_awareness_model_enabled=True,
        ocr_reader_screen_awareness_model_path="screen-model.json",
        memory_reader_hook_codes=["/HSN-4@1234"],
    )

    assert len(fields(GalgameConfig)) == 7
    assert cfg.bridge.bridge_root == tmp_path / "bridge"
    assert cfg.bridge_root == tmp_path / "bridge"
    assert cfg.llm.llm_target_entry_ref == "entry-1"
    assert cfg.ocr_reader.ocr_reader_enabled is True
    assert cfg.ocr_reader_enabled is True
    assert cfg.ocr_reader.ocr_reader_screen_templates == [{"id": "demo", "stage": "title_stage"}]
    assert cfg.ocr_reader.ocr_reader_screen_awareness_model_enabled is True
    assert cfg.ocr_reader_screen_awareness_model_path == "screen-model.json"
    assert cfg.memory_reader.memory_reader_hook_codes == ["/HSN-4@1234"]

    cfg.ocr_reader_backend_selection = "rapidocr"

    assert cfg.ocr_reader.ocr_reader_backend_selection == "rapidocr"
