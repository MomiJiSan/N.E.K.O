from __future__ import annotations

from pathlib import Path

import pytest

from plugin.plugins.galgame_plugin import (
    _replace_toml_section_number_value,
    _replace_toml_section_value,
)
from plugin.plugins.galgame_plugin.install_tasks import install_task_state_path
from plugin.plugins.galgame_plugin.models import json_copy


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
