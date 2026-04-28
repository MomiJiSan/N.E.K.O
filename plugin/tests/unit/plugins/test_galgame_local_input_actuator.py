from __future__ import annotations

import pytest

from plugin.plugins.galgame_plugin import local_input_actuator as local_input


pytestmark = pytest.mark.plugin_unit


def test_local_input_visible_choice_detection_accepts_menu_flag_or_choices() -> None:
    assert local_input._snapshot_has_visible_choices(
        {"latest_snapshot": {"is_menu_open": True, "choices": []}}
    ) is True
    assert local_input._snapshot_has_visible_choices(
        {"latest_snapshot": {"is_menu_open": False, "choices": [{"text": "Left"}]}}
    ) is True
    assert local_input._snapshot_has_visible_choices(
        {"latest_snapshot": {"is_menu_open": False, "choices": []}}
    ) is False


def test_local_input_choose_index_uses_choice_payload_index() -> None:
    assert local_input._choose_index(
        {
            "candidate_index": 0,
            "candidate_choices": [
                {"text": "First visible", "index": 2},
                {"text": "Second visible", "index": 4},
            ],
        }
    ) == 2


def test_local_input_virtual_mouse_skips_forbidden_candidate() -> None:
    target = local_input._resolve_virtual_mouse_dialogue_target(
        {"virtual_mouse_target_id": "unsafe"},
        (100, 200, 1100, 1000),
        candidates=(
            {"target_id": "unsafe", "relative_x": 0.9, "relative_y": 0.9},
            {"target_id": "safe", "relative_x": 0.3, "relative_y": 0.7},
        ),
    )

    assert target["success"] is True
    assert target["target_id"] == "safe"
    assert target["screen_x"] == 400
    assert target["screen_y"] == 760
    assert target["skipped_candidates"][0]["forbidden_zone"] == "bottom_toolbar"

