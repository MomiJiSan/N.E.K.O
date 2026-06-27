import json
from typing import Any

from plugin.plugins.game_companion.profiles.tft.insights import (
    ALLOWED_INSIGHT_TYPES,
    FORBIDDEN_DIRECTIVE_PHRASES,
    generate_insights,
    generate_tft_insights,
)


def _by_type(insights: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for insight in insights:
        grouped.setdefault(insight["type"], []).append(insight)
    return grouped


def test_generate_tft_insights_returns_only_allowed_insight_schema() -> None:
    state = {
        "level": 7,
        "gold": 54,
        "board_units": [{"name": "Ahri"}, {"name": "Neeko"}],
        "bench_units": [{"name": "Ahri"}],
        "shop_units": [{"name": "Syndra"}],
        "items": [
            {"name": "Needlessly Large Rod", "holder": "Ahri"},
            {"name": "Tear of the Goddess"},
        ],
        "traits": [
            {
                "name": "Star Guardian",
                "active": 2,
                "thresholds": [2, 3, 5],
                "units": ["Ahri", "Neeko"],
            },
            {
                "name": "Sorcerer",
                "active": 2,
                "thresholds": [2, 4, 6],
                "units": ["Ahri"],
            },
        ],
    }

    insights = generate_tft_insights(state)

    assert insights
    for insight in insights:
        assert set(insight) == {"type", "severity", "title", "detail", "related_units"}
        assert insight["type"] in ALLOWED_INSIGHT_TYPES
        assert isinstance(insight["title"], str)
        assert isinstance(insight["detail"], str)
        assert isinstance(insight["related_units"], list)

    rendered = json.dumps(insights, ensure_ascii=False)
    for phrase in FORBIDDEN_DIRECTIVE_PHRASES:
        assert phrase not in rendered


def test_generate_tft_insights_detects_available_signal_types() -> None:
    state = {
        "level": 7,
        "gold": 54,
        "board_units": [{"name": "Ahri"}, {"name": "Neeko"}, {"name": "Syndra"}],
        "bench_units": [{"name": "Ahri"}],
        "shop_units": [],
        "items": [
            {"name": "Needlessly Large Rod", "holder": "Ahri"},
            {"name": "Tear of the Goddess"},
        ],
        "traits": [
            {
                "name": "Star Guardian",
                "active": 2,
                "thresholds": [2, 3, 5],
                "units": ["Ahri", "Neeko"],
            },
            {
                "name": "Sorcerer",
                "active": 2,
                "thresholds": [2, 4, 6],
                "units": ["Ahri", "Syndra"],
            },
        ],
    }

    grouped = _by_type(generate_tft_insights(state))

    assert grouped["trait_gap"][0]["related_units"] == ["Ahri", "Neeko"]
    assert grouped["pairs"][0]["related_units"] == ["Ahri"]
    assert grouped["item_direction"][0]["title"] == "Magic damage item signal"
    assert grouped["composition_direction"][0]["title"] == "Active trait cluster"
    assert grouped["level_option"][0]["title"] == "High-economy level option"


def test_generate_tft_insights_accepts_analysis_dict_without_state_coupling() -> None:
    analysis = {
        "trait_gaps": [{"name": "Duelist", "needed": 1, "units": ["Fiora"]}],
        "pairs": [{"unit": "Fiora", "copies": 2}],
        "item_directions": [{"direction": "Attack damage", "related_units": ["Fiora"]}],
        "composition_direction": {
            "direction": "Duelist",
            "related_units": ["Fiora", "Yasuo"],
        },
        "level_options": [
            {"option": "Slow economy", "detail": "Economy can remain flexible this round."}
        ],
    }

    grouped = _by_type(generate_tft_insights({}, analysis))

    assert set(grouped) == {
        "trait_gap",
        "pairs",
        "item_direction",
        "composition_direction",
        "level_option",
    }
    assert grouped["trait_gap"][0]["title"] == "Duelist breakpoint nearby"
    assert grouped["composition_direction"][0]["related_units"] == ["Fiora", "Yasuo"]


def test_generate_insights_alias_matches_tft_generator() -> None:
    state = {"level": 8, "gold": 51}

    assert generate_insights(state) == generate_tft_insights(state)


def test_forbidden_directive_analysis_text_is_filtered() -> None:
    analysis = {
        "level_options": [
            {
                "option": "Aggressive",
                "detail": (
                    f"{FORBIDDEN_DIRECTIVE_PHRASES[4]} and "
                    f"{FORBIDDEN_DIRECTIVE_PHRASES[2]} are directive text."
                ),
            },
            {
                "option": "Flexible tempo",
                "detail": "Economy and level timing remain open.",
            },
        ]
    }

    insights = generate_tft_insights({}, analysis)

    assert len(insights) == 1
    assert insights[0]["title"] == "Flexible tempo option"


def test_english_and_action_directives_are_filtered() -> None:
    analysis = {
        "level_options": [
            {"option": "Force comp", "detail": "Must buy every visible copy."},
            {"option": "Click line", "detail": "Click the shop and roll down now."},
            {"option": "Tempo window", "detail": "Economy and board-space options remain open."},
        ],
        "composition_directions": [
            {"direction": "All in reroll", "detail": "All in and force this comp."},
            {"direction": "Flexible frontline", "detail": "Visible traits point toward frontline coverage."},
        ],
    }

    insights = generate_tft_insights({}, analysis)

    rendered = json.dumps(insights, ensure_ascii=False).casefold()
    assert "must buy" not in rendered
    assert "click " not in rendered
    assert "all in" not in rendered
    assert {insight["title"] for insight in insights} == {
        "Flexible frontline composition signal",
        "Tempo window option",
    }
