from plugin.plugins.game_companion.profiles.tft.items import calculate_item_biases
from plugin.plugins.game_companion.profiles.tft.state import TFTUnit
from plugin.plugins.game_companion.profiles.tft.state_parser import (
    calculate_upgrade_opportunities,
    parse_tft_state,
)
from plugin.plugins.game_companion.profiles.tft.traits import (
    calculate_active_traits,
    calculate_trait_gaps,
)


TRAIT_DEFINITIONS = {
    "Bruiser": {"tiers": (2, 4, 6), "units": ("Vi", "Sejuani", "Sett", "Cho")},
    "Arcanist": {"tiers": (2, 4), "units": ("Lux", "Ahri", "Zoe")},
    "Sniper": {"tiers": (2, 4), "units": ("Ashe", "Caitlyn")},
}


def test_calculate_active_traits_counts_unique_board_champions() -> None:
    board = (
        TFTUnit(name="Vi", traits=("Bruiser",), location="board"),
        TFTUnit(name="Vi", traits=("Bruiser",), location="board"),
        TFTUnit(name="Sejuani", traits=("Bruiser",), location="board"),
        TFTUnit(name="Lux", traits=("Arcanist",), location="board"),
    )

    active_traits = calculate_active_traits(board, TRAIT_DEFINITIONS)

    assert len(active_traits) == 1
    assert active_traits[0].trait == "Bruiser"
    assert active_traits[0].count == 2
    assert active_traits[0].active_tier == 2
    assert active_traits[0].next_tier == 4
    assert active_traits[0].unit_names == ("Sejuani", "Vi")


def test_calculate_trait_gaps_reports_one_and_two_unit_breakpoints() -> None:
    board = (
        TFTUnit(name="Vi", traits=("Bruiser",), location="board"),
        TFTUnit(name="Lux", traits=("Arcanist",), location="board"),
    )
    candidates = (
        TFTUnit(name="Sejuani", traits=("Bruiser",), location="bench"),
        TFTUnit(name="Ahri", traits=("Arcanist",), location="shop"),
        TFTUnit(name="Caitlyn", traits=("Sniper",), location="shop"),
    )

    gaps = calculate_trait_gaps(board, TRAIT_DEFINITIONS, candidates)
    by_trait = {gap.trait: gap for gap in gaps}

    assert by_trait["Bruiser"].units_needed == 1
    assert by_trait["Bruiser"].candidate_unit_names == ("Sejuani",)
    assert by_trait["Bruiser"].source_locations == ("bench",)
    assert by_trait["Arcanist"].units_needed == 1
    assert by_trait["Arcanist"].source_locations == ("shop",)
    assert by_trait["Sniper"].current_count == 0
    assert by_trait["Sniper"].target_count == 2
    assert by_trait["Sniper"].units_needed == 2


def test_calculate_item_biases_prefers_ad_or_frontline_from_components() -> None:
    offensive = calculate_item_biases(("B.F. Sword", "Bow", "Sparring Gloves"))
    defensive = calculate_item_biases(("Chain Vest", "Giant's Belt", "Negatron Cloak"))

    assert offensive[0].bias == "ad_carry"
    assert offensive[0].score == 7
    assert "Recurve Bow" in offensive[0].components
    assert defensive[0].bias == "frontline"
    assert defensive[0].score == 9


def test_calculate_upgrade_opportunities_counts_board_bench_and_shop() -> None:
    board = (
        TFTUnit(name="Lux", traits=("Arcanist",), location="board"),
        TFTUnit(name="Ashe", traits=("Sniper",), location="board"),
    )
    bench = (
        TFTUnit(name="Lux", traits=("Arcanist",), location="bench"),
        TFTUnit(name="Ashe", traits=("Sniper",), location="bench"),
    )
    shop = (TFTUnit(name="Lux", traits=("Arcanist",), location="shop"),)

    opportunities = calculate_upgrade_opportunities(board, bench, shop)
    by_name = {opportunity.champion_name: opportunity for opportunity in opportunities}

    assert by_name["Lux"].kind == "upgrade"
    assert by_name["Lux"].owned_copies == 3
    assert by_name["Lux"].needed_copies == 0
    assert by_name["Lux"].target_stars == 2
    assert by_name["Lux"].source_counts == (("board", 1), ("bench", 1), ("shop", 1))
    assert by_name["Ashe"].kind == "pair"
    assert by_name["Ashe"].owned_copies == 2
    assert by_name["Ashe"].needed_copies == 1


def test_parse_tft_state_builds_analyzed_state_from_mock_payload() -> None:
    state = parse_tft_state(
        {
            "stage": "3-2",
            "level": "6",
            "gold": "41",
            "board_units": [
                {"name": "Vi", "traits": ["Bruiser"]},
                {"name": "Sejuani", "traits": ["Bruiser"]},
                {"name": "Lux", "traits": ["Arcanist"]},
                {"name": "Ashe", "traits": ["Sniper"]},
            ],
            "bench_units": [
                {"name": "Ahri", "traits": ["Arcanist"]},
                {"name": "Sett", "traits": ["Bruiser"]},
                {"name": "Lux", "traits": ["Arcanist"]},
                {"name": "Ashe", "traits": ["Sniper"]},
            ],
            "shop_units": [
                {"name": "Zoe", "traits": ["Arcanist"]},
                {"name": "Lux", "traits": ["Arcanist"]},
            ],
            "items": [{"name": "Sword"}, {"component": "Bow"}, "Glove"],
        },
        TRAIT_DEFINITIONS,
    )

    active_traits = {trait.trait: trait for trait in state.active_traits}
    gaps = {gap.trait: gap for gap in state.trait_gaps}
    upgrades = {opportunity.champion_name: opportunity for opportunity in state.upgrade_opportunities}

    assert state.stage == "3-2"
    assert state.level == 6
    assert state.gold == 41
    assert active_traits["Bruiser"].active_tier == 2
    assert gaps["Arcanist"].units_needed == 1
    assert gaps["Bruiser"].units_needed == 2
    assert state.item_biases[0].bias == "ad_carry"
    assert upgrades["Lux"].kind == "upgrade"
    assert upgrades["Ashe"].kind == "pair"
