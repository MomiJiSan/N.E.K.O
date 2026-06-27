from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from .items import calculate_item_biases, normalize_component
from .state import TFTState, TFTUnit, TFTUpgradeOpportunity, normalized_key, normalized_name
from .traits import TraitDefinitionsInput, calculate_active_traits, calculate_trait_gaps

LOCATION_ORDER = ("board", "bench", "shop")


def parse_tft_state(
    payload: Mapping[str, Any],
    trait_definitions: TraitDefinitionsInput | None = None,
    *,
    analyze: bool = True,
) -> TFTState:
    state = TFTState(
        stage=_optional_str(payload.get("stage")),
        level=_optional_int(payload.get("level")),
        gold=_optional_int(payload.get("gold")),
        board_units=parse_units(payload.get("board_units") or payload.get("board") or (), "board"),
        bench_units=parse_units(payload.get("bench_units") or payload.get("bench") or (), "bench"),
        shop_units=parse_units(payload.get("shop_units") or payload.get("shop") or (), "shop"),
        item_components=parse_item_components(
            payload.get("item_components")
            or payload.get("components")
            or payload.get("items")
            or ()
        ),
    )
    if not analyze:
        return state
    return analyze_tft_state(state, trait_definitions)


def analyze_tft_state(
    state: TFTState,
    trait_definitions: TraitDefinitionsInput | None = None,
) -> TFTState:
    candidate_units = state.bench_units + state.shop_units
    return replace(
        state,
        active_traits=calculate_active_traits(state.board_units, trait_definitions),
        trait_gaps=calculate_trait_gaps(state.board_units, trait_definitions, candidate_units),
        item_biases=calculate_item_biases(state.item_components),
        upgrade_opportunities=calculate_upgrade_opportunities(
            state.board_units,
            state.bench_units,
            state.shop_units,
        ),
    )


def parse_units(values: Iterable[Any], location: str) -> tuple[TFTUnit, ...]:
    return tuple(_parse_unit(value, location) for value in values)


def parse_item_components(values: Iterable[Any]) -> tuple[str, ...]:
    components: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            raw_value = value.get("component") or value.get("name") or value.get("item")
        else:
            raw_value = value
        component = normalize_component(raw_value)
        if component:
            components.append(component)
    return tuple(components)


def calculate_upgrade_opportunities(
    board_units: Iterable[TFTUnit],
    bench_units: Iterable[TFTUnit],
    shop_units: Iterable[TFTUnit],
) -> tuple[TFTUpgradeOpportunity, ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for unit in [*board_units, *bench_units, *shop_units]:
        if not unit.name:
            continue
        key = normalized_key(unit.name)
        entry = grouped.setdefault(
            key,
            {
                "name": unit.name,
                "owned_copies": 0,
                "highest_stars": 1,
                "source_counts": defaultdict(int),
            },
        )
        entry["owned_copies"] += unit.copy_value
        entry["highest_stars"] = max(entry["highest_stars"], unit.stars)
        entry["source_counts"][unit.location] += unit.copy_value

    opportunities: list[TFTUpgradeOpportunity] = []
    for entry in grouped.values():
        owned_copies = int(entry["owned_copies"])
        highest_stars = int(entry["highest_stars"])
        target_copies, target_stars = _next_upgrade_target(owned_copies, highest_stars)
        needed_copies = max(0, target_copies - owned_copies)
        if not _is_upgrade_opportunity(target_stars, needed_copies, owned_copies):
            continue
        opportunities.append(
            TFTUpgradeOpportunity(
                champion_name=entry["name"],
                owned_copies=owned_copies,
                needed_copies=needed_copies,
                target_stars=target_stars,
                kind=_upgrade_kind(target_stars, needed_copies),
                source_counts=tuple(
                    (location, int(entry["source_counts"].get(location, 0)))
                    for location in LOCATION_ORDER
                    if entry["source_counts"].get(location, 0)
                ),
            )
        )

    return tuple(
        sorted(
            opportunities,
            key=lambda opportunity: (
                opportunity.needed_copies,
                opportunity.target_stars,
                opportunity.champion_name,
            ),
        )
    )


def _parse_unit(value: Any, location: str) -> TFTUnit:
    if isinstance(value, TFTUnit):
        return replace(value, location=location if value.location == "unknown" else value.location)
    if not isinstance(value, Mapping):
        return TFTUnit(name=str(value), location=location)
    return TFTUnit(
        name=str(
            value.get("name")
            or value.get("champion")
            or value.get("champion_name")
            or value.get("unit")
            or value.get("unit_id")
            or value.get("id")
            or value.get("display_name")
            or ""
        ),
        traits=_parse_str_sequence(value.get("traits") or value.get("trait_names") or ()),
        location=str(value.get("location") or location),
        stars=_optional_int(value.get("stars") or value.get("star") or value.get("tier")) or 1,
        cost=_optional_int(value.get("cost")),
        items=_parse_str_sequence(value.get("items") or ()),
        count=_optional_int(value.get("count") or value.get("copies")) or 1,
    )


def _next_upgrade_target(owned_copies: int, highest_stars: int) -> tuple[int, int]:
    if highest_stars >= 2 or owned_copies >= 6:
        return 9, 3
    return 3, 2


def _is_upgrade_opportunity(target_stars: int, needed_copies: int, owned_copies: int) -> bool:
    if needed_copies <= 0:
        return True
    if target_stars == 2:
        return owned_copies >= 2
    return needed_copies <= 2


def _upgrade_kind(target_stars: int, needed_copies: int) -> str:
    if needed_copies <= 0:
        return "upgrade"
    if target_stars == 2 and needed_copies == 1:
        return "pair"
    return "near_upgrade"


def _parse_str_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.replace("/", ",").split(",")
        return tuple(normalized_name(part) for part in parts if normalized_name(part))
    if isinstance(value, Sequence):
        return tuple(normalized_name(part) for part in value if normalized_name(part))
    return ()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = normalized_name(value)
    return text or None


__all__ = [
    "analyze_tft_state",
    "calculate_upgrade_opportunities",
    "parse_item_components",
    "parse_tft_state",
    "parse_units",
]
