from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .state import (
    TFTTraitDefinition,
    TFTTraitGap,
    TFTTraitStatus,
    TFTUnit,
    normalized_key,
    normalized_name,
)

TraitDefinitionsInput = Mapping[str, Any] | Iterable[TFTTraitDefinition]


def build_trait_definitions(definitions: TraitDefinitionsInput | None) -> dict[str, TFTTraitDefinition]:
    if definitions is None:
        return {}
    if isinstance(definitions, Mapping):
        return {
            normalized_key(definition.name): definition
            for definition in (_coerce_trait_definition(name, value) for name, value in definitions.items())
            if definition.name and definition.tiers
        }
    return {
        normalized_key(definition.name): definition
        for definition in definitions
        if definition.name and definition.tiers
    }


def calculate_active_traits(
    board_units: Iterable[TFTUnit],
    trait_definitions: TraitDefinitionsInput | None = None,
) -> tuple[TFTTraitStatus, ...]:
    definitions = build_trait_definitions(trait_definitions)
    counts, units_by_trait = _board_trait_counts(board_units)
    active_traits: list[TFTTraitStatus] = []

    for trait_key, count in counts.items():
        definition = definitions.get(trait_key)
        if definition is None:
            continue
        active_tier = _active_tier(count, definition.tiers)
        if active_tier is None:
            continue
        active_traits.append(
            TFTTraitStatus(
                trait=definition.name,
                count=count,
                active_tier=active_tier,
                next_tier=_next_tier(count, definition.tiers),
                unit_names=tuple(sorted(units_by_trait.get(trait_key, ()))),
            )
        )

    return tuple(sorted(active_traits, key=lambda trait: (trait.trait, trait.active_tier)))


def calculate_trait_gaps(
    board_units: Iterable[TFTUnit],
    trait_definitions: TraitDefinitionsInput | None = None,
    candidate_units: Iterable[TFTUnit] = (),
    max_units_needed: int = 2,
) -> tuple[TFTTraitGap, ...]:
    definitions = build_trait_definitions(trait_definitions)
    counts, units_by_trait = _board_trait_counts(board_units)
    candidates_by_trait = _candidate_units_by_trait(candidate_units, units_by_trait)
    gaps: list[TFTTraitGap] = []

    for trait_key, definition in definitions.items():
        current_count = counts.get(trait_key, 0)
        target_count = _next_tier(current_count, definition.tiers)
        if target_count is None:
            continue
        units_needed = target_count - current_count
        if units_needed < 1 or units_needed > max_units_needed:
            continue

        candidate_names, source_locations = candidates_by_trait.get(trait_key, ((), ()))
        if not candidate_names:
            owned_names = units_by_trait.get(trait_key, frozenset())
            owned_name_keys = {normalized_key(unit) for unit in owned_names}
            candidate_names = tuple(
                sorted(unit for unit in definition.units if normalized_key(unit) not in owned_name_keys)
            )

        gaps.append(
            TFTTraitGap(
                trait=definition.name,
                current_count=current_count,
                target_count=target_count,
                units_needed=units_needed,
                active_tier=_active_tier(current_count, definition.tiers),
                candidate_unit_names=candidate_names,
                source_locations=source_locations,
            )
        )

    return tuple(sorted(gaps, key=lambda gap: (gap.units_needed, gap.trait)))


def _coerce_trait_definition(name: str, value: Any) -> TFTTraitDefinition:
    if isinstance(value, TFTTraitDefinition):
        return value
    if isinstance(value, Mapping):
        tiers = value.get("tiers") or value.get("breakpoints") or value.get("thresholds") or ()
        units = value.get("units") or value.get("champions") or ()
        return TFTTraitDefinition(name=str(value.get("name") or name), tiers=_int_tuple(tiers), units=_str_tuple(units))
    if isinstance(value, Sequence) and not isinstance(value, str):
        return TFTTraitDefinition(name=name, tiers=_int_tuple(value))
    return TFTTraitDefinition(name=name, tiers=())


def _board_trait_counts(board_units: Iterable[TFTUnit]) -> tuple[dict[str, int], dict[str, frozenset[str]]]:
    units_by_trait: dict[str, dict[str, str]] = {}
    anonymous_index = 0

    for unit in board_units:
        unit_name = unit.name or f"anonymous:{anonymous_index}"
        unit_key = normalized_key(unit_name) or f"anonymous:{anonymous_index}"
        anonymous_index += 1
        for trait in unit.traits:
            trait_key = normalized_key(trait)
            if not trait_key:
                continue
            units_by_trait.setdefault(trait_key, {})[unit_key] = unit_name

    return (
        {trait_key: len(unit_names) for trait_key, unit_names in units_by_trait.items()},
        {trait_key: frozenset(unit_names.values()) for trait_key, unit_names in units_by_trait.items()},
    )


def _candidate_units_by_trait(
    candidate_units: Iterable[TFTUnit],
    board_units_by_trait: Mapping[str, frozenset[str]],
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    names: dict[str, set[str]] = {}
    locations: dict[str, set[str]] = {}
    for unit in candidate_units:
        if not unit.name:
            continue
        for trait in unit.traits:
            trait_key = normalized_key(trait)
            board_unit_keys = {normalized_key(name) for name in board_units_by_trait.get(trait_key, frozenset())}
            if normalized_key(unit.name) in board_unit_keys:
                continue
            names.setdefault(trait_key, set()).add(unit.name)
            locations.setdefault(trait_key, set()).add(unit.location)
    return {
        trait_key: (tuple(sorted(unit_names)), tuple(sorted(locations.get(trait_key, ()))))
        for trait_key, unit_names in names.items()
    }


def _active_tier(count: int, tiers: Iterable[int]) -> int | None:
    active = [tier for tier in tiers if tier <= count]
    return max(active) if active else None


def _next_tier(count: int, tiers: Iterable[int]) -> int | None:
    next_tiers = [tier for tier in tiers if tier > count]
    return min(next_tiers) if next_tiers else None


def _int_tuple(values: Any) -> tuple[int, ...]:
    if isinstance(values, int):
        return (values,)
    if isinstance(values, str):
        return tuple(int(part) for part in values.replace("/", ",").split(",") if part.strip().isdigit())
    return tuple(int(value) for value in values)


def _str_tuple(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        return tuple(normalized_name(part) for part in values.split(",") if normalized_name(part))
    return tuple(normalized_name(value) for value in values if normalized_name(value))


__all__ = [
    "TraitDefinitionsInput",
    "build_trait_definitions",
    "calculate_active_traits",
    "calculate_trait_gaps",
]
