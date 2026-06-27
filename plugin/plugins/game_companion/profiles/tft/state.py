from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def normalized_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalized_name(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


@dataclass(frozen=True, slots=True)
class TFTUnit:
    name: str
    traits: tuple[str, ...] = ()
    location: str = "unknown"
    stars: int = 1
    cost: int | None = None
    items: tuple[str, ...] = ()
    count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalized_name(self.name))
        object.__setattr__(self, "traits", tuple(normalized_name(trait) for trait in self.traits if normalized_name(trait)))
        object.__setattr__(self, "location", normalized_key(self.location) or "unknown")
        object.__setattr__(self, "stars", max(1, min(3, int(self.stars or 1))))
        object.__setattr__(self, "items", tuple(normalized_name(item) for item in self.items if normalized_name(item)))
        object.__setattr__(self, "count", max(1, int(self.count or 1)))

    @property
    def copy_value(self) -> int:
        return self.count * (3 ** (self.stars - 1))


@dataclass(frozen=True, slots=True)
class TFTTraitDefinition:
    name: str
    tiers: tuple[int, ...]
    units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tiers = tuple(sorted({int(tier) for tier in self.tiers if int(tier) > 0}))
        object.__setattr__(self, "name", normalized_name(self.name))
        object.__setattr__(self, "tiers", tiers)
        object.__setattr__(self, "units", tuple(normalized_name(unit) for unit in self.units if normalized_name(unit)))


@dataclass(frozen=True, slots=True)
class TFTTraitStatus:
    trait: str
    count: int
    active_tier: int
    next_tier: int | None = None
    unit_names: tuple[str, ...] = ()

    @property
    def is_active(self) -> bool:
        return self.active_tier > 0


@dataclass(frozen=True, slots=True)
class TFTTraitGap:
    trait: str
    current_count: int
    target_count: int
    units_needed: int
    active_tier: int | None = None
    candidate_unit_names: tuple[str, ...] = ()
    source_locations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TFTItemBias:
    bias: str
    score: int
    components: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TFTUpgradeOpportunity:
    champion_name: str
    owned_copies: int
    needed_copies: int
    target_stars: int
    kind: str
    source_counts: tuple[tuple[str, int], ...] = ()

    @property
    def can_upgrade_now(self) -> bool:
        return self.needed_copies <= 0


@dataclass(frozen=True, slots=True)
class TFTState:
    stage: str | None = None
    level: int | None = None
    gold: int | None = None
    board_units: tuple[TFTUnit, ...] = field(default_factory=tuple)
    bench_units: tuple[TFTUnit, ...] = field(default_factory=tuple)
    shop_units: tuple[TFTUnit, ...] = field(default_factory=tuple)
    item_components: tuple[str, ...] = field(default_factory=tuple)
    active_traits: tuple[TFTTraitStatus, ...] = field(default_factory=tuple)
    trait_gaps: tuple[TFTTraitGap, ...] = field(default_factory=tuple)
    item_biases: tuple[TFTItemBias, ...] = field(default_factory=tuple)
    upgrade_opportunities: tuple[TFTUpgradeOpportunity, ...] = field(default_factory=tuple)

    @property
    def items(self) -> tuple[str, ...]:
        return self.item_components

    @property
    def traits(self) -> tuple[TFTTraitStatus, ...]:
        return self.active_traits


__all__ = [
    "TFTItemBias",
    "TFTState",
    "TFTTraitDefinition",
    "TFTTraitGap",
    "TFTTraitStatus",
    "TFTUnit",
    "TFTUpgradeOpportunity",
    "normalized_key",
    "normalized_name",
]
