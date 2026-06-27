from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from .state import TFTItemBias, normalized_key

COMPONENT_ALIASES: Mapping[str, str] = {
    "bf sword": "B.F. Sword",
    "b.f. sword": "B.F. Sword",
    "sword": "B.F. Sword",
    "recurve bow": "Recurve Bow",
    "bow": "Recurve Bow",
    "needlessly large rod": "Needlessly Large Rod",
    "rod": "Needlessly Large Rod",
    "tear of the goddess": "Tear of the Goddess",
    "tear": "Tear of the Goddess",
    "chain vest": "Chain Vest",
    "vest": "Chain Vest",
    "negatron cloak": "Negatron Cloak",
    "cloak": "Negatron Cloak",
    "giant's belt": "Giant's Belt",
    "giants belt": "Giant's Belt",
    "belt": "Giant's Belt",
    "sparring gloves": "Sparring Gloves",
    "glove": "Sparring Gloves",
    "gloves": "Sparring Gloves",
    "spatula": "Spatula",
    "frying pan": "Frying Pan",
    "pan": "Frying Pan",
}

BIAS_COMPONENT_WEIGHTS: Mapping[str, Mapping[str, int]] = {
    "ad_carry": {
        "B.F. Sword": 3,
        "Recurve Bow": 2,
        "Sparring Gloves": 2,
    },
    "attack_speed": {
        "Recurve Bow": 3,
        "B.F. Sword": 1,
        "Negatron Cloak": 1,
    },
    "ap_carry": {
        "Needlessly Large Rod": 3,
        "Tear of the Goddess": 2,
        "Sparring Gloves": 1,
    },
    "mana_caster": {
        "Tear of the Goddess": 3,
        "Needlessly Large Rod": 1,
    },
    "frontline": {
        "Chain Vest": 3,
        "Giant's Belt": 3,
        "Negatron Cloak": 3,
    },
    "flex_utility": {
        "Spatula": 3,
        "Frying Pan": 3,
        "Giant's Belt": 1,
        "Negatron Cloak": 1,
        "Tear of the Goddess": 1,
    },
}


def normalize_component(component: object) -> str:
    key = normalized_key(component)
    return COMPONENT_ALIASES.get(key, " ".join(str(component or "").strip().split()))


def calculate_item_biases(
    components: Iterable[object],
    bias_weights: Mapping[str, Mapping[str, int]] = BIAS_COMPONENT_WEIGHTS,
) -> tuple[TFTItemBias, ...]:
    component_counts = Counter(
        normalized
        for component in components
        if (normalized := normalize_component(component))
    )
    biases: list[TFTItemBias] = []

    for bias, weights in bias_weights.items():
        used_components: list[str] = []
        reasons: list[str] = []
        score = 0
        for component, count in sorted(component_counts.items()):
            weight = weights.get(component, 0)
            if weight <= 0:
                continue
            score += weight * count
            used_components.append(component)
            reasons.append(f"{component} x{count}")
        if score > 0:
            biases.append(
                TFTItemBias(
                    bias=bias,
                    score=score,
                    components=tuple(used_components),
                    reasons=tuple(reasons),
                )
            )

    return tuple(sorted(biases, key=lambda bias: (-bias.score, bias.bias)))


__all__ = [
    "BIAS_COMPONENT_WEIGHTS",
    "COMPONENT_ALIASES",
    "calculate_item_biases",
    "normalize_component",
]

