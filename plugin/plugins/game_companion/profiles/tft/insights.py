from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

ALLOWED_INSIGHT_TYPES = frozenset(
    {
        "trait_gap",
        "item_direction",
        "pairs",
        "composition_direction",
        "level_option",
    }
)
_FORBIDDEN_DIRECTIVE_CODEPOINTS = (
    (0x4E70, 0x8FD9, 0x4E2A),
    (0x5356, 0x8FD9, 0x4E2A),
    (0x5FC5, 0x62FF),
    (0x6700, 0x4F18, 0x7AD9, 0x4F4D),
    (0x7ACB, 0x523B, 0x8F6C, 0x9635),
    (0x6700, 0x4F18, 0x8F6C, 0x9635),
    (0x7ACB, 0x523B, 0x4E70),
    (0x7ACB, 0x523B, 0x5356),
    (0x5FC5, 0x987B, 0x4E70),
    (0x5FC5, 0x987B, 0x5356),
    (0x5F3A, 0x5236),
    (0x81EA, 0x52A8, 0x70B9, 0x51FB),
    (0x5237, 0x65B0, 0x5546, 0x5E97),
)
FORBIDDEN_DIRECTIVE_PHRASES = tuple(
    "".join(chr(codepoint) for codepoint in phrase)
    for phrase in _FORBIDDEN_DIRECTIVE_CODEPOINTS
)
FORBIDDEN_DIRECTIVE_PHRASES += (
    "buy this",
    "sell this",
    "must buy",
    "must sell",
    "always buy",
    "always sell",
    "click ",
    "auto click",
    "roll down",
    "reroll now",
    "all in",
    "force this",
    "force comp",
    "slam this",
    "equip this",
    "reposition to",
    "move unit",
    "position here",
)
FORBIDDEN_DIRECTIVE_TERMS = FORBIDDEN_DIRECTIVE_PHRASES
MAX_INSIGHTS = 8


def generate_tft_insights(
    state: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return read-only TFT insights from already-derived state and analysis data."""
    safe_state = _mapping_or_empty(state)
    safe_analysis = _mapping_or_empty(analysis)
    insights: list[dict[str, Any]] = []

    _extend(insights, _trait_gap_insights(safe_state, safe_analysis))
    _extend(insights, _pair_insights(safe_state, safe_analysis))
    _extend(insights, _item_direction_insights(safe_state, safe_analysis))
    _extend(insights, _composition_direction_insights(safe_state, safe_analysis))
    _extend(insights, _level_option_insights(safe_state, safe_analysis))
    return insights[:MAX_INSIGHTS]


def generate_insights(
    state: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if analysis is None and isinstance(state, Mapping) and _looks_like_analysis(state):
        analysis = state
        state = _mapping_or_empty(analysis.get("state"))
    return generate_tft_insights(state, analysis)


def _trait_gap_insights(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    explicit_gaps = _as_list(analysis.get("trait_gaps"))
    if explicit_gaps:
        for gap in explicit_gaps:
            mapping = _mapping_or_empty(gap)
            trait_name = _name(gap) or _name(mapping.get("trait"))
            if not trait_name:
                continue
            needed = _int_or_none(mapping.get("needed", mapping.get("gap"))) or 1
            yield _insight(
                "trait_gap",
                "info",
                f"{trait_name} breakpoint nearby",
                f"{trait_name} is {needed} unit away from a listed trait breakpoint.",
                _related_units(mapping),
            )
        return

    for trait in _as_list(state.get("traits")):
        mapping = _mapping_or_empty(trait)
        trait_name = _name(trait) or _name(mapping.get("trait"))
        current = _int_or_none(mapping.get("active", mapping.get("current", mapping.get("count"))))
        next_threshold = _next_trait_threshold(mapping, current)
        needed = _int_or_none(mapping.get("needed", mapping.get("gap")))
        if needed is None and current is not None and next_threshold is not None:
            needed = next_threshold - current
        if trait_name and needed == 1:
            yield _insight(
                "trait_gap",
                "info",
                f"{trait_name} breakpoint nearby",
                f"{trait_name} is 1 unit away from its next trait breakpoint.",
                _related_units(mapping),
            )


def _pair_insights(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    explicit_pairs = _as_list(analysis.get("pairs"))
    if explicit_pairs:
        for pair in explicit_pairs:
            mapping = _mapping_or_empty(pair)
            unit_name = _name(pair) or _name(mapping.get("unit"))
            if not unit_name:
                continue
            copies = _int_or_none(mapping.get("copies", mapping.get("count"))) or 2
            yield _insight(
                "pairs",
                "info",
                f"{unit_name} pair visible",
                f"{copies} copies of {unit_name} are visible in the provided analysis.",
                [unit_name],
            )
        return

    counts: Counter[str] = Counter()
    for zone in ("board_units", "bench_units", "shop_units"):
        for unit in _as_list(state.get(zone)):
            unit_name = _unit_name(unit)
            if unit_name:
                counts[unit_name] += 1
    for unit_name, copies in sorted(counts.items()):
        if copies == 2:
            yield _insight(
                "pairs",
                "info",
                f"{unit_name} pair visible",
                f"Two copies of {unit_name} are visible across board, bench, or shop.",
                [unit_name],
            )


def _item_direction_insights(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    explicit_directions = _as_list(analysis.get("item_directions"))
    if not explicit_directions and analysis.get("item_direction"):
        explicit_directions = [analysis.get("item_direction")]
    if explicit_directions:
        for direction in explicit_directions:
            mapping = _mapping_or_empty(direction)
            label = _name(direction) or _text(mapping.get("direction")) or "Item direction"
            detail = _text(mapping.get("detail")) or f"Items point toward {label} lines."
            yield _insight(
                "item_direction",
                "info",
                f"{label} item signal",
                detail,
                _related_units(mapping),
            )
        return

    tags = Counter[str]()
    holders: list[str] = []
    for item in _as_list(state.get("items")):
        item_name = _name(item)
        if not item_name:
            continue
        tags.update(_item_tags(item_name))
        holder = _holder_name(item)
        if holder:
            holders.append(holder)
    if tags:
        direction, _count = tags.most_common(1)[0]
        yield _insight(
            "item_direction",
            "info",
            f"{direction} item signal",
            f"Current components lean toward {direction.lower()} item lines.",
            holders,
        )


def _composition_direction_insights(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    explicit_directions = _as_list(analysis.get("composition_directions"))
    if not explicit_directions and analysis.get("composition_direction"):
        explicit_directions = [analysis.get("composition_direction")]
    if explicit_directions:
        for direction in explicit_directions:
            mapping = _mapping_or_empty(direction)
            label = _name(direction) or _text(mapping.get("direction")) or "Composition"
            detail = _text(mapping.get("detail")) or f"{label} is the clearest board signal."
            yield _insight("composition_direction", "info", f"{label} composition signal", detail, _related_units(mapping))
        return

    active_traits = _as_list(analysis.get("active_traits")) or _as_list(state.get("traits"))
    names = [_name(trait) for trait in active_traits if _name(trait)]
    if names:
        shown = " and ".join(names[:2])
        yield _insight(
            "composition_direction",
            "info",
            "Active trait cluster",
            f"{shown} are the strongest visible trait signals.",
            _unit_names(state.get("board_units"))[:5],
        )


def _level_option_insights(
    state: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    explicit_options = _as_list(analysis.get("level_options"))
    if explicit_options:
        for option in explicit_options:
            mapping = _mapping_or_empty(option)
            label = _name(option) or _text(mapping.get("option")) or "Level timing"
            detail = _text(mapping.get("detail")) or f"{label} is available in the analysis."
            yield _insight(
                "level_option",
                "info",
                f"{label} option",
                detail,
                _related_units(mapping),
            )
        return

    gold = _int_or_none(state.get("gold"))
    level = _int_or_none(state.get("level"))
    if level is None:
        return
    if gold is not None and gold >= 50 and level < 9:
        yield _insight(
            "level_option",
            "info",
            "High-economy level option",
            f"Level {level} with {gold} gold leaves a level timing option open.",
            [],
        )
    elif level < 9:
        yield _insight(
            "level_option",
            "info",
            f"Level {level + 1} board-space option",
            f"Level {level + 1} would add one more unit slot for trait or frontline coverage.",
            _unit_names(state.get("bench_units"))[:5],
        )


def _extend(target: list[dict[str, Any]], candidates: Iterable[dict[str, Any]]) -> None:
    seen = {(item["type"], item["title"], item["detail"]) for item in target}
    for candidate in candidates:
        if (
            candidate["type"] not in ALLOWED_INSIGHT_TYPES
            or _has_forbidden_directive(candidate)
        ):
            continue
        key = (candidate["type"], candidate["title"], candidate["detail"])
        if key in seen:
            continue
        target.append(candidate)
        seen.add(key)
        if len(target) >= MAX_INSIGHTS:
            return


def _insight(
    insight_type: str,
    severity: str,
    title: str,
    detail: str,
    related_units: Iterable[Any],
) -> dict[str, Any]:
    return {
        "type": insight_type,
        "severity": severity,
        "title": str(title).strip(),
        "detail": str(detail).strip(),
        "related_units": _unique_text(related_units),
    }


def _looks_like_analysis(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "active_traits",
            "trait_gaps",
            "item_direction",
            "item_directions",
            "pairs",
            "composition_direction",
            "composition_directions",
            "level_options",
        )
    )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("name", "unit", "unit_id", "champion", "display_name", "trait", "trait_id", "id"):
            text = _text(value.get(key))
            if text:
                return text
    return ""


def _unit_name(value: Any) -> str:
    return _name(value)


def _holder_name(value: Any) -> str:
    mapping = _mapping_or_empty(value)
    for key in ("holder", "unit", "champion"):
        text = _text(mapping.get(key))
        if text:
            return text
    return ""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_trait_threshold(mapping: Mapping[str, Any], current: int | None) -> int | None:
    explicit = _int_or_none(mapping.get("next", mapping.get("next_threshold", mapping.get("threshold"))))
    if explicit is not None:
        return explicit
    if current is None:
        return None
    thresholds = sorted(
        threshold
        for threshold in (_int_or_none(value) for value in _as_list(mapping.get("thresholds")))
        if threshold is not None
    )
    return next((threshold for threshold in thresholds if threshold > current), None)


def _related_units(mapping: Mapping[str, Any]) -> list[str]:
    for key in ("related_units", "units", "champions"):
        units = _unique_text(_as_list(mapping.get(key)))
        if units:
            return units
    unit = _name(mapping.get("unit"))
    return [unit] if unit else []


def _unit_names(units: Any) -> list[str]:
    return _unique_text(_unit_name(unit) for unit in _as_list(units))


def _unique_text(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _name(value) or _text(value)
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _item_tags(item_name: str) -> list[str]:
    normalized = item_name.lower()
    tags: list[str] = []
    if any(token in normalized for token in ("rod", "tear", "staff", "blue", "archangel")):
        tags.append("Magic damage")
    if any(token in normalized for token in ("sword", "bow", "edge", "rageblade")):
        tags.append("Attack damage")
    if any(token in normalized for token in ("belt", "chain", "cloak", "warmog", "bramble")):
        tags.append("Frontline durability")
    if any(token in normalized for token in ("glove", "thief", "guardbreaker")):
        tags.append("Flexible utility")
    return tags or ["Flexible utility"]


def _has_forbidden_directive(insight: Mapping[str, Any]) -> bool:
    text = " ".join(
        (
            _text(insight.get("title")),
            _text(insight.get("detail")),
            " ".join(_unique_text(_as_list(insight.get("related_units")))),
        )
    ).casefold()
    return any(phrase in text for phrase in FORBIDDEN_DIRECTIVE_PHRASES)
