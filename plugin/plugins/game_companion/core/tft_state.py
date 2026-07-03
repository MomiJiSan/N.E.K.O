from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..profiles.tft.screen_regions import LAYOUT_AUGMENT_SELECT, LAYOUT_COMBAT, LAYOUT_NORMAL_SHOP

TFT_STATE_SCHEMA_VERSION = 1
LAYOUT_AUGMENT = "augment"
LAYOUT_POPUP = "popup"
LAYOUT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class TftShopSlot:
    slot: int
    state: str
    name: str | None
    cost: int | None
    confidence: float
    name_confidence: float
    cost_confidence: float
    name_source: str
    cost_source: str
    cost_inference: dict[str, Any] | None
    missing_fields: list[str]


@dataclass(frozen=True)
class TftShopState:
    slots: list[TftShopSlot]
    occupied_count: int
    slot_count: int
    partial_count: int
    slot_issues: list[dict[str, Any]]


@dataclass(frozen=True)
class TftAugmentOption:
    slot: int
    title: str
    description: str
    confidence: float


@dataclass(frozen=True)
class TftAugmentState:
    options: list[TftAugmentOption]
    option_count: int


@dataclass(frozen=True)
class TftCombatState:
    status: str
    details: list[dict[str, Any]]


@dataclass(frozen=True)
class TftFrameQuality:
    hover_contaminated: bool
    ocr_ready: bool
    blocked: bool


@dataclass(frozen=True)
class TftFrameState:
    type: str
    schema_version: int
    game: str
    layout: str
    readiness: str
    confidence: float
    source_frame: str
    timestamp: float | None
    source_context: dict[str, Any]
    shop: TftShopState | None
    augment: TftAugmentState | None
    combat: TftCombatState | None
    blockers: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    quality: TftFrameQuality
    summary: str


def build_tft_state(
    recognition: dict[str, Any],
    *,
    timestamp: float | None = None,
    source_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert focused TFT recognition into the stable runtime state consumed by NEKO."""

    source_context = dict(source_context or {})
    layout = _runtime_layout(str(recognition.get("layout") or LAYOUT_UNKNOWN), recognition)
    readiness = _readiness_status(recognition)
    blockers = _blockers(recognition)
    warnings = _warnings(recognition)
    success = bool(recognition.get("success"))
    confidence = _float(recognition.get("confidence"))
    quality = TftFrameQuality(
        hover_contaminated=readiness == "contaminated" or any(item.get("code") == "contaminated_by_hover" for item in blockers),
        ocr_ready=success and readiness in {"ready", "partial", "contaminated"},
        blocked=readiness == "blocked",
    )
    shop = _shop_state(recognition) if success and layout == LAYOUT_NORMAL_SHOP else None
    augment = _augment_state(recognition) if success and layout == LAYOUT_AUGMENT else None
    combat = TftCombatState(status="observed", details=[]) if success and layout == LAYOUT_COMBAT else None
    state = TftFrameState(
        type="tft_frame_state",
        schema_version=TFT_STATE_SCHEMA_VERSION,
        game="tft",
        layout=layout,
        readiness=readiness,
        confidence=confidence,
        source_frame=str(recognition.get("image_path") or ""),
        timestamp=timestamp,
        source_context=source_context,
        shop=shop,
        augment=augment,
        combat=combat,
        blockers=blockers,
        warnings=warnings,
        quality=quality,
        summary=_summary(
            success=success,
            layout=layout,
            readiness=readiness,
            shop=shop,
            augment=augment,
            blockers=blockers,
        ),
    )
    return asdict(state)


def _runtime_layout(layout: str, recognition: dict[str, Any]) -> str:
    if _readiness_status(recognition) == "contaminated":
        return LAYOUT_POPUP
    if layout == LAYOUT_AUGMENT_SELECT:
        return LAYOUT_AUGMENT
    if layout in {LAYOUT_NORMAL_SHOP, LAYOUT_COMBAT, LAYOUT_POPUP, LAYOUT_AUGMENT}:
        return layout
    return LAYOUT_UNKNOWN if not layout else layout


def _readiness_status(recognition: dict[str, Any]) -> str:
    if not recognition.get("success", False):
        return "blocked"
    readiness = recognition.get("readiness")
    if isinstance(readiness, dict):
        status = str(readiness.get("status") or "").strip().lower()
        if status:
            return status
    return "ready"


def _shop_state(recognition: dict[str, Any]) -> TftShopState:
    slots = [_shop_slot(slot) for slot in recognition.get("shop") or [] if isinstance(slot, dict)]
    slot_issues = [
        {"slot": slot.slot, "state": slot.state, "missing_fields": list(slot.missing_fields)}
        for slot in slots
        if slot.missing_fields
    ]
    return TftShopState(
        slots=slots,
        occupied_count=sum(1 for slot in slots if slot.state == "occupied"),
        slot_count=len(slots),
        partial_count=len(slot_issues),
        slot_issues=slot_issues,
    )


def _shop_slot(slot: dict[str, Any]) -> TftShopSlot:
    return TftShopSlot(
        slot=_int(slot.get("slot")),
        state=str(slot.get("state") or "unknown"),
        name=_str_or_none(slot.get("name") or slot.get("name_candidate")),
        cost=_slot_cost(slot),
        confidence=_float(slot.get("confidence")),
        name_confidence=_float(slot.get("name_confidence")),
        cost_confidence=_float(slot.get("cost_confidence")),
        name_source=_source(slot.get("name_candidate_source") or slot.get("name_source")),
        cost_source=_source(slot.get("cost_candidate_source") or slot.get("cost_source")),
        cost_inference=_cost_inference(slot.get("cost_inference")),
        missing_fields=_shop_slot_missing_fields(slot),
    )


def _shop_slot_missing_fields(slot: dict[str, Any]) -> list[str]:
    if slot.get("state") != "occupied":
        return []
    missing = []
    if not (slot.get("name") or slot.get("name_candidate")):
        missing.append("name")
    if _slot_cost(slot) is None:
        missing.append("cost")
    return missing


def _slot_cost(slot: dict[str, Any]) -> int | None:
    return _int_or_none(slot.get("cost") if slot.get("cost") is not None else slot.get("cost_candidate"))


def _source(value: Any) -> str:
    return str(value or "").strip()


def _cost_inference(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    allowed = {
        key: value.get(key)
        for key in ("method", "matched_name", "confidence")
        if value.get(key) is not None
    }
    return allowed or None


def _augment_state(recognition: dict[str, Any]) -> TftAugmentState:
    options = [_augment_option(option) for option in recognition.get("augments") or [] if isinstance(option, dict)]
    return TftAugmentState(options=options, option_count=len(options))


def _augment_option(option: dict[str, Any]) -> TftAugmentOption:
    return TftAugmentOption(
        slot=_int(option.get("slot") or option.get("option") or option.get("index")),
        title=str(option.get("title") or option.get("name") or ""),
        description=str(option.get("description") or ""),
        confidence=_float(option.get("confidence")),
    )


def _blockers(recognition: dict[str, Any]) -> list[dict[str, Any]]:
    if not recognition.get("success", False):
        error = recognition.get("error") if isinstance(recognition.get("error"), dict) else {}
        return [{"code": str(error.get("code") or "recognition_failed"), "message": str(error.get("message") or "")}]
    readiness = recognition.get("readiness")
    raw = readiness.get("blockers") if isinstance(readiness, dict) else []
    if not raw and isinstance(readiness, dict):
        raw = readiness.get("blocking_issues")
    blockers: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        blockers.extend({"code": str(code), "count": count} for code, count in raw.items())
    elif isinstance(raw, list):
        blockers.extend(dict(item) for item in raw if isinstance(item, dict))
    if not blockers and isinstance(readiness, dict) and readiness.get("main_blocker"):
        blockers.append({"code": str(readiness["main_blocker"])})
    return blockers


def _warnings(recognition: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in recognition.get("warnings") or [] if isinstance(item, dict)]


def _summary(
    *,
    success: bool,
    layout: str,
    readiness: str,
    shop: TftShopState | None,
    augment: TftAugmentState | None,
    blockers: list[dict[str, Any]],
) -> str:
    if not success:
        code = blockers[0].get("code") if blockers else "recognition_failed"
        return f"当前截图无法识别：{code}。"
    if readiness == "contaminated":
        return "当前画面被弹窗或悬浮提示遮挡，暂不作为可用识别状态。"
    if readiness == "blocked" and blockers:
        return f"当前画面暂不可用，主要阻塞原因是 {blockers[0].get('code')}。"
    if layout == LAYOUT_NORMAL_SHOP and readiness == "partial" and shop is not None:
        return _partial_shop_summary(shop)
    if layout == LAYOUT_NORMAL_SHOP and shop is not None:
        return f"当前是商店界面，{shop.slot_count} 个商店栏位可识别，{shop.occupied_count} 个有棋子。"
    if layout == LAYOUT_AUGMENT and augment is not None:
        return f"当前是强化符文选择界面，{augment.option_count} 个选项可识别。"
    if layout == LAYOUT_COMBAT:
        return "当前是战斗画面，商店和强化识别为不适用。"
    return "当前云顶画面已转换为运行时状态。"


def _partial_shop_summary(shop: TftShopState) -> str:
    issue_parts = []
    for issue in shop.slot_issues[:3]:
        fields = [str(field) for field in issue.get("missing_fields", [])]
        translated = "和".join("费用" if field == "cost" else "名字" if field == "name" else field for field in fields)
        if translated:
            issue_parts.append(f"slot {issue.get('slot')} 缺{translated}")
    detail = "，".join(issue_parts) if issue_parts else "部分栏位缺信息"
    return (
        f"当前是商店界面，{shop.slot_count} 个商店栏位可识别，{shop.occupied_count} 个有棋子；"
        f"商店可部分识别，{detail}。"
    )


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
