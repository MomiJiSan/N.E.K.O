from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GameType(str, Enum):
    TYPE_A = "pure_offline"
    TYPE_B = "offline_onlineable"
    TYPE_C = "online_non_competitive"
    TYPE_D = "online_competitive"


class RuntimeMode(str, Enum):
    OFFLINE = "offline"
    ONLINE = "online"


class Capability(str, Enum):
    SCREEN_OBSERVE = "screen_observe"
    OCR = "ocr"
    VISION_CLASSIFY = "vision_classify"
    NEKO_CONTEXT = "neko_context"
    INPUT_CONTROL = "input_control"
    AUTO_CLICK = "auto_click"
    MEMORY_READ = "memory_read"
    PACKET_READ = "packet_read"
    AUTOMATED_GAMEPLAY = "automated_gameplay"

    @classmethod
    def coerce(cls, value: "Capability | str") -> "Capability":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown capability: {value!r}") from exc


DEFAULT_ALLOWED_CAPABILITIES: tuple[Capability, ...] = (
    Capability.SCREEN_OBSERVE,
    Capability.OCR,
    Capability.VISION_CLASSIFY,
    Capability.NEKO_CONTEXT,
)

DEFAULT_DENIED_CAPABILITIES: tuple[Capability, ...] = (
    Capability.INPUT_CONTROL,
    Capability.AUTO_CLICK,
    Capability.MEMORY_READ,
    Capability.PACKET_READ,
    Capability.AUTOMATED_GAMEPLAY,
)


@dataclass(frozen=True)
class CapabilityGate:
    allowed: tuple[Capability | str, ...] = field(
        default_factory=lambda: DEFAULT_ALLOWED_CAPABILITIES
    )
    denied: tuple[Capability | str, ...] = field(
        default_factory=lambda: DEFAULT_DENIED_CAPABILITIES
    )

    def __post_init__(self) -> None:
        allowed = tuple(Capability.coerce(capability) for capability in self.allowed)
        denied = tuple(Capability.coerce(capability) for capability in self.denied)
        overlap = set(allowed) & set(denied)
        if overlap:
            names = ", ".join(sorted(capability.value for capability in overlap))
            raise ValueError(f"capabilities cannot be both allowed and denied: {names}")
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "denied", denied)

    def allows(self, capability: Capability | str) -> bool:
        return Capability.coerce(capability) in self.allowed

    def denies(self, capability: Capability | str) -> bool:
        return Capability.coerce(capability) in self.denied

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "allowed": [capability.value for capability in self.allowed],
            "denied": [capability.value for capability in self.denied],
        }
