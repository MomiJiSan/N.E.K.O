from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..safety.models import Capability, CapabilityGate, GameType, RuntimeMode


@dataclass(frozen=True)
class ProfileMetadata:
    profile_id: str
    display_name: str
    game_type: GameType
    default_runtime_mode: RuntimeMode = RuntimeMode.OFFLINE
    description: str = ""
    capability_gate: CapabilityGate = field(default_factory=CapabilityGate)
    capabilities: tuple[Capability | str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = self.profile_id.strip().lower()
        if not normalized:
            raise ValueError("profile_id is required")
        object.__setattr__(self, "profile_id", normalized)
        if not isinstance(self.capability_gate, CapabilityGate):
            raise TypeError("capability_gate must be a CapabilityGate")

        capabilities = (
            tuple(Capability.coerce(capability) for capability in self.capabilities)
            if self.capabilities
            else self.capability_gate.allowed
        )
        denied = set(self.capability_gate.denied)
        blocked = [capability.value for capability in capabilities if capability in denied]
        if blocked:
            names = ", ".join(sorted(blocked))
            raise ValueError(f"profile capabilities are denied by gate: {names}")
        object.__setattr__(self, "capabilities", capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "display_name": self.display_name,
            "game_type": self.game_type.value,
            "default_runtime_mode": self.default_runtime_mode.value,
            "description": self.description,
            "capabilities": [capability.value for capability in self.capabilities],
            "capability_gate": self.capability_gate.to_dict(),
        }


class ProfileRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, ProfileMetadata] = {}

    def register(self, profile: ProfileMetadata) -> None:
        if profile.profile_id in self._profiles:
            raise ValueError(f"duplicate profile_id: {profile.profile_id}")
        self._profiles[profile.profile_id] = profile

    def has(self, profile_id: str) -> bool:
        return str(profile_id or "").strip().lower() in self._profiles

    def get(self, profile_id: str) -> ProfileMetadata | None:
        return self._profiles.get(str(profile_id or "").strip().lower())

    def list(self) -> list[ProfileMetadata]:
        return [self._profiles[key] for key in sorted(self._profiles)]
