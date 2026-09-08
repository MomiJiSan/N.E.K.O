"""Lightweight type contracts for Core voice-session activation wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from main_logic.voice_turn.contracts import VoiceIngressToken

from .contracts import (
    ActivationDecision,
    ActivationGeneration,
    ActivationState,
    AudioFrame,
    OutputCommit,
)


@dataclass(frozen=True, slots=True)
class VoiceSessionActivationRouteContext:
    speech_probability: float | None
    rnnoise_available: bool | None
    rnnoise_evidence: object | None
    ingress_token: VoiceIngressToken | None
    captured_at: float | None


class VoiceSessionActivationRuntime(Protocol):
    generation: ActivationGeneration
    state: ActivationState

    async def prepare(self) -> ActivationDecision: ...

    async def feed(
        self,
        frame: AudioFrame,
        *,
        voice_activity: bool,
    ) -> ActivationDecision: ...

    async def close(self) -> None: ...


class VoiceSessionActivationFactory(Protocol):
    activation_generation: str

    def create(
        self,
        generation: ActivationGeneration,
        output: Callable[[AudioFrame], Awaitable[OutputCommit]],
        *,
        status_callback: Callable[[ActivationDecision], None] | None = None,
    ) -> VoiceSessionActivationRuntime: ...

    def close(self) -> None: ...


__all__ = [
    "VoiceSessionActivationFactory",
    "VoiceSessionActivationRouteContext",
    "VoiceSessionActivationRuntime",
]
