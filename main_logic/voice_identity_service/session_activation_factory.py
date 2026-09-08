"""Repeatable Owner voice-session activation runtime factory."""

from __future__ import annotations

import copy
import threading

from main_logic.voice_identity.profile import SpeakerProfile
from main_logic.voice_input.activation import ActivationGeneration

from .activation_runtime import (
    ActivationStatusCallback,
    VoiceSessionActivationRuntime,
    VoiceSessionActivationRuntimeConfig,
)
from .activation_scoring import CampPlusActivationScorer


class OwnerVoiceSessionActivationFactory:
    """Own profile material and create one scorer per microphone authority."""

    def __init__(
        self,
        runtime_owner: object,
        profile: SpeakerProfile,
        *,
        activation_generation: str,
        enforce: bool,
        config: VoiceSessionActivationRuntimeConfig | None = None,
    ) -> None:
        del runtime_owner
        if type(profile) is not SpeakerProfile:
            raise TypeError("profile must be SpeakerProfile")
        if type(activation_generation) is not str or not activation_generation.strip():
            raise ValueError("activation_generation must be a non-empty string")
        if type(enforce) is not bool:
            raise TypeError("enforce must be bool")
        self._profile = copy.copy(profile)
        self._activation_generation = activation_generation
        self._enforce = enforce
        self._config = config
        self._lock = threading.Lock()
        self._scorer_generation = 0
        self._closed = False

    @property
    def activation_generation(self) -> str:
        return self._activation_generation

    def create(
        self,
        generation: ActivationGeneration,
        output,
        *,
        status_callback: ActivationStatusCallback | None = None,
    ) -> VoiceSessionActivationRuntime:
        with self._lock:
            if self._closed:
                raise RuntimeError("Owner voice session activation factory is closed")
            self._scorer_generation += 1
            scorer_generation = self._scorer_generation
            profile = copy.copy(self._profile)
        try:
            scorer = CampPlusActivationScorer(
                profile,
                scorer_generation=scorer_generation,
            )
        finally:
            profile.close()
        return VoiceSessionActivationRuntime(
            generation,
            scorer,
            output,
            config=self._config,
            status_callback=status_callback,
            enabled=self._enforce,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._profile.close()


__all__ = ["OwnerVoiceSessionActivationFactory"]
