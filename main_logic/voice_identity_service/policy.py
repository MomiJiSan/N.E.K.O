"""Pure Owner-speaker policy for the independent-ASR soft filter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
    SpeakerShadowObservationKind,
)


class OwnerVoiceDecision(StrEnum):
    FORWARD = "forward"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class OwnerVoicePolicyResult:
    decision: OwnerVoiceDecision
    reason: str


class OwnerVoicePolicy:
    """Require two explicit low scores and otherwise forward the candidate."""

    FIRST_CHECKPOINT_MS = 1_500
    SECOND_CHECKPOINT_MS = 3_000
    SIMILARITY_THRESHOLD = 0.40
    DEFAULT_CANDIDATE_CAPACITY = 256

    def __init__(self, *, candidate_capacity: int = DEFAULT_CANDIDATE_CAPACITY) -> None:
        if type(candidate_capacity) is not int or candidate_capacity <= 0:
            raise ValueError("candidate_capacity must be a positive integer")
        self._candidate_capacity = candidate_capacity
        self._first_low: dict[SpeakerShadowCandidateKey, bool] = {}

    @property
    def pending_candidate_count(self) -> int:
        return len(self._first_low)

    def observe(
        self,
        *,
        candidate: SpeakerShadowCandidateKey,
        checkpoint_ms: int | None,
        similarity: float,
        enforce: bool,
        observation_kind: SpeakerShadowObservationKind = "checkpoint",
        audio_ms: int | None = None,
    ) -> OwnerVoicePolicyResult:
        if type(candidate) is not SpeakerShadowCandidateKey:
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "invalid_candidate"
            )
        if (
            type(observation_kind) is not str
            or observation_kind not in ("checkpoint", "completion_confirmation")
            or type(similarity) not in {int, float}
            or not math.isfinite(float(similarity))
            or not -1.0 <= float(similarity) <= 1.0
        ):
            self._first_low.pop(candidate, None)
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "invalid_observation"
            )

        if observation_kind == "completion_confirmation":
            return self._observe_completion_confirmation(
                candidate=candidate,
                checkpoint_ms=checkpoint_ms,
                audio_ms=audio_ms,
                similarity=float(similarity),
                enforce=enforce,
            )

        if type(checkpoint_ms) is not int or checkpoint_ms not in {
            self.FIRST_CHECKPOINT_MS,
            self.SECOND_CHECKPOINT_MS,
        }:
            self._first_low.pop(candidate, None)
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "invalid_observation"
            )

        low = float(similarity) < self.SIMILARITY_THRESHOLD
        if checkpoint_ms == self.FIRST_CHECKPOINT_MS:
            if self._first_low.pop(candidate, False):
                return OwnerVoicePolicyResult(
                    OwnerVoiceDecision.FORWARD, "invalid_observation"
                )
            if low:
                self._remember(candidate)
                return OwnerVoicePolicyResult(
                    OwnerVoiceDecision.FORWARD,
                    "awaiting_second_low_observation",
                )
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "owner_or_uncertain"
            )

        first_low = self._first_low.pop(candidate, False)
        if not first_low:
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "invalid_observation"
            )
        if first_low and low and enforce is True:
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.REJECT, "stable_clear_mismatch"
            )
        if first_low and low:
            return OwnerVoicePolicyResult(OwnerVoiceDecision.FORWARD, "shadow_only")
        return OwnerVoicePolicyResult(OwnerVoiceDecision.FORWARD, "owner_or_uncertain")

    def _observe_completion_confirmation(
        self,
        *,
        candidate: SpeakerShadowCandidateKey,
        checkpoint_ms: int | None,
        audio_ms: int | None,
        similarity: float,
        enforce: bool,
    ) -> OwnerVoicePolicyResult:
        first_low = self._first_low.pop(candidate, False)
        if (
            type(checkpoint_ms) is not int
            or checkpoint_ms != self.FIRST_CHECKPOINT_MS
            or type(audio_ms) is not int
            or not self.FIRST_CHECKPOINT_MS < audio_ms < self.SECOND_CHECKPOINT_MS
            or not first_low
        ):
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.FORWARD, "invalid_observation"
            )

        low = similarity < self.SIMILARITY_THRESHOLD
        if low and enforce is True:
            return OwnerVoicePolicyResult(
                OwnerVoiceDecision.REJECT, "stable_clear_mismatch"
            )
        if low:
            return OwnerVoicePolicyResult(OwnerVoiceDecision.FORWARD, "shadow_only")
        return OwnerVoicePolicyResult(OwnerVoiceDecision.FORWARD, "owner_or_uncertain")

    def forget(self, candidate: SpeakerShadowCandidateKey) -> None:
        if type(candidate) is SpeakerShadowCandidateKey:
            self._first_low.pop(candidate, None)

    def reset(self) -> None:
        self._first_low.clear()

    def _remember(self, candidate: SpeakerShadowCandidateKey) -> None:
        self._first_low[candidate] = True
        while len(self._first_low) > self._candidate_capacity:
            self._first_low.pop(next(iter(self._first_low)), None)


__all__ = ["OwnerVoiceDecision", "OwnerVoicePolicy", "OwnerVoicePolicyResult"]
