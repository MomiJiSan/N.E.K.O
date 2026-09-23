"""Sample-scoped local speech evidence; independent of endpoint authority.

Thresholds are experiment defaults, not ASR confidence or proof of identity.
The owner supplies normalized 16 kHz window offsets exactly once and seals at
its endpoint boundary. Network waits and replayed PCM never advance evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math


class AdmissionDecision(str, Enum):
    PENDING = "pending"
    ADMIT = "admit"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    sample_rate: int = 16_000
    minimum_voiced_ms: int = 224
    window_ms: int = 480
    minimum_occupancy: float = 0.65
    maximum_gap_ms: int = 96
    maximum_candidate_ms: int = 640
    onset_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000:
            raise ValueError("admission requires normalized 16 kHz audio")
        if (
            not 0
            < self.minimum_voiced_ms
            <= self.window_ms
            <= self.maximum_candidate_ms
        ):
            raise ValueError("invalid admission duration limits")
        if not 0 < self.maximum_gap_ms < self.maximum_candidate_ms:
            raise ValueError("invalid admission gap limit")
        if not 0 < self.minimum_occupancy <= 1 or not 0 < self.onset_probability <= 1:
            raise ValueError("invalid admission probability")


@dataclass(frozen=True, slots=True)
class SpeechEvidence:
    scope_id: str
    candidate_id: int
    audio_start_sample: int
    audio_end_sample: int
    observed_samples: int
    voiced_samples: int
    longest_speech_run_samples: int
    longest_gap_samples: int
    probability_mean: float
    probability_peak: float
    decision: AdmissionDecision
    reason: str
    rnnoise_mean: float | None = None
    playback_active: bool | None = None

    @property
    def observed_audio_ms(self) -> float:
        return self.observed_samples / 16

    @property
    def voiced_audio_ms(self) -> float:
        return self.voiced_samples / 16


class CandidateAdmission:
    """Bounded pending candidate plus frozen decision; not a session FSM.

    An accepted utterance stays accepted through silence until its owner seals
    it. A rejected candidate can be replaced by a later voiced window, with a
    new identity. A returned frozen snapshot never changes with its successor.
    """

    def __init__(self, scope_id: str, config: AdmissionConfig | None = None) -> None:
        self.scope_id = scope_id
        self.config = config or AdmissionConfig()
        self._cursor = 0
        self._candidate_sequence = 0
        self._snapshot: SpeechEvidence | None = None
        self._window: deque[tuple[int, int, bool]] = deque()
        self._speech_run = self._gap = 0
        self._weighted_probability = 0.0

    @property
    def snapshot(self) -> SpeechEvidence | None:
        return self._snapshot

    def seal(self) -> SpeechEvidence | None:
        snapshot = self._snapshot
        self._snapshot = None
        self._window.clear()
        self._speech_run = self._gap = 0
        self._weighted_probability = 0.0
        return snapshot

    def observe(
        self, start_sample: int, end_sample: int, probability: float
    ) -> SpeechEvidence | None:
        if start_sample < self._cursor:
            raise ValueError("duplicate or overlapping admission evidence")
        if end_sample <= start_sample or start_sample < 0:
            raise ValueError("invalid audio sample interval")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("invalid local speech probability")
        previous_cursor = self._cursor
        self._cursor = end_sample
        voiced = probability >= self.config.onset_probability
        old = self._snapshot
        if old is not None and old.decision is AdmissionDecision.REJECT:
            if not voiced:
                return old
            self.seal()
            old = None
        if old is None:
            if not voiced:
                return None
            self._candidate_sequence += 1
            old = SpeechEvidence(
                self.scope_id,
                self._candidate_sequence,
                start_sample,
                start_sample,
                0,
                0,
                0,
                0,
                0.0,
                0.0,
                AdmissionDecision.PENDING,
                "collecting",
            )
        # Missing windows are gaps, never inferred silence or human speech.
        missing = max(0, start_sample - max(previous_cursor, old.audio_start_sample))
        size = end_sample - start_sample
        self._gap += missing
        longest_gap = max(old.longest_gap_samples, self._gap)
        if voiced:
            self._speech_run = (self._speech_run if not missing else 0) + size
            self._gap = 0
        else:
            self._gap += size
            self._speech_run = 0
        longest_gap = max(longest_gap, self._gap)
        self._weighted_probability += probability * size
        observed = old.observed_samples + size
        self._window.append((start_sample, end_sample, voiced))
        cutoff = end_sample - self.config.window_ms * 16
        while self._window and self._window[0][1] <= cutoff:
            self._window.popleft()
        window_voiced = sum(
            end - max(start, cutoff) for start, end, speech in self._window if speech
        )
        span = end_sample - max(old.audio_start_sample, cutoff)
        decision, reason = old.decision, old.reason
        if decision is AdmissionDecision.PENDING:
            if longest_gap > self.config.maximum_gap_ms * 16:
                decision, reason = AdmissionDecision.REJECT, "speech_gap"
            elif (
                end_sample - old.audio_start_sample
                > self.config.maximum_candidate_ms * 16
            ):
                decision, reason = AdmissionDecision.REJECT, "candidate_timeout"
            elif (
                window_voiced >= self.config.minimum_voiced_ms * 16
                and window_voiced / span >= self.config.minimum_occupancy
            ):
                decision, reason = AdmissionDecision.ADMIT, "speech_window"
        self._snapshot = SpeechEvidence(
            self.scope_id,
            old.candidate_id,
            old.audio_start_sample,
            end_sample,
            observed,
            old.voiced_samples + (size if voiced else 0),
            max(old.longest_speech_run_samples, self._speech_run),
            longest_gap,
            self._weighted_probability / observed,
            max(old.probability_peak, probability),
            decision,
            reason,
        )
        return self._snapshot
