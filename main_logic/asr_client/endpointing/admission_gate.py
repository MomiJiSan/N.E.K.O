"""Separate raw endpoint activity from permission to open an ASR turn."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import logging
from uuid import uuid4

from main_logic.voice_turn.admission import (
    AdmissionDecision,
    CandidateAdmission,
    SpeechEvidence,
)
from main_logic.voice_turn.contracts import SpeechActivityEvent
from .silero_vad import SileroActivityGate, SileroVad

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdmissionActivity:
    activity: SpeechActivityEvent
    evidence: SpeechEvidence
    audio_start_sample: int


class AdmissionActivityGate(SileroActivityGate):
    """One Silero inference, two event views. Raw events retain their meaning.

    Only ``admission_events`` may open a new external user turn. ``feed`` still
    returns raw events for SmartTurn continuation and endpoint evaluation.
    Reset is owned by the detector worker; provider seal only ends admission.
    """

    def reset(self) -> None:
        super().reset()
        self._admission = CandidateAdmission(uuid4().hex)
        self._sample_cursor = 0
        self.admission_events: tuple[SpeechActivityEvent, ...] = ()
        self.admission_records: tuple[AdmissionActivity, ...] = ()
        self._published = False
        self._seen_samples = 0
        self._source_offset = 0
        self._rejected_end = 0
        self.event_evidence = None
        self.event_audio_start_sample = None
        self.had_admission = False

    @property
    def evidence(self):
        value = self._admission.snapshot
        if value is None:
            return None
        return replace(
            value,
            audio_start_sample=value.audio_start_sample + self._source_offset,
            audio_end_sample=value.audio_end_sample + self._source_offset,
        )

    @property
    def retained_start_sample(self):
        value = self._admission.snapshot
        if value is None:
            return None
        return self._source_offset + max(
            self._rejected_end, value.audio_start_sample - 2048, 0
        )

    def feed_at(self, pcm16: bytes, source_end_sample: int):
        self._seen_samples += len(pcm16) // 2
        self._source_offset = source_end_sample - self._seen_samples
        return self.feed(pcm16)

    def seal_admission(self):
        sealed = self._admission.seal()
        self._published = False
        self.admission_events = ()
        self._rejected_end = self._sample_cursor
        self.event_evidence = None
        self.event_audio_start_sample = None
        self.had_admission = False
        return sealed

    def process_probabilities(self, probabilities: Iterable[float]):
        raw_events = []
        records = []
        for probability in probabilities:
            # Boundaries belong to model windows, not websocket packets. One
            # large packet can contain a pause and a successor candidate.
            raw = super().process_probabilities((probability,))
            raw_events.extend(raw)
            previous = self._admission.snapshot
            end = self._sample_cursor + SileroVad.WINDOW_SAMPLES
            local = self._admission.observe(self._sample_cursor, end, probability)
            self._sample_cursor = end
            if local is not None and local.decision is AdmissionDecision.REJECT:
                self._rejected_end = max(self._rejected_end, local.audio_end_sample)
            evidence = self.evidence
            if evidence is not None and (
                previous is None
                or (previous.candidate_id, previous.decision)
                != (evidence.candidate_id, evidence.decision)
            ):
                logger.info(
                    "[voice-admission] scope=%s candidate=%s start_sample=%s end_sample=%s "
                    "observed_ms=%.1f voiced_ms=%.1f longest_run_ms=%.1f longest_gap_ms=%.1f "
                    "probability_mean=%.3f probability_peak=%.3f rnnoise_mean=%s "
                    "playback_active=%s decision=%s reason=%s",
                    evidence.scope_id,
                    evidence.candidate_id,
                    evidence.audio_start_sample,
                    evidence.audio_end_sample,
                    evidence.observed_audio_ms,
                    evidence.voiced_audio_ms,
                    evidence.longest_speech_run_samples / 16,
                    evidence.longest_gap_samples / 16,
                    evidence.probability_mean,
                    evidence.probability_peak,
                    evidence.rnnoise_mean,
                    evidence.playback_active,
                    evidence.decision.value,
                    evidence.reason,
                )
            if evidence is not None and evidence.decision is AdmissionDecision.ADMIT:
                admitted = (
                    raw if self._published else (SpeechActivityEvent.SPEECH_STARTED,)
                )
                self._published = True
                self.had_admission = True
                records.extend(
                    AdmissionActivity(event, evidence, self.retained_start_sample)
                    for event in admitted
                )
            if SpeechActivityEvent.CANDIDATE_PAUSE in raw:
                # Only the potential next onset needs new evidence. Streaming
                # ASR audio and raw SmartTurn continuation stay uninterrupted.
                self._admission.seal()
                self._published = False
                self._rejected_end = self._sample_cursor
        self.admission_records = tuple(records)
        self.admission_events = tuple(record.activity for record in records)
        self.event_evidence = records[-1].evidence if records else self.evidence
        self.event_audio_start_sample = (
            records[-1].audio_start_sample if records else self.retained_start_sample
        )
        return tuple(raw_events)
