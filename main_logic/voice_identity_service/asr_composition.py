"""Bind one Owner profile activation to one independent-ASR evidence sink."""

from __future__ import annotations

import copy
import threading
from typing import Protocol

from main_logic.asr_client.admission.contracts import (
    CaptureClosed,
    SpeakerCheckpointKind,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
)
from main_logic.asr_client.speaker_shadow.asset_manifest import (
    CAMPPLUS_MODEL_ID,
    CAMPPLUS_MODEL_REVISION,
)
from main_logic.asr_client.speaker_shadow.campplus import (
    CAMPPLUS_EMBEDDING_DIM,
    CampPlusBackendFactory,
)
from main_logic.asr_client.speaker_shadow.contracts import (
    MAX_SPEAKER_SHADOW_CANDIDATE_AUDIO_MS,
    SpeakerShadowCompletion,
    SpeakerShadowConfig,
    SpeakerShadowEvidenceEvent,
    SpeakerShadowObservation,
)
from main_logic.asr_client.speaker_shadow.runtime import SpeakerShadowRuntime
from main_logic.voice_identity.contracts import SpeakerModelIdentity
from main_logic.voice_identity.profile import SpeakerProfile

from .policy import OwnerVoiceClassification, OwnerVoicePolicy


class _OwnerVoiceEvidenceSink(Protocol):
    """One-way bridge; it has no transcript or reservation operations."""

    def _accept_speaker_evidence_fact(
        self,
        fact: SpeakerLow | SpeakerHigh | SpeakerUnavailable,
        *,
        activation_generation: str,
        enforce: bool,
    ) -> bool: ...

    def _close_speaker_evidence(
        self,
        closed: CaptureClosed,
        *,
        activation_generation: str,
        enforce: bool,
        evidence_complete: bool,
    ) -> bool: ...

    def _mark_speaker_evidence_backend_degraded(
        self,
        *,
        activation_generation: str,
    ) -> None: ...

    def _mark_speaker_evidence_backend_healthy(
        self,
        *,
        activation_generation: str,
    ) -> None: ...


class OwnerVoiceAsrCompositionFactory:
    """Create repeatable observers for one activation generation."""

    def __init__(
        self,
        runtime: _OwnerVoiceEvidenceSink,
        profile: SpeakerProfile,
        *,
        activation_generation: str,
        enforce: bool,
    ) -> None:
        required_methods = (
            "_accept_speaker_evidence_fact",
            "_close_speaker_evidence",
            "_mark_speaker_evidence_backend_degraded",
            "_mark_speaker_evidence_backend_healthy",
        )
        if any(not callable(getattr(runtime, name, None)) for name in required_methods):
            raise TypeError("runtime must provide the Owner voice evidence sink")
        if type(profile) is not SpeakerProfile:
            raise TypeError("profile must be SpeakerProfile")
        if type(activation_generation) is not str or not activation_generation.strip():
            raise ValueError("activation_generation must be a non-empty string")
        if type(enforce) is not bool:
            raise TypeError("enforce must be bool")
        self._runtime = runtime
        self._profile = copy.copy(profile)
        self._activation_generation = activation_generation
        self._enforce = enforce
        self._lock = threading.Lock()
        self._closed = False
        self._diagnostics = {
            "observation_count": 0,
            "first_checkpoint_count": 0,
            "second_checkpoint_count": 0,
            "low_checkpoint_count": 0,
            "reject_decision_count": 0,
            "speaker_completion_count": 0,
            "speaker_completion_before_first_checkpoint_count": 0,
            "speaker_completion_after_first_checkpoint_count": 0,
            "speaker_completion_stale_count": 0,
        }

    @property
    def activation_generation(self) -> str:
        return self._activation_generation

    def diagnostics_snapshot(self) -> dict[str, int]:
        """Return aggregate decision counters without biometric material."""

        with self._lock:
            return dict(self._diagnostics)

    def __call__(self) -> SpeakerShadowRuntime:
        with self._lock:
            if self._closed:
                raise RuntimeError("Owner voice composition factory is closed")
            reference = self._profile.clone_reference()
        embedding = None
        try:
            expected_identity = SpeakerModelIdentity(
                CAMPPLUS_MODEL_ID,
                CAMPPLUS_MODEL_REVISION,
                CAMPPLUS_EMBEDDING_DIM,
            )
            if reference.model_identity != expected_identity:
                raise ValueError("speaker profile model identity does not match CAM++")
            embedding = reference.copy_embedding()
            backend_factory = CampPlusBackendFactory(embedding)
        finally:
            if embedding is not None:
                embedding.fill(0.0)
            reference.close()

        runtime = self._runtime
        generation = self._activation_generation
        enforce = self._enforce

        def on_evidence(event: SpeakerShadowEvidenceEvent) -> None:
            with self._lock:
                factory_closed = self._closed
            if factory_closed:
                if isinstance(event, SpeakerShadowCompletion):
                    with self._lock:
                        self._diagnostics["speaker_completion_stale_count"] += 1
                return

            if isinstance(event, SpeakerShadowCompletion):
                self._record_completion(event)
                through_sequence_no = event.through_sequence_no
                if not event.evidence_complete:
                    through_sequence_no = max(1, through_sequence_no)
                    runtime._accept_speaker_evidence_fact(
                        SpeakerUnavailable(
                            candidate=event.candidate,
                            sequence_no=through_sequence_no,
                        ),
                        activation_generation=generation,
                        enforce=enforce,
                    )
                runtime._close_speaker_evidence(
                    CaptureClosed(
                        candidate=event.candidate,
                        through_sequence_no=through_sequence_no,
                    ),
                    activation_generation=generation,
                    enforce=enforce,
                    evidence_complete=event.evidence_complete,
                )
                return

            assert isinstance(event, SpeakerShadowObservation)
            checkpoint_kind = self._checkpoint_kind(event)
            with self._lock:
                self._diagnostics["observation_count"] += 1
                if checkpoint_kind is SpeakerCheckpointKind.FIRST:
                    self._diagnostics["first_checkpoint_count"] += 1
                elif checkpoint_kind is SpeakerCheckpointKind.SECOND:
                    self._diagnostics["second_checkpoint_count"] += 1

            if not event.evidence_available:
                fact: SpeakerLow | SpeakerHigh | SpeakerUnavailable = (
                    SpeakerUnavailable(event.candidate, event.sequence_no)
                )
            else:
                result = OwnerVoicePolicy.classify(
                    checkpoint_ms=event.checkpoint_ms,
                    similarity=event.similarity,
                    observation_kind=event.observation_kind,
                    audio_ms=event.audio_ms,
                )
                if (
                    result.classification is OwnerVoiceClassification.LOW
                    and checkpoint_kind is not None
                ):
                    fact = SpeakerLow(
                        event.candidate,
                        event.sequence_no,
                        checkpoint_kind,
                    )
                    with self._lock:
                        self._diagnostics["low_checkpoint_count"] += 1
                        if checkpoint_kind in {
                            SpeakerCheckpointKind.SECOND,
                            SpeakerCheckpointKind.COMPLETION_CONFIRMATION,
                        }:
                            self._diagnostics["reject_decision_count"] += 1
                elif result.classification is OwnerVoiceClassification.HIGH:
                    fact = SpeakerHigh(event.candidate, event.sequence_no)
                else:
                    fact = SpeakerUnavailable(event.candidate, event.sequence_no)
            runtime._accept_speaker_evidence_fact(
                fact,
                activation_generation=generation,
                enforce=enforce,
            )

        def on_backend_degraded() -> None:
            runtime._mark_speaker_evidence_backend_degraded(
                activation_generation=generation,
            )

        def on_backend_recovered() -> None:
            runtime._mark_speaker_evidence_backend_healthy(
                activation_generation=generation,
            )

        return SpeakerShadowRuntime(
            backend_factory=backend_factory,
            config=SpeakerShadowConfig(
                enabled=True,
                similarity_thresholds=(OwnerVoicePolicy.SIMILARITY_THRESHOLD,),
                minimum_audio_ms=OwnerVoicePolicy.FIRST_CHECKPOINT_MS,
                maximum_audio_ms=MAX_SPEAKER_SHADOW_CANDIDATE_AUDIO_MS,
                observation_checkpoints_ms=(
                    OwnerVoicePolicy.FIRST_CHECKPOINT_MS,
                    OwnerVoicePolicy.SECOND_CHECKPOINT_MS,
                ),
                completion_confirmation_scopes=(
                    ("provider_candidate",) if enforce else ()
                ),
                pending_observation_gate_scopes=(
                    ("provider_candidate",) if enforce else ()
                ),
                backend_prewarm_scopes=(
                    ("provider_candidate",) if enforce else ()
                ),
            ),
            on_evidence=on_evidence,
            on_backend_degraded=on_backend_degraded,
            on_backend_recovered=on_backend_recovered,
        )

    @staticmethod
    def _checkpoint_kind(
        observation: SpeakerShadowObservation,
    ) -> SpeakerCheckpointKind | None:
        if observation.observation_kind == "completion_confirmation":
            return SpeakerCheckpointKind.COMPLETION_CONFIRMATION
        if observation.checkpoint_ms == OwnerVoicePolicy.FIRST_CHECKPOINT_MS:
            return SpeakerCheckpointKind.FIRST
        if observation.checkpoint_ms == OwnerVoicePolicy.SECOND_CHECKPOINT_MS:
            return SpeakerCheckpointKind.SECOND
        return None

    def _record_completion(self, completion: SpeakerShadowCompletion) -> None:
        with self._lock:
            self._diagnostics["speaker_completion_count"] += 1
            if completion.last_checkpoint_ms is None:
                self._diagnostics[
                    "speaker_completion_before_first_checkpoint_count"
                ] += 1
            else:
                self._diagnostics[
                    "speaker_completion_after_first_checkpoint_count"
                ] += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._profile.close()


__all__ = ["OwnerVoiceAsrCompositionFactory"]
