"""Bind one Owner profile activation to one independent-ASR runtime."""

from __future__ import annotations

import asyncio
import copy
import threading

from main_logic.asr_client.runtime import IndependentAsrRuntime
from main_logic.asr_client.speaker_shadow.asset_manifest import (
    CAMPPLUS_MODEL_ID,
    CAMPPLUS_MODEL_REVISION,
)
from main_logic.asr_client.speaker_shadow.campplus import (
    CAMPPLUS_EMBEDDING_DIM,
    CampPlusBackendFactory,
)
from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
    SpeakerShadowConfig,
    SpeakerShadowObservation,
)
from main_logic.asr_client.speaker_shadow.runtime import SpeakerShadowRuntime
from main_logic.voice_identity.contracts import SpeakerModelIdentity
from main_logic.voice_identity.profile import SpeakerProfile

from .policy import OwnerVoiceDecision, OwnerVoicePolicy


class OwnerVoiceAsrCompositionFactory:
    """Create repeatable observers for one runtime and activation generation."""

    def __init__(
        self,
        runtime: IndependentAsrRuntime,
        profile: SpeakerProfile,
        *,
        activation_generation: str,
        enforce: bool,
    ) -> None:
        if not isinstance(runtime, IndependentAsrRuntime):
            raise TypeError("runtime must be IndependentAsrRuntime")
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
        self._armed_candidates: set[SpeakerShadowCandidateKey] = set()
        self._diagnostics = {
            "observation_count": 0,
            "first_checkpoint_count": 0,
            "second_checkpoint_count": 0,
            "low_checkpoint_count": 0,
            "reject_decision_count": 0,
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

        policy = OwnerVoicePolicy()
        runtime = self._runtime
        generation = self._activation_generation
        enforce = self._enforce

        async def on_observation(observation: SpeakerShadowObservation) -> None:
            with self._lock:
                self._diagnostics["observation_count"] += 1
                if observation.checkpoint_ms == OwnerVoicePolicy.FIRST_CHECKPOINT_MS:
                    self._diagnostics["first_checkpoint_count"] += 1
                elif observation.checkpoint_ms == OwnerVoicePolicy.SECOND_CHECKPOINT_MS:
                    self._diagnostics["second_checkpoint_count"] += 1
                if any(blocked for _, blocked in observation.would_block):
                    self._diagnostics["low_checkpoint_count"] += 1
            if runtime._speaker_verifier_activation_generation != generation:
                self._resolve_armed_candidates()
                return
            result = policy.observe(
                candidate=observation.candidate,
                checkpoint_ms=observation.checkpoint_ms,
                similarity=observation.similarity,
                enforce=enforce,
            )
            if (
                enforce
                and observation.candidate.scope == "provider_candidate"
                and result.reason == "awaiting_second_low_observation"
            ):
                try:
                    armed = await runtime._arm_speaker_candidate_decision(
                        observation.candidate,
                        activation_generation=generation,
                    )
                except asyncio.CancelledError:
                    self._resolve_candidates((observation.candidate,))
                    raise
                except Exception:
                    self._resolve_candidates((observation.candidate,))
                    return
                if not armed:
                    return
                with self._lock:
                    keep_armed = bool(
                        not self._closed
                        and runtime._speaker_verifier_activation_generation
                        == generation
                    )
                    if keep_armed:
                        self._armed_candidates.add(observation.candidate)
                if not keep_armed:
                    self._resolve_candidates((observation.candidate,))
                return

            with self._lock:
                was_armed = observation.candidate in self._armed_candidates
                self._armed_candidates.discard(observation.candidate)
            if result.decision is OwnerVoiceDecision.REJECT:
                with self._lock:
                    self._diagnostics["reject_decision_count"] += 1
                if (
                    observation.candidate.scope == "provider_candidate"
                    and not was_armed
                ):
                    return
                try:
                    scheduled = runtime.request_speaker_candidate_rejection(
                        observation.candidate,
                        activation_generation=generation,
                    )
                except Exception:
                    scheduled = False
                if not scheduled:
                    self._resolve_candidates((observation.candidate,))
                return
            if was_armed:
                self._resolve_candidates((observation.candidate,))

        def on_backend_degraded() -> None:
            self._resolve_armed_candidates()
            if runtime._speaker_verifier_activation_generation == generation:
                runtime._mark_speaker_verifier_degraded()

        def on_backend_recovered() -> None:
            if runtime._speaker_verifier_activation_generation == generation:
                runtime._mark_speaker_verifier_healthy()

        return SpeakerShadowRuntime(
            backend_factory=backend_factory,
            config=SpeakerShadowConfig(
                enabled=True,
                similarity_thresholds=(OwnerVoicePolicy.SIMILARITY_THRESHOLD,),
                minimum_audio_ms=OwnerVoicePolicy.FIRST_CHECKPOINT_MS,
                maximum_audio_ms=OwnerVoicePolicy.SECOND_CHECKPOINT_MS,
                observation_checkpoints_ms=(
                    OwnerVoicePolicy.FIRST_CHECKPOINT_MS,
                    OwnerVoicePolicy.SECOND_CHECKPOINT_MS,
                ),
            ),
            on_observation=on_observation,
            on_backend_degraded=on_backend_degraded,
            on_backend_recovered=on_backend_recovered,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            armed_candidates = tuple(self._armed_candidates)
            self._armed_candidates.clear()
            self._profile.close()
        self._resolve_candidates(armed_candidates)

    def _resolve_armed_candidates(self) -> None:
        with self._lock:
            armed_candidates = tuple(self._armed_candidates)
            self._armed_candidates.clear()
        self._resolve_candidates(armed_candidates)

    def _resolve_candidates(
        self,
        candidates: tuple[SpeakerShadowCandidateKey, ...],
    ) -> None:
        for candidate in candidates:
            try:
                self._runtime._resolve_speaker_candidate_decision(
                    candidate,
                    activation_generation=self._activation_generation,
                    rejected=False,
                )
            except Exception:
                continue


__all__ = ["OwnerVoiceAsrCompositionFactory"]
