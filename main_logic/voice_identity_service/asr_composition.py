"""Bind one Owner profile activation to one independent-ASR runtime."""

from __future__ import annotations

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
    MAX_SPEAKER_SHADOW_CANDIDATE_AUDIO_MS,
    SpeakerShadowCandidateKey,
    SpeakerShadowCompletion,
    SpeakerShadowConfig,
    SpeakerShadowObservation,
)
from main_logic.asr_client.speaker_shadow.runtime import SpeakerShadowRuntime
from main_logic.voice_identity.contracts import SpeakerModelIdentity
from main_logic.voice_identity.profile import SpeakerProfile

from .policy import OwnerVoiceDecision, OwnerVoicePolicy


class OwnerVoiceAsrCompositionFactory:
    """Create repeatable observers for one runtime and activation generation."""

    _ARMED_CANDIDATE_CAPACITY = 256
    _TERMINAL_CANDIDATE_CAPACITY = 256

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
        # Insertion ordered so capacity eviction is deterministic. This remains
        # bookkeeping only; IndependentAsrRuntime's exact gate owns authority.
        self._armed_candidates: dict[SpeakerShadowCandidateKey, bool] = {}
        # Bounded tombstones fence detached observations that finish after the
        # same candidate's terminal callback has already cleaned policy state.
        self._terminal_candidates: dict[SpeakerShadowCandidateKey, None] = {}
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

        policy = OwnerVoicePolicy()
        runtime = self._runtime
        generation = self._activation_generation
        enforce = self._enforce

        async def on_observation(observation: SpeakerShadowObservation) -> None:
            with self._lock:
                self._diagnostics["observation_count"] += 1
                if observation.observation_kind == "checkpoint":
                    if (
                        observation.checkpoint_ms
                        == OwnerVoicePolicy.FIRST_CHECKPOINT_MS
                    ):
                        self._diagnostics["first_checkpoint_count"] += 1
                    elif (
                        observation.checkpoint_ms
                        == OwnerVoicePolicy.SECOND_CHECKPOINT_MS
                    ):
                        self._diagnostics["second_checkpoint_count"] += 1
                    if any(blocked for _, blocked in observation.would_block):
                        self._diagnostics["low_checkpoint_count"] += 1
                ignore_terminal = bool(
                    self._closed
                    or observation.candidate in self._terminal_candidates
                )
            if ignore_terminal:
                policy.forget(observation.candidate)
                return
            if (
                runtime._speaker_verifier_activation_generation != generation
                or runtime._speaker_verifier_degraded
            ):
                self._resolve_armed_candidates()
                policy.forget(observation.candidate)
                return
            result = policy.observe(
                candidate=observation.candidate,
                checkpoint_ms=observation.checkpoint_ms,
                similarity=observation.similarity,
                enforce=enforce,
                observation_kind=observation.observation_kind,
                audio_ms=observation.audio_ms,
            )
            if (
                enforce
                and observation.candidate.scope == "provider_candidate"
                and result.reason == "awaiting_second_low_observation"
            ):
                with self._lock:
                    stale_same_key = self._armed_candidates.pop(
                        observation.candidate,
                        False,
                    )
                if stale_same_key:
                    self._resolve_candidates((observation.candidate,))
                try:
                    armed = runtime._request_speaker_candidate_decision_arm(
                        observation.candidate,
                        activation_generation=generation,
                    )
                except Exception:
                    self._resolve_candidates((observation.candidate,))
                    return
                if not armed:
                    return
                evicted_candidates: tuple[SpeakerShadowCandidateKey, ...] = ()
                with self._lock:
                    keep_armed = bool(
                        not self._closed
                        and runtime._speaker_verifier_activation_generation
                        == generation
                        and not runtime._speaker_verifier_degraded
                        and observation.candidate
                        not in self._terminal_candidates
                    )
                    if keep_armed:
                        self._armed_candidates[observation.candidate] = True
                        if (
                            len(self._armed_candidates)
                            > self._ARMED_CANDIDATE_CAPACITY
                        ):
                            evicted = next(iter(self._armed_candidates))
                            self._armed_candidates.pop(evicted, None)
                            evicted_candidates = (evicted,)
                if not keep_armed:
                    policy.forget(observation.candidate)
                    self._resolve_candidates((observation.candidate,))
                elif evicted_candidates:
                    self._resolve_candidates(evicted_candidates)
                return

            with self._lock:
                was_armed = self._armed_candidates.pop(
                    observation.candidate,
                    False,
                )
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
            if (
                enforce
                and observation.candidate.scope == "provider_candidate"
            ) or was_armed:
                self._resolve_candidates((observation.candidate,))

        async def on_completion(completion: SpeakerShadowCompletion) -> None:
            policy.forget(completion.candidate)
            activation_is_stale = bool(
                runtime._speaker_verifier_activation_generation != generation
            )
            with self._lock:
                factory_is_closed = self._closed
                if not factory_is_closed:
                    self._terminal_candidates.pop(completion.candidate, None)
                    self._terminal_candidates[completion.candidate] = None
                    while (
                        len(self._terminal_candidates)
                        > self._TERMINAL_CANDIDATE_CAPACITY
                    ):
                        oldest = next(iter(self._terminal_candidates))
                        self._terminal_candidates.pop(oldest, None)
                self._diagnostics["speaker_completion_count"] += 1
                if completion.last_checkpoint_ms is None:
                    self._diagnostics[
                        "speaker_completion_before_first_checkpoint_count"
                    ] += 1
                else:
                    self._diagnostics[
                        "speaker_completion_after_first_checkpoint_count"
                    ] += 1
                if factory_is_closed or activation_is_stale:
                    self._diagnostics["speaker_completion_stale_count"] += 1
                was_armed = False
                if completion.candidate.scope == "provider_candidate":
                    was_armed = self._armed_candidates.pop(
                        completion.candidate,
                        False,
                    )
            if (
                enforce
                and completion.candidate.scope == "provider_candidate"
            ) or was_armed:
                self._resolve_candidates((completion.candidate,))

        def on_backend_degraded() -> None:
            policy.reset()
            if runtime._speaker_verifier_activation_generation == generation:
                runtime._mark_speaker_verifier_degraded()
            self._resolve_armed_candidates()

        def on_backend_recovered() -> None:
            if runtime._speaker_verifier_activation_generation == generation:
                runtime._mark_speaker_verifier_healthy()

        return SpeakerShadowRuntime(
            backend_factory=backend_factory,
            config=SpeakerShadowConfig(
                enabled=True,
                similarity_thresholds=(OwnerVoicePolicy.SIMILARITY_THRESHOLD,),
                minimum_audio_ms=OwnerVoicePolicy.FIRST_CHECKPOINT_MS,
                # Exact Provider reconciliation may merge a short head with
                # a longer tentative tail (for example 0.8s + 2.5s).  Keep
                # the scoring checkpoints at 1.5s/3.0s while allowing the
                # shared bounded runtime to retain the full reconciled range.
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
            on_observation=on_observation,
            on_completion=on_completion,
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
            self._terminal_candidates.clear()
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
