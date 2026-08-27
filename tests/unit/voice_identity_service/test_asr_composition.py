from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import numpy as np
import pytest

from main_logic.asr_client.runtime import AsrRuntimeCallbacks, IndependentAsrRuntime
from main_logic.asr_client.speaker_shadow.asset_manifest import (
    CAMPPLUS_MODEL_ID,
    CAMPPLUS_MODEL_REVISION,
)
from main_logic.asr_client.speaker_shadow.campplus import CAMPPLUS_EMBEDDING_DIM
from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
    SpeakerShadowObservation,
)
from main_logic.voice_identity.contracts import SpeakerModelIdentity
from main_logic.voice_identity.profile import SpeakerProfile
from main_logic.voice_identity.reference import SpeakerReference
from main_logic.voice_identity_service.asr_composition import (
    OwnerVoiceAsrCompositionFactory,
)


def _runtime() -> IndependentAsrRuntime:
    return IndependentAsrRuntime(
        AsrRuntimeCallbacks(
            display_name=lambda: "owner-voice-composition-test",
            on_prepare_turn=AsyncMock(return_value=True),
            on_partial=AsyncMock(),
            on_final=AsyncMock(),
            on_turn_abandoned=AsyncMock(),
            on_failure=AsyncMock(),
            on_status=AsyncMock(),
            on_lifecycle=AsyncMock(),
        )
    )


def _profile(
    identity: SpeakerModelIdentity | None = None,
) -> SpeakerProfile:
    model_identity = identity or SpeakerModelIdentity(
        CAMPPLUS_MODEL_ID,
        CAMPPLUS_MODEL_REVISION,
        CAMPPLUS_EMBEDDING_DIM,
    )
    embedding = np.arange(
        1,
        model_identity.embedding_dimension + 1,
        dtype=np.float32,
    )
    reference = SpeakerReference(model_identity, embedding)
    embedding.fill(0.0)
    try:
        return SpeakerProfile("profile-generation", reference)
    finally:
        reference.close()


def _observation(
    candidate: SpeakerShadowCandidateKey,
    *,
    checkpoint_ms: int,
    similarity: float,
) -> SpeakerShadowObservation:
    return SpeakerShadowObservation(
        candidate=candidate,
        similarity=similarity,
        would_block=((0.40, similarity < 0.40),),
        audio_ms=checkpoint_ms,
        checkpoint_ms=checkpoint_ms,
    )


def _install_gate_spies(
    monkeypatch: pytest.MonkeyPatch,
    runtime: IndependentAsrRuntime,
    *,
    arm_result: bool = True,
) -> tuple[
    list[tuple[SpeakerShadowCandidateKey, str]],
    list[tuple[SpeakerShadowCandidateKey, str, bool]],
]:
    armed: list[tuple[SpeakerShadowCandidateKey, str]] = []
    resolved: list[tuple[SpeakerShadowCandidateKey, str, bool]] = []

    async def arm(
        candidate: SpeakerShadowCandidateKey,
        *,
        activation_generation: str,
    ) -> bool:
        armed.append((candidate, activation_generation))
        return arm_result

    def resolve(
        candidate: SpeakerShadowCandidateKey,
        *,
        activation_generation: str,
        rejected: bool = False,
    ) -> bool:
        resolved.append((candidate, activation_generation, rejected))
        return True

    monkeypatch.setattr(
        runtime,
        "_arm_speaker_candidate_decision",
        arm,
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_resolve_speaker_candidate_decision",
        resolve,
        raising=False,
    )
    return armed, resolved


@pytest.mark.parametrize(
    ("enforce", "expected_requests"),
    [(True, 1), (False, 0)],
    ids=["enforce", "shadow"],
)
async def test_composition_uses_two_checkpoints_and_enforces_only_in_enforce_mode(
    monkeypatch: pytest.MonkeyPatch,
    enforce: bool,
    expected_requests: int,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-7"
    profile = _profile()
    requests: list[tuple[SpeakerShadowCandidateKey, str]] = []
    armed, resolved = _install_gate_spies(monkeypatch, runtime)

    def request_rejection(
        candidate: SpeakerShadowCandidateKey,
        *,
        activation_generation: str,
    ) -> bool:
        requests.append((candidate, activation_generation))
        return True

    monkeypatch.setattr(
        runtime,
        "request_speaker_candidate_rejection",
        request_rejection,
    )
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-7",
        enforce=enforce,
    )
    shadow = factory()
    candidate = SpeakerShadowCandidateKey(3, 9, "provider_candidate")

    assert shadow._config.minimum_audio_ms == 1_500
    assert shadow._config.maximum_audio_ms == 3_000
    assert shadow._config.observation_checkpoints_ms == (1_500, 3_000)
    assert shadow._config.similarity_thresholds == (0.40,)
    callback = shadow._on_observation
    assert callback is not None

    await callback(
        _observation(candidate, checkpoint_ms=1_500, similarity=0.20)
    )
    assert requests == []
    await callback(
        _observation(candidate, checkpoint_ms=3_000, similarity=0.20)
    )

    assert factory.diagnostics_snapshot() == {
        "observation_count": 2,
        "first_checkpoint_count": 1,
        "second_checkpoint_count": 1,
        "low_checkpoint_count": 2,
        "reject_decision_count": expected_requests,
    }
    assert len(requests) == expected_requests
    if expected_requests:
        assert requests == [(candidate, "activation-7")]
        assert armed == [(candidate, "activation-7")]
    else:
        assert armed == []
    assert resolved == []
    await shadow.close()
    factory.close()
    profile.close()


@pytest.mark.parametrize(
    ("checkpoint_ms", "similarity"),
    [(3_000, 0.80), (2_500, 0.20)],
    ids=["owner", "invalid"],
)
async def test_second_forward_or_invalid_observation_resolves_armed_gate(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_ms: int,
    similarity: float,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-forward"
    profile = _profile()
    armed, resolved = _install_gate_spies(monkeypatch, runtime)
    requests: list[SpeakerShadowCandidateKey] = []
    monkeypatch.setattr(
        runtime,
        "request_speaker_candidate_rejection",
        lambda candidate, **_kwargs: requests.append(candidate) or True,
    )
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-forward",
        enforce=True,
    )
    shadow = factory()
    callback = shadow._on_observation
    assert callback is not None
    candidate = SpeakerShadowCandidateKey(4, 2, "provider_candidate")

    await callback(_observation(candidate, checkpoint_ms=1_500, similarity=0.20))
    await callback(
        _observation(
            candidate,
            checkpoint_ms=checkpoint_ms,
            similarity=similarity,
        )
    )

    assert armed == [(candidate, "activation-forward")]
    assert resolved == [(candidate, "activation-forward", False)]
    assert requests == []
    await shadow.close()
    factory.close()
    profile.close()


async def test_rejection_schedule_failure_resolves_armed_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-failed-schedule"
    profile = _profile()
    armed, resolved = _install_gate_spies(monkeypatch, runtime)
    monkeypatch.setattr(
        runtime,
        "request_speaker_candidate_rejection",
        lambda *_args, **_kwargs: False,
    )
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-failed-schedule",
        enforce=True,
    )
    shadow = factory()
    callback = shadow._on_observation
    assert callback is not None
    candidate = SpeakerShadowCandidateKey(5, 3, "provider_candidate")

    await callback(_observation(candidate, checkpoint_ms=1_500, similarity=0.20))
    await callback(_observation(candidate, checkpoint_ms=3_000, similarity=0.20))

    assert armed == [(candidate, "activation-failed-schedule")]
    assert resolved == [(candidate, "activation-failed-schedule", False)]
    await shadow.close()
    factory.close()
    profile.close()


@pytest.mark.parametrize("boundary", ["arm_failed", "degraded"])
async def test_second_low_never_rejects_without_live_armed_gate(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-no-gate"
    profile = _profile()
    armed, resolved = _install_gate_spies(
        monkeypatch,
        runtime,
        arm_result=boundary != "arm_failed",
    )
    requests: list[SpeakerShadowCandidateKey] = []
    monkeypatch.setattr(
        runtime,
        "request_speaker_candidate_rejection",
        lambda candidate, **_kwargs: requests.append(candidate) or True,
    )
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-no-gate",
        enforce=True,
    )
    shadow = factory()
    callback = shadow._on_observation
    assert callback is not None
    candidate = SpeakerShadowCandidateKey(7, 5, "provider_candidate")

    await callback(_observation(candidate, checkpoint_ms=1_500, similarity=0.20))
    if boundary == "degraded":
        shadow._mark_backend_degraded()
    await callback(_observation(candidate, checkpoint_ms=3_000, similarity=0.20))

    assert armed == [(candidate, "activation-no-gate")]
    assert requests == []
    assert resolved == (
        [(candidate, "activation-no-gate", False)]
        if boundary == "degraded"
        else []
    )
    await shadow.close()
    factory.close()
    profile.close()


async def test_smart_turn_candidate_keeps_ungated_rejection_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-smart-turn"
    profile = _profile()
    armed, resolved = _install_gate_spies(monkeypatch, runtime)
    requests: list[tuple[SpeakerShadowCandidateKey, str]] = []

    def request_rejection(
        candidate: SpeakerShadowCandidateKey,
        *,
        activation_generation: str,
    ) -> bool:
        requests.append((candidate, activation_generation))
        return True

    monkeypatch.setattr(
        runtime,
        "request_speaker_candidate_rejection",
        request_rejection,
    )
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-smart-turn",
        enforce=True,
    )
    shadow = factory()
    callback = shadow._on_observation
    assert callback is not None
    candidate = SpeakerShadowCandidateKey(8, 6, "smart_turn_turn")

    await callback(_observation(candidate, checkpoint_ms=1_500, similarity=0.20))
    await callback(_observation(candidate, checkpoint_ms=3_000, similarity=0.20))

    assert armed == []
    assert resolved == []
    assert requests == [(candidate, "activation-smart-turn")]
    await shadow.close()
    factory.close()
    profile.close()


@pytest.mark.parametrize("boundary", ["degraded", "close", "activation"])
async def test_gate_is_released_by_fail_open_boundary(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    runtime = _runtime()
    runtime._speaker_verifier_activation_generation = "activation-boundary"
    profile = _profile()
    armed, resolved = _install_gate_spies(monkeypatch, runtime)
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-boundary",
        enforce=True,
    )
    shadow = factory()
    callback = shadow._on_observation
    assert callback is not None
    candidate = SpeakerShadowCandidateKey(6, 4, "provider_candidate")
    await callback(_observation(candidate, checkpoint_ms=1_500, similarity=0.20))

    if boundary == "degraded":
        shadow._mark_backend_degraded()
    elif boundary == "close":
        factory.close()
    else:
        runtime._speaker_verifier_activation_generation = "activation-replacement"
        await callback(
            _observation(candidate, checkpoint_ms=3_000, similarity=0.20)
        )

    assert armed == [(candidate, "activation-boundary")]
    assert resolved == [(candidate, "activation-boundary", False)]
    await shadow.close()
    factory.close()
    profile.close()


async def test_factory_profile_and_backend_close_wipe_owned_biometric_material() -> None:
    runtime = _runtime()
    profile = _profile()
    source_embedding = profile._reference._embedding
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-8",
        enforce=True,
    )
    factory_profile = factory._profile
    factory_profile_embedding = factory_profile._reference._embedding

    profile.close()
    assert profile.closed is True
    assert not np.any(source_embedding)
    assert np.any(factory_profile_embedding)

    shadow = factory()
    backend_factory = shadow._backend_factory
    assert backend_factory is not None
    backend_storage = backend_factory._reference._storage
    assert any(backend_storage)

    factory.close()
    factory.close()
    assert factory_profile.closed is True
    assert not np.any(factory_profile_embedding)
    with pytest.raises(RuntimeError, match="factory is closed"):
        factory()

    await shadow.close()
    assert not any(backend_storage)


async def test_backend_health_marks_runtime_degraded() -> None:
    runtime = _runtime()
    profile = _profile()
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-8",
        enforce=True,
    )
    shadow = factory()
    runtime._speaker_verifier_activation_generation = "activation-8"

    shadow._mark_backend_degraded()

    assert runtime._speaker_verifier_degraded
    await shadow.close()
    factory.close()
    profile.close()


async def test_backend_health_recovery_clears_runtime_degraded() -> None:
    runtime = _runtime()
    profile = _profile()
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-8",
        enforce=True,
    )
    shadow = factory()
    runtime._speaker_verifier_activation_generation = "activation-8"

    shadow._mark_backend_degraded()
    assert runtime._speaker_verifier_degraded
    shadow._mark_backend_recovered()

    assert not runtime._speaker_verifier_degraded
    await shadow.close()
    factory.close()
    profile.close()


async def test_backend_health_ignores_stale_activation_generation() -> None:
    runtime = _runtime()
    profile = _profile()
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-8",
        enforce=True,
    )
    shadow = factory()
    runtime._speaker_verifier_activation_generation = "activation-9"

    shadow._mark_backend_degraded()
    assert not runtime._speaker_verifier_degraded

    runtime._speaker_verifier_degraded = True
    shadow._mark_backend_recovered()
    assert runtime._speaker_verifier_degraded

    await shadow.close()
    factory.close()
    profile.close()


def test_wrong_model_identity_raises_and_wipes_temporary_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_identity = SpeakerModelIdentity(
        "wrong-model",
        "wrong-revision",
        CAMPPLUS_EMBEDDING_DIM,
    )
    profile = _profile(wrong_identity)
    runtime = _runtime()
    captured_references: list[SpeakerReference] = []
    captured_embeddings: list[np.ndarray] = []
    original_clone: Callable[[SpeakerProfile], SpeakerReference] = (
        SpeakerProfile.clone_reference
    )

    def capture_clone(owner: SpeakerProfile) -> SpeakerReference:
        reference = original_clone(owner)
        captured_references.append(reference)
        captured_embeddings.append(reference._embedding)
        return reference

    monkeypatch.setattr(SpeakerProfile, "clone_reference", capture_clone)
    factory = OwnerVoiceAsrCompositionFactory(
        runtime,
        profile,
        activation_generation="activation-9",
        enforce=True,
    )

    with pytest.raises(ValueError, match=r"model identity does not match CAM\+\+"):
        factory()

    assert len(captured_references) == 1
    assert captured_references[0].closed is True
    assert not np.any(captured_embeddings[0])
    factory.close()
    profile.close()
