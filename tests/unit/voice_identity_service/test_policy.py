import math

import pytest

from main_logic.asr_client.speaker_shadow.contracts import SpeakerShadowCandidateKey
from main_logic.voice_identity_service.policy import (
    OwnerVoiceDecision,
    OwnerVoicePolicy,
)


def _candidate(generation: int = 1) -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(2, generation, "provider_candidate")


def test_policy_requires_two_strictly_low_checkpoints_to_reject() -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()

    first = policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.39,
        enforce=True,
    )
    second = policy.observe(
        candidate=candidate,
        checkpoint_ms=3_000,
        similarity=0.39,
        enforce=True,
    )

    assert first.decision is OwnerVoiceDecision.FORWARD
    assert second.decision is OwnerVoiceDecision.REJECT
    assert policy.pending_candidate_count == 0


@pytest.mark.parametrize("audio_ms", [1_501, 2_999])
def test_policy_accepts_completion_confirmation_as_second_low_evidence(
    audio_ms: int,
) -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
    )

    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
        observation_kind="completion_confirmation",
        audio_ms=audio_ms,
    )

    assert result.decision is OwnerVoiceDecision.REJECT
    assert result.reason == "stable_clear_mismatch"
    assert policy.pending_candidate_count == 0


@pytest.mark.parametrize(
    ("similarity", "enforce", "expected_reason"),
    [(0.40, True, "owner_or_uncertain"), (0.20, False, "shadow_only")],
)
def test_policy_completion_confirmation_preserves_fail_open_modes(
    similarity: float,
    enforce: bool,
    expected_reason: str,
) -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=enforce,
    )

    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=similarity,
        enforce=enforce,
        observation_kind="completion_confirmation",
        audio_ms=2_999,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD
    assert result.reason == expected_reason
    assert policy.pending_candidate_count == 0


@pytest.mark.parametrize(
    ("checkpoint_ms", "audio_ms"),
    [(1_500, None), (1_500, 1_500), (1_500, 3_000), (3_000, 2_000)],
)
def test_policy_malformed_completion_confirmation_fails_open_and_clears_state(
    checkpoint_ms: int,
    audio_ms: int | None,
) -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
    )

    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=checkpoint_ms,
        similarity=0.20,
        enforce=True,
        observation_kind="completion_confirmation",
        audio_ms=audio_ms,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD
    assert result.reason == "invalid_observation"
    assert policy.pending_candidate_count == 0


def test_policy_duplicate_or_out_of_order_observation_fails_open() -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()

    out_of_order = policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
        observation_kind="completion_confirmation",
        audio_ms=2_000,
    )
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
    )
    duplicate = policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
    )

    assert out_of_order.reason == "invalid_observation"
    assert duplicate.reason == "invalid_observation"
    assert policy.pending_candidate_count == 0


@pytest.mark.parametrize(
    ("observation_kind", "checkpoint_ms"),
    [([], 1_500), ("completion_confirmation", 1_500.0)],
)
def test_policy_invalid_completion_types_fail_open_and_clear_state(
    observation_kind: object,
    checkpoint_ms: object,
) -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=0.20,
        enforce=True,
    )

    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=checkpoint_ms,  # type: ignore[arg-type]
        similarity=0.20,
        enforce=True,
        observation_kind=observation_kind,  # type: ignore[arg-type]
        audio_ms=2_000,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD
    assert result.reason == "invalid_observation"
    assert policy.pending_candidate_count == 0


@pytest.mark.parametrize(
    "first,second",
    [(0.40, 0.39), (0.39, 0.40), (0.9, 0.1), (0.1, 0.9)],
)
def test_policy_forwards_owner_or_uncertain_observations(
    first: float,
    second: float,
) -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()

    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=first,
        enforce=True,
    )
    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=3_000,
        similarity=second,
        enforce=True,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD


def test_policy_shadow_mode_never_rejects() -> None:
    policy = OwnerVoicePolicy()
    candidate = _candidate()
    policy.observe(
        candidate=candidate,
        checkpoint_ms=1_500,
        similarity=-0.5,
        enforce=False,
    )

    result = policy.observe(
        candidate=candidate,
        checkpoint_ms=3_000,
        similarity=-0.5,
        enforce=False,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD
    assert result.reason == "shadow_only"


@pytest.mark.parametrize(
    "checkpoint,similarity",
    [(None, 0.1), (2_000, 0.1), (1_500, math.nan), (1_500, math.inf)],
)
def test_policy_invalid_or_missing_observation_fails_open(
    checkpoint: int | None,
    similarity: float,
) -> None:
    result = OwnerVoicePolicy().observe(
        candidate=_candidate(),
        checkpoint_ms=checkpoint,
        similarity=similarity,
        enforce=True,
    )

    assert result.decision is OwnerVoiceDecision.FORWARD
    assert result.reason == "invalid_observation"


def test_policy_bounds_and_can_forget_candidate_state() -> None:
    policy = OwnerVoicePolicy(candidate_capacity=2)
    candidates = [_candidate(index) for index in range(3)]
    for candidate in candidates:
        policy.observe(
            candidate=candidate,
            checkpoint_ms=1_500,
            similarity=0.1,
            enforce=True,
        )

    assert policy.pending_candidate_count == 2
    policy.forget(candidates[-1])
    assert policy.pending_candidate_count == 1
    policy.reset()
    assert policy.pending_candidate_count == 0
