from __future__ import annotations

import pytest

from main_logic.asr_client.speaker_shadow.contracts import (
    MAX_SPEAKER_SHADOW_BUFFERED_CANDIDATES,
    MAX_SPEAKER_SHADOW_CALLBACK_TIMEOUT_SECONDS,
    MAX_SPEAKER_SHADOW_CANDIDATE_PCM_BYTES,
    MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES,
    MAX_SPEAKER_SHADOW_FINALIZED_CANDIDATES,
    MAX_SPEAKER_SHADOW_QUEUE_CAPACITY,
    MAX_SPEAKER_SHADOW_RETAINED_PCM_BYTES,
    MAX_SPEAKER_SHADOW_SHUTDOWN_GRACE_SECONDS,
    MAX_SPEAKER_SHADOW_THRESHOLDS,
    SpeakerShadowCandidateKey,
    SpeakerShadowConfig,
    SpeakerShadowObservation,
)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"similarity_thresholds": ()}, "similarity_thresholds"),
        ({"similarity_thresholds": (0.5, 0.4)}, "similarity_thresholds"),
        ({"minimum_audio_ms": 0}, "minimum_audio_ms"),
        (
            {"minimum_audio_ms": 2_000, "maximum_audio_ms": 1_000},
            "maximum_audio_ms",
        ),
        ({"maximum_audio_ms": 4_001}, "maximum_audio_ms"),
        ({"observation_checkpoints_ms": ()}, "observation_checkpoints_ms"),
        (
            {"observation_checkpoints_ms": (1_500, 1_500)},
            "observation_checkpoints_ms",
        ),
        (
            {"observation_checkpoints_ms": (1_499, 3_000)},
            "observation_checkpoints_ms",
        ),
        (
            {"observation_checkpoints_ms": (1_500, 4_001)},
            "observation_checkpoints_ms",
        ),
        ({"queue_capacity": 0}, "queue_capacity"),
        (
            {"queue_capacity": MAX_SPEAKER_SHADOW_QUEUE_CAPACITY + 1},
            "queue_capacity",
        ),
        ({"buffered_candidate_capacity": 0}, "buffered_candidate_capacity"),
        (
            {
                "buffered_candidate_capacity": (
                    MAX_SPEAKER_SHADOW_BUFFERED_CANDIDATES + 1
                )
            },
            "buffered_candidate_capacity",
        ),
        (
            {"queue_capacity": 4, "finalized_candidate_capacity": 3},
            "finalized_candidate_capacity",
        ),
        (
            {
                "finalized_candidate_capacity": (
                    MAX_SPEAKER_SHADOW_FINALIZED_CANDIDATES + 1
                )
            },
            "finalized_candidate_capacity",
        ),
        (
            {
                "load_retry_initial_seconds": 2.0,
                "load_retry_max_seconds": 1.0,
            },
            "load_retry_max_seconds",
        ),
        ({"shutdown_grace_seconds": 0.0}, "shutdown_grace_seconds"),
        (
            {
                "shutdown_grace_seconds": (
                    MAX_SPEAKER_SHADOW_SHUTDOWN_GRACE_SECONDS + 0.1
                )
            },
            "shutdown_grace_seconds",
        ),
        ({"callback_timeout_seconds": 0.0}, "callback_timeout_seconds"),
        (
            {
                "callback_timeout_seconds": (
                    MAX_SPEAKER_SHADOW_CALLBACK_TIMEOUT_SECONDS + 0.1
                )
            },
            "callback_timeout_seconds",
        ),
        ({"backend_load_timeout_seconds": float("inf")}, "backend_load"),
        ({"backend_score_timeout_seconds": 0.0}, "backend_score"),
        ({"backend_close_timeout_seconds": 3.0}, "backend_close"),
        ({"process_terminate_timeout_seconds": 3.0}, "process_terminate"),
        (
            {
                "similarity_thresholds": tuple(
                    index / (MAX_SPEAKER_SHADOW_THRESHOLDS + 1)
                    for index in range(MAX_SPEAKER_SHADOW_THRESHOLDS + 1)
                )
            },
            "similarity_thresholds",
        ),
    ],
)
def test_config_rejects_unsafe_resource_bounds(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SpeakerShadowConfig(**overrides)


def test_config_is_default_off_and_caps_candidate_audio() -> None:
    config = SpeakerShadowConfig()

    assert config.enabled is False
    assert config.maximum_audio_ms == 4_000
    assert config.observation_checkpoints_ms is None
    assert config.completion_confirmation_scopes == ()
    assert MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES == 128_000
    assert MAX_SPEAKER_SHADOW_CANDIDATE_PCM_BYTES == 128_000
    assert MAX_SPEAKER_SHADOW_RETAINED_PCM_BYTES < 8 * 1024 * 1024


def test_config_accepts_provider_completion_confirmation_with_two_checkpoints() -> None:
    config = SpeakerShadowConfig(
        observation_checkpoints_ms=(1_500, 3_000),
        completion_confirmation_scopes=("provider_candidate",),
    )

    assert config.completion_confirmation_scopes == ("provider_candidate",)


def test_config_preserves_legacy_positional_argument_order() -> None:
    config = SpeakerShadowConfig(False, (0.40,), 1_500, 4_000, None, 60.0)

    assert config.idle_unload_seconds == 60.0
    assert config.completion_confirmation_scopes == ()


@pytest.mark.parametrize(
    "completion_confirmation_scopes",
    [
        ["provider_candidate"],
        ("unsupported",),
        ("provider_candidate", "provider_candidate"),
    ],
)
def test_config_rejects_invalid_completion_confirmation_scopes(
    completion_confirmation_scopes: object,
) -> None:
    with pytest.raises(ValueError, match="completion_confirmation_scopes"):
        SpeakerShadowConfig(
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=completion_confirmation_scopes,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "observation_checkpoints_ms",
    [None, (1_500,)],
)
def test_completion_confirmation_requires_two_explicit_checkpoints(
    observation_checkpoints_ms: tuple[int, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="at least two explicit"):
        SpeakerShadowConfig(
            observation_checkpoints_ms=observation_checkpoints_ms,
            completion_confirmation_scopes=("provider_candidate",),
        )


def test_observation_kind_defaults_to_checkpoint_and_accepts_confirmation() -> None:
    candidate = SpeakerShadowCandidateKey(1, 2, "provider_candidate")
    observation = SpeakerShadowObservation(
        candidate=candidate,
        similarity=0.2,
        would_block=((0.4, True),),
        audio_ms=1_500,
        checkpoint_ms=1_500,
    )

    assert observation.observation_kind == "checkpoint"
    assert (
        SpeakerShadowObservation(
            candidate=candidate,
            similarity=0.2,
            would_block=((0.4, True),),
            audio_ms=2_999,
            checkpoint_ms=1_500,
            observation_kind="completion_confirmation",
        ).observation_kind
        == "completion_confirmation"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (-1, 0, "provider_candidate"),
        (0, -1, "provider_candidate"),
        (True, 0, "provider_candidate"),
        (0, 0, "unsupported"),
    ],
)
def test_candidate_key_rejects_identity_shapes_outside_fixed_contract(
    arguments: tuple[object, object, object],
) -> None:
    with pytest.raises(ValueError):
        SpeakerShadowCandidateKey(*arguments)  # type: ignore[arg-type]
