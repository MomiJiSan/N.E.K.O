from __future__ import annotations

import numpy as np
import pytest

from main_logic.voice_identity_service.enrollment import (
    ENROLLMENT_MAXIMUM_PCM_BYTES,
    ENROLLMENT_MINIMUM_PCM_BYTES,
    EnrollmentAudioError,
    validate_enrollment_pcm16,
)


def _pcm(milliseconds: int, *, amplitude: int = 2_000) -> bytes:
    sample_count = 16_000 * milliseconds // 1_000
    samples = np.full(sample_count, amplitude, dtype="<i2")
    return samples.tobytes()


def test_accepts_four_seconds_of_usable_pcm() -> None:
    validate_enrollment_pcm16(_pcm(4_000))


@pytest.mark.parametrize("milliseconds", [1_500, 4_000, 8_000])
def test_accepts_supported_variable_recording_lengths(milliseconds: int) -> None:
    validate_enrollment_pcm16(_pcm(milliseconds))


def test_accepts_pause_when_total_audio_is_under_maximum() -> None:
    parts = np.concatenate(
        [
            np.full(2 * 16_000, 2_000, dtype="<i2"),
            np.zeros(2 * 16_000, dtype="<i2"),
            np.full(2 * 16_000, 2_000, dtype="<i2"),
        ]
    )

    validate_enrollment_pcm16(parts.tobytes())


@pytest.mark.parametrize(
    ("pcm16", "code"),
    [
        pytest.param(b"\x00", "invalid_pcm", id="odd-byte-count"),
        pytest.param(_pcm(1_499), "speech_too_short", id="too-short"),
        pytest.param(
            b"\x00\x00" * 32_000,
            "speech_too_short",
            id="silence",
        ),
        pytest.param(_pcm(8_001), "audio_too_long", id="too-long"),
        pytest.param(
            _pcm(4_000, amplitude=32_767),
            "severe_clipping",
            id="clipped",
        ),
    ],
)
def test_rejects_unusable_pcm(pcm16: bytes, code: str) -> None:
    with pytest.raises(EnrollmentAudioError) as caught:
        validate_enrollment_pcm16(pcm16)

    assert caught.value.code == code


def test_payload_ceiling_matches_eight_second_pcm16() -> None:
    assert ENROLLMENT_MINIMUM_PCM_BYTES == 48_000
    assert ENROLLMENT_MAXIMUM_PCM_BYTES == 256_000
