from __future__ import annotations

import asyncio

import pytest

from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from main_logic.voice_identity_service.enrollment_audio import (
    OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES,
    EnrollmentAudioNormalizationError,
    EnrollmentAudioNormalizer,
)


class _Pipeline:
    def __init__(
        self,
        *,
        rnnoise_available: bool = True,
        fail: bool = False,
        block: asyncio.Event | None = None,
    ) -> None:
        self.rnnoise_available = rnnoise_available
        self.fail = fail
        self.block = block
        self.chunks: list[bytes] = []
        self.closed = False

    async def process(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
    ) -> ProcessedVoiceFrame:
        assert sample_rate_hz == 48_000
        self.chunks.append(pcm16)
        if self.block is not None:
            await self.block.wait()
        if self.fail:
            raise RuntimeError("processor failed")
        # The real 48 -> 16 kHz pipeline emits one third as many samples.
        output = bytes(len(pcm16) // 3)
        return ProcessedVoiceFrame(
            output,
            16_000,
            0.9 if self.rnnoise_available else None,
            self.rnnoise_available,
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uses_runtime_chunking_trims_exactly_and_isolates_segments() -> None:
    pipelines: list[_Pipeline] = []

    def factory() -> _Pipeline:
        pipeline = _Pipeline()
        pipelines.append(pipeline)
        return pipeline

    normalizer = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=factory,
    )
    raw = bytes(48_000 * 31 // 10 * 2)

    first = await normalizer.normalize(
        raw,
        sample_rate_hz=48_000,
        target_samples=48_000,
    )
    second = await normalizer.normalize(
        raw,
        sample_rate_hz=48_000,
        target_samples=48_000,
    )

    assert len(first) == len(second) == 48_000 * 2
    assert len(pipelines) == 2
    assert pipelines[0] is not pipelines[1]
    assert all(pipeline.closed for pipeline in pipelines)
    assert all(
        len(chunk) == OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES
        for chunk in pipelines[0].chunks
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pcm16", "sample_rate_hz"),
    [
        (b"\x00", 48_000),
        (b"\x00\x00", 48_000),
        (bytes(OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES), 44_100),
        (bytearray(OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES), 48_000),
    ],
)
async def test_rejects_non_contract_source_pcm(
    pcm16: bytes,
    sample_rate_hz: int,
) -> None:
    normalizer = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=_Pipeline,
    )

    with pytest.raises(EnrollmentAudioNormalizationError, match="invalid_pcm"):
        await normalizer.normalize(
            pcm16,  # type: ignore[arg-type]
            sample_rate_hz=sample_rate_hz,
            target_samples=16_000,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_requires_rnnoise_only_when_contract_enables_it() -> None:
    enabled_pipeline = _Pipeline(rnnoise_available=False)
    enabled = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=lambda: enabled_pipeline,
    )
    disabled_pipeline = _Pipeline(rnnoise_available=False)
    disabled = EnrollmentAudioNormalizer(
        nr_enabled=False,
        pipeline_factory=lambda: disabled_pipeline,
    )
    raw = bytes(48_000 * 2)

    with pytest.raises(
        EnrollmentAudioNormalizationError,
        match="audio_processing_unavailable",
    ):
        await enabled.normalize(
            raw,
            sample_rate_hz=48_000,
            target_samples=16_000,
        )
    normalized = await disabled.normalize(
        raw,
        sample_rate_hz=48_000,
        target_samples=16_000,
    )

    assert len(normalized) == 32_000
    assert enabled_pipeline.closed
    assert disabled_pipeline.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failure_and_cancellation_close_the_pipeline() -> None:
    failed_pipeline = _Pipeline(fail=True)
    failed = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=lambda: failed_pipeline,
    )
    with pytest.raises(
        EnrollmentAudioNormalizationError,
        match="audio_processing_unavailable",
    ):
        await failed.normalize(
            bytes(OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES),
            sample_rate_hz=48_000,
            target_samples=1,
        )
    assert failed_pipeline.closed

    block = asyncio.Event()
    cancelled_pipeline = _Pipeline(block=block)
    cancelled = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=lambda: cancelled_pipeline,
    )
    task = asyncio.create_task(
        cancelled.normalize(
            bytes(OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES),
            sample_rate_hz=48_000,
            target_samples=1,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_pipeline.closed


@pytest.mark.unit
@pytest.mark.asyncio
async def test_short_normalized_output_is_not_misreported_as_speaker_low() -> None:
    pipeline = _Pipeline()
    normalizer = EnrollmentAudioNormalizer(
        nr_enabled=True,
        pipeline_factory=lambda: pipeline,
    )

    with pytest.raises(
        EnrollmentAudioNormalizationError,
        match="speech_too_short",
    ):
        await normalizer.normalize(
            bytes(OWNER_CAMPPLUS_RUNTIME_CHUNK_BYTES),
            sample_rate_hz=48_000,
            target_samples=16_000,
        )
    assert pipeline.closed
