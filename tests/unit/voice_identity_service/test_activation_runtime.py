from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from main_logic.voice_identity_service.activation_runtime import (
    VoiceSessionActivationRuntime,
    VoiceSessionActivationRuntimeConfig,
)
from main_logic.voice_identity_service.activation_scoring import (
    ActivationScoreResult,
    ActivationScoreStatus,
)
from main_logic.voice_input.activation import (
    ActivationGeneration,
    ActivationState,
    AudioFrame,
    OutputCommit,
    VoiceActivationController,
)


def _generation() -> ActivationGeneration:
    return ActivationGeneration("session", 1, 1, 1, 1, "core_chat")


def _frame(sequence: int, *, captured_at: float | None = None) -> AudioFrame:
    samples = 1_600
    return AudioFrame(
        sequence=sequence,
        sample_start=sequence * samples,
        sample_end=(sequence + 1) * samples,
        captured_at=sequence * 0.1 if captured_at is None else captured_at,
        sample_rate=16_000,
        pcm=bytes([sequence % 251]) * (samples * 2),
        generation=_generation(),
        context={"sequence": sequence},
    )


class _Scorer:
    profile_generation = "profile"
    scorer_generation = 1

    def __init__(self, similarity: float = 0.75) -> None:
        self.similarity = similarity
        self.closed = False
        self.calls = 0

    async def prepare(self) -> ActivationScoreStatus:
        return ActivationScoreStatus.READY

    async def score(self, identity, pcm16: bytes, *, sample_rate_hz: int):
        assert pcm16
        assert sample_rate_hz == 16_000
        self.calls += 1
        return ActivationScoreResult(
            identity,
            ActivationScoreStatus.READY,
            self.similarity,
        )

    async def close(self) -> None:
        self.closed = True


async def _settle(runtime: VoiceSessionActivationRuntime) -> None:
    for _ in range(20):
        await asyncio.sleep(0)
        if runtime.state in {ActivationState.ACTIVE, ActivationState.UNAVAILABLE}:
            return


@pytest.mark.asyncio
async def test_owner_activation_replays_then_forwards_live_without_rescoring() -> None:
    sent: list[AudioFrame] = []

    async def output(frame: AudioFrame) -> OutputCommit:
        sent.append(frame)
        return OutputCommit.TRANSPORT_WRITTEN

    scorer = _Scorer()
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    assert (await runtime.prepare()).state is ActivationState.WAITING
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await _settle(runtime)
    assert runtime.state is ActivationState.ACTIVE
    assert [frame.sequence for frame in sent] == list(range(15))
    assert [frame.context for frame in sent] == [
        {"sequence": sequence} for sequence in range(15)
    ]
    assert scorer.calls == 1

    await runtime.feed(_frame(15), voice_activity=True)
    await asyncio.sleep(0)
    assert sent[-1].sequence == 15
    assert scorer.calls == 1
    await runtime.close()
    assert scorer.closed is True


@pytest.mark.asyncio
async def test_new_input_retries_once_after_inflight_output_was_not_sent() -> None:
    first_attempt_started = asyncio.Event()
    release_first_attempt = asyncio.Event()
    delivered: list[int] = []
    attempts = 0

    async def output(frame: AudioFrame) -> OutputCommit:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_attempt_started.set()
            await release_first_attempt.wait()
            return OutputCommit.NOT_SENT
        delivered.append(frame.sequence)
        return OutputCommit.TRANSPORT_WRITTEN

    runtime = VoiceSessionActivationRuntime(
        _generation(),
        _Scorer(),  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await asyncio.wait_for(first_attempt_started.wait(), timeout=1)

    await runtime.feed(_frame(15), voice_activity=True)
    release_first_attempt.set()
    await _settle(runtime)

    assert runtime.state is ActivationState.ACTIVE
    assert delivered == list(range(16))
    assert attempts == 17
    await runtime.close()


@pytest.mark.asyncio
async def test_new_input_never_retries_an_unknown_output_commit() -> None:
    first_attempt_started = asyncio.Event()
    release_first_attempt = asyncio.Event()
    attempts = 0

    async def output(_frame: AudioFrame) -> OutputCommit:
        nonlocal attempts
        attempts += 1
        first_attempt_started.set()
        await release_first_attempt.wait()
        return OutputCommit.UNKNOWN

    runtime = VoiceSessionActivationRuntime(
        _generation(),
        _Scorer(),  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await asyncio.wait_for(first_attempt_started.wait(), timeout=1)

    await runtime.feed(_frame(15), voice_activity=True)
    release_first_attempt.set()
    for _ in range(20):
        await asyncio.sleep(0)

    assert runtime.state is ActivationState.UNAVAILABLE
    assert attempts == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_close_is_bounded_when_output_swallows_cancellation() -> None:
    output_started = asyncio.Event()
    release_output = asyncio.Event()

    async def output(_frame: AudioFrame) -> OutputCommit:
        output_started.set()
        while not release_output.is_set():
            try:
                await release_output.wait()
            except asyncio.CancelledError:
                continue
        return OutputCommit.NOT_SENT

    runtime = VoiceSessionActivationRuntime(
        _generation(),
        _Scorer(),  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
        config=VoiceSessionActivationRuntimeConfig(shutdown_timeout_seconds=0.01),
    )
    await runtime.prepare()
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await asyncio.wait_for(output_started.wait(), timeout=1)

    try:
        await asyncio.wait_for(runtime.close(), timeout=0.2)
        assert runtime.state is ActivationState.CLOSED
    finally:
        release_output.set()
        output_task = runtime._output_task
        if output_task is not None:
            await asyncio.wait_for(output_task, timeout=1)


@pytest.mark.asyncio
async def test_short_waiting_audio_does_not_leave_the_process() -> None:
    sent: list[AudioFrame] = []

    async def output(frame: AudioFrame) -> OutputCommit:
        sent.append(frame)
        return OutputCommit.TRANSPORT_WRITTEN

    scorer = _Scorer()
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()
    for sequence in range(5):
        await runtime.feed(_frame(sequence), voice_activity=True)
    for sequence in range(5, 11):
        await runtime.feed(_frame(sequence), voice_activity=False)
    await asyncio.sleep(0)
    assert runtime.state is ActivationState.WAITING
    assert sent == []
    assert scorer.calls == 0
    await runtime.close()


@pytest.mark.asyncio
async def test_preparing_accumulates_voice_without_starting_verification() -> None:
    scorer = _Scorer()
    output = AsyncMock(return_value=OutputCommit.TRANSPORT_WRITTEN)
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
    )

    for sequence in range(15):
        decision = await runtime.feed(_frame(sequence), voice_activity=True)

    assert decision.state is ActivationState.PREPARING
    assert scorer.calls == 0
    output.assert_not_awaited()
    await runtime.close()


@pytest.mark.asyncio
async def test_sub_threshold_silence_cannot_pad_a_verification_checkpoint() -> None:
    sent: list[AudioFrame] = []

    async def output(frame: AudioFrame) -> OutputCommit:
        sent.append(frame)
        return OutputCommit.TRANSPORT_WRITTEN

    scorer = _Scorer()
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 3.0),
    )
    await runtime.prepare()

    sequence = 0
    for _ in range(3):
        for _ in range(4):
            await runtime.feed(_frame(sequence), voice_activity=True)
            sequence += 1
        for _ in range(4):
            await runtime.feed(_frame(sequence), voice_activity=False)
            sequence += 1
    await asyncio.sleep(0)

    assert sequence == 24
    assert scorer.calls == 0
    assert runtime.state is ActivationState.WAITING
    assert sent == []

    for _ in range(3):
        await runtime.feed(_frame(sequence), voice_activity=True)
        sequence += 1
    await _settle(runtime)
    assert scorer.calls == 1
    assert runtime.state is ActivationState.ACTIVE
    await runtime.close()


@pytest.mark.asyncio
async def test_half_second_continuous_silence_clears_accumulated_voice() -> None:
    scorer = _Scorer()
    output = AsyncMock(return_value=OutputCommit.TRANSPORT_WRITTEN)
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 3.0),
    )
    await runtime.prepare()

    for sequence in range(10):
        await runtime.feed(_frame(sequence), voice_activity=True)
    for sequence in range(10, 15):
        await runtime.feed(_frame(sequence), voice_activity=False)
    for sequence in range(15, 20):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await asyncio.sleep(0)

    assert scorer.calls == 0
    assert runtime.state is ActivationState.WAITING
    output.assert_not_awaited()
    await runtime.close()


@pytest.mark.asyncio
async def test_capture_time_silence_gap_resets_before_next_voiced_frame() -> None:
    scorer = _Scorer()
    output = AsyncMock(return_value=OutputCommit.TRANSPORT_WRITTEN)
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 3.0),
    )
    await runtime.prepare()

    for sequence in range(10):
        await runtime.feed(_frame(sequence), voice_activity=True)
    for sequence in range(10, 15):
        await runtime.feed(
            _frame(sequence, captured_at=1.5 + (sequence - 10) * 0.1),
            voice_activity=True,
        )
    await asyncio.sleep(0)

    assert scorer.calls == 0
    assert runtime.state is ActivationState.WAITING
    output.assert_not_awaited()
    await runtime.close()


@pytest.mark.asyncio
async def test_duplicate_voiced_frame_is_not_counted_twice() -> None:
    scorer = _Scorer()
    output = AsyncMock(return_value=OutputCommit.TRANSPORT_WRITTEN)
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()

    for sequence in range(14):
        await runtime.feed(_frame(sequence), voice_activity=True)
    for _ in range(5):
        duplicate = await runtime.feed(_frame(13), voice_activity=True)
        assert duplicate.reason == "duplicate_or_stale_frame"
    await asyncio.sleep(0)

    assert scorer.calls == 0
    await runtime.feed(_frame(14), voice_activity=True)
    await _settle(runtime)
    assert scorer.calls == 1
    assert runtime.state is ActivationState.ACTIVE
    await runtime.close()


@pytest.mark.asyncio
async def test_low_score_does_not_activate() -> None:
    sent: list[AudioFrame] = []

    async def output(frame: AudioFrame) -> OutputCommit:
        sent.append(frame)
        return OutputCommit.TRANSPORT_WRITTEN

    scorer = _Scorer(similarity=0.2)
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await _settle(runtime)
    assert runtime.state is ActivationState.WAITING
    assert sent == []
    assert scorer.calls == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_idle_timeout_returns_to_waiting_without_retracting_output() -> None:
    sent: list[AudioFrame] = []

    async def output(frame: AudioFrame) -> OutputCommit:
        sent.append(frame)
        return OutputCommit.TRANSPORT_WRITTEN

    scorer = _Scorer()
    runtime = VoiceSessionActivationRuntime(
        _generation(),
        scorer,  # type: ignore[arg-type]
        output,
        controller=VoiceActivationController(clock=lambda: 1.5),
    )
    await runtime.prepare()
    for sequence in range(15):
        await runtime.feed(_frame(sequence), voice_activity=True)
    await _settle(runtime)
    assert runtime.state is ActivationState.ACTIVE
    committed = len(sent)
    decision = await runtime.tick(now=31.6)
    assert decision.state is ActivationState.WAITING
    assert len(sent) == committed
    await runtime.close()
