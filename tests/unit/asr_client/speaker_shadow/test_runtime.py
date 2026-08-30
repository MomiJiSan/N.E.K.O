from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from main_logic.asr_client.speaker_shadow.contracts import (
    MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES,
    SPEAKER_SHADOW_SAMPLE_RATE_HZ,
    SpeakerShadowCandidateKey,
    SpeakerShadowCompletion,
    SpeakerShadowConfig,
    SpeakerShadowObservation,
)
from main_logic.asr_client.speaker_shadow.runtime import (
    SpeakerShadowRuntime,
    _AudioFrame,
    _BackendProcessHost,
    _CandidateActivated,
    _CandidateDeferred,
    _CandidateFinished,
    _CandidateToken,
    _backend_host_main,
)


@dataclass(slots=True)
class _BackendFactory:
    score_value: float = 0.9
    load_ok: bool = True
    load_error: bool = False
    score_error: bool = False
    score_error_after: int | None = None
    close_error: bool = False
    expected_pcm: bytes | None = None
    block_stage: str | None = None
    stage_started: Any = None
    stage_release: Any = None
    parent_close_calls: int = 0
    parent_profile: bytearray | None = None

    def __call__(self) -> _Backend:
        return _Backend(self)

    def close(self) -> None:
        self.parent_close_calls += 1
        if self.parent_profile is not None:
            self.parent_profile[:] = b"\x00" * len(self.parent_profile)


class _Backend:
    def __init__(self, settings: _BackendFactory) -> None:
        self._settings = settings
        self._score_calls = 0

    def _maybe_block(self, stage: str) -> None:
        settings = self._settings
        if settings.block_stage != stage:
            return
        if settings.stage_started is not None:
            settings.stage_started.set()
        if settings.stage_release is None:
            while True:
                time.sleep(60.0)
        settings.stage_release.wait()

    def load(self) -> bool:
        self._maybe_block("load")
        if self._settings.load_error:
            raise RuntimeError("load failed")
        return self._settings.load_ok

    def score(self, pcm16: bytes, sample_rate_hz: int) -> float:
        self._maybe_block("score")
        should_fail = self._settings.score_error or (
            self._settings.score_error_after is not None
            and self._score_calls >= self._settings.score_error_after
        )
        self._score_calls += 1
        if should_fail:
            raise RuntimeError("score failed")
        if self._settings.expected_pcm is not None:
            assert pcm16 == self._settings.expected_pcm
        assert sample_rate_hz == SPEAKER_SHADOW_SAMPLE_RATE_HZ
        return self._settings.score_value

    def close(self) -> None:
        self._maybe_block("close")
        if self._settings.close_error:
            raise RuntimeError("close failed")


def _pcm(duration_ms: int) -> bytes:
    return b"\x01\x00" * (
        SPEAKER_SHADOW_SAMPLE_RATE_HZ * duration_ms // 1_000
    )


def _candidate(
    generation: int,
    scope: str = "provider_candidate",
) -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(1, generation, scope)  # type: ignore[arg-type]


def _config(**overrides: object) -> SpeakerShadowConfig:
    values: dict[str, object] = {
        "enabled": True,
        "minimum_audio_ms": 20,
        "maximum_audio_ms": 100,
        "idle_unload_seconds": 60.0,
        "queue_capacity": 8,
        "buffered_candidate_capacity": 4,
        "finalized_candidate_capacity": 16,
        "shutdown_grace_seconds": 0.05,
        "callback_timeout_seconds": 0.05,
        "backend_load_timeout_seconds": 3.0,
        "backend_score_timeout_seconds": 1.0,
        "backend_close_timeout_seconds": 1.0,
        "process_terminate_timeout_seconds": 0.5,
    }
    values.update(overrides)
    return SpeakerShadowConfig(**values)


def _provider_gate_config(
    *,
    prewarm: bool = False,
    **overrides: object,
) -> SpeakerShadowConfig:
    values: dict[str, object] = {
        "minimum_audio_ms": 1_500,
        "maximum_audio_ms": 4_000,
        "observation_checkpoints_ms": (1_500, 3_000),
        "completion_confirmation_scopes": ("provider_candidate",),
        "pending_observation_gate_scopes": ("provider_candidate",),
        "backend_prewarm_scopes": (
            ("provider_candidate",) if prewarm else ()
        ),
    }
    values.update(overrides)
    return _config(**values)


def _spawn_event() -> Any:
    return multiprocessing.get_context("spawn").Event()


def _speaker_host_pids() -> set[int]:
    return {
        process.pid
        for process in multiprocessing.active_children()
        if process.pid is not None
        and process.name == "speaker-shadow-backend"
        and process.is_alive()
    }


async def test_deferred_candidate_buffers_then_scores_in_order_after_activation() -> (
    None
):
    observations: list[SpeakerShadowObservation] = []
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(9_001)

    assert runtime.defer_candidate(candidate)
    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    assert observations == []
    assert runtime.requires_provisional_decision(candidate) is False
    assert runtime.snapshot()["retained_pcm_bytes"] == len(_pcm(2_999))

    assert runtime.activate_candidate(candidate)
    assert runtime.requires_provisional_decision(candidate) is True
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert [item.observation_kind for item in observations] == [
        "checkpoint",
        "completion_confirmation",
    ]
    assert [item.audio_ms for item in observations] == [1_500, 2_999]
    assert completions == [
        SpeakerShadowCompletion(candidate, "scored", 1_500)
    ]
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_deferred_finish_before_activation_is_fail_open_and_wipes_pcm() -> None:
    observations: list[SpeakerShadowObservation] = []
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(9_002)

    assert runtime.defer_candidate(candidate)
    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    assert runtime.activate_candidate(candidate) is False
    await runtime.wait_idle()

    assert observations == []
    assert completions == [
        SpeakerShadowCompletion(candidate, "insufficient", None)
    ]
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_defer_must_precede_first_pcm_and_reset_wipes_buffer() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_provider_gate_config(),
    )
    ordinary_candidate = _candidate(9_003)
    deferred_candidate = _candidate(9_004)

    assert runtime.submit(
        _pcm(100),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=ordinary_candidate,
    )
    assert runtime.defer_candidate(ordinary_candidate) is False
    assert runtime.defer_candidate(deferred_candidate)
    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=deferred_candidate,
    )
    await runtime.wait_idle()
    assert runtime.snapshot()["retained_pcm_bytes"] > 0

    await runtime.reset()

    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_defer_requires_scope_enabled_for_pending_observation_gate() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=SpeakerShadowConfig(
            enabled=True,
            minimum_audio_ms=1_500,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
        ),
    )
    provider_candidate = _candidate(9_005)
    smart_turn_candidate = SpeakerShadowCandidateKey(
        detector_epoch=1,
        shadow_generation=9_006,
        scope="smart_turn_turn",
    )

    assert runtime.supports_deferred_candidate(provider_candidate) is False
    assert runtime.supports_deferred_candidate(smart_turn_candidate) is False
    assert runtime.defer_candidate(provider_candidate) is False
    assert runtime.defer_candidate(smart_turn_candidate) is False
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_deferred_first_frame_prewarms_without_scoring() -> None:
    score_started = _spawn_event()
    score_release = _spawn_event()
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            block_stage="score",
            stage_started=score_started,
            stage_release=score_release,
        ),
        config=_provider_gate_config(prewarm=True),
    )
    candidate = _candidate(9_005)

    assert runtime.defer_candidate(candidate)
    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    assert runtime.snapshot()["backend_loaded_count"] == 1
    assert score_started.is_set() is False
    assert runtime.requires_provisional_decision(candidate) is False
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_deferred_predeclarations_are_bounded_without_evicting_owner() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_provider_gate_config(buffered_candidate_capacity=1),
    )
    retained = _candidate(9_006)
    rejected = _candidate(9_007)

    assert runtime.defer_candidate(retained)
    assert runtime.defer_candidate(rejected) is False
    assert runtime.activate_candidate(retained)
    assert runtime.finish_candidate(retained)
    await runtime.wait_idle()

    assert runtime.snapshot()["finished_candidate_count"] == 1
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def _wait_until(predicate: Any, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.005)


class _InspectingHostConnection:
    def __init__(self) -> None:
        self._messages = iter(
            [
                ("load",),
                ("score", len(_pcm(10)), SPEAKER_SHADOW_SAMPLE_RATE_HZ),
                ("close",),
            ]
        )
        self.responses: list[tuple[object, ...]] = []
        self.score_pcm_cleared = False
        self.closed = False

    def recv(self) -> tuple[object, ...]:
        message = next(self._messages)
        if message[0] == "close":
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            retained_pcm = frame.f_back.f_locals.get("pcm16")
            self.score_pcm_cleared = retained_pcm is None or not any(retained_pcm)
        return message

    def send(self, response: tuple[object, ...]) -> None:
        self.responses.append(response)

    def close(self) -> None:
        self.closed = True


class _PollBlindProcess:
    """A live child whose only job is to keep the request path going."""

    pid = -1
    exitcode = None

    def is_alive(self) -> bool:
        return True

    # Reaping hooks, so a host that gives up on the response fails with the
    # timeout it actually hit instead of an AttributeError that hides it.
    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None

    def close(self) -> None:
        return None


class _PollBlindConnection:
    """A pipe end that answers a read but never admits readiness to ``poll``."""

    def __init__(self, response: tuple[object, ...]) -> None:
        self._response = response
        self.sent: list[tuple[object, ...]] = []
        self.polls = 0

    def poll(self, timeout: float = 0.0) -> bool:
        self.polls += 1
        return False

    def send(self, message: tuple[object, ...]) -> None:
        self.sent.append(message)

    def recv(self) -> tuple[object, ...]:
        return self._response

    def close(self) -> None:
        return None


async def test_host_response_survives_a_pipe_that_never_polls_ready() -> None:
    # Windows loses a response when the host waits on ``poll(0)``: each poll
    # starts an overlapped pipe read and cancels it in the same breath, and
    # that cancellation can swallow the very answer it asked about. The child
    # stays alive, the answer never comes back, and the request spins to a
    # false timeout — a live backend reported as "failed" roughly half the
    # time on a loaded runner. So the request path must not gate the read on
    # a readiness poll at all: a connection that answers ``recv`` but always
    # reports "not ready" has to complete the request anyway.
    host = _BackendProcessHost(
        factory=_BackendFactory(),
        terminate_timeout_seconds=0.1,
    )
    parent_connection = host._connection
    child_connection = host._child_connection
    assert parent_connection is not None and child_connection is not None
    parent_connection.close()
    child_connection.close()
    connection = _PollBlindConnection((True, True))
    host._connection = connection  # type: ignore[assignment]
    host._child_connection = None
    host._process = _PollBlindProcess()  # type: ignore[assignment]

    assert await host.load(timeout_seconds=1.0) is True
    assert connection.sent == [("load",)]


def test_backend_host_wipes_score_pcm_before_waiting_for_next_command() -> None:
    pcm16 = _pcm(10)
    pcm_buffer = bytearray(pcm16)
    connection = _InspectingHostConnection()

    _backend_host_main(  # type: ignore[arg-type]
        _BackendFactory(expected_pcm=pcm16),
        connection,
        pcm_buffer,
    )

    assert connection.score_pcm_cleared is True
    assert connection.closed is True


@pytest.mark.parametrize(
    ("config", "has_factory"),
    [
        (SpeakerShadowConfig(enabled=False), True),
        (SpeakerShadowConfig(enabled=True), False),
    ],
)
async def test_disabled_or_missing_factory_does_zero_work(
    config: SpeakerShadowConfig,
    has_factory: bool,
) -> None:
    before = _speaker_host_pids()
    factory = _BackendFactory() if has_factory else None
    runtime = SpeakerShadowRuntime(
        backend_factory=factory,
        config=config,
    )
    candidate = _candidate(1)

    assert runtime.enabled is False
    assert runtime.submit(
        _pcm(20),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False
    assert runtime.finish_candidate(candidate) is False
    await runtime.wait_idle()
    await runtime.close()

    metrics = runtime.snapshot()
    assert metrics["submitted_frame_count"] == 0
    assert metrics["queued_item_count"] == 0
    assert metrics["worker_task_count"] == 0
    assert metrics["cleanup_task_count"] == 0
    assert metrics["backend_loaded_count"] == 0
    assert metrics["backend_process_count"] == 0
    assert _speaker_host_pids() == before
    if factory is not None:
        assert factory.parent_close_calls == 1


async def test_ordered_finish_scores_once_and_rejects_late_pcm() -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    first = _pcm(10)
    second = _pcm(10)
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            score_value=0.2,
            expected_pcm=first + second,
        ),
        config=_config(),
        on_observation=observe,
    )
    candidate = _candidate(2)

    assert runtime.submit(
        first,
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.submit(
        second,
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert observations == [
        SpeakerShadowObservation(
            candidate=candidate,
            similarity=pytest.approx(0.2),
            would_block=(
                (0.40, True),
                (0.44, True),
                (0.48, True),
                (0.52, True),
                (0.55, True),
            ),
            audio_ms=20,
        )
    ]
    assert runtime.submit(
        first,
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False
    assert runtime.finish_candidate(candidate)
    metrics = runtime.snapshot()
    assert metrics["scored_candidate_count"] == 1
    assert metrics["finished_candidate_count"] == 1
    assert metrics["evaluated_candidate_count"] == 1
    await runtime.close()


async def test_explicit_checkpoints_emit_1500ms_and_3000ms_observations() -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
    )
    candidate = _candidate(47)

    assert runtime.submit(
        _pcm(3_000),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    assert [item.checkpoint_ms for item in observations] == [1_500, 3_000]
    assert [item.audio_ms for item in observations] == [1_500, 3_000]
    assert [item.candidate for item in observations] == [candidate, candidate]
    metrics = runtime.snapshot()
    assert metrics["scored_candidate_count"] == 1
    assert metrics["evaluated_candidate_count"] == 1
    assert metrics["would_block_at_0_4_count"] == 2
    assert metrics["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_3000ms_completion_follows_both_observations() -> None:
    events: list[tuple[str, int | None]] = []
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        events.append(("observation", observation.checkpoint_ms))

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)
        events.append(("completion", completion.last_checkpoint_ms))

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(147)

    assert runtime.submit(
        _pcm(3_000),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert events == [
        ("observation", 1_500),
        ("observation", 3_000),
        ("completion", 3_000),
    ]
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="scored",
            last_checkpoint_ms=3_000,
        )
    ]
    await runtime.close()


async def test_duplicate_finish_emits_completion_once() -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=20),
        on_completion=complete,
    )
    candidate = _candidate(148)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()
    assert runtime.finish_candidate(candidate)

    assert len(completions) == 1
    assert completions[0].candidate == candidate
    assert runtime.snapshot()["completion_count"] == 1
    await runtime.close()


async def test_score_failure_completion_is_ordered_after_prior_checkpoint() -> None:
    events: list[tuple[str, int | None, str | None]] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        events.append(("observation", observation.checkpoint_ms, None))

    async def complete(completion: SpeakerShadowCompletion) -> None:
        events.append(
            (
                "completion",
                completion.last_checkpoint_ms,
                completion.terminal_reason,
            )
        )

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_error_after=1),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(149)

    assert runtime.submit(
        _pcm(3_000),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert events == [
        ("observation", 1_500, None),
        ("completion", 1_500, "failed"),
    ]
    assert runtime.snapshot()["completion_count"] == 1
    await runtime.close()


async def test_short_candidate_emits_no_3000ms_confirmation() -> None:
    observations: list[SpeakerShadowObservation] = []
    completions: list[SpeakerShadowCompletion] = []
    event_order: list[str] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)
        event_order.append(f"observation:{observation.checkpoint_ms}")

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)
        event_order.append("completion")

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(48)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert [item.checkpoint_ms for item in observations] == [1_500]
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="scored",
            last_checkpoint_ms=1_500,
        )
    ]
    assert event_order == ["observation:1500", "completion"]
    metrics = runtime.snapshot()
    assert metrics["scored_candidate_count"] == 1
    assert metrics["insufficient_candidate_count"] == 0
    assert metrics["finished_candidate_count"] == 1
    assert metrics["evaluated_candidate_count"] == 1
    assert metrics["completion_count"] == 1
    assert metrics["completion_after_first_checkpoint_count"] == 1
    assert metrics["completion_before_first_checkpoint_count"] == 0
    assert metrics["retained_pcm_bytes"] == 0
    await runtime.close()


@pytest.mark.parametrize(
    ("duration_ms", "expected_kinds"),
    [
        (1_499, []),
        (1_500, ["checkpoint"]),
        (1_501, ["checkpoint", "completion_confirmation"]),
        (2_999, ["checkpoint", "completion_confirmation"]),
        (3_000, ["checkpoint", "checkpoint"]),
    ],
)
async def test_provider_completion_confirmation_respects_checkpoint_boundaries(
    duration_ms: int,
    expected_kinds: list[str],
) -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
        ),
        on_observation=observe,
    )
    candidate = _candidate(248 + duration_ms)

    assert runtime.submit(
        _pcm(duration_ms),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert [item.observation_kind for item in observations] == expected_kinds
    if "completion_confirmation" in expected_kinds:
        confirmation = observations[-1]
        assert confirmation.checkpoint_ms == 1_500
        assert confirmation.audio_ms == duration_ms
        assert confirmation.candidate == candidate
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_completion_confirmation_observation_precedes_single_completion() -> None:
    events: list[tuple[str, str | None, int | None]] = []
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        events.append(
            (
                "observation",
                observation.observation_kind,
                observation.checkpoint_ms,
            )
        )

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)
        events.append(("completion", None, completion.last_checkpoint_ms))

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(349)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert events == [
        ("observation", "checkpoint", 1_500),
        ("observation", "completion_confirmation", 1_500),
        ("completion", None, 1_500),
    ]
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="scored",
            last_checkpoint_ms=1_500,
        )
    ]
    assert runtime.snapshot()["completion_count"] == 1
    await runtime.close()


@pytest.mark.parametrize(
    ("scope", "score_value"),
    [
        ("provider_candidate", 0.9),
        ("smart_turn_turn", 0.2),
    ],
    ids=["first-owner", "non-provider"],
)
async def test_completion_confirmation_does_not_rescore_unarmed_candidate(
    scope: str,
    score_value: float,
) -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=score_value),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
        ),
        on_observation=observe,
    )
    candidate = _candidate(350, scope)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert [item.observation_kind for item in observations] == ["checkpoint"]
    assert [item.checkpoint_ms for item in observations] == [1_500]
    await runtime.close()


async def test_completion_confirmation_score_failure_still_completes_fail_open() -> None:
    events: list[tuple[str, str]] = []
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        events.append(("observation", observation.observation_kind))

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)
        events.append(("completion", completion.terminal_reason))

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2, score_error_after=1),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(351)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    assert events == [
        ("observation", "checkpoint"),
        ("completion", "failed"),
    ]
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="failed",
            last_checkpoint_ms=1_500,
        )
    ]
    metrics = runtime.snapshot()
    assert metrics["completion_count"] == 1
    assert metrics["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_uncooperative_confirmation_callback_cannot_suppress_completion() -> None:
    confirmation_started = asyncio.Event()
    release_confirmation = asyncio.Event()
    completion_delivered = asyncio.Event()
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        if observation.observation_kind != "completion_confirmation":
            return
        confirmation_started.set()
        while not release_confirmation.is_set():
            try:
                await release_confirmation.wait()
            except asyncio.CancelledError:
                continue

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)
        completion_delivered.set()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
            callback_timeout_seconds=0.01,
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(353)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await asyncio.wait_for(confirmation_started.wait(), 2.0)
    await asyncio.wait_for(completion_delivered.wait(), 1.0)

    release_confirmation.set()
    await runtime.wait_idle()
    await asyncio.sleep(0)

    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="scored",
            last_checkpoint_ms=1_500,
        )
    ]
    metrics = runtime.snapshot()
    assert metrics["completion_count"] == 1
    assert metrics["callback_failure_count"] == 1
    assert metrics["completion_callback_failure_count"] == 0
    assert metrics["retained_pcm_bytes"] == 0
    assert metrics["callback_task_count"] == 0
    await runtime.close()


async def test_close_tracks_and_reaps_detached_confirmation_callback() -> None:
    confirmation_started = asyncio.Event()
    completion_delivered = asyncio.Event()
    cancellation_count = 0

    async def observe(observation: SpeakerShadowObservation) -> None:
        nonlocal cancellation_count
        if observation.observation_kind != "completion_confirmation":
            return
        confirmation_started.set()
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_count += 1
                if cancellation_count >= 4:
                    raise

    async def complete(_completion: SpeakerShadowCompletion) -> None:
        completion_delivered.set()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
            callback_timeout_seconds=0.01,
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(354)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await asyncio.wait_for(confirmation_started.wait(), 2.0)
    await asyncio.wait_for(completion_delivered.wait(), 1.0)
    await runtime.wait_idle()

    assert cancellation_count == 2
    assert runtime.snapshot()["callback_task_count"] == 1

    await asyncio.wait_for(runtime.close(), 2.0)

    assert cancellation_count >= 4
    assert runtime.snapshot()["callback_task_count"] == 0


async def test_reset_invalidates_in_flight_completion_confirmation() -> None:
    confirmation_started = asyncio.Event()
    completions: list[SpeakerShadowCompletion] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        if observation.observation_kind == "completion_confirmation":
            confirmation_started.set()
            await asyncio.Event().wait()

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
            completion_confirmation_scopes=("provider_candidate",),
            callback_timeout_seconds=1.0,
        ),
        on_observation=observe,
        on_completion=complete,
    )
    candidate = _candidate(352)

    assert runtime.submit(
        _pcm(2_999),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await asyncio.wait_for(confirmation_started.wait(), 2.0)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False

    await runtime.reset()
    await runtime.wait_idle()

    assert completions == []
    metrics = runtime.snapshot()
    assert metrics["retained_pcm_bytes"] == 0
    assert metrics["buffered_candidate_count"] == 0
    await runtime.close()


async def test_provisional_decision_tracks_first_checkpoint_delivery() -> None:
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()

    async def observe(_observation: SpeakerShadowObservation) -> None:
        callback_started.set()
        await callback_release.wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(callback_timeout_seconds=1.0),
        on_observation=observe,
    )
    candidate = _candidate(353)

    assert runtime.submit(
        _pcm(1_499),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.requires_provisional_decision(candidate) is False
    assert runtime.submit(
        _pcm(1),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.requires_provisional_decision(candidate) is True

    await asyncio.wait_for(callback_started.wait(), 2.0)
    assert runtime.requires_provisional_decision(candidate) is True
    callback_release.set()
    await runtime.wait_idle()

    assert runtime.requires_provisional_decision(candidate) is False
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


async def test_failed_checkpoint_callback_keeps_provisional_decision() -> None:
    async def observe(_observation: SpeakerShadowObservation) -> None:
        raise RuntimeError("checkpoint callback failed")

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(),
        on_observation=observe,
    )
    candidate = _candidate(354)

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    assert runtime.requires_provisional_decision(candidate) is True
    assert runtime.snapshot()["callback_failure_count"] == 1
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


async def test_reset_prevents_late_callback_from_delivering_checkpoint() -> None:
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()

    async def observe(_observation: SpeakerShadowObservation) -> None:
        callback_started.set()
        try:
            await callback_release.wait()
        except asyncio.CancelledError:
            await callback_release.wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(callback_timeout_seconds=1.0),
        on_observation=observe,
    )
    candidate = _candidate(355)

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await asyncio.wait_for(callback_started.wait(), 2.0)
    assert runtime.requires_provisional_decision(candidate) is True

    await runtime.reset()
    assert runtime.requires_provisional_decision(candidate) is False
    callback_release.set()
    await runtime.wait_idle()

    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


@pytest.mark.parametrize(
    ("config", "scope"),
    [
        (_config(minimum_audio_ms=1_500, maximum_audio_ms=4_000),
         "provider_candidate"),
        (_provider_gate_config(), "smart_turn_turn"),
    ],
    ids=["default-off", "smart-turn"],
)
async def test_provisional_decision_is_scope_gated(
    config: SpeakerShadowConfig,
    scope: str,
) -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=config,
    )
    candidate = _candidate(356, scope)

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.wait_idle()
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


async def test_provider_prewarm_loads_once_without_scoring() -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_provider_gate_config(prewarm=True),
        on_observation=observe,
    )
    candidate = _candidate(357)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["load_count"] == 1
    assert metrics["evaluated_candidate_count"] == 0
    assert metrics["would_block_count"] == 0
    assert metrics["active_audio_bytes"] == 0
    assert observations == []
    await runtime.close()


@pytest.mark.parametrize(
    ("config", "scope"),
    [
        (_provider_gate_config(), "provider_candidate"),
        (_provider_gate_config(prewarm=True), "smart_turn_turn"),
    ],
    ids=["prewarm-default-off", "smart-turn"],
)
async def test_backend_prewarm_is_scope_gated(
    config: SpeakerShadowConfig,
    scope: str,
) -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=config,
    )
    candidate = _candidate(358, scope)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    assert runtime.snapshot()["load_count"] == 0
    assert runtime.snapshot()["backend_process_count"] == 0
    await runtime.close()


async def test_backend_prewarm_failure_finalizes_candidate_fail_open() -> None:
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(load_ok=False),
        config=_provider_gate_config(prewarm=True),
        on_observation=observe,
    )
    candidate = _candidate(359)

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["failed_candidate_count"] == 1
    assert metrics["retained_pcm_bytes"] == 0
    assert observations == []
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


async def test_reset_invalidates_in_flight_backend_prewarm() -> None:
    load_started = _spawn_event()
    load_release = _spawn_event()
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            block_stage="load",
            stage_started=load_started,
            stage_release=load_release,
        ),
        config=_provider_gate_config(prewarm=True),
    )
    candidate = _candidate(360)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await _wait_until(load_started.is_set)

    await runtime.reset()
    load_release.set()
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["stale_result_count"] == 1
    assert metrics["retained_pcm_bytes"] == 0
    assert runtime.requires_provisional_decision(candidate) is False
    await runtime.close()


async def test_close_cancels_in_flight_backend_prewarm() -> None:
    load_started = _spawn_event()
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            block_stage="load",
            stage_started=load_started,
        ),
        config=_provider_gate_config(
            prewarm=True,
            shutdown_grace_seconds=0.05,
        ),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(361),
    )
    await _wait_until(load_started.is_set)

    await asyncio.wait_for(runtime.close(), 2.0)

    metrics = runtime.snapshot()
    assert metrics["worker_task_count"] == 0
    assert metrics["backend_process_count"] == 0
    assert metrics["retained_pcm_bytes"] == 0


async def test_checkpoint_callback_failure_does_not_block_confirmation() -> None:
    seen_checkpoints: list[int | None] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        if observation.checkpoint_ms == 1_500:
            raise RuntimeError("first checkpoint callback failed")
        seen_checkpoints.append(observation.checkpoint_ms)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.2),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
    )

    assert runtime.submit(
        _pcm(3_000),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(49),
    )
    await runtime.wait_idle()

    assert seen_checkpoints == [3_000]
    assert runtime.snapshot()["callback_failure_count"] == 1
    assert runtime.snapshot()["scored_candidate_count"] == 1
    await runtime.close()


async def test_checkpoint_accepts_pcm_while_intermediate_score_is_running() -> None:
    score_started = _spawn_event()
    score_release = _spawn_event()
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            score_value=0.2,
            block_stage="score",
            stage_started=score_started,
            stage_release=score_release,
        ),
        config=SpeakerShadowConfig(
            enabled=True,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_observation=observe,
    )
    candidate = _candidate(50)

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await _wait_until(score_started.is_set)
    for _ in range(250):
        assert runtime.submit(
            _pcm(10),
            sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
            candidate=candidate,
        )
    score_release.set()
    await runtime.wait_idle()

    assert [item.checkpoint_ms for item in observations] == [1_500, 3_000]
    assert runtime.snapshot()["scored_candidate_count"] == 1
    assert runtime.snapshot()["submitted_audio_ms"] == 4_000
    await runtime.close()


async def test_intermediate_score_failure_wipes_buffer_and_fails_open() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_error=True),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
    )

    assert runtime.submit(
        _pcm(1_500),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(51),
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["failed_candidate_count"] == 1
    assert metrics["retained_pcm_bytes"] == 0
    assert metrics["buffered_candidate_count"] == 0
    await runtime.close()


async def test_finish_releases_short_buffer_without_starting_host() -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=1_500,
            maximum_audio_ms=4_000,
            observation_checkpoints_ms=(1_500, 3_000),
        ),
        on_completion=complete,
    )
    candidate = _candidate(3)

    assert runtime.submit(
        _pcm(1_499),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["insufficient_candidate_count"] == 1
    assert metrics["finished_candidate_count"] == 1
    assert metrics["backend_process_count"] == 0
    assert metrics["buffered_audio_bytes"] == 0
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="insufficient",
            last_checkpoint_ms=None,
        )
    ]
    assert metrics["completion_count"] == 1
    assert metrics["completion_before_first_checkpoint_count"] == 1
    assert metrics["completion_after_first_checkpoint_count"] == 0
    await runtime.close()


async def test_candidate_pcm_is_capped_at_four_seconds_across_frames() -> None:
    expected = _pcm(1_000) * 4
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(expected_pcm=expected),
        config=_config(
            minimum_audio_ms=4_000,
            maximum_audio_ms=4_000,
            queue_capacity=8,
        ),
    )
    candidate = _candidate(4)

    for _ in range(4):
        assert runtime.submit(
            _pcm(1_000),
            sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
            candidate=candidate,
        )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False
    await runtime.wait_idle()

    assert runtime.snapshot()["submitted_audio_ms"] == 4_000
    assert runtime.snapshot()["scored_candidate_count"] == 1
    await runtime.close()


async def test_single_lifecycle_preroll_payload_above_one_second_is_accepted() -> None:
    preroll = _pcm(2_500)
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(expected_pcm=preroll),
        config=_config(
            minimum_audio_ms=2_500,
            maximum_audio_ms=4_000,
        ),
    )
    candidate = _candidate(5)

    assert runtime.submit(
        preroll,
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["submitted_audio_ms"] == 2_500
    assert metrics["scored_candidate_count"] == 1
    assert metrics["retained_pcm_bytes"] == 0
    await runtime.close()


async def test_warm_worker_releases_the_last_parent_pcm_frame() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=10, idle_unload_seconds=60.0),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(46),
    )
    await runtime.wait_idle()

    worker = runtime._worker_task
    assert worker is not None and not worker.done()
    worker_frame = worker.get_coro().cr_frame
    assert worker_frame is not None
    assert worker_frame.f_locals.get("item") is None
    assert runtime.snapshot()["retained_pcm_bytes"] == 0
    await runtime.close()


@pytest.mark.parametrize(
    ("pcm16", "sample_rate_hz", "candidate"),
    [
        (b"", SPEAKER_SHADOW_SAMPLE_RATE_HZ, _candidate(5)),
        (b"\x00", SPEAKER_SHADOW_SAMPLE_RATE_HZ, _candidate(6)),
        (_pcm(10), 48_000, _candidate(7)),
        (_pcm(10), SPEAKER_SHADOW_SAMPLE_RATE_HZ, (1, 1)),
        (
            b"\x00" * (MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES + 2),
            SPEAKER_SHADOW_SAMPLE_RATE_HZ,
            _candidate(8),
        ),
    ],
    ids=["empty", "odd", "wrong-rate", "wrong-key", "oversized"],
)
async def test_invalid_or_oversized_frames_fail_open_without_starting_host(
    pcm16: bytes,
    sample_rate_hz: int,
    candidate: object,
) -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(),
    )

    assert runtime.submit(  # type: ignore[arg-type]
        pcm16,
        sample_rate_hz=sample_rate_hz,
        candidate=candidate,
    ) is False
    await runtime.wait_idle()
    assert runtime.snapshot()["backend_process_count"] == 0
    await runtime.close()


async def test_queue_saturation_drops_only_shadow_candidate() -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=20,
            queue_capacity=1,
            finalized_candidate_capacity=2,
        ),
        on_completion=complete,
    )
    candidate = _candidate(9)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["dropped_frame_count"] == 1
    assert metrics["dropped_candidate_count"] == 1
    assert metrics["finished_candidate_count"] == 1
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="dropped",
            last_checkpoint_ms=None,
        )
    ]
    assert metrics["backend_process_count"] == 0
    await runtime.close()


async def test_pcm_capacity_saturation_preserves_finish_admission() -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=20,
            queue_capacity=1,
            terminal_queue_capacity=1,
            completion_queue_capacity=1,
        ),
        on_completion=complete,
    )
    candidate = _candidate(901)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    token = runtime._candidate_tokens[candidate]
    assert runtime.snapshot()["queued_item_count"] == 1

    assert runtime.finish_candidate(candidate)
    assert token.finish_state == "queued"
    queued = runtime.snapshot()
    assert queued["pending_terminal_count"] == 1
    assert queued["terminal_queued_count"] == 1
    assert queued["terminal_overflow_count"] == 0

    await runtime.wait_idle()

    assert token.finish_state == "processed"
    assert token.completion_state == "attempted"
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="insufficient",
            last_checkpoint_ms=None,
        )
    ]
    completed = runtime.snapshot()
    assert completed["pending_terminal_count"] == 0
    assert completed["pending_completion_count"] == 0
    assert completed["completion_count"] == 1
    assert completed["completion_dispatched_count"] == 1
    assert completed["completion_attempted_count"] == 1
    await runtime.close()


async def test_terminal_capacity_overflow_never_claims_finish_processed() -> None:
    degraded_calls = 0

    def degraded() -> None:
        nonlocal degraded_calls
        degraded_calls += 1

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            queue_capacity=1,
            terminal_queue_capacity=1,
            completion_queue_capacity=1,
        ),
        on_backend_degraded=degraded,
    )
    hold_worker = asyncio.Event()
    fake_worker = asyncio.create_task(hold_worker.wait())
    runtime._worker_task = fake_worker
    queued_candidate = _candidate(902)
    overflowed_candidate = _candidate(903)

    try:
        assert runtime.finish_candidate(queued_candidate)
        queued_token = runtime._candidate_tokens[queued_candidate]
        assert queued_token.finish_state == "queued"

        assert runtime.finish_candidate(overflowed_candidate) is False
        overflowed = runtime._finalized[overflowed_candidate]
        assert overflowed.token is not None
        assert overflowed.token.finish_state == "abandoned"
        assert overflowed.token.finish_state != "processed"

        metrics = runtime.snapshot()
        assert metrics["pending_terminal_count"] == 1
        assert metrics["terminal_queued_count"] == 1
        assert metrics["terminal_overflow_count"] == 1
        assert metrics["terminal_abandoned_count"] == 1
        assert metrics["delivery_degraded_count"] == 1
        assert metrics["delivery_degraded_cause_count"] == 1
        assert metrics["completion_count"] == 0
        assert degraded_calls == 1
    finally:
        hold_worker.set()
        await fake_worker
        await runtime.close()


async def test_worker_start_failure_abandons_unaccepted_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    degraded_calls = 0

    def degraded() -> None:
        nonlocal degraded_calls
        degraded_calls += 1

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            terminal_queue_capacity=1,
            completion_queue_capacity=1,
        ),
        on_backend_degraded=degraded,
    )
    candidate = _candidate(909)
    monkeypatch.setattr(runtime, "_ensure_worker", lambda: False)

    assert runtime.finish_candidate(candidate) is False

    finalized = runtime._finalized[candidate]
    assert finalized.token is not None
    assert finalized.token.finish_state == "abandoned"
    metrics = runtime.snapshot()
    assert metrics["terminal_queued_count"] == 0
    assert metrics["pending_terminal_count"] == 0
    assert metrics["terminal_overflow_count"] == 0
    assert metrics["terminal_abandoned_count"] == 1
    assert metrics["worker_start_failure_count"] == 1
    assert metrics["delivery_degraded_count"] == 1
    assert metrics["delivery_degraded_cause_count"] == 1
    assert degraded_calls == 1
    await runtime.close()


async def test_finish_processing_exception_still_delivers_failed_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    async def fail_process_finish(_marker: _CandidateFinished) -> None:
        raise RuntimeError("finish processing failed")

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            terminal_queue_capacity=1,
            completion_queue_capacity=1,
        ),
        on_completion=complete,
    )
    monkeypatch.setattr(runtime, "_process_finish", fail_process_finish)
    candidate = _candidate(910)

    assert runtime.finish_candidate(candidate)
    token = runtime._candidate_tokens[candidate]
    await runtime.wait_idle()

    assert token.finish_state == "processed"
    assert token.completion_state == "attempted"
    assert token.terminal_reason == "failed"
    assert completions == [
        SpeakerShadowCompletion(
            candidate=candidate,
            terminal_reason="failed",
            last_checkpoint_ms=None,
        )
    ]
    metrics = runtime.snapshot()
    assert metrics["inference_failure_count"] == 1
    assert metrics["completion_count"] == 1
    assert metrics["completion_dispatched_count"] == 1
    assert metrics["completion_attempted_count"] == 1
    assert metrics["pending_terminal_count"] == 0
    assert metrics["pending_completion_count"] == 0
    await runtime.close()


async def test_finalized_candidate_keeps_completion_when_other_pcm_fills_queue(
) -> None:
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(score_value=0.8),
        config=_config(minimum_audio_ms=10, queue_capacity=1),
        on_completion=complete,
    )
    finalized_candidate = _candidate(91)
    queued_candidate = _candidate(92)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=finalized_candidate,
    )
    await runtime.wait_idle()

    assert runtime.submit(
        _pcm(1),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=queued_candidate,
    )
    assert runtime.finish_candidate(finalized_candidate)
    await runtime.wait_idle()

    assert completions == [
        SpeakerShadowCompletion(
            candidate=finalized_candidate,
            terminal_reason="scored",
            last_checkpoint_ms=10,
        )
    ]
    metrics = runtime.snapshot()
    assert metrics["completion_count"] == 1
    assert metrics["dropped_frame_count"] == 0
    assert metrics["dropped_candidate_count"] == 0
    await runtime.close()


async def test_completion_callbacks_never_overlap_after_ignored_cancellation(
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    active_callbacks = 0
    maximum_active_callbacks = 0
    order: list[tuple[str, int]] = []
    first_candidate = _candidate(93)
    second_candidate = _candidate(94)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        nonlocal active_callbacks, maximum_active_callbacks
        active_callbacks += 1
        maximum_active_callbacks = max(maximum_active_callbacks, active_callbacks)
        order.append(("start", completion.candidate.shadow_generation))
        try:
            if completion.candidate == first_candidate:
                first_started.set()
                while not release_first.is_set():
                    try:
                        await release_first.wait()
                    except asyncio.CancelledError:
                        continue
            else:
                second_started.set()
        finally:
            order.append(("end", completion.candidate.shadow_generation))
            active_callbacks -= 1

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=20,
            queue_capacity=2,
            callback_timeout_seconds=0.05,
        ),
        on_completion=complete,
    )
    assert runtime.finish_candidate(first_candidate)
    await asyncio.wait_for(first_started.wait(), 1.0)
    await runtime.wait_idle()
    assert runtime._completion_callback_task is not None

    assert runtime.finish_candidate(second_candidate)
    await asyncio.sleep(0)
    assert second_started.is_set() is False
    assert maximum_active_callbacks == 1
    release_first.set()
    await runtime.wait_idle()

    assert second_started.is_set()
    assert maximum_active_callbacks == 1
    assert order == [
        ("start", first_candidate.shadow_generation),
        ("end", first_candidate.shadow_generation),
        ("start", second_candidate.shadow_generation),
        ("end", second_candidate.shadow_generation),
    ]
    await runtime.close()


async def test_blocked_completion_keeps_next_completion_queued_until_fifo_turn(
) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    order: list[tuple[str, int]] = []
    first_candidate = _candidate(904)
    second_candidate = _candidate(905)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        generation = completion.candidate.shadow_generation
        order.append(("start", generation))
        try:
            if completion.candidate == first_candidate:
                first_started.set()
                while not release_first.is_set():
                    try:
                        await release_first.wait()
                    except asyncio.CancelledError:
                        continue
            else:
                second_started.set()
        finally:
            order.append(("end", generation))

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            queue_capacity=2,
            terminal_queue_capacity=2,
            completion_queue_capacity=2,
            callback_timeout_seconds=0.01,
        ),
        on_completion=complete,
    )

    assert runtime.finish_candidate(first_candidate)
    first_token = runtime._candidate_tokens[first_candidate]
    await asyncio.wait_for(first_started.wait(), 1.0)
    assert runtime.finish_candidate(second_candidate)
    second_token = runtime._candidate_tokens[second_candidate]

    await asyncio.wait_for(runtime.wait_idle(), 1.0)

    stalled = runtime.snapshot()
    assert first_token.completion_state == "dispatched"
    assert second_token.completion_state == "queued"
    assert second_started.is_set() is False
    assert stalled["completion_count"] == 2
    assert stalled["completion_dispatched_count"] == 1
    assert stalled["completion_attempted_count"] == 0
    assert stalled["completion_stall_count"] == 1
    assert stalled["pending_completion_count"] == 1
    assert stalled["delivery_degraded_cause_count"] == 1

    release_first.set()
    await asyncio.wait_for(second_started.wait(), 1.0)
    await asyncio.wait_for(runtime.wait_idle(), 1.0)

    assert first_token.completion_state == "attempted"
    assert second_token.completion_state == "attempted"
    assert order == [
        ("start", first_candidate.shadow_generation),
        ("end", first_candidate.shadow_generation),
        ("start", second_candidate.shadow_generation),
        ("end", second_candidate.shadow_generation),
    ]
    delivered = runtime.snapshot()
    assert delivered["completion_dispatched_count"] == 2
    assert delivered["completion_attempted_count"] == 2
    assert delivered["pending_completion_count"] == 0
    await runtime.close()


async def test_completion_outbox_overflow_is_explicit_and_never_dispatched() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    delivered_candidates: list[SpeakerShadowCandidateKey] = []
    first_candidate = _candidate(906)
    second_candidate = _candidate(907)
    overflowed_candidate = _candidate(908)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        delivered_candidates.append(completion.candidate)
        if completion.candidate == first_candidate:
            first_started.set()
            while not release_first.is_set():
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    continue
        elif completion.candidate == second_candidate:
            second_started.set()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            queue_capacity=3,
            terminal_queue_capacity=3,
            completion_queue_capacity=1,
            callback_timeout_seconds=0.01,
        ),
        on_completion=complete,
    )

    assert runtime.finish_candidate(first_candidate)
    await asyncio.wait_for(first_started.wait(), 1.0)
    assert runtime.finish_candidate(second_candidate)
    second_token = runtime._candidate_tokens[second_candidate]
    deadline = asyncio.get_running_loop().time() + 1.0
    while second_token.completion_state != "queued":
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0)

    assert runtime.finish_candidate(overflowed_candidate)
    overflowed_token = runtime._candidate_tokens[overflowed_candidate]
    await asyncio.wait_for(runtime.wait_idle(), 1.0)

    overflowed = runtime.snapshot()
    assert overflowed_token.finish_state == "processed"
    assert overflowed_token.completion_state == "abandoned"
    assert second_token.completion_state == "queued"
    assert second_started.is_set() is False
    assert overflowed["completion_count"] == 2
    assert overflowed["completion_dispatched_count"] == 1
    assert overflowed["completion_overflow_count"] == 1
    assert overflowed["completion_abandoned_count"] == 1
    assert overflowed["pending_completion_count"] == 1
    assert overflowed["delivery_degraded_count"] == 1
    assert overflowed["delivery_degraded_cause_count"] >= 1

    release_first.set()
    await asyncio.wait_for(second_started.wait(), 1.0)
    await asyncio.wait_for(runtime.wait_idle(), 1.0)

    assert delivered_candidates == [first_candidate, second_candidate]
    assert overflowed_candidate not in delivered_candidates
    assert runtime.snapshot()["completion_dispatched_count"] == 2
    await runtime.close()


async def test_reset_detaches_uncooperative_completion_without_false_busy() -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    async def complete(_completion: SpeakerShadowCompletion) -> None:
        callback_started.set()
        while not release_callback.is_set():
            try:
                await release_callback.wait()
            except asyncio.CancelledError:
                continue

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(callback_timeout_seconds=0.01),
        on_completion=complete,
    )
    candidate = _candidate(912)
    assert runtime.finish_candidate(candidate)
    token = runtime._candidate_tokens[candidate]
    await asyncio.wait_for(callback_started.wait(), 1.0)

    await asyncio.wait_for(runtime.reset(), 1.0)
    await asyncio.wait_for(runtime.wait_idle(), 0.2)

    detached = runtime.snapshot()
    assert token.completion_state == "dispatched"
    assert detached["detached_callback_task_count"] == 1
    assert detached["callback_task_count"] == 1

    release_callback.set()
    await _wait_until(lambda: token.completion_state == "attempted")
    completed = runtime.snapshot()
    assert completed["completion_attempted_count"] == 1
    assert completed["detached_callback_task_count"] == 0
    assert completed["callback_task_count"] == 0
    await runtime.close()


async def test_reset_stalled_completion_recovers_when_detached_task_finishes() -> None:
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()
    degraded_count = 0
    recovered_count = 0

    async def complete(_completion: SpeakerShadowCompletion) -> None:
        callback_started.set()
        while not release_callback.is_set():
            try:
                await release_callback.wait()
            except asyncio.CancelledError:
                continue

    def on_degraded() -> None:
        nonlocal degraded_count
        degraded_count += 1

    def on_recovered() -> None:
        nonlocal recovered_count
        recovered_count += 1

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(callback_timeout_seconds=0.01),
        on_completion=complete,
        on_backend_degraded=on_degraded,
        on_backend_recovered=on_recovered,
    )
    candidate = _candidate(913)
    assert runtime.finish_candidate(candidate)
    token = runtime._candidate_tokens[candidate]
    await asyncio.wait_for(callback_started.wait(), 1.0)
    await asyncio.wait_for(runtime.wait_idle(), 1.0)
    assert runtime.snapshot()["delivery_degraded_cause_count"] == 1

    await asyncio.wait_for(runtime.reset(), 1.0)
    release_callback.set()
    await _wait_until(lambda: token.completion_state == "attempted")

    recovered = runtime.snapshot()
    assert degraded_count == 1
    assert recovered_count == 1
    assert recovered["delivery_degraded_cause_count"] == 0
    assert recovered["completion_attempted_count"] == 1
    await runtime.close()


async def test_new_generation_completion_waits_for_detached_predecessor() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    first_candidate = _candidate(915)
    second_candidate = _candidate(916)

    async def complete(completion: SpeakerShadowCompletion) -> None:
        if completion.candidate == first_candidate:
            first_started.set()
            while not release_first.is_set():
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    continue
            return
        second_started.set()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(callback_timeout_seconds=0.01),
        on_completion=complete,
    )
    assert runtime.finish_candidate(first_candidate)
    await asyncio.wait_for(first_started.wait(), 1.0)
    await asyncio.wait_for(runtime.reset(), 1.0)

    assert runtime.finish_candidate(second_candidate)
    second_token = runtime._candidate_tokens[second_candidate]
    await asyncio.wait_for(runtime.wait_idle(), 0.2)

    assert second_token.completion_state == "queued"
    assert second_started.is_set() is False
    assert runtime.snapshot()["detached_callback_task_count"] == 1

    release_first.set()
    await asyncio.wait_for(second_started.wait(), 1.0)
    await asyncio.wait_for(runtime.wait_idle(), 1.0)
    assert second_token.completion_state == "attempted"
    await runtime.close()


async def test_unexpected_worker_cancel_abandons_consumed_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_started = asyncio.Event()

    async def block_finish(_marker: _CandidateFinished) -> None:
        finish_started.set()
        await asyncio.Event().wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(terminal_queue_capacity=1),
    )
    monkeypatch.setattr(runtime, "_process_finish", block_finish)
    candidate = _candidate(914)
    assert runtime.finish_candidate(candidate)
    token = runtime._candidate_tokens[candidate]
    await asyncio.wait_for(finish_started.wait(), 1.0)

    worker = runtime._worker_task
    assert worker is not None
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker

    assert token.finish_state == "abandoned"
    assert token.completion_state == "abandoned"
    assert runtime.finish_candidate(candidate) is False
    metrics = runtime.snapshot()
    assert metrics["pending_terminal_count"] == 0
    assert metrics["worker_start_failure_count"] == 1
    assert metrics["delivery_degraded_cause_count"] == 1
    await runtime.close()


async def test_unexpected_worker_cancel_abandons_processed_finish_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finish_processed = asyncio.Event()

    async def block_after_process(marker: _CandidateFinished) -> None:
        runtime._mark_finish_processed(marker.token)
        finish_processed.set()
        await asyncio.Event().wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(terminal_queue_capacity=1),
    )
    monkeypatch.setattr(runtime, "_process_finish", block_after_process)
    candidate = _candidate(917)
    assert runtime.finish_candidate(candidate)
    token = runtime._candidate_tokens[candidate]
    await asyncio.wait_for(finish_processed.wait(), 1.0)

    worker = runtime._worker_task
    assert worker is not None
    worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker

    assert token.finish_state == "processed"
    assert token.completion_state == "abandoned"
    assert runtime.snapshot()["completion_abandoned_count"] == 1
    await runtime.close()


async def test_buffers_and_tombstones_remain_bounded() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=20,
            queue_capacity=4,
            buffered_candidate_capacity=2,
            finalized_candidate_capacity=4,
        ),
    )

    for generation in range(10, 20):
        candidate = _candidate(generation)
        assert runtime.submit(
            _pcm(1),
            sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
            candidate=candidate,
        )
        await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["buffered_candidate_count"] == 2
    assert metrics["dropped_candidate_count"] == 8
    assert metrics["finalized_tombstone_count"] <= 4
    assert metrics["buffered_audio_bytes"] <= len(_pcm(1)) * 2
    await runtime.close()


async def test_evicted_tombstone_keeps_late_finish_idempotent() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            queue_capacity=1,
            finalized_candidate_capacity=1,
        ),
    )
    evicted_candidate = _candidate(20)

    assert runtime.finish_candidate(evicted_candidate)
    await runtime.wait_idle()
    assert runtime.finish_candidate(_candidate(21))
    await runtime.wait_idle()
    before_duplicate = runtime.snapshot()

    assert runtime.finish_candidate(evicted_candidate)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=evicted_candidate,
    ) is False
    await runtime.wait_idle()

    after_duplicate = runtime.snapshot()
    assert after_duplicate["finished_candidate_count"] == before_duplicate[
        "finished_candidate_count"
    ]
    assert after_duplicate["insufficient_candidate_count"] == before_duplicate[
        "insufficient_candidate_count"
    ]
    assert after_duplicate["finalized_tombstone_count"] == 1
    await runtime.close()


async def test_eviction_watermark_preserves_older_live_candidate() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            queue_capacity=1,
            finalized_candidate_capacity=1,
        ),
    )
    live_candidate = _candidate(20)

    assert runtime.submit(
        _pcm(1),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=live_candidate,
    )
    await runtime.wait_idle()
    for generation in (21, 22):
        assert runtime.finish_candidate(_candidate(generation))
        await runtime.wait_idle()

    assert runtime.finish_candidate(live_candidate)
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["buffered_candidate_count"] == 0
    assert metrics["finished_candidate_count"] == 3
    assert metrics["insufficient_candidate_count"] == 3
    await runtime.close()


async def test_queued_work_ignores_evicted_candidate_watermark() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(queue_capacity=1, finalized_candidate_capacity=1),
    )
    candidate = _candidate(20)
    marker = _CandidateFinished(
        runtime._generation,
        candidate,
        _CandidateToken(candidate, 0),
    )
    pcm16 = bytearray(_pcm(10))
    frame = _AudioFrame(
        runtime._generation,
        candidate,
        _CandidateToken(candidate, SPEAKER_SHADOW_SAMPLE_RATE_HZ),
        pcm16,
        SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        len(pcm16) // 2,
    )
    runtime._record_evicted_candidate(candidate)
    before_work = runtime.snapshot()

    await runtime._process_finish(marker)
    await runtime._process_frame(frame)

    assert runtime.snapshot() == before_work
    await runtime.close()


def test_threshold_metric_keys_preserve_distinct_float_values() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=None,
        config=_config(similarity_thresholds=(0.4, 0.404)),
    )

    metrics = runtime.snapshot()

    assert metrics["would_block_at_0_4_count"] == 0
    assert metrics["would_block_at_0_404_count"] == 0
    asyncio.run(runtime.close())


@pytest.mark.parametrize("failure_stage", ["load", "score", "callback"])
async def test_failures_stay_inside_shadow_and_have_one_terminal_state(
    failure_stage: str,
) -> None:
    async def callback(_observation: SpeakerShadowObservation) -> None:
        if failure_stage == "callback":
            raise RuntimeError("callback failed")

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            load_error=failure_stage == "load",
            score_error=failure_stage == "score",
        ),
        config=_config(minimum_audio_ms=10),
        on_observation=callback,
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(20, "smart_turn_turn"),
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    terminal_count = sum(
        metrics[f"{reason}_candidate_count"]
        for reason in ("scored", "insufficient", "dropped", "failed")
    )
    assert terminal_count == 1
    if failure_stage == "callback":
        assert metrics["scored_candidate_count"] == 1
        assert metrics["callback_failure_count"] == 1
    else:
        assert metrics["failed_candidate_count"] == 1
    await runtime.close()
    assert runtime.snapshot()["backend_process_count"] == 0


async def test_reset_discards_in_flight_result_by_generation() -> None:
    started = _spawn_event()
    release = _spawn_event()
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            block_stage="score",
            stage_started=started,
            stage_release=release,
        ),
        config=_config(minimum_audio_ms=10),
        on_observation=observe,
    )
    before = _candidate(21)
    after = _candidate(22)

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=before,
    )
    await _wait_until(started.is_set)
    old_generation = runtime.generation
    await runtime.reset()
    assert runtime.generation == old_generation + 1
    release.set()
    await runtime.wait_idle()

    assert observations == []
    assert runtime.snapshot()["stale_result_count"] == 1
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=after,
    )
    await runtime.wait_idle()
    assert [item.candidate for item in observations] == [after]
    await runtime.close()


async def test_reset_cancels_stale_observation_delivery() -> None:
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    observations: list[SpeakerShadowObservation] = []

    async def observe(observation: SpeakerShadowObservation) -> None:
        callback_started.set()
        await callback_release.wait()
        observations.append(observation)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=10),
        on_observation=observe,
    )
    candidate = _candidate(23)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    await asyncio.wait_for(callback_started.wait(), 2.0)

    await runtime.reset()
    callback_release.set()
    await runtime.wait_idle()

    assert observations == []
    assert runtime.snapshot()["stale_result_count"] == 1
    assert runtime.snapshot()["callback_task_count"] == 0
    await runtime.close()


async def test_reset_cancels_stale_completion_delivery() -> None:
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    completions: list[SpeakerShadowCompletion] = []

    async def complete(completion: SpeakerShadowCompletion) -> None:
        callback_started.set()
        await callback_release.wait()
        completions.append(completion)

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=20),
        on_completion=complete,
    )
    candidate = _candidate(123)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await asyncio.wait_for(callback_started.wait(), 2.0)

    await runtime.reset()
    callback_release.set()
    await runtime.wait_idle()

    assert completions == []
    assert runtime.snapshot()["stale_result_count"] == 1
    assert runtime.snapshot()["callback_task_count"] == 0
    await runtime.close()


async def test_completion_callback_failure_is_counted_and_contained() -> None:
    async def complete(_completion: SpeakerShadowCompletion) -> None:
        raise RuntimeError("completion failed")

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=20),
        on_completion=complete,
    )
    candidate = _candidate(124)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    assert runtime.finish_candidate(candidate)
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["completion_count"] == 1
    assert metrics["completion_callback_failure_count"] == 1
    assert metrics["callback_failure_count"] == 1
    await runtime.close()


def test_sync_observation_callback_is_rejected() -> None:
    with pytest.raises(TypeError, match="callback must be async"):
        SpeakerShadowRuntime(
            backend_factory=_BackendFactory(),
            config=_config(),
            on_observation=lambda _observation: None,  # type: ignore[arg-type]
        )


def test_sync_completion_callback_is_rejected() -> None:
    with pytest.raises(TypeError, match="completion callback must be async"):
        SpeakerShadowRuntime(
            backend_factory=_BackendFactory(),
            config=_config(),
            on_completion=lambda _completion: None,  # type: ignore[arg-type]
        )


async def test_idle_unload_releases_then_reloads_one_serial_host() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=10, idle_unload_seconds=0.05),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(24),
    )
    await runtime.wait_idle()
    assert runtime.snapshot()["backend_process_count"] == 1
    await _wait_until(lambda: runtime.snapshot()["unload_count"] == 1)
    assert runtime.snapshot()["backend_process_count"] == 0
    assert runtime.snapshot()["unload_count"] == 1

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(25),
    )
    await runtime.wait_idle()
    assert runtime.snapshot()["load_count"] == 2
    assert runtime.snapshot()["backend_process_count"] == 1
    await runtime.close()
    assert runtime.snapshot()["backend_process_count"] == 0


async def test_close_error_is_fail_open_and_next_candidate_can_reload() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(close_error=True),
        config=_config(minimum_audio_ms=10, idle_unload_seconds=0.05),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(26),
    )
    await runtime.wait_idle()
    await _wait_until(lambda: runtime.snapshot()["unload_failure_count"] == 1)
    assert runtime.snapshot()["backend_process_count"] == 0
    assert runtime.snapshot()["unload_failure_count"] == 1
    assert runtime.enabled is True

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(27),
    )
    await runtime.wait_idle()
    assert runtime.snapshot()["load_count"] == 2
    await runtime.close()
    assert runtime.snapshot()["backend_process_count"] == 0


async def test_parent_factory_profile_is_wiped_once_on_idempotent_close() -> None:
    profile = bytearray(b"private-profile")
    factory = _BackendFactory(parent_profile=profile)
    runtime = SpeakerShadowRuntime(
        backend_factory=factory,
        config=_config(minimum_audio_ms=10),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(28),
    )
    await runtime.wait_idle()
    assert profile == bytearray(b"private-profile")

    await runtime.close()
    await runtime.close()

    assert profile == bytearray(len(profile))
    assert factory.parent_close_calls == 1
    assert runtime.snapshot()["backend_process_count"] == 0


async def test_unpicklable_factory_fails_open_without_process_leak() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=lambda: _BackendFactory()(),
        config=_config(minimum_audio_ms=10),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(29),
    )
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    assert metrics["failed_candidate_count"] == 1
    assert metrics["load_failure_count"] == 1
    assert metrics["host_start_task_count"] == 0
    assert metrics["backend_process_count"] == 0
    await runtime.close()


@pytest.mark.parametrize("blocked_stage", ["load", "score"])
async def test_backend_operation_timeout_is_fail_open_and_reaps_host(
    blocked_stage: str,
) -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(block_stage=blocked_stage),
        config=_config(
            minimum_audio_ms=10,
            backend_load_timeout_seconds=0.1,
            backend_score_timeout_seconds=0.1,
            process_terminate_timeout_seconds=0.2,
        ),
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(40 + (blocked_stage == "score")),
    )

    await asyncio.wait_for(runtime.wait_idle(), 3.0)

    metrics = runtime.snapshot()
    assert metrics["failed_candidate_count"] == 1
    assert metrics["backend_timeout_count"] == 1
    assert metrics["backend_process_termination_count"] == 1
    assert metrics["backend_process_count"] == 0
    await runtime.close()


@pytest.mark.parametrize(
    "factory",
    [
        _BackendFactory(load_ok=False),
        _BackendFactory(score_value=float("nan")),
    ],
    ids=["unavailable-load", "invalid-score"],
)
async def test_backend_invalid_results_fail_open(
    factory: _BackendFactory,
) -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=factory,
        config=_config(minimum_audio_ms=10),
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(42),
    )

    await runtime.wait_idle()

    assert runtime.snapshot()["failed_candidate_count"] == 1
    await runtime.close()
    assert runtime.snapshot()["backend_process_count"] == 0


async def test_close_cancels_in_flight_observation_and_reaps_host() -> None:
    callback_started = asyncio.Event()

    async def observe(_observation: SpeakerShadowObservation) -> None:
        callback_started.set()
        await asyncio.Event().wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(minimum_audio_ms=10, shutdown_grace_seconds=0.05),
        on_observation=observe,
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(43),
    )
    await asyncio.wait_for(callback_started.wait(), 2.0)

    await asyncio.wait_for(runtime.close(), 2.0)

    metrics = runtime.snapshot()
    assert metrics["worker_task_count"] == 0
    assert metrics["callback_task_count"] == 0
    assert metrics["backend_process_count"] == 0


async def test_close_retries_after_callback_consumes_first_cancellation() -> None:
    callback_started = asyncio.Event()
    first_cancellation_seen = asyncio.Event()
    force_release = asyncio.Event()
    profile = bytearray(b"private-profile")
    factory = _BackendFactory(parent_profile=profile)

    async def observe(_observation: SpeakerShadowObservation) -> None:
        callback_started.set()
        try:
            await force_release.wait()
        except asyncio.CancelledError:
            first_cancellation_seen.set()
        await force_release.wait()

    runtime = SpeakerShadowRuntime(
        backend_factory=factory,
        config=_config(
            minimum_audio_ms=10,
            callback_timeout_seconds=0.05,
            shutdown_grace_seconds=0.05,
        ),
        on_observation=observe,
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(47),
    )
    await asyncio.wait_for(callback_started.wait(), 2.0)

    try:
        await asyncio.wait_for(runtime.close(), 2.0)

        metrics = runtime.snapshot()
        assert first_cancellation_seen.is_set()
        assert metrics["worker_task_count"] == 0
        assert metrics["cleanup_task_count"] == 0
        assert metrics["backend_process_count"] == 0
        assert metrics["callback_task_count"] == 0
        assert factory.parent_close_calls == 1
        assert profile == bytearray(len(profile))
    finally:
        force_release.set()
        callback_task = runtime._observation_task
        if callback_task is not None and not callback_task.done():
            callback_task.cancel()
            await asyncio.wait({callback_task}, timeout=2.0)
        await runtime.close()


async def test_finish_without_audio_and_late_pcm_has_one_terminal_state() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(),
    )
    candidate = _candidate(44)

    assert runtime.finish_candidate(candidate)
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    ) is False
    await runtime.wait_idle()

    metrics = runtime.snapshot()
    terminal_count = sum(
        metrics[f"{reason}_candidate_count"]
        for reason in ("scored", "insufficient", "dropped", "failed")
    )
    assert terminal_count == 1
    assert metrics["dropped_candidate_count"] == 1
    assert runtime.finish_candidate(candidate)
    assert runtime.finish_candidate(candidate)
    assert runtime.snapshot()["finished_candidate_count"] == 1
    await runtime.close()


def test_submit_without_running_loop_fails_open_and_drains_pcm() -> None:
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(),
    )

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(45),
    ) is False

    metrics = runtime.snapshot()
    assert metrics["worker_start_failure_count"] == 1
    assert metrics["queued_item_count"] == 0
    assert metrics["queued_audio_bytes"] == 0
    asyncio.run(runtime.close())


async def test_close_tracks_cancelled_off_loop_host_start_without_leaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from main_logic.asr_client.speaker_shadow import runtime as runtime_module

    start_entered = threading.Event()
    start_release = threading.Event()
    original_start = runtime_module._BackendProcessHost.create_started

    def delayed_start(**kwargs: object) -> Any:
        start_entered.set()
        start_release.wait(timeout=2.0)
        return original_start(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        runtime_module._BackendProcessHost,
        "create_started",
        delayed_start,
    )
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(),
        config=_config(
            minimum_audio_ms=10,
            shutdown_grace_seconds=0.05,
            process_terminate_timeout_seconds=0.05,
        ),
    )
    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=_candidate(30),
    )
    await _wait_until(start_entered.is_set)

    loop_progressed = False

    async def mark_loop_progress() -> None:
        nonlocal loop_progressed
        await asyncio.sleep(0)
        loop_progressed = True

    await mark_loop_progress()
    close_task = asyncio.create_task(runtime.close())
    # Exceed both cleanup cancellation budgets. Repeated cancellation must not
    # detach the off-loop start task and orphan the host it eventually returns.
    await asyncio.sleep(0.35)
    assert loop_progressed is True
    assert close_task.done() is False

    start_release.set()
    await asyncio.wait_for(close_task, 2.0)
    await _wait_until(lambda: not _speaker_host_pids())
    metrics = runtime.snapshot()
    assert metrics["host_start_task_count"] == 0
    assert metrics["worker_task_count"] == 0
    assert metrics["backend_process_count"] == 0


@pytest.mark.parametrize("blocked_stage", ["load", "score", "close"])
async def test_permanently_blocked_backend_close_is_bounded_and_leaves_no_resources(
    blocked_stage: str,
) -> None:
    started = _spawn_event()
    runtime = SpeakerShadowRuntime(
        backend_factory=_BackendFactory(
            block_stage=blocked_stage,
            stage_started=started,
        ),
        config=_config(
            minimum_audio_ms=10,
            shutdown_grace_seconds=0.05,
            backend_load_timeout_seconds=3.0,
            backend_score_timeout_seconds=3.0,
            backend_close_timeout_seconds=0.1,
            process_terminate_timeout_seconds=0.2,
        ),
    )
    candidate = _candidate(30 + ("load", "score", "close").index(blocked_stage))

    assert runtime.submit(
        _pcm(10),
        sample_rate_hz=SPEAKER_SHADOW_SAMPLE_RATE_HZ,
        candidate=candidate,
    )
    if blocked_stage == "close":
        await runtime.wait_idle()
        close_task = asyncio.create_task(runtime.close())
        await _wait_until(started.is_set)
    else:
        await _wait_until(started.is_set)
        close_task = asyncio.create_task(runtime.close())

    await asyncio.wait_for(close_task, 1.0)
    await _wait_until(lambda: not _speaker_host_pids())

    metrics = runtime.snapshot()
    assert metrics["worker_task_count"] == 0
    assert metrics["callback_task_count"] == 0
    assert metrics["cleanup_task_count"] == 0
    assert metrics["backend_loaded_count"] == 0
    assert metrics["backend_process_count"] == 0
    assert metrics["retained_pcm_bytes"] == 0
    assert metrics["backend_process_termination_count"] >= 1
    assert metrics["shutdown_timeout_count"] + metrics["backend_timeout_count"] >= 1
