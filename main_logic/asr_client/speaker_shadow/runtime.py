"""Bounded, fail-open runtime for observation-only speaker scoring."""

from __future__ import annotations

import asyncio
import inspect
import math
import multiprocessing
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Literal

from .contracts import (
    CompletionCallback,
    MAX_SPEAKER_SHADOW_CANDIDATE_PCM_BYTES,
    MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES,
    MAX_SPEAKER_SHADOW_RETAINED_PCM_BYTES,
    ObservationCallback,
    SpeakerShadowBackend,
    SpeakerShadowBackendFactory,
    SpeakerShadowCandidateKey,
    SpeakerShadowConfig,
    SpeakerShadowCompletion,
    SpeakerShadowMetrics,
    SpeakerShadowObservation,
    SpeakerShadowScope,
    SPEAKER_SHADOW_SAMPLE_RATE_HZ,
    SpeakerShadowTerminalReason,
)

_HOST_POLL_INTERVAL_SECONDS = 0.005
_HostOperation = Literal["load", "score", "close"]
_DegradedCause = Literal[
    "backend_unavailable",
    "terminal_overflow",
    "completion_overflow",
    "completion_stalled",
    "worker_start_failure",
    "dispatcher_start_failure",
    "resetting",
]


class _FinishState(StrEnum):
    OPEN = "open"
    QUEUED = "queued"
    PROCESSED = "processed"
    ABANDONED = "abandoned"


class _CompletionState(StrEnum):
    NONE = "none"
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    ATTEMPTED = "attempted"
    ABANDONED = "abandoned"


class _BackendHostError(RuntimeError):
    pass


class _BackendHostTimeout(_BackendHostError):
    pass


def _backend_host_error_name(exc: BaseException) -> str:
    """Return a non-sensitive error identity safe to cross the process pipe."""

    return type(exc).__name__


def _backend_host_main(
    factory: SpeakerShadowBackendFactory,
    connection: Connection,
    pcm_buffer: Any,
) -> None:
    """Own one blocking backend session inside a killable spawn process."""

    backend: SpeakerShadowBackend | None = None
    factory_closed = False

    def close_owned_resources() -> str | None:
        nonlocal backend, factory_closed
        error_name: str | None = None
        if backend is not None:
            owned_backend, backend = backend, None
            try:
                owned_backend.close()
            except BaseException as exc:  # process boundary must contain backend faults
                error_name = _backend_host_error_name(exc)
        close_factory = getattr(factory, "close", None)
        if not factory_closed and callable(close_factory):
            factory_closed = True
            try:
                close_factory()
            except BaseException as exc:  # process boundary must contain factory faults
                error_name = error_name or _backend_host_error_name(exc)
        return error_name

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            operation = message[0]
            try:
                if operation == "load":
                    if backend is None:
                        backend = factory()
                    connection.send((True, bool(backend.load())))
                    continue
                if operation == "score":
                    if backend is None:
                        raise RuntimeError("backend is not loaded")
                    pcm_length = int(message[1])
                    sample_rate_hz = int(message[2])
                    pcm16 = bytearray(
                        memoryview(pcm_buffer).cast("B")[:pcm_length]
                    )
                    try:
                        similarity = float(backend.score(bytes(pcm16), sample_rate_hz))
                    finally:
                        pcm16[:] = b"\x00" * len(pcm16)
                        del pcm16
                    connection.send((True, similarity))
                    continue
                if operation == "close":
                    error_name = close_owned_resources()
                    connection.send((error_name is None, error_name))
                    return
                raise RuntimeError("unsupported backend-host operation")
            except BaseException as exc:  # backend faults stay inside this process
                try:
                    connection.send((False, _backend_host_error_name(exc)))
                except (BrokenPipeError, EOFError, OSError):
                    return
    finally:
        close_owned_resources()
        connection.close()


class _BackendProcessHost:
    """One serial spawn-process host for one backend session."""

    def __init__(
        self,
        *,
        factory: SpeakerShadowBackendFactory,
        terminate_timeout_seconds: float,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        pcm_buffer = context.RawArray("B", MAX_SPEAKER_SHADOW_CANDIDATE_PCM_BYTES)
        process = context.Process(
            target=_backend_host_main,
            args=(factory, child_connection, pcm_buffer),
            name="speaker-shadow-backend",
            daemon=True,
        )
        self._connection: Connection | None = parent_connection
        self._child_connection: Connection | None = child_connection
        # Abandoned host reads keep a strong reference here until the thread
        # they are blocked in unwinds, so the event loop cannot drop them.
        self._pending_responses: set[asyncio.Future[Any]] = set()
        self._pcm_buffer = pcm_buffer
        self._process: BaseProcess | None = process
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self.loaded = False
        self.was_terminated = False
        self.timed_out = False
        self.pcm_bytes_in_use = 0

    @classmethod
    def create_started(
        cls,
        *,
        factory: SpeakerShadowBackendFactory,
        terminate_timeout_seconds: float,
    ) -> _BackendProcessHost:
        """Construct IPC resources and spawn outside the asyncio event loop."""

        host = cls(
            factory=factory,
            terminate_timeout_seconds=terminate_timeout_seconds,
        )
        host.start()
        return host

    @property
    def alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    @property
    def process_count(self) -> int:
        return int(self.alive)

    def start(self) -> None:
        process = self._process
        child_connection = self._child_connection
        if process is None or child_connection is None:
            raise _BackendHostError("backend host is already closed")
        try:
            process.start()
        except BaseException:
            self._dispose_handles()
            raise
        finally:
            child_connection.close()
            self._child_connection = None

    async def load(self, *, timeout_seconds: float) -> bool:
        available = bool(
            await self._request("load", timeout_seconds=timeout_seconds)
        )
        self.loaded = available
        return available

    async def score(
        self,
        pcm16: bytes | bytearray,
        *,
        timeout_seconds: float,
    ) -> float:
        if len(pcm16) > MAX_SPEAKER_SHADOW_CANDIDATE_PCM_BYTES:
            raise _BackendHostError("candidate PCM exceeds host buffer")
        if self._pcm_buffer is None:
            raise _BackendHostError("backend host PCM buffer is closed")
        pcm_view = memoryview(self._pcm_buffer).cast("B")
        pcm_view[: len(pcm16)] = pcm16
        self.pcm_bytes_in_use = len(pcm16)
        try:
            return float(
                await self._request(
                    "score",
                    len(pcm16),
                    SPEAKER_SHADOW_SAMPLE_RATE_HZ,
                    timeout_seconds=timeout_seconds,
                )
            )
        finally:
            pcm_view[: len(pcm16)] = b"\x00" * len(pcm16)
            self.pcm_bytes_in_use = 0

    async def close(self, *, timeout_seconds: float) -> bool:
        success = True
        if self.alive:
            try:
                await self._request("close", timeout_seconds=timeout_seconds)
            except _BackendHostError:
                success = False
        self.loaded = False
        if self.alive and not await self._wait_for_exit(timeout_seconds):
            success = False
            await self.terminate()
        await asyncio.to_thread(self._dispose_handles)
        return success

    async def terminate(self) -> None:
        process = self._process
        self.loaded = False
        if process is None:
            await asyncio.to_thread(self._dispose_handles)
            return
        if process.is_alive():
            self.was_terminated = True
            process.terminate()
            if not await self._wait_for_exit(self._terminate_timeout_seconds):
                process.kill()
                if not await self._wait_for_exit(self._terminate_timeout_seconds):
                    raise _BackendHostError("backend host could not be terminated")
        await asyncio.to_thread(self._dispose_handles)

    async def _request(
        self,
        operation: _HostOperation,
        *payload: object,
        timeout_seconds: float,
    ) -> object:
        connection = self._connection
        process = self._process
        if connection is None or process is None or not process.is_alive():
            await asyncio.to_thread(self._dispose_handles)
            raise _BackendHostError("backend host is not alive")
        try:
            connection.send((operation, *payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            await self.terminate()
            raise _BackendHostError("backend host command failed") from exc

        # One blocking read off the event loop, not a ``poll(0)`` spin. Each
        # zero-timeout poll starts an overlapped pipe read and cancels it in
        # the same breath, and on Windows that cancellation races the very
        # response it asked about: the child answers and stays alive, the
        # answer is swallowed, and the parent spins to a false timeout that no
        # later poll can recover. A plain ``recv`` issues one read and never
        # cancels it. Nothing else waits on this connection, so the read is
        # released either by the response or by the host dying — including the
        # ``terminate`` below, which closes the pipe on timeout and on
        # cancellation.
        response = asyncio.ensure_future(asyncio.to_thread(connection.recv))
        self._pending_responses.add(response)
        response.add_done_callback(self._consume_response_result)
        try:
            done, _ = await asyncio.wait({response}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            await self.terminate()
            raise
        if not done:
            self.timed_out = True
            # Terminating kills the child first, which breaks the pipe and
            # releases the blocked read before the handles are disposed.
            await self.terminate()
            await asyncio.wait({response}, timeout=self._terminate_timeout_seconds)
            raise _BackendHostTimeout(f"backend {operation} timed out")
        try:
            succeeded, value = response.result()
        except (BrokenPipeError, EOFError, OSError) as exc:
            if not process.is_alive():
                await asyncio.to_thread(self._dispose_handles)
                raise _BackendHostError(
                    "backend host exited without a response"
                ) from exc
            await self.terminate()
            raise _BackendHostError("backend host response failed") from exc
        if succeeded:
            return value
        raise _BackendHostError(f"backend operation failed: {value}")

    def _consume_response_result(self, response: asyncio.Future[Any]) -> None:
        """Retire an abandoned host read once its blocked thread unwinds."""

        self._pending_responses.discard(response)
        if response.cancelled():
            return
        response.exception()

    async def _wait_for_exit(self, timeout_seconds: float) -> bool:
        process = self._process
        if process is None:
            return True
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while process.is_alive():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_HOST_POLL_INTERVAL_SECONDS, remaining))
        await asyncio.to_thread(process.join, 0)
        return True

    def _dispose_handles(self) -> None:
        for connection_name in ("_connection", "_child_connection"):
            connection = getattr(self, connection_name)
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
                setattr(self, connection_name, None)
        process = self._process
        if process is not None and not process.is_alive():
            # ``Process.join()`` raises when ``start()`` failed before a PID was
            # assigned, which would otherwise mask the original spawn error.
            if process.pid is not None:
                process.join(timeout=0)
            try:
                process.close()
            except ValueError:
                pass
            self._process = None
        pcm_buffer = self._pcm_buffer
        if pcm_buffer is not None:
            pcm_view = memoryview(pcm_buffer).cast("B")
            pcm_view[:] = b"\x00" * len(pcm_view)
            self._pcm_buffer = None
            self.pcm_bytes_in_use = 0


@dataclass(frozen=True, slots=True)
class _AudioFrame:
    generation: int
    candidate: SpeakerShadowCandidateKey
    token: _CandidateToken
    pcm16: bytearray
    sample_rate_hz: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class _CandidateFinished:
    generation: int
    candidate: SpeakerShadowCandidateKey
    token: _CandidateToken


@dataclass(frozen=True, slots=True)
class _CandidateDeferred:
    generation: int
    candidate: SpeakerShadowCandidateKey
    token: _CandidateToken


@dataclass(frozen=True, slots=True)
class _CandidateActivated:
    generation: int
    candidate: SpeakerShadowCandidateKey
    token: _CandidateToken


@dataclass(slots=True)
class _CandidateToken:
    candidate: SpeakerShadowCandidateKey
    sample_rate_hz: int
    accepted_sample_count: int = 0
    terminal_reason: SpeakerShadowTerminalReason | None = None
    finish_state: _FinishState = _FinishState.OPEN
    last_checkpoint_ms: int | None = None
    last_delivered_checkpoint_ms: int | None = None
    completion_state: _CompletionState = _CompletionState.NONE
    deferred_requested: bool = False
    defer_processed: bool = False
    scoring_deferred: bool = False
    activation_queued: bool = False


@dataclass(slots=True)
class _CandidateBuffer:
    token: _CandidateToken
    sample_rate_hz: int
    pcm16: bytearray
    sample_count: int = 0
    next_checkpoint_index: int = 0
    completion_confirmation_checkpoint_ms: int | None = None
    backend_prewarm_attempted: bool = False

    @property
    def audio_ms(self) -> int:
        return self.sample_count * 1_000 // self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class _FinalizedCandidate:
    finish_state: _FinishState
    terminal_reason: SpeakerShadowTerminalReason
    token: _CandidateToken | None = None

    @property
    def finish_seen(self) -> bool:
        return self.finish_state is _FinishState.PROCESSED


@dataclass(frozen=True, slots=True)
class _CompletionEnvelope:
    generation: int
    candidate: SpeakerShadowCandidateKey
    token: _CandidateToken
    completion: SpeakerShadowCompletion


_STOP = object()
_COMPLETION_STOP = object()
_QueueItem = (
    _AudioFrame
    | _CandidateDeferred
    | _CandidateActivated
    | _CandidateFinished
    | object
)


class SpeakerShadowRuntime:
    """Score accepted candidate PCM without controlling the ASR path.

    ``submit`` and ``finish_candidate`` are non-blocking. Queue pressure and all
    backend/callback failures terminate shadow work locally and never escape to
    the ASR task graph. Observation callbacks are cancellation-cooperative;
    shutdown uses bounded repeated cancellation so a callback can finish cleanup
    after consuming its first cancellation request.
    """

    def __init__(
        self,
        *,
        backend_factory: SpeakerShadowBackendFactory | None,
        config: SpeakerShadowConfig | None = None,
        on_observation: ObservationCallback | None = None,
        on_completion: CompletionCallback | None = None,
        on_backend_degraded: Callable[[], None] | None = None,
        on_backend_recovered: Callable[[], None] | None = None,
    ) -> None:
        self._config = config or SpeakerShadowConfig()
        self._backend_factory = backend_factory
        self._on_backend_degraded = on_backend_degraded
        self._on_backend_recovered = on_backend_recovered
        if on_observation is not None and not (
            inspect.iscoroutinefunction(on_observation)
            or inspect.iscoroutinefunction(getattr(on_observation, "__call__", None))
        ):
            raise TypeError("SpeakerShadowRuntime observation callback must be async")
        self._on_observation = on_observation
        if on_completion is not None and not (
            inspect.iscoroutinefunction(on_completion)
            or inspect.iscoroutinefunction(getattr(on_completion, "__call__", None))
        ):
            raise TypeError("SpeakerShadowRuntime completion callback must be async")
        self._on_completion = on_completion
        self._metrics = SpeakerShadowMetrics()
        self._would_block_counts = {
            threshold: 0 for threshold in self._config.similarity_thresholds
        }
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue(
            maxsize=(
                self._config.queue_capacity
                + self._config.terminal_queue_capacity
                + 1
            )
        )
        self._queued_data_item_count = 0
        self._queued_terminal_count = 0
        self._completion_queue: asyncio.Queue[_CompletionEnvelope | object] = (
            asyncio.Queue(maxsize=self._config.completion_queue_capacity + 1)
        )
        self._queued_pcm_bytes = 0
        self._active_pcm_bytes = 0
        self._buffers: OrderedDict[SpeakerShadowCandidateKey, _CandidateBuffer] = (
            OrderedDict()
        )
        self._finalized: OrderedDict[
            SpeakerShadowCandidateKey, _FinalizedCandidate
        ] = OrderedDict()
        self._finalized_through: dict[SpeakerShadowScope, tuple[int, int]] = {}
        self._candidate_tokens: OrderedDict[
            SpeakerShadowCandidateKey, _CandidateToken
        ] = OrderedDict()
        self._worker_task: asyncio.Task[None] | None = None
        self._completion_dispatcher_task: asyncio.Task[None] | None = None
        self._completion_dispatch_in_progress = False
        self._observation_task: asyncio.Task[None] | None = None
        self._completion_callback_task: asyncio.Task[None] | None = None
        self._completion_callback_token: _CandidateToken | None = None
        self._detached_callback_tasks: set[asyncio.Task[None]] = set()
        self._detached_completion_tokens: dict[
            asyncio.Task[None], _CandidateToken
        ] = {}
        self._reset_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._host_start_task: asyncio.Task[_BackendProcessHost] | None = None
        self._active_evaluation: tuple[int, SpeakerShadowCandidateKey] | None = None
        self._active_evaluation_terminal = False
        self._active_terminal_token: _CandidateToken | None = None
        self._backend_host: _BackendProcessHost | None = None
        self._load_failure_streak = 0
        self._next_load_attempt_at = 0.0
        self._generation = 0
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._degraded_causes: set[_DegradedCause] = set()
        self._resetting = False
        self._closed = False
        self._factory_closed = False

    @property
    def enabled(self) -> bool:
        """Whether submissions can do work.

        A missing factory is treated exactly like disabled configuration: no
        PCM is queued and no task or model-loading attempt is created.
        """

        return (
            self._config.enabled
            and self._backend_factory is not None
            and not self._closed
        )

    @property
    def generation(self) -> int:
        return self._generation

    def supports_deferred_candidate(
        self,
        candidate: SpeakerShadowCandidateKey,
    ) -> bool:
        """Whether this exact candidate scope may use buffer-only admission."""

        return bool(
            self.enabled
            and isinstance(candidate, SpeakerShadowCandidateKey)
            and candidate.scope in self._config.pending_observation_gate_scopes
        )

    def defer_candidate(self, candidate: SpeakerShadowCandidateKey) -> bool:
        """Predeclare one candidate as buffer-only before accepting its first PCM."""

        if (
            self._resetting
            or not self.supports_deferred_candidate(candidate)
            or candidate in self._finalized
            or self._candidate_was_evicted(candidate)
        ):
            return False
        token = self._candidate_tokens.get(candidate)
        if token is not None:
            return bool(
                token.deferred_requested
                and token.terminal_reason is None
                and token.finish_state is _FinishState.OPEN
            )
        if len(self._candidate_tokens) >= self._config.buffered_candidate_capacity:
            return False
        token = _CandidateToken(
            candidate,
            0,
            deferred_requested=True,
            scoring_deferred=True,
        )
        marker = _CandidateDeferred(self._generation, candidate, token)
        if not self._admit_data_item(marker):
            return False
        self._candidate_tokens[candidate] = token
        self._candidate_tokens.move_to_end(candidate)
        return True

    def activate_candidate(self, candidate: SpeakerShadowCandidateKey) -> bool:
        """Order scoring activation behind all PCM already accepted for a defer."""

        if (
            self._resetting
            or not self.enabled
            or not isinstance(candidate, SpeakerShadowCandidateKey)
        ):
            return False
        token = self._candidate_tokens.get(candidate)
        if (
            token is None
            or not token.deferred_requested
            or token.terminal_reason is not None
            or token.finish_state is not _FinishState.OPEN
            or candidate in self._finalized
            or self._candidate_was_evicted(candidate, token=token)
        ):
            return False
        if not token.scoring_deferred or token.activation_queued:
            return True
        marker = _CandidateActivated(self._generation, candidate, token)
        if not self._admit_data_item(marker):
            self._drop_candidate(candidate, token=token)
            return False
        token.activation_queued = True
        return True

    def requires_provisional_decision(
        self,
        candidate: SpeakerShadowCandidateKey,
    ) -> bool:
        """Whether accepted PCM has reached a still-undelivered checkpoint."""

        if (
            self._resetting
            or not self.enabled
            or not isinstance(candidate, SpeakerShadowCandidateKey)
            or candidate.scope
            not in self._config.pending_observation_gate_scopes
            or candidate in self._finalized
        ):
            return False
        token = self._candidate_tokens.get(candidate)
        if (
            token is None
            or token.candidate != candidate
            or token.terminal_reason is not None
            or self._candidate_was_evicted(candidate, token=token)
        ):
            return False
        explicit_checkpoints = self._config.observation_checkpoints_ms
        first_checkpoint_ms = (
            explicit_checkpoints[0]
            if explicit_checkpoints is not None
            else self._config.minimum_audio_ms
        )
        first_checkpoint_samples = math.ceil(
            token.sample_rate_hz * first_checkpoint_ms / 1_000
        )
        delivered_checkpoint_ms = token.last_delivered_checkpoint_ms
        return (
            token.sample_rate_hz == SPEAKER_SHADOW_SAMPLE_RATE_HZ
            and (not token.scoring_deferred or token.activation_queued)
            and token.accepted_sample_count >= first_checkpoint_samples
            and (
                delivered_checkpoint_ms is None
                or delivered_checkpoint_ms < first_checkpoint_ms
            )
        )

    def snapshot(self) -> dict[str, int]:
        buffered_audio_bytes = sum(
            len(buffer.pcm16) for buffer in self._buffers.values()
        )
        host_pcm_bytes = (
            self._backend_host.pcm_bytes_in_use
            if self._backend_host is not None
            else 0
        )
        snapshot = self._metrics.snapshot()
        snapshot.update(
            buffered_candidate_count=len(self._buffers),
            buffered_audio_bytes=buffered_audio_bytes,
            queued_audio_bytes=self._queued_pcm_bytes,
            active_audio_bytes=self._active_pcm_bytes,
            retained_pcm_bytes=(
                buffered_audio_bytes
                + self._queued_pcm_bytes
                + self._active_pcm_bytes
                + host_pcm_bytes
            ),
            finalized_tombstone_count=len(self._finalized),
            queued_item_count=self._queue.qsize(),
            pending_terminal_count=self._queued_terminal_count,
            pending_completion_count=self._completion_queue.qsize(),
            detached_callback_task_count=sum(
                not task.done() for task in self._detached_callback_tasks
            ),
            delivery_degraded_cause_count=len(self._degraded_causes),
            in_flight_candidate_count=int(self._active_evaluation is not None),
            worker_task_count=int(
                self._worker_task is not None and not self._worker_task.done()
            ),
            callback_task_count=int(
                self._observation_task is not None
                and not self._observation_task.done()
            )
            + int(
                self._completion_callback_task is not None
                and not self._completion_callback_task.done()
            )
            + sum(not task.done() for task in self._detached_callback_tasks),
            completion_dispatcher_task_count=int(
                self._completion_dispatcher_task is not None
                and not self._completion_dispatcher_task.done()
            ),
            cleanup_task_count=int(
                self._cleanup_task is not None and not self._cleanup_task.done()
            ),
            host_start_task_count=int(
                self._host_start_task is not None
                and not self._host_start_task.done()
            ),
            backend_loaded_count=int(
                self._backend_host is not None
                and self._backend_host.alive
                and self._backend_host.loaded
            ),
            backend_process_count=(
                self._backend_host.process_count
                if self._backend_host is not None
                else 0
            ),
            backend_close_failed_count=0,
        )
        snapshot.update(
            {
                self._threshold_metric_key(threshold): count
                for threshold, count in self._would_block_counts.items()
            }
        )
        return snapshot

    def submit(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
        candidate: SpeakerShadowCandidateKey,
    ) -> bool:
        """Queue immutable PCM accepted by the current candidate fence."""

        if self._resetting or not self.enabled:
            return False
        if not isinstance(candidate, SpeakerShadowCandidateKey):
            return False
        if not isinstance(pcm16, bytes) or not pcm16 or len(pcm16) % 2:
            return False
        if sample_rate_hz != SPEAKER_SHADOW_SAMPLE_RATE_HZ:
            self._metrics.dropped_frame_count += 1
            return False
        if len(pcm16) > MAX_SPEAKER_SHADOW_FRAME_PCM_BYTES:
            self._metrics.dropped_frame_count += 1
            self._metrics.dropped_audio_ms += self._audio_ms(
                len(pcm16) // 2,
                SPEAKER_SHADOW_SAMPLE_RATE_HZ,
            )
            self._drop_candidate(candidate)
            return False

        identity = (self._generation, candidate)
        if (
            candidate in self._finalized
            or self._candidate_was_evicted(candidate)
            or (
                identity == self._active_evaluation
                and self._active_evaluation_terminal
            )
        ):
            return False

        token = self._candidate_tokens.get(candidate)
        if token is not None and (
            token.sample_rate_hz != sample_rate_hz
            and not (token.sample_rate_hz == 0 and token.deferred_requested)
        ):
            self._metrics.dropped_frame_count += 1
            self._metrics.dropped_audio_ms += self._audio_ms(
                len(pcm16) // 2,
                sample_rate_hz,
            )
            self._drop_candidate(candidate, token=token)
            return False
        if token is None:
            token = _CandidateToken(candidate, sample_rate_hz)
        elif token.sample_rate_hz == 0 and token.deferred_requested:
            token.sample_rate_hz = sample_rate_hz
        accepted_sample_count = token.accepted_sample_count
        maximum_samples = (
            sample_rate_hz * self._config.maximum_audio_ms // 1_000
        )
        remaining_samples = maximum_samples - accepted_sample_count
        if remaining_samples <= 0:
            self._metrics.dropped_frame_count += 1
            self._metrics.dropped_audio_ms += self._audio_ms(
                len(pcm16) // 2,
                sample_rate_hz,
            )
            return False
        input_sample_count = len(pcm16) // 2
        sample_count = min(input_sample_count, remaining_samples)
        if sample_count <= 0:
            return False
        if sample_count < input_sample_count:
            self._metrics.dropped_audio_ms += self._audio_ms(
                input_sample_count - sample_count,
                sample_rate_hz,
            )
        bounded_pcm16 = bytearray(memoryview(pcm16)[: sample_count * 2])
        frame = _AudioFrame(
            generation=self._generation,
            candidate=candidate,
            token=token,
            pcm16=bounded_pcm16,
            sample_rate_hz=sample_rate_hz,
            sample_count=sample_count,
        )
        if self._retained_pcm_bytes() + len(bounded_pcm16) > (
            MAX_SPEAKER_SHADOW_RETAINED_PCM_BYTES
        ):
            self._metrics.dropped_frame_count += 1
            self._metrics.dropped_audio_ms += self._audio_ms(
                sample_count,
                sample_rate_hz,
            )
            self._wipe_bytearray(bounded_pcm16)
            self._drop_candidate(candidate, token=token)
            return False
        if not self._admit_data_item(frame):
            self._metrics.dropped_frame_count += 1
            self._metrics.dropped_audio_ms += self._audio_ms(
                sample_count, sample_rate_hz
            )
            self._wipe_bytearray(bounded_pcm16)
            self._drop_candidate(candidate, token=token)
            return False
        self._queued_pcm_bytes += len(bounded_pcm16)
        token.accepted_sample_count = accepted_sample_count + sample_count
        self._candidate_tokens[candidate] = token
        self._candidate_tokens.move_to_end(candidate)
        self._metrics.submitted_frame_count += 1
        self._metrics.submitted_audio_ms += self._audio_ms(
            sample_count, sample_rate_hz
        )
        return True

    def finish_candidate(self, candidate: SpeakerShadowCandidateKey) -> bool:
        """Order the terminal boundary behind all previously accepted PCM."""

        if self._resetting or not self.enabled:
            return False
        if not isinstance(candidate, SpeakerShadowCandidateKey):
            return False
        finalized = self._finalized.get(candidate)
        if finalized is not None and finalized.finish_seen:
            return True
        if self._candidate_was_evicted(candidate):
            return True
        token = self._candidate_tokens.get(candidate)
        if token is None and finalized is not None:
            token = finalized.token
        if token is None:
            token = _CandidateToken(candidate, 0)
            if finalized is not None:
                token.terminal_reason = finalized.terminal_reason
        if token.finish_state in {
            _FinishState.QUEUED,
            _FinishState.PROCESSED,
        }:
            return True
        if token.finish_state is _FinishState.ABANDONED:
            return False
        marker = _CandidateFinished(self._generation, candidate, token)
        if not self._admit_terminal_item(marker):
            self._abandon_terminal(candidate, token=token)
            return False
        self._candidate_tokens[candidate] = token
        self._candidate_tokens.move_to_end(candidate)
        token.finish_state = _FinishState.QUEUED
        self._metrics.terminal_queued_count += 1
        return True

    def _admit_data_item(self, item: _QueueItem) -> bool:
        if self._resetting or self._closed:
            return False
        if self._queued_data_item_count >= self._config.queue_capacity:
            return False
        if not self._ensure_worker():
            self._metrics.worker_start_failure_count += 1
            self._set_degraded_cause("worker_start_failure")
            return False
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            return False
        self._queued_data_item_count += 1
        return True

    def _admit_terminal_item(self, marker: _CandidateFinished) -> bool:
        if self._resetting or self._closed:
            return False
        if self._queued_terminal_count >= self._config.terminal_queue_capacity:
            self._metrics.terminal_overflow_count += 1
            self._set_degraded_cause("terminal_overflow")
            return False
        if not self._ensure_worker():
            self._metrics.worker_start_failure_count += 1
            self._set_degraded_cause("worker_start_failure")
            return False
        try:
            self._queue.put_nowait(marker)
        except asyncio.QueueFull:
            self._metrics.terminal_overflow_count += 1
            self._set_degraded_cause("terminal_overflow")
            return False
        self._queued_terminal_count += 1
        return True

    def _abandon_terminal(
        self,
        candidate: SpeakerShadowCandidateKey,
        *,
        token: _CandidateToken,
    ) -> None:
        self._abandon_completion(token)
        if token.finish_state in {
            _FinishState.PROCESSED,
            _FinishState.ABANDONED,
        }:
            return
        token.finish_state = _FinishState.ABANDONED
        self._metrics.terminal_abandoned_count += 1
        self._drop_candidate(candidate, token=token)

    async def wait_idle(self) -> None:
        """Wait for accepted work, excluding the warm-backend idle timer."""

        reset_task = self._reset_task
        if (
            reset_task is not None
            and reset_task is not asyncio.current_task()
            and not reset_task.done()
        ):
            await asyncio.shield(reset_task)
        while not self._queue.empty():
            worker = self._worker_task
            if worker is None or worker.done():
                if self._closed or self._resetting:
                    self._drain_queue()
                    break
                if not self._ensure_worker():
                    self._metrics.worker_start_failure_count += 1
                    self._set_degraded_cause("worker_start_failure")
                    self._drain_queue()
                    break
            await asyncio.sleep(0)
        await self._queue.join()
        await asyncio.sleep(0)
        while True:
            callback = self._completion_callback_task
            completion_idle = (
                self._completion_queue.empty()
                and not self._completion_dispatch_in_progress
                and (callback is None or callback.done())
            )
            if completion_idle:
                return
            if "dispatcher_start_failure" in self._degraded_causes:
                return
            if (
                "completion_stalled" in self._degraded_causes
                and (
                    (callback is not None and not callback.done())
                    or bool(self._detached_completion_tokens)
                )
            ):
                return
            dispatcher = self._completion_dispatcher_task
            if (
                not self._completion_queue.empty()
                and (dispatcher is None or dispatcher.done())
                and not self._ensure_completion_dispatcher()
            ):
                self._set_degraded_cause("dispatcher_start_failure")
                return
            await asyncio.sleep(0)

    async def reset(self) -> None:
        """Invalidate queued/in-flight results while retaining a warm backend."""

        if self._closed:
            return
        reset_task = self._reset_task
        if reset_task is None or reset_task.done():
            self._resetting = True
            self._generation += 1
            self._set_degraded_cause("resetting")
            reset_task = asyncio.create_task(
                self._reset_impl(),
                name="speaker-shadow-reset",
            )
            self._reset_task = reset_task
            reset_task.add_done_callback(self._consume_reset_result)
        await asyncio.shield(reset_task)

    async def _reset_impl(self) -> None:
        owned_tokens = list(self._owned_candidate_tokens())
        observation = self._observation_task
        try:
            if observation is not None and not observation.done():
                cancelled = await self._cancel_callback_bounded(observation)
                if not cancelled:
                    self._detach_callback(observation)
            if self._observation_task is observation:
                self._observation_task = None
            await self._cancel_completion_dispatcher_bounded()
        finally:
            try:
                self._drain_completion_queue()
                self._drain_queue()
                owned_tokens.extend(self._owned_candidate_tokens())
                self._sweep_reset_tokens(owned_tokens)
                self._clear_buffers()
                self._retire_finalized_candidates()
                self._candidate_tokens.clear()
                self._load_failure_streak = 0
                self._next_load_attempt_at = 0.0
            finally:
                self._resetting = False
                self._clear_degraded_cause("resetting")

    async def close(self) -> None:
        """Stop accepting work and release every tracked resource exactly once.

        Blocking backend calls live only in the dedicated spawn process. If the
        serial worker misses its grace period, cleanup terminates that process
        before joining the worker, so close has a hard resource boundary.
        """

        cleanup = self._cleanup_task
        if not self._closed:
            self._closed = True
            self._generation += 1
            reset_task = self._reset_task
            cleanup = asyncio.create_task(
                self._close_after_reset(reset_task),
                name="speaker-shadow-cleanup",
            )
            self._cleanup_task = cleanup
            cleanup.add_done_callback(self._consume_cleanup_result)
        if cleanup is None:
            self._close_parent_factory()
            return
        await asyncio.shield(cleanup)

    async def _close_after_reset(
        self,
        reset_task: asyncio.Task[None] | None,
    ) -> None:
        if reset_task is not None and not reset_task.done():
            try:
                await asyncio.shield(reset_task)
            except asyncio.CancelledError:
                if not reset_task.done():
                    raise
            except Exception:
                # Reset already ran its mandatory local cleanup in ``finally``.
                pass
        self._cancel_observation_callback()
        self._drain_queue()
        self._drain_completion_queue()
        self._clear_buffers()
        self._finalized.clear()
        self._candidate_tokens.clear()
        worker = self._worker_task
        if worker is not None and not worker.done():
            self._queue.put_nowait(_STOP)
        dispatcher = self._completion_dispatcher_task
        if dispatcher is not None and not dispatcher.done():
            self._completion_queue.put_nowait(_COMPLETION_STOP)
        needs_cleanup = (
            worker is not None
            or dispatcher is not None
            or self._backend_host is not None
            or self._host_start_task is not None
            or self._observation_task is not None
            or self._completion_callback_task is not None
            or bool(self._detached_callback_tasks)
        )
        if needs_cleanup:
            await self._cleanup_after_worker(worker)
        else:
            self._close_parent_factory()

    def _ensure_worker(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            return False
        if self._worker_task is not None and not self._worker_task.done():
            return True
        worker = loop.create_task(
            self._run(), name="speaker-shadow-runtime"
        )
        worker.add_done_callback(self._consume_worker_result)
        self._worker_task = worker
        self._clear_degraded_cause("worker_start_failure")
        return True

    async def _run(self) -> None:
        while True:
            try:
                work_items = self._queue
                item = await asyncio.wait_for(
                    work_items.get(),
                    timeout=self._config.idle_unload_seconds,
                )
            except asyncio.TimeoutError:
                await self._unload_backend()
                if self._queue.empty():
                    return
                continue
            try:
                if item is _STOP:
                    return
                if isinstance(item, _CandidateFinished):
                    self._active_terminal_token = item.token
                    await self._process_finish(item)
                elif isinstance(item, _CandidateDeferred):
                    self._process_defer(item)
                elif isinstance(item, _CandidateActivated):
                    await self._process_activate(item)
                else:
                    assert isinstance(item, _AudioFrame)
                    await self._process_frame(item)
            except asyncio.CancelledError:
                if not self._closed and not self._resetting:
                    if isinstance(item, _CandidateFinished):
                        if item.token.finish_state is _FinishState.QUEUED:
                            self._abandon_terminal(item.candidate, token=item.token)
                        self._abandon_completion(item.token)
                    elif isinstance(
                        item,
                        (_AudioFrame, _CandidateDeferred, _CandidateActivated),
                    ):
                        self._drop_candidate(item.candidate, token=item.token)
                raise
            except Exception:
                # A defensive final fence: shadow errors never reach ASR.
                self._metrics.inference_failure_count += 1
                if isinstance(item, _CandidateFinished):
                    self._recover_failed_finish(item)
                elif isinstance(
                    item,
                    (_AudioFrame, _CandidateDeferred, _CandidateActivated),
                ):
                    self._finalize_candidate(
                        item.candidate,
                        "failed",
                        token=item.token,
                    )
            finally:
                if isinstance(item, _AudioFrame):
                    self._queued_pcm_bytes = max(
                        0,
                        self._queued_pcm_bytes - len(item.pcm16),
                    )
                    self._wipe_bytearray(item.pcm16)
                self._retire_queued_item(item)
                self._queue.task_done()
                if (
                    isinstance(item, _CandidateFinished)
                    and self._active_terminal_token is item.token
                ):
                    self._active_terminal_token = None
                item = None
            if self._queue.empty() and self._backend_host is None:
                return

    def _retire_queued_item(self, item: _QueueItem) -> None:
        if isinstance(item, _CandidateFinished):
            self._queued_terminal_count = max(0, self._queued_terminal_count - 1)
            if self._queued_terminal_count == 0:
                self._clear_degraded_cause("terminal_overflow")
            return
        if isinstance(item, (_AudioFrame, _CandidateDeferred, _CandidateActivated)):
            self._queued_data_item_count = max(0, self._queued_data_item_count - 1)

    def _process_defer(self, marker: _CandidateDeferred) -> None:
        if not self._identity_is_current(
            marker.generation,
            marker.candidate,
            marker.token,
        ):
            return
        marker.token.defer_processed = True

    async def _process_activate(self, marker: _CandidateActivated) -> None:
        token = marker.token
        if not self._identity_is_current(
            marker.generation,
            marker.candidate,
            token,
        ):
            return
        token.activation_queued = False
        if not token.deferred_requested or not token.defer_processed:
            self._drop_candidate(marker.candidate, token=token)
            return
        token.scoring_deferred = False
        buffer = self._buffers.get(marker.candidate)
        if buffer is None:
            return
        if buffer.token is not token:
            self._drop_candidate(marker.candidate, token=token)
            return
        if not await self._prewarm_candidate_backend(
            generation=marker.generation,
            candidate=marker.candidate,
            token=token,
            buffer=buffer,
        ):
            return
        await self._process_buffer_checkpoints(
            generation=marker.generation,
            candidate=marker.candidate,
            token=token,
            buffer=buffer,
        )

    async def _process_frame(self, frame: _AudioFrame) -> None:
        if frame.generation != self._generation:
            return
        if (
            frame.token.terminal_reason is not None
            or frame.candidate in self._finalized
            or self._candidate_was_evicted(
                frame.candidate,
                token=frame.token,
            )
        ):
            return
        buffer = self._buffers.get(frame.candidate)
        if buffer is None:
            if len(self._buffers) >= self._config.buffered_candidate_capacity:
                dropped_candidate, dropped_buffer = self._buffers.popitem(last=False)
                self._metrics.dropped_audio_ms += dropped_buffer.audio_ms
                self._wipe_bytearray(dropped_buffer.pcm16)
                self._finalize_candidate(
                    dropped_candidate,
                    "dropped",
                    token=dropped_buffer.token,
                )
            buffer = _CandidateBuffer(
                token=frame.token,
                sample_rate_hz=frame.sample_rate_hz,
                pcm16=bytearray(),
            )
            self._buffers[frame.candidate] = buffer
            self._metrics.started_candidate_count += 1
        elif buffer.token is not frame.token:
            return
        elif buffer.sample_rate_hz != frame.sample_rate_hz:
            self._buffers.pop(frame.candidate, None)
            self._wipe_bytearray(buffer.pcm16)
            self._finalize_candidate(
                frame.candidate,
                "failed",
                token=frame.token,
            )
            return
        else:
            self._buffers.move_to_end(frame.candidate)

        maximum_samples = (
            buffer.sample_rate_hz * self._config.maximum_audio_ms // 1_000
        )
        allowed_samples = min(
            frame.sample_count,
            maximum_samples - buffer.sample_count,
        )
        if allowed_samples > 0:
            buffer.pcm16.extend(frame.pcm16[: allowed_samples * 2])
            buffer.sample_count += allowed_samples
        if not await self._prewarm_candidate_backend(
            generation=frame.generation,
            candidate=frame.candidate,
            token=frame.token,
            buffer=buffer,
        ):
            return
        if frame.token.scoring_deferred:
            return
        await self._process_buffer_checkpoints(
            generation=frame.generation,
            candidate=frame.candidate,
            token=frame.token,
            buffer=buffer,
        )

    async def _prewarm_candidate_backend(
        self,
        *,
        generation: int,
        candidate: SpeakerShadowCandidateKey,
        token: _CandidateToken,
        buffer: _CandidateBuffer,
    ) -> bool:
        if (
            buffer.sample_count > 0
            and not buffer.backend_prewarm_attempted
            and candidate.scope in self._config.backend_prewarm_scopes
        ):
            buffer.backend_prewarm_attempted = True
            backend_host = await self._ensure_backend()
            if not self._identity_is_current(
                generation,
                candidate,
                token,
            ):
                self._metrics.stale_result_count += 1
                retained_buffer = self._buffers.get(candidate)
                if retained_buffer is buffer:
                    self._buffers.pop(candidate, None)
                    self._wipe_bytearray(buffer.pcm16)
                return False
            if backend_host is None:
                self._mark_backend_degraded()
                retained_buffer = self._buffers.get(candidate)
                if retained_buffer is buffer:
                    self._buffers.pop(candidate, None)
                    self._wipe_bytearray(buffer.pcm16)
                self._finalize_candidate(
                    candidate,
                    "failed",
                    token=token,
                )
                return False
        return True

    async def _process_buffer_checkpoints(
        self,
        *,
        generation: int,
        candidate: SpeakerShadowCandidateKey,
        token: _CandidateToken,
        buffer: _CandidateBuffer,
    ) -> None:
        explicit_checkpoints = self._config.observation_checkpoints_ms
        checkpoints = explicit_checkpoints or (self._config.minimum_audio_ms,)
        while buffer.next_checkpoint_index < len(checkpoints):
            checkpoint_index = buffer.next_checkpoint_index
            checkpoint_ms = checkpoints[checkpoint_index]
            checkpoint_samples = math.ceil(
                buffer.sample_rate_hz * checkpoint_ms / 1_000
            )
            if buffer.sample_count < checkpoint_samples:
                return

            terminal = checkpoint_index == len(checkpoints) - 1
            buffer.next_checkpoint_index += 1
            score_sample_count = (
                buffer.sample_count
                if explicit_checkpoints is None
                else checkpoint_samples
            )
            candidate_pcm = bytearray(buffer.pcm16[: score_sample_count * 2])
            if terminal:
                self._buffers.pop(candidate, None)
                self._wipe_bytearray(buffer.pcm16)
            try:
                would_block = await self._evaluate_candidate(
                    generation=generation,
                    candidate=candidate,
                    token=token,
                    pcm16=candidate_pcm,
                    sample_rate_hz=buffer.sample_rate_hz,
                    audio_ms=(
                        buffer.audio_ms
                        if explicit_checkpoints is None
                        else checkpoint_ms
                    ),
                    checkpoint_ms=(
                        checkpoint_ms
                        if explicit_checkpoints is not None
                        else None
                    ),
                    terminal=terminal,
                )
            except BaseException:
                retained_buffer = self._buffers.get(candidate)
                if retained_buffer is buffer:
                    self._buffers.pop(candidate, None)
                    self._wipe_bytearray(buffer.pcm16)
                raise
            finally:
                self._wipe_bytearray(candidate_pcm)
            if not self._identity_is_current(
                generation,
                candidate,
                token,
            ):
                retained_buffer = self._buffers.get(candidate)
                if retained_buffer is buffer:
                    self._buffers.pop(candidate, None)
                    self._wipe_bytearray(buffer.pcm16)
                return
            if (
                not terminal
                and candidate.scope in self._config.completion_confirmation_scopes
                and buffer.next_checkpoint_index < len(checkpoints)
            ):
                buffer.completion_confirmation_checkpoint_ms = (
                    checkpoint_ms if would_block else None
                )

    async def _evaluate_candidate(
        self,
        *,
        generation: int,
        candidate: SpeakerShadowCandidateKey,
        token: _CandidateToken,
        pcm16: bytearray,
        sample_rate_hz: int,
        audio_ms: int,
        checkpoint_ms: int | None,
        terminal: bool,
        observation_kind: Literal[
            "checkpoint", "completion_confirmation"
        ] = "checkpoint",
    ) -> bool | None:
        self._active_evaluation = (generation, candidate)
        self._active_evaluation_terminal = terminal
        self._active_pcm_bytes = len(pcm16)
        try:
            backend_host = await self._ensure_backend()
            if not self._identity_is_current(generation, candidate, token):
                self._metrics.stale_result_count += 1
                return
            if backend_host is None:
                self._mark_backend_degraded()
                self._finalize_candidate(candidate, "failed", token=token)
                return
            started = time.perf_counter()
            try:
                similarity = float(
                    await backend_host.score(
                        pcm16,
                        timeout_seconds=self._config.backend_score_timeout_seconds,
                    )
                )
                if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
                    raise ValueError(
                        "speaker cosine similarity must be within [-1, 1]"
                    )
                self._mark_backend_recovered()
            except asyncio.CancelledError:
                raise
            except _BackendHostTimeout:
                self._metrics.backend_timeout_count += 1
                self._discard_backend_host(backend_host)
                self._metrics.inference_failure_count += 1
                self._mark_backend_degraded()
                if self._identity_is_current(generation, candidate, token):
                    self._finalize_candidate(candidate, "failed", token=token)
                return
            except Exception:
                if not backend_host.alive:
                    self._discard_backend_host(backend_host)
                self._metrics.inference_failure_count += 1
                self._mark_backend_degraded()
                if self._identity_is_current(generation, candidate, token):
                    self._finalize_candidate(candidate, "failed", token=token)
                return
            finally:
                self._metrics.inference_ms += int(
                    (time.perf_counter() - started) * 1_000
                )
            if not self._identity_is_current(generation, candidate, token):
                self._metrics.stale_result_count += 1
                return

            would_block = tuple(
                (threshold, similarity < threshold)
                for threshold in self._config.similarity_thresholds
            )
            blocked_at_any_threshold = any(
                blocked for _, blocked in would_block
            )
            token.last_checkpoint_ms = (
                checkpoint_ms
                if checkpoint_ms is not None
                else self._config.minimum_audio_ms
            )
            if terminal:
                self._finalize_candidate(candidate, "scored", token=token)
                self._metrics.evaluated_candidate_count += 1
            if blocked_at_any_threshold:
                self._metrics.would_block_count += 1
            for threshold, blocked in would_block:
                if blocked:
                    self._would_block_counts[threshold] += 1
            callback = self._on_observation
            if callback is None:
                return blocked_at_any_threshold
            existing_callback_task = self._observation_task
            if existing_callback_task is not None:
                if not existing_callback_task.done():
                    self._metrics.callback_failure_count += 1
                    return blocked_at_any_threshold
                self._consume_callback_result(existing_callback_task)
            observation = SpeakerShadowObservation(
                candidate=candidate,
                similarity=similarity,
                would_block=would_block,
                audio_ms=audio_ms,
                checkpoint_ms=checkpoint_ms,
                observation_kind=observation_kind,
            )
            callback_task = asyncio.create_task(
                callback(observation),
                name="speaker-shadow-observation",
            )
            self._observation_task = callback_task
            callback_task.add_done_callback(self._consume_callback_result)
            try:
                done, _ = await asyncio.wait(
                    {callback_task},
                    timeout=self._config.callback_timeout_seconds,
                )
            except asyncio.CancelledError:
                cancelled = await self._cancel_callback_bounded(callback_task)
                if not cancelled:
                    self._detach_callback(callback_task)
                if self._observation_task is callback_task:
                    self._observation_task = None
                raise
            if not done:
                self._metrics.callback_failure_count += 1
                cancelled = await self._cancel_callback_bounded(callback_task)
                if not cancelled:
                    self._detach_callback(callback_task)
                    if self._observation_task is callback_task:
                        self._observation_task = None
                return blocked_at_any_threshold
            try:
                callback_task.result()
            except asyncio.CancelledError:
                self._metrics.stale_result_count += 1
            except Exception:
                self._metrics.callback_failure_count += 1
            else:
                if (
                    observation_kind == "checkpoint"
                    and checkpoint_ms is not None
                    and self._identity_is_current(
                        generation,
                        candidate,
                        token,
                    )
                ):
                    token.last_delivered_checkpoint_ms = checkpoint_ms
            return blocked_at_any_threshold
        finally:
            if self._active_evaluation == (generation, candidate):
                self._active_evaluation = None
                self._active_evaluation_terminal = False
                self._active_pcm_bytes = 0

    async def _ensure_backend(self) -> _BackendProcessHost | None:
        existing_host = self._backend_host
        if (
            existing_host is not None
            and existing_host.alive
            and existing_host.loaded
        ):
            return existing_host
        if existing_host is not None:
            self._discard_backend_host(existing_host)
        if time.monotonic() < self._next_load_attempt_at:
            self._metrics.load_retry_suppressed_count += 1
            return None
        factory = self._backend_factory
        if factory is None:
            return None

        started = time.perf_counter()
        start_task = asyncio.create_task(
            asyncio.to_thread(
                _BackendProcessHost.create_started,
                factory=factory,
                terminate_timeout_seconds=(
                    self._config.process_terminate_timeout_seconds
                ),
            ),
            name="speaker-shadow-host-start",
        )
        self._host_start_task = start_task
        host: _BackendProcessHost | None = None
        try:
            try:
                host = await asyncio.shield(start_task)
            except asyncio.CancelledError:
                # ``to_thread`` cannot stop an in-progress ``Process.start``.
                # Keep ownership across repeated worker cancellations so a host
                # that finishes starting after shutdown is always retrieved and
                # terminated by the outer cancellation handler.
                while not start_task.done():
                    try:
                        await asyncio.shield(start_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if start_task.done() and not start_task.cancelled():
                    try:
                        host = start_task.result()
                    except Exception:
                        host = None
                raise
            available = await host.load(
                timeout_seconds=self._config.backend_load_timeout_seconds
            )
            if not available:
                await self._close_host(host)
                self._record_load_failure()
                return None
        except asyncio.CancelledError:
            if host is not None:
                await self._terminate_host(host)
            raise
        except _BackendHostTimeout:
            self._metrics.backend_timeout_count += 1
            if host is not None:
                self._record_host_termination(host)
            self._record_load_failure()
            return None
        except Exception:
            if host is not None:
                await self._close_host(host)
            self._record_load_failure()
            return None
        finally:
            if self._host_start_task is start_task:
                self._host_start_task = None
            self._metrics.load_ms += int(
                (time.perf_counter() - started) * 1_000
            )
        assert host is not None
        self._backend_host = host
        self._load_failure_streak = 0
        self._next_load_attempt_at = 0.0
        self._metrics.load_count += 1
        self._mark_backend_recovered()
        return host

    def _record_load_failure(self) -> None:
        self._load_failure_streak += 1
        retry_seconds = self._config.load_retry_initial_seconds
        for _ in range(self._load_failure_streak - 1):
            if retry_seconds >= self._config.load_retry_max_seconds:
                break
            retry_seconds = min(
                self._config.load_retry_max_seconds,
                retry_seconds * 2,
            )
        self._next_load_attempt_at = time.monotonic() + retry_seconds
        self._metrics.load_failure_count += 1
        self._mark_backend_degraded()

    def _mark_backend_degraded(self) -> None:
        self._set_degraded_cause("backend_unavailable")

    def _mark_backend_recovered(self) -> None:
        self._clear_degraded_cause("backend_unavailable")

    def _set_degraded_cause(self, cause: _DegradedCause) -> None:
        if cause in self._degraded_causes:
            return
        notify = not self._degraded_causes
        self._degraded_causes.add(cause)
        if not notify:
            return
        self._metrics.delivery_degraded_count += 1
        callback = self._on_backend_degraded
        if callback is None:
            return
        try:
            callback()
        except Exception:
            self._metrics.callback_failure_count += 1

    def _clear_degraded_cause(self, cause: _DegradedCause) -> None:
        if cause not in self._degraded_causes:
            return
        self._degraded_causes.discard(cause)
        if self._degraded_causes:
            return
        if self._closed:
            return
        callback = self._on_backend_recovered
        if callback is None:
            return
        try:
            callback()
        except Exception:
            self._metrics.callback_failure_count += 1

    async def _unload_backend(self) -> bool:
        host = self._backend_host
        if host is None:
            return True
        closed = await self._close_host(host)
        if self._backend_host is host:
            self._backend_host = None
        if closed:
            self._metrics.unload_count += 1
        else:
            self._metrics.unload_failure_count += 1
        self._load_failure_streak = 0
        self._next_load_attempt_at = 0.0
        return closed

    async def _close_host(self, host: _BackendProcessHost) -> bool:
        try:
            closed = await host.close(
                timeout_seconds=self._config.backend_close_timeout_seconds
            )
        except Exception:
            await self._terminate_host(host)
            return False
        if host.timed_out:
            self._metrics.backend_timeout_count += 1
        self._record_host_termination(host)
        return closed

    async def _terminate_host(self, host: _BackendProcessHost) -> None:
        try:
            await host.terminate()
        except Exception:
            self._metrics.unload_failure_count += 1
        self._record_host_termination(host)

    def _discard_backend_host(self, host: _BackendProcessHost) -> None:
        if self._backend_host is host:
            self._backend_host = None
        self._record_host_termination(host)

    def _record_host_termination(self, host: _BackendProcessHost) -> None:
        if host.was_terminated:
            self._metrics.backend_process_termination_count += 1
            host.was_terminated = False

    async def _cleanup_after_worker(
        self,
        worker: asyncio.Task[None] | None,
    ) -> None:
        try:
            if worker is not None and not worker.done():
                done, _ = await asyncio.wait(
                    {worker},
                    timeout=self._config.shutdown_grace_seconds,
                )
                if not done:
                    self._metrics.shutdown_timeout_count += 1
                    cancellation_timeout = (
                        self._config.process_terminate_timeout_seconds * 2
                        + _HOST_POLL_INTERVAL_SECONDS
                    )
                    for attempt in range(2):
                        worker.cancel()
                        done, _ = await asyncio.wait(
                            {worker},
                            timeout=cancellation_timeout,
                        )
                        if done:
                            break
                        if attempt == 0:
                            host, self._backend_host = self._backend_host, None
                            if host is not None:
                                await self._terminate_host(host)
                    if not worker.done():
                        # A thread already inside ``Process.start`` cannot be
                        # cancelled. Keep cleanup attached until the worker
                        # retrieves and terminates any host it eventually
                        # returns; close must not leave that ownership orphaned.
                        await asyncio.wait({worker})
            if worker is not None and worker.done():
                self._consume_worker_result(worker)
            observation = self._observation_task
            if observation is not None:
                cancelled = await self._cancel_callback_bounded(observation)
                if not cancelled:
                    self._detach_callback(observation)
                if self._observation_task is observation:
                    self._observation_task = None
            await self._cancel_completion_dispatcher_bounded()
            await self._cancel_detached_callbacks_bounded()
        finally:
            try:
                await self._unload_backend()
            finally:
                self._close_parent_factory()

    def _close_parent_factory(self) -> None:
        """Release the parent-owned profile exactly once without exposing it."""

        if self._factory_closed:
            return
        self._factory_closed = True
        close_factory = getattr(self._backend_factory, "close", None)
        if not callable(close_factory):
            return
        try:
            # Factory.close is a parent-memory wipe contract. It must be
            # idempotent and non-blocking; running a copy elsewhere would not
            # clear the parent-owned profile or embedding.
            close_factory()
        except Exception:
            self._metrics.unload_failure_count += 1

    async def _process_finish(self, marker: _CandidateFinished) -> None:
        if marker.generation != self._generation:
            self._abandon_terminal(marker.candidate, token=marker.token)
            return
        if marker.token.finish_state is _FinishState.ABANDONED:
            self._abandon_completion(marker.token)
            return
        if marker.token.finish_state is _FinishState.PROCESSED:
            return
        if (
            marker.token.finish_state is not _FinishState.QUEUED
            and self._candidate_was_evicted(
                marker.candidate,
                token=marker.token,
            )
        ):
            return
        # Only an accepted QUEUED marker outranks the tombstone watermark.
        self._mark_finish_processed(marker.token)
        if marker.token.terminal_reason is not None:
            self._record_token_finish(marker.token)
            self._enqueue_completion(
                marker,
                terminal_reason=marker.token.terminal_reason,
            )
            return
        finalized = self._finalized.get(marker.candidate)
        if finalized is not None:
            self._record_finish(marker.candidate, finalized)
            completion_token = finalized.token or marker.token
            self._enqueue_completion(
                _CandidateFinished(
                    marker.generation,
                    marker.candidate,
                    completion_token,
                ),
                terminal_reason=finalized.terminal_reason,
            )
            return
        buffer = self._buffers.get(marker.candidate)
        explicit_checkpoints = self._config.observation_checkpoints_ms
        confirmation_checkpoint_ms = (
            buffer.completion_confirmation_checkpoint_ms
            if buffer is not None
            else None
        )
        should_confirm = (
            buffer is not None
            and buffer.token is marker.token
            and self._identity_is_current(
                marker.generation,
                marker.candidate,
                marker.token,
            )
            and marker.candidate.scope
            in self._config.completion_confirmation_scopes
            and explicit_checkpoints is not None
            and confirmation_checkpoint_ms is not None
            and 0 < buffer.next_checkpoint_index < len(explicit_checkpoints)
            and explicit_checkpoints[buffer.next_checkpoint_index - 1]
            == confirmation_checkpoint_ms
            and buffer.audio_ms > confirmation_checkpoint_ms
            and buffer.audio_ms
            < explicit_checkpoints[buffer.next_checkpoint_index]
        )
        if should_confirm:
            assert buffer is not None
            assert confirmation_checkpoint_ms is not None
            candidate_pcm = bytearray(buffer.pcm16)
            audio_ms = buffer.audio_ms
            sample_rate_hz = buffer.sample_rate_hz
            try:
                try:
                    await self._evaluate_candidate(
                        generation=marker.generation,
                        candidate=marker.candidate,
                        token=marker.token,
                        pcm16=candidate_pcm,
                        sample_rate_hz=sample_rate_hz,
                        audio_ms=audio_ms,
                        checkpoint_ms=confirmation_checkpoint_ms,
                        terminal=True,
                        observation_kind="completion_confirmation",
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if self._identity_is_current(
                        marker.generation,
                        marker.candidate,
                        marker.token,
                    ):
                        self._finalize_candidate(
                            marker.candidate,
                            "failed",
                            token=marker.token,
                        )
            finally:
                self._wipe_bytearray(candidate_pcm)
                retained_buffer = self._buffers.get(marker.candidate)
                if retained_buffer is buffer:
                    self._buffers.pop(marker.candidate, None)
                self._wipe_bytearray(buffer.pcm16)

            if marker.generation != self._generation or self._closed:
                self._abandon_completion(marker.token)
                return
            terminal_reason = marker.token.terminal_reason
            if terminal_reason is None:
                self._finalize_candidate(
                    marker.candidate,
                    "failed",
                    token=marker.token,
                )
                terminal_reason = marker.token.terminal_reason or "failed"
            self._record_token_finish(marker.token)
            self._enqueue_completion(
                marker,
                terminal_reason=terminal_reason,
            )
            return

        buffer = self._buffers.pop(marker.candidate, None)
        terminal_reason: SpeakerShadowTerminalReason = "insufficient"
        if buffer is not None:
            if buffer.next_checkpoint_index > 0:
                terminal_reason = "scored"
                self._metrics.evaluated_candidate_count += 1
            self._wipe_bytearray(buffer.pcm16)
        self._finalize_candidate(
            marker.candidate,
            terminal_reason,
            token=marker.token,
        )
        self._record_token_finish(marker.token)
        self._enqueue_completion(
            marker,
            terminal_reason=terminal_reason,
        )

    def _recover_failed_finish(self, marker: _CandidateFinished) -> None:
        """Convert a consumed finish fault into one explicit failed completion."""

        if marker.generation != self._generation or self._closed:
            self._abandon_completion(marker.token)
            return
        self._mark_finish_processed(marker.token)
        if marker.token.terminal_reason is None:
            self._finalize_candidate(
                marker.candidate,
                "failed",
                token=marker.token,
            )
        terminal_reason = marker.token.terminal_reason or "failed"
        self._record_token_finish(marker.token)
        self._enqueue_completion(marker, terminal_reason=terminal_reason)

    def _enqueue_completion(
        self,
        marker: _CandidateFinished,
        *,
        terminal_reason: SpeakerShadowTerminalReason,
    ) -> bool:
        """Accept one terminal notice into the bounded ordered outbox."""

        token = marker.token
        if (
            marker.generation != self._generation
            or self._closed
        ):
            self._abandon_completion(token)
            return False
        if token.completion_state in {
            _CompletionState.QUEUED,
            _CompletionState.DISPATCHED,
            _CompletionState.ATTEMPTED,
        }:
            return True
        if token.completion_state is _CompletionState.ABANDONED:
            return False
        if self._completion_queue.qsize() >= self._config.completion_queue_capacity:
            self._metrics.completion_overflow_count += 1
            self._set_degraded_cause("completion_overflow")
            self._abandon_completion(token)
            return False
        if not self._ensure_completion_dispatcher():
            self._set_degraded_cause("dispatcher_start_failure")
            self._abandon_completion(token)
            return False
        completion = SpeakerShadowCompletion(
            candidate=marker.candidate,
            terminal_reason=terminal_reason,
            last_checkpoint_ms=token.last_checkpoint_ms,
        )
        envelope = _CompletionEnvelope(
            generation=marker.generation,
            candidate=marker.candidate,
            token=token,
            completion=completion,
        )
        try:
            self._completion_queue.put_nowait(envelope)
        except asyncio.QueueFull:
            self._metrics.completion_overflow_count += 1
            self._set_degraded_cause("completion_overflow")
            self._abandon_completion(token)
            return False
        token.completion_state = _CompletionState.QUEUED
        self._metrics.completion_count += 1
        if token.last_checkpoint_ms is None:
            self._metrics.completion_before_first_checkpoint_count += 1
        else:
            self._metrics.completion_after_first_checkpoint_count += 1
        return True

    def _ensure_completion_dispatcher(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            return False
        dispatcher = self._completion_dispatcher_task
        if dispatcher is not None and not dispatcher.done():
            return True
        dispatcher = loop.create_task(
            self._run_completion_dispatcher(),
            name="speaker-shadow-completion-dispatcher",
        )
        dispatcher.add_done_callback(self._consume_completion_dispatcher_result)
        self._completion_dispatcher_task = dispatcher
        self._clear_degraded_cause("dispatcher_start_failure")
        return True

    async def _run_completion_dispatcher(self) -> None:
        while True:
            item = await self._completion_queue.get()
            try:
                if item is _COMPLETION_STOP:
                    return
                assert isinstance(item, _CompletionEnvelope)
                self._completion_dispatch_in_progress = True
                try:
                    await self._dispatch_completion(item)
                except asyncio.CancelledError:
                    if item.token.completion_state is _CompletionState.QUEUED:
                        self._abandon_completion(item.token)
                    raise
                except Exception:
                    self._set_degraded_cause("dispatcher_start_failure")
                    self._abandon_completion(item.token)
            finally:
                self._completion_dispatch_in_progress = False
                self._completion_queue.task_done()
            if self._completion_queue.empty():
                self._clear_degraded_cause("completion_overflow")
                if (
                    self._completion_callback_task is None
                    or self._completion_callback_task.done()
                ):
                    self._clear_degraded_cause("completion_stalled")

    async def _dispatch_completion(self, envelope: _CompletionEnvelope) -> None:
        token = envelope.token
        if (
            envelope.generation != self._generation
            or self._closed
            or token.completion_state is not _CompletionState.QUEUED
        ):
            self._abandon_completion(token)
            return

        await self._wait_for_detached_completion_callbacks()
        if (
            envelope.generation != self._generation
            or self._closed
            or token.completion_state is not _CompletionState.QUEUED
        ):
            self._abandon_completion(token)
            return

        observation_task = self._observation_task
        if observation_task is not None and not observation_task.done():
            observation_task.cancel()
            self._detach_callback(observation_task)
            self._observation_task = None

        callback = self._on_completion
        if callback is None:
            token.completion_state = _CompletionState.ATTEMPTED
            self._metrics.completion_attempted_count += 1
            return
        try:
            callback_task = asyncio.create_task(
                callback(envelope.completion),
                name="speaker-shadow-completion",
            )
        except Exception:
            self._set_degraded_cause("dispatcher_start_failure")
            raise
        self._completion_callback_task = callback_task
        self._completion_callback_token = token
        token.completion_state = _CompletionState.DISPATCHED
        self._metrics.completion_dispatched_count += 1
        callback_task.add_done_callback(self._consume_callback_result)
        try:
            done, _ = await asyncio.wait(
                {callback_task},
                timeout=self._config.callback_timeout_seconds,
            )
            if not done:
                self._metrics.callback_failure_count += 1
                self._metrics.completion_callback_failure_count += 1
                self._metrics.completion_stall_count += 1
                self._set_degraded_cause("completion_stalled")
                cancelled = await self._cancel_callback_bounded(callback_task)
                if not cancelled:
                    try:
                        await asyncio.shield(callback_task)
                    except asyncio.CancelledError:
                        self._detach_callback(
                            callback_task,
                            completion_token=token,
                        )
                        raise
            self._consume_completion_callback_result(callback_task)
        except asyncio.CancelledError:
            cancelled = await self._cancel_callback_bounded(callback_task)
            if not cancelled:
                self._detach_callback(
                    callback_task,
                    completion_token=token,
                )
            else:
                self._consume_completion_callback_result(callback_task)
            raise
        finally:
            if token.completion_state is _CompletionState.DISPATCHED and callback_task.done():
                token.completion_state = _CompletionState.ATTEMPTED
                self._metrics.completion_attempted_count += 1
            if self._completion_callback_task is callback_task and callback_task.done():
                self._completion_callback_task = None
                self._completion_callback_token = None

    def _consume_completion_callback_result(
        self,
        callback_task: asyncio.Task[None],
    ) -> None:
        try:
            callback_task.result()
        except asyncio.CancelledError:
            self._metrics.stale_result_count += 1
        except Exception:
            self._metrics.callback_failure_count += 1
            self._metrics.completion_callback_failure_count += 1

    def _abandon_completion(self, token: _CandidateToken) -> None:
        if token.completion_state in {
            _CompletionState.DISPATCHED,
            _CompletionState.ATTEMPTED,
            _CompletionState.ABANDONED,
        }:
            return
        token.completion_state = _CompletionState.ABANDONED
        self._metrics.completion_abandoned_count += 1

    def _detach_callback(
        self,
        task: asyncio.Task[None],
        *,
        completion_token: _CandidateToken | None = None,
    ) -> None:
        if completion_token is not None:
            if "completion_stalled" not in self._degraded_causes:
                self._metrics.completion_stall_count += 1
                self._set_degraded_cause("completion_stalled")
            self._detached_completion_tokens[task] = completion_token
            if self._completion_callback_task is task:
                self._completion_callback_task = None
                self._completion_callback_token = None
        if self._observation_task is task:
            self._observation_task = None
        if task in self._detached_callback_tasks:
            return
        self._detached_callback_tasks.add(task)
        task.add_done_callback(self._consume_detached_callback_result)

    async def _wait_for_detached_completion_callbacks(self) -> None:
        while self._detached_completion_tokens:
            tasks = tuple(self._detached_completion_tokens)
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                self._consume_detached_callback_result(task)

    def _drop_candidate(
        self,
        candidate: SpeakerShadowCandidateKey,
        *,
        token: _CandidateToken | None = None,
    ) -> None:
        buffer = self._buffers.pop(candidate, None)
        if buffer is not None:
            self._metrics.dropped_audio_ms += buffer.audio_ms
            self._wipe_bytearray(buffer.pcm16)
            if token is None:
                token = buffer.token
        self._finalize_candidate(
            candidate,
            "dropped",
            token=token,
        )

    def _finalize_candidate(
        self,
        candidate: SpeakerShadowCandidateKey,
        terminal_reason: SpeakerShadowTerminalReason,
        *,
        token: _CandidateToken | None = None,
    ) -> None:
        if token is None:
            token = self._candidate_tokens.get(candidate)
        if token is not None and token.terminal_reason is not None:
            return
        if token is not None:
            token.terminal_reason = terminal_reason
            if self._candidate_tokens.get(candidate) is token:
                self._candidate_tokens.pop(candidate, None)
        previous = self._finalized.pop(candidate, None)
        if previous is not None:
            self._finalized[candidate] = _FinalizedCandidate(
                finish_state=(
                    _FinishState.PROCESSED
                    if previous.finish_seen
                    else token.finish_state if token is not None else previous.finish_state
                ),
                terminal_reason=previous.terminal_reason,
                token=previous.token or token,
            )
            return
        self._finalized[candidate] = _FinalizedCandidate(
            finish_state=(token.finish_state if token is not None else _FinishState.OPEN),
            terminal_reason=terminal_reason,
            token=token,
        )
        counter_name = f"{terminal_reason}_candidate_count"
        setattr(
            self._metrics,
            counter_name,
            getattr(self._metrics, counter_name) + 1,
        )
        while len(self._finalized) > self._config.finalized_candidate_capacity:
            evicted_candidate, _ = self._finalized.popitem(last=False)
            self._record_evicted_candidate(evicted_candidate)

    def _candidate_was_evicted(
        self,
        candidate: SpeakerShadowCandidateKey,
        *,
        token: _CandidateToken | None = None,
    ) -> bool:
        current_token = self._candidate_tokens.get(candidate)
        buffer = self._buffers.get(candidate)
        finalized = self._finalized.get(candidate)
        if token is None and (
            current_token is not None
            or buffer is not None
            or self._active_evaluation == (self._generation, candidate)
        ):
            return False
        if token is not None and (
            current_token is token
            or (buffer is not None and buffer.token is token)
            or (finalized is not None and finalized.token is token)
        ):
            return False
        finalized_through = self._finalized_through.get(candidate.scope)
        if finalized_through is None:
            return False
        return (
            candidate.detector_epoch,
            candidate.shadow_generation,
        ) <= finalized_through

    def _record_evicted_candidate(
        self,
        candidate: SpeakerShadowCandidateKey,
    ) -> None:
        position = (candidate.detector_epoch, candidate.shadow_generation)
        previous = self._finalized_through.get(candidate.scope)
        if previous is None or position > previous:
            self._finalized_through[candidate.scope] = position

    def _retire_finalized_candidates(self) -> None:
        for candidate in self._finalized:
            self._record_evicted_candidate(candidate)
        self._finalized.clear()

    @staticmethod
    def _threshold_metric_key(threshold: float) -> str:
        # ``repr`` is the shortest round-trippable float representation, so
        # distinct configured thresholds cannot collapse into one metric key.
        suffix = (
            repr(threshold)
            .replace("-", "m")
            .replace("+", "p")
            .replace(".", "_")
        )
        return f"would_block_at_{suffix}_count"

    def _record_finish(
        self,
        candidate: SpeakerShadowCandidateKey,
        finalized: _FinalizedCandidate,
    ) -> None:
        if finalized.finish_seen:
            return
        if finalized.token is not None:
            finalized.token.finish_state = _FinishState.PROCESSED
        self._finalized.pop(candidate, None)
        self._finalized[candidate] = _FinalizedCandidate(
            finish_state=_FinishState.PROCESSED,
            terminal_reason=finalized.terminal_reason,
            token=finalized.token,
        )

    def _mark_finish_processed(self, token: _CandidateToken) -> None:
        if token.finish_state is _FinishState.PROCESSED:
            return
        if token.finish_state is _FinishState.ABANDONED:
            return
        token.finish_state = _FinishState.PROCESSED
        self._metrics.finished_candidate_count += 1

    def _record_token_finish(self, token: _CandidateToken) -> None:
        self._mark_finish_processed(token)
        if self._candidate_tokens.get(token.candidate) is token:
            self._candidate_tokens.pop(token.candidate, None)
        finalized = self._finalized.get(token.candidate)
        if finalized is not None:
            self._finalized.pop(token.candidate, None)
            self._finalized[token.candidate] = _FinalizedCandidate(
                finish_state=_FinishState.PROCESSED,
                terminal_reason=finalized.terminal_reason,
                token=finalized.token or token,
            )

    def _identity_is_current(
        self,
        generation: int,
        candidate: SpeakerShadowCandidateKey,
        token: _CandidateToken,
    ) -> bool:
        return (
            generation == self._generation
            and not self._closed
            and not self._resetting
            and candidate not in self._finalized
            and token.terminal_reason is None
            and self._candidate_tokens.get(candidate) is token
        )

    def _owned_candidate_tokens(self) -> tuple[_CandidateToken, ...]:
        tokens: list[_CandidateToken] = []
        seen: set[int] = set()

        def append(token: _CandidateToken | None) -> None:
            if token is None or id(token) in seen:
                return
            seen.add(id(token))
            tokens.append(token)

        for token in self._candidate_tokens.values():
            append(token)
        for buffer in self._buffers.values():
            append(buffer.token)
        for finalized in self._finalized.values():
            append(finalized.token)
        append(self._active_terminal_token)
        append(self._completion_callback_token)
        for token in self._detached_completion_tokens.values():
            append(token)
        return tuple(tokens)

    def _sweep_reset_tokens(self, tokens: list[_CandidateToken]) -> None:
        seen: set[int] = set()
        for token in tokens:
            if id(token) in seen:
                continue
            seen.add(id(token))
            if token.finish_state is _FinishState.QUEUED:
                self._abandon_terminal(token.candidate, token=token)
            elif token.finish_state in {
                _FinishState.PROCESSED,
                _FinishState.ABANDONED,
            }:
                self._abandon_completion(token)

    def _drain_queue(self, *, abandon_data_candidates: bool = False) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                if isinstance(item, _AudioFrame):
                    self._queued_pcm_bytes = max(
                        0,
                        self._queued_pcm_bytes - len(item.pcm16),
                    )
                    self._wipe_bytearray(item.pcm16)
                    if abandon_data_candidates:
                        self._drop_candidate(item.candidate, token=item.token)
                elif isinstance(item, (_CandidateDeferred, _CandidateActivated)):
                    if abandon_data_candidates:
                        self._drop_candidate(item.candidate, token=item.token)
                elif isinstance(item, _CandidateFinished):
                    self._abandon_terminal(item.candidate, token=item.token)
                self._retire_queued_item(item)
                self._queue.task_done()

    def _drain_completion_queue(self) -> None:
        while True:
            try:
                item = self._completion_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, _CompletionEnvelope):
                self._abandon_completion(item.token)
            self._completion_queue.task_done()
        self._clear_degraded_cause("completion_overflow")

    def _retained_pcm_bytes(self) -> int:
        host_pcm_bytes = (
            self._backend_host.pcm_bytes_in_use
            if self._backend_host is not None
            else 0
        )
        return (
            self._queued_pcm_bytes
            + sum(len(buffer.pcm16) for buffer in self._buffers.values())
            + self._active_pcm_bytes
            + host_pcm_bytes
        )

    def _clear_buffers(self) -> None:
        for buffer in self._buffers.values():
            self._wipe_bytearray(buffer.pcm16)
        self._buffers.clear()

    @staticmethod
    def _wipe_bytearray(value: bytearray) -> None:
        value[:] = b"\x00" * len(value)

    def _cancel_observation_callback(self) -> None:
        callback_task = self._observation_task
        if callback_task is not None and not callback_task.done():
            callback_task.cancel()

    async def _cancel_callback_bounded(
        self,
        task: asyncio.Task[None],
    ) -> bool:
        callback_task = task
        for _ in range(2):
            if callback_task.done():
                self._consume_callback_result(callback_task)
                return True
            callback_task.cancel()
            done, _ = await asyncio.wait(
                {callback_task},
                timeout=self._config.callback_timeout_seconds,
            )
            if done:
                self._consume_callback_result(callback_task)
                return True
        return False

    async def _cancel_completion_dispatcher_bounded(self) -> bool:
        dispatcher = self._completion_dispatcher_task
        if dispatcher is None:
            callback = self._completion_callback_task
            if callback is None:
                return True
            token = self._completion_callback_token
            cancelled = await self._cancel_callback_bounded(callback)
            if not cancelled:
                self._detach_callback(
                    callback,
                    completion_token=token,
                )
            elif token is not None:
                self._consume_completion_callback_result(callback)
                if token.completion_state is _CompletionState.DISPATCHED:
                    token.completion_state = _CompletionState.ATTEMPTED
                    self._metrics.completion_attempted_count += 1
            if self._completion_callback_task is callback:
                self._completion_callback_task = None
                self._completion_callback_token = None
            return cancelled
        if not dispatcher.done():
            dispatcher.cancel()
            timeout = max(
                self._config.callback_timeout_seconds * 3,
                _HOST_POLL_INTERVAL_SECONDS,
            )
            done, _ = await asyncio.wait({dispatcher}, timeout=timeout)
            if not done:
                dispatcher.cancel()
                done, _ = await asyncio.wait({dispatcher}, timeout=timeout)
            if not done:
                return False
        self._consume_completion_dispatcher_result(dispatcher)
        return True

    async def _cancel_detached_callbacks_bounded(self) -> bool:
        detached_tasks = tuple(self._detached_callback_tasks)
        if not detached_tasks:
            return True
        results = await asyncio.gather(
            *(self._cancel_callback_bounded(task) for task in detached_tasks)
        )
        for task in detached_tasks:
            if task.done():
                self._consume_detached_callback_result(task)
        return all(results)

    def _consume_callback_result(self, task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        finally:
            if self._observation_task is task and task.done():
                self._observation_task = None

    def _consume_detached_callback_result(self, task: asyncio.Task[None]) -> None:
        self._detached_callback_tasks.discard(task)
        token = self._detached_completion_tokens.pop(task, None)
        if token is None:
            return
        self._consume_completion_callback_result(task)
        if token.completion_state is _CompletionState.DISPATCHED:
            token.completion_state = _CompletionState.ATTEMPTED
            self._metrics.completion_attempted_count += 1
        if (
            self._completion_callback_task is None
            and not self._detached_completion_tokens
            and self._completion_queue.empty()
        ):
            self._clear_degraded_cause("completion_stalled")

    def _consume_completion_dispatcher_result(
        self,
        task: asyncio.Task[None],
    ) -> None:
        abnormal = False
        try:
            abnormal = task.exception() is not None
        except asyncio.CancelledError:
            pass
        if self._completion_dispatcher_task is task and task.done():
            self._completion_dispatcher_task = None
        if abnormal and not self._closed:
            self._set_degraded_cause("dispatcher_start_failure")
            self._drain_completion_queue()

    def _consume_worker_result(self, task: asyncio.Task[None]) -> None:
        abnormal = False
        try:
            abnormal = task.exception() is not None
        except asyncio.CancelledError:
            abnormal = True
        if self._worker_task is task and task.done():
            self._worker_task = None
        if self._closed or self._resetting:
            return
        if abnormal:
            self._metrics.worker_start_failure_count += 1
            self._set_degraded_cause("worker_start_failure")
            self._drain_queue(abandon_data_candidates=True)
            return
        if not self._queue.empty() and not self._ensure_worker():
            self._metrics.worker_start_failure_count += 1
            self._set_degraded_cause("worker_start_failure")
            self._drain_queue(abandon_data_candidates=True)

    @staticmethod
    def _audio_ms(sample_count: int, sample_rate_hz: int) -> int:
        return max(1, sample_count * 1_000 // sample_rate_hz)

    @staticmethod
    def _consume_cleanup_result(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    @staticmethod
    def _consume_reset_result(task: asyncio.Task[None]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            return
