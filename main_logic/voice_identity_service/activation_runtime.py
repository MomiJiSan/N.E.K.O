"""Runtime coordinator for Owner-activated voice sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import math

from main_logic.voice_input.activation import (
    ActivationDecision,
    ActivationGeneration,
    ActivationState,
    AudioFrame,
    OutputCommit,
    VerificationRequest,
    VerificationResultKind,
    VoiceActivationController,
)

from .activation_scoring import (
    ActivationScoreIdentity,
    ActivationScoreStatus,
    CampPlusActivationScorer,
)


ActivationOutput = Callable[[AudioFrame], Awaitable[OutputCommit]]
ActivationStatusCallback = Callable[[ActivationDecision], None]


@dataclass(frozen=True, slots=True)
class VoiceSessionActivationRuntimeConfig:
    owner_similarity_threshold: float = 0.40
    first_checkpoint_seconds: float = 1.5
    second_checkpoint_seconds: float = 3.0
    candidate_silence_seconds: float = 0.5
    shutdown_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.owner_similarity_threshold)
            or not -1.0 <= self.owner_similarity_threshold <= 1.0
        ):
            raise ValueError("owner_similarity_threshold must be within [-1, 1]")
        if self.first_checkpoint_seconds <= 0:
            raise ValueError("first_checkpoint_seconds must be positive")
        if self.second_checkpoint_seconds <= self.first_checkpoint_seconds:
            raise ValueError("second checkpoint must follow first checkpoint")
        if self.candidate_silence_seconds <= 0:
            raise ValueError("candidate_silence_seconds must be positive")
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")


class VoiceSessionActivationRuntime:
    """Drive scoring and the single output writer around the pure controller."""

    def __init__(
        self,
        generation: ActivationGeneration,
        scorer: CampPlusActivationScorer,
        output: ActivationOutput,
        *,
        controller: VoiceActivationController | None = None,
        config: VoiceSessionActivationRuntimeConfig | None = None,
        status_callback: ActivationStatusCallback | None = None,
        enabled: bool = True,
    ) -> None:
        if not callable(output):
            raise TypeError("output must be callable")
        if status_callback is not None and not callable(status_callback):
            raise TypeError("status_callback must be callable or None")
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self._generation = generation
        self._scorer = scorer
        self._output = output
        self._controller = controller or VoiceActivationController()
        self._config = config or VoiceSessionActivationRuntimeConfig()
        self._status_callback = status_callback
        self._enabled = enabled
        self._lock = asyncio.Lock()
        self._verification_task: asyncio.Task[None] | None = None
        self._output_task: asyncio.Task[None] | None = None
        self._output_retry_requested = False
        self._idle_task: asyncio.Task[None] | None = None
        self._closed = False
        self._candidate_start_sequence: int | None = None
        self._candidate_start_sample: int | None = None
        self._candidate_voice_samples = 0
        self._last_voice_end_at: float | None = None
        self._attempted_checkpoints: set[float] = set()
        self._controller.start(generation, enabled=enabled)

    @property
    def state(self) -> ActivationState:
        return self._controller.state

    @property
    def generation(self) -> ActivationGeneration:
        return self._generation

    async def prepare(self) -> ActivationDecision:
        if not self._enabled:
            return self._publish(self._controller.disable())
        status = await self._scorer.prepare()
        async with self._lock:
            if self._closed:
                return self._publish(self._controller.close())
            if status is ActivationScoreStatus.READY:
                return self._publish(self._controller.mark_ready(self._generation))
            return self._publish(
                self._controller.mark_unavailable(
                    self._generation,
                    status.value,
                )
            )

    async def feed(
        self,
        frame: AudioFrame,
        *,
        voice_activity: bool,
    ) -> ActivationDecision:
        request: VerificationRequest | None = None
        async with self._lock:
            if self._closed:
                return self._publish(self._controller.close())
            decision = self._controller.ingest(frame, voice_activity=voice_activity)
            if (
                decision.reason == "frame_buffered"
                and decision.state is ActivationState.PREPARING
            ):
                self._advance_candidate(
                    frame,
                    voice_activity=voice_activity,
                    allow_verification=False,
                )
            elif decision.reason == "frame_buffered" and decision.state in {
                ActivationState.WAITING,
                ActivationState.VERIFYING,
            }:
                request = self._advance_candidate(frame, voice_activity=voice_activity)
            elif decision.state in {ActivationState.ACTIVE, ActivationState.REPLAYING}:
                self._clear_candidate()
            decision = self._publish(decision)
            self._ensure_output_task_locked()
            self._ensure_idle_task_locked()
            if request is not None:
                self._ensure_verification_task_locked(request)
            return decision

    async def tick(self, *, now: float | None = None) -> ActivationDecision:
        async with self._lock:
            decision = self._controller.tick(now)
            if decision.state is ActivationState.WAITING:
                self._clear_candidate()
            return self._publish(decision)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._publish(self._controller.close())
            verification_task = self._verification_task
            output_task = self._output_task
            idle_task = self._idle_task
        for task in (verification_task, output_task, idle_task):
            if task is not None and not task.done():
                task.cancel()
        await self._scorer.close()
        await self._join_tasks(
            tuple(
                task
                for task in (verification_task, output_task, idle_task)
                if task is not None
            ),
            timeout_seconds=self._config.shutdown_timeout_seconds,
        )

    def _advance_candidate(
        self,
        frame: AudioFrame,
        *,
        voice_activity: bool,
        allow_verification: bool = True,
    ) -> VerificationRequest | None:
        if voice_activity:
            if (
                self._last_voice_end_at is not None
                and frame.captured_at - self._last_voice_end_at
                >= self._config.candidate_silence_seconds
            ):
                self._clear_candidate()
            if self._candidate_start_sequence is None:
                self._candidate_start_sequence = frame.sequence
                self._candidate_start_sample = frame.sample_start
                self._attempted_checkpoints.clear()
            self._candidate_voice_samples += frame.sample_end - frame.sample_start
            self._last_voice_end_at = frame.captured_end_at
        elif (
            self._last_voice_end_at is not None
            and frame.captured_end_at - self._last_voice_end_at
            >= self._config.candidate_silence_seconds
        ):
            self._clear_candidate()
            return None

        start_sequence = self._candidate_start_sequence
        start_sample = self._candidate_start_sample
        if start_sequence is None or start_sample is None:
            return None
        if not allow_verification:
            return None
        duration = self._candidate_voice_samples / frame.sample_rate
        checkpoint = next(
            (
                value
                for value in (
                    self._config.first_checkpoint_seconds,
                    self._config.second_checkpoint_seconds,
                )
                if duration >= value and value not in self._attempted_checkpoints
            ),
            None,
        )
        if checkpoint is None:
            return None
        self._attempted_checkpoints.add(checkpoint)
        decision = self._controller.request_verification(
            candidate_start_sequence=start_sequence,
            candidate_end_sequence=frame.sequence,
        )
        self._publish(decision)
        return decision.verification_request

    def _ensure_verification_task_locked(self, request: VerificationRequest) -> None:
        task = self._verification_task
        if task is not None and not task.done():
            return
        self._verification_task = asyncio.create_task(
            self._verify(request),
            name="voice-session-activation-verify",
        )

    async def _verify(self, request: VerificationRequest) -> None:
        async with self._lock:
            verification_input = self._controller.claim_verification_input(request)
        if verification_input is None:
            return
        score_identity = ActivationScoreIdentity(
            self._scorer.profile_generation,
            self._scorer.scorer_generation,
            request.request_id,
        )
        score = await self._scorer.score(
            score_identity,
            verification_input.pcm,
            sample_rate_hz=verification_input.sample_rate,
        )
        if score.status is ActivationScoreStatus.READY:
            kind = (
                VerificationResultKind.OWNER
                if float(score.similarity) >= self._config.owner_similarity_threshold
                else VerificationResultKind.NOT_OWNER
            )
        elif score.status is ActivationScoreStatus.INVALID_AUDIO:
            kind = VerificationResultKind.INSUFFICIENT
        else:
            kind = VerificationResultKind.FAILED

        next_request: VerificationRequest | None = None
        async with self._lock:
            if self._closed:
                return
            decision = self._controller.apply_verification_result(request, kind)
            self._publish(decision)
            next_request = decision.verification_request
            self._ensure_output_task_locked()
            self._ensure_idle_task_locked()
            self._verification_task = None
            if next_request is not None:
                self._ensure_verification_task_locked(next_request)

    def _ensure_output_task_locked(self) -> None:
        if self._closed:
            return
        task = self._output_task
        if task is not None and not task.done():
            # A frame arrived while the sole writer was awaiting downstream.
            # If that attempt proves NOT_SENT, consume this edge exactly once
            # so the newly arrived input can trigger a safe retry after the
            # old writer releases its lease.
            self._output_retry_requested = True
            return
        lease = self._controller.claim_output()
        if lease is None:
            return
        # claim_output is exclusive. Release this speculative lease as NOT_SENT
        # so the actual writer task can claim it after this synchronous check.
        self._controller.complete_output(lease, OutputCommit.NOT_SENT)
        self._output_retry_requested = False
        self._output_task = asyncio.create_task(
            self._drain_output(),
            name="voice-session-activation-output",
        )

    async def _drain_output(self) -> None:
        while True:
            async with self._lock:
                if self._closed:
                    return
                lease = self._controller.claim_output()
                if lease is None:
                    self._output_task = None
                    return
            try:
                commit = await self._output(lease.frame)
            except asyncio.CancelledError:
                async with self._lock:
                    if not self._closed:
                        self._publish(
                            self._controller.complete_output(
                                lease,
                                OutputCommit.UNKNOWN,
                            )
                        )
                raise
            except Exception:
                commit = OutputCommit.UNKNOWN
            async with self._lock:
                decision = self._controller.complete_output(lease, commit)
                self._publish(decision)
                self._ensure_idle_task_locked()
                if commit in {OutputCommit.NOT_SENT, OutputCommit.UNKNOWN}:
                    self._output_task = None
                    if (
                        commit is OutputCommit.NOT_SENT
                        and self._output_retry_requested
                    ):
                        self._output_retry_requested = False
                        self._output_task = asyncio.create_task(
                            self._drain_output(),
                            name="voice-session-activation-output",
                        )
                    return

    def _ensure_idle_task_locked(self) -> None:
        if self._closed or self._controller.state not in {
            ActivationState.REPLAYING,
            ActivationState.ACTIVE,
        }:
            return
        task = self._idle_task
        if task is None or task.done():
            self._idle_task = asyncio.create_task(
                self._run_idle_timer(),
                name="voice-session-activation-idle",
            )

    async def _run_idle_timer(self) -> None:
        current = asyncio.current_task()
        try:
            while True:
                await asyncio.sleep(1.0)
                async with self._lock:
                    if self._closed:
                        return
                    decision = self._controller.tick()
                    self._publish(decision)
                    if decision.state not in {
                        ActivationState.REPLAYING,
                        ActivationState.ACTIVE,
                    }:
                        if decision.state is ActivationState.WAITING:
                            self._clear_candidate()
                        return
        finally:
            if self._idle_task is current:
                self._idle_task = None

    def _clear_candidate(self) -> None:
        self._candidate_start_sequence = None
        self._candidate_start_sample = None
        self._candidate_voice_samples = 0
        self._last_voice_end_at = None
        self._attempted_checkpoints.clear()

    def _publish(self, decision: ActivationDecision) -> ActivationDecision:
        if self._status_callback is not None:
            self._status_callback(decision)
        return decision

    @classmethod
    async def _join_tasks(
        cls,
        tasks: tuple[asyncio.Task[None], ...],
        *,
        timeout_seconds: float,
    ) -> None:
        if not tasks:
            return
        done, pending = await asyncio.wait(set(tasks), timeout=timeout_seconds)
        for task in done:
            cls._consume_task_result(task)
        for task in pending:
            task.add_done_callback(cls._consume_task_result)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass


__all__ = [
    "ActivationOutput",
    "ActivationStatusCallback",
    "VoiceSessionActivationRuntime",
    "VoiceSessionActivationRuntimeConfig",
]
