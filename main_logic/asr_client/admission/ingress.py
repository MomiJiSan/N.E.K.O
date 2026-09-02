"""FIFO ingress for every mutation of the admission coordinator.

The lane is intentionally separate from effect execution.  One worker reduces
events in the exact order in which synchronous producers enqueue them, then
completes a future only after the coordinator lock has been released.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from main_logic.voice_turn.contracts import VoiceTurnToken

from .contracts import (
    AdmissionEffect,
    AdmissionEvent,
    BoundaryExact,
    CandidateBound,
    CaptureClosed,
    Close,
    Reset,
    RouteReplaced,
    SpeakerAuthorityPending,
    SpeakerAuthorityUnavailable,
    SpeakerAuthorityUnarmed,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
    VoiceTurnAdmissionRecord,
)
from .coordinator import AdmissionBulkResult, VoiceTurnAdmissionCoordinator


_CapacityClass: TypeAlias = Literal["data", "control", "speaker_control"]


class AdmissionIngressCapacityError(RuntimeError):
    """One bounded ingress partition has no remaining reservation."""

    def __init__(
        self,
        turn_token: VoiceTurnToken | None,
        event: AdmissionEvent | None,
        *,
        capacity_class: _CapacityClass = "data",
    ) -> None:
        self.turn_token = turn_token
        self.event = event
        self.event_type = type(event)
        self.capacity_class = capacity_class
        super().__init__(
            f"ASR_ADMISSION_INGRESS_{capacity_class.upper()}_CAPACITY_EXHAUSTED"
        )


class AdmissionIngressClosedError(RuntimeError):
    """The admission ingress lane is not accepting new events."""


@dataclass(slots=True)
class _IngressItem:
    turn_token: VoiceTurnToken | None
    event: AdmissionEvent | None
    now: float | None
    result: asyncio.Future["_IngressResult"]
    capacity_class: _CapacityClass
    coalescing_key: (
        tuple[
            VoiceTurnToken | None,
            AdmissionEvent,
            float | None,
        ]
        | None
    )
    retires_turn: bool = False


_IngressResult: TypeAlias = (
    bool
    | VoiceTurnAdmissionRecord
    | tuple[AdmissionEffect, ...]
    | tuple[AdmissionBulkResult, ...]
)


_DATA_EVENT_TYPES = (BoundaryExact,)
_SPEAKER_CONTROL_EVENT_TYPES = (
    CandidateBound,
    CaptureClosed,
    SpeakerAuthorityPending,
    SpeakerAuthorityUnavailable,
    SpeakerAuthorityUnarmed,
    SpeakerHigh,
    SpeakerLow,
    SpeakerUnavailable,
)


class AdmissionIngressLane:
    """Single-consumer admission event lane with control capacity isolation.

    Optional boundary data, general controls, and ordered speaker facts use
    separate finite reservations in one FIFO.  General/data pressure therefore
    cannot consume the slots required for the two authoritative speaker
    checkpoints.  Identical controls are still coalesced, and every queued item
    (including open/retire operations) consumes exactly one bounded slot.
    """

    def __init__(
        self,
        coordinator: VoiceTurnAdmissionCoordinator,
        *,
        data_capacity: int = 64,
        control_capacity: int = 256,
        speaker_control_capacity: int = 128,
    ) -> None:
        if type(coordinator) is not VoiceTurnAdmissionCoordinator:
            raise TypeError("coordinator must be VoiceTurnAdmissionCoordinator")
        if type(data_capacity) is not int or data_capacity <= 0:
            raise ValueError("data_capacity must be a positive integer")
        if type(control_capacity) is not int or control_capacity <= 0:
            raise ValueError("control_capacity must be a positive integer")
        if (
            type(speaker_control_capacity) is not int
            or speaker_control_capacity <= 0
        ):
            raise ValueError(
                "speaker_control_capacity must be a positive integer"
            )
        self._coordinator = coordinator
        self._data_capacity = data_capacity
        self._control_capacity = control_capacity
        self._speaker_control_capacity = speaker_control_capacity
        self._items: deque[_IngressItem] = deque()
        self._data_pending = 0
        self._control_pending = 0
        self._speaker_control_pending = 0
        self._pending_controls: dict[
            tuple[VoiceTurnToken | None, AdmissionEvent, float | None],
            asyncio.Future[_IngressResult],
        ] = {}
        self._pending_retirements: dict[
            VoiceTurnToken,
            asyncio.Future[_IngressResult],
        ] = {}
        self._available: asyncio.Event | None = None
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._closed = False

    @property
    def data_capacity(self) -> int:
        return self._data_capacity

    @property
    def pending_data_count(self) -> int:
        return self._data_pending

    @property
    def pending_control_count(self) -> int:
        return self._control_pending + self._speaker_control_pending

    @property
    def control_capacity(self) -> int:
        return self._control_capacity

    @property
    def speaker_control_capacity(self) -> int:
        return self._speaker_control_capacity

    @property
    def pending_speaker_control_count(self) -> int:
        return self._speaker_control_pending

    def _reserve_capacity(
        self,
        capacity_class: _CapacityClass,
        turn_token: VoiceTurnToken | None,
        event: AdmissionEvent | None,
    ) -> None:
        pending = {
            "data": self._data_pending,
            "control": self._control_pending,
            "speaker_control": self._speaker_control_pending,
        }[capacity_class]
        capacity = {
            "data": self._data_capacity,
            "control": self._control_capacity,
            "speaker_control": self._speaker_control_capacity,
        }[capacity_class]
        if pending >= capacity:
            raise AdmissionIngressCapacityError(
                turn_token,
                event,
                capacity_class=capacity_class,
            )
        if capacity_class == "data":
            self._data_pending += 1
        elif capacity_class == "control":
            self._control_pending += 1
        else:
            self._speaker_control_pending += 1

    def _release_capacity(self, capacity_class: _CapacityClass) -> None:
        if capacity_class == "data":
            self._data_pending -= 1
        elif capacity_class == "control":
            self._control_pending -= 1
        else:
            self._speaker_control_pending -= 1

    async def start(self) -> None:
        """Bind the lane to the running loop and start its only consumer."""

        if self._closed or self._closing:
            raise AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
        loop = asyncio.get_running_loop()
        if self._worker is not None:
            if self._loop is not loop:
                raise RuntimeError("ASR_ADMISSION_INGRESS_LOOP_MISMATCH")
            return
        self._loop = loop
        self._available = asyncio.Event()
        self._worker = loop.create_task(
            self._run(),
            name="voice-turn-admission-ingress",
        )

    def post_nowait(
        self,
        turn_token: VoiceTurnToken,
        event: AdmissionEvent,
        *,
        now: float | None = None,
    ) -> asyncio.Future[tuple[AdmissionEffect, ...]]:
        """Append synchronously so callback return cannot reorder two facts."""

        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if self._closing or self._closed:
            raise AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
        loop = asyncio.get_running_loop()
        if self._worker is None or self._available is None or self._loop is None:
            raise RuntimeError("ASR_ADMISSION_INGRESS_NOT_STARTED")
        if loop is not self._loop:
            raise RuntimeError("ASR_ADMISSION_INGRESS_LOOP_MISMATCH")
        is_data = isinstance(event, _DATA_EVENT_TYPES)
        capacity_class: _CapacityClass = (
            "data"
            if is_data
            else (
                "speaker_control"
                if isinstance(event, _SPEAKER_CONTROL_EVENT_TYPES)
                else "control"
            )
        )
        coalescing_key: (
            tuple[
                VoiceTurnToken | None,
                AdmissionEvent,
                float | None,
            ]
            | None
        ) = None
        if not is_data:
            coalescing_key = (turn_token, event, now)
            existing = self._pending_controls.get(coalescing_key)
            if existing is not None:
                follower = self._effectless_follower(existing)
                return cast(asyncio.Future[tuple[AdmissionEffect, ...]], follower)
        self._reserve_capacity(capacity_class, turn_token, event)
        result: asyncio.Future[_IngressResult] = loop.create_future()
        self._items.append(
            _IngressItem(
                turn_token,
                event,
                now,
                result,
                capacity_class,
                coalescing_key,
            )
        )
        if not is_data:
            assert coalescing_key is not None
            self._pending_controls[coalescing_key] = result
        self._available.set()
        return cast(asyncio.Future[tuple[AdmissionEffect, ...]], result)

    async def post(
        self,
        turn_token: VoiceTurnToken,
        event: AdmissionEvent,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionEffect, ...]:
        """Await one queued reduction without transferring cancellation ownership."""

        return await asyncio.shield(self.post_nowait(turn_token, event, now=now))

    def open_turn_nowait(
        self,
        turn_token: VoiceTurnToken,
    ) -> asyncio.Future[VoiceTurnAdmissionRecord]:
        """Allocate one admission record in the same FIFO as every fact."""

        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if self._closing or self._closed:
            raise AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
        loop = asyncio.get_running_loop()
        if self._worker is None or self._available is None or self._loop is None:
            raise RuntimeError("ASR_ADMISSION_INGRESS_NOT_STARTED")
        if loop is not self._loop:
            raise RuntimeError("ASR_ADMISSION_INGRESS_LOOP_MISMATCH")
        self._reserve_capacity("control", turn_token, None)
        result: asyncio.Future[_IngressResult] = loop.create_future()
        self._items.append(
            _IngressItem(turn_token, None, None, result, "control", None)
        )
        self._available.set()
        return cast(asyncio.Future[VoiceTurnAdmissionRecord], result)

    async def open_turn(
        self,
        turn_token: VoiceTurnToken,
    ) -> VoiceTurnAdmissionRecord:
        """Await FIFO record allocation without transferring cancellation."""

        return await asyncio.shield(self.open_turn_nowait(turn_token))

    def retire_turn_nowait(
        self,
        turn_token: VoiceTurnToken,
    ) -> asyncio.Future[bool]:
        """Queue one idempotent settled-record retirement in the same FIFO."""

        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if self._closing or self._closed:
            raise AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
        loop = asyncio.get_running_loop()
        if self._worker is None or self._available is None or self._loop is None:
            raise RuntimeError("ASR_ADMISSION_INGRESS_NOT_STARTED")
        if loop is not self._loop:
            raise RuntimeError("ASR_ADMISSION_INGRESS_LOOP_MISMATCH")
        existing = self._pending_retirements.get(turn_token)
        if existing is not None:
            follower: asyncio.Future[bool] = loop.create_future()

            def transfer_result(completed: asyncio.Future[_IngressResult]) -> None:
                if follower.done():
                    return
                if completed.cancelled():
                    follower.cancel()
                    return
                error = completed.exception()
                if error is not None:
                    follower.set_exception(error)
                else:
                    follower.set_result(bool(completed.result()))

            existing.add_done_callback(transfer_result)
            return follower
        self._reserve_capacity("control", turn_token, None)
        result: asyncio.Future[_IngressResult] = loop.create_future()
        self._items.append(
            _IngressItem(turn_token, None, None, result, "control", None, True)
        )
        self._pending_retirements[turn_token] = result
        self._available.set()
        return cast(asyncio.Future[bool], result)

    async def retire_turn(self, turn_token: VoiceTurnToken) -> bool:
        """Await one FIFO retirement check without transferring cancellation."""

        return await asyncio.shield(self.retire_turn_nowait(turn_token))

    def invalidate_all_nowait(
        self,
        event: Reset | Close | RouteReplaced,
        *,
        now: float | None = None,
    ) -> asyncio.Future[tuple[AdmissionBulkResult, ...]]:
        """Enqueue one bulk route fence in the same FIFO as per-turn facts."""

        if type(event) not in {Reset, Close, RouteReplaced}:
            raise TypeError("event must be Reset, Close, or RouteReplaced")
        if self._closing or self._closed:
            raise AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
        loop = asyncio.get_running_loop()
        if self._worker is None or self._available is None or self._loop is None:
            raise RuntimeError("ASR_ADMISSION_INGRESS_NOT_STARTED")
        if loop is not self._loop:
            raise RuntimeError("ASR_ADMISSION_INGRESS_LOOP_MISMATCH")
        coalescing_key: tuple[
            VoiceTurnToken | None,
            AdmissionEvent,
            float | None,
        ] = (None, event, now)
        existing = self._pending_controls.get(coalescing_key)
        if existing is not None:
            follower = self._effectless_follower(existing)
            return cast(asyncio.Future[tuple[AdmissionBulkResult, ...]], follower)
        self._reserve_capacity("control", None, event)
        result: asyncio.Future[_IngressResult] = loop.create_future()
        self._items.append(
            _IngressItem(None, event, now, result, "control", coalescing_key)
        )
        self._pending_controls[coalescing_key] = result
        self._available.set()
        return cast(asyncio.Future[tuple[AdmissionBulkResult, ...]], result)

    async def invalidate_all(
        self,
        event: Reset | Close | RouteReplaced,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionBulkResult, ...]:
        """Await a bulk route fence without bypassing ingress ordering."""

        return await asyncio.shield(self.invalidate_all_nowait(event, now=now))

    def _effectless_follower(
        self,
        leader: asyncio.Future[_IngressResult],
    ) -> asyncio.Future[_IngressResult]:
        """Follow one coalesced reduction without acquiring its effect ownership."""

        assert self._loop is not None
        follower: asyncio.Future[_IngressResult] = self._loop.create_future()

        def transfer_result(completed: asyncio.Future[_IngressResult]) -> None:
            if follower.done():
                return
            if completed.cancelled():
                follower.cancel()
                return
            error = completed.exception()
            if error is not None:
                follower.set_exception(error)
            else:
                follower.set_result(())

        leader.add_done_callback(transfer_result)
        return follower

    async def close(self) -> None:
        """Stop accepting events, drain the FIFO, and join the worker."""

        if self._closed:
            return
        if self._worker is None or self._available is None:
            self._closing = True
            self._closed = True
            return
        self._closing = True
        self._available.set()
        worker = self._worker
        if worker is asyncio.current_task():
            raise RuntimeError("ASR_ADMISSION_INGRESS_SELF_CLOSE")
        await asyncio.shield(worker)

    async def _run(self) -> None:
        assert self._available is not None
        try:
            while True:
                await self._available.wait()
                while self._items:
                    item = self._items.popleft()
                    try:
                        if item.retires_turn:
                            assert item.turn_token is not None
                            assert item.event is None
                            effects = await self._coordinator.retire(item.turn_token)
                        elif item.turn_token is None:
                            assert type(item.event) in {Reset, Close, RouteReplaced}
                            bulk_event = cast(
                                Reset | Close | RouteReplaced,
                                item.event,
                            )
                            effects = await self._coordinator.invalidate_all(
                                bulk_event,
                                now=item.now,
                            )
                        elif item.event is None:
                            effects = await self._coordinator.open_turn(item.turn_token)
                        else:
                            effects = await self._coordinator.post(
                                item.turn_token,
                                item.event,
                                now=item.now,
                            )
                    except Exception as exc:
                        if not item.result.done():
                            item.result.set_exception(exc)
                    else:
                        if not item.result.done():
                            item.result.set_result(effects)
                    finally:
                        self._release_capacity(item.capacity_class)
                        if item.retires_turn:
                            assert item.turn_token is not None
                            self._pending_retirements.pop(item.turn_token, None)
                        if item.coalescing_key is not None:
                            self._pending_controls.pop(item.coalescing_key, None)
                self._available.clear()
                if self._closing:
                    return
        finally:
            self._closed = True
            self._closing = True
            error = AdmissionIngressClosedError("ASR_ADMISSION_INGRESS_CLOSED")
            while self._items:
                item = self._items.popleft()
                self._release_capacity(item.capacity_class)
                if item.coalescing_key is not None:
                    self._pending_controls.pop(item.coalescing_key, None)
                if item.retires_turn:
                    assert item.turn_token is not None
                    self._pending_retirements.pop(item.turn_token, None)
                if not item.result.done():
                    item.result.set_exception(error)


__all__ = [
    "AdmissionIngressCapacityError",
    "AdmissionIngressClosedError",
    "AdmissionIngressLane",
]
