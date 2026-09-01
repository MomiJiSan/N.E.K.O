"""Single-writer storage around the pure admission reducer."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from main_logic.voice_turn.contracts import VoiceTurnToken

from .._provider_events import ProviderUtteranceKey
from ..speaker_shadow.contracts import SpeakerShadowCandidateKey
from .contracts import (
    AdmissionEffect,
    AdmissionEvent,
    CandidateBindingState,
    CaptureState,
    ProviderBindingState,
    SettlementState,
    Close,
    Reset,
    RouteReplaced,
    VoiceTurnAdmissionRecord,
)
from .reducer import reduce


class AdmissionCapacityError(RuntimeError):
    """A core admission record could not be reserved without data loss."""


class AdmissionIdentityError(RuntimeError):
    """A logical turn token or one of its aliases was reused inconsistently."""


@dataclass(frozen=True, slots=True)
class AdmissionBulkResult:
    """Effects produced for one turn by an atomic bulk invalidation snapshot."""

    turn_token: VoiceTurnToken
    effects: tuple[AdmissionEffect, ...]


class VoiceTurnAdmissionCoordinator:
    """Own admission records while leaving every asynchronous effect outside."""

    def __init__(
        self,
        *,
        capacity: int = 8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._capacity = capacity
        self._clock = clock
        self._records: dict[VoiceTurnToken, VoiceTurnAdmissionRecord] = {}
        self._retired_turn_high_water: dict[object, int] = {}
        self._record_generation = 0
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    async def open_turn(
        self,
        turn_token: VoiceTurnToken,
        *,
        provider_key: ProviderUtteranceKey | None = None,
        speaker_candidate: SpeakerShadowCandidateKey | None = None,
    ) -> VoiceTurnAdmissionRecord:
        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        async with self._lock:
            existing = self._records.get(turn_token)
            if existing is not None:
                if (
                    provider_key is not None
                    and existing.provider_key != provider_key
                ) or (
                    speaker_candidate is not None
                    and existing.speaker_candidate != speaker_candidate
                ):
                    raise AdmissionIdentityError("ASR_ADMISSION_ALIAS_CONFLICT")
                return existing
            if turn_token.turn_id <= self._retired_turn_high_water.get(
                turn_token.ingress,
                0,
            ):
                raise AdmissionIdentityError("ASR_ADMISSION_TURN_ALREADY_RETIRED")
            if len(self._records) >= self._capacity:
                raise AdmissionCapacityError("ASR_ADMISSION_CAPACITY_EXHAUSTED")
            self._record_generation += 1
            record = VoiceTurnAdmissionRecord(
                turn_token=turn_token,
                record_generation=self._record_generation,
                provider_binding_state=(
                    ProviderBindingState.BOUND
                    if provider_key is not None
                    else ProviderBindingState.UNBOUND
                ),
                candidate_binding_state=(
                    CandidateBindingState.BOUND
                    if speaker_candidate is not None
                    else CandidateBindingState.UNBOUND
                ),
                capture_state=(
                    CaptureState.COLLECTING
                    if speaker_candidate is not None
                    else CaptureState.NONE
                ),
                provider_key=provider_key,
                speaker_candidate=speaker_candidate,
            )
            self._records[turn_token] = record
            return record

    async def post(
        self,
        turn_token: VoiceTurnToken,
        event: AdmissionEvent,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionEffect, ...]:
        """Reduce under the short lock and return effects without executing them."""

        async with self._lock:
            record = self._records.get(turn_token)
            if record is None:
                raise KeyError(turn_token)
            reduced, effects = reduce(
                record,
                event,
                self._clock() if now is None else now,
            )
            self._records[turn_token] = reduced
            return effects

    async def get_record(
        self,
        turn_token: VoiceTurnToken,
    ) -> VoiceTurnAdmissionRecord | None:
        async with self._lock:
            return self._records.get(turn_token)

    async def live_turn_tokens(self) -> tuple[VoiceTurnToken, ...]:
        """Return one insertion-ordered snapshot without exposing the record table."""

        async with self._lock:
            return tuple(self._records)

    async def invalidate_all(
        self,
        event: Reset | Close | RouteReplaced,
        *,
        now: float | None = None,
    ) -> tuple[AdmissionBulkResult, ...]:
        """Reduce one route invalidation against the complete live snapshot.

        The reducer is run for every record while this coordinator remains the
        single writer.  Effects are only returned after the lock is released;
        callers execute them and post their acknowledgements through ingress.
        """

        if type(event) not in {Reset, Close, RouteReplaced}:
            raise TypeError("event must be Reset, Close, or RouteReplaced")
        effective_now = self._clock() if now is None else now
        async with self._lock:
            results: list[AdmissionBulkResult] = []
            for turn_token, record in self._records.items():
                reduced, effects = reduce(record, event, effective_now)
                self._records[turn_token] = reduced
                results.append(AdmissionBulkResult(turn_token, effects))
            return tuple(results)

    async def retire(self, turn_token: VoiceTurnToken) -> bool:
        """Remove only an already-settled record; never evict live admission."""

        async with self._lock:
            record = self._records.get(turn_token)
            if record is None:
                return False
            if record.terminal_disposition is None or any(
                state not in {SettlementState.SETTLED, SettlementState.DEGRADED}
                for state in (
                    record.core_settlement_state,
                    record.transport_settlement_state,
                    record.lifecycle_settlement_state,
                )
            ):
                return False
            if record.revoked_rejection_ticket is not None:
                return False
            if record.pending_revocations:
                return False
            if record.revocation_degraded:
                return False
            if record.namespace_poison_ticket is not None:
                return False
            if record.rejection_capability is not None:
                return False
            self._records.pop(turn_token, None)
            self._retired_turn_high_water[turn_token.ingress] = max(
                self._retired_turn_high_water.get(turn_token.ingress, 0),
                turn_token.turn_id,
            )
            return True


__all__ = [
    "AdmissionBulkResult",
    "AdmissionCapacityError",
    "AdmissionIdentityError",
    "VoiceTurnAdmissionCoordinator",
]
