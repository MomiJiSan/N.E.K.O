"""Pure Provider-key correlation, separate from audio ownership execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, TypeAlias

from main_logic.voice_turn.contracts import VoiceTurnToken

from .._provider_events import ProviderAudioRange, ProviderUtteranceKey
from .contracts import AdmissionResolutionTicket, BoundaryProof, PendingProviderFinal


BoundaryResultQuality: TypeAlias = Literal["exact", "unknown"]


@dataclass(frozen=True, slots=True)
class ProviderBoundaryResult:
    quality: BoundaryResultQuality
    audio_range: ProviderAudioRange | None
    proof: BoundaryProof | None

    def __post_init__(self) -> None:
        if self.quality == "exact":
            if self.audio_range is None or self.proof is None:
                raise ValueError("exact boundary requires range and proof")
        elif self.quality == "unknown":
            if self.audio_range is not None or self.proof is not None:
                raise ValueError("unknown boundary cannot carry authority")
        else:
            raise ValueError("unsupported boundary quality")

    @classmethod
    def unknown(cls) -> "ProviderBoundaryResult":
        return cls(quality="unknown", audio_range=None, proof=None)


@dataclass(slots=True)
class ProviderAliasRecord:
    provider_key: ProviderUtteranceKey
    boundary_result: ProviderBoundaryResult | None = None
    ordered_seen: bool = False
    bound_turn_token: VoiceTurnToken | None = None
    pending_final: PendingProviderFinal | None = None


class ProviderAliasConflictError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderAliasCompletionResult:
    completed: bool
    retired_proofs: tuple[BoundaryProof, ...] = ()


class ProviderTurnCorrelator:
    """Bind aliases only in the ordered lane; optional exact proof is bounded."""

    def __init__(
        self,
        *,
        namespace: tuple[int, int],
        proof_capacity: int = 8,
        completed_capacity: int = 256,
    ) -> None:
        if type(proof_capacity) is not int or proof_capacity <= 0:
            raise ValueError("proof_capacity must be a positive integer")
        if type(completed_capacity) is not int or completed_capacity <= 0:
            raise ValueError("completed_capacity must be a positive integer")
        if (
            type(namespace) is not tuple
            or len(namespace) != 2
            or any(type(value) is not int or value < 0 for value in namespace)
        ):
            raise ValueError("namespace must be a non-negative generation/epoch pair")
        self._proof_capacity = proof_capacity
        self._completed_capacity = completed_capacity
        self._records: dict[ProviderUtteranceKey, ProviderAliasRecord] = {}
        self._token_bindings: dict[VoiceTurnToken, ProviderUtteranceKey] = {}
        self._exact_proofs: OrderedDict[
            ProviderUtteranceKey,
            ProviderBoundaryResult,
        ] = OrderedDict()
        self._completed: OrderedDict[ProviderUtteranceKey, None] = OrderedDict()
        self._namespace = namespace
        self._completed_high_water = 0

    @property
    def completed_tombstone_count(self) -> int:
        return len(self._completed)

    def is_completed(self, key: ProviderUtteranceKey) -> bool:
        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        return bool(
            key in self._completed
            or (
                self._namespace == (key.generation, key.buffer_epoch)
                and key.utterance_id <= self._completed_high_water
            )
        )

    def _accept_namespace(self, key: ProviderUtteranceKey) -> bool:
        namespace = (key.generation, key.buffer_epoch)
        return namespace == self._namespace

    def record_for(self, key: ProviderUtteranceKey) -> ProviderAliasRecord | None:
        return self._records.get(key)

    def _record(self, key: ProviderUtteranceKey) -> ProviderAliasRecord:
        record = self._records.get(key)
        if record is None:
            record = ProviderAliasRecord(provider_key=key)
            self._records[key] = record
        return record

    def record_boundary_result(
        self,
        key: ProviderUtteranceKey,
        result: ProviderBoundaryResult,
    ) -> ProviderBoundaryResult:
        """Record ownership only; this API deliberately accepts no turn token."""

        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        if type(result) is not ProviderBoundaryResult:
            raise TypeError("result must be ProviderBoundaryResult")
        if not self._accept_namespace(key) or self.is_completed(key):
            return ProviderBoundaryResult.unknown()
        if (
            key not in self._records
            and sum(not record.ordered_seen for record in self._records.values())
            >= self._proof_capacity
        ):
            return ProviderBoundaryResult.unknown()
        record = self._record(key)
        if (
            result.quality == "exact"
            and result.proof is not None
            and result.proof.provider_key != key
        ):
            result = ProviderBoundaryResult.unknown()
        existing = record.boundary_result
        if existing is not None:
            if existing == result:
                return existing
            # Authority never recovers after a duplicate/conflicting boundary.
            self._exact_proofs.pop(key, None)
            record.boundary_result = ProviderBoundaryResult.unknown()
            return record.boundary_result
        if result.quality == "exact" and len(self._exact_proofs) >= self._proof_capacity:
            record.boundary_result = ProviderBoundaryResult.unknown()
            return record.boundary_result
        record.boundary_result = result
        if result.quality == "exact":
            self._exact_proofs[key] = result
        return record.boundary_result

    def mark_ordered(self, key: ProviderUtteranceKey) -> ProviderAliasRecord:
        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        if not self._accept_namespace(key):
            raise ProviderAliasConflictError("PROVIDER_NAMESPACE_MISMATCH")
        if self.is_completed(key):
            raise ProviderAliasConflictError("PROVIDER_KEY_ALREADY_COMPLETED")
        record = self._record(key)
        record.ordered_seen = True
        return record

    def bind_ordered(
        self,
        key: ProviderUtteranceKey,
        turn_token: VoiceTurnToken,
    ) -> ProviderAliasRecord:
        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        if type(turn_token) is not VoiceTurnToken:
            raise TypeError("turn_token must be VoiceTurnToken")
        if not self._accept_namespace(key):
            raise ProviderAliasConflictError("PROVIDER_NAMESPACE_MISMATCH")
        if self.is_completed(key):
            raise ProviderAliasConflictError("PROVIDER_KEY_ALREADY_COMPLETED")
        record = self._records.get(key)
        if record is None or not record.ordered_seen:
            raise ProviderAliasConflictError("PROVIDER_ALIAS_BIND_REQUIRES_ORDERED")
        if record.bound_turn_token not in {None, turn_token}:
            raise ProviderAliasConflictError("PROVIDER_KEY_ALREADY_BOUND")
        existing_key = self._token_bindings.get(turn_token)
        if existing_key not in {None, key}:
            raise ProviderAliasConflictError("VOICE_TURN_ALREADY_BOUND")
        record.bound_turn_token = turn_token
        self._token_bindings[turn_token] = key
        return record

    def record_final(
        self,
        key: ProviderUtteranceKey,
        final: PendingProviderFinal,
    ) -> ProviderAliasRecord:
        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        if type(final) is not PendingProviderFinal:
            raise TypeError("final must be PendingProviderFinal")
        if not self._accept_namespace(key):
            raise ProviderAliasConflictError("PROVIDER_NAMESPACE_MISMATCH")
        if self.is_completed(key):
            raise ProviderAliasConflictError("PROVIDER_KEY_ALREADY_COMPLETED")
        record = self._records.get(key)
        if record is None or not record.ordered_seen:
            raise ProviderAliasConflictError("PROVIDER_FINAL_REQUIRES_ORDERED")
        if final.provider_key != key:
            raise ProviderAliasConflictError("PROVIDER_FINAL_KEY_MISMATCH")
        if record.pending_final is None:
            record.pending_final = final
        elif record.pending_final != final:
            raise ProviderAliasConflictError("PROVIDER_FINAL_CONFLICT")
        return record

    def complete(
        self,
        key: ProviderUtteranceKey,
        resolution: AdmissionResolutionTicket,
    ) -> ProviderAliasCompletionResult:
        if type(key) is not ProviderUtteranceKey:
            raise TypeError("key must be ProviderUtteranceKey")
        if type(resolution) is not AdmissionResolutionTicket:
            raise TypeError("resolution must be AdmissionResolutionTicket")
        record = self._records.get(key)
        if (
            record is None
            or record.pending_final is None
            or record.bound_turn_token is None
            or resolution.turn_token != record.bound_turn_token
            or any(
                other.provider_key.utterance_id < key.utterance_id
                for other in self._records.values()
                if other is not record and other.ordered_seen
            )
        ):
            return ProviderAliasCompletionResult(False)
        retired_keys = tuple(
            candidate_key
            for candidate_key in self._records
            if candidate_key.utterance_id <= key.utterance_id
        )
        retired_proofs = tuple(
            proof_result.proof
            for retired_key in retired_keys
            if (proof_result := self._exact_proofs.get(retired_key)) is not None
            and proof_result.proof is not None
        )
        for retired_key in retired_keys:
            retired = self._records.pop(retired_key)
            self._exact_proofs.pop(retired_key, None)
            retired_token = retired.bound_turn_token
            if (
                retired_token is not None
                and self._token_bindings.get(retired_token) == retired_key
            ):
                self._token_bindings.pop(retired_token, None)
        token = record.bound_turn_token
        if token is not None and self._token_bindings.get(token) == key:
            self._token_bindings.pop(token, None)
        self._completed[key] = None
        self._completed_high_water = max(
            self._completed_high_water,
            key.utterance_id,
        )
        while len(self._completed) > self._completed_capacity:
            self._completed.popitem(last=False)
        return ProviderAliasCompletionResult(True, retired_proofs)


__all__ = [
    "ProviderAliasConflictError",
    "ProviderAliasCompletionResult",
    "ProviderAliasRecord",
    "ProviderBoundaryResult",
    "ProviderTurnCorrelator",
]
