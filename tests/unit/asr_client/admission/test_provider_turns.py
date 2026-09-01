from __future__ import annotations

import pytest

from main_logic.asr_client._provider_events import ProviderAudioRange, ProviderUtteranceKey
from main_logic.asr_client.admission.contracts import (
    AdmissionDisposition,
    AdmissionResolutionTicket,
    BoundaryProof,
    PendingProviderFinal,
)
from main_logic.asr_client.admission.provider_turns import (
    ProviderAliasConflictError,
    ProviderBoundaryResult,
    ProviderTurnCorrelator,
)
from main_logic.voice_turn.contracts import VoiceIngressToken, VoiceTurnToken


def _key(utterance_id: int) -> ProviderUtteranceKey:
    return ProviderUtteranceKey(1, 0, utterance_id)


def _token(turn_id: int) -> VoiceTurnToken:
    return VoiceTurnToken(VoiceIngressToken(1, "socket", 2, 3, 4), turn_id)


def _exact(key: ProviderUtteranceKey, turn_id: int) -> ProviderBoundaryResult:
    proof = BoundaryProof(
        proof_id=turn_id,
        owner_generation=7,
        provider_key=key,
    )
    return ProviderBoundaryResult(
        quality="exact",
        audio_range=ProviderAudioRange((turn_id - 1) * 100, turn_id * 100),
        proof=proof,
    )


def _final(key: ProviderUtteranceKey, text: str, received: float) -> PendingProviderFinal:
    return PendingProviderFinal(key, "qwen", text, received, received + 0.2)


def _resolution(turn_id: int, nonce: int = 1) -> AdmissionResolutionTicket:
    return AdmissionResolutionTicket(
        turn_token=_token(turn_id),
        record_generation=turn_id,
        resolution_nonce=nonce,
        disposition=AdmissionDisposition.FORWARD,
    )


def test_boundary_phase_cannot_bind_current_turn():
    correlator = ProviderTurnCorrelator(namespace=(1, 0))
    key = _key(1)
    correlator.record_boundary_result(key, _exact(key, 1))
    with pytest.raises(
        ProviderAliasConflictError,
        match="PROVIDER_ALIAS_BIND_REQUIRES_ORDERED",
    ):
        correlator.bind_ordered(key, _token(1))


def test_ordered_key_binds_exactly_one_voice_turn_token():
    correlator = ProviderTurnCorrelator(namespace=(1, 0))
    first_key = _key(1)
    second_key = _key(2)
    correlator.mark_ordered(first_key)
    record = correlator.bind_ordered(first_key, _token(1))
    assert record.bound_turn_token == _token(1)
    assert correlator.bind_ordered(first_key, _token(1)) is record

    correlator.mark_ordered(second_key)
    with pytest.raises(ProviderAliasConflictError, match="VOICE_TURN_ALREADY_BOUND"):
        correlator.bind_ordered(second_key, _token(1))


def test_conflicting_boundary_downgrades_only_same_key():
    correlator = ProviderTurnCorrelator(namespace=(1, 0))
    first_key, second_key = _key(1), _key(2)
    correlator.record_boundary_result(first_key, _exact(first_key, 1))
    correlator.record_boundary_result(second_key, _exact(second_key, 2))

    conflict = ProviderBoundaryResult(
        quality="exact",
        audio_range=ProviderAudioRange(0, 99),
        proof=_exact(first_key, 1).proof,
    )
    result = correlator.record_boundary_result(first_key, conflict)
    assert result.quality == "unknown"
    assert correlator.record_for(second_key).boundary_result.quality == "exact"


def test_optional_proof_overflow_downgrades_new_key_without_dropping_final():
    correlator = ProviderTurnCorrelator(namespace=(1, 0), proof_capacity=1)
    first_key, second_key = _key(1), _key(2)
    correlator.record_boundary_result(first_key, _exact(first_key, 1))
    result = correlator.record_boundary_result(second_key, _exact(second_key, 2))
    assert result.quality == "unknown"

    correlator.mark_ordered(second_key)
    correlator.bind_ordered(second_key, _token(2))
    final = _final(second_key, "second", 12.0)
    record = correlator.record_final(second_key, final)
    assert record.pending_final is final


def test_provider_final_deadline_is_preserved_while_waiting_for_earlier_key():
    correlator = ProviderTurnCorrelator(namespace=(1, 0))
    first_key, second_key = _key(1), _key(2)
    correlator.mark_ordered(first_key)
    correlator.mark_ordered(second_key)
    first = _final(first_key, "first", 10.0)
    second = _final(second_key, "second", 10.05)
    correlator.record_final(second_key, second)
    correlator.record_final(first_key, first)
    assert correlator.record_for(second_key).pending_final.admission_deadline == 10.25


def test_complete_retires_only_finalized_record_and_bounds_tombstones():
    correlator = ProviderTurnCorrelator(namespace=(1, 0), completed_capacity=1)
    first_key, second_key = _key(1), _key(2)

    correlator.mark_ordered(first_key)
    correlator.bind_ordered(first_key, _token(1))
    assert correlator.complete(first_key, _resolution(1)).completed is False
    assert correlator.record_for(first_key) is not None

    correlator.record_final(first_key, _final(first_key, "first", 10.0))
    assert correlator.complete(first_key, _resolution(1)).completed is True
    assert correlator.record_for(first_key) is None
    assert correlator.is_completed(first_key) is True
    assert correlator.completed_tombstone_count == 1
    assert correlator.record_boundary_result(first_key, _exact(first_key, 1)).quality == (
        "unknown"
    )
    assert correlator.record_for(first_key) is None
    with pytest.raises(
        ProviderAliasConflictError,
        match="PROVIDER_KEY_ALREADY_COMPLETED",
    ):
        correlator.mark_ordered(first_key)

    correlator.mark_ordered(second_key)
    correlator.bind_ordered(second_key, _token(2))
    correlator.record_final(second_key, _final(second_key, "second", 11.0))
    assert correlator.complete(second_key, _resolution(2)).completed is True
    assert correlator.completed_tombstone_count == 1
    assert correlator.is_completed(second_key) is True
    assert correlator.is_completed(first_key) is True
    assert correlator.record_boundary_result(first_key, _exact(first_key, 1)).quality == (
        "unknown"
    )
    with pytest.raises(
        ProviderAliasConflictError,
        match="PROVIDER_KEY_ALREADY_COMPLETED",
    ):
        correlator.mark_ordered(first_key)


def test_pending_final_requires_exact_absolute_200ms_budget():
    key = _key(1)
    with pytest.raises(ValueError, match="exactly 200ms"):
        PendingProviderFinal(key, "qwen", "too short", 10.0, 10.0)
    with pytest.raises(ValueError, match="exactly 200ms"):
        PendingProviderFinal(key, "qwen", "too long", 10.0, 40.0)
