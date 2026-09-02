from __future__ import annotations

import asyncio

import pytest

from main_logic.asr_client.admission.contracts import (
    AdmissionDisposition,
    CandidateBound,
    CaptureClosed,
    CoreSettled,
    EvidenceState,
    LifecycleSettled,
    PendingProviderFinal,
    ProviderFinalReceived,
    ResolveReserved,
    RouteReplaced,
    SpeakerCheckpointKind,
    SpeakerLow,
    SpeakerUnavailable,
    TransportSettled,
)
from main_logic.asr_client.admission.coordinator import (
    VoiceTurnAdmissionCoordinator,
)
from main_logic.asr_client.admission.ingress import (
    AdmissionIngressCapacityError,
    AdmissionIngressClosedError,
    AdmissionIngressLane,
)
from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCandidateKey,
)
from main_logic.voice_turn.contracts import VoiceIngressToken, VoiceTurnToken


pytestmark = pytest.mark.asyncio


def _token(turn_id: int = 1) -> VoiceTurnToken:
    return VoiceTurnToken(VoiceIngressToken(1, "socket", 2, 3, 4), turn_id)


def _candidate() -> SpeakerShadowCandidateKey:
    return SpeakerShadowCandidateKey(2, 1, "provider_candidate")


async def test_post_nowait_preserves_fact_then_completion_fifo():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token()
    await coordinator.open_turn(token, speaker_candidate=_candidate())
    lane = AdmissionIngressLane(coordinator, data_capacity=2)
    await lane.start()

    first = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    second = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    closed = lane.post_nowait(token, CaptureClosed(_candidate(), 2))
    await first
    await second
    await closed

    record = await coordinator.get_record(token)
    assert record is not None
    assert record.last_speaker_sequence_no == 2
    assert record.evidence_state is EvidenceState.REJECT_REQUESTED
    await lane.close()


async def test_open_turn_is_ordered_before_immediate_speaker_fact():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token()
    lane = AdmissionIngressLane(coordinator, data_capacity=1)
    await lane.start()

    opened = lane.open_turn_nowait(token)
    bound = lane.post_nowait(token, CandidateBound(_candidate()))
    fact = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )

    await opened
    await bound
    await fact
    record = await coordinator.get_record(token)
    assert record is not None
    assert record.evidence_state is EvidenceState.FIRST_LOW
    await lane.close()


async def test_terminal_settlement_retires_capacity_through_same_fifo():
    coordinator = VoiceTurnAdmissionCoordinator(capacity=1, clock=lambda: 10.0)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    first = _token(1)

    await lane.open_turn(first)
    effects = await lane.post(
        first,
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
        ),
    )
    resolution = next(
        effect for effect in effects if isinstance(effect, ResolveReserved)
    )
    await lane.post(first, CoreSettled(resolution.ticket))
    await lane.post(first, TransportSettled(resolution.ticket))
    assert await lane.retire_turn(first) is False
    await lane.post(first, LifecycleSettled(resolution.ticket))
    assert await lane.retire_turn(first) is True

    await lane.open_turn(_token(2))
    assert await coordinator.get_record(first) is None
    assert await coordinator.get_record(_token(2)) is not None
    await lane.close()


async def test_open_turn_then_bulk_fence_then_fact_is_one_fifo():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token()
    lane = AdmissionIngressLane(coordinator)
    await lane.start()

    opened = lane.open_turn_nowait(token)
    fenced = lane.invalidate_all_nowait(RouteReplaced())
    fact = lane.post_nowait(token, SpeakerUnavailable(_candidate(), 1))

    await opened
    bulk = await fenced
    await fact
    assert len(bulk) == 1
    record = await coordinator.get_record(token)
    assert record is not None
    assert record.admission_state.value == "abandoned"
    assert record.evidence_state is EvidenceState.NONE
    await lane.close()


async def test_data_overflow_is_explicit_but_unavailable_and_final_are_reserved_control():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token()
    await coordinator.open_turn(token, speaker_candidate=_candidate())
    lane = AdmissionIngressLane(coordinator, data_capacity=1)
    await lane.start()

    first = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    overflowed = SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND)
    with pytest.raises(
        AdmissionIngressCapacityError,
        match="DATA_CAPACITY_EXHAUSTED",
    ) as error:
        lane.post_nowait(token, overflowed)
    assert error.value.turn_token == token
    assert error.value.event is overflowed

    unavailable = lane.post_nowait(token, SpeakerUnavailable(_candidate(), 2))
    final = lane.post_nowait(
        token,
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
        ),
    )
    await first
    await unavailable
    effects = await final

    resolution = next(effect for effect in effects if isinstance(effect, ResolveReserved))
    assert resolution.disposition is AdmissionDisposition.FORWARD
    record = await coordinator.get_record(token)
    assert record is not None and record.evidence_state is EvidenceState.UNAVAILABLE
    await lane.close()


async def test_failed_item_does_not_stop_single_consumer_and_close_drains():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token()
    await coordinator.open_turn(token)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()

    missing = lane.post_nowait(
        _token(2),
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "missing", 10.0, 10.2)
        ),
    )
    accepted = lane.post_nowait(
        token,
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "accepted", 10.0, 10.2)
        ),
    )
    with pytest.raises(KeyError):
        await missing
    effects = await accepted
    assert any(isinstance(effect, ResolveReserved) for effect in effects)

    await lane.close()
    with pytest.raises(AdmissionIngressClosedError, match="INGRESS_CLOSED"):
        lane.post_nowait(token, SpeakerUnavailable(_candidate(), 1))
    with pytest.raises(AdmissionIngressClosedError, match="INGRESS_CLOSED"):
        await lane.start()


async def test_identical_control_retry_has_no_effect_execution_ownership():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token()
    await coordinator.open_turn(token)
    lane = AdmissionIngressLane(coordinator, data_capacity=1)
    await lane.start()
    event = ProviderFinalReceived(
        PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
    )

    first = lane.post_nowait(token, event)
    duplicate = lane.post_nowait(token, event)

    assert duplicate is not first
    assert lane.pending_control_count == 1
    leader_effects = await first
    follower_effects = await duplicate
    assert sum(isinstance(effect, ResolveReserved) for effect in leader_effects) == 1
    assert follower_effects == ()
    assert lane.pending_control_count == 0
    await lane.close()


async def test_control_follower_cancellation_does_not_cancel_effect_owner():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token()
    await coordinator.open_turn(token)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    await coordinator._lock.acquire()
    event = ProviderFinalReceived(
        PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
    )
    leader = lane.post_nowait(token, event)
    follower = lane.post_nowait(token, event)

    follower.cancel()
    coordinator._lock.release()
    leader_effects = await leader

    assert follower.cancelled() is True
    assert sum(isinstance(effect, ResolveReserved) for effect in leader_effects) == 1
    await lane.close()


async def test_control_follower_propagates_leader_exception():
    coordinator = VoiceTurnAdmissionCoordinator()
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    token = _token()
    event = SpeakerUnavailable(_candidate(), 1)
    leader = lane.post_nowait(token, event)
    follower = lane.post_nowait(token, event)

    with pytest.raises(KeyError) as leader_error:
        await leader
    with pytest.raises(KeyError) as follower_error:
        await follower

    assert follower_error.value is leader_error.value
    await lane.close()


async def test_identical_bulk_retry_has_no_effect_execution_ownership():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token()
    await coordinator.open_turn(token)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()

    leader = lane.invalidate_all_nowait(RouteReplaced())
    follower = lane.invalidate_all_nowait(RouteReplaced())
    leader_results = await leader
    follower_results = await follower

    assert len(leader_results) == 1
    assert sum(
        isinstance(effect, ResolveReserved)
        for result in leader_results
        for effect in result.effects
    ) == 1
    assert follower_results == ()
    await lane.close()


async def test_bulk_route_fence_is_ordered_with_per_turn_facts():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token()
    await coordinator.open_turn(token, speaker_candidate=_candidate())
    lane = AdmissionIngressLane(coordinator, data_capacity=2)
    await lane.start()

    before = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 1, SpeakerCheckpointKind.FIRST),
    )
    fenced = lane.invalidate_all_nowait(RouteReplaced())
    after = lane.post_nowait(
        token,
        SpeakerLow(_candidate(), 2, SpeakerCheckpointKind.SECOND),
    )
    await before
    bulk_results = await fenced
    await after

    assert len(bulk_results) == 1
    assert any(
        isinstance(effect, ResolveReserved)
        and effect.disposition is AdmissionDisposition.ABANDON
        for effect in bulk_results[0].effects
    )
    record = await coordinator.get_record(token)
    assert record is not None
    assert record.admission_state.value == "abandoned"
    assert record.last_speaker_sequence_no == 1
    await lane.close()


async def test_waiter_cancellation_does_not_cancel_accepted_final_ownership():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token()
    await coordinator.open_turn(token)
    lane = AdmissionIngressLane(coordinator)
    await lane.start()
    await coordinator._lock.acquire()
    waiter = asyncio.create_task(
        lane.post(
            token,
            ProviderFinalReceived(
                PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
            ),
        )
    )
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    coordinator._lock.release()
    await lane.close()

    record = await coordinator.get_record(token)
    assert record is not None
    assert record.admission_state.value == "forwarded"
