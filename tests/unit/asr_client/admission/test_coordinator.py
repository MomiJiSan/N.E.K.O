from __future__ import annotations

import pytest

from main_logic.asr_client.admission.contracts import (
    AdmissionDisposition,
    AdmissionState,
    CoreSettled,
    LifecycleSettled,
    PendingProviderFinal,
    ProviderFinalReceived,
    ResolveReserved,
    TransportSettled,
)
from main_logic.asr_client.admission.coordinator import (
    AdmissionCapacityError,
    AdmissionIdentityError,
    VoiceTurnAdmissionCoordinator,
)
from main_logic.asr_client._provider_events import ProviderUtteranceKey
from main_logic.voice_turn.contracts import VoiceIngressToken, VoiceTurnToken


pytestmark = pytest.mark.asyncio


def _token(turn_id: int) -> VoiceTurnToken:
    return VoiceTurnToken(VoiceIngressToken(1, "socket", 2, 3, 4), turn_id)


async def test_post_reduces_under_single_writer_and_returns_effects_without_awaiting_them():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token(1)
    await coordinator.open_turn(token)
    effects = await coordinator.post(
        token,
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
        ),
    )
    resolution = next(effect for effect in effects if isinstance(effect, ResolveReserved))
    record = await coordinator.get_record(token)
    assert resolution.disposition is AdmissionDisposition.FORWARD
    assert record is not None and record.admission_state is AdmissionState.FORWARDED


async def test_record_capacity_failure_is_explicit_not_silent():
    coordinator = VoiceTurnAdmissionCoordinator(capacity=1)
    await coordinator.open_turn(_token(1))
    with pytest.raises(AdmissionCapacityError, match="ASR_ADMISSION_CAPACITY_EXHAUSTED"):
        await coordinator.open_turn(_token(2))


async def test_retire_requires_all_three_settlements_for_same_resolution():
    coordinator = VoiceTurnAdmissionCoordinator(clock=lambda: 10.0)
    token = _token(1)
    await coordinator.open_turn(token)
    effects = await coordinator.post(
        token,
        ProviderFinalReceived(
            PendingProviderFinal(None, "qwen", "hello", 10.0, 10.2)
        ),
    )
    resolution = next(effect for effect in effects if isinstance(effect, ResolveReserved))
    assert await coordinator.retire(token) is False

    await coordinator.post(token, CoreSettled(resolution.ticket))
    await coordinator.post(token, TransportSettled(resolution.ticket))
    assert await coordinator.retire(token) is False
    await coordinator.post(token, LifecycleSettled(resolution.ticket))
    assert await coordinator.retire(token) is True
    assert await coordinator.get_record(token) is None

    with pytest.raises(AdmissionIdentityError, match="TURN_ALREADY_RETIRED"):
        await coordinator.open_turn(token)


async def test_reopening_live_token_cannot_silently_add_or_replace_aliases():
    coordinator = VoiceTurnAdmissionCoordinator()
    token = _token(1)
    await coordinator.open_turn(token)
    with pytest.raises(AdmissionIdentityError, match="ALIAS_CONFLICT"):
        await coordinator.open_turn(token, provider_key=ProviderUtteranceKey(1, 0, 1))
