"""Pending Smart Turn preparation owns bounded, still-unsent PCM."""
import asyncio
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from main_logic.asr_client.endpointing.detector_runtime import DetectorFeedResult
from main_logic.asr_client.lifecycle import AudioDisposition
from main_logic.asr_client.runtime import IndependentAsrRuntime
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from main_logic.voice_turn.contracts import AsrSubmitStatus
from tests.unit.test_asr_client import (
    _install_pending_runtime_state,
    _runtime_callbacks,
    _RuntimeDetectorStub,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.runtime]


def _pending():
    entered, release = asyncio.Event(), asyncio.Event()
    failures, sites, leases = [], [], []

    class Detector(_RuntimeDetectorStub):
        admission_enabled = False

        async def prepare_endpointing(self, token):
            entered.set()
            await release.wait()
            lease = await super().prepare_endpointing(token)
            leases.append(lease)
            return lease

        async def feed(self, *args, **kwargs):
            return DetectorFeedResult((), True)

    runtime = IndependentAsrRuntime(_runtime_callbacks(failures=failures))
    _install_pending_runtime_state(runtime, Detector())
    session = runtime._asr_session
    session.stream_audio = AsyncMock()
    original_error = runtime._handle_independent_asr_error

    async def observed_error(*args, **kwargs):
        sites.extend(f"{item.name}:{item.lineno}" for item in traceback.extract_stack()[-4:])
        return await original_error(*args, **kwargs)

    runtime._handle_independent_asr_error = observed_error
    return SimpleNamespace(runtime=runtime, lifecycle=runtime._asr_lifecycle, session=session,
                           entered=entered, release=release, failures=failures, sites=sites, leases=leases)


async def _submit(rig, value):
    return await rig.runtime.submit(ProcessedVoiceFrame(value, 16000, None),
                                    ingress_token=rig.runtime._asr_current_ingress_token)


async def _activation(rig):
    task = asyncio.create_task(rig.runtime._activate_pending_independent_turn(rig.runtime._asr_session_epoch))
    await asyncio.wait_for(rig.entered.wait(), 1)
    return task


async def _close(rig, activation):
    rig.release.set()
    await asyncio.gather(activation, return_exceptions=True)
    await rig.runtime.close()


async def test_pending_activation_buffers_two_anonymous_frames_then_flushes_once_in_order():
    rig = _pending()
    activation = await _activation(rig)
    try:
        prefix, second, third = b"\1\0" * 160, b"\2\0" * 160, b"\3\0" * 160
        for frame in (second, third):
            result = await _submit(rig, frame)
            assert not rig.failures, (rig.failures, rig.sites)
            assert result.status is AsrSubmitStatus.ACCEPTED
        assert rig.lifecycle.peek_active_start_audio() == prefix + second + third
        assert len(rig.lifecycle.peek_active_start_audio()) <= rig.lifecycle.prefix_capacity_bytes
        assert all(span.start is None for span in rig.lifecycle.peek_active_start_spans())
        assert not rig.runtime._asr_input_delivery.enabled, "Do not invent delivery ownership for legacy PCM"
        rig.session.stream_audio.assert_not_awaited()
        rig.release.set()
        await asyncio.wait_for(activation, 1)
        await rig.runtime._asr_audio_dispatcher.wait_idle()
        assert not rig.failures, (rig.failures, rig.sites)
        assert b"".join(call.args[0] for call in rig.session.stream_audio.await_args_list) == prefix + second + third
        assert rig.lifecycle.peek_active_start_audio() == b""
        assert rig.runtime._asr_pending_activation_turn is None
    finally:
        await _close(rig, activation)


async def test_pending_activation_capacity_blocks_without_evicting_or_sending_prefix():
    rig = _pending()
    activation = await _activation(rig)
    try:
        prefix = rig.lifecycle.peek_active_start_audio()
        remaining = rig.lifecycle.prefix_capacity_bytes - len(prefix)
        rest = b"\2\0" * (remaining // 2)
        assert (await _submit(rig, rest)).status is AsrSubmitStatus.ACCEPTED
        assert rig.lifecycle.peek_active_start_audio() == prefix + rest
        decision = rig.lifecycle.buffer_active_start_audio(b"\3\0", start_sample=None)
        assert decision.disposition is AudioDisposition.BLOCK and decision.backpressure
        assert rig.lifecycle.peek_active_start_audio() == prefix + rest
        assert len(rig.lifecycle.peek_active_start_audio()) == rig.lifecycle.prefix_capacity_bytes
        rig.session.stream_audio.assert_not_awaited()
    finally:
        await _close(rig, activation)


async def test_invalidated_pending_activation_releases_prefix_and_never_sends_to_successor():
    rig = _pending()
    activation = await _activation(rig)
    try:
        assert (await _submit(rig, b"\2\0" * 160)).status is AsrSubmitStatus.ACCEPTED
        old_token = rig.runtime._asr_current_ingress_token
        assert len(rig.lifecycle.peek_active_start_audio()) == 640
        await rig.runtime.close()
        assert rig.runtime._asr_current_ingress_token != old_token
        assert rig.lifecycle._active_start_audio == b"" and rig.lifecycle._active_start_spans == ()
        rig.release.set()
        await asyncio.wait_for(activation, 1)
        rig.session.stream_audio.assert_not_awaited()
        assert rig.leases and rig.leases[0].release.await_count == 1
        assert not rig.failures
    finally:
        await _close(rig, activation)


@pytest.mark.parametrize("known_prefix", [False, True], ids=["anonymous-to-ranged", "ranged-to-anonymous"])
async def test_pending_prefix_does_not_mix_anonymous_and_tracked_audio(known_prefix):
    from main_logic.asr_client.lifecycle import (
        VoiceInputLifecycleController,
        VoiceLifecycleEvent,
        VoiceRouteMode,
    )
    from main_logic.asr_client.provider_policy import resolve_provider_policy

    lifecycle = VoiceInputLifecycleController(provider_policy=resolve_provider_policy("local_streaming", "manual"), shadow_mode=False)
    lifecycle.open(route_mode=VoiceRouteMode.INDEPENDENT)
    for event in (VoiceLifecycleEvent.SOFT_WAKE, VoiceLifecycleEvent.SPEECH_CONFIRMED, VoiceLifecycleEvent.TURN_SEALED):
        lifecycle.transition(event)
    lifecycle.mark_pending_turn_speech()
    prefix = b"\1\0" * 160
    lifecycle.accept_audio(prefix, sample_rate_hz=16000, start_sample=100 if known_prefix else None)
    lifecycle.transition(VoiceLifecycleEvent.PROVIDER_FINAL)
    assert lifecycle.begin_pending_turn() == prefix
    start = None if known_prefix else 260
    result = lifecycle.buffer_active_start_audio(b"\2\0" * 160, start_sample=start)
    assert result.disposition is AudioDisposition.BLOCK
    assert lifecycle.admission_failure_reason == "candidate_audio_range_discontinuous"
    assert lifecycle.peek_active_start_audio() == prefix
