import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import main_logic.asr_client.runtime as runtime_module
from main_logic.asr_client.runtime import AsrRuntimeCallbacks, IndependentAsrRuntime
from tests.support.asr_fakes import _Runtime
from tests.unit.test_voice_admission_integration import make_runtime


def runtime():
    return IndependentAsrRuntime(AsrRuntimeCallbacks(
        display_name=lambda: "lease-test", on_prepare_turn=AsyncMock(return_value=True),
        on_partial=AsyncMock(), on_final=AsyncMock(), on_turn_abandoned=AsyncMock(),
        on_failure=AsyncMock(), on_status=AsyncMock(), on_lifecycle=AsyncMock(),
    ))


@pytest.mark.asyncio
async def test_voice_resource_survives_transport_retirement_and_closes_with_runtime():
    voice, _, session, _, _ = make_runtime(False)
    lease = SimpleNamespace(close=AsyncMock())
    voice._asr_resource_lease = lease
    # A warm transport is independent of the voice-mode resource owner.
    await voice._close_asr_session(session)
    lease.close.assert_not_awaited()
    await voice.close()
    lease.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_retired_resource_cleanup_cannot_close_successor():
    voice, _, _, _, _ = make_runtime(False)
    old = SimpleNamespace(close=AsyncMock())
    new = SimpleNamespace(close=AsyncMock())
    voice._asr_resource_lease = old
    cleanup = voice._detach_independent_asr()
    voice._asr_resource_lease = new
    await cleanup
    old.close.assert_awaited_once()
    new.close.assert_not_awaited()
    await voice.close()
    new.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_or_cancelled_start_releases_only_its_ready_resource(monkeypatch, cancel):
    voice = runtime()
    lease = SimpleNamespace(close=AsyncMock())
    entered = asyncio.Event()
    async def connect():
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("connection failed")
    session = SimpleNamespace(connect=connect, close=AsyncMock(), last_failure_code="ASR_LOCAL_NOT_PREPARED")
    selection = SimpleNamespace(provider_key="qwen", endpointing_mode="manual",
                                _resource_lease_factory=lambda: lease)
    monkeypatch.setattr(runtime_module, "_resolve_asr_selection", lambda *a, **k: selection)
    monkeypatch.setattr(runtime_module, "_create_asr_session_from_selection", lambda *a, **k: session)
    task = asyncio.create_task(voice.start(route_key="qwen", resource_optimization_enabled=False))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.failure_code == "ASR_INDEPENDENT_FAILED"
        lease.close.assert_awaited_once()
        assert voice._asr_resource_lease is None
    finally:
        await voice.close()


@pytest.mark.asyncio
async def test_unadopted_start_releases_resource_when_notification_declines_without_detach(monkeypatch):
    voice = runtime()
    lease = SimpleNamespace(close=AsyncMock())
    session = SimpleNamespace(connect=AsyncMock(), close=AsyncMock(), is_ready=True)
    selection = SimpleNamespace(provider_key="qwen", endpointing_mode="manual",
                                _resource_lease_factory=lambda: lease)
    monkeypatch.setattr(runtime_module, "_resolve_asr_selection", lambda *a, **k: selection)
    monkeypatch.setattr(runtime_module, "_create_asr_session_from_selection", lambda *a, **k: session)
    monkeypatch.setattr(runtime_module, "DetectorRuntime", lambda **k: SimpleNamespace(close=AsyncMock()))
    voice._send_asr_lifecycle_state = AsyncMock(return_value=False)
    try:
        result = await voice.start(route_key="qwen", resource_optimization_enabled=False)
        assert result.failure_code == "ASR_START_STALE"
        lease.close.assert_awaited_once()
        assert voice._asr_resource_lease is None
    finally:
        await voice.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", [True, False, "cancel"])
async def test_normal_capture_end_releases_resource_without_ending_llm_session(finish):
    core = _Runtime()
    lease = SimpleNamespace(close=AsyncMock())
    core._asr_runtime._asr_resource_lease = lease
    if finish == "cancel":
        core._finish_normal_microphone_capture = AsyncMock(side_effect=asyncio.CancelledError)
    elif finish is False:
        core._finish_normal_microphone_capture = AsyncMock(return_value=finish)
    try:
        result = core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
        if finish == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await result
        else:
            assert await result is finish
        lease.close.assert_awaited_once()
        assert core._asr_runtime._asr_resource_lease is None
    finally:
        await core._asr_runtime.close()


@pytest.mark.asyncio
async def test_terminal_failure_releases_resource_and_clears_renew_owner():
    voice, _, _, _, _ = make_runtime(False)
    lease = SimpleNamespace(close=AsyncMock())
    voice._asr_resource_lease = lease
    try:
        await voice._handle_independent_asr_error(voice._asr_session_epoch, "qwen", allow_recovery=False)
        await asyncio.gather(*tuple(voice._asr_close_tasks), return_exceptions=True)
        lease.close.assert_awaited_once()
        assert voice._asr_resource_lease is None
    finally:
        await voice.close()


@pytest.mark.asyncio
async def test_capture_end_late_cleanup_does_not_release_a_successor_resource():
    core = _Runtime()
    old, new = SimpleNamespace(close=AsyncMock()), SimpleNamespace(close=AsyncMock())
    core._asr_runtime._asr_resource_lease = old
    async def finish():
        cleanup = core._asr_runtime._detach_independent_asr()
        core._asr_runtime._asr_resource_lease = new
        await cleanup
        return True
    core._finish_normal_microphone_capture = finish
    try:
        assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
        old.close.assert_awaited_once()
        new.close.assert_not_awaited()
    finally:
        await core._asr_runtime.close()
