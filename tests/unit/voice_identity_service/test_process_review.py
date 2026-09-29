import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from main_logic.voice_identity_service.pvad import models as ecapa
from main_logic.voice_identity_service.tse import worker as tse
from main_logic.voice_identity_service.tse.contracts import TseModelError
from main_logic.voice_identity_service.process_startup import (
    PendingStartupRetirement,
    start_owned_process,
)

pytestmark = pytest.mark.runtime


@pytest.fixture(params=["ecapa", "tse"])
def process_api(request, tmp_path):
    if request.param == "ecapa":
        return SimpleNamespace(
            module=ecapa,
            start="_start_ecapa_process",
            stop="_stop_ecapa_process",
            invoke=lambda timeout: ecapa.extract_activity_reference(
                tmp_path, bytes(48000), timeout=timeout
            ),
            failure=RuntimeError,
            retiring=ecapa.EcapaStartupRetirementError,
        )
    return SimpleNamespace(
        module=tse,
        start="_start_encoder_process",
        stop="_stop_encoder_process",
        invoke=lambda timeout: tse.extract_extraction_reference(
            tmp_path, [np.ones(48000)] * 3, timeout=timeout
        ),
        failure=TseModelError,
        retiring=tse.TseEncoderRetirementError,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["late_result", "empty", "eof", "os_error"])
async def test_process_exit_rechecks_pipe_and_normalizes_failed_receive(
    monkeypatch, process_api, outcome
):
    state = SimpleNamespace(polls=0, stopped=False, closed=False)

    def poll():
        state.polls += 1
        return outcome != "empty" and (outcome != "late_result" or state.polls >= 2)

    def recv():
        if outcome == "eof":
            raise EOFError()
        if outcome == "os_error":
            raise OSError("broken pipe")
        return True, np.ones(192, np.float32)

    receiver = SimpleNamespace(
        poll=poll, recv=recv, close=lambda: setattr(state, "closed", True)
    )
    process = SimpleNamespace(is_alive=lambda: False)
    monkeypatch.setattr(
        process_api.module, process_api.start, lambda *args: (process, receiver)
    )
    monkeypatch.setattr(
        process_api.module,
        process_api.stop,
        lambda proc: setattr(state, "stopped", True),
    )
    if outcome == "late_result":
        reference = await process_api.invoke(1)
        reference.close()
        assert state.polls >= 2
    else:
        with pytest.raises(process_api.failure):
            await process_api.invoke(1)
    assert state.stopped and state.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("startup_fails", [False, True])
async def test_blocked_startup_returns_owner_and_retires_after_waiter_cancel(
    monkeypatch,
    process_api,
    cancel,
    startup_fails,
):
    entered, release = threading.Event(), threading.Event()
    stopped, closed = threading.Event(), threading.Event()
    receiver = SimpleNamespace(close=closed.set)

    def start(*args):
        entered.set()
        assert release.wait(5)
        if startup_fails:
            raise RuntimeError("startup cleaned up then failed")
        return object(), receiver

    monkeypatch.setattr(process_api.module, process_api.start, start)
    monkeypatch.setattr(
        process_api.module, process_api.stop, lambda proc: stopped.set()
    )
    task = asyncio.create_task(process_api.invoke(1 if cancel else 0.03))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        if cancel:
            task.cancel()
        with pytest.raises(process_api.retiring) as raised:
            await asyncio.wait_for(task, 1)
        error = raised.value
        assert not error.retirement_owner.confirmed_stopped
        error.retirement_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await error.retirement_task
    finally:
        release.set()
    async with asyncio.timeout(2):
        while not error.retirement_owner.confirmed_stopped:
            await asyncio.sleep(0.01)
    assert stopped.is_set() is (not startup_fails)
    assert closed.is_set() is (not startup_fails)


@pytest.mark.asyncio
async def test_partial_startup_failure_keeps_handles_through_retirement_retry():
    started, release = threading.Event(), threading.Event()
    attempted, closed = threading.Event(), threading.Event()
    receiver = SimpleNamespace(close=closed.set)
    process = object()
    recordings = [np.ones(8)]

    def start():
        started.set()
        assert release.wait(5)
        raise tse._EncoderStartupRetirementPending(process, receiver)

    def stop(actual):
        assert actual is process
        if not attempted.is_set():
            attempted.set()
            raise TseModelError("retry native retirement")

    startup = start_owned_process(start)
    owner = PendingStartupRetirement(
        startup,
        stop,
        recordings=recordings,
        pending_error=tse._EncoderStartupRetirementPending,
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert not owner.confirmed_stopped
        assert recordings[0].all()
    finally:
        release.set()
    assert await asyncio.to_thread(attempted.wait, 1)
    await asyncio.wait_for(asyncio.shield(owner.retirement_task), 2)
    assert owner.confirmed_stopped and closed.is_set()
    assert not recordings[0].any()
