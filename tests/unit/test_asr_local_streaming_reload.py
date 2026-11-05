"""Per-utterance model binding preserves the existing delivery state machine."""
import asyncio
from uuid import uuid4

import pytest

from tests.unit.test_asr_local_streaming_worker import (
    _event,
    _request,
    _Socket,
    _start,
    _stop,
    _until,
)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    monkeypatch.delenv("NEKO_ASR_URL", raising=False)
    for name in ("NEKO_ASR_SERVICE_PYTHON", "NEKO_ASR_SERVICE_ROOT", "NEKO_ASR_MODEL_DIR", "NEKO_ASR_SMOKE_WAV"):
        monkeypatch.delenv(name, raising=False)


class BindingSocket(_Socket):
    def __init__(self, *, bad_kind=None, **kwargs):
        self.bindings = {}
        self.next_binding = (str(uuid4()), 1, "version-a")
        self.bad_kind = bad_kind
        super().__init__(**kwargs)

    def emit(self, **event):
        kind, identity = event["type"], event.get("stream_id")
        if kind == "hello":
            event["per_utterance_binding"] = True
        if kind == "opened":
            self.bindings[identity] = self.next_binding
            event["bundle"] = {**event["bundle"], "bundle_id": self.next_binding[2]}
        if identity in self.bindings:
            worker, generation, bundle = self.bindings[identity]
            event.setdefault("worker_instance", worker)
            event.setdefault("route_generation", generation)
            event.setdefault("bundle_id", bundle)
        if kind == self.bad_kind:
            event["worker_instance"] = str(uuid4())
        super().emit(**event)


@pytest.mark.asyncio
async def test_one_connection_can_publish_and_rollback_only_between_utterances():
    socket = BindingSocket()
    requests, responses, task, _, _ = await _start(socket)
    finals = []
    try:
        worker_a = socket.next_binding[0]
        for utterance, binding in enumerate([(worker_a, 1, "version-a"),
                                              (str(uuid4()), 2, "version-b"),
                                              (worker_a, 3, "version-a")], 1):
            socket.next_binding = binding
            await requests.put(_request("audio", utterance=utterance, audio=b"\1\0" * 640))
            await requests.put(_request("commit", utterance=utterance))
            finals.append(await _event(responses, "final"))
        assert [event.utterance_id for event in finals] == [1, 2, 3]
        assert len({command["stream_id"] for command in socket.commands("open")}) == 3
        assert len(socket.frames) == 3
        assert not task.done()
    finally:
        await _stop(requests, responses, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["accepted", "consumed", "partial", "final", "cancelled"])
async def test_worker_identity_drift_within_one_utterance_fails_closed_and_releases_holds(kind):
    socket = BindingSocket(bad_kind=kind)
    requests, responses, task, _, _ = await _start(socket)
    try:
        await requests.put(_request("audio", audio=b"\1\0" * 640))
        if kind in {"partial", "final", "cancelled"}:
            await _until(lambda: len(socket.frames) == 1)
        identity = socket.commands("open")[0]["stream_id"] if socket.commands("open") else None
        if kind == "partial":
            socket.emit(type="partial", stream_id=identity, revision=1, text="wrong owner")
        elif kind == "final":
            await requests.put(_request("commit"))
        elif kind == "cancelled":
            await requests.put(_request("clear", epoch=1))
        error = await _event(responses, "error")
        assert error.error_code == "ASR_LOCAL_PROTOCOL_ERROR"
        await asyncio.wait_for(task, 1)
        assert requests.waiting_audio_bytes == 0
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("generation", [0, 1])
async def test_a_new_worker_requires_a_new_route_generation(generation):
    socket = BindingSocket()
    requests, responses, task, _, _ = await _start(socket)
    try:
        await requests.put(_request("commit"))
        await _event(responses, "final")
        socket.next_binding = (str(uuid4()), generation, "version-b")
        await requests.put(_request("commit", utterance=2))
        assert (await _event(responses, "error")).error_code == "ASR_LOCAL_PROTOCOL_ERROR"
        await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_retired_partial_error_and_duplicate_final_never_corrupt_the_successor():
    socket = BindingSocket(final=False)
    requests, responses, task, _, _ = await _start(socket)
    try:
        await requests.put(_request("commit"))
        await _until(lambda: len(socket.commands("commit")) == 1)
        old = socket.commands("open")[0]["stream_id"]
        socket.emit(type="final", stream_id=old, text="first", sample_end=0)
        assert (await _event(responses, "final")).text == "first"
        socket.emit(type="partial", stream_id=old, text="late", revision=999, worker_instance=str(uuid4()))
        socket.next_binding = (str(uuid4()), 2, "version-b")
        await requests.put(_request("commit", utterance=2))
        await _until(lambda: len(socket.commands("commit")) == 2)
        new = socket.commands("open")[1]["stream_id"]
        socket.emit(type="error", stream_id=old, code="LATE_OLD_WORKER_FAILURE", terminal=True)
        socket.emit(type="final", stream_id=old, text="duplicate", sample_end=0)
        socket.emit(type="final", stream_id=new, text="second", sample_end=0)
        final = await _event(responses, "final")
        assert (final.utterance_id, final.text) == (2, "second")
        await requests.join()
        assert responses.empty()
        assert not task.done()
    finally:
        await _stop(requests, responses, task)
