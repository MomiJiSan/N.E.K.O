from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from main_logic.asr_client._infra import (
    AsrSessionConfig,
    _AsrRequestQueue,
    _AsrWorkerRequest,
)
from main_logic.asr_client.workers import local_streaming as local
from main_logic.voice_turn.audio_delivery import AudioDeliverySpan, AudioDeliveryTag
from main_logic.voice_turn.interception_events import InterceptionDeliveryStage as Stage

_BUNDLE = {
    "sample_rate_hz": 16000, "channels": 1, "pcm_format": "pcm16le",
    "capabilities": {"streaming": True, "manual_commit": True, "partial_semantics": "snapshot"},
    "finish_contract": {"version": 1, "fresh_stream_per_utterance": True},
}


class _Socket:
    def __init__(self, *, consume=True, cancel=True, fail_frame=0, opened=True, final=True):
        self.incoming = asyncio.Queue()
        self.calls = []
        self.consume = consume
        self.auto_cancel = cancel
        self.auto_opened = opened
        self.auto_final = final
        self.fail_frame = fail_frame
        self.frames = []
        self.samples = {}
        self.closed = False
        self.emit(type="hello", protocol_version=1, supervisor_instance=str(uuid4()), worker_instance=str(uuid4()))

    def emit(self, **event):
        self.incoming.put_nowait(json.dumps(event))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def send(self, raw):
        self.calls.append(raw)
        if isinstance(raw, bytes):
            _magic, identity, sequence, start = local._HEADER.unpack_from(raw)
            from uuid import UUID
            identity = str(UUID(bytes=identity))
            self.frames.append((identity, sequence, start, raw[32:]))
            if len(self.frames) == self.fail_frame:
                raise OSError("never expose this secret or PCM")
            end = start + (len(raw) - 32) // 2
            self.samples[identity] = end
            self.emit(type="accepted", stream_id=identity, seq=sequence, sample_end=end,
                      available_bytes=64000 - len(raw) + 32)
            if self.consume:
                self.consumed(identity, sequence, end)
                self.emit(type="partial", stream_id=identity, revision=sequence + 1, text=f"partial {sequence}")
            return
        value = json.loads(raw)
        kind = value["type"]
        identity = value.get("stream_id")
        if kind == "open":
            self.samples[identity] = 0
            if self.auto_opened:
                self.opened(identity)
        elif kind == "commit":
            assert value["sample_end"] == self.samples[identity]
            if self.auto_final:
                self.emit(type="final", stream_id=identity, text="recognized" if value["sample_end"] else "",
                          sample_end=value["sample_end"])
        elif kind == "cancel" and self.auto_cancel:
            self.emit(type="cancelled", stream_id=identity, terminal_state="cancelled")
        elif kind == "finish":
            self.emit(type="finished")

    def opened(self, identity):
        self.emit(type="opened", protocol_version=1, stream_id=identity, bundle=_BUNDLE, max_buffer_bytes=64000)

    def consumed(self, identity, sequence, end):
        self.emit(type="consumed", stream_id=identity, seq=sequence, sample_end=end, available_bytes=64000)

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(OSError("closed"))

    def commands(self, kind):
        return [json.loads(value) for value in self.calls if isinstance(value, str) and json.loads(value)["type"] == kind]


class _Connect:
    def __init__(self, *sockets):
        self.sockets = list(sockets)
        self.calls = []

    async def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.sockets[len(self.calls) - 1]


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    monkeypatch.delenv("NEKO_ASR_URL", raising=False)


def _request(kind, *, epoch=0, utterance=1, audio=b"", spans=()):
    return _AsrWorkerRequest(kind=kind, generation=1, buffer_epoch=epoch,
                             utterance_id=utterance, audio=audio, delivery_spans=spans)


async def _event(responses, kind):
    async with asyncio.timeout(2):
        while True:
            event = await responses.get()
            if event.kind == "error":
                assert kind == "error", event
                return event
            if event.kind == kind:
                return event


async def _until(condition):
    async with asyncio.timeout(2):
        while not condition():
            await asyncio.sleep(0)


async def _start(socket=None, *, queue=None, connector=None, config=None):
    socket = socket or _Socket()
    connector = connector or _Connect(socket)
    requests = queue or _AsrRequestQueue()
    responses = asyncio.Queue()
    task = asyncio.create_task(local.local_streaming_asr_worker(
        requests, responses, "unused-core-cloud-key", config or AsrSessionConfig(), connect_factory=connector))
    await _event(responses, "ready")
    return requests, responses, task, socket, connector


async def _stop(requests, responses, task):
    if not task.done():
        await requests.put(_request("shutdown"))
    await asyncio.wait_for(task, 2)
    await asyncio.wait_for(requests.join(), 2)


@pytest.mark.parametrize("url", [
    "ws://example.com:1/v1/asr/stream", "wss://localhost:1/v1/asr/stream",
    "ws://127.0.0.1:1/v1/asr/stream?token=secret", "ws://secret@localhost:1/v1/asr/stream",
    "ws://localhost:1/v1/asr/stream#fragment", "ws://localhost:1/wrong", "ws://localhost:bad/v1/asr/stream",
    "ws://127.0.0.2:1/v1/asr/stream", "ws://localhost/v1/asr/stream",
])
def test_configuration_rejects_remote_credentials_query_and_invalid_authority(monkeypatch, url):
    monkeypatch.setenv("NEKO_ASR_URL", url)
    assert not local.configuration_is_available()


@pytest.mark.parametrize("token", ["", "short", "x" * 257, "x" * 31 + " ", "x" * 31 + "中"])
def test_configuration_rejects_invalid_token(monkeypatch, token):
    monkeypatch.setenv("NEKO_ASR_TOKEN", token)
    assert not local.configuration_is_available()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_configuration_accepts_explicit_loopback(monkeypatch, host):
    monkeypatch.setenv("NEKO_ASR_URL", f"ws://{host}:50120/v1/asr/stream")
    assert local.configuration_is_available()


async def test_stream_partial_commit_final_and_finish_use_existing_worker_identity():
    requests, responses, task, socket, connector = await _start()
    await requests.put(_request("audio", audio=bytes(1280)))
    partial = await _event(responses, "partial")
    assert (partial.generation, partial.buffer_epoch, partial.utterance_id) == (1, 0, 1)
    assert not socket.commands("commit")
    await requests.put(_request("commit"))
    final = await _event(responses, "final")
    assert final.text == "recognized"
    identity = socket.frames[0][0]
    socket.emit(type="partial", stream_id=identity, revision=2, text="late")
    socket.emit(type="final", stream_id=identity, sample_end=640, text="duplicate")
    await requests.put(_request("finish"))
    await _event(responses, "finished")
    await task
    assert [event.kind for event in list(responses._queue)] == ["closed"]
    assert requests.held_audio_bytes == 0
    assert connector.calls[0][1]["additional_headers"] == {"Authorization": "Bearer " + "x" * 32}
    assert connector.calls[0][1]["proxy"] is None
    assert "unused-core-cloud-key" not in str(connector.calls)


async def test_empty_commit_is_empty_final_and_utterances_get_fresh_wire_ids():
    requests, responses, task, socket, _ = await _start()
    await requests.put(_request("commit"))
    assert (await _event(responses, "final")).text == ""
    await requests.put(_request("audio", utterance=2, audio=bytes(1280)))
    await requests.put(_request("commit", utterance=2))
    assert (await _event(responses, "final")).text == "recognized"
    assert len({command["stream_id"] for command in socket.commands("open")}) == 2
    await _stop(requests, responses, task)


async def test_inflight_hold_is_released_by_consumed_not_accepted():
    requests, responses, task, socket, _ = await _start(_Socket(consume=False))
    await requests.put(_request("audio", audio=bytes(6400)))
    await _until(lambda: len(socket.frames) == 1)
    await asyncio.sleep(0)
    assert requests.waiting_audio_bytes == 6400
    identity, seq, _sample_start, _pcm = socket.frames[0]
    socket.consumed(identity, seq, 3200)
    await _until(lambda: requests.waiting_audio_bytes == 0)
    await _stop(requests, responses, task)


async def test_split_whole_request_has_exact_sample_positions_and_safe_source_evidence():
    stages = []
    tag = AudioDeliveryTag(7000, lambda stage: stages.append(stage))
    spans = (AudioDeliverySpan(tag, 0, 7000),)
    requests, responses, task, socket, _ = await _start(_Socket(consume=False))
    await requests.put(_request("audio", audio=bytes(14000), spans=spans))
    await _until(lambda: len(socket.frames) == 3)
    assert [(seq, start, len(pcm)) for _id, seq, start, pcm in socket.frames] == [(0, 0, 6400), (1, 3200, 6400), (2, 6400, 1200)]
    assert Stage.TRANSPORT_OWNED in stages
    assert Stage.PROVIDER_CONFIRMED not in stages
    assert requests.held_audio_bytes == 14000
    for identity, seq, start, pcm in socket.frames[:2]:
        socket.consumed(identity, seq, start + len(pcm) // 2)
    await asyncio.sleep(0)
    assert requests.held_audio_bytes == 14000
    identity, seq, start, pcm = socket.frames[2]
    socket.consumed(identity, seq, start + len(pcm) // 2)
    await _until(lambda: requests.held_audio_bytes == 0)
    await _stop(requests, responses, task)


async def test_partial_send_failure_is_unknown_and_never_replayed():
    stages = []
    tag = AudioDeliveryTag(7000, lambda stage: stages.append(stage))
    requests, responses, task, socket, connector = await _start(_Socket(fail_frame=2))
    await requests.put(_request("audio", audio=bytes(14000), spans=(AudioDeliverySpan(tag, 0, 7000),)))
    error = await _event(responses, "error")
    await task
    assert error.error_code == "ASR_LOCAL_SERVICE_UNAVAILABLE"
    assert Stage.UNKNOWN in stages
    assert Stage.TRANSPORT_OWNED not in stages
    assert Stage.PROVIDER_CONFIRMED not in stages
    assert len(connector.calls) == 1 and len(socket.frames) == 2
    assert "secret" not in error.error_message
    assert requests.held_audio_bytes == 0 and socket.closed
    await requests.join()


async def test_clear_fences_old_partial_before_cancel_ack_and_isolates_next_scope():
    requests, responses, task, socket, _ = await _start(_Socket(consume=False, cancel=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _until(lambda: len(socket.frames) == 1)
    old_identity = socket.frames[0][0]
    await requests.put(_request("clear", epoch=1))
    await requests.put(_request("audio", epoch=1, audio=bytes(1280)))
    await _until(lambda: len(socket.commands("cancel")) == 1)
    socket.emit(type="partial", stream_id=old_identity, revision=1, text="must not leak")
    await asyncio.sleep(0)
    assert responses.empty()
    assert len(socket.commands("open")) == 1
    assert requests.waiting_audio_bytes == 2560
    socket.emit(type="cancelled", stream_id=old_identity, terminal_state="cancelled")
    await _until(lambda: len(socket.frames) == 2)
    new_identity, seq, _sample_start, pcm = socket.frames[-1]
    assert new_identity != old_identity
    socket.consumed(new_identity, seq, len(pcm) // 2)
    await requests.put(_request("commit", epoch=1))
    final = await _event(responses, "final")
    assert final.buffer_epoch == 1
    assert requests.held_audio_bytes == 0
    socket.auto_cancel = True
    await _stop(requests, responses, task)


async def test_unsolicited_final_without_commit_fails_closed():
    requests, responses, task, socket, _ = await _start(_Socket(cancel=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _event(responses, "partial")
    identity = socket.frames[0][0]
    await requests.put(_request("clear", epoch=1))
    await _until(lambda: len(socket.commands("cancel")) == 1)
    socket.emit(type="final", stream_id=identity, sample_end=640, text="unsolicited")
    error = await _event(responses, "error")
    assert error.error_code == "ASR_LOCAL_PROTOCOL_ERROR"
    await task


async def test_final_wins_cancel_race_without_replaying_or_leaking_old_result():
    requests, responses, task, socket, _ = await _start(_Socket(cancel=False, final=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _event(responses, "partial")
    identity = socket.frames[0][0]
    await requests.put(_request("commit"))
    await _until(lambda: len(socket.commands("commit")) == 1)
    await requests.put(_request("clear", epoch=1))
    await _until(lambda: len(socket.commands("cancel")) == 1)
    socket.emit(type="final", stream_id=identity, sample_end=640, text="must not leak")
    socket.emit(type="cancelled", stream_id=identity, terminal_state="final")
    socket.auto_final = True
    socket.auto_cancel = True
    await requests.put(_request("audio", epoch=1, audio=bytes(1280)))
    await requests.put(_request("commit", epoch=1))
    result = await _event(responses, "final")
    assert result.buffer_epoch == 1 and result.text == "recognized"
    assert len(socket.commands("open")) == 2 and len(socket.frames) == 2
    await _stop(requests, responses, task)


async def test_clear_during_open_waits_old_ack_and_never_sends_old_pcm():
    requests, responses, task, socket, _ = await _start(_Socket(opened=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _until(lambda: len(socket.commands("open")) == 1)
    identity = socket.commands("open")[0]["stream_id"]
    await requests.put(_request("clear", epoch=1))
    await requests.put(_request("audio", epoch=1, audio=bytes(1280)))
    await _until(lambda: requests.held_audio_bytes == 2560)
    socket.auto_opened = True
    socket.opened(identity)
    await _until(lambda: len(socket.frames) == 1)
    assert socket.frames[0][0] != identity
    await requests.put(_request("commit", epoch=1))
    assert (await _event(responses, "final")).buffer_epoch == 1
    await _stop(requests, responses, task)


async def test_clear_while_next_utterance_waits_final_never_opens_superseded_sentence():
    requests, responses, task, socket, _ = await _start(_Socket(final=False, cancel=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _event(responses, "partial")
    old = socket.frames[0][0]
    await requests.put(_request("commit"))
    await _until(lambda: len(socket.commands("commit")) == 1)
    await requests.put(_request("audio", utterance=2, audio=bytes(1280)))
    await requests.put(_request("clear", epoch=1))
    await _until(lambda: len(socket.commands("cancel")) == 1)
    socket.emit(type="cancelled", stream_id=old, terminal_state="cancelled")
    await _until(lambda: requests.held_audio_bytes == 0)
    assert len(socket.commands("open")) == 1
    socket.auto_final = True
    socket.auto_cancel = True
    await requests.put(_request("audio", epoch=1, audio=bytes(1280)))
    await requests.put(_request("commit", epoch=1))
    assert (await _event(responses, "final")).buffer_epoch == 1
    assert len(socket.commands("open")) == 2
    await _stop(requests, responses, task)


async def test_shutdown_fences_queued_audio_before_native_open():
    queue = _AsrRequestQueue()
    queue.put_nowait(_request("audio", audio=bytes(1280)))
    queue.put_nowait(_request("shutdown"))
    requests, responses, task, socket, _ = await _start(queue=queue)
    await task
    assert not socket.commands("open") and not socket.frames
    assert requests.held_audio_bytes == 0
    await requests.join()


class _BarrierSocket(_Socket):
    def __init__(self):
        super().__init__()
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()

    async def send(self, raw):
        if isinstance(raw, bytes) and len(self.frames) == 1:
            self.second_started.set()
            await self.release_second.wait()
        await super().send(raw)


async def test_clear_during_multiframe_write_seals_hold_without_false_delivery_success():
    stages = []
    tag = AudioDeliveryTag(10000, lambda stage: stages.append(stage))
    requests, responses, task, socket, _ = await _start(_BarrierSocket())
    await requests.put(_request("audio", audio=bytes(20000), spans=(AudioDeliverySpan(tag, 0, 10000),)))
    await socket.second_started.wait()
    await requests.put(_request("clear", epoch=1))
    await _until(lambda: Stage.UNKNOWN in stages)
    socket.release_second.set()
    await _until(lambda: len(socket.commands("cancel")) == 1)
    await _until(lambda: requests.held_audio_bytes == 0)
    assert len(socket.frames) == 2
    assert Stage.TRANSPORT_OWNED not in stages
    await requests.put(_request("audio", epoch=1, audio=bytes(1280)))
    await requests.put(_request("commit", epoch=1))
    assert (await _event(responses, "final")).buffer_epoch == 1
    await _stop(requests, responses, task)


async def test_unsolicited_finished_cannot_leave_ready_worker_hanging():
    requests, responses, task, socket, _ = await _start()
    socket.emit(type="finished")
    assert (await _event(responses, "error")).error_code == "ASR_LOCAL_PROTOCOL_ERROR"
    await task
    assert socket.closed


async def test_partial_before_opened_is_rejected():
    requests, responses, task, socket, _ = await _start(_Socket(opened=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _until(lambda: len(socket.commands("open")) == 1)
    identity = socket.commands("open")[0]["stream_id"]
    socket.emit(type="partial", stream_id=identity, revision=1, text="invalid")
    assert (await _event(responses, "error")).error_code == "ASR_LOCAL_PROTOCOL_ERROR"
    await task


@pytest.mark.parametrize(("hint", "expected"), [("auto", None), ("zh-CN", "zh"), ("en-US", "en")])
def test_supported_language_hints(hint, expected):
    assert local.normalize_language(hint) == expected


def test_unsupported_language_hint_is_explicit():
    with pytest.raises(ValueError, match="ASR_LANGUAGE_NOT_SUPPORTED"):
        local.normalize_language("ja")


async def test_connection_rotates_only_between_utterances_after_64_without_replay():
    first, second = _Socket(), _Socket()
    connector = _Connect(first, second)
    requests, responses, task, _socket, _ = await _start(connector=connector)
    for number in range(1, 66):
        await requests.put(_request("audio", utterance=number, audio=bytes(1280)))
        await requests.put(_request("commit", utterance=number))
        assert (await _event(responses, "final")).utterance_id == number
    assert len(connector.calls) == 2
    assert len(first.frames) == 64 and len(second.frames) == 1
    assert first.closed and not second.closed
    await _stop(requests, responses, task)
    assert second.closed


@pytest.mark.parametrize("event", [
    {"type": "accepted", "seq": 1, "sample_end": 640, "available_bytes": 64000},
    {"type": "consumed", "seq": 0, "sample_end": 999, "available_bytes": 64000},
    {"type": "partial", "revision": True, "text": "invalid"},
    {"type": "mystery"},
])
async def test_malformed_receipts_fail_closed(event):
    requests, responses, task, socket, connector = await _start(_Socket(consume=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _until(lambda: len(socket.frames) == 1)
    socket.emit(stream_id=socket.frames[0][0], **event)
    assert (await _event(responses, "error")).error_code == "ASR_LOCAL_PROTOCOL_ERROR"
    await task
    assert requests.held_audio_bytes == 0
    assert len(connector.calls) == 1


async def test_missing_opened_ack_times_out_and_releases_every_hold(monkeypatch):
    monkeypatch.setattr(local, "_TIMEOUT", 0.03)
    requests, responses, task, socket, _ = await _start(_Socket(opened=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    assert (await _event(responses, "error")).error_code == "ASR_LOCAL_SERVICE_TIMEOUT"
    await task
    assert requests.held_audio_bytes == 0 and socket.closed


async def test_worker_cancellation_closes_transport_and_releases_owned_and_queued_pcm():
    requests, responses, task, socket, _ = await _start(_Socket(consume=False))
    await requests.put(_request("audio", audio=bytes(1280)))
    await _until(lambda: len(socket.frames) == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert requests.held_audio_bytes == 0 and socket.closed
    await requests.join()
