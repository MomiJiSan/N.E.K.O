"""Loopback streaming ASR adapter; endpoint and transcript ownership stay in Core.

The reader fences clear immediately, the actor preserves command order, and the
receiver translates service events. Dequeued PCM remains charged to Core until
the matching service consumed receipt (or completed cancellation).
"""
from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import websockets

from .. import local_service
from .._infra import AsrSessionConfig, _AsrWorkerEvent, _AsrWorkerRequest
from ..delivery import (
    begin_transport_write,
    complete_transport_write,
    retire_interval_deliveries,
)

LOCAL_ASR_URL_ENV = "NEKO_ASR_URL"
LOCAL_ASR_TOKEN_ENV = "NEKO_ASR_TOKEN"
DEFAULT_LOCAL_ASR_URL = "ws://127.0.0.1:50120/v1/asr/stream"
_PACKET_BYTES = 6400
_WINDOW_BYTES = 64000
_HEADER = struct.Struct("!4s16sIQ")
_TIMEOUT = 10.0
_Key = tuple[int, int, int]


class _Failure(Exception):
    def __init__(self, code: str = "ASR_LOCAL_PROTOCOL_ERROR") -> None:
        self.code = code
        super().__init__(code)


def _endpoint_snapshot():
    try:
        return local_service.get_endpoint()
    except local_service.LocalServiceError as error:
        raise _Failure(error.code) from None


def _configuration() -> tuple[str, str]:
    endpoint = _endpoint_snapshot()
    return endpoint.url, endpoint.token


def configuration_is_available() -> bool:
    """Pure configuration probe; never connect, load a model, or log credentials."""
    return local_service.configuration_is_available()


def normalize_language(language: str) -> str | None:
    """Validate a hint against the fixed bilingual model, without inventing support."""
    base = str(language or "auto").strip().replace("_", "-").lower().split("-", 1)[0]
    if base == "auto":
        return None
    if base not in {"zh", "en"}:
        raise ValueError("ASR_LANGUAGE_NOT_SUPPORTED")
    return base


def _uuid(value: Any) -> bool:
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        return False


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _message(raw: Any) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError

    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 8192:
            raise ValueError
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
        if not isinstance(result, dict) or not isinstance(result.get("type"), str):
            raise ValueError
        return result
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _Failure() from error


@dataclass(eq=False)
class _Work:
    request: _AsrWorkerRequest
    hold: Any = None
    frames: int = 0
    sent: bool = False
    done: bool = False


@dataclass
class _Stream:
    key: _Key
    identity: str = field(default_factory=lambda: str(uuid4()))
    opened: asyncio.Event = field(default_factory=asyncio.Event)
    terminal: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_requested: asyncio.Event = field(default_factory=asyncio.Event)
    window: asyncio.Event = field(default_factory=asyncio.Event)
    suppressed: bool = False
    committed: bool = False
    terminal_state: str = ""
    sequence: int = 0
    samples: int = 0
    accepted: int = 0
    consumed: int = 0
    consumed_samples: int = 0
    revision: int = 0
    frames: dict[int, tuple[int, _Work]] = field(default_factory=dict)
    inflight_bytes: int = 0
    binding: tuple[str, int, str] | None = None


class _Worker:
    def __init__(self, requests, responses, config, connector):
        self.requests, self.responses, self.config = requests, responses, config
        self.connector = connector
        self.ws = None
        self.receiver = None
        self.stream: _Stream | None = None
        self.scope: tuple[int, int] | None = None
        self.last = _AsrWorkerRequest(kind="shutdown", generation=0)
        self.last_key: _Key | None = None
        self.works: set[_Work] = set()
        self.queue: asyncio.Queue[_Work] = asyncio.Queue()
        self.finished = asyncio.Event()
        self.finish_requested = False
        self.stopping = False
        self.connection_streams = 0
        self.retired_ids: set[str] = set()
        self.rotating = False
        self.connection_ready = asyncio.Event()
        endpoint = _endpoint_snapshot()
        self.url, self.token = endpoint.url, endpoint.token
        self.supervisor_instance = endpoint.instance_id
        self.per_utterance_binding = False
        self.last_binding: tuple[str, int, str] | None = None

    async def send(self, payload):
        async with asyncio.timeout(2):
            await self.ws.send(json.dumps(payload) if isinstance(payload, dict) else payload)

    async def connect(self):
        async with asyncio.timeout(_TIMEOUT):
            self.ws = await self.connector(
                self.url, additional_headers={"Authorization": "Bearer " + self.token},
                proxy=None, open_timeout=5, close_timeout=2, ping_interval=15,
                max_size=8192, max_queue=16, compression=None,
            )
            hello = _message(await self.ws.recv())
            if (hello.get("type") != "hello" or type(hello.get("protocol_version")) is not int
                    or hello["protocol_version"] != 1
                    or not _uuid(hello.get("supervisor_instance")) or not _uuid(hello.get("worker_instance"))
                    or ("per_utterance_binding" in hello and type(hello["per_utterance_binding"]) is not bool)
                    or (self.supervisor_instance is not None and hello.get("supervisor_instance") != self.supervisor_instance)):
                raise _Failure()
            self.per_utterance_binding = hello.get("per_utterance_binding", False)
            self.last_binding = None
        self.connection_streams = 0
        self.retired_ids.clear()
        if self.receiver is None:
            self.receiver = asyncio.create_task(self.receive(), name="local-asr-receive")

    def release(self, work, *, discard=False):
        if discard or (work.sent and work.frames == 0):
            if work.hold is not None:
                work.hold.release()
                work.hold = None
            if work.done:
                self.works.discard(work)

    def done(self, work, *, discard=False):
        if not work.done:
            work.done = True
            self.requests.task_done()
        self.release(work, discard=discard)
        if work.hold is None and (discard or not work.frames):
            self.works.discard(work)

    def drop_frames(self, stream):
        for _end, work in stream.frames.values():
            work.frames -= 1
            self.release(work)
        stream.frames.clear()
        stream.inflight_bytes = 0
        stream.window.set()

    async def read_requests(self):
        while True:
            getter = getattr(self.requests, "get_with_audio_hold", None)
            if getter is None:
                request, hold = await self.requests.get(), None  # noqa: ASYNC_BLOCK # awaited asyncio request queue, not HTTP.
            else:
                request, hold = await getter()
            work = _Work(request=request, hold=hold)
            self.works.add(work)
            self.last = request
            if self.scope is None or request.kind == "clear":
                self.scope = (request.generation, request.buffer_epoch)
            if request.kind in {"clear", "shutdown"}:
                if request.kind == "shutdown":
                    self.stopping = True
                if self.stream is not None:
                    self.stream.suppressed = True
                    self.stream.cancel_requested.set()
                    self.stream.window.set()
            self.queue.put_nowait(work)
            if request.kind in {"shutdown", "finish"}:
                return

    async def wait(self, event):
        async with asyncio.timeout(_TIMEOUT):
            await event.wait()

    async def wait_final(self, stream):
        if stream.terminal.is_set():
            return
        final = asyncio.create_task(stream.terminal.wait())
        cancel = asyncio.create_task(stream.cancel_requested.wait())
        try:
            async with asyncio.timeout(_TIMEOUT):
                await asyncio.wait({final, cancel}, return_when=asyncio.FIRST_COMPLETED)
            if stream.cancel_requested.is_set():
                await self.cancel(stream)
        finally:
            for task in (final, cancel):
                task.cancel()
            await asyncio.gather(final, cancel, return_exceptions=True)

    async def cancel(self, stream):
        stream.suppressed = True
        stream.cancel_requested.set()
        if not stream.terminal.is_set():
            await self.wait(stream.opened)
            if not stream.terminal.is_set():
                await self.send({"type": "cancel", "stream_id": stream.identity})
                # Wait the acknowledgement even if final wins the server race;
                # otherwise the late cancel could target a successor's lease.
                await self.wait(stream.cancelled)
        self.drop_frames(stream)

    async def rotate(self):
        self.rotating = True
        self.connection_ready.clear()
        async with asyncio.timeout(3):
            await self.ws.close()
        await self.connect()
        self.rotating = False
        self.connection_ready.set()

    async def ensure_stream(self, key):
        previous = self.stream
        if previous is not None:
            if previous.key == key:
                if previous.committed or previous.terminal.is_set():
                    raise _Failure()
                return previous
            await self.wait_final(previous)
            self.retired_ids.add(previous.identity)
            self.stream = None
        if self.stopping or key[:2] != self.scope:
            return None
        if self.last_key is not None and key <= self.last_key:
            raise _Failure()
        if self.connection_streams >= 64:
            await self.rotate()
        stream = _Stream(key)
        self.stream = stream
        self.last_key = key
        self.connection_streams += 1
        await self.send({"type": "open", "protocol_version": 1, "stream_id": stream.identity})
        await self.wait(stream.opened)
        return stream

    async def audio(self, work, stream):
        request = work.request
        if not request.audio or len(request.audio) % 2 or stream.samples + len(request.audio) // 2 > 960000:
            raise _Failure()
        evidence = None
        for offset in range(0, len(request.audio), _PACKET_BYTES):
            pcm = request.audio[offset:offset + _PACKET_BYTES]
            while stream.inflight_bytes + len(pcm) > _WINDOW_BYTES or len(stream.frames) >= 256:
                if stream.cancel_requested.is_set():
                    break
                stream.window.clear()
                await self.wait(stream.window)
            if stream.cancel_requested.is_set() or stream.key[:2] != self.scope:
                return
            if evidence is None:
                evidence = begin_transport_write(self.requests, delivery_spans=request.delivery_spans)
            sequence = stream.sequence
            stream.sequence += 1
            start = stream.samples
            stream.samples += len(pcm) // 2
            stream.frames[sequence] = (stream.samples, work)
            stream.inflight_bytes += len(pcm)
            work.frames += 1
            await self.send(_HEADER.pack(b"NAS1", UUID(stream.identity).bytes, sequence, start) + pcm)
        # The registered sidecar owns the whole request. A failed partial send
        # stays UNKNOWN rather than claiming unregistered per-frame slices.
        complete_transport_write(evidence, len(request.audio), generation=request.generation,
                                 buffer_epoch=request.buffer_epoch, provider="local_streaming",
                                 delivery_spans=request.delivery_spans, takes_ownership=True)
        work.sent = True
        self.release(work)

    async def act(self):
        while True:
            work = await self.queue.get()
            request = work.request
            discard = True
            try:
                if request.kind in {"clear", "shutdown"}:
                    if self.stream is not None:
                        await self.cancel(self.stream)
                    retire_interval_deliveries(self.requests, keep_scope=self.scope if request.kind == "clear" else None)
                    if request.kind == "shutdown":
                        return
                    continue
                if request.kind == "finish":
                    stream = self.stream
                    if stream is not None and not stream.terminal.is_set():
                        if not stream.committed:
                            stream.committed = True
                            await self.send({"type": "commit", "stream_id": stream.identity,
                                             "last_seq": stream.sequence - 1, "sample_end": stream.samples})
                        await self.wait_final(stream)
                    self.finish_requested = True
                    await self.send({"type": "finish"})
                    await self.wait(self.finished)
                    await self.responses.put(_AsrWorkerEvent(kind="finished", generation=request.generation,
                                                            buffer_epoch=request.buffer_epoch,
                                                            utterance_id=request.utterance_id))
                    return
                if request.kind == "activity":
                    continue
                if request.kind not in {"audio", "commit"} or type(request.utterance_id) is not int:
                    raise _Failure()
                key = (request.generation, request.buffer_epoch, request.utterance_id)
                if self.stopping or key[:2] != self.scope:
                    continue
                stream = await self.ensure_stream(key)
                if stream is None:
                    continue
                if stream.cancel_requested.is_set() or key[:2] != self.scope:
                    await self.cancel(stream)
                    continue
                if request.kind == "audio":
                    await self.audio(work, stream)
                    work.sent = True  # Sending ended; remaining frame receipts still own the hold.
                    self.release(work)
                    discard = False
                else:
                    stream.committed = True
                    await self.send({"type": "commit", "stream_id": stream.identity,
                                     "last_seq": stream.sequence - 1, "sample_end": stream.samples})
            finally:
                self.done(work, discard=discard)
                self.queue.task_done()

    def validate_binding(self, event, stream, *, opening=False):
        if not self.per_utterance_binding:
            return
        worker, generation, bundle = event.get("worker_instance"), event.get("route_generation"), event.get("bundle_id")
        if (not _uuid(worker) or not _integer(generation, 0, 2**63 - 1)
                or not isinstance(bundle, str) or not 1 <= len(bundle) <= 256
                or not bundle.isprintable()):
            raise _Failure()
        binding = (worker, generation, bundle)
        if opening:
            previous = self.last_binding
            if previous is not None and (generation < previous[1] or (generation == previous[1] and binding != previous)):
                raise _Failure()
            descriptor = event.get("bundle", {})
            if descriptor.get("bundle_id", bundle) != bundle:
                raise _Failure()
            stream.binding = binding
            self.last_binding = binding
        elif stream.binding != binding:
            # Never attribute an old worker's callback to a new utterance.
            raise _Failure()

    async def receive(self):
        while True:
            if self.rotating:
                await self.connection_ready.wait()
            connection = self.ws
            try:
                raw = await connection.recv()
            except Exception:
                if connection is not self.ws:
                    continue
                if not self.rotating:
                    raise
                await self.connection_ready.wait()
                continue
            if connection is not self.ws:
                continue
            event = _message(raw)
            kind = event["type"]
            if kind == "error":
                identity = event.get("stream_id")
                stream = self.stream
                if (isinstance(identity, str) and identity in self.retired_ids) or (stream is not None and identity == stream.identity and stream.terminal.is_set()):
                    continue
                if stream is not None and identity == stream.identity and stream.opened.is_set():
                    self.validate_binding(event, stream)
                raise _Failure("ASR_LOCAL_SERVICE_UNAVAILABLE")
            if kind == "finished":
                if not self.finish_requested:
                    raise _Failure()
                self.finished.set()
                return
            stream = self.stream
            if stream is None or event.get("stream_id") != stream.identity:
                if isinstance(event.get("stream_id"), str) and event["stream_id"] in self.retired_ids:
                    continue
                raise _Failure()
            if kind == "cancelled" or (kind != "opened" and stream.opened.is_set() and not stream.terminal.is_set()):
                self.validate_binding(event, stream)
            if kind == "opened":
                bundle = event.get("bundle", {})
                caps = bundle.get("capabilities", {}) if isinstance(bundle, dict) else {}
                contract = bundle.get("finish_contract", {}) if isinstance(bundle, dict) else {}
                if (stream.opened.is_set() or type(event.get("protocol_version")) is not int
                        or event.get("protocol_version") != 1
                        or not _integer(event.get("max_buffer_bytes"), _WINDOW_BYTES, _WINDOW_BYTES)
                        or not isinstance(bundle, dict) or not isinstance(caps, dict) or not isinstance(contract, dict)
                        or not _integer(bundle.get("sample_rate_hz"), 16000, 16000)
                        or not _integer(bundle.get("channels"), 1, 1)
                        or bundle.get("pcm_format") != "pcm16le"
                        or caps.get("streaming") is not True or caps.get("manual_commit") is not True
                        or caps.get("partial_semantics") != "snapshot"
                        or not _integer(contract.get("version"), 1, 1)
                        or contract.get("fresh_stream_per_utterance") is not True):
                    raise _Failure()
                self.validate_binding(event, stream, opening=True)
                stream.opened.set()
            elif kind == "cancelled":
                if (not stream.opened.is_set() or not stream.cancel_requested.is_set()
                        or event.get("terminal_state", "cancelled") not in {"final", "cancelled"}
                        or (event.get("terminal_state") == "final" and stream.terminal_state != "final")):
                    raise _Failure()
                stream.terminal_state = event.get("terminal_state", "cancelled")
                stream.cancelled.set()
                stream.terminal.set()
                self.drop_frames(stream)
            elif stream.terminal.is_set():
                # Terminal tombstone blocks late partial and duplicate final.
                continue
            elif not stream.opened.is_set():
                raise _Failure()
            elif kind in {"accepted", "consumed"}:
                sequence = event.get("seq")
                expected = stream.accepted if kind == "accepted" else stream.consumed
                frame = stream.frames.get(sequence) if type(sequence) is int else None
                if (sequence != expected or frame is None
                        or not _integer(event.get("sample_end"), frame[0], frame[0])
                        or not _integer(event.get("available_bytes"), 0, _WINDOW_BYTES)):
                    raise _Failure()
                if kind == "accepted":
                    stream.accepted += 1
                else:
                    if sequence >= stream.accepted:
                        raise _Failure()
                    end, work = stream.frames.pop(sequence)
                    start = stream.consumed_samples
                    stream.inflight_bytes -= (end - start) * 2
                    stream.consumed_samples = end
                    stream.consumed += 1
                    work.frames -= 1
                    self.release(work)
                    stream.window.set()
            elif kind == "partial":
                if not _integer(event.get("revision"), stream.revision + 1, 2**31 - 1) or not isinstance(event.get("text"), str):
                    raise _Failure()
                stream.revision = event["revision"]
                if not stream.suppressed and stream.key[:2] == self.scope:
                    await self.responses.put(_AsrWorkerEvent(kind="partial", generation=stream.key[0],
                                                            buffer_epoch=stream.key[1], utterance_id=stream.key[2],
                                                            text=event["text"]))
            elif kind == "final":
                if (not stream.committed or stream.frames
                        or not _integer(event.get("sample_end"), stream.samples, stream.samples)
                        or not isinstance(event.get("text"), str)):
                    raise _Failure()
                stream.terminal_state = "final"
                stream.terminal.set()
                if not stream.suppressed and stream.key[:2] == self.scope:
                    await self.responses.put(_AsrWorkerEvent(kind="final", generation=stream.key[0],
                                                            buffer_epoch=stream.key[1], utterance_id=stream.key[2],
                                                            text=event["text"]))
            else:
                raise _Failure()

    async def run(self):
        tasks = []
        try:
            if self.config.endpointing_mode != "manual" or self.config.input_sample_rate_hz != 16000:
                raise _Failure("ASR_ENDPOINTING_NOT_SUPPORTED")
            try:
                normalize_language(self.config.language)
            except ValueError as error:
                raise _Failure("ASR_LANGUAGE_NOT_SUPPORTED") from error
            await self.connect()
            await self.responses.put(_AsrWorkerEvent(kind="ready", generation=0))
            reader = asyncio.create_task(self.read_requests(), name="local-asr-requests")
            actor = asyncio.create_task(self.act(), name="local-asr-actor")
            tasks = [reader, actor]
            while True:
                watched = {actor, self.receiver}
                if not reader.done():
                    watched.add(reader)
                elif not reader.cancelled() and reader.exception() is not None:
                    raise reader.exception()
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if actor in done:
                    return
                if self.receiver in done:
                    if not self.finished.is_set():
                        raise _Failure("ASR_LOCAL_SERVICE_UNAVAILABLE")
                    await actor
                    return
        finally:
            if self.receiver is not None:
                tasks.append(self.receiver)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.ws is not None:
                try:
                    async with asyncio.timeout(3):
                        await self.ws.close()
                except Exception:
                    pass
            retire_interval_deliveries(self.requests)
            for work in tuple(self.works):
                self.done(work, discard=True)
            while not self.requests.empty():
                self.requests.get_nowait()
                self.requests.task_done()


async def local_streaming_asr_worker(request_queue, response_queue, api_key: str,
                                     config: AsrSessionConfig, *, connect_factory=None) -> None:
    """Translate the v1 local service into the existing RealtimeAsrSession worker API."""
    worker = None
    try:
        worker = _Worker(request_queue, response_queue, config, connect_factory or websockets.connect)
        await worker.run()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        code = error.code if isinstance(error, _Failure) else (
            "ASR_LOCAL_SERVICE_TIMEOUT" if isinstance(error, TimeoutError) else "ASR_LOCAL_SERVICE_UNAVAILABLE")
        request = worker.last if worker is not None else _AsrWorkerRequest(kind="shutdown", generation=0)
        await response_queue.put(_AsrWorkerEvent(kind="error", generation=request.generation,
                                               buffer_epoch=request.buffer_epoch, utterance_id=request.utterance_id,
                                               error_code=code, error_message="Local ASR service could not complete the request"))
    finally:
        generation = worker.last.generation if worker is not None else 0
        await response_queue.put(_AsrWorkerEvent(kind="closed", generation=generation))
