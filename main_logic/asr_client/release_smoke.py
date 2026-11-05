"""Frozen host release gate: real managed sidecar and public PCM, no devices."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import wave
from pathlib import Path
from uuid import UUID, uuid4

import websockets

from .local_service import LocalServiceManager, ManagedConfig, configuration


def _read_pcm(filename):
    with wave.open(str(filename), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError("ASR_RELEASE_SMOKE_WAV_FORMAT")
        if not 1 <= source.getnframes() <= 30 * 16000:
            raise ValueError("ASR_RELEASE_SMOKE_WAV_DURATION")
        return source.readframes(source.getnframes())


async def _recognize(endpoint, pcm):
    identity = str(uuid4())
    partials = 0
    finals = []
    async with websockets.connect(endpoint.url, additional_headers={"Authorization": "Bearer " + endpoint.token},
                                  max_size=32768, open_timeout=10, proxy=None) as socket:
        async def receive():
            value = json.loads(await socket.recv())
            if not isinstance(value, dict) or value.get("type") == "error":
                raise ValueError("ASR_RELEASE_SMOKE_PROTOCOL")
            if value.get("stream_id", identity) != identity:
                raise ValueError("ASR_RELEASE_SMOKE_IDENTITY")
            return value

        hello = await receive()
        if hello.get("type") != "hello" or hello.get("protocol_version") != 1:
            raise ValueError("ASR_RELEASE_SMOKE_HELLO")
        await socket.send(json.dumps({"type": "open", "protocol_version": 1, "stream_id": identity}))
        opened = await receive()
        if opened.get("type") != "opened":
            raise ValueError("ASR_RELEASE_SMOKE_OPEN")
        end = 0
        for seq, offset in enumerate(range(0, len(pcm), 6400)):
            chunk = pcm[offset:offset + 6400]
            await socket.send(struct.pack("!4s16sIQ", b"NAS1", UUID(identity).bytes, seq, end) + chunk)
            end += len(chunk) // 2
            accepted = False
            while True:
                value = await receive()
                if value.get("type") == "partial":
                    partials += 1
                elif value.get("type") == "accepted":
                    if accepted or value.get("seq") != seq or value.get("sample_end") != end:
                        raise ValueError("ASR_RELEASE_SMOKE_ACCEPTED")
                    accepted = True
                elif value.get("type") == "consumed":
                    if not accepted or value.get("seq") != seq or value.get("sample_end") != end:
                        raise ValueError("ASR_RELEASE_SMOKE_CONSUMED")
                    break
                else:
                    raise ValueError("ASR_RELEASE_SMOKE_EARLY_TERMINAL")
        await socket.send(json.dumps({"type": "commit", "stream_id": identity, "last_seq": seq, "sample_end": end}))
        while not finals:
            value = await receive()
            if value.get("type") == "partial":
                partials += 1
            elif value.get("type") == "final":
                finals.append(value)
            else:
                raise ValueError("ASR_RELEASE_SMOKE_FINAL")
        if finals[0].get("sample_end") != end or not str(finals[0].get("text", "")).strip():
            raise ValueError("ASR_RELEASE_SMOKE_FINAL_CONTENT")
        await socket.send(json.dumps({"type": "finish"}))
        if (await receive()).get("type") != "finished":
            raise ValueError("ASR_RELEASE_SMOKE_DUPLICATE_OR_UNFINISHED")
        text = finals[0]["text"]
        return {"samples_consumed": end, "partial_count": partials, "final_count": len(finals),
                "final_text_chars": len(text), "final_text_sha256": hashlib.sha256(text.encode()).hexdigest()}


async def _run():
    config = configuration()
    if not isinstance(config, ManagedConfig) or not config.executable:
        raise ValueError("ASR_RELEASE_SMOKE_REQUIRES_EXECUTABLE")
    pcm = await asyncio.to_thread(_read_pcm, config.smoke_wav)
    manager = LocalServiceManager(config)
    process = None
    async with asyncio.timeout(60):
        try:
            result = await manager.prepare(str(uuid4()))
            while not result.get("ready"):
                if result.get("state") in {"failed", "cancelled", "expired"}:
                    raise ValueError("ASR_RELEASE_SMOKE_PREPARE")
                await asyncio.sleep(.05)
                result = await manager.status(result["operation_id"])
            process = manager._owned.process
            recognition = await _recognize(manager.control_endpoint(), pcm)
            return {"passed": True, "scope": "Frozen host manager and real ASR WS; public PCM; no microphone, cloud LLM/TTS, screen or audible output",
                    "managed_executable": True, "pcm_sha256": hashlib.sha256(pcm).hexdigest(), **recognition}
        finally:
            await manager.close()
            if process is not None and process.returncode is None:
                raise ValueError("ASR_RELEASE_SMOKE_OWNED_PROCESS_REMAINED")


def main():
    try:
        result = asyncio.run(_run())
        result["owned_service_exited"] = True
    except Exception as error:
        detail = str(error)
        safe_code = detail if detail.startswith("ASR_RELEASE_SMOKE_") and detail.replace("_", "").isalnum() else "ASR_RELEASE_SMOKE_FAILED"
        result = {"passed": False, "error_type": type(error).__name__,
                  "code": getattr(error, "code", safe_code)}
    report = os.environ.get("NEKO_ASR_RELEASE_SMOKE_REPORT")
    if report:
        filename = Path(report)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1
