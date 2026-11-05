"""Opt-in repeated voice restarts retaining one Core and external chat sink."""
import asyncio
import json
import os
import time
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from main_logic.asr_client import local_service
from main_logic.asr_client.workers import local_streaming
from main_logic.voice_turn.audio_input import ProcessedVoiceFrame
from tests.integration.test_local_streaming_asr_cycles import ProcessSampler
from tests.integration.test_local_streaming_asr_soak import (
    Socket,
    State,
    core_fixture,
    until,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration_serial]


async def test_repeated_voice_restarts_keep_same_core_and_chat_sink(tmp_path, monkeypatch):
    if os.getenv("NEKO_ASR_SAME_CORE") != "1":
        pytest.skip("Explicit NEKO_ASR_SAME_CORE=1 and managed paths required")
    monkeypatch.setenv("NEKO_ASR_IDLE_SECONDS", "0.4")
    config = local_service.configuration()
    assert isinstance(config, local_service.ManagedConfig)
    with wave.open(config.smoke_wav, "rb") as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        pcm = source.readframes(source.getnframes())
    audio = b"\0" * 16000 + pcm + b"\0" * 96000
    state = State()
    core = core_fixture(state, monkeypatch)
    runtime, chat_sink = core._asr_runtime, core.session
    original = local_streaming.websockets.connect

    async def observed(*args, **kwargs):
        state.counts["connections"] += 1
        return Socket(await original(*args, **kwargs), state)

    monkeypatch.setattr(local_streaming.websockets, "connect", observed)
    await local_service.startup()
    manager = local_service.get_manager()
    sampler = ProcessSampler()
    sampler.manager = manager
    worker_ids, turn_ids, stream_ids, epochs = set(), set(), set(), []
    result = {"passed": False, "cycles": [], "boundary": "One retained production Core/runtime and typed recorder chat sink; real ASR/VAD/Smart Turn and capture_end. No LLM, microphone or TTS."}
    output = Path(os.getenv("NEKO_ASR_SAME_CORE_REPORT", str(tmp_path / "same-core.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        for cycle in range(8):
            if cycle:
                # All previous owners have closed. Reset counters, retaining the
                # same State object captured by the Core's bounded observers.
                state.__init__()
            request = str(uuid4())
            prepared = await manager.acquire_ready(request)
            endpoint = manager.endpoint()
            worker = prepared["worker_instance"]
            assert worker not in worker_ids
            worker_ids.add(worker)
            core._set_microphone_route("independent")
            started = await runtime.start(route_key="qwen", resource_optimization_enabled=bool(cycle % 2),
                                          provider_preference="local_streaming", user_language="zh-CN")
            assert started.provider == "local_streaming" and runtime._asr_session.is_ready
            assert core._asr_runtime is runtime and core.session is chat_sink
            epochs.append(runtime._asr_session_epoch)
            assert len(set(epochs)) == len(epochs)
            await manager.cancel_request(request)
            token = core._capture_ingress_token()
            deadline = time.monotonic()
            async with asyncio.timeout(90):
                while (not state.counts["core_handoffs"] or state.final_queue or state.handoff_queue
                       or any(stream["first_partial"] for stream in state.streams.values())):
                    for offset in range(0, len(audio), 1024):
                        assert not state.errors, list(state.errors)
                        frame = audio[offset:offset + 1024]
                        await runtime.submit(ProcessedVoiceFrame(frame, 16000, None), ingress_token=token)
                        deadline += len(frame) / 32000
                        await asyncio.sleep(max(0, deadline - time.monotonic()))
                    await runtime.wait_transcript_idle()
            assert state.counts["final"] == state.counts["formal_notifications"] == state.counts["core_handoffs"] == state.counts["sync_messages"]
            assert not (turn_ids & state.all_turn_ids) and not (stream_ids & state.all_stream_ids)
            turn_ids.update(state.all_turn_ids)
            stream_ids.update(state.all_stream_ids)
            assert await core._handle_voice_input_control("capture_end", core._voice_lease_generation + 1)
            await runtime.close()
            if core._bg_tasks:
                await asyncio.gather(*core._bg_tasks)
            assert runtime._asr_resource_lease is None and not manager._leases

            async def unloaded():
                health = await manager._request(endpoint, "GET", "/v1/health", statuses=(200, 503))
                return health["state"] == "stopped" and not health["ready"]

            await until(unloaded)
            result["cycles"].append({"cycle": cycle + 1, "epoch": epochs[-1], "worker": worker,
                                     "finals": state.counts["final"], "handoffs": state.counts["core_handoffs"],
                                     "resource_released": True, "idle_unloaded": True})
        result["passed"] = True
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            await runtime.close()
            await local_service.close()
            sampler.manager = None
            await sampler.close()
            result["owned_processes_exited"] = True
        except BaseException:
            result["passed"] = False
            raise
        finally:
            result.update(elapsed_seconds=time.monotonic() - began, epochs=epochs,
                          unique_workers=len(worker_ids), unique_turns=len(turn_ids),
                          observed_owned_processes=sampler.retired)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
