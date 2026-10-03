"""Failures retire actual resource workers and never publish partial bundles."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import threading
import tarfile
import time

import pytest

from config import voice_wake_word
from main_logic.voice_identity_service import resource_manager
from main_logic.voice_identity_service.enrollment import EnrollmentAudioError, validate_enrollment_pcm16
from main_logic.voice_identity_service import wake_word_bundle as model_bundle


def _archive(tmp_path, value=b"first"):
    path = tmp_path / (hashlib.sha256(value).hexdigest() + ".tar.bz2")
    with tarfile.open(path, "w:bz2") as bundle:
        for name in model_bundle.ASSETS:
            member = tarfile.TarInfo(f"{model_bundle.MODEL_NAME}/{name}")
            member.size = len(value)
            bundle.addfile(member, io.BytesIO(value))
    return path


def _install(tmp_path, monkeypatch, value=b"first"):
    source = _archive(tmp_path, value)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", digest)
    root = tmp_path / "cache"
    return root, model_bundle.install_bundle(root, source)


def test_pointer_publication_failure_keeps_old_complete_version(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    pointer = (root / "current.json").read_bytes()
    source = _archive(tmp_path, b"second")
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    replace = model_bundle.os.replace
    def fail_pointer(source, target):
        if Path(target).name == "current.json":
            raise PermissionError("Windows file lock")
        return replace(source, target)
    monkeypatch.setattr(model_bundle.os, "replace", fail_pointer)
    with pytest.raises(PermissionError):
        model_bundle.install_bundle(root, source)
    assert (root / "current.json").read_bytes() == pointer
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    assert not tuple(root.glob("stage-*"))


def test_prepare_failure_cannot_publish_or_change_preference(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    voice_wake_word.save_wake_word_preference(True, root)
    pointer = (root / "current.json").read_bytes()
    source = _archive(tmp_path, b"second")
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    def rejected(_path):
        raise RuntimeError("native model construction failed")
    with pytest.raises(RuntimeError):
        model_bundle.install_bundle(root, source, validate=rejected)
    assert (root / "current.json").read_bytes() == pointer
    assert voice_wake_word.wake_word_preference(root)["enabled"] is True
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)


def test_cancel_at_validation_boundary_never_publishes(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    pointer = (root / "current.json").read_bytes()
    source = _archive(tmp_path, b"second")
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    cancel = threading.Event()
    with pytest.raises(model_bundle.WakeWordBundleError, match="operation_cancelled"):
        model_bundle.install_bundle(root, source, cancel=cancel, validate=lambda _path: cancel.set())
    assert (root / "current.json").read_bytes() == pointer
    assert previous.is_dir()


def test_os_lock_rejects_concurrent_install_and_releases_after_error(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    root = tmp_path / "cache"
    root.mkdir()
    with model_bundle._installation_lock(root):
        with pytest.raises(model_bundle.WakeWordBundleError, match="resource_operation_busy"):
            model_bundle.install_bundle(root, source)
    assert model_bundle.install_bundle(root, source).is_dir()


def test_corrupt_pointer_and_changed_asset_are_not_ready(tmp_path, monkeypatch):
    root, installed = _install(tmp_path, monkeypatch)
    (installed / model_bundle.ASSETS[0]).write_bytes(b"corrupt")
    with pytest.raises(model_bundle.WakeWordBundleError, match="wake_model_invalid"):
        model_bundle.resolve_cached_model_dir(root)
    (root / "current.json").write_text(json.dumps({"schema": 1, "version": "../../outside", "model": model_bundle.MODEL_NAME}))
    with pytest.raises(model_bundle.WakeWordBundleError, match="wake_model_invalid"):
        model_bundle.resolve_cached_model_dir(root)


def test_cache_budget_retains_existing_version(tmp_path, monkeypatch):
    root, installed = _install(tmp_path, monkeypatch)
    pointer = (root / "current.json").read_bytes()
    monkeypatch.setattr(model_bundle, "MAX_CACHE_BYTES", 1)
    with pytest.raises(model_bundle.WakeWordBundleError, match="resource_cache_full"):
        model_bundle.install_bundle(root, _archive(tmp_path))
    assert (root / "current.json").read_bytes() == pointer
    assert installed.is_dir()


def test_download_does_not_enable_wake_words_and_corrupt_preference_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEKO_WAKE_WORD_ENABLED", raising=False)
    root, _installed = _install(tmp_path, monkeypatch)
    assert voice_wake_word.wake_word_preference(root)["enabled"] is False
    (root / "preference.json").write_bytes(b"broken")
    with pytest.raises(ValueError, match="wake_preference_unavailable"):
        voice_wake_word.save_wake_word_preference(True, root)
    assert (root / "preference.json").read_bytes() == b"broken"


def test_deployment_configuration_has_precedence_and_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("NEKO_WAKE_WORD_MODEL_DIR", "deployed-assets")
    assert voice_wake_word.wake_word_preference(tmp_path) == {"enabled": True, "managed": True, "reason": None}
    with pytest.raises(ValueError, match="wake_preference_managed"):
        voice_wake_word.save_wake_word_preference(False, tmp_path)
    monkeypatch.setenv("NEKO_WAKE_WORD_ENABLED", "false")
    assert voice_wake_word.wake_word_preference(tmp_path)["enabled"] is False


def test_insufficient_active_audio_retains_code_and_explains_duration():
    import numpy as np
    audio = np.zeros(48_000, dtype="<i2")
    audio[:16_000] = 4000
    with pytest.raises(EnrollmentAudioError) as caught:
        validate_enrollment_pcm16(audio.tobytes())
    assert caught.value.code == "volume_too_low"
    assert caught.value.diagnostics["active_seconds"] == 1.0
    assert caught.value.diagnostics["rms"] > .008
    assert audio[:16_000].min() == 4000


def _hung_worker(connection, kind, nr_enabled, wake_path, pcm16):
    Path(wake_path).write_text(str(multiprocessing.current_process().pid))
    while True:
        time.sleep(.1)


@pytest.mark.asyncio
async def test_worker_timeout_physically_retires_spawned_process(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_manager, "_resource_worker", _hung_worker)
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(resource_manager.VoiceResourceError, match="resource_prepare_timeout"):
        await resource_manager._run_worker("prepare", False, str(tmp_path / "started"), timeout=1)
    assert {child.pid for child in multiprocessing.active_children()} <= before


@pytest.mark.asyncio
async def test_worker_cancellation_retires_process_without_waiting_for_native_return(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_manager, "_resource_worker", _hung_worker)
    ready = tmp_path / "started"
    before = {child.pid for child in multiprocessing.active_children()}
    task = asyncio.create_task(resource_manager._run_worker("prepare", False, str(ready)))
    deadline = asyncio.get_running_loop().time() + 10
    while not ready.exists():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(.025)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert {child.pid for child in multiprocessing.active_children()} <= before


@pytest.mark.asyncio
async def test_late_prepare_result_cannot_replace_new_operation(monkeypatch, tmp_path):
    started = asyncio.Event()
    finish = asyncio.Event()
    async def controlled(*args, **kwargs):
        started.set()
        await finish.wait()
        return {"campp": {"state": "ready", "required": True, "reason": None}}
    monkeypatch.setattr(resource_manager, "_run_worker", controlled)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    first = manager.start("prepare")
    await started.wait()
    with pytest.raises(resource_manager.VoiceResourceError, match="resource_operation_busy"):
        manager.start("prepare")
    await manager.cancel(first["operation_id"])
    assert manager.operation(first["operation_id"])["state"] == "cancelled"
    finish.set()
    second = manager.start("prepare")
    await manager._current.task
    assert manager.operation(second["operation_id"])["state"] == "succeeded"
    assert manager.operation(first["operation_id"])["state"] == "cancelled"
    await manager.close()


@pytest.mark.asyncio
async def test_native_result_after_cancel_does_not_publish_ready(monkeypatch, tmp_path):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finish = asyncio.Event()
    async def late_result(*args, **kwargs):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()
        return {"campp": {"state": "ready", "required": True, "reason": None}}
    monkeypatch.setattr(resource_manager, "_run_worker", late_result)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    operation = manager.start("prepare")
    await started.wait()
    cancelling = asyncio.create_task(manager.cancel(operation["operation_id"]))
    await cancelled.wait()
    with pytest.raises(resource_manager.VoiceResourceError, match="resource_operation_busy"):
        manager.start("prepare")
    finish.set()
    await cancelling
    assert manager.operation(operation["operation_id"])["state"] == "cancelled"
    assert manager._prepared == {}
    await manager.close()


@pytest.mark.asyncio
async def test_contract_change_during_trial_rejects_previously_accepted_result(monkeypatch, tmp_path):
    state = [False]
    async def changed(*args, **kwargs):
        state[0] = True
        await asyncio.sleep(0)
        return {"accepted": True, "reason": None}
    monkeypatch.setattr(resource_manager, "_run_worker", changed)
    manager = resource_manager.VoiceResourceManager(lambda: state[0], cache_root=tmp_path)
    with pytest.raises(resource_manager.VoiceResourceError, match="audio_contract_changed"):
        await manager.check_audio(b"\0" * 288000, noise_reduction_enabled=False)
    assert manager._trial_task is None
    await manager.close()


@pytest.mark.asyncio
async def test_actual_preference_worker_is_atomic_and_does_not_change_owner_filter(monkeypatch, tmp_path):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEKO_WAKE_WORD_ENABLED", raising=False)
    owner_filter = tmp_path / "filter.json"
    owner_filter.write_text('{"enabled": false}')
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    result = await manager.save_preference(True)
    assert result["enabled"] is True
    assert voice_wake_word.wake_word_preference(tmp_path)["enabled"] is True
    assert owner_filter.read_text() == '{"enabled": false}'
    assert not tuple(tmp_path.glob("preference-*.tmp"))
    await manager.close()


def test_repair_corrupt_bundle_publishes_new_directory_and_keeps_owned_old_files(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    source = _archive(tmp_path)
    original_pointer = (root / "current.json").read_bytes()
    (previous / model_bundle.ASSETS[0]).write_bytes(b"damaged")
    with pytest.raises(model_bundle.WakeWordBundleError, match="wake_model_invalid"):
        model_bundle.resolve_cached_model_dir(root)
    verified = []
    repaired = model_bundle.install_bundle(root, source, validate=lambda path: verified.append(path))
    assert verified
    assert repaired != previous
    assert (root / "current.json").read_bytes() != original_pointer
    assert model_bundle.resolve_cached_model_dir(root) == repaired
    assert all((repaired / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    assert (previous / model_bundle.ASSETS[0]).read_bytes() == b"damaged"
    # Re-running repair does not accumulate another directory.
    repeated = model_bundle.install_bundle(root, source, validate=lambda path: None)
    assert repeated == repaired
    assert len(tuple((root / "versions").iterdir())) == 2


@pytest.mark.asyncio
async def test_optional_corrupt_cache_does_not_block_primary_prepare(tmp_path, monkeypatch):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEKO_WAKE_WORD_ENABLED", raising=False)
    root, previous = _install(tmp_path, monkeypatch)
    (previous / model_bundle.ASSETS[0]).write_bytes(b"damaged")
    observed = []
    async def primary_prepare(kind, nr, wake_path):
        observed.append(wake_path)
        return {name: {"state": "ready", "reason": None, "required": True}
                for name in ("campp", "silero", "noise_reduction")}
    monkeypatch.setattr(resource_manager, "_run_worker", primary_prepare)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root)
    operation = manager.start("prepare")
    await manager._current.task
    result = manager.operation(operation["operation_id"])
    assert result["state"] == "succeeded"
    assert observed == [None]
    assert all(result["result"][name]["state"] == "ready" for name in ("campp", "silero", "noise_reduction"))
    assert result["result"]["wake_model"]["reason"] == "WAKE_WORD_MODEL_INVALID"
    assert result["result"]["wake_model"]["required"] is False
    await manager.close()


def test_unmarked_stage_directory_is_never_removed(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    root = tmp_path / "cache"
    unrelated = root / "stage-neko-kws-user-owned"
    unrelated.mkdir(parents=True)
    (unrelated / "important.txt").write_bytes(b"keep")
    with pytest.raises(model_bundle.WakeWordBundleError, match="resource_cache_unsafe"):
        model_bundle.install_bundle(root, source)
    assert (unrelated / "important.txt").read_bytes() == b"keep"


def test_interrupted_marked_stage_is_cleaned_before_retry(tmp_path, monkeypatch):
    source = _archive(tmp_path)
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    root = tmp_path / "cache"
    orphan = root / "stage-neko-kws-abandoned"
    orphan.mkdir(parents=True)
    (orphan / ".owner").write_bytes(model_bundle._STAGE_OWNER)
    (orphan / "archive.tar.bz2").write_bytes(b"incomplete")
    model_bundle.install_bundle(root, source)
    assert not orphan.exists()
    assert model_bundle.resolve_cached_model_dir(root) is not None


def test_preference_pending_crash_artifact_is_reused_under_os_lock(tmp_path, monkeypatch):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEKO_WAKE_WORD_ENABLED", raising=False)
    voice_wake_word.save_wake_word_preference(False, tmp_path)
    (tmp_path / "preference.pending").write_bytes(b"partial")
    voice_wake_word.save_wake_word_preference(True, tmp_path)
    assert voice_wake_word.wake_word_preference(tmp_path)["enabled"] is True
    assert not (tmp_path / "preference.pending").exists()


@pytest.mark.asyncio
async def test_changed_trial_contract_is_rejected_before_model_load_or_suppression(tmp_path):
    from tests.support.voice_identity_fakes import _service
    from main_logic.voice_identity_service.audio_contract import desktop_audio_contract_snapshot
    from main_logic.voice_identity_service.service import VoiceIdentityServiceError
    from unittest.mock import Mock
    service, model, activations, suppression = _service(tmp_path, enrollment_noise_reduction_enabled=False)
    model.load = Mock(wraps=model.load)
    await service.initialize()
    with pytest.raises(VoiceIdentityServiceError) as caught:
        await service.start_enrollment(expected_audio_contract=desktop_audio_contract_snapshot(noise_reduction_enabled=True))
    assert caught.value.code == "audio_contract_changed"
    assert service.status().enrollment is None
    assert suppression == []
    assert activations == []
    assert model.closed is False
    model.load.assert_not_called()
    await service.close()


@pytest.mark.asyncio
async def test_real_trial_pipeline_does_not_mistake_loud_tone_for_voice(tmp_path):
    import numpy as np
    from main_logic.asr_client.endpointing.asset_manifest import AssetManifestError, resolve_verified_assets
    try:
        resolve_verified_assets(("silero_vad.onnx",))
    except AssetManifestError:
        pytest.skip("Pinned Silero asset is prepared by the desktop release acceptance job")
    samples = np.arange(48_000 * 3, dtype=np.float64)
    tone = (np.sin(samples * (2 * np.pi * 220 / 48_000)) * 3276).astype("<i2").tobytes()
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    result = await manager.check_audio(tone, noise_reduction_enabled=False)
    assert result["accepted"] is False
    assert result["reason"] == "no_speech_detected"
    assert result["diagnostics"]["rms"] > .008
    assert result["diagnostics"]["active_seconds"] >= 1.5
    assert result["audio_contract"]["noise_reduction_enabled"] is False
    assert not tuple(tmp_path.iterdir())
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_before_operation_task_starts_has_terminal_result(tmp_path):
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    operation = manager.start("prepare")
    cancelled = await manager.cancel(operation["operation_id"])
    assert cancelled["state"] == "cancelled"
    assert cancelled["reason"] == "operation_cancelled"
    assert manager._current.task.done()
    successor = manager.start("prepare")
    await manager.cancel(successor["operation_id"])
    await manager.close()


def _exiting_worker(connection, kind, nr_enabled, wake_path, pcm16):
    import os
    os._exit(4)


@pytest.mark.asyncio
async def test_native_crash_reports_stable_worker_failure_and_retires_process(monkeypatch):
    monkeypatch.setattr(resource_manager, "_resource_worker", _exiting_worker)
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(resource_manager.VoiceResourceError, match="resource_worker_failed"):
        await resource_manager._run_worker("prepare", False)
    assert {child.pid for child in multiprocessing.active_children()} <= before


def _holding_install_lock(connection, root):
    with model_bundle._installation_lock(Path(root)):
        connection.send("locked")
        while True:
            time.sleep(.1)


def test_cross_process_install_lock_is_released_after_worker_termination(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_holding_install_lock, args=(child, str(tmp_path)))
    try:
        process.start()
        child.close()
        assert parent.poll(5)
        assert parent.recv() == "locked"
        with pytest.raises(model_bundle.WakeWordBundleError, match="resource_operation_busy"):
            with model_bundle._installation_lock(tmp_path):
                pytest.fail("A second process acquired the same exclusive lock")
        process.terminate()
        process.join(3)
        assert not process.is_alive()
        with model_bundle._installation_lock(tmp_path):
            pass
    finally:
        parent.close()
        child.close()
        if process.is_alive():
            process.kill()
            process.join(3)
        process.close()


class _WorkerReply:
    """Capture the public worker envelope without retaining submitted audio."""
    def __init__(self):
        self.message = None
        self.closed = False

    def send(self, message):
        self.message = message

    def close(self):
        self.closed = True


def test_resource_worker_prepares_real_primary_models_and_closes_reply(monkeypatch):
    from types import SimpleNamespace
    from utils import audio_processor
    try:
        resource_manager.resolve_verified_campplus_asset()
        resource_manager.resolve_verified_assets(("silero_vad.onnx",))
    except (resource_manager.CampPlusAssetError, resource_manager.AssetManifestError):
        pytest.skip("Pinned primary assets are prepared by desktop acceptance")
    monkeypatch.setitem(resource_manager.sys.modules, "sherpa_onnx", SimpleNamespace(__version__="incompatible", version="incompatible"))
    monkeypatch.setattr(audio_processor, "DEBUG_SAVE_AUDIO", True)
    reply = _WorkerReply()
    resource_manager._resource_worker(reply, "prepare", False, None, b"")
    assert reply.closed and reply.message["ok"]
    result = reply.message["result"]
    assert all(result[name]["state"] == "ready" for name in ("campp", "silero", "noise_reduction"))
    assert result["wake_runtime"]["reason"] == "WAKE_WORD_RUNTIME_FIX_REQUIRED"
    assert audio_processor.DEBUG_SAVE_AUDIO is False


def test_resource_worker_checks_real_pcm_without_publishing_raw_audio(monkeypatch):
    import numpy as np
    from utils import audio_processor
    try:
        resource_manager.resolve_verified_assets(("silero_vad.onnx",))
    except resource_manager.AssetManifestError:
        pytest.skip("Pinned Silero asset is prepared by desktop acceptance")
    monkeypatch.setattr(audio_processor, "DEBUG_SAVE_AUDIO", True)
    samples = np.arange(48_000 * 3)
    pcm = (np.sin(samples * (2 * np.pi * 220 / 48_000)) * 3276).astype("<i2").tobytes()
    reply = _WorkerReply()
    resource_manager._resource_worker(reply, "audio", False, None, pcm)
    assert reply.closed and reply.message["ok"]
    assert reply.message["result"]["accepted"] is False
    assert reply.message["result"]["reason"] == "no_speech_detected"
    assert "pcm" not in json.dumps(reply.message)
    assert audio_processor.DEBUG_SAVE_AUDIO is False


def test_incompatible_runtime_is_rejected_before_download(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setitem(resource_manager.sys.modules, "sherpa_onnx", SimpleNamespace(__version__="wrong", version="wrong"))
    def unexpected(*args, **kwargs):
        pytest.fail("Download must not run when native runtime is incompatible")
    monkeypatch.setattr(resource_manager, "install_bundle", unexpected)
    reply = _WorkerReply()
    resource_manager._resource_worker(reply, "download", False, str(tmp_path), b"")
    assert reply.closed
    assert reply.message == {"ok": False, "reason": "WAKE_WORD_RUNTIME_FIX_REQUIRED"}
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_partial_prepare_failure_preserves_each_resource_and_closes_handles(monkeypatch):
    from types import SimpleNamespace
    from main_logic.asr_client.speaker_shadow import campplus
    from utils import audio_processor
    closed = []
    class Model:
        def load(self):
            return False
        def close(self):
            closed.append("campp")
    class Validator:
        async def load(self):
            return False
        async def close(self):
            closed.append("silero")
    class Processor:
        rnnoise_available = False
        def __init__(self, **kwargs):
            pass
        def process_chunk(self, pcm):
            assert len(pcm) == 960
        def close(self):
            closed.append("nr")
    async def invalid_wake(path):
        raise resource_manager.VoiceResourceError("WAKE_WORD_MODEL_INVALID")
    monkeypatch.setattr(campplus, "CampPlusEmbeddingModel", Model)
    monkeypatch.setattr(resource_manager, "SileroEnrollmentSpeechValidator", Validator)
    monkeypatch.setattr(audio_processor, "AudioProcessor", Processor)
    monkeypatch.setattr(resource_manager, "_prepare_wake", invalid_wake)
    monkeypatch.setitem(resource_manager.sys.modules, "sherpa_onnx", SimpleNamespace(__version__=resource_manager.SUPPORTED_RUNTIME_VERSION, version=resource_manager.SUPPORTED_RUNTIME_VERSION))
    result = await resource_manager._prepare_resources(True, "model")
    assert closed == ["campp", "silero", "nr"]
    assert all(value["state"] == "unavailable" for value in result.values())
    assert result["campp"]["reason"] == result["silero"]["reason"] == "model_unavailable"
    assert result["noise_reduction"]["reason"] == "audio_processing_unavailable"
    assert result["wake_model"]["reason"] == result["wake_runtime"]["reason"] == "WAKE_WORD_MODEL_INVALID"


@pytest.mark.asyncio
async def test_snapshot_tracks_missing_assets_and_corrupt_preference_without_loading(monkeypatch, tmp_path):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    monkeypatch.delenv("NEKO_WAKE_WORD_ENABLED", raising=False)
    def absent_campp():
        raise resource_manager.CampPlusAssetError("missing")
    def absent_silero(*args):
        raise resource_manager.AssetManifestError("missing")
    monkeypatch.setattr(resource_manager, "resolve_verified_campplus_asset", absent_campp)
    monkeypatch.setattr(resource_manager, "resolve_verified_assets", absent_silero)
    (tmp_path / "preference.json").write_bytes(b"broken")
    manager = resource_manager.VoiceResourceManager(lambda: True, cache_root=tmp_path)
    snapshot = await manager.resources()
    assert snapshot["can_enroll"] is False
    assert snapshot["resources"]["campp"]["state"] == snapshot["resources"]["silero"]["state"] == "missing"
    assert snapshot["wake_enabled"] is True
    assert snapshot["resources"]["wake_model"]["reason"] == "WAKE_WORD_PREFERENCE_UNAVAILABLE"
    assert (tmp_path / "preference.json").read_bytes() == b"broken"
    assert manager._current is None
    await manager.close()


@pytest.mark.asyncio
async def test_prepared_readiness_is_invalidated_when_dsp_changes(monkeypatch, tmp_path):
    monkeypatch.delenv("NEKO_WAKE_WORD_MODEL_DIR", raising=False)
    state = [False]
    async def prepared(*args):
        return {name: {"state": "ready", "reason": None, "required": True}
                for name in ("campp", "silero", "noise_reduction")}
    monkeypatch.setattr(resource_manager, "_run_worker", prepared)
    monkeypatch.setattr(resource_manager, "resolve_verified_campplus_asset", lambda: None)
    monkeypatch.setattr(resource_manager, "resolve_verified_assets", lambda *args: None)
    manager = resource_manager.VoiceResourceManager(lambda: state[0], cache_root=tmp_path)
    manager.start("prepare")
    await manager._current.task
    assert (await manager.resources())["can_enroll"] is True
    state[0] = True
    snapshot = await manager.resources()
    assert snapshot["can_enroll"] is False
    assert snapshot["resources"]["noise_reduction"]["state"] == "unchecked"
    assert snapshot["audio_contract"]["noise_reduction_enabled"] is True
    await manager.close()


@pytest.mark.asyncio
async def test_prepare_callback_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    async def prepared(*args):
        return {}
    async def refresh(operation_id):
        raise resource_manager.VoiceResourceError("runtime_degraded")
    monkeypatch.setattr(resource_manager, "_run_worker", prepared)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path, on_ready=refresh)
    operation = manager.start("prepare")
    await manager._current.task
    assert manager.operation(operation["operation_id"])["state"] == "failed"
    assert manager.operation(operation["operation_id"])["reason"] == "runtime_degraded"
    await manager.close()


@pytest.mark.asyncio
async def test_close_physically_cancels_in_progress_trial_and_rejects_late_use(monkeypatch, tmp_path):
    started = asyncio.Event()
    retired = asyncio.Event()
    async def waiting(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            retired.set()
    monkeypatch.setattr(resource_manager, "_run_worker", waiting)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path)
    task = asyncio.create_task(manager.check_audio(b"\0" * 288000, noise_reduction_enabled=False))
    await started.wait()
    await manager.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert retired.is_set() and manager._trial_task is None
    with pytest.raises(resource_manager.VoiceResourceError, match="runtime_degraded"):
        await manager.resources()
    with pytest.raises(resource_manager.VoiceResourceError, match="runtime_degraded"):
        manager.start("prepare")


def _staged_download_worker(connection, kind, nr_enabled, wake_path, pcm16):
    """Real immutable delivery/publication with small authenticated fixtures."""
    root = Path(wake_path)
    source = root.parent / "download.tar.bz2"
    model_bundle.MODEL_SHA256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if kind == "download":
        installed = model_bundle.install_bundle(root, source, validate=lambda path: None, publish=False)
        (root / "staged").write_text(installed.name)
        if (root.parent / "hold-download").exists():
            while True:
                time.sleep(.025)
        connection.send({"ok": True, "result": {"version": installed.name}})
        connection.close()
    elif (root.parent / "drop-publication-reply").exists():
        model_bundle.publish_bundle_version(root, pcm16.decode("ascii"))
        import os
        os._exit(7)
    else:
        resource_manager._resource_worker(connection, kind, nr_enabled, wake_path, pcm16)


def _prepare_download_fixture(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    pointer = (root / "current.json").read_bytes()
    source = _archive(tmp_path, b"second")
    source.rename(tmp_path / "download.tar.bz2")
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256((tmp_path / "download.tar.bz2").read_bytes()).hexdigest())
    monkeypatch.setattr(resource_manager, "_resource_worker", _staged_download_worker)
    monkeypatch.setattr(resource_manager.platform, "system", lambda: "Windows")
    monkeypatch.setattr(resource_manager.platform, "machine", lambda: "amd64")
    return root, previous, pointer


@pytest.mark.asyncio
async def test_actual_download_cancel_before_commit_keeps_previous_pointer(tmp_path, monkeypatch):
    root, previous, pointer = _prepare_download_fixture(tmp_path, monkeypatch)
    (tmp_path / "hold-download").touch()
    refreshed = []
    async def refresh(operation_id):
        refreshed.append(operation_id)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("download")
    async def staged():
        while not (root / "staged").exists():
            await asyncio.sleep(.025)
    await asyncio.wait_for(staged(), timeout=8)
    result = await asyncio.wait_for(manager.cancel(operation["operation_id"]), timeout=3)
    assert result["state"] == "cancelled" and result["committed"] is False
    assert (root / "current.json").read_bytes() == pointer
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    assert not refreshed
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh_fails", [False, True])
async def test_actual_download_cancel_after_publication_reports_commit_and_refresh(tmp_path, monkeypatch, refresh_fails):
    root, previous, pointer = _prepare_download_fixture(tmp_path, monkeypatch)
    refresh_entered = asyncio.Event()
    refresh_finish = asyncio.Event()
    calls = []
    async def refresh(operation_id):
        calls.append(operation_id)
        refresh_entered.set()
        await refresh_finish.wait()
        if refresh_fails:
            raise resource_manager.VoiceResourceError("runtime_degraded")
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("download")
    await asyncio.wait_for(refresh_entered.wait(), timeout=8)
    assert (root / "current.json").read_bytes() != pointer
    assert manager.operation(operation["operation_id"])["committed"] is True
    cancellation = asyncio.create_task(manager.cancel(operation["operation_id"]))
    await asyncio.sleep(0)
    assert not cancellation.done()
    refresh_finish.set()
    result = await asyncio.wait_for(cancellation, timeout=3)
    assert result["state"] == ("failed" if refresh_fails else "succeeded")
    assert result["committed"] is True and result["result"] == {"installed": True}
    assert result["reason"] == ("runtime_degraded" if refresh_fails else None)
    assert calls == [operation["operation_id"]]
    current = model_bundle.resolve_cached_model_dir(root)
    assert current != previous and all((current / name).read_bytes() == b"second" for name in model_bundle.ASSETS)
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    await manager.close()


@pytest.mark.asyncio
async def test_actual_publication_without_ipc_reply_recovers_committed_pointer(tmp_path, monkeypatch):
    root, previous, pointer = _prepare_download_fixture(tmp_path, monkeypatch)
    (tmp_path / "drop-publication-reply").touch()
    refreshed = []
    async def refresh(operation_id):
        refreshed.append(operation_id)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("download")
    await asyncio.wait_for(manager._current.task, timeout=8)
    result = manager.operation(operation["operation_id"])
    assert result["state"] == "succeeded" and result["committed"] is True
    assert result["result"] == {"installed": True}
    assert refreshed == [operation["operation_id"]]
    assert (root / "current.json").read_bytes() != pointer
    assert model_bundle.resolve_cached_model_dir(root) != previous
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_prepare_refresh_never_publishes_new_ready_snapshot(monkeypatch, tmp_path):
    ready = {name: {"state": "ready", "reason": None, "required": True}
             for name in ("campp", "silero", "noise_reduction")}
    async def prepared(*args):
        return ready
    refresh_entered = asyncio.Event()
    async def refresh(operation_id):
        refresh_entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(resource_manager, "_run_worker", prepared)
    monkeypatch.setattr(resource_manager, "resolve_verified_campplus_asset", lambda: None)
    monkeypatch.setattr(resource_manager, "resolve_verified_assets", lambda *args: None)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=tmp_path, on_ready=refresh)
    operation = manager.start("prepare")
    await refresh_entered.wait()
    assert (await manager.resources())["can_enroll"] is False
    result = await manager.cancel(operation["operation_id"])
    assert result["state"] == "cancelled" and result["committed"] is False
    assert manager._prepared == {} and manager._prepared_noise_reduction is None
    assert (await manager.resources())["can_enroll"] is False
    await manager.close()


@pytest.mark.asyncio
async def test_actual_service_close_retires_postpublication_refresh_waiting_on_its_lock(tmp_path, monkeypatch):
    from tests.support.voice_identity_fakes import _service
    service, *_ = _service(tmp_path)
    await service.initialize()
    await service._resource_manager.close()
    source = _archive(tmp_path)
    monkeypatch.setattr(model_bundle, "MODEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    root = tmp_path / "cache"
    staged = model_bundle.install_bundle(root, source, publish=False)
    publish_started = asyncio.Event()
    close_entered = asyncio.Event()
    refresh_entered = asyncio.Event()
    async def work(kind, nr, path, **kwargs):
        if kind == "download":
            return {"version": staged.name}
        publish_started.set()
        await close_entered.wait()
        await asyncio.to_thread(model_bundle.publish_bundle_version, root, staged.name)
        return {"installed": True}
    async def refresh(operation_id):
        refresh_entered.set()
        await service._refresh_after_resource_operation(operation_id)
    manager = resource_manager.VoiceResourceManager(lambda: True, cache_root=root, on_ready=refresh)
    service._resource_manager = manager
    original_close = manager.close
    async def close_after_waiter_enters():
        assert service._operation_lock.locked()
        close_entered.set()
        await refresh_entered.wait()
        # refresh's actual service-lock acquisition has yielded by here.
        await original_close()
    monkeypatch.setattr(manager, "close", close_after_waiter_enters)
    monkeypatch.setattr(resource_manager, "_run_worker", work)
    monkeypatch.setattr(resource_manager.platform, "system", lambda: "Windows")
    monkeypatch.setattr(resource_manager.platform, "machine", lambda: "amd64")
    operation = manager.start("download")
    await publish_started.wait()
    await asyncio.wait_for(service.close(), timeout=3)
    result = manager.operation(operation["operation_id"])
    assert result["committed"] is True and result["state"] == "failed"
    assert result["reason"] == "runtime_degraded"
    assert result["result"] == {"installed": True}
    assert manager._current.refresh_task is None
    assert model_bundle.resolve_cached_model_dir(root) == staged


@pytest.mark.asyncio
async def test_missing_home_is_stable_closed_preference_without_blocking_primary_snapshot(monkeypatch):
    from main_logic.voice_identity_service.wake_resources import resolve_wake_word_resources
    for variable in ("LOCALAPPDATA", "XDG_CACHE_HOME", "NEKO_WAKE_WORD_MODEL_DIR", "NEKO_WAKE_WORD_ENABLED"):
        monkeypatch.delenv(variable, raising=False)
    def no_home():
        raise RuntimeError("Can't determine home directory")
    monkeypatch.setattr(Path, "home", no_home)
    assert voice_wake_word.wake_word_preference()["reason"] == "wake_preference_unavailable"
    assert resolve_wake_word_resources().reason == "WAKE_WORD_PREFERENCE_UNAVAILABLE"
    with pytest.raises(ValueError, match="wake_preference_unavailable"):
        voice_wake_word.save_wake_word_preference(True)
    manager = resource_manager.VoiceResourceManager(lambda: False)
    snapshot = await manager.resources()
    assert snapshot["wake_preference_reason"] == "WAKE_WORD_PREFERENCE_UNAVAILABLE"
    with pytest.raises(resource_manager.VoiceResourceError, match="resource_storage_unavailable"):
        await manager.save_preference(True)
    await manager.close()


def _locked_pointer_publish_worker(connection, kind, nr_enabled, wake_path, pcm16):
    if kind == "download":
        return _staged_download_worker(connection, kind, nr_enabled, wake_path, pcm16)
    root = Path(wake_path)
    model_bundle.MODEL_SHA256 = hashlib.sha256((root.parent / "download.tar.bz2").read_bytes()).hexdigest()
    original_replace = model_bundle.os.replace
    def locked_replace(source, target):
        if Path(target).name == "current.json":
            raise PermissionError("Windows current pointer sharing lock")
        return original_replace(source, target)
    model_bundle.os.replace = locked_replace
    resource_manager._resource_worker(connection, kind, nr_enabled, wake_path, pcm16)


@pytest.mark.asyncio
async def test_actual_parent_publication_failure_keeps_previous_version_and_reports_failure(tmp_path, monkeypatch):
    root, previous, pointer = _prepare_download_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(resource_manager, "_resource_worker", _locked_pointer_publish_worker)
    refreshed = []
    async def refresh(operation_id):
        refreshed.append(operation_id)
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("download")
    await asyncio.wait_for(manager._current.task, timeout=8)
    result = await manager.cancel(operation["operation_id"])
    assert result["state"] == "failed" and result["committed"] is False
    assert result["reason"] == "resource_storage_unavailable"
    assert (root / "current.json").read_bytes() == pointer
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    assert not (root / "current.pending").exists()
    assert not refreshed
    await manager.close()


@pytest.mark.asyncio
async def test_committed_download_refresh_timeout_keeps_installation_and_reports_runtime_failure(tmp_path, monkeypatch):
    from main_logic.voice_identity_service.state import VoiceIdentityEffectiveReason
    root, previous, pointer = _prepare_download_fixture(tmp_path, monkeypatch)
    refresh_entered = asyncio.Event()
    refresh_retired = asyncio.Event()
    refresh_calls = []
    async def refresh(operation_id):
        refresh_calls.append(operation_id)
        refresh_entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            refresh_retired.set()
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("download")
    await asyncio.wait_for(refresh_entered.wait(), timeout=8)
    assert (root / "current.json").read_bytes() != pointer
    # Cancellation during refresh resolves the real bounded refresh deadline;
    # the installed package is neither rolled back nor reported as cancelled.
    result = await asyncio.wait_for(manager.cancel(operation["operation_id"]), timeout=7)
    assert result["state"] == "failed" and result["committed"] is True
    assert result["reason"] == VoiceIdentityEffectiveReason.RUNTIME_DEGRADED.value
    assert result["reason"] != "resource_prepare_timeout"
    assert result["result"] == {"installed": True}
    assert refresh_retired.is_set() and manager._current.refresh_task is None
    assert refresh_calls == [operation["operation_id"]]
    assert not manager._prepared
    current = model_bundle.resolve_cached_model_dir(root)
    assert current != previous and all((current / name).read_bytes() == b"second" for name in model_bundle.ASSETS)
    assert all((previous / name).read_bytes() == b"first" for name in model_bundle.ASSETS)
    await manager.close()


@pytest.mark.asyncio
async def test_precommit_resource_worker_timeout_retains_preparation_reason_and_old_pointer(tmp_path, monkeypatch):
    root, previous = _install(tmp_path, monkeypatch)
    pointer = (root / "current.json").read_bytes()
    monkeypatch.setattr(resource_manager, "_resource_worker", _hung_worker)
    real_worker = resource_manager._run_worker
    async def short_worker(kind, nr_enabled, wake_path):
        return await real_worker(kind, nr_enabled, str(tmp_path / "started"), timeout=.5)
    monkeypatch.setattr(resource_manager, "_run_worker", short_worker)
    refreshed = []
    async def refresh(operation_id):
        refreshed.append(operation_id)
    before = {child.pid for child in multiprocessing.active_children()}
    manager = resource_manager.VoiceResourceManager(lambda: False, cache_root=root, on_ready=refresh)
    operation = manager.start("prepare")
    await asyncio.wait_for(manager._current.task, timeout=3)
    result = manager.operation(operation["operation_id"])
    assert result["state"] == "failed" and result["committed"] is False
    assert result["reason"] == "resource_prepare_timeout"
    assert result["result"] is None and not refreshed and not manager._prepared
    assert (root / "current.json").read_bytes() == pointer and previous.is_dir()
    assert {child.pid for child in multiprocessing.active_children()} <= before
    await manager.close()
