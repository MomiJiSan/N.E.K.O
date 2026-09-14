from __future__ import annotations

import asyncio

from pathlib import Path

import threading

from types import SimpleNamespace

import numpy as np

import pytest

import main_logic.voice_identity_service.profile_store as store_module

from main_logic.asr_client import VoiceIdentityActivationResult

from main_logic.asr_client.speaker_shadow.campplus import CAMPPLUS_EMBEDDING_DIM

from main_logic.voice_identity.contracts import SpeakerModelIdentity

from main_logic.voice_identity.profile import SpeakerProfile

from main_logic.voice_identity.reference import SpeakerReference

from main_logic.voice_identity_service.preference_store import (
    VoiceIdentityPreferenceStore,
    VoiceIdentityPreferenceStoreError,
)

from main_logic.voice_identity_service.enrollment import EnrollmentSpeechResult

from main_logic.voice_identity_service.audio_contract import (
    OWNER_CAMPPLUS_DESKTOP_CONTRACT_ID,
    desktop_audio_contract_snapshot,
)

from main_logic.voice_identity_service.enrollment_audio import (
    EnrollmentAudioNormalizationError,
)

from main_logic.voice_identity_service.profile_store import (
    SecureStorageUnavailableError,
    VoiceIdentityProfileCorruptError,
    VoiceIdentityProfileIncompatibleError,
    VoiceIdentityProfileStore,
    VoiceIdentityProfileStoreError,
)

from main_logic.voice_identity_service.service import (
    VoiceIdentityService,
    VoiceIdentityServiceError,
)

from main_logic.voice_input.suppression import VoiceInputSuppressionController

from .test_profile_store import _TestKeyProtector

class _Model:
    model_id = "3d-speaker-campplus-zh-en"
    model_revision = "2025-06-16-sherpa-onnx-campplus"

    def __init__(
        self,
        *,
        loads: bool = True,
        embeddings: list[np.ndarray] | None = None,
    ) -> None:
        self.loads = loads
        self.closed = False
        self.embeddings = list(embeddings or [])
        self.inference_count = 0

    def load(self) -> bool:
        return self.loads

    def cancel_load(self) -> None:
        return

    def embedding_from_pcm16(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
    ) -> np.ndarray:
        assert pcm16
        assert sample_rate_hz == 16_000
        self.inference_count += 1
        if self.embeddings:
            return self.embeddings.pop(0)
        result = np.zeros(CAMPPLUS_EMBEDDING_DIM, dtype=np.float32)
        result[0] = 1.0
        return result

    def cancel_inference(self) -> None:
        return

    def close(self) -> None:
        self.closed = True

class _SpeechValidator:
    def __init__(self, *, loads: bool = True) -> None:
        self.loads = loads
        self.closed = False

    async def load(self) -> bool:
        return self.loads

    async def validate_pcm16(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int = 16_000,
    ) -> EnrollmentSpeechResult:
        assert pcm16
        assert sample_rate_hz == 16_000
        return EnrollmentSpeechResult(window_count=96, active_window_count=96)

    async def close(self) -> None:
        self.closed = True

class _AudioNormalizer:
    def __init__(self, nr_enabled: bool, *, failure_code: str | None = None) -> None:
        self.nr_enabled = nr_enabled
        self.failure_code = failure_code
        self.calls: list[tuple[int, int, int]] = []

    async def normalize(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
        target_samples: int,
    ) -> bytes:
        self.calls.append((len(pcm16), sample_rate_hz, target_samples))
        if self.failure_code is not None:
            raise EnrollmentAudioNormalizationError(self.failure_code)
        assert sample_rate_hz == 48_000
        assert target_samples in (48_000, 80_000)
        required_bytes = target_samples * 2
        if len(pcm16) < required_bytes:
            raise EnrollmentAudioNormalizationError("speech_too_short")
        return pcm16[:required_bytes]

def _pcm() -> bytes:
    samples = np.full(48_000, 4_000, dtype="<i2")
    return samples.tobytes()

def _verification_pcm(milliseconds: int = 5_000) -> bytes:
    samples = np.full(48_000 * milliseconds // 1_000, 4_000, dtype="<i2")
    return samples.tobytes()

async def _wait_until(predicate, *, timeout_seconds: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("condition was not satisfied before timeout")
        await asyncio.sleep(0.005)

def _service(
    tmp_path: Path,
    *,
    model: _Model | None = None,
    activation_results: list[bool | VoiceIdentityActivationResult] | None = None,
    runtime_status_results: list[VoiceIdentityActivationResult] | None = None,
    enrollment_ttl_seconds: float = 30.0,
    model_timeout_seconds: float = 1.0,
    runtime_mode: str = "enforce",
    speech_validator: _SpeechValidator | None = None,
    audio_normalizer_factory=None,
    enrollment_noise_reduction_enabled: bool = True,
) -> tuple[
    VoiceIdentityService,
    _Model,
    list[tuple[SpeakerProfile | None, str]],
    list[str],
]:
    selected_model = model or _Model()
    selected_validator = speech_validator or _SpeechValidator()
    activations: list[tuple[SpeakerProfile | None, str]] = []
    results = activation_results or []
    runtime_results = runtime_status_results or []
    suppression_events: list[str] = []

    async def activate(
        profile: SpeakerProfile | None,
        generation: str,
        **_authority,
    ) -> bool:
        activations.append((profile, generation))
        return results.pop(0) if results else True

    async def suppress(reason: str) -> None:
        suppression_events.append(f"suppress:{reason}")

    async def restore(reason: str) -> None:
        suppression_events.append(f"restore:{reason}")

    def runtime_status() -> VoiceIdentityActivationResult:
        return (
            runtime_results[-1]
            if runtime_results
            else VoiceIdentityActivationResult.READY
        )

    service = VoiceIdentityService(
        VoiceIdentityProfileStore(
            tmp_path / "voice_identity.profile",
            key_protector=_TestKeyProtector(),
        ),
        VoiceIdentityPreferenceStore(tmp_path / "voice_identity.preference"),
        VoiceInputSuppressionController(
            suppress,
            restore,
            default_ttl_seconds=enrollment_ttl_seconds,
            hard_ttl_seconds=max(1.0, enrollment_ttl_seconds),
        ),
        lambda: selected_model,
        activate,
        runtime_mode=runtime_mode,  # type: ignore[arg-type]
        enrollment_ttl_seconds=enrollment_ttl_seconds,
        model_timeout_seconds=model_timeout_seconds,
        activation_timeout_seconds=1.0,
        runtime_status_callback=(runtime_status if runtime_status_results else None),
        speech_validator_factory=lambda: selected_validator,
        enrollment_audio_normalizer_factory=(
            audio_normalizer_factory or _AudioNormalizer
        ),
        enrollment_noise_reduction_enabled=enrollment_noise_reduction_enabled,
    )

    production_submit_enrollment_segment = service.submit_enrollment_segment

    async def submit_enrollment_segment(
        enrollment_id: str,
        profile_id: str,
        segment_index: int,
        pcm16: bytes,
        *,
        sample_rate_hz: int = 48_000,
        audio_contract_id: str = OWNER_CAMPPLUS_DESKTOP_CONTRACT_ID,
    ):
        return await production_submit_enrollment_segment(
            enrollment_id,
            profile_id,
            segment_index,
            pcm16,
            sample_rate_hz=sample_rate_hz,
            audio_contract_id=audio_contract_id,
        )

    service.submit_enrollment_segment = submit_enrollment_segment  # type: ignore[method-assign]

    async def complete_enrollment(
        enrollment_id: str,
        profile_id: str,
        pcm16: bytes,
    ):
        status = service.status()
        for segment_index in range(1, 5):
            status = await service.submit_enrollment_segment(
                enrollment_id,
                profile_id,
                segment_index,
                _verification_pcm() if segment_index == 4 else pcm16,
            )
        return status

    # Keep the legacy tests focused on transaction semantics while the production
    # service exposes only the four-segment API.
    service.complete_enrollment = complete_enrollment  # type: ignore[attr-defined]
    return service, selected_model, activations, suppression_events

def _embedding(axis: int = 0) -> np.ndarray:
    result = np.zeros(CAMPPLUS_EMBEDDING_DIM, dtype=np.float32)
    result[axis] = 1.0
    return result

async def test_segment_progress_is_server_owned_idempotent_and_profile_bound(
    tmp_path: Path,
) -> None:
    service, model, _activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    assert enrollment.profile_id is None
    assert enrollment.next_segment_index == 1
    assert enrollment.required_segments == 4

    with pytest.raises(VoiceIdentityServiceError, match="segment_out_of_order"):
        await service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-a",
            2,
            _pcm(),
        )

    first = await service.submit_enrollment_segment(
        enrollment.enrollment_id,
        "profile-a",
        1,
        _pcm(),
    )
    assert first.enrollment is not None
    assert first.enrollment.profile_id == "profile-a"
    assert first.enrollment.accepted_segments == 1
    assert first.enrollment.next_segment_index == 2
    assert model.inference_count == 1

    retry = await service.submit_enrollment_segment(
        enrollment.enrollment_id,
        "profile-a",
        1,
        _pcm(),
    )
    assert retry.enrollment == first.enrollment
    assert model.inference_count == 1
    with pytest.raises(VoiceIdentityServiceError, match="stale_enrollment"):
        await service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-b",
            1,
            _pcm(),
        )
    await service.cancel_enrollment(enrollment.enrollment_id)
    await service.close()

async def test_normalization_unavailable_never_replaces_existing_profile(
    tmp_path: Path,
) -> None:
    failure_code: list[str | None] = [None]

    def factory(enabled: bool) -> _AudioNormalizer:
        return _AudioNormalizer(enabled, failure_code=failure_code[0])

    service, model, _activations, _events = _service(
        tmp_path,
        audio_normalizer_factory=factory,
    )
    await service.initialize()
    enrollment = await service.start_enrollment()
    completed = await service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )
    old_generation = completed.profile_generation
    old_inference_count = model.inference_count

    failure_code[0] = "audio_processing_unavailable"
    reenrollment = await service.start_enrollment()
    with pytest.raises(
        VoiceIdentityServiceError,
        match="audio_processing_unavailable",
    ):
        await service.submit_enrollment_segment(
            reenrollment.enrollment_id,
            "profile-b",
            1,
            _pcm(),
        )

    current = service.status()
    assert current.state.has_profile
    assert current.state.effective_reason == "ready"
    assert current.profile_generation == old_generation
    assert current.enrollment is None
    assert model.inference_count == old_inference_count
    await service.close()

async def test_normalization_unavailable_without_profile_degrades_runtime(
    tmp_path: Path,
) -> None:
    def factory(enabled: bool) -> _AudioNormalizer:
        return _AudioNormalizer(
            enabled,
            failure_code="audio_processing_unavailable",
        )

    service, model, _activations, _events = _service(
        tmp_path,
        audio_normalizer_factory=factory,
    )
    await service.initialize()
    enrollment = await service.start_enrollment()

    with pytest.raises(
        VoiceIdentityServiceError,
        match="audio_processing_unavailable",
    ):
        await service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-a",
            1,
            _pcm(),
        )

    current = service.status()
    assert not current.state.has_profile
    assert not current.state.effective_enabled
    assert current.state.effective_reason == "runtime_degraded"
    assert current.enrollment is None
    assert model.inference_count == 0
    await service.close()

async def test_commit_linearizes_before_concurrent_profile_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    for segment_index in (1, 2, 3):
        await service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-a",
            segment_index,
            _pcm(),
        )

    profile_store = service._profile_store  # type: ignore[attr-defined]
    original_stage = profile_store.stage
    stage_started = threading.Event()
    stage_release = threading.Event()

    def blocking_stage(profile: SpeakerProfile, *, audio_contract):
        stage_started.set()
        assert stage_release.wait(1.0)
        return original_stage(profile, audio_contract=audio_contract)

    monkeypatch.setattr(profile_store, "stage", blocking_stage)
    completion = asyncio.create_task(
        service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-a",
            4,
            _verification_pcm(),
        )
    )
    assert await asyncio.to_thread(stage_started.wait, 1.0)
    deletion = asyncio.create_task(service.delete_profile())
    await asyncio.sleep(0)
    assert not deletion.done()
    stage_release.set()

    committed = await completion
    deleted = await deletion
    assert committed.profile_generation == "profile-a"
    assert deleted.profile_generation is None
    assert not deleted.state.has_profile
    await service.close()

async def test_unsupported_route_saves_profile_without_reporting_ready(
    tmp_path: Path,
) -> None:
    service, model, _activations, suppression_events = _service(
        tmp_path,
        activation_results=[VoiceIdentityActivationResult.UNSUPPORTED_ASR_ROUTE],
    )
    await service.initialize()
    enrollment = await service.start_enrollment()

    status = await service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )

    assert status.state.requested_enabled
    assert not status.state.effective_enabled
    assert status.state.effective_reason == "unsupported_asr_route"
    assert status.state.has_profile
    assert status.profile_generation == "profile-a"
    assert model.closed
    assert suppression_events[-1] == "restore:voice_identity_enrollment"
    await service.close()

async def test_cancelled_profile_staging_aborts_completed_worker_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    profile_store = service._profile_store  # type: ignore[attr-defined]
    original_stage = profile_store.stage
    stage_started = threading.Event()
    stage_release = threading.Event()

    def blocking_stage(profile: SpeakerProfile, *, audio_contract):
        stage_started.set()
        assert stage_release.wait(1.0)
        return original_stage(profile, audio_contract=audio_contract)

    monkeypatch.setattr(profile_store, "stage", blocking_stage)
    completion = asyncio.create_task(
        service.complete_enrollment(
            enrollment.enrollment_id,
            "profile-a",
            _pcm(),
        )
    )
    assert await asyncio.to_thread(stage_started.wait, 1.0)
    completion.cancel()
    stage_release.set()

    with pytest.raises(asyncio.CancelledError):
        await completion

    assert list(tmp_path.glob(".*.tmp")) == []
    assert not (tmp_path / "voice_identity.profile").exists()
    await service.close()

async def test_status_stays_valid_while_profile_delete_detaches(
    tmp_path: Path,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    await service.complete_enrollment(enrollment.enrollment_id, "profile-a", _pcm())
    detach_started = asyncio.Event()
    detach_release = asyncio.Event()

    async def blocking_activate(
        profile: SpeakerProfile | None,
        generation: str,
        **_authority,
    ) -> bool:
        del generation
        if profile is None:
            detach_started.set()
            await detach_release.wait()
        return True

    service._activation_callback = blocking_activate  # type: ignore[attr-defined]
    deletion = asyncio.create_task(service.delete_profile())
    await asyncio.wait_for(detach_started.wait(), 1.0)

    status = service.status()
    assert not status.state.requested_enabled
    assert not status.state.effective_enabled
    assert status.state.effective_reason == "disabled"

    detach_release.set()
    await deletion
    await service.close()

async def test_initialize_restores_encrypted_profile_and_preference(
    tmp_path: Path,
) -> None:
    first_service, _model, _activations, _events = _service(tmp_path)
    await first_service.initialize()
    enrollment = await first_service.start_enrollment()
    await first_service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )
    await first_service.close()

    restored, _restored_model, activations, _restored_events = _service(tmp_path)
    status = await restored.initialize()

    assert status.state.requested_enabled
    assert status.state.effective_enabled
    assert status.state.has_profile
    assert activations[-1][1] == "profile-a"
    await restored.close()

async def test_shadow_mode_records_profile_without_reporting_enforced(
    tmp_path: Path,
) -> None:
    service, _model, activations, _events = _service(
        tmp_path,
        runtime_mode="shadow",
    )
    await service.initialize()
    enrollment = await service.start_enrollment()

    status = await service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )

    assert status.runtime_mode == "shadow"
    assert status.state.requested_enabled
    assert not status.state.effective_enabled
    assert status.state.effective_reason == "shadow_mode"
    assert activations[-1][1] == "profile-a"
    await service.close()

async def test_cancelled_profile_commit_keeps_memory_and_disk_on_new_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, activations, _events = _service(tmp_path)
    await service.initialize()
    first = await service.start_enrollment()
    await service.complete_enrollment(first.enrollment_id, "profile-a", _pcm())
    second = await service.start_enrollment()
    replace_started = threading.Event()
    replace_release = threading.Event()
    original_replace = store_module._replace

    def blocking_replace(source: Path, destination: Path) -> None:
        replace_started.set()
        if not replace_release.wait(1.0):
            raise TimeoutError("test did not release profile commit")
        original_replace(source, destination)

    monkeypatch.setattr(store_module, "_replace", blocking_replace)
    completion = asyncio.create_task(
        service.complete_enrollment(second.enrollment_id, "profile-b", _pcm())
    )
    assert await asyncio.to_thread(replace_started.wait, 1.0)
    completion.cancel()
    replace_release.set()

    with pytest.raises(asyncio.CancelledError):
        await completion

    status = service.status()
    assert status.state.effective_enabled
    assert status.profile_generation == "profile-b"
    assert activations[-1][1] == "profile-b"
    stored = await service._profile_store.aload()  # type: ignore[attr-defined]
    assert stored is not None
    try:
        assert stored.profile.generation == "profile-b"
    finally:
        stored.close()
    await service.close()

async def test_initialize_maps_profile_storage_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    reason: str,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service._preference_store.asave(True)  # type: ignore[attr-defined]
    authority_calls: list[dict[str, object]] = []

    async def capture_unavailable_authority(
        profile: SpeakerProfile | None,
        generation: str,
        **authority,
    ) -> bool:
        del generation
        assert profile is None
        authority_calls.append(authority)
        return True

    service._activation_callback = capture_unavailable_authority  # type: ignore[attr-defined]

    async def fail_load():
        raise failure

    monkeypatch.setattr(
        service._profile_store,  # type: ignore[attr-defined]
        "aload",
        fail_load,
    )
    status = await service.initialize()
    assert status.state.effective_reason == reason
    assert authority_calls[-1]["activation_required"] is True
    assert authority_calls[-1]["noise_reduction_enabled"] is True
    await service.close()

async def test_initialize_maps_preference_failure_and_incompatible_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken, _model, _activations, _events = _service(tmp_path / "broken")

    async def fail_preference():
        raise VoiceIdentityPreferenceStoreError("corrupt")

    monkeypatch.setattr(
        broken._preference_store,  # type: ignore[attr-defined]
        "aload",
        fail_preference,
    )
    assert (await broken.initialize()).state.effective_reason == "runtime_degraded"
    await broken.close()

    service, _model, _activations, _events = _service(
        tmp_path / "incompatible",
        runtime_status_results=[VoiceIdentityActivationResult.READY],
    )
    reference = SpeakerReference(
        SpeakerModelIdentity("other-model", "v1", 2),
        [1.0, 0.0],
    )
    try:
        profile = SpeakerProfile("incompatible", reference)
    finally:
        reference.close()
    try:
        await service._profile_store.asave(  # type: ignore[attr-defined]
            profile,
            audio_contract=desktop_audio_contract_snapshot(
                noise_reduction_enabled=True,
            ),
        )
        await service._preference_store.asave(True)  # type: ignore[attr-defined]
    finally:
        profile.close()

    status = await service.initialize()
    assert status.state.effective_reason == "profile_incompatible"
    await service.close()

async def test_delete_rolls_back_profile_when_preference_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    await service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )

    async def fail_preference(_enabled: bool) -> None:
        raise VoiceIdentityPreferenceStoreError("write failed")

    monkeypatch.setattr(
        service._preference_store,  # type: ignore[attr-defined]
        "asave",
        fail_preference,
    )
    with pytest.raises(VoiceIdentityServiceError, match="runtime_degraded"):
        await service.delete_profile()

    restored = await service._profile_store.aload()  # type: ignore[attr-defined]
    assert restored is not None
    try:
        assert restored.profile.generation == "profile-a"
    finally:
        restored.close()
    status = service.status()
    assert status.state.requested_enabled
    assert status.state.has_profile
    assert status.state.effective_enabled
    assert activations[-1][0] is not None
    await service.close()

async def test_cancelled_profile_delete_reconciles_memory_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    await service.complete_enrollment(enrollment.enrollment_id, "profile-a", _pcm())
    old_profile = service._profile  # type: ignore[attr-defined]
    assert old_profile is not None
    delete_started = threading.Event()
    delete_release = threading.Event()
    profile_store = service._profile_store  # type: ignore[attr-defined]
    original_delete = profile_store.delete

    def blocking_delete() -> bool:
        delete_started.set()
        assert delete_release.wait(1.0)
        return original_delete()

    monkeypatch.setattr(profile_store, "delete", blocking_delete)
    deletion = asyncio.create_task(service.delete_profile())
    assert await asyncio.to_thread(delete_started.wait, 1.0)
    deletion.cancel()
    delete_release.set()

    with pytest.raises(asyncio.CancelledError):
        await deletion

    status = service.status()
    assert not status.state.requested_enabled
    assert not status.state.has_profile
    assert not status.state.effective_enabled
    assert status.state.effective_reason == "disabled"
    assert old_profile.closed
    assert await profile_store.aload() is None
    assert activations[-1][0] is None
    await service.close()
