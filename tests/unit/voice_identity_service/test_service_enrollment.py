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

async def test_enrollment_audio_timeout_detaches_cancellation_swallowing_normalizer(
    tmp_path: Path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _StuckAudioNormalizer:
        def __init__(self, nr_enabled: bool) -> None:
            self.nr_enabled = nr_enabled

        async def normalize(self, *_args, **_kwargs) -> bytes:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return _pcm()

    service, _model, _activations, _events = _service(
        tmp_path,
        model_timeout_seconds=0.1,
        audio_normalizer_factory=_StuckAudioNormalizer,
    )
    await service.initialize()
    enrollment = await service.start_enrollment()
    submit = asyncio.create_task(
        service.submit_enrollment_segment(
            enrollment.enrollment_id,
            "profile-a",
            1,
            _pcm(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        with pytest.raises(
            VoiceIdentityServiceError,
            match="audio_processing_unavailable",
        ):
            await asyncio.wait_for(asyncio.shield(submit), timeout=0.5)
        cleanup = getattr(service, "_enrollment_audio_cleanup_task", None)
        assert cleanup is not None and not cleanup.done()
        with pytest.raises(
            VoiceIdentityServiceError,
            match="audio_processing_unavailable",
        ):
            await service.start_enrollment()
    finally:
        release.set()
        await asyncio.gather(submit, return_exceptions=True)
        await _wait_until(
            lambda: getattr(service, "_enrollment_audio_cleanup_task", None) is None
        )
        await service.close()

async def test_first_enrollment_loads_before_suppression_and_enables(
    tmp_path: Path,
) -> None:
    service, model, activations, suppression_events = _service(tmp_path)
    await service.initialize()

    enrollment = await service.start_enrollment()
    assert suppression_events == ["suppress:voice_identity_enrollment"]
    status = await service.complete_enrollment(
        enrollment.enrollment_id,
        "profile-a",
        _pcm(),
    )

    assert status.state.requested_enabled
    assert status.state.effective_enabled
    assert status.state.has_profile
    assert status.profile_generation == "profile-a"
    assert activations[-1][1] == "profile-a"
    assert suppression_events[-1] == "restore:voice_identity_enrollment"
    assert model.closed
    await service.close()

async def test_enrollment_expiry_uses_suppression_lease_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, model, _activations, suppression_events = _service(
        tmp_path,
        enrollment_ttl_seconds=0.2,
    )
    await service.initialize()
    lease_released = False

    class ShortLease:
        def __init__(self, expires_at: float) -> None:
            self.expires_at = expires_at

        async def release(self) -> bool:
            nonlocal lease_released
            lease_released = True
            suppression_events.append("restore:voice_identity_enrollment")
            return True

    async def slow_acquire(reason: str, *, ttl_seconds: float):
        del ttl_seconds
        suppression_events.append(f"suppress:{reason}")
        expires_at = asyncio.get_running_loop().time() + 0.02
        await asyncio.sleep(0.05)
        return ShortLease(expires_at)

    monkeypatch.setattr(service._suppression_controller, "acquire", slow_acquire)  # type: ignore[attr-defined]

    await service.start_enrollment()
    await _wait_until(
        lambda: (
            lease_released
            and model.closed
            and service.status().enrollment is None
        ),
        timeout_seconds=0.1,
    )

    assert lease_released
    assert model.closed
    assert suppression_events[-1] == "restore:voice_identity_enrollment"
    await service.close()

async def test_expired_enrollment_completion_is_rejected_and_cleaned(
    tmp_path: Path,
) -> None:
    service, model, _activations, suppression_events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    session = service._enrollment  # type: ignore[attr-defined]
    assert session is not None
    session.expires_at = asyncio.get_running_loop().time() - 0.001

    with pytest.raises(VoiceIdentityServiceError, match="stale_enrollment"):
        await service.complete_enrollment(enrollment.enrollment_id, "profile-a", _pcm())

    assert service.status().enrollment is None
    assert model.closed
    assert suppression_events[-1] == "restore:voice_identity_enrollment"
    await service.close()

async def test_enrollment_completion_preserves_latest_explicit_filter_choice(
    tmp_path: Path, has_profile: bool, enabled: bool,
) -> None:
    service, _model, activations, _events = _service(tmp_path)
    await service.initialize()
    try:
        if has_profile:
            first = await service.start_enrollment()
            await service.complete_enrollment(first.enrollment_id, "old", _pcm())
            await service.set_filter(not enabled)
        enrollment = await service.start_enrollment()
        for index in range(1, 4):
            await service.submit_enrollment_segment(
                enrollment.enrollment_id, "new", index, _pcm(),
            )
        # The active enrollment survives a page reload/retry. A filter request
        # can arrive before its last segment is committed.
        await service.set_filter(enabled)
        status = await service.submit_enrollment_segment(
            enrollment.enrollment_id, "new", 4, _verification_pcm(),
        )
        assert status.profile_generation == "new"
        assert status.state.requested_enabled is enabled
        assert status.state.effective_enabled is enabled
        assert await service._preference_store.aload() is enabled
        if enabled:
            assert activations[-1][1] == "new"
        else:
            assert activations[-1][0] is None
    finally:
        await service.close()

async def test_failed_filter_save_does_not_replace_enrollment_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service.initialize()
    try:
        enrollment = await service.start_enrollment()

        async def fail_save(_enabled):
            raise VoiceIdentityPreferenceStoreError("cannot save")

        with monkeypatch.context() as patch:
            patch.setattr(service._preference_store, "asave", fail_save)
            with pytest.raises(VoiceIdentityServiceError, match="runtime_degraded"):
                await service.set_filter(False)
        status = await service.complete_enrollment(enrollment.enrollment_id, "new", _pcm())
        assert status.state.requested_enabled
        assert status.state.effective_enabled
        assert await service._preference_store.aload()
    finally:
        await service.close()

async def test_reenrollment_while_disabled_keeps_user_preference(
    tmp_path: Path,
) -> None:
    service, _model, activations, _events = _service(tmp_path)
    await service.initialize()
    first = await service.start_enrollment()
    await service.complete_enrollment(first.enrollment_id, "profile-a", _pcm())
    await service.set_filter(False)
    activation_count = len(activations)

    second = await service.start_enrollment()
    status = await service.complete_enrollment(
        second.enrollment_id,
        "profile-b",
        _pcm(),
    )

    assert not status.state.requested_enabled
    assert not status.state.effective_enabled
    assert status.state.has_profile
    assert len(activations) == activation_count
    await service.close()

async def test_cancelled_enrollment_cancel_retains_cleanup_to_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _model, _activations, _events = _service(tmp_path)
    await service.initialize()
    enrollment = await service.start_enrollment()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_completed = False

    async def blocking_cleanup(session) -> bool:
        nonlocal cleanup_completed
        cleanup_started.set()
        await cleanup_release.wait()
        cleanup_completed = True
        return True

    monkeypatch.setattr(service, "_cleanup_session", blocking_cleanup)
    cancellation = asyncio.create_task(service.cancel_enrollment(enrollment.enrollment_id))
    await asyncio.wait_for(cleanup_started.wait(), 1.0)
    cancellation.cancel()
    cleanup_release.set()

    with pytest.raises(asyncio.CancelledError):
        await cancellation

    assert cleanup_completed
    assert service.status().state.effective_reason == "disabled"
    await service.close()
