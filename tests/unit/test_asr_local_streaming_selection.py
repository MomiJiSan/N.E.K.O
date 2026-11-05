"""Local service selection stays inside existing route and delivery contracts."""
from unittest.mock import AsyncMock

import pytest

import main_logic.asr_client as client
from main_logic.asr_client._registry_meta import (
    ASR_PROVIDER_REGISTRY,
    AsrProviderAvailability,
)
from main_logic.asr_client.provider_policy import resolve_provider_policy


def test_explicit_local_selection_does_not_read_cloud_credentials(monkeypatch):
    monkeypatch.delenv("ASR_PROVIDER", raising=False)
    def forbidden():
        raise AssertionError("local selection must not read Core API keys")
    monkeypatch.setattr(client, "_load_core_config", forbidden)
    selected = client._resolve_asr_selection("qwen", provider_preference="local_streaming")
    assert selected.provider_key == "local_streaming"
    assert selected.endpointing_mode == "manual"
    assert selected._api_key == ""
    assert selected.availability is AsrProviderAvailability.IMPLEMENTED


def test_local_selection_cannot_override_native_only_core(monkeypatch):
    monkeypatch.delenv("ASR_PROVIDER", raising=False)
    monkeypatch.setattr(client, "_load_core_config", lambda: {})
    selected = client._resolve_asr_selection("free", provider_preference="local_streaming")
    assert selected.provider_key == "free"
    assert selected.availability is AsrProviderAvailability.BLOCKED_BACKEND


def test_local_policy_reuses_streaming_and_smart_turn():
    policy = resolve_provider_policy("local_streaming", "manual")
    assert policy.transport == "streaming"
    assert policy.endpoint_authority == "smart_turn"
    assert policy.smart_turn_required
    assert policy.replay_policy == "none"
    assert policy.supports_result_preserving_finish
    assert policy.supports_interception_delivery
    assert ASR_PROVIDER_REGISTRY["local_streaming"].category == "local_streaming"
    with pytest.raises(RuntimeError, match="ASR_ENDPOINTING_NOT_SUPPORTED"):
        resolve_provider_policy("local_streaming", "provider")


@pytest.mark.parametrize("language,expected", [("zh-CN", "zh-CN"), ("en-US", "en-US"), ("ja", "auto"), ("es", "auto")])
def test_language_hint_uses_existing_automatic_detection_fallback(language, expected):
    assert client._resolve_session_language("local_streaming", language) == expected


def test_service_registration_does_not_make_whisper_ui_available(monkeypatch):
    monkeypatch.setattr(client, "_optional_dependency_available", lambda _: False)
    monkeypatch.setenv("NEKO_ASR_TOKEN", "x" * 32)
    monkeypatch.delenv("NEKO_ASR_URL", raising=False)
    assert client.is_local_streaming_asr_configured()
    assert not client.is_local_asr_available()


@pytest.mark.asyncio
async def test_selected_service_missing_configuration_fails_without_cloud_fallback(monkeypatch):
    monkeypatch.delenv("NEKO_ASR_TOKEN", raising=False)
    selected = client._resolve_asr_selection("qwen", provider_preference="local_streaming")
    session = client._create_asr_session_from_selection(
        "qwen", selection=selected, external_endpointing_runtime=True,
        on_input_transcript=AsyncMock(), on_connection_error=AsyncMock(),
    )
    try:
        with pytest.raises(RuntimeError):
            await session.connect()
        assert session.last_failure_code == "ASR_LOCAL_CONFIG_INVALID"
        assert not session.is_ready
    finally:
        await session.close()
