from utils import preferences
from utils.conversation_settings_constants import (
    normalize_independent_asr_provider_preference_handshake,
)


def test_local_streaming_preference_survives_persistence_validation_and_handshake():
    value = {"independentAsrProviderPreference": "local_streaming"}
    assert preferences._validate_conversation_settings(value) == value
    assert normalize_independent_asr_provider_preference_handshake("local_streaming") == "local_streaming"
    assert normalize_independent_asr_provider_preference_handshake(None) is None
    assert normalize_independent_asr_provider_preference_handshake("remote_unregistered") == "auto"
