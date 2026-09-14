"""Shared activation test harness exports for PR #3078."""
from tests.unit.test_voice_activation_handoff import _Clock, _Factory, _harness, _until
from tests.unit.test_voice_activation_cold_prefix import _cold_harness, _feed
__all__ = ["_Clock", "_Factory", "_harness", "_until", "_cold_harness", "_feed"]
