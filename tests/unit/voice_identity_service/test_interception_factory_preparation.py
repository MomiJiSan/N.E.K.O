"""Preparation rejects incompatible input before any model owner is created."""

from dataclasses import replace

import pytest

from main_logic.voice_identity_service.interception_runtime import PrewireInterceptionFactory
from main_logic.voice_identity_service.prewire_gate.scheduler import ScorerCapabilities
from tests.unit.voice_identity_service.test_interception_runtime import _config, _Scorer, _Classifier


@pytest.mark.parametrize("changes", [
    {"max_held_pcm_bytes": 3200},
    {"window_samples": 14400, "scorer_capabilities": ScorerCapabilities(16000, 24000, (24000,), "model")},
])
def test_invalid_configuration_cannot_allocate_candidate_or_separator_resources(changes):
    allocations = []

    def allocate(stream):
        allocations.append(stream)
        raise AssertionError("model resources allocated before preparation")

    factory = PrewireInterceptionFactory(
        replace(_config(), **changes), score_backend=_Scorer(), classifier=_Classifier(),
        tse_factory=allocate, candidate_factory=allocate, candidate_source_factory=allocate,
    )
    with pytest.raises(ValueError):
        factory.create("generation", ingress_token=None)
    assert allocations == []
    assert factory._runtimes == []
