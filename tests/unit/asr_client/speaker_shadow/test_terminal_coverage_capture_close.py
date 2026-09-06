"""Terminal coverage must publish one ordered close, preserving scored facts."""

import pytest

from main_logic.asr_client.speaker_shadow.contracts import (
    SpeakerShadowCompletion, SpeakerShadowObservation, SpeakerShadowTerminalCoverageRequest,
)
from main_logic.asr_client.speaker_shadow.runtime import SpeakerShadowRuntime
from tests.unit.asr_client.test_provider_speaker_continuity import _GatedScoreHost
from .test_runtime import (
    _BackendFactory, _candidate, _pcm, _provider_gate_config,
    _reconcile_source, _finalize_provider_candidate_score,
)


@pytest.mark.parametrize("score", [.2, .95])
@pytest.mark.parametrize("operation", ["commit", "abort", "revoke", "reset"])
async def test_terminal_coverage_close_respects_commit_and_revoke(score, operation):
    evidence = []
    runtime = SpeakerShadowRuntime(backend_factory=_BackendFactory(score_value=score),
        config=_provider_gate_config(), on_evidence=evidence.append)
    host = _GatedScoreHost(score, False)
    host.ready.set()
    runtime._backend_host = host
    target, successor = _candidate(110001), _candidate(110002)
    try:
        await _finalize_provider_candidate_score(runtime, target)
        original = runtime._finalized[target].token
        assert len(evidence) == 2
        receipt = runtime.prepare_finalized_candidate_coverage(SpeakerShadowTerminalCoverageRequest(
            sources=(_reconcile_source(target, 12_000, keep_end_ms=10_000),),
            target=target, suffix=successor, provider_exact_start_sample=0,
            provider_exact_end_sample=160_000, scored_window_start_sample=0,
            scored_window_end_sample=48_000))
        assert receipt is not None
        assert len(evidence) == 2
        if operation == "abort":
            assert runtime.abort_finalized_candidate_coverage(receipt)
        else:
            assert runtime.commit_finalized_candidate_coverage(receipt)
            if operation == "revoke":
                runtime.revoke_terminal_coverage(receipt)
            elif operation == "reset":
                await runtime.reset()
        await runtime.wait_idle()
        closes = [e for e in evidence if isinstance(e, SpeakerShadowCompletion)
            and e.candidate == target and e.terminal_reason == "scored" and e.evidence_complete]
        if operation != "commit":
            assert closes == []
            return
        assert len(closes) == 1
        assert [type(e) for e in evidence] == [SpeakerShadowObservation, SpeakerShadowObservation, SpeakerShadowCompletion]
        assert closes[0].through_sequence_no == 2
        assert [e.similarity for e in evidence[:2]] == [score, score]
        assert runtime._finalized[target].token is original
        assert host.calls == 2
        assert runtime.finish_candidate(target)
        await runtime.wait_idle()
        assert len(evidence) == 3
        assert runtime.complete_reconciliation(receipt, successor=successor) == "completed"
        runtime.revoke_terminal_coverage(receipt)
        assert runtime.submit(_pcm(100), sample_rate_hz=16_000, candidate=successor)
        await runtime.wait_idle()
        assert runtime._buffers[successor].sample_count == 1600
        assert host.calls == 2
    finally:
        await runtime.close()
