from __future__ import annotations

from main_routers import debug_router


def test_debug_health_voice_identity_diagnostics_keep_only_safe_counters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        debug_router,
        "_VOICE_IDENTITY_DIAGNOSTICS_PROVIDER",
        lambda: {
            "observation_count": 3,
            "rejection_task_applied_count": 1,
            "speaker_gate_armed_count": 2,
            "speaker_gate_armed_while_preparing_count": 1,
            "speaker_gate_waited_count": 1,
            "speaker_gate_resolved_forward_count": 1,
            "speaker_gate_resolved_reject_count": 1,
            "speaker_gate_timeout_count": 0,
            "speaker_gate_stale_count": 0,
            "speaker_gate_released_prepare_failure_count": 1,
            "micro_event_candidate_count": 3,
            "micro_event_evidence_complete_count": 2,
            "micro_event_evidence_unavailable_count": 1,
            "micro_event_would_suppress_count": 1,
            "micro_event_suppressed_count": 0,
            "micro_event_shadow_forward_count": 1,
            "micro_event_fail_open_count": 1,
            "micro_event_stale_fence_count": 0,
            "micro_event_rnnoise_unavailable_count": 1,
            "similarity": 0.12,
            "embedding": [1.0, 0.0],
            "unexpected": 99,
            "negative": -1,
            "boolean": True,
        },
    )

    assert debug_router._safe_voice_identity_diagnostics() == {
        "observation_count": 3,
        "rejection_task_applied_count": 1,
        "speaker_gate_armed_count": 2,
        "speaker_gate_armed_while_preparing_count": 1,
        "speaker_gate_waited_count": 1,
        "speaker_gate_resolved_forward_count": 1,
        "speaker_gate_resolved_reject_count": 1,
        "speaker_gate_timeout_count": 0,
        "speaker_gate_stale_count": 0,
        "speaker_gate_released_prepare_failure_count": 1,
        "micro_event_candidate_count": 3,
        "micro_event_evidence_complete_count": 2,
        "micro_event_evidence_unavailable_count": 1,
        "micro_event_would_suppress_count": 1,
        "micro_event_suppressed_count": 0,
        "micro_event_shadow_forward_count": 1,
        "micro_event_fail_open_count": 1,
        "micro_event_stale_fence_count": 0,
        "micro_event_rnnoise_unavailable_count": 1,
    }
