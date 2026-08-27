"""Provider-neutral contracts for aggregate voice-identity diagnostics."""

VOICE_IDENTITY_DIAGNOSTIC_COUNTERS = frozenset(
    {
        "observation_count",
        "first_checkpoint_count",
        "second_checkpoint_count",
        "low_checkpoint_count",
        "reject_decision_count",
        "rejection_request_failed_count",
        "rejection_task_scheduled_count",
        "rejection_task_applied_count",
        "rejection_task_stale_count",
        "rejection_stale_initial_count",
        "rejection_stale_prepare_count",
        "rejection_stale_runtime_fence_count",
        "rejection_stale_candidate_fence_count",
        "rejection_stale_smart_turn_count",
        "rejection_stale_commit_count",
        "rejection_prepare_detector_closed_count",
        "rejection_prepare_candidate_closed_count",
        "rejection_prepare_epoch_mismatch_count",
        "rejection_prepare_shadow_mismatch_count",
        "rejection_prepare_unbound_count",
        "rejection_task_cleanup_degraded_count",
        "rejection_task_failure_count",
        "rejection_task_cancelled_count",
        "rejection_task_pending_count",
        "rejection_in_progress_count",
        "verifier_installed_count",
        "verifier_degraded_count",
        "registered_manager_count",
        "diagnostic_runtime_count",
    }
)

__all__ = ["VOICE_IDENTITY_DIAGNOSTIC_COUNTERS"]
