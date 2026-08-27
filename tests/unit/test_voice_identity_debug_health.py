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
    }
