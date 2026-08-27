from __future__ import annotations

import app.main_server.voice_identity_runtime as voice_identity_runtime
from main_routers import debug_router


def test_debug_health_voice_identity_diagnostics_keep_only_safe_counters(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        voice_identity_runtime,
        "get_voice_identity_diagnostics",
        lambda: {
            "observation_count": 3,
            "rejection_task_applied_count": 1,
            "similarity": 0.12,
            "embedding": [1.0, 0.0],
            "negative": -1,
            "boolean": True,
        },
    )

    assert debug_router._safe_voice_identity_diagnostics() == {
        "observation_count": 3,
        "rejection_task_applied_count": 1,
    }
