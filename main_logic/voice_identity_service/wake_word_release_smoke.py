"""Offline release gate executed by the frozen binary, before app startup."""

from __future__ import annotations

import os


def main() -> int:
    try:
        import sherpa_onnx
        from main_logic.voice_input.wake_word.sherpa_backend import SUPPORTED_RUNTIME_VERSION, SherpaWakeWordConfig, validate_wake_word_resources
        from config.voice_wake_word import DEFAULT_WAKE_WORD_KEYWORDS
        if sherpa_onnx.__version__ != SUPPORTED_RUNTIME_VERSION or sherpa_onnx.version != SUPPORTED_RUNTIME_VERSION:
            raise ValueError("WAKE_WORD_RUNTIME_FIX_REQUIRED")
        model_dir = os.environ.get("NEKO_WAKE_WORD_MODEL_DIR", "")
        if not model_dir:
            raise ValueError("WAKE_WORD_MODEL_MISSING")
        validate_wake_word_resources(SherpaWakeWordConfig(model_dir=model_dir, keywords=DEFAULT_WAKE_WORD_KEYWORDS))
    except Exception:
        print("WAKE_WORD_RELEASE_SMOKE_FAILED", flush=True)
        return 1
    print("WAKE_WORD_RELEASE_SMOKE_READY", flush=True)
    return 0
