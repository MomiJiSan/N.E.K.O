"""Resolve deployment preference and immutable installed resources off-loop."""

from dataclasses import dataclass

from config.voice_wake_word import wake_word_model_dir, wake_word_preference
from .errors import WakeWordFailureReason
from .model_bundle import WakeWordBundleError, resolve_cached_model_dir
from .sherpa_backend import WakeWordBackendError


@dataclass(frozen=True, slots=True)
class WakeWordResources:
    enabled: bool
    model_dir: str | None = None
    reason: str | None = None


def resolve_wake_word_resources() -> WakeWordResources:
    """Call on a worker thread. Discovery never enables the capability."""
    preference = wake_word_preference()
    if preference["reason"]:
        return WakeWordResources(True, reason=WakeWordFailureReason.PREFERENCE_UNAVAILABLE.value)
    if not preference["enabled"]:
        return WakeWordResources(False)
    explicit = wake_word_model_dir()
    if explicit:
        return WakeWordResources(True, explicit)
    try:
        cached = resolve_cached_model_dir()
    except WakeWordBundleError:
        return WakeWordResources(True, reason=WakeWordFailureReason.MODEL_INVALID.value)
    return (WakeWordResources(True, str(cached)) if cached is not None else
            WakeWordResources(True, reason=WakeWordFailureReason.MODEL_MISSING.value))


class UnavailableWakeWordDetector:
    """Preserve configured wake failure as a closed activation authority."""

    inference_timeout_seconds = 2.0
    runtime_info = None

    def __init__(self, reason: str):
        self.reason = reason

    async def prepare(self):
        raise WakeWordBackendError(self.reason)

    async def feed_batch(self, frames, epoch):
        raise WakeWordBackendError(self.reason)

    async def close(self):
        pass
