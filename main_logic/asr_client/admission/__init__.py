"""Provider-neutral transcript admission state machine."""

from .contracts import (
    AdmissionDisposition,
    AdmissionEffect,
    AdmissionEvent,
    AdmissionState,
    VoiceTurnAdmissionRecord,
)
from .coordinator import (
    AdmissionCapacityError,
    AdmissionIdentityError,
    VoiceTurnAdmissionCoordinator,
)
from .reducer import maybe_resolve, reduce

__all__ = [
    "AdmissionCapacityError",
    "AdmissionDisposition",
    "AdmissionIdentityError",
    "AdmissionEffect",
    "AdmissionEvent",
    "AdmissionState",
    "VoiceTurnAdmissionCoordinator",
    "VoiceTurnAdmissionRecord",
    "maybe_resolve",
    "reduce",
]
