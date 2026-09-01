"""Provider-neutral transcript admission state machine."""

from .contracts import (
    AdmissionDisposition,
    AdmissionEffect,
    AdmissionEvent,
    AdmissionState,
    VoiceTurnAdmissionRecord,
)
from .coordinator import (
    AdmissionBulkResult,
    AdmissionCapacityError,
    AdmissionIdentityError,
    VoiceTurnAdmissionCoordinator,
)
from .ingress import (
    AdmissionIngressCapacityError,
    AdmissionIngressClosedError,
    AdmissionIngressLane,
)
from .reducer import maybe_resolve, reduce

__all__ = [
    "AdmissionBulkResult",
    "AdmissionCapacityError",
    "AdmissionDisposition",
    "AdmissionIdentityError",
    "AdmissionIngressCapacityError",
    "AdmissionIngressClosedError",
    "AdmissionIngressLane",
    "AdmissionEffect",
    "AdmissionEvent",
    "AdmissionState",
    "VoiceTurnAdmissionCoordinator",
    "VoiceTurnAdmissionRecord",
    "maybe_resolve",
    "reduce",
]
