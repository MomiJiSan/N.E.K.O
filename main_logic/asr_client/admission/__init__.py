"""Provider-neutral transcript admission state machine."""

from .contracts import (
    AdmissionDisposition,
    AdmissionEffect,
    AdmissionEvent,
    AdmissionState,
    SpeakerCaptureLeaseRecord,
    SpeakerCaptureLeaseToken,
    SpeakerLeaseAbandoned,
    SpeakerLeaseCaptureClosed,
    SpeakerLeaseEvent,
    SpeakerLeaseHigh,
    SpeakerLeaseLow,
    SpeakerLeaseState,
    SpeakerLeaseUnavailable,
    VoiceTurnAdmissionRecord,
)
from .coordinator import (
    AdmissionBulkResult,
    AdmissionCapacityError,
    AdmissionIdentityError,
    SpeakerLeaseCapacityError,
    VoiceTurnAdmissionCoordinator,
)
from .ingress import (
    AdmissionIngressCapacityError,
    AdmissionIngressClosedError,
    AdmissionIngressLane,
)
from .reducer import maybe_resolve, reduce
from .speaker_leases import (
    MAX_SPEAKER_LEASE_CHILDREN,
    MAX_SPEAKER_LEASES,
    SpeakerLeaseChildCapacityError,
    SpeakerLeaseIdentityError,
    SpeakerLeaseTerminalError,
    bind_speaker_lease_child,
    reduce_speaker_lease,
)

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
    "MAX_SPEAKER_LEASE_CHILDREN",
    "MAX_SPEAKER_LEASES",
    "SpeakerCaptureLeaseRecord",
    "SpeakerCaptureLeaseToken",
    "SpeakerLeaseAbandoned",
    "SpeakerLeaseCapacityError",
    "SpeakerLeaseCaptureClosed",
    "SpeakerLeaseChildCapacityError",
    "SpeakerLeaseEvent",
    "SpeakerLeaseHigh",
    "SpeakerLeaseIdentityError",
    "SpeakerLeaseLow",
    "SpeakerLeaseState",
    "SpeakerLeaseTerminalError",
    "SpeakerLeaseUnavailable",
    "VoiceTurnAdmissionCoordinator",
    "VoiceTurnAdmissionRecord",
    "bind_speaker_lease_child",
    "maybe_resolve",
    "reduce",
    "reduce_speaker_lease",
]
