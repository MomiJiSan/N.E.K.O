"""Provider-neutral internal endpoint boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


ProviderBoundaryQuality: TypeAlias = Literal["exact", "unknown"]
ProviderEndpointPhase: TypeAlias = Literal["boundary", "ordered"]


@dataclass(frozen=True, slots=True)
class ProviderUtteranceKey:
    """One provider utterance inside a session generation and buffer epoch."""

    generation: int
    buffer_epoch: int
    utterance_id: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
            or isinstance(self.buffer_epoch, bool)
            or not isinstance(self.buffer_epoch, int)
            or self.buffer_epoch < 0
            or isinstance(self.utterance_id, bool)
            or not isinstance(self.utterance_id, int)
            or self.utterance_id < 1
        ):
            raise ValueError("ASR_PROVIDER_ENDPOINT_KEY_INVALID")


@dataclass(frozen=True, slots=True)
class ProviderAudioRange:
    """One half-open canonical 16 kHz mono sample range."""

    start_sample_16k: int
    end_sample_16k: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_sample_16k, bool)
            or not isinstance(self.start_sample_16k, int)
            or isinstance(self.end_sample_16k, bool)
            or not isinstance(self.end_sample_16k, int)
            or self.start_sample_16k < 0
            or self.end_sample_16k <= self.start_sample_16k
        ):
            raise ValueError("ASR_PROVIDER_AUDIO_RANGE_INVALID")


@dataclass(frozen=True, slots=True)
class ProviderEndpointNotification:
    """A keyed provider boundary update or its ordered final-delivery fence."""

    phase: ProviderEndpointPhase
    generation: int
    buffer_epoch: int
    utterance_id: int
    boundary_quality: ProviderBoundaryQuality
    audio_range: ProviderAudioRange | None

    def __post_init__(self) -> None:
        if self.phase not in ("boundary", "ordered"):
            raise ValueError("ASR_PROVIDER_ENDPOINT_PHASE_INVALID")
        ProviderUtteranceKey(
            generation=self.generation,
            buffer_epoch=self.buffer_epoch,
            utterance_id=self.utterance_id,
        )
        if self.boundary_quality == "exact":
            if not isinstance(self.audio_range, ProviderAudioRange):
                raise ValueError("ASR_PROVIDER_ENDPOINT_BOUNDARY_INVALID")
        elif self.boundary_quality == "unknown":
            if self.audio_range is not None:
                raise ValueError("ASR_PROVIDER_ENDPOINT_BOUNDARY_INVALID")
        else:
            raise ValueError("ASR_PROVIDER_ENDPOINT_QUALITY_INVALID")

    @property
    def key(self) -> ProviderUtteranceKey:
        return ProviderUtteranceKey(
            generation=self.generation,
            buffer_epoch=self.buffer_epoch,
            utterance_id=self.utterance_id,
        )

    @property
    def namespace(self) -> tuple[int, int]:
        """Return the physical Provider audio-timeline namespace."""

        return (self.generation, self.buffer_epoch)
