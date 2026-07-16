"""Immutable domain models for normalized carrier events."""

from datetime import date
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from shipment_triage.domain.statuses import CanonicalStatus

ShipmentId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]


class DomainModel(BaseModel):
    """Strict immutable base for trusted domain values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RawRecordRef(DomainModel):
    line_number: int = Field(ge=1)
    raw_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataQualityIssue(DomainModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=400)
    record_refs: tuple[RawRecordRef, ...] = ()


class Location(DomainModel):
    city: str | None = Field(default=None, min_length=1, max_length=120)
    state: str | None = Field(default=None, min_length=1, max_length=40)


class NormalizedEvent(DomainModel):
    event_id: str = Field(pattern=r"^evt_[0-9a-f]{20}$")
    shipment_id: ShipmentId
    carrier: str = Field(min_length=2, max_length=10)
    status: CanonicalStatus
    raw_status: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=300)
    occurred_at: AwareDatetime
    location: Location | None = None
    promised_date: date | None = None
    provenance: tuple[RawRecordRef, ...] = Field(min_length=1)
    data_quality_issues: tuple[DataQualityIssue, ...] = ()


class RejectedRecord(DomainModel):
    line_number: int = Field(ge=1)
    raw_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=400)
    excerpt: str = Field(max_length=240)


class FeedLoadResult(DomainModel):
    raw_record_count: int = Field(ge=0)
    exact_duplicate_count: int = Field(ge=0)
    semantic_merge_count: int = Field(ge=0)
    events: tuple[NormalizedEvent, ...]
    rejected_records: tuple[RejectedRecord, ...]


__all__ = [
    "DataQualityIssue",
    "DomainModel",
    "FeedLoadResult",
    "Location",
    "NormalizedEvent",
    "RawRecordRef",
    "RejectedRecord",
    "ShipmentId",
]
