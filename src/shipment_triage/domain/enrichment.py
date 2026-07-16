"""Trusted tracking-enrichment domain models."""

from datetime import date
from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from shipment_triage.domain.models import DataQualityIssue, DomainModel


class EnrichmentStatus(StrEnum):
    VALID = "VALID"
    NOT_FOUND = "NOT_FOUND"
    FAILED = "FAILED"


class DataCompleteness(StrEnum):
    ENRICHED = "ENRICHED"
    FEED_ONLY = "FEED_ONLY"


class EnrichmentFailureReason(StrEnum):
    AUTH = "AUTH"
    AUTH_CIRCUIT_OPEN = "AUTH_CIRCUIT_OPEN"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    SERVER_ERROR = "SERVER_ERROR"
    CLIENT_ERROR = "CLIENT_ERROR"
    TIMEOUT = "TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    MISMATCHED_SHIPMENT = "MISMATCHED_SHIPMENT"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class AttemptOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    INVALID_BODY = "INVALID_BODY"
    MISMATCHED_SHIPMENT = "MISMATCHED_SHIPMENT"


class TrackingLocation(DomainModel):
    city: str | None = Field(default=None, min_length=1, max_length=120)
    state: str | None = Field(default=None, min_length=1, max_length=40)


class TrackingScan(DomainModel):
    time: AwareDatetime
    status: str = Field(min_length=1, max_length=160)
    city: str | None = Field(default=None, min_length=1, max_length=120)
    state: str | None = Field(default=None, min_length=1, max_length=40)


class ReferenceNumbers(DomainModel):
    order_id: str | None = Field(default=None, max_length=120)
    po_number: str | None = Field(default=None, max_length=120)
    bol_number: str | None = Field(default=None, max_length=120)


class TrackingDetail(DomainModel):
    shipment_id: str
    scac: str = Field(min_length=2, max_length=10)
    current_status: str = Field(min_length=1, max_length=160)
    status_reason: str | None = Field(default=None, max_length=300)
    last_event_time: AwareDatetime
    promised_delivery_date: date | None = None
    estimated_delivery_date: date | None = None
    origin: TrackingLocation | None = None
    destination: TrackingLocation | None = None
    reference_numbers: ReferenceNumbers | None = None
    pieces: int | None = Field(default=None, ge=0)
    weight_lbs: int | float | None = Field(default=None, ge=0)
    scan_history: tuple[TrackingScan, ...]
    exception_notes: str | None = Field(default=None, max_length=1000)


class AttemptRecord(DomainModel):
    attempt: int = Field(ge=1)
    outcome: AttemptOutcome
    http_status: int | None = Field(default=None, ge=100, le=599)
    retryable: bool
    duration_ms: float = Field(ge=0)
    next_delay_seconds: float | None = Field(default=None, ge=0)
    detail: str = Field(min_length=1, max_length=240)


class EnrichmentResult(DomainModel):
    status: EnrichmentStatus
    data_completeness: DataCompleteness
    detail: TrackingDetail | None = None
    attempts: tuple[AttemptRecord, ...]
    failure_reason: EnrichmentFailureReason | None = None
    data_quality_issues: tuple[DataQualityIssue, ...] = ()

    @model_validator(mode="after")
    def validate_consistency(self) -> "EnrichmentResult":
        if self.status is EnrichmentStatus.VALID:
            if self.detail is None or self.data_completeness is not DataCompleteness.ENRICHED:
                raise ValueError("valid enrichment requires trusted detail")
            if self.failure_reason is not None:
                raise ValueError("valid enrichment cannot have a failure reason")
        elif self.detail is not None or self.data_completeness is not DataCompleteness.FEED_ONLY:
            raise ValueError("failed enrichment must remain feed-only")
        elif self.failure_reason is None:
            raise ValueError("failed enrichment requires a failure reason")
        return self


__all__ = [
    "AttemptOutcome",
    "AttemptRecord",
    "DataCompleteness",
    "EnrichmentFailureReason",
    "EnrichmentResult",
    "EnrichmentStatus",
    "ReferenceNumbers",
    "TrackingDetail",
    "TrackingLocation",
    "TrackingScan",
]
