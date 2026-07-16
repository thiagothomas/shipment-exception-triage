"""Provider-neutral evidence and classification contracts."""

from datetime import date
from enum import StrEnum

from pydantic import AwareDatetime, Field, field_validator

from shipment_triage.domain.enrichment import DataCompleteness
from shipment_triage.domain.models import DomainModel
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.timelines import TerminalState
from shipment_triage.domain.triggers import TriggerRule


class ProblemCategory(StrEnum):
    CARRIER_DELAY_WEATHER = "CARRIER_DELAY_WEATHER"
    CARRIER_DELAY_MECHANICAL = "CARRIER_DELAY_MECHANICAL"
    DELIVERY_FAILED_MISSED_APPOINTMENT = "DELIVERY_FAILED_MISSED_APPOINTMENT"
    HELD_CONSIGNEE_UNAVAILABLE = "HELD_CONSIGNEE_UNAVAILABLE"
    DAMAGED_IN_TRANSIT = "DAMAGED_IN_TRANSIT"
    STALLED_NO_SCANS = "STALLED_NO_SCANS"
    SLA_BREACH_LATE = "SLA_BREACH_LATE"
    TERMINAL_STATUS_CONFLICT = "TERMINAL_STATUS_CONFLICT"
    OTHER_EXCEPTION = "OTHER_EXCEPTION"


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


SEVERITY_ORDER = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class RecommendedAction(StrEnum):
    ESCALATE_TO_CARRIER = "ESCALATE_TO_CARRIER"
    CONTACT_CONSIGNEE = "CONTACT_CONSIGNEE"
    FILE_CLAIM_INVESTIGATION = "FILE_CLAIM_INVESTIGATION"
    MONITOR_24H = "MONITOR_24H"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ClassificationSource(StrEnum):
    OPENAI = "openai"
    FALLBACK_RULES = "fallback-rules"


class ClassificationAttemptOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    REFUSED = "REFUSED"


class EvidenceEvent(DomainModel):
    ref: str
    occurred_at: AwareDatetime
    status: CanonicalStatus
    raw_status: str
    description: str | None = None
    city: str | None = None
    state: str | None = None
    promised_date: date | None = None


class EvidenceTrigger(DomainModel):
    ref: str
    rule: TriggerRule
    matched: bool
    observed: str


class EvidenceTrackingScan(DomainModel):
    ref: str
    time: AwareDatetime
    status: str
    city: str | None = None
    state: str | None = None


class EvidenceTracking(DomainModel):
    current_status: str
    status_reason: str | None = None
    last_event_time: AwareDatetime
    promised_delivery_date: date | None = None
    estimated_delivery_date: date | None = None
    exception_notes: str | None = None
    scans: tuple[EvidenceTrackingScan, ...]


class EvidencePack(DomainModel):
    schema_version: str = "1"
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    shipment_id: str
    carrier: str
    terminal_state: TerminalState
    data_completeness: DataCompleteness
    events: tuple[EvidenceEvent, ...]
    triggers: tuple[EvidenceTrigger, ...]
    tracking: EvidenceTracking | None = None
    data_quality_issue_codes: tuple[str, ...] = ()
    allowed_evidence_refs: tuple[str, ...]


class Classification(DomainModel):
    shipment_id: str
    category: ProblemCategory
    severity: Severity
    recommended_action: RecommendedAction
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=600)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=20)

    @field_validator("evidence_refs")
    @classmethod
    def evidence_refs_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence references must be unique")
        return value


class ClassificationAttempt(DomainModel):
    attempt: int = Field(ge=1)
    outcome: ClassificationAttemptOutcome
    duration_ms: float = Field(ge=0)
    detail: str = Field(min_length=1, max_length=240)


class GuardrailOverride(DomainModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)


class ClassificationResult(DomainModel):
    provider_output: Classification | None
    effective: Classification
    source: ClassificationSource
    provider: str | None
    model: str | None
    prompt_version: str
    schema_version: str
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempts: tuple[ClassificationAttempt, ...]
    overrides: tuple[GuardrailOverride, ...] = ()


__all__ = [
    "SEVERITY_ORDER",
    "Classification",
    "ClassificationAttempt",
    "ClassificationAttemptOutcome",
    "ClassificationResult",
    "ClassificationSource",
    "EvidenceEvent",
    "EvidencePack",
    "EvidenceTracking",
    "EvidenceTrackingScan",
    "EvidenceTrigger",
    "GuardrailOverride",
    "ProblemCategory",
    "RecommendedAction",
    "Severity",
]
