"""Provider-neutral carrier-escalation draft and reservation contracts."""

from datetime import date
from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from shipment_triage.domain.classification import ProblemCategory
from shipment_triage.domain.enrichment import DataCompleteness
from shipment_triage.domain.models import DomainModel, ShipmentId
from shipment_triage.domain.policy import VerificationState
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.triggers import TriggerRule


class EscalationCause(StrEnum):
    NONE_REPORTED = "NONE_REPORTED"
    MISSED_APPOINTMENT = "MISSED_APPOINTMENT"
    MECHANICAL = "MECHANICAL"
    WEATHER = "WEATHER"


class EscalationDraft(DomainModel):
    schema_version: str = "1"
    shipment_id: ShipmentId
    carrier_shipment_id: str = Field(min_length=1, max_length=64)
    scac: str = Field(pattern=r"^[A-Z0-9]{2,10}$")
    bol_number: str | None = Field(default=None, min_length=1, max_length=120)
    po_number: str | None = Field(default=None, min_length=1, max_length=120)
    prepared_at: AwareDatetime
    event_at: AwareDatetime
    actual_status: CanonicalStatus
    cause: EscalationCause
    city: str | None = Field(default=None, min_length=1, max_length=120)
    state: str | None = Field(default=None, min_length=1, max_length=40)
    category: ProblemCategory
    trigger_rule: TriggerRule
    idle_hours: int | None = Field(default=None, ge=0, le=999)
    promised_date: date | None = None
    data_completeness: DataCompleteness
    verification_state: VerificationState

    @model_validator(mode="after")
    def validate_verification_state(self) -> "EscalationDraft":
        expected = (
            VerificationState.READY_FOR_HUMAN_REVIEW
            if self.data_completeness is DataCompleteness.ENRICHED
            else VerificationState.DRAFT_UNVERIFIED
        )
        if self.verification_state is not expected:
            raise ValueError("draft verification state must match data completeness")
        cause_by_category = {
            ProblemCategory.CARRIER_DELAY_WEATHER: EscalationCause.WEATHER,
            ProblemCategory.CARRIER_DELAY_MECHANICAL: EscalationCause.MECHANICAL,
            ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT: (
                EscalationCause.MISSED_APPOINTMENT
            ),
        }
        expected_cause = cause_by_category.get(self.category, EscalationCause.NONE_REPORTED)
        if self.cause is not expected_cause:
            raise ValueError("draft cause must match its primary classification category")
        return self


class EdiControlNumbers(DomainModel):
    isa: int = Field(ge=1, le=999_999_999)
    gs: int = Field(ge=1, le=999_999_999)
    st: int = Field(ge=1, le=999_999_999)


class ReservationState(StrEnum):
    PENDING = "PENDING"
    FINALIZED = "FINALIZED"


class EscalationReservation(DomainModel):
    decision_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: str = Field(min_length=1, max_length=80)
    sender_id: str = Field(min_length=1, max_length=80)
    receiver_id: str = Field(min_length=1, max_length=80)
    controls: EdiControlNumbers
    state: ReservationState
    artifact_path: str | None = Field(default=None, min_length=1, max_length=500)
    artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_finalization(self) -> "EscalationReservation":
        has_artifact = self.artifact_path is not None and self.artifact_hash is not None
        if self.state is ReservationState.FINALIZED and not has_artifact:
            raise ValueError("finalized reservation requires artifact path and hash")
        if self.state is ReservationState.PENDING and (
            self.artifact_path is not None or self.artifact_hash is not None
        ):
            raise ValueError("pending reservation cannot reference a finalized artifact")
        return self


__all__ = [
    "EdiControlNumbers",
    "EscalationCause",
    "EscalationDraft",
    "EscalationReservation",
    "ReservationState",
]
