"""Auditable per-shipment decisions and bounded run summaries."""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from shipment_triage.domain.classification import ClassificationResult
from shipment_triage.domain.enrichment import EnrichmentResult
from shipment_triage.domain.escalation import EdiControlNumbers
from shipment_triage.domain.models import DomainModel
from shipment_triage.domain.policy import FinalDisposition, PolicyDecision
from shipment_triage.domain.timelines import ShipmentTimeline
from shipment_triage.domain.triggers import TriggerEvaluation


class AsOfSource(StrEnum):
    EXPLICIT = "explicit"
    FEED_MAX = "feed_max"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    DEGRADED = "degraded"


class EscalationArtifactStatus(StrEnum):
    CREATED = "CREATED"
    REUSED = "REUSED"
    REGENERATED = "REGENERATED"
    NOT_CREATED = "NOT_CREATED"


class StoredArtifactStatus(StrEnum):
    MATCHED = "MATCHED"
    RESTORED = "RESTORED"
    CONFLICT = "CONFLICT"


class StoredArtifactResult(DomainModel):
    status: StoredArtifactStatus
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EscalationRecord(DomainModel):
    status: EscalationArtifactStatus
    decision_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    artifact_path: str | None = Field(default=None, min_length=1, max_length=500)
    artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    controls: EdiControlNumbers | None = None
    failure_code: str | None = Field(default=None, min_length=1, max_length=80)
    failure_message: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_shape(self) -> "EscalationRecord":
        created = self.status is not EscalationArtifactStatus.NOT_CREATED
        has_artifact = all(
            value is not None
            for value in (self.decision_key, self.artifact_path, self.artifact_hash, self.controls)
        )
        if created != has_artifact:
            raise ValueError("created escalation records require complete artifact metadata")
        has_failure = self.failure_code is not None and self.failure_message is not None
        if (not created) != has_failure:
            raise ValueError("non-created escalation records require a safe failure reason")
        return self


class DecisionMetadata(DomainModel):
    schema_version: str = "1"
    run_id: str = Field(pattern=r"^\d{8}T\d{6}Z-[0-9a-f]{8}-[0-9a-f]{6}$")
    run_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_of: AwareDatetime
    as_of_source: AsOfSource
    provider: str
    model: str | None
    trigger_policy_version: str
    classification_policy_version: str
    escalation_policy_version: str
    edi_profile_version: str


class ShipmentDecision(DomainModel):
    metadata: DecisionMetadata
    shipment_id: str
    artifact_key: str = Field(pattern=r"^[0-9a-f]{20}$")
    timeline: ShipmentTimeline
    trigger: TriggerEvaluation
    selected: bool
    skip_reason: str | None = Field(default=None, max_length=80)
    enrichment: EnrichmentResult | None = None
    evidence_path: str | None = Field(default=None, max_length=500)
    classification: ClassificationResult | None = None
    policy: PolicyDecision
    escalation: EscalationRecord | None = None

    @model_validator(mode="after")
    def validate_stage_shape(self) -> "ShipmentDecision":
        if (
            self.shipment_id != self.timeline.shipment_id
            or self.shipment_id != self.trigger.shipment_id
        ):
            raise ValueError("decision inputs must describe the same shipment")
        if self.selected:
            if any(
                value is None
                for value in (
                    self.enrichment,
                    self.evidence_path,
                    self.classification,
                )
            ):
                raise ValueError("selected shipment is missing a completed pipeline stage")
            if self.skip_reason is not None:
                raise ValueError("selected shipment cannot have a skip reason")
        else:
            if self.skip_reason != "NOT_FLAGGED":
                raise ValueError("unselected shipment must explain that it was not flagged")
            if any(
                value is not None
                for value in (
                    self.enrichment,
                    self.evidence_path,
                    self.classification,
                    self.escalation,
                )
            ):
                raise ValueError("unselected shipment cannot contain downstream stage output")
            if self.policy.final_disposition is not FinalDisposition.NO_ACTION:
                raise ValueError("unselected shipment must have a no-action policy")
        return self


class RunArtifactPaths(DomainModel):
    decisions: str
    rejected_records: str
    report: str
    summary: str


class RunSummary(DomainModel):
    schema_version: str = "1"
    run_id: str = Field(pattern=r"^\d{8}T\d{6}Z-[0-9a-f]{8}-[0-9a-f]{6}$")
    run_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RunStatus
    as_of: AwareDatetime
    as_of_source: AsOfSource
    raw_records: int = Field(ge=0)
    canonical_events: int = Field(ge=0)
    shipments: int = Field(ge=0)
    flagged: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    enriched: int = Field(ge=0)
    feed_only: int = Field(ge=0)
    provider_classifications: int = Field(ge=0)
    fallback_classifications: int = Field(ge=0)
    provider_input_tokens: int = Field(ge=0)
    provider_output_tokens: int = Field(ge=0)
    edi_created: int = Field(ge=0)
    edi_reused: int = Field(ge=0)
    manual_review: int = Field(ge=0)
    degraded_reasons: tuple[str, ...]
    artifacts: RunArtifactPaths


class TriageRunResult(DomainModel):
    summary: RunSummary
    decisions: tuple[ShipmentDecision, ...]


__all__ = [
    "AsOfSource",
    "DecisionMetadata",
    "EscalationArtifactStatus",
    "EscalationRecord",
    "RunArtifactPaths",
    "RunStatus",
    "RunSummary",
    "ShipmentDecision",
    "StoredArtifactResult",
    "StoredArtifactStatus",
    "TriageRunResult",
]
