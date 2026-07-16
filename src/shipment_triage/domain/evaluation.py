"""Versioned labels and machine-readable evaluation results."""

from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from shipment_triage.domain.classification import (
    Classification,
    ClassificationSource,
    ProblemCategory,
    RecommendedAction,
    Severity,
)
from shipment_triage.domain.policy import FinalDisposition, PolicyDecision


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvalSplit(StrEnum):
    DEV = "dev"
    TEST = "test"


class EvalLabel(EvalModel):
    case_id: str = Field(pattern=r"^(dev|test)-[a-z0-9-]+$")
    split: EvalSplit
    shipment_id: str
    carrier: str
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_category: ProblemCategory
    allowed_severities: tuple[Severity, ...] = Field(min_length=1)
    allowed_actions: tuple[RecommendedAction, ...] = Field(min_length=1)
    required_evidence_refs: tuple[str, ...] = Field(min_length=1)
    expected_disposition: FinalDisposition
    human_review_required: bool
    label_rationale: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_sets(self) -> "EvalLabel":
        if len(set(self.allowed_severities)) != len(self.allowed_severities):
            raise ValueError("allowed severities must be unique")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed actions must be unique")
        if len(set(self.required_evidence_refs)) != len(self.required_evidence_refs):
            raise ValueError("required evidence references must be unique")
        return self


class EvalDataset(EvalModel):
    version: str = Field(min_length=1, max_length=40)
    as_of: AwareDatetime
    cases: tuple[EvalLabel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self) -> "EvalDataset":
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation case IDs must be unique")
        if len({case.shipment_id for case in self.cases}) != len(self.cases):
            raise ValueError("evaluation shipment IDs must be unique")
        if {case.split for case in self.cases} != {EvalSplit.DEV, EvalSplit.TEST}:
            raise ValueError("evaluation dataset must contain dev and test cases")
        return self


class ClassificationMetrics(EvalModel):
    cases: int = Field(ge=0)
    missing_predictions: int = Field(ge=0)
    category_exact: int = Field(ge=0)
    category_accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    severity_admissible: int = Field(ge=0)
    severity_ordinal_mae: float = Field(ge=0)
    action_admissible: int = Field(ge=0)
    required_evidence_recall: float = Field(ge=0, le=1)
    invalid_evidence_references: int = Field(ge=0)
    confusion_matrix: dict[str, dict[str, int]]


class SystemMetrics(EvalModel):
    disposition_exact: int = Field(ge=0)
    human_review_exact: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    guardrail_override_count: int = Field(ge=0)
    prohibited_edi_count: int = Field(ge=0)
    policy_escape_count: int = Field(ge=0)
    effective_schema_valid: int = Field(ge=0)
    hard_gates_passed: bool
    provider_input_tokens: int = Field(ge=0)
    provider_output_tokens: int = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)


class EvalCaseResult(EvalModel):
    case_id: str
    repeat: int = Field(ge=1)
    classification_source: ClassificationSource
    raw: Classification | None
    effective: Classification
    policy: PolicyDecision


class EvalRunMetrics(EvalModel):
    repeat: int = Field(ge=1)
    raw: ClassificationMetrics
    effective: ClassificationMetrics
    system: SystemMetrics


class EvaluationReport(EvalModel):
    schema_version: str = "1"
    dataset_version: str
    split: str
    provider: str
    model: str | None
    repeats: int = Field(ge=1)
    case_results: tuple[EvalCaseResult, ...]
    runs: tuple[EvalRunMetrics, ...]
    category_consistency: float = Field(ge=0, le=1)
    action_consistency: float = Field(ge=0, le=1)
    hard_gates_passed: bool


__all__ = [
    "ClassificationMetrics",
    "EvalCaseResult",
    "EvalDataset",
    "EvalLabel",
    "EvalRunMetrics",
    "EvalSplit",
    "EvaluationReport",
    "SystemMetrics",
]
