"""Code-owned final operational disposition policy."""

from enum import StrEnum

from pydantic import Field, model_validator

from shipment_triage.domain.classification import (
    ClassificationResult,
    ProblemCategory,
    RecommendedAction,
)
from shipment_triage.domain.enrichment import EnrichmentResult, EnrichmentStatus
from shipment_triage.domain.models import DomainModel
from shipment_triage.domain.statuses import EXCEPTION_STATUSES
from shipment_triage.domain.timelines import ShipmentTimeline, TerminalState
from shipment_triage.domain.triggers import TriggerEvaluation, TriggerRule


class FinalDisposition(StrEnum):
    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    CONTACT_CONSIGNEE = "CONTACT_CONSIGNEE"
    FILE_CLAIM_INVESTIGATION = "FILE_CLAIM_INVESTIGATION"
    PREPARE_CARRIER_ESCALATION = "PREPARE_CARRIER_ESCALATION"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class VerificationState(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    READY_FOR_HUMAN_REVIEW = "READY_FOR_HUMAN_REVIEW"
    DRAFT_UNVERIFIED = "DRAFT_UNVERIFIED"


class PolicyOverride(DomainModel):
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=300)


class PolicyDecision(DomainModel):
    requested_disposition: FinalDisposition
    final_disposition: FinalDisposition
    human_review_required: bool
    verification_state: VerificationState
    overrides: tuple[PolicyOverride, ...] = ()

    @model_validator(mode="after")
    def validate_draft_state(self) -> "PolicyDecision":
        prepares_draft = self.final_disposition is FinalDisposition.PREPARE_CARRIER_ESCALATION
        if prepares_draft and self.verification_state is VerificationState.NOT_APPLICABLE:
            raise ValueError("carrier escalation requires a draft verification state")
        if not prepares_draft and self.verification_state is not VerificationState.NOT_APPLICABLE:
            raise ValueError("non-EDI disposition cannot have a draft verification state")
        if prepares_draft and not self.human_review_required:
            raise ValueError("every EDI draft requires human review")
        return self


_ACTION_TO_DISPOSITION = {
    RecommendedAction.ESCALATE_TO_CARRIER: FinalDisposition.PREPARE_CARRIER_ESCALATION,
    RecommendedAction.CONTACT_CONSIGNEE: FinalDisposition.CONTACT_CONSIGNEE,
    RecommendedAction.FILE_CLAIM_INVESTIGATION: FinalDisposition.FILE_CLAIM_INVESTIGATION,
    RecommendedAction.MONITOR_24H: FinalDisposition.MONITOR,
    RecommendedAction.MANUAL_REVIEW: FinalDisposition.MANUAL_REVIEW,
}


def _matched(trigger: TriggerEvaluation, rule: TriggerRule) -> bool:
    return any(fact.rule is rule and fact.matched for fact in trigger.facts)


def decide_disposition(
    timeline: ShipmentTimeline,
    trigger: TriggerEvaluation,
    enrichment: EnrichmentResult,
    classification: ClassificationResult,
) -> PolicyDecision:
    """Translate deterministic facts and bounded model advice into an operator action."""

    if not (timeline.shipment_id == trigger.shipment_id == classification.effective.shipment_id):
        raise ValueError("policy inputs must describe the same shipment")

    effective = classification.effective
    requested = _ACTION_TO_DISPOSITION[effective.recommended_action]
    past_promise = _matched(trigger, TriggerRule.PAST_PROMISE)
    stalled = _matched(trigger, TriggerRule.STALLED)
    unknown = _matched(trigger, TriggerRule.UNKNOWN_STATUS)
    latest_event_time = max(event.occurred_at for event in timeline.events)
    idle_hours = (trigger.as_of - latest_event_time).total_seconds() / 3600

    if timeline.terminal_state is TerminalState.CONFLICTED or unknown:
        final = FinalDisposition.MANUAL_REVIEW
    elif effective.category is ProblemCategory.DAMAGED_IN_TRANSIT:
        final = (
            FinalDisposition.MANUAL_REVIEW
            if effective.recommended_action is RecommendedAction.MANUAL_REVIEW
            else FinalDisposition.FILE_CLAIM_INVESTIGATION
        )
    elif effective.category is ProblemCategory.HELD_CONSIGNEE_UNAVAILABLE:
        final = FinalDisposition.CONTACT_CONSIGNEE
    elif effective.category is ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT:
        final = (
            FinalDisposition.CONTACT_CONSIGNEE
            if effective.recommended_action is RecommendedAction.CONTACT_CONSIGNEE
            else FinalDisposition.PREPARE_CARRIER_ESCALATION
        )
    elif (
        (
            effective.category
            in {
                ProblemCategory.CARRIER_DELAY_WEATHER,
                ProblemCategory.CARRIER_DELAY_MECHANICAL,
            }
            and past_promise
        )
        or (stalled and (past_promise or idle_hours >= 96))
        or past_promise
    ):
        final = FinalDisposition.PREPARE_CARRIER_ESCALATION
    else:
        latest_status = timeline.events[-1].status
        recovered = (
            _matched(trigger, TriggerRule.EXCEPTION_STATUS)
            and latest_status not in EXCEPTION_STATUSES
        )
        if recovered:
            final = FinalDisposition.MONITOR
        elif requested in {
            FinalDisposition.MONITOR,
            FinalDisposition.CONTACT_CONSIGNEE,
            FinalDisposition.MANUAL_REVIEW,
        }:
            final = requested
        else:
            final = FinalDisposition.MANUAL_REVIEW

    overrides: tuple[PolicyOverride, ...] = ()
    if final is not requested:
        overrides = (
            PolicyOverride(
                code="MODEL_ACTION_OVERRIDDEN",
                message=(
                    "Deterministic policy replaced a recommendation outside its allowed envelope."
                ),
            ),
        )
    prepares = final is FinalDisposition.PREPARE_CARRIER_ESCALATION
    verification_state = VerificationState.NOT_APPLICABLE
    if prepares:
        verification_state = (
            VerificationState.READY_FOR_HUMAN_REVIEW
            if enrichment.status is EnrichmentStatus.VALID
            else VerificationState.DRAFT_UNVERIFIED
        )
    human_review = final not in {FinalDisposition.NO_ACTION, FinalDisposition.MONITOR}
    return PolicyDecision(
        requested_disposition=requested,
        final_disposition=final,
        human_review_required=human_review,
        verification_state=verification_state,
        overrides=overrides,
    )


__all__ = [
    "FinalDisposition",
    "PolicyDecision",
    "PolicyOverride",
    "VerificationState",
    "decide_disposition",
]
