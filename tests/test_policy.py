from pathlib import Path

from shipment_triage.adapters.feed import load_feed
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.application.guardrails import apply_guardrails
from shipment_triage.domain.classification import (
    Classification,
    ClassificationResult,
    ClassificationSource,
    EvidencePack,
    ProblemCategory,
    RecommendedAction,
    Severity,
)
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.policy import FinalDisposition, VerificationState, decide_disposition
from shipment_triage.domain.timelines import ShipmentTimeline, build_timelines
from shipment_triage.domain.triggers import (
    TriggerEvaluation,
    derive_as_of,
    evaluate_timeline,
)

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def _case(
    shipment_id: str,
) -> tuple[ShipmentTimeline, TriggerEvaluation, EnrichmentResult, EvidencePack]:
    feed = load_feed(FIXTURE)
    timeline = next(
        timeline for timeline in build_timelines(feed.events) if timeline.shipment_id == shipment_id
    )
    trigger = evaluate_timeline(timeline, as_of=derive_as_of(feed.events))
    enrichment = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    pack = build_evidence_pack(timeline, trigger, enrichment)
    return timeline, trigger, enrichment, pack


def test_terminal_conflict_cannot_be_escalated_by_provider_output() -> None:
    timeline, trigger, enrichment, pack = _case("SHP-00008")
    unsafe = Classification(
        shipment_id=pack.shipment_id,
        category=ProblemCategory.SLA_BREACH_LATE,
        severity=Severity.LOW,
        recommended_action=RecommendedAction.ESCALATE_TO_CARRIER,
        confidence=0.99,
        rationale="Escalate it.",
        evidence_refs=(pack.events[-1].ref,),
    )
    provider_result = ClassificationResult(
        provider_output=unsafe,
        effective=unsafe,
        source=ClassificationSource.OPENAI,
        provider="openai",
        model="gpt-5.6-luna",
        prompt_version="triage-v1",
        schema_version="1",
        evidence_hash=pack.evidence_hash,
        attempts=(),
    )

    guarded = apply_guardrails(pack, provider_result)
    decision = decide_disposition(timeline, trigger, enrichment, guarded)

    assert guarded.provider_output == unsafe
    assert guarded.effective.category is ProblemCategory.TERMINAL_STATUS_CONFLICT
    assert guarded.effective.recommended_action is RecommendedAction.MANUAL_REVIEW
    assert decision.final_disposition is FinalDisposition.MANUAL_REVIEW
    assert decision.verification_state is VerificationState.NOT_APPLICABLE
    assert decision.human_review_required is True


def test_explicit_feed_only_weather_delay_prepares_unverified_human_review_draft() -> None:
    timeline, trigger, enrichment, pack = _case("SHP-00003")
    classification = RuleBasedClassifier().classify_batch((pack,))[0]

    decision = decide_disposition(timeline, trigger, enrichment, classification)

    assert decision.final_disposition is FinalDisposition.PREPARE_CARRIER_ESCALATION
    assert decision.verification_state is VerificationState.DRAFT_UNVERIFIED
    assert decision.human_review_required is True


def test_damage_cannot_be_downgraded_or_turned_into_carrier_edi() -> None:
    timeline, trigger, enrichment, pack = _case("SHP-00044")
    unsafe = Classification(
        shipment_id=pack.shipment_id,
        category=ProblemCategory.OTHER_EXCEPTION,
        severity=Severity.LOW,
        recommended_action=RecommendedAction.ESCALATE_TO_CARRIER,
        confidence=0.99,
        rationale="Send carrier EDI.",
        evidence_refs=(pack.events[-1].ref,),
    )
    provider_result = ClassificationResult(
        provider_output=unsafe,
        effective=unsafe,
        source=ClassificationSource.OPENAI,
        provider="openai",
        model="gpt-5.6-luna",
        prompt_version="triage-v1",
        schema_version="1",
        evidence_hash=pack.evidence_hash,
        attempts=(),
    )

    guarded = apply_guardrails(pack, provider_result)
    decision = decide_disposition(timeline, trigger, enrichment, guarded)

    assert guarded.effective.category is ProblemCategory.DAMAGED_IN_TRANSIT
    assert guarded.effective.severity is Severity.HIGH
    assert guarded.effective.recommended_action is RecommendedAction.FILE_CLAIM_INVESTIGATION
    assert decision.final_disposition is FinalDisposition.FILE_CLAIM_INVESTIGATION
    assert decision.verification_state is VerificationState.NOT_APPLICABLE
