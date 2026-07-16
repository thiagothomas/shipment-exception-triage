from datetime import UTC, datetime
from pathlib import Path

import pytest

from shipment_triage.adapters.feed import load_feed
from shipment_triage.application.escalation import (
    UnrepresentableEscalation,
    build_escalation_draft,
)
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.domain.classification import ClassificationResult
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
    ReferenceNumbers,
    TrackingDetail,
    TrackingScan,
)
from shipment_triage.domain.escalation import EscalationCause
from shipment_triage.domain.policy import (
    FinalDisposition,
    PolicyDecision,
    VerificationState,
    decide_disposition,
)
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.timelines import ShipmentTimeline, build_timelines
from shipment_triage.domain.triggers import TriggerEvaluation, derive_as_of, evaluate_timeline

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def _timeline_and_trigger(shipment_id: str) -> tuple[ShipmentTimeline, TriggerEvaluation]:
    feed = load_feed(FIXTURE)
    timeline = next(
        item for item in build_timelines(feed.events) if item.shipment_id == shipment_id
    )
    return timeline, evaluate_timeline(timeline, as_of=derive_as_of(feed.events))


def _feed_only() -> EnrichmentResult:
    return EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )


def _classification(
    timeline: ShipmentTimeline,
    trigger: TriggerEvaluation,
    enrichment: EnrichmentResult,
) -> ClassificationResult:
    pack = build_evidence_pack(timeline, trigger, enrichment)
    return RuleBasedClassifier().classify_batch((pack,))[0]


def test_feed_only_weather_draft_uses_preceding_movement_and_delay_time() -> None:
    timeline, trigger = _timeline_and_trigger("SHP-00003")
    enrichment = _feed_only()
    classification = _classification(timeline, trigger, enrichment)
    decision = decide_disposition(timeline, trigger, enrichment, classification)

    draft = build_escalation_draft(
        timeline,
        trigger,
        enrichment,
        classification,
        decision,
    )

    assert draft.actual_status is CanonicalStatus.OUT_FOR_DELIVERY
    assert draft.event_at == datetime(2026, 6, 29, 4, 0, tzinfo=UTC)
    assert draft.cause is EscalationCause.WEATHER
    assert (draft.city, draft.state) == ("Charlotte", "NC")
    assert draft.verification_state is VerificationState.DRAFT_UNVERIFIED
    assert draft.bol_number is None


def test_valid_enrichment_adds_references_and_latest_trusted_location() -> None:
    timeline, trigger = _timeline_and_trigger("SHP-00003")
    enrichment = EnrichmentResult(
        status=EnrichmentStatus.VALID,
        data_completeness=DataCompleteness.ENRICHED,
        detail=TrackingDetail(
            shipment_id=timeline.shipment_id,
            scac=timeline.carrier,
            current_status="DELAYED",
            status_reason="WEATHER",
            last_event_time=datetime(2026, 6, 29, 5, 0, tzinfo=UTC),
            reference_numbers=ReferenceNumbers(
                po_number="PO456",
                bol_number="BOL123",
            ),
            scan_history=(
                TrackingScan(
                    time=datetime(2026, 6, 29, 5, 0, tzinfo=UTC),
                    status="DELAYED",
                    city="Raleigh",
                    state="NC",
                ),
            ),
        ),
        attempts=(),
    )
    classification = _classification(timeline, trigger, enrichment)
    decision = decide_disposition(timeline, trigger, enrichment, classification)

    draft = build_escalation_draft(
        timeline,
        trigger,
        enrichment,
        classification,
        decision,
    )

    assert (draft.bol_number, draft.po_number) == ("BOL123", "PO456")
    assert (draft.city, draft.state) == ("Raleigh", "NC")
    assert draft.verification_state is VerificationState.READY_FOR_HUMAN_REVIEW


def test_delay_without_preceding_movement_becomes_unrepresentable() -> None:
    timeline, trigger = _timeline_and_trigger("SHP-00019")
    enrichment = _feed_only()
    classification = _classification(timeline, trigger, enrichment)
    forced_draft_decision = PolicyDecision(
        requested_disposition=FinalDisposition.PREPARE_CARRIER_ESCALATION,
        final_disposition=FinalDisposition.PREPARE_CARRIER_ESCALATION,
        human_review_required=True,
        verification_state=VerificationState.DRAFT_UNVERIFIED,
    )

    with pytest.raises(UnrepresentableEscalation, match="preceding mappable movement"):
        build_escalation_draft(
            timeline,
            trigger,
            enrichment,
            classification,
            forced_draft_decision,
        )
