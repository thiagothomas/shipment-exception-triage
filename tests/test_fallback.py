from pathlib import Path

from shipment_triage.adapters.feed import load_feed
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.fallback import RuleBasedClassifier
from shipment_triage.domain.classification import (
    ClassificationSource,
    EvidencePack,
    EvidenceTracking,
    ProblemCategory,
    RecommendedAction,
)
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.timelines import build_timelines
from shipment_triage.domain.triggers import derive_as_of, evaluate_timeline

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def _feed_only_pack(shipment_id: str) -> EvidencePack:
    feed = load_feed(FIXTURE)
    timeline = next(
        item for item in build_timelines(feed.events) if item.shipment_id == shipment_id
    )
    trigger = evaluate_timeline(timeline, as_of=derive_as_of(feed.events))
    feed_only = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    return build_evidence_pack(timeline, trigger, feed_only)


def test_rule_fallback_classifies_weather_delay_with_grounded_references() -> None:
    feed = load_feed(FIXTURE)
    timeline = next(
        timeline for timeline in build_timelines(feed.events) if timeline.shipment_id == "SHP-00019"
    )
    trigger = evaluate_timeline(timeline, as_of=derive_as_of(feed.events))
    feed_only = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    pack = build_evidence_pack(timeline, trigger, feed_only)

    result = RuleBasedClassifier().classify_batch((pack,))[0]

    assert result.source is ClassificationSource.FALLBACK_RULES
    assert result.effective.category is ProblemCategory.CARRIER_DELAY_WEATHER
    assert result.effective.recommended_action is RecommendedAction.ESCALATE_TO_CARRIER
    assert set(result.effective.evidence_refs) <= set(pack.allowed_evidence_refs)


def test_tracking_only_cause_uses_the_exact_tracking_reference() -> None:
    pack = _feed_only_pack("SHP-00019")
    neutral_events = tuple(
        event.model_copy(update={"raw_status": "DELAYED", "description": None})
        for event in pack.events
    )
    pack = pack.model_copy(
        update={
            "events": neutral_events,
            "tracking": EvidenceTracking(
                current_status="DELAYED",
                status_reason="WEATHER",
                last_event_time=pack.events[-1].occurred_at,
                scans=(),
            ),
            "allowed_evidence_refs": (
                *pack.allowed_evidence_refs,
                "tracking:current_status",
                "tracking:status_reason",
            ),
        }
    )

    result = RuleBasedClassifier().classify_batch((pack,))[0]

    assert result.effective.category is ProblemCategory.CARRIER_DELAY_WEATHER
    assert result.effective.evidence_refs == ("tracking:status_reason",)


def test_terms_from_different_events_do_not_create_a_false_match() -> None:
    pack = _feed_only_pack("SHP-00003")
    events = tuple(
        event.model_copy(
            update={
                "raw_status": ("MISSED SCAN" if index == 0 else "APPOINTMENT SCHEDULED"),
                "description": None,
            }
        )
        for index, event in enumerate(pack.events)
    )
    pack = pack.model_copy(update={"events": events})

    result = RuleBasedClassifier().classify_batch((pack,))[0]

    assert result.effective.category is not ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT
    assert set(result.effective.evidence_refs) <= set(pack.allowed_evidence_refs)


def test_rule_fallback_produces_grounded_output_for_every_flagged_shipment() -> None:
    feed = load_feed(FIXTURE)
    as_of = derive_as_of(feed.events)
    feed_only = EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.SERVER_ERROR,
    )
    cases = [
        (timeline, evaluate_timeline(timeline, as_of=as_of))
        for timeline in build_timelines(feed.events)
    ]
    packs = tuple(
        build_evidence_pack(timeline, trigger, feed_only)
        for timeline, trigger in cases
        if trigger.flagged
    )

    results = RuleBasedClassifier().classify_batch(packs)

    assert len(results) == 52
    assert all(
        set(result.effective.evidence_refs) <= set(pack.allowed_evidence_refs)
        for pack, result in zip(packs, results, strict=True)
    )
    assert {result.effective.category for result in results} >= {
        ProblemCategory.CARRIER_DELAY_MECHANICAL,
        ProblemCategory.CARRIER_DELAY_WEATHER,
        ProblemCategory.DAMAGED_IN_TRANSIT,
        ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT,
        ProblemCategory.HELD_CONSIGNEE_UNAVAILABLE,
        ProblemCategory.SLA_BREACH_LATE,
        ProblemCategory.STALLED_NO_SCANS,
        ProblemCategory.TERMINAL_STATUS_CONFLICT,
    }
