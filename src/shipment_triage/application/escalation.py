"""Build profile-neutral escalation facts from trusted pipeline decisions."""

from collections.abc import Callable

from shipment_triage.domain.classification import ClassificationResult, ProblemCategory
from shipment_triage.domain.enrichment import EnrichmentResult, EnrichmentStatus
from shipment_triage.domain.escalation import EscalationCause, EscalationDraft
from shipment_triage.domain.models import NormalizedEvent
from shipment_triage.domain.policy import FinalDisposition, PolicyDecision
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.timelines import ShipmentTimeline, TerminalState
from shipment_triage.domain.triggers import TriggerEvaluation, TriggerRule


class UnrepresentableEscalation(ValueError):
    """Trusted facts cannot produce a truthful draft for the selected profile."""


_MOVEMENT_STATUSES = frozenset(
    {
        CanonicalStatus.PICKED_UP,
        CanonicalStatus.IN_TRANSIT,
        CanonicalStatus.ARRIVED_FACILITY,
        CanonicalStatus.DEPARTED_FACILITY,
        CanonicalStatus.OUT_FOR_DELIVERY,
    }
)
_CAUSE_BY_CATEGORY = {
    ProblemCategory.CARRIER_DELAY_WEATHER: EscalationCause.WEATHER,
    ProblemCategory.CARRIER_DELAY_MECHANICAL: EscalationCause.MECHANICAL,
    ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT: EscalationCause.MISSED_APPOINTMENT,
    ProblemCategory.STALLED_NO_SCANS: EscalationCause.NONE_REPORTED,
    ProblemCategory.SLA_BREACH_LATE: EscalationCause.NONE_REPORTED,
}
_TRIGGER_BY_CATEGORY = {
    ProblemCategory.CARRIER_DELAY_WEATHER: TriggerRule.EXCEPTION_STATUS,
    ProblemCategory.CARRIER_DELAY_MECHANICAL: TriggerRule.EXCEPTION_STATUS,
    ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT: TriggerRule.EXCEPTION_STATUS,
    ProblemCategory.STALLED_NO_SCANS: TriggerRule.STALLED,
    ProblemCategory.SLA_BREACH_LATE: TriggerRule.PAST_PROMISE,
}


def _event_text(event: NormalizedEvent) -> str:
    return f"{event.raw_status} {event.description or ''}".lower()


def _latest_matching(
    timeline: ShipmentTimeline,
    predicate: Callable[[NormalizedEvent], bool],
) -> NormalizedEvent | None:
    matches = [event for event in timeline.events if predicate(event)]
    return max(matches, key=lambda event: (event.occurred_at, event.event_id), default=None)


def _cause_event(
    timeline: ShipmentTimeline,
    category: ProblemCategory,
) -> NormalizedEvent | None:
    if category is ProblemCategory.CARRIER_DELAY_WEATHER:
        return _latest_matching(timeline, lambda event: "weather" in _event_text(event))
    if category is ProblemCategory.CARRIER_DELAY_MECHANICAL:
        return _latest_matching(timeline, lambda event: "mechanical" in _event_text(event))
    if category is ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT:
        return _latest_matching(
            timeline,
            lambda event: event.status is CanonicalStatus.MISSED_APPOINTMENT,
        )
    return None


def _matched(trigger: TriggerEvaluation, rule: TriggerRule) -> bool:
    return any(fact.rule is rule and fact.matched for fact in trigger.facts)


def build_escalation_draft(
    timeline: ShipmentTimeline,
    trigger: TriggerEvaluation,
    enrichment: EnrichmentResult,
    classification: ClassificationResult,
    decision: PolicyDecision,
) -> EscalationDraft:
    """Select truthful status/time/location facts for a human-review EDI draft."""

    shipment_ids = {
        timeline.shipment_id,
        trigger.shipment_id,
        classification.effective.shipment_id,
    }
    if len(shipment_ids) != 1:
        raise ValueError("escalation inputs must describe the same shipment")
    if decision.final_disposition is not FinalDisposition.PREPARE_CARRIER_ESCALATION:
        raise ValueError("only carrier-escalation decisions can produce an EDI draft")
    if not decision.human_review_required:
        raise ValueError("every EDI draft must require human review")
    if timeline.terminal_state is not TerminalState.ACTIVE:
        raise UnrepresentableEscalation("only active, non-conflicted timelines can produce EDI")

    category = classification.effective.category
    cause = _CAUSE_BY_CATEGORY.get(category)
    preferred_trigger = _TRIGGER_BY_CATEGORY.get(category)
    if cause is None or preferred_trigger is None:
        raise UnrepresentableEscalation("classification category has no exercise EDI mapping")
    if not _matched(trigger, preferred_trigger):
        matched_rules = trigger.matched_rules
        if not matched_rules:
            raise UnrepresentableEscalation("EDI draft requires a matched deterministic trigger")
        preferred_trigger = matched_rules[0]

    cause_event = _cause_event(timeline, category)
    if cause is EscalationCause.MISSED_APPOINTMENT:
        if cause_event is None:
            raise UnrepresentableEscalation("missed appointment lacks a truthful AT7 event")
        actual_event = cause_event
        event_at = cause_event.occurred_at
    elif cause in {EscalationCause.WEATHER, EscalationCause.MECHANICAL}:
        if cause_event is None:
            raise UnrepresentableEscalation("delay category lacks explicit carrier cause evidence")
        movement_events = [
            event
            for event in timeline.events
            if event.status in _MOVEMENT_STATUSES and event.occurred_at <= cause_event.occurred_at
        ]
        if not movement_events:
            raise UnrepresentableEscalation("delay has no preceding mappable movement state")
        actual_event = max(
            movement_events,
            key=lambda event: (event.occurred_at, event.event_id),
        )
        event_at = cause_event.occurred_at
    else:
        movement_events = [event for event in timeline.events if event.status in _MOVEMENT_STATUSES]
        if not movement_events:
            raise UnrepresentableEscalation("derived escalation has no mappable movement state")
        actual_event = max(
            movement_events,
            key=lambda event: (event.occurred_at, event.event_id),
        )
        event_at = actual_event.occurred_at

    location_event = _latest_matching(
        timeline,
        lambda event: event.occurred_at <= event_at and event.location is not None,
    )
    city = location_event.location.city if location_event and location_event.location else None
    state = location_event.location.state if location_event and location_event.location else None
    bol_number = None
    po_number = None
    if enrichment.status is EnrichmentStatus.VALID and enrichment.detail is not None:
        references = enrichment.detail.reference_numbers
        if references is not None:
            bol_number = references.bol_number
            po_number = references.po_number
        located_scans = [
            scan
            for scan in enrichment.detail.scan_history
            if scan.city is not None or scan.state is not None
        ]
        if located_scans:
            latest_scan = max(located_scans, key=lambda scan: scan.time)
            city = latest_scan.city or city
            state = latest_scan.state or state

    latest_event_at = max(event.occurred_at for event in timeline.events)
    idle_hours = max(int((trigger.as_of - latest_event_at).total_seconds() // 3600), 0)
    return EscalationDraft(
        shipment_id=timeline.shipment_id,
        carrier_shipment_id=timeline.shipment_id,
        scac=timeline.carrier,
        bol_number=bol_number,
        po_number=po_number,
        prepared_at=trigger.as_of,
        event_at=event_at,
        actual_status=actual_event.status,
        cause=cause,
        city=city,
        state=state,
        category=category,
        trigger_rule=preferred_trigger,
        idle_hours=idle_hours,
        promised_date=timeline.promised_date,
        data_completeness=enrichment.data_completeness,
        verification_state=decision.verification_state,
    )


__all__ = ["UnrepresentableEscalation", "build_escalation_draft"]
