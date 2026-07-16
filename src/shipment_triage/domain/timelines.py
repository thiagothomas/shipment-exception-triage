"""Shipment timeline construction and terminal-state policy."""

from collections import defaultdict
from collections.abc import Iterable
from datetime import date
from enum import StrEnum

from shipment_triage.domain.models import DataQualityIssue, DomainModel, NormalizedEvent
from shipment_triage.domain.statuses import CanonicalStatus


class TerminalState(StrEnum):
    ACTIVE = "ACTIVE"
    DELIVERED = "DELIVERED"
    CONFLICTED = "CONFLICTED"


class ShipmentTimeline(DomainModel):
    shipment_id: str
    carrier: str
    events: tuple[NormalizedEvent, ...]
    promised_date: date | None
    terminal_state: TerminalState
    data_quality_issues: tuple[DataQualityIssue, ...] = ()


def _terminal_state(events: tuple[NormalizedEvent, ...]) -> TerminalState:
    delivery_times = [
        event.occurred_at for event in events if event.status is CanonicalStatus.DELIVERED
    ]
    if not delivery_times:
        return TerminalState.ACTIVE

    non_delivery_times = [
        event.occurred_at for event in events if event.status is not CanonicalStatus.DELIVERED
    ]
    latest_time = max(event.occurred_at for event in events)
    latest_statuses = {event.status for event in events if event.occurred_at == latest_time}

    if CanonicalStatus.DELIVERED in latest_statuses and len(latest_statuses) > 1:
        return TerminalState.CONFLICTED
    if non_delivery_times and max(non_delivery_times) > max(delivery_times):
        return TerminalState.CONFLICTED
    if latest_statuses == {CanonicalStatus.DELIVERED}:
        return TerminalState.DELIVERED
    return TerminalState.ACTIVE


def _latest_promised_date(events: tuple[NormalizedEvent, ...]) -> date | None:
    dated = [event for event in events if event.promised_date is not None]
    if not dated:
        return None
    latest = max(dated, key=lambda event: (event.occurred_at, event.event_id))
    return latest.promised_date


def build_timelines(events: Iterable[NormalizedEvent]) -> tuple[ShipmentTimeline, ...]:
    """Group normalized events and derive terminal state without file-order tie breaking."""

    grouped: dict[str, list[NormalizedEvent]] = defaultdict(list)
    for event in events:
        grouped[event.shipment_id].append(event)

    timelines: list[ShipmentTimeline] = []
    for shipment_id, shipment_events in grouped.items():
        ordered = tuple(
            sorted(
                shipment_events,
                key=lambda event: (event.occurred_at, event.carrier, event.event_id),
            )
        )
        carriers = sorted({event.carrier for event in ordered})
        issues: list[DataQualityIssue] = []
        if len(carriers) > 1:
            issues.append(
                DataQualityIssue(
                    code="SHIPMENT_CARRIER_CONFLICT",
                    message="Shipment appears under more than one carrier.",
                    record_refs=tuple(ref for event in ordered for ref in event.provenance),
                )
            )
        terminal_state = _terminal_state(ordered)
        if terminal_state is TerminalState.CONFLICTED:
            issues.append(
                DataQualityIssue(
                    code="TERMINAL_STATUS_CONFLICT",
                    message="Delivered and later or same-time non-delivered states conflict.",
                    record_refs=tuple(ref for event in ordered for ref in event.provenance),
                )
            )
        timelines.append(
            ShipmentTimeline(
                shipment_id=shipment_id,
                carrier=carriers[0],
                events=ordered,
                promised_date=_latest_promised_date(ordered),
                terminal_state=terminal_state,
                data_quality_issues=tuple(issues),
            )
        )

    return tuple(sorted(timelines, key=lambda timeline: timeline.shipment_id))


__all__ = ["ShipmentTimeline", "TerminalState", "build_timelines"]
