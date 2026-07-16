"""Pure, deterministic shipment-attention trigger policy."""

from collections.abc import Iterable
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum

from pydantic import AwareDatetime

from shipment_triage.domain.models import DomainModel, NormalizedEvent
from shipment_triage.domain.statuses import EXCEPTION_STATUSES, CanonicalStatus
from shipment_triage.domain.timelines import ShipmentTimeline, TerminalState


class TriggerRule(StrEnum):
    EXCEPTION_STATUS = "EXCEPTION_STATUS"
    PAST_PROMISE = "PAST_PROMISE"
    STALLED = "STALLED"
    TERMINAL_STATUS_CONFLICT = "TERMINAL_STATUS_CONFLICT"
    UNKNOWN_STATUS = "UNKNOWN_STATUS"


class TriggerFact(DomainModel):
    rule: TriggerRule
    matched: bool
    observed: str
    rationale: str


class TriggerEvaluation(DomainModel):
    shipment_id: str
    as_of: AwareDatetime
    facts: tuple[TriggerFact, ...]

    @property
    def flagged(self) -> bool:
        return any(fact.matched for fact in self.facts)

    @property
    def matched_rules(self) -> tuple[TriggerRule, ...]:
        return tuple(fact.rule for fact in self.facts if fact.matched)


def derive_as_of(events: Iterable[NormalizedEvent]) -> datetime:
    """Derive the deterministic fixture clock from the latest normalized event."""

    timestamps = [event.occurred_at for event in events]
    if not timestamps:
        raise ValueError("cannot derive as_of from an empty feed")
    return max(timestamps)


def _fact(rule: TriggerRule, matched: bool, observed: str, rationale: str) -> TriggerFact:
    return TriggerFact(rule=rule, matched=matched, observed=observed, rationale=rationale)


def evaluate_timeline(
    timeline: ShipmentTimeline,
    *,
    as_of: datetime,
    stall_after: timedelta = timedelta(hours=48),
) -> TriggerEvaluation:
    """Evaluate every rule and retain concrete observed values, including false results."""

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if stall_after <= timedelta(0):
        raise ValueError("stall_after must be positive")

    cleanly_delivered = timeline.terminal_state is TerminalState.DELIVERED
    exception_events = [event for event in timeline.events if event.status in EXCEPTION_STATUSES]
    exception_candidate = bool(exception_events)

    promise_deadline = (
        datetime.combine(timeline.promised_date, time.max, tzinfo=UTC)
        if timeline.promised_date is not None
        else None
    )
    past_promise_candidate = promise_deadline is not None and as_of > promise_deadline

    latest_time = max(event.occurred_at for event in timeline.events)
    idle = as_of - latest_time
    stalled_candidate = idle >= stall_after
    idle_hours = idle.total_seconds() / 3600
    threshold_hours = stall_after.total_seconds() / 3600

    latest_events = tuple(event for event in timeline.events if event.occurred_at == latest_time)
    unknown_candidate = any(event.status is CanonicalStatus.UNKNOWN for event in latest_events)
    terminal_conflict = timeline.terminal_state is TerminalState.CONFLICTED

    eligible = not cleanly_delivered
    facts = (
        _fact(
            TriggerRule.EXCEPTION_STATUS,
            eligible and exception_candidate,
            f"exception_events={len(exception_events)}; terminal_state={timeline.terminal_state}",
            "Explicit carrier exception states require attention unless a later clean delivery "
            "resolves the shipment.",
        ),
        _fact(
            TriggerRule.PAST_PROMISE,
            eligible and past_promise_candidate,
            f"promise={timeline.promised_date}; as_of={as_of.isoformat()}",
            "The configured observation time is after promise end-of-day.",
        ),
        _fact(
            TriggerRule.STALLED,
            eligible and stalled_candidate,
            f"idle_hours={idle_hours:.3f}; threshold_hours={threshold_hours:.3f}",
            "The latest carrier event is older than the configured stall threshold.",
        ),
        _fact(
            TriggerRule.TERMINAL_STATUS_CONFLICT,
            terminal_conflict,
            f"terminal_state={timeline.terminal_state}",
            "Delivered and non-delivered states conflict at or after the latest delivery.",
        ),
        _fact(
            TriggerRule.UNKNOWN_STATUS,
            eligible and unknown_candidate,
            "latest_raw_statuses=" + ",".join(sorted(event.raw_status for event in latest_events)),
            "An unmapped latest carrier status requires human interpretation.",
        ),
    )
    return TriggerEvaluation(shipment_id=timeline.shipment_id, as_of=as_of, facts=facts)


def evaluate_timelines(
    timelines: Iterable[ShipmentTimeline],
    *,
    as_of: datetime,
    stall_after: timedelta = timedelta(hours=48),
) -> tuple[TriggerEvaluation, ...]:
    return tuple(
        evaluate_timeline(timeline, as_of=as_of, stall_after=stall_after) for timeline in timelines
    )


__all__ = [
    "TriggerEvaluation",
    "TriggerFact",
    "TriggerRule",
    "derive_as_of",
    "evaluate_timeline",
    "evaluate_timelines",
]
