from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from shipment_triage.adapters.feed import load_feed
from shipment_triage.domain.timelines import build_timelines
from shipment_triage.domain.triggers import (
    TriggerRule,
    derive_as_of,
    evaluate_timeline,
    evaluate_timelines,
)

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def test_fixture_trigger_counts_are_explainable_and_stable() -> None:
    feed = load_feed(FIXTURE)
    evaluations = evaluate_timelines(build_timelines(feed.events), as_of=derive_as_of(feed.events))

    counts = Counter(
        fact.rule for evaluation in evaluations for fact in evaluation.facts if fact.matched
    )

    assert len(evaluations) == 125
    assert sum(evaluation.flagged for evaluation in evaluations) == 52
    assert counts == {
        TriggerRule.EXCEPTION_STATUS: 39,
        TriggerRule.PAST_PROMISE: 37,
        TriggerRule.STALLED: 29,
        TriggerRule.TERMINAL_STATUS_CONFLICT: 6,
    }


def test_stall_boundary_is_inclusive_and_every_rule_is_recorded() -> None:
    timeline = next(
        timeline
        for timeline in build_timelines(load_feed(FIXTURE).events)
        if timeline.shipment_id == "SHP-00004"
    )
    latest = max(event.occurred_at for event in timeline.events)

    at_boundary = evaluate_timeline(timeline, as_of=latest + timedelta(hours=48))
    before_boundary = evaluate_timeline(
        timeline,
        as_of=latest + timedelta(hours=48) - timedelta(microseconds=1),
    )

    assert len(at_boundary.facts) == len(TriggerRule)
    assert TriggerRule.STALLED in at_boundary.matched_rules
    assert TriggerRule.STALLED not in before_boundary.matched_rules


def test_trigger_clock_must_be_timezone_aware() -> None:
    timeline = build_timelines(load_feed(FIXTURE).events)[0]

    with pytest.raises(ValueError, match="timezone-aware"):
        # A deliberately naive timestamp verifies that wall-clock ambiguity is rejected.
        evaluate_timeline(timeline, as_of=datetime(2026, 6, 30, 11, 0))  # noqa: DTZ001

    assert derive_as_of(timeline.events).tzinfo is UTC
