import json
from pathlib import Path

from shipment_triage.adapters.feed import load_feed
from shipment_triage.domain.timelines import TerminalState, build_timelines

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def test_fixture_terminal_conflicts_are_order_independent() -> None:
    timelines = build_timelines(load_feed(FIXTURE).events)

    conflicted = {
        timeline.shipment_id
        for timeline in timelines
        if timeline.terminal_state is TerminalState.CONFLICTED
    }

    assert len(timelines) == 125
    assert conflicted == {
        "SHP-00008",
        "SHP-00027",
        "SHP-00035",
        "SHP-00083",
        "SHP-00100",
        "SHP-00119",
    }


def test_later_clean_delivery_resolves_an_earlier_same_time_conflict(tmp_path: Path) -> None:
    records = [
        {
            "shipmentId": "SHP-RESOLVE",
            "scac": "UPSN",
            "statusCode": status,
            "statusText": status,
            "ts": occurred_at,
        }
        for status, occurred_at in (
            ("DL", "2026-06-29T09:00:00Z"),
            ("IT", "2026-06-29T09:00:00Z"),
            ("DL", "2026-06-29T10:00:00Z"),
        )
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    timeline = build_timelines(load_feed(path).events)[0]

    assert timeline.terminal_state is TerminalState.DELIVERED
    assert timeline.data_quality_issues == ()
