import json
from pathlib import Path

from shipment_triage.adapters.feed import load_feed
from shipment_triage.domain.statuses import CanonicalStatus

FIXTURE = Path(__file__).parents[1] / "events.jsonl"


def test_fixture_profile_counts_are_stable() -> None:
    result = load_feed(FIXTURE)

    assert result.raw_record_count == 260
    assert result.exact_duplicate_count == 6
    assert len(result.events) == 245
    assert len({event.shipment_id for event in result.events}) == 125
    assert result.rejected_records == ()


def test_semantic_duplicates_merge_complementary_fields_and_report_conflicts(
    tmp_path: Path,
) -> None:
    records = [
        {
            "shipmentId": "SHP-MERGE",
            "scac": "UPSN",
            "statusCode": "IT",
            "statusText": "In Transit",
            "ts": "2026-06-29T09:00:00Z",
            "city": "Reno",
        },
        {
            "shipmentId": "SHP-MERGE",
            "scac": "UPSN",
            "statusCode": "IT",
            "statusText": "In Transit",
            "ts": "2026-06-29T09:00:00Z",
            "state": "NV",
        },
        {
            "shipmentId": "SHP-MERGE",
            "scac": "UPSN",
            "statusCode": "IT",
            "statusText": "In Transit",
            "ts": "2026-06-29T09:00:00Z",
            "city": "Denver",
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    result = load_feed(path)

    assert result.semantic_merge_count == 2
    assert len(result.events) == 1
    event = result.events[0]
    assert event.location is not None
    assert event.location.city == "Reno"
    assert event.location.state == "NV"
    assert len(event.provenance) == 3
    assert {issue.code for issue in event.data_quality_issues} == {"DUPLICATE_FIELD_CONFLICT"}


def test_invalid_lines_are_quarantined_and_unknown_statuses_remain_visible(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    unknown = {
        "shipmentId": "SHP-UNKNOWN",
        "scac": "UPSN",
        "statusCode": "ZZ",
        "statusText": "Novel status",
        "ts": "2026-06-29T09:00:00Z",
    }
    path.write_text(f"not-json\n{json.dumps(unknown)}\n", encoding="utf-8")

    result = load_feed(path)

    assert len(result.rejected_records) == 1
    assert result.rejected_records[0].code == "INVALID_RECORD"
    assert len(result.events) == 1
    assert result.events[0].status is CanonicalStatus.UNKNOWN
    assert {issue.code for issue in result.events[0].data_quality_issues} == {"UNKNOWN_STATUS"}
