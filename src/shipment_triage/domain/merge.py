"""Deterministic semantic duplicate coalescing."""

from collections.abc import Iterable

from shipment_triage.domain.models import (
    DataQualityIssue,
    Location,
    NormalizedEvent,
    RawRecordRef,
)


def _merge_scalar[T](
    left: T | None,
    right: T | None,
    *,
    field_name: str,
    refs: tuple[RawRecordRef, ...],
) -> tuple[T | None, DataQualityIssue | None]:
    if left is None:
        return right, None
    if right is None or left == right:
        return left, None
    return left, DataQualityIssue(
        code="DUPLICATE_FIELD_CONFLICT",
        message=f"Semantic duplicate values disagree for {field_name}.",
        record_refs=refs,
    )


def _merge_location(
    left: Location | None,
    right: Location | None,
    refs: tuple[RawRecordRef, ...],
) -> tuple[Location | None, tuple[DataQualityIssue, ...]]:
    if left is None:
        return right, ()
    if right is None:
        return left, ()
    city, city_issue = _merge_scalar(left.city, right.city, field_name="location.city", refs=refs)
    state, state_issue = _merge_scalar(
        left.state,
        right.state,
        field_name="location.state",
        refs=refs,
    )
    issues = tuple(issue for issue in (city_issue, state_issue) if issue is not None)
    return Location(city=city, state=state), issues


def merge_semantic_duplicates(
    events: Iterable[NormalizedEvent],
) -> tuple[NormalizedEvent, ...]:
    """Coalesce events with the same carrier business identity."""

    merged: dict[tuple[str, str, object, str], NormalizedEvent] = {}
    for event in events:
        identity = (event.carrier, event.shipment_id, event.occurred_at, event.raw_status)
        current = merged.get(identity)
        if current is None:
            merged[identity] = event
            continue

        refs = tuple(
            sorted((*current.provenance, *event.provenance), key=lambda ref: ref.line_number)
        )
        location, location_issues = _merge_location(current.location, event.location, refs)
        promised_date, promised_issue = _merge_scalar(
            current.promised_date,
            event.promised_date,
            field_name="promised_date",
            refs=refs,
        )
        description, description_issue = _merge_scalar(
            current.description,
            event.description,
            field_name="description",
            refs=refs,
        )
        new_issues = (*location_issues, promised_issue, description_issue)
        merged[identity] = current.model_copy(
            update={
                "description": description,
                "location": location,
                "promised_date": promised_date,
                "provenance": refs,
                "data_quality_issues": (
                    *current.data_quality_issues,
                    *event.data_quality_issues,
                    *(issue for issue in new_issues if issue is not None),
                ),
            }
        )

    return tuple(
        sorted(
            merged.values(),
            key=lambda event: (
                event.shipment_id,
                event.occurred_at,
                event.carrier,
                event.raw_status,
            ),
        )
    )


__all__ = ["merge_semantic_duplicates"]
