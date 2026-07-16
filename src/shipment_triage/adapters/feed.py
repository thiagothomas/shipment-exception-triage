"""Bounded JSONL loader and carrier normalization adapters."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from shipment_triage.domain.merge import merge_semantic_duplicates
from shipment_triage.domain.models import (
    DataQualityIssue,
    FeedLoadResult,
    Location,
    NormalizedEvent,
    RawRecordRef,
    RejectedRecord,
)
from shipment_triage.domain.statuses import CanonicalStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_UPSN_STATUSES = {
    "PU": CanonicalStatus.PICKED_UP,
    "IT": CanonicalStatus.IN_TRANSIT,
    "AR": CanonicalStatus.ARRIVED_FACILITY,
    "DP": CanonicalStatus.DEPARTED_FACILITY,
    "OD": CanonicalStatus.OUT_FOR_DELIVERY,
    "DL": CanonicalStatus.DELIVERED,
    "EX": CanonicalStatus.EXCEPTION,
    "DE": CanonicalStatus.EXCEPTION,
    "HL": CanonicalStatus.HELD,
}

_FXFE_STATUSES = {
    100: CanonicalStatus.PICKED_UP,
    200: CanonicalStatus.IN_TRANSIT,
    300: CanonicalStatus.ARRIVED_FACILITY,
    320: CanonicalStatus.DEPARTED_FACILITY,
    400: CanonicalStatus.OUT_FOR_DELIVERY,
    500: CanonicalStatus.DELIVERED,
    850: CanonicalStatus.HELD,
    900: CanonicalStatus.EXCEPTION,
    950: CanonicalStatus.DELAYED,
}

_ESTE_STATUSES = {
    "PICKED UP": CanonicalStatus.PICKED_UP,
    "IN TRANSIT": CanonicalStatus.IN_TRANSIT,
    "ARRIVED AT TERMINAL": CanonicalStatus.ARRIVED_FACILITY,
    "DEPARTED TERMINAL": CanonicalStatus.DEPARTED_FACILITY,
    "OUT FOR DELIVERY": CanonicalStatus.OUT_FOR_DELIVERY,
    "DELIVERED": CanonicalStatus.DELIVERED,
    "HELD - CONSIGNEE CLOSED": CanonicalStatus.HELD,
    "DELAYED - WEATHER": CanonicalStatus.DELAYED,
    "DELAYED - MECHANICAL": CanonicalStatus.DELAYED,
    "DAMAGED IN TRANSIT": CanonicalStatus.DAMAGED,
    "MISSED DELIVERY APPOINTMENT": CanonicalStatus.MISSED_APPOINTMENT,
}


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class _UpsnRecord(_BoundaryModel):
    shipmentId: str
    scac: str
    statusCode: str
    statusText: str
    ts: str
    promisedDeliveryDate: str | None = None
    city: str | None = None
    state: str | None = None


class _FxfeEvent(_BoundaryModel):
    code: int
    description: str


class _FxfeLocation(_BoundaryModel):
    city: str | None = None
    region: str | None = None


class _FxfeRecord(_BoundaryModel):
    tracking_number: str
    carrier: str
    event: _FxfeEvent
    event_time: int
    location: _FxfeLocation | None = None
    sla_date: str | None = None


class _EsteRecord(_BoundaryModel):
    pro_number: str
    scac: str
    status: str
    datetime: str
    terminal: str | None = None
    appt: str | None = None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _event_id(*, carrier: str, shipment_id: str, occurred_at: datetime, raw_status: str) -> str:
    identity = json.dumps(
        [carrier, shipment_id, occurred_at.isoformat(), raw_status],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return f"evt_{_sha256(identity)[:20]}"


def _parse_date(value: str | None, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 date") from exc


def _parse_aware_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed


def _location(city: str | None, state: str | None) -> Location | None:
    normalized_city = city.strip() if city and city.strip() else None
    normalized_state = state.strip() if state and state.strip() else None
    if normalized_city is None and normalized_state is None:
        return None
    return Location(city=normalized_city, state=normalized_state)


def _unknown_issue(raw_status: str, ref: RawRecordRef) -> DataQualityIssue:
    return DataQualityIssue(
        code="UNKNOWN_STATUS",
        message=f"Carrier status is not mapped: {raw_status[:120]}",
        record_refs=(ref,),
    )


def _build_event(
    *,
    carrier: str,
    shipment_id: str,
    status: CanonicalStatus,
    raw_status: str,
    description: str | None,
    occurred_at: datetime,
    location: Location | None,
    promised_date: date | None,
    ref: RawRecordRef,
    issues: tuple[DataQualityIssue, ...] = (),
) -> NormalizedEvent:
    status_issues = (
        *issues,
        *(() if status is not CanonicalStatus.UNKNOWN else (_unknown_issue(raw_status, ref),)),
    )
    return NormalizedEvent(
        event_id=_event_id(
            carrier=carrier,
            shipment_id=shipment_id,
            occurred_at=occurred_at,
            raw_status=raw_status,
        ),
        shipment_id=shipment_id,
        carrier=carrier,
        status=status,
        raw_status=raw_status,
        description=description.strip() if description else None,
        occurred_at=occurred_at,
        location=location,
        promised_date=promised_date,
        provenance=(ref,),
        data_quality_issues=status_issues,
    )


def _normalize_upsn(raw: Mapping[str, Any], ref: RawRecordRef) -> NormalizedEvent:
    record = _UpsnRecord.model_validate(raw)
    if record.scac != "UPSN":
        raise ValueError("UPSN record has an unexpected SCAC")
    occurred_at = _parse_aware_datetime(record.ts, "ts")
    raw_status = record.statusCode.strip().upper()
    return _build_event(
        carrier="UPSN",
        shipment_id=record.shipmentId,
        status=_UPSN_STATUSES.get(raw_status, CanonicalStatus.UNKNOWN),
        raw_status=raw_status,
        description=record.statusText,
        occurred_at=occurred_at,
        location=_location(record.city, record.state),
        promised_date=_parse_date(record.promisedDeliveryDate, "promisedDeliveryDate"),
        ref=ref,
    )


def _normalize_fxfe(raw: Mapping[str, Any], ref: RawRecordRef) -> NormalizedEvent:
    record = _FxfeRecord.model_validate(raw)
    if record.carrier != "FXFE":
        raise ValueError("FXFE record has an unexpected carrier")
    try:
        occurred_at = datetime.fromtimestamp(record.event_time / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise ValueError("event_time must be a valid epoch-millisecond timestamp") from exc
    raw_status = str(record.event.code)
    return _build_event(
        carrier="FXFE",
        shipment_id=record.tracking_number,
        status=_FXFE_STATUSES.get(record.event.code, CanonicalStatus.UNKNOWN),
        raw_status=raw_status,
        description=record.event.description,
        occurred_at=occurred_at,
        location=(
            _location(record.location.city, record.location.region)
            if record.location is not None
            else None
        ),
        promised_date=_parse_date(record.sla_date, "sla_date"),
        ref=ref,
    )


def _parse_terminal(value: str | None) -> Location | None:
    if value is None:
        return None
    city, separator, state = value.rpartition(",")
    if not separator:
        return _location(value, None)
    return _location(city, state)


def _normalize_este(raw: Mapping[str, Any], ref: RawRecordRef) -> NormalizedEvent:
    record = _EsteRecord.model_validate(raw)
    if record.scac != "ESTE":
        raise ValueError("ESTE record has an unexpected SCAC")
    try:
        occurred_at = datetime.strptime(record.datetime, "%m/%d/%Y %H:%M").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("datetime must use MM/DD/YYYY HH:MM") from exc
    timezone_issue = DataQualityIssue(
        code="ASSUMED_TIMEZONE",
        message="Naive ESTE fixture timestamp interpreted as UTC.",
        record_refs=(ref,),
    )
    raw_status = record.status.strip().upper()
    return _build_event(
        carrier="ESTE",
        shipment_id=record.pro_number,
        status=_ESTE_STATUSES.get(raw_status, CanonicalStatus.UNKNOWN),
        raw_status=raw_status,
        description=record.status,
        occurred_at=occurred_at,
        location=_parse_terminal(record.terminal),
        promised_date=_parse_date(record.appt, "appt"),
        ref=ref,
        issues=(timezone_issue,),
    )


def _normalize(raw: Mapping[str, Any], ref: RawRecordRef) -> NormalizedEvent:
    if "shipmentId" in raw:
        return _normalize_upsn(raw, ref)
    if "tracking_number" in raw:
        return _normalize_fxfe(raw, ref)
    if "pro_number" in raw:
        return _normalize_este(raw, ref)
    raise ValueError("record does not match a supported carrier schema")


def _reject(line_number: int, raw_line: bytes, code: str, message: str) -> RejectedRecord:
    excerpt = raw_line.decode("utf-8", errors="replace").strip()
    safe_excerpt = "".join(character if character.isprintable() else "?" for character in excerpt)
    return RejectedRecord(
        line_number=line_number,
        raw_hash=_sha256(raw_line),
        code=code,
        message=message[:400],
        excerpt=safe_excerpt[:240],
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def load_feed(path: Path, *, max_line_bytes: int = 64 * 1024) -> FeedLoadResult:
    """Load, validate, exact-deduplicate, normalize, and semantically merge a JSONL feed."""

    raw_count = 0
    exact_duplicate_count = 0
    unique_events: list[NormalizedEvent] = []
    event_index_by_raw_hash: dict[str, int] = {}
    rejected: list[RejectedRecord] = []

    with path.open("rb") as feed:
        for line_number, raw_line in enumerate(feed, start=1):
            raw_count += 1
            if len(raw_line) > max_line_bytes:
                rejected.append(
                    _reject(line_number, raw_line, "LINE_TOO_LARGE", "Feed line is too large.")
                )
                continue
            if not raw_line.strip():
                rejected.append(_reject(line_number, raw_line, "EMPTY_LINE", "Feed line is empty."))
                continue
            try:
                decoded = json.loads(raw_line, parse_constant=_reject_constant)
                if not isinstance(decoded, dict):
                    raise ValueError("JSON value must be an object")
                canonical = json.dumps(
                    decoded,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
                raw_hash = _sha256(canonical)
                ref = RawRecordRef(line_number=line_number, raw_hash=raw_hash)
                duplicate_index = event_index_by_raw_hash.get(raw_hash)
                if duplicate_index is not None:
                    exact_duplicate_count += 1
                    duplicate = unique_events[duplicate_index]
                    unique_events[duplicate_index] = duplicate.model_copy(
                        update={"provenance": (*duplicate.provenance, ref)}
                    )
                    continue
                event = _normalize(decoded, ref)
            except (UnicodeDecodeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                rejected.append(_reject(line_number, raw_line, "INVALID_RECORD", str(exc)))
                continue
            event_index_by_raw_hash[raw_hash] = len(unique_events)
            unique_events.append(event)

    merged_events = merge_semantic_duplicates(unique_events)
    return FeedLoadResult(
        raw_record_count=raw_count,
        exact_duplicate_count=exact_duplicate_count,
        semantic_merge_count=len(unique_events) - len(merged_events),
        events=merged_events,
        rejected_records=tuple(rejected),
    )


__all__ = ["load_feed"]
