"""Trustworthy synchronous client for the unreliable tracking boundary."""

from __future__ import annotations

import re
import secrets
import time
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shipment_triage.domain.enrichment import (
    AttemptOutcome,
    AttemptRecord,
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
    ReferenceNumbers,
    TrackingDetail,
    TrackingLocation,
    TrackingScan,
)
from shipment_triage.domain.models import DataQualityIssue
from shipment_triage.domain.statuses import CanonicalStatus

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_SHIPMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_KNOWN_API_STATUSES = frozenset(status.value for status in CanonicalStatus)
_RANDOM = secrets.SystemRandom()


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)


class _LocationDto(_BoundaryModel):
    city: str | None = None
    state: str | None = None


class _ReferenceNumbersDto(_BoundaryModel):
    order_id: str | None = Field(default=None, alias="orderId")
    po_number: str | None = Field(default=None, alias="poNumber")
    bol_number: str | None = Field(default=None, alias="bolNumber")


class _ScanDto(_BoundaryModel):
    time: str
    status: str = Field(min_length=1)
    city: str | None = None
    state: str | None = None


class _TrackingDto(_BoundaryModel):
    shipment_id: str = Field(alias="shipmentId")
    scac: str = Field(min_length=2)
    current_status: str = Field(alias="currentStatus", min_length=1)
    status_reason: str | None = Field(default=None, alias="statusReason")
    last_event_time: str = Field(alias="lastEventTime")
    promised_delivery_date: str | None = Field(default=None, alias="promisedDeliveryDate")
    estimated_delivery_date: str | None = Field(default=None, alias="estimatedDeliveryDate")
    origin: _LocationDto | None = None
    destination: _LocationDto | None = None
    reference_numbers: _ReferenceNumbersDto | None = Field(default=None, alias="referenceNumbers")
    pieces: int | None = Field(default=None, ge=0)
    weight_lbs: int | float | None = Field(default=None, alias="weightLbs", ge=0)
    scan_history: list[_ScanDto] = Field(alias="scanHistory")
    exception_notes: str | None = Field(default=None, alias="exceptionNotes")


class _ResponseValidationError(ValueError):
    pass


class _ShipmentMismatchError(_ResponseValidationError):
    pass


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _full_jitter(maximum: float) -> float:
    return _RANDOM.uniform(0.0, maximum)


def _parse_aware(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _ResponseValidationError(f"{field_name} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _ResponseValidationError(f"{field_name} must include a timezone")
    return parsed


def _parse_date(value: str | None, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _ResponseValidationError(f"{field_name} is not a valid date") from exc


def _location(value: _LocationDto | None) -> TrackingLocation | None:
    if value is None or (value.city is None and value.state is None):
        return None
    return TrackingLocation(city=value.city, state=value.state)


def _references(value: _ReferenceNumbersDto | None) -> ReferenceNumbers | None:
    if value is None:
        return None
    return ReferenceNumbers(
        order_id=value.order_id,
        po_number=value.po_number,
        bol_number=value.bol_number,
    )


def _parse_detail(
    payload: Mapping[str, Any],
    *,
    requested_shipment_id: str,
    now: datetime,
) -> tuple[TrackingDetail, tuple[DataQualityIssue, ...]]:
    if payload.get("ok") is False:
        raise _ResponseValidationError("response explicitly reports failure")
    try:
        dto = _TrackingDto.model_validate(payload)
    except ValidationError as exc:
        raise _ResponseValidationError("response does not match the tracking schema") from exc
    if dto.shipment_id != requested_shipment_id:
        raise _ShipmentMismatchError("response shipment ID does not match the request")

    scans = tuple(
        TrackingScan(
            time=_parse_aware(scan.time, "scanHistory.time"),
            status=scan.status,
            city=scan.city,
            state=scan.state,
        )
        for scan in dto.scan_history
    )
    last_event_time = _parse_aware(dto.last_event_time, "lastEventTime")
    detail = TrackingDetail(
        shipment_id=dto.shipment_id,
        scac=dto.scac,
        current_status=dto.current_status,
        status_reason=dto.status_reason,
        last_event_time=last_event_time,
        promised_delivery_date=_parse_date(dto.promised_delivery_date, "promisedDeliveryDate"),
        estimated_delivery_date=_parse_date(dto.estimated_delivery_date, "estimatedDeliveryDate"),
        origin=_location(dto.origin),
        destination=_location(dto.destination),
        reference_numbers=_references(dto.reference_numbers),
        pieces=dto.pieces,
        weight_lbs=dto.weight_lbs,
        scan_history=scans,
        exception_notes=dto.exception_notes,
    )

    issues: list[DataQualityIssue] = []
    if dto.current_status not in _KNOWN_API_STATUSES:
        issues.append(
            DataQualityIssue(
                code="API_UNKNOWN_STATUS",
                message="Tracking API returned an unmapped current status.",
            )
        )
    scan_times = [scan.time for scan in scans]
    if scan_times != sorted(scan_times):
        issues.append(
            DataQualityIssue(
                code="API_SCAN_ORDER",
                message="Tracking API scan history is not in chronological order.",
            )
        )
    if scan_times and max(scan_times) != last_event_time:
        issues.append(
            DataQualityIssue(
                code="API_LAST_EVENT_MISMATCH",
                message="Tracking API last-event time disagrees with its scan history.",
            )
        )
    if last_event_time > now + timedelta(minutes=5):
        issues.append(
            DataQualityIssue(
                code="API_FUTURE_EVENT",
                message="Tracking API last event is implausibly in the future.",
            )
        )
    return detail, tuple(issues)


def _retry_after(response: httpx.Response, now: datetime, maximum: float) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            seconds = (retry_at - now).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(max(seconds, 0.0), maximum)


class TrackingClient:
    """Accept a tracking response only after status, schema, and identity checks."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        http_client: httpx.Client,
        max_attempts: int = 4,
        timeout: httpx.Timeout | float | None = None,
        backoff_cap_seconds: float = 8.0,
        retry_after_cap_seconds: float = 30.0,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[float], float] = _full_jitter,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _now_utc,
    ) -> None:
        parsed_base = httpx.URL(base_url)
        if parsed_base.scheme not in {"http", "https"} or parsed_base.host is None:
            raise ValueError("tracking base URL must be absolute HTTP(S)")
        if parsed_base.query or parsed_base.fragment or parsed_base.userinfo:
            raise ValueError("tracking base URL cannot contain credentials, query, or fragment")
        if not api_key:
            raise ValueError("tracking API key is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._base_url = str(parsed_base).rstrip("/")
        self._api_key = api_key
        self._http = http_client
        self._max_attempts = max_attempts
        self._timeout = timeout if timeout is not None else httpx.Timeout(10.0, connect=3.0)
        self._backoff_cap = backoff_cap_seconds
        self._retry_after_cap = retry_after_cap_seconds
        self._sleep = sleeper
        self._jitter = jitter
        self._clock = clock
        self._now = now

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = _retry_after(response, self._now(), self._retry_after_cap)
            if retry_after is not None:
                return retry_after
        maximum = min(self._backoff_cap, 2 ** (attempt - 1))
        return self._jitter(float(maximum))

    def _failed(
        self,
        *,
        attempts: list[AttemptRecord],
        reason: EnrichmentFailureReason,
        status: EnrichmentStatus = EnrichmentStatus.FAILED,
    ) -> EnrichmentResult:
        return EnrichmentResult(
            status=status,
            data_completeness=DataCompleteness.FEED_ONLY,
            detail=None,
            attempts=tuple(attempts),
            failure_reason=reason,
        )

    def enrich(self, shipment_id: str) -> EnrichmentResult:
        if _SHIPMENT_ID.fullmatch(shipment_id) is None:
            raise ValueError("shipment ID contains unsupported characters")
        encoded_id = quote(shipment_id, safe="")
        url = f"{self._base_url}/tracking/{encoded_id}"
        attempts: list[AttemptRecord] = []
        last_reason = EnrichmentFailureReason.TRANSPORT_ERROR

        for attempt in range(1, self._max_attempts + 1):
            started = self._clock()
            response: httpx.Response | None = None
            outcome: AttemptOutcome
            retryable = False
            detail_message: str
            http_status: int | None = None
            try:
                response = self._http.get(
                    url,
                    headers={"x-api-key": self._api_key},
                    timeout=self._timeout,
                )
                http_status = response.status_code
                if response.status_code in {401, 403}:
                    last_reason = EnrichmentFailureReason.AUTH
                    outcome = AttemptOutcome.HTTP_ERROR
                    detail_message = "Tracking authentication failed."
                elif response.status_code == 404:
                    last_reason = EnrichmentFailureReason.NOT_FOUND
                    outcome = AttemptOutcome.HTTP_ERROR
                    detail_message = "Shipment was not found."
                elif response.status_code in _RETRYABLE_STATUSES:
                    last_reason = (
                        EnrichmentFailureReason.RATE_LIMITED
                        if response.status_code == 429
                        else EnrichmentFailureReason.SERVER_ERROR
                    )
                    outcome = AttemptOutcome.HTTP_ERROR
                    retryable = True
                    detail_message = f"Retryable HTTP {response.status_code}."
                elif not response.is_success:
                    last_reason = EnrichmentFailureReason.CLIENT_ERROR
                    outcome = AttemptOutcome.HTTP_ERROR
                    detail_message = f"Non-retryable HTTP {response.status_code}."
                else:
                    try:
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise _ResponseValidationError("response JSON must be an object")
                        trusted, issues = _parse_detail(
                            payload,
                            requested_shipment_id=shipment_id,
                            now=self._now(),
                        )
                    except _ShipmentMismatchError:
                        last_reason = EnrichmentFailureReason.MISMATCHED_SHIPMENT
                        outcome = AttemptOutcome.MISMATCHED_SHIPMENT
                        retryable = True
                        detail_message = "Response shipment ID did not match the request."
                    except (ValueError, ValidationError, _ResponseValidationError):
                        last_reason = EnrichmentFailureReason.INVALID_RESPONSE
                        outcome = AttemptOutcome.INVALID_BODY
                        retryable = True
                        detail_message = "Response body was not trustworthy."
                    else:
                        attempts.append(
                            AttemptRecord(
                                attempt=attempt,
                                outcome=AttemptOutcome.SUCCESS,
                                http_status=http_status,
                                retryable=False,
                                duration_ms=max((self._clock() - started) * 1000, 0.0),
                                detail="Validated tracking response.",
                            )
                        )
                        return EnrichmentResult(
                            status=EnrichmentStatus.VALID,
                            data_completeness=DataCompleteness.ENRICHED,
                            detail=trusted,
                            attempts=tuple(attempts),
                            data_quality_issues=issues,
                        )
            except httpx.TimeoutException:
                last_reason = EnrichmentFailureReason.TIMEOUT
                outcome = AttemptOutcome.TIMEOUT
                retryable = True
                detail_message = "Tracking request timed out."
            except httpx.TransportError:
                last_reason = EnrichmentFailureReason.TRANSPORT_ERROR
                outcome = AttemptOutcome.TRANSPORT_ERROR
                retryable = True
                detail_message = "Tracking transport failed."

            can_retry = retryable and attempt < self._max_attempts
            delay = self._delay(attempt, response) if can_retry else None
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    outcome=outcome,
                    http_status=http_status,
                    retryable=retryable,
                    duration_ms=max((self._clock() - started) * 1000, 0.0),
                    next_delay_seconds=delay,
                    detail=detail_message,
                )
            )
            if not can_retry:
                status = (
                    EnrichmentStatus.NOT_FOUND
                    if last_reason is EnrichmentFailureReason.NOT_FOUND
                    else EnrichmentStatus.FAILED
                )
                return self._failed(attempts=attempts, reason=last_reason, status=status)
            self._sleep(delay if delay is not None else 0.0)

        raise AssertionError("tracking retry loop must return")


__all__ = ["TrackingClient"]
