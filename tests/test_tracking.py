from collections.abc import Callable

import httpx
import pytest

from shipment_triage.adapters.tracking import TrackingClient
from shipment_triage.domain.enrichment import (
    AttemptOutcome,
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentStatus,
)


def _valid_payload(shipment_id: str = "SHP-00001") -> dict[str, object]:
    return {
        "shipmentId": shipment_id,
        "scac": "ESTE",
        "currentStatus": "EXCEPTION",
        "statusReason": "MISSED_APPOINTMENT",
        "lastEventTime": "2026-06-28T11:00:00Z",
        "promisedDeliveryDate": "2026-06-27",
        "estimatedDeliveryDate": "2026-06-27",
        "origin": {"city": "Phoenix", "state": "AZ"},
        "destination": {"city": "Atlanta", "state": "GA"},
        "referenceNumbers": {
            "orderId": "ORD-345185",
            "poNumber": "PO-28996",
            "bolNumber": "BOL-8537091",
        },
        "pieces": 3,
        "weightLbs": 1096,
        "scanHistory": [
            {
                "time": "2026-06-28T11:00:00Z",
                "status": "EXCEPTION",
                "city": "Atlanta",
                "state": "GA",
            }
        ],
        "exceptionNotes": "MISSED_APPOINTMENT reported by carrier",
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> TrackingClient:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return TrackingClient(
        base_url="https://tracking.example",
        api_key="test-secret",
        http_client=http_client,
        sleeper=lambda _delay: None,
        jitter=lambda _maximum: 0.0,
    )


def test_valid_response_requires_schema_id_match_and_header_auth() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://tracking.example/tracking/SHP-00001")
        assert request.headers["x-api-key"] == "test-secret"
        assert request.url.query == b""
        return httpx.Response(200, json=_valid_payload(), request=request)

    result = _client(handler).enrich("SHP-00001")

    assert result.status is EnrichmentStatus.VALID
    assert result.data_completeness is DataCompleteness.ENRICHED
    assert result.detail is not None
    assert result.detail.shipment_id == "SHP-00001"
    assert result.failure_reason is None
    assert len(result.attempts) == 1


def test_untrustworthy_success_and_wrong_id_are_retried_before_acceptance() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = _valid_payload()
        if calls == 1:
            payload["scanHistory"] = "unavailable"
        elif calls == 2:
            payload["shipmentId"] = "SHP-99999"
        return httpx.Response(200, json=payload, request=request)

    result = _client(handler).enrich("SHP-00001")

    assert result.status is EnrichmentStatus.VALID
    assert [attempt.outcome for attempt in result.attempts] == [
        AttemptOutcome.INVALID_BODY,
        AttemptOutcome.MISMATCHED_SHIPMENT,
        AttemptOutcome.SUCCESS,
    ]


@pytest.mark.parametrize(
    ("http_status", "expected_status", "expected_reason"),
    [
        (401, EnrichmentStatus.FAILED, EnrichmentFailureReason.AUTH),
        (404, EnrichmentStatus.NOT_FOUND, EnrichmentFailureReason.NOT_FOUND),
        (422, EnrichmentStatus.FAILED, EnrichmentFailureReason.CLIENT_ERROR),
    ],
)
def test_non_retryable_http_failures_are_explicit(
    http_status: int,
    expected_status: EnrichmentStatus,
    expected_reason: EnrichmentFailureReason,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(http_status, request=request)

    result = _client(handler).enrich("SHP-00001")

    assert calls == 1
    assert result.status is expected_status
    assert result.data_completeness is DataCompleteness.FEED_ONLY
    assert result.failure_reason is expected_reason


def test_retry_after_is_honored_for_rate_limit() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, json=_valid_payload(), request=request)

    client = TrackingClient(
        base_url="https://tracking.example",
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=sleeps.append,
        jitter=lambda _maximum: 0.0,
    )

    result = client.enrich("SHP-00001")

    assert result.status is EnrichmentStatus.VALID
    assert sleeps == [2.0]
    assert result.attempts[0].next_delay_seconds == 2.0


def test_schema_valid_inconsistencies_are_issues_not_silent_corrections() -> None:
    payload = _valid_payload()
    payload["currentStatus"] = "NEW_CARRIER_STATE"
    payload["lastEventTime"] = "2026-06-28T12:00:00Z"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    result = _client(handler).enrich("SHP-00001")

    assert result.status is EnrichmentStatus.VALID
    assert {issue.code for issue in result.data_quality_issues} == {
        "API_LAST_EVENT_MISMATCH",
        "API_UNKNOWN_STATUS",
    }


def test_timeout_exhaustion_returns_feed_only_attempt_history() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("simulated", request=request)

    result = _client(handler).enrich("SHP-00001")

    assert calls == 4
    assert result.status is EnrichmentStatus.FAILED
    assert result.failure_reason is EnrichmentFailureReason.TIMEOUT
    assert {attempt.outcome for attempt in result.attempts} == {AttemptOutcome.TIMEOUT}


def test_unsafe_shipment_id_is_rejected_before_network_use() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("network must not be called")

    with pytest.raises(ValueError, match="unsupported characters"):
        _client(handler).enrich("../../secret")
