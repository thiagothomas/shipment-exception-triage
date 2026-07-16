from shipment_triage.application.enrichment import enrich_shipments
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)


class _AuthFailingEnricher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enrich(self, shipment_id: str) -> EnrichmentResult:
        self.calls.append(shipment_id)
        return EnrichmentResult(
            status=EnrichmentStatus.FAILED,
            data_completeness=DataCompleteness.FEED_ONLY,
            attempts=(),
            failure_reason=EnrichmentFailureReason.AUTH,
        )


class _EchoEnricher:
    def enrich(self, shipment_id: str) -> EnrichmentResult:
        if shipment_id == "SHP-00002":
            raise RuntimeError("simulated adapter bug")
        return EnrichmentResult(
            status=EnrichmentStatus.NOT_FOUND,
            data_completeness=DataCompleteness.FEED_ONLY,
            attempts=(),
            failure_reason=EnrichmentFailureReason.NOT_FOUND,
        )


def test_auth_failure_opens_circuit_before_unscheduled_calls() -> None:
    enricher = _AuthFailingEnricher()

    results = enrich_shipments(
        enricher,
        ("SHP-00001", "SHP-00002", "SHP-00003"),
        workers=1,
    )

    assert enricher.calls == ["SHP-00001"]
    assert tuple(results) == ("SHP-00001", "SHP-00002", "SHP-00003")
    assert results["SHP-00001"].failure_reason is EnrichmentFailureReason.AUTH
    assert results["SHP-00002"].failure_reason is EnrichmentFailureReason.AUTH_CIRCUIT_OPEN
    assert results["SHP-00003"].failure_reason is EnrichmentFailureReason.AUTH_CIRCUIT_OPEN


def test_executor_preserves_input_order_and_contains_adapter_exceptions() -> None:
    results = enrich_shipments(
        _EchoEnricher(),
        ("SHP-00001", "SHP-00002", "SHP-00003"),
        workers=2,
    )

    assert tuple(results) == ("SHP-00001", "SHP-00002", "SHP-00003")
    assert results["SHP-00002"].failure_reason is EnrichmentFailureReason.UNEXPECTED_ERROR
    assert results["SHP-00001"].failure_reason is EnrichmentFailureReason.NOT_FOUND
