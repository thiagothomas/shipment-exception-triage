"""Bounded tracking fan-out with a run-wide authentication circuit."""

from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from shipment_triage.application.ports import Enricher
from shipment_triage.domain.enrichment import (
    DataCompleteness,
    EnrichmentFailureReason,
    EnrichmentResult,
    EnrichmentStatus,
)
from shipment_triage.domain.models import DataQualityIssue


def _circuit_result() -> EnrichmentResult:
    return EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.AUTH_CIRCUIT_OPEN,
    )


def _unexpected_result() -> EnrichmentResult:
    return EnrichmentResult(
        status=EnrichmentStatus.FAILED,
        data_completeness=DataCompleteness.FEED_ONLY,
        attempts=(),
        failure_reason=EnrichmentFailureReason.UNEXPECTED_ERROR,
        data_quality_issues=(
            DataQualityIssue(
                code="ENRICHER_UNEXPECTED_ERROR",
                message="Enrichment adapter raised outside its declared result contract.",
            ),
        ),
    )


def enrich_shipments(
    enricher: Enricher,
    shipment_ids: Iterable[str],
    *,
    workers: int = 6,
) -> dict[str, EnrichmentResult]:
    """Enrich unique shipments while preventing new calls after an auth failure."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    ordered_ids = tuple(shipment_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("shipment IDs must be unique")
    if not ordered_ids:
        return {}

    results: list[EnrichmentResult | None] = [None] * len(ordered_ids)
    next_index = 0
    auth_circuit_open = False

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tracking") as executor:
        pending: dict[Future[EnrichmentResult], tuple[int, str]] = {}

        def submit_available() -> None:
            nonlocal next_index
            while (
                not auth_circuit_open and next_index < len(ordered_ids) and len(pending) < workers
            ):
                shipment_id = ordered_ids[next_index]
                pending[executor.submit(enricher.enrich, shipment_id)] = (next_index, shipment_id)
                next_index += 1

        submit_available()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda item: pending[item][0]):
                index, _shipment_id = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = _unexpected_result()
                results[index] = result
                if result.failure_reason is EnrichmentFailureReason.AUTH:
                    # Only work that was already in flight is allowed to finish.
                    auth_circuit_open = True
            submit_available()

    if auth_circuit_open:
        for index in range(next_index, len(ordered_ids)):
            results[index] = _circuit_result()

    if any(result is None for result in results):
        raise AssertionError("every shipment must receive an enrichment result")
    return {
        shipment_id: result
        for shipment_id, result in zip(ordered_ids, results, strict=True)
        if result is not None
    }


__all__ = ["enrich_shipments"]
