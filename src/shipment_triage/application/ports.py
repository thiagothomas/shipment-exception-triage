"""Narrow protocols at genuine external I/O boundaries."""

from typing import Protocol

from shipment_triage.domain.enrichment import EnrichmentResult


class Enricher(Protocol):
    def enrich(self, shipment_id: str) -> EnrichmentResult: ...


__all__ = ["Enricher"]
