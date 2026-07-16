"""Narrow protocols at genuine external I/O boundaries."""

from collections.abc import Sequence
from typing import Protocol

from shipment_triage.domain.classification import ClassificationResult, EvidencePack
from shipment_triage.domain.enrichment import EnrichmentResult


class Enricher(Protocol):
    def enrich(self, shipment_id: str) -> EnrichmentResult: ...


class Classifier(Protocol):
    def classify_batch(self, packs: Sequence[EvidencePack]) -> tuple[ClassificationResult, ...]: ...


__all__ = ["Classifier", "Enricher"]
