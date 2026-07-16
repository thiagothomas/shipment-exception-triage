"""Narrow protocols at genuine external I/O boundaries."""

from collections.abc import Sequence
from typing import Protocol

from shipment_triage.domain.classification import ClassificationResult, EvidencePack
from shipment_triage.domain.enrichment import EnrichmentResult
from shipment_triage.domain.escalation import (
    EdiControlNumbers,
    EscalationDraft,
    EscalationReservation,
)


class Enricher(Protocol):
    def enrich(self, shipment_id: str) -> EnrichmentResult: ...


class Classifier(Protocol):
    def classify_batch(self, packs: Sequence[EvidencePack]) -> tuple[ClassificationResult, ...]: ...


class EdiRenderer(Protocol):
    def render(self, draft: EscalationDraft, controls: EdiControlNumbers) -> bytes: ...


class EscalationStore(Protocol):
    def reserve_or_get(
        self,
        decision_key: str,
        *,
        profile: str,
        sender_id: str,
        receiver_id: str,
    ) -> EscalationReservation: ...

    def finalize(
        self,
        reservation: EscalationReservation,
        *,
        artifact_path: str,
        artifact_hash: str,
    ) -> EscalationReservation: ...


__all__ = ["Classifier", "EdiRenderer", "Enricher", "EscalationStore"]
