"""Narrow protocols at genuine external I/O boundaries."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from shipment_triage.domain.classification import ClassificationResult, EvidencePack
from shipment_triage.domain.enrichment import EnrichmentResult
from shipment_triage.domain.escalation import (
    EdiControlNumbers,
    EscalationDraft,
    EscalationReservation,
)
from shipment_triage.domain.runs import StoredArtifactResult


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


class ArtifactWriter(Protocol):
    def write_bytes(self, relative: str, payload: bytes) -> tuple[str, str]: ...

    def write_json(
        self,
        relative: str,
        value: BaseModel | dict[str, Any],
    ) -> tuple[str, str]: ...

    def write_jsonl(self, relative: str, values: tuple[BaseModel, ...]) -> tuple[str, str]: ...

    def output_relative(self, run_relative: str) -> str: ...

    def inspect_or_restore(
        self,
        output_relative: str,
        *,
        expected_hash: str,
        payload: bytes,
    ) -> StoredArtifactResult: ...


class ArtifactWriterFactory(Protocol):
    def __call__(self, output_root: str | Path, run_id: str) -> ArtifactWriter: ...


__all__ = [
    "ArtifactWriter",
    "ArtifactWriterFactory",
    "Classifier",
    "EdiRenderer",
    "Enricher",
    "EscalationStore",
]
