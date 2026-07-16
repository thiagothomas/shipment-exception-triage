"""Single-process orchestration shared by the CLI and HTTP adapters."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from shipment_triage.adapters.feed import load_feed
from shipment_triage.application.enrichment import enrich_shipments
from shipment_triage.application.escalation import (
    UnrepresentableEscalation,
    build_escalation_draft,
)
from shipment_triage.application.evidence import build_evidence_pack
from shipment_triage.application.guardrails import apply_guardrails
from shipment_triage.application.idempotency import compute_decision_key
from shipment_triage.domain.classification import ClassificationSource
from shipment_triage.domain.enrichment import EnrichmentStatus
from shipment_triage.domain.escalation import ReservationState
from shipment_triage.domain.policy import (
    FinalDisposition,
    PolicyDecision,
    PolicyOverride,
    VerificationState,
    decide_disposition,
)
from shipment_triage.domain.runs import (
    AsOfSource,
    DecisionMetadata,
    EscalationArtifactStatus,
    EscalationRecord,
    RunArtifactPaths,
    RunStatus,
    RunSummary,
    ShipmentDecision,
    StoredArtifactStatus,
    TriageRunResult,
)
from shipment_triage.domain.timelines import build_timelines
from shipment_triage.domain.triggers import derive_as_of, evaluate_timelines

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pydantic import BaseModel

    from shipment_triage.application.ports import (
        ArtifactWriter,
        ArtifactWriterFactory,
        Classifier,
        EdiRenderer,
        Enricher,
        EscalationStore,
    )
    from shipment_triage.domain.classification import ClassificationResult, EvidencePack
    from shipment_triage.domain.enrichment import EnrichmentResult
    from shipment_triage.domain.timelines import ShipmentTimeline
    from shipment_triage.domain.triggers import TriggerEvaluation

_TRIGGER_POLICY_VERSION = "1"
_CLASSIFICATION_POLICY_VERSION = "1"
_ESCALATION_POLICY_VERSION = "1"


class PipelineInputError(ValueError):
    """The requested run cannot start from the supplied feed or options."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    events_path: Path
    output_root: Path
    provider: str
    model: str | None
    as_of: datetime | None = None
    tracking_workers: int = 6
    classification_batch_size: int = 8
    max_feed_bytes: int = 10 * 1024 * 1024
    shipment_id: str | None = None
    limit: int | None = None
    edi_profile_version: str = "exercise-generic-4010-v1"
    edi_sender_id: str = "SHIPOPS"
    edi_receiver_template: str = "CARRIER-{scac}"

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider name is required")
        if self.as_of is not None and (self.as_of.tzinfo is None or self.as_of.utcoffset() is None):
            raise ValueError("as_of must be timezone-aware")
        if self.tracking_workers < 1:
            raise ValueError("tracking_workers must be positive")
        if not 1 <= self.classification_batch_size <= 8:
            raise ValueError("classification_batch_size must be between one and eight")
        if self.max_feed_bytes < 1:
            raise ValueError("max_feed_bytes must be positive")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.edi_receiver_template.count("{scac}") != 1:
            raise ValueError("EDI receiver template must contain one {scac} placeholder")


@dataclass(frozen=True, slots=True)
class PipelineDependencies:
    enricher: Enricher
    classifier: Classifier
    edi_renderer: EdiRenderer
    escalation_store: EscalationStore
    artifact_writer_factory: ArtifactWriterFactory


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_key(shipment_id: str) -> str:
    return hashlib.sha256(shipment_id.encode()).hexdigest()[:20]


def _run_key(feed_hash: str, as_of: datetime, config: PipelineConfig) -> str:
    canonical = json.dumps(
        {
            "as_of": as_of.isoformat(),
            "classification_batch_size": config.classification_batch_size,
            "classification_policy_version": _CLASSIFICATION_POLICY_VERSION,
            "edi_profile_version": config.edi_profile_version,
            "escalation_policy_version": _ESCALATION_POLICY_VERSION,
            "feed_hash": feed_hash,
            "limit": config.limit,
            "model": config.model,
            "provider": config.provider,
            "shipment_id": config.shipment_id,
            "trigger_policy_version": _TRIGGER_POLICY_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _no_action_policy() -> PolicyDecision:
    return PolicyDecision(
        requested_disposition=FinalDisposition.NO_ACTION,
        final_disposition=FinalDisposition.NO_ACTION,
        human_review_required=False,
        verification_state=VerificationState.NOT_APPLICABLE,
    )


def _manual_policy(decision: PolicyDecision, *, code: str, message: str) -> PolicyDecision:
    return decision.model_copy(
        update={
            "final_disposition": FinalDisposition.MANUAL_REVIEW,
            "human_review_required": True,
            "verification_state": VerificationState.NOT_APPLICABLE,
            "overrides": (*decision.overrides, PolicyOverride(code=code, message=message)),
        }
    )


def _not_created(code: str, message: str) -> EscalationRecord:
    return EscalationRecord(
        status=EscalationArtifactStatus.NOT_CREATED,
        failure_code=code,
        failure_message=message,
    )


def _prepare_escalation(
    *,
    config: PipelineConfig,
    dependencies: PipelineDependencies,
    writer: ArtifactWriter,
    timeline: ShipmentTimeline,
    trigger: TriggerEvaluation,
    enrichment: EnrichmentResult,
    classification: ClassificationResult,
    decision: PolicyDecision,
    artifact_key: str,
) -> tuple[PolicyDecision, EscalationRecord, str | None]:
    try:
        draft = build_escalation_draft(
            timeline,
            trigger,
            enrichment,
            classification,
            decision,
        )
        decision_key = compute_decision_key(
            draft,
            escalation_policy_version=_ESCALATION_POLICY_VERSION,
            edi_profile_version=config.edi_profile_version,
        )
        receiver_id = config.edi_receiver_template.format(scac=draft.scac)
        reservation = dependencies.escalation_store.reserve_or_get(
            decision_key,
            profile=config.edi_profile_version,
            sender_id=config.edi_sender_id,
            receiver_id=receiver_id,
        )
        payload = dependencies.edi_renderer.render(draft, reservation.controls)
        payload_hash = hashlib.sha256(payload).hexdigest()
    except (UnrepresentableEscalation, ValueError) as exc:
        message = str(exc)[:300] or "Escalation facts cannot be represented by the EDI profile."
        return (
            _manual_policy(
                decision,
                code="EDI_UNREPRESENTABLE",
                message="Exercise EDI profile could not represent the trusted shipment facts.",
            ),
            _not_created("EDI_UNREPRESENTABLE", message),
            "edi_unrepresentable",
        )
    except RuntimeError:
        return (
            _manual_policy(
                decision,
                code="EDI_STATE_ERROR",
                message="Draft reservation failed and requires manual review.",
            ),
            _not_created("EDI_STATE_ERROR", "Draft reservation or rendering state failed."),
            "edi_state_error",
        )

    if reservation.state is ReservationState.FINALIZED:
        if reservation.artifact_path is None or reservation.artifact_hash is None:
            raise AssertionError("finalized reservation is missing artifact metadata")
        restored = writer.inspect_or_restore(
            reservation.artifact_path,
            expected_hash=reservation.artifact_hash,
            payload=payload,
        )
        if (
            restored.status is StoredArtifactStatus.CONFLICT
            or payload_hash != reservation.artifact_hash
        ):
            return (
                _manual_policy(
                    decision,
                    code="EDI_ARTIFACT_CONFLICT",
                    message="Existing logical draft bytes differ from the current rendering.",
                ),
                _not_created(
                    "EDI_ARTIFACT_CONFLICT",
                    "Existing draft hash differs; no artifact was overwritten.",
                ),
                "edi_artifact_conflict",
            )
        status = (
            EscalationArtifactStatus.REGENERATED
            if restored.status is StoredArtifactStatus.RESTORED
            else EscalationArtifactStatus.REUSED
        )
        return (
            decision,
            EscalationRecord(
                status=status,
                decision_key=decision_key,
                artifact_path=reservation.artifact_path,
                artifact_hash=reservation.artifact_hash,
                controls=reservation.controls,
            ),
            None,
        )

    relative_path = f"edi/{artifact_key}.edi"
    _, artifact_hash = writer.write_bytes(relative_path, payload)
    output_relative = writer.output_relative(relative_path)
    try:
        finalized = dependencies.escalation_store.finalize(
            reservation,
            artifact_path=output_relative,
            artifact_hash=artifact_hash,
        )
    except RuntimeError:
        return (
            _manual_policy(
                decision,
                code="EDI_FINALIZATION_ERROR",
                message="Draft bytes were not finalized in idempotency state.",
            ),
            _not_created(
                "EDI_FINALIZATION_ERROR",
                "Draft state finalization failed; operator review is required.",
            ),
            "edi_finalization_error",
        )
    return (
        decision,
        EscalationRecord(
            status=EscalationArtifactStatus.CREATED,
            decision_key=decision_key,
            artifact_path=finalized.artifact_path,
            artifact_hash=finalized.artifact_hash,
            controls=finalized.controls,
        ),
        None,
    )


def _report(summary: RunSummary) -> bytes:
    lines = [
        "# Shipment exception triage report",
        "",
        f"Run: `{summary.run_id}`",
        f"Status: **{summary.status.value}**",
        f"Observation time: `{summary.as_of.isoformat()}` ({summary.as_of_source.value})",
        "",
        "## Funnel",
        "",
        f"- Raw records: {summary.raw_records}",
        f"- Canonical events: {summary.canonical_events}",
        f"- Shipments evaluated: {summary.shipments}",
        f"- Flagged shipments: {summary.flagged}",
        f"- Enriched / feed-only: {summary.enriched} / {summary.feed_only}",
        f"- Provider / fallback classifications: "
        f"{summary.provider_classifications} / {summary.fallback_classifications}",
        f"- EDI created / reused: {summary.edi_created} / {summary.edi_reused}",
        f"- Manual review: {summary.manual_review}",
        "",
        "All EDI files are drafts for human review; nothing is transmitted.",
    ]
    if summary.degraded_reasons:
        lines.extend(("", "## Degraded reasons", ""))
        lines.extend(f"- {reason}" for reason in summary.degraded_reasons)
    return ("\n".join(lines) + "\n").encode()


def _select_timelines(
    timelines: tuple[ShipmentTimeline, ...],
    config: PipelineConfig,
) -> tuple[ShipmentTimeline, ...]:
    selected = timelines
    if config.shipment_id is not None:
        selected = tuple(
            timeline for timeline in selected if timeline.shipment_id == config.shipment_id
        )
        if not selected:
            raise PipelineInputError("requested shipment does not exist in the feed")
    if config.limit is not None:
        selected = selected[: config.limit]
    return selected


def _write_model_jsonl(
    writer: ArtifactWriter,
    relative: str,
    values: tuple[BaseModel, ...],
) -> None:
    writer.write_jsonl(relative, values)


def run_triage(
    config: PipelineConfig,
    dependencies: PipelineDependencies,
    *,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(3),
) -> TriageRunResult:
    if not config.events_path.is_file():
        raise PipelineInputError("events path must be a readable file")
    if config.events_path.stat().st_size > config.max_feed_bytes:
        raise PipelineInputError("events file exceeds the configured byte limit")
    feed_hash = _hash_file(config.events_path)
    feed = load_feed(config.events_path)
    if not feed.events:
        raise PipelineInputError("events feed contains no usable records")
    timelines = _select_timelines(build_timelines(feed.events), config)
    as_of_source = AsOfSource.EXPLICIT if config.as_of is not None else AsOfSource.FEED_MAX
    as_of = config.as_of or derive_as_of(feed.events)
    execution_at = clock().astimezone(UTC)
    nonce = nonce_factory()
    if not len(nonce) == 6 or any(char not in "0123456789abcdef" for char in nonce):
        raise ValueError("run nonce must contain exactly six lowercase hexadecimal characters")
    run_id = f"{execution_at:%Y%m%dT%H%M%SZ}-{feed_hash[:8]}-{nonce}"
    run_key = _run_key(feed_hash, as_of, config)
    writer = dependencies.artifact_writer_factory(config.output_root, run_id)
    metadata = DecisionMetadata(
        run_id=run_id,
        run_key=run_key,
        as_of=as_of,
        as_of_source=as_of_source,
        provider=config.provider,
        model=config.model,
        trigger_policy_version=_TRIGGER_POLICY_VERSION,
        classification_policy_version=_CLASSIFICATION_POLICY_VERSION,
        escalation_policy_version=_ESCALATION_POLICY_VERSION,
        edi_profile_version=config.edi_profile_version,
    )

    triggers = evaluate_timelines(timelines, as_of=as_of)
    trigger_by_id = {trigger.shipment_id: trigger for trigger in triggers}
    flagged = tuple(
        timeline for timeline in timelines if trigger_by_id[timeline.shipment_id].flagged
    )
    enrichments = enrich_shipments(
        dependencies.enricher,
        (timeline.shipment_id for timeline in flagged),
        workers=config.tracking_workers,
    )
    packs: tuple[EvidencePack, ...] = tuple(
        build_evidence_pack(
            timeline,
            trigger_by_id[timeline.shipment_id],
            enrichments[timeline.shipment_id],
        )
        for timeline in flagged
    )
    classifications: list[ClassificationResult] = []
    for start in range(0, len(packs), config.classification_batch_size):
        batch = packs[start : start + config.classification_batch_size]
        classifications.extend(dependencies.classifier.classify_batch(batch))
    if len(classifications) != len(packs):
        raise AssertionError("classifier must return one result per selected evidence pack")

    pack_by_id = {pack.shipment_id: pack for pack in packs}
    classification_by_id = {result.effective.shipment_id: result for result in classifications}
    decisions: list[ShipmentDecision] = []
    degradation: set[str] = set()
    for timeline in timelines:
        trigger = trigger_by_id[timeline.shipment_id]
        artifact_key = _artifact_key(timeline.shipment_id)
        if not trigger.flagged:
            decisions.append(
                ShipmentDecision(
                    metadata=metadata,
                    shipment_id=timeline.shipment_id,
                    artifact_key=artifact_key,
                    timeline=timeline,
                    trigger=trigger,
                    selected=False,
                    skip_reason="NOT_FLAGGED",
                    policy=_no_action_policy(),
                )
            )
            continue

        enrichment = enrichments[timeline.shipment_id]
        pack = pack_by_id[timeline.shipment_id]
        evidence_path, _ = writer.write_json(f"evidence/{artifact_key}.json", pack)
        raw_classification = classification_by_id[timeline.shipment_id]
        classification = apply_guardrails(pack, raw_classification)
        policy = decide_disposition(timeline, trigger, enrichment, classification)
        escalation = None
        if policy.final_disposition is FinalDisposition.PREPARE_CARRIER_ESCALATION:
            policy, escalation, degraded_reason = _prepare_escalation(
                config=config,
                dependencies=dependencies,
                writer=writer,
                timeline=timeline,
                trigger=trigger,
                enrichment=enrichment,
                classification=classification,
                decision=policy,
                artifact_key=artifact_key,
            )
            if degraded_reason is not None:
                degradation.add(degraded_reason)
        decisions.append(
            ShipmentDecision(
                metadata=metadata,
                shipment_id=timeline.shipment_id,
                artifact_key=artifact_key,
                timeline=timeline,
                trigger=trigger,
                selected=True,
                enrichment=enrichment,
                evidence_path=evidence_path,
                classification=classification,
                policy=policy,
                escalation=escalation,
            )
        )

    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.shipment_id))
    if feed.rejected_records:
        degradation.add("rejected_records")
    feed_only = sum(
        decision.enrichment is not None and decision.enrichment.status is not EnrichmentStatus.VALID
        for decision in ordered_decisions
    )
    if feed_only:
        degradation.add("tracking_failures")
    fallback_count = sum(
        decision.classification is not None
        and decision.classification.source is ClassificationSource.FALLBACK_RULES
        for decision in ordered_decisions
    )
    if fallback_count and config.provider != "fallback-rules":
        degradation.add("classifier_fallback")

    seen_interactions: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    for decision in ordered_decisions:
        if decision.classification is None:
            continue
        for attempt in decision.classification.attempts:
            if attempt.interaction_id is None or attempt.interaction_id in seen_interactions:
                continue
            seen_interactions.add(attempt.interaction_id)
            input_tokens += attempt.input_tokens or 0
            output_tokens += attempt.output_tokens or 0

    artifact_paths = RunArtifactPaths(
        decisions="decisions.jsonl",
        rejected_records="rejected_records.jsonl",
        report="triage_report.md",
        summary="summary.json",
    )
    summary = RunSummary(
        run_id=run_id,
        run_key=run_key,
        status=RunStatus.DEGRADED if degradation else RunStatus.COMPLETED,
        as_of=as_of,
        as_of_source=as_of_source,
        raw_records=feed.raw_record_count,
        canonical_events=len(feed.events),
        shipments=len(timelines),
        flagged=len(flagged),
        rejected_records=len(feed.rejected_records),
        enriched=len(flagged) - feed_only,
        feed_only=feed_only,
        provider_classifications=len(flagged) - fallback_count,
        fallback_classifications=fallback_count,
        provider_input_tokens=input_tokens,
        provider_output_tokens=output_tokens,
        edi_created=sum(
            decision.escalation is not None
            and decision.escalation.status is EscalationArtifactStatus.CREATED
            for decision in ordered_decisions
        ),
        edi_reused=sum(
            decision.escalation is not None
            and decision.escalation.status
            in {EscalationArtifactStatus.REUSED, EscalationArtifactStatus.REGENERATED}
            for decision in ordered_decisions
        ),
        manual_review=sum(
            decision.policy.final_disposition is FinalDisposition.MANUAL_REVIEW
            for decision in ordered_decisions
        ),
        degraded_reasons=tuple(sorted(degradation)),
        artifacts=artifact_paths,
    )
    _write_model_jsonl(
        writer,
        artifact_paths.decisions,
        tuple(ordered_decisions),
    )
    _write_model_jsonl(
        writer,
        artifact_paths.rejected_records,
        tuple(feed.rejected_records),
    )
    writer.write_bytes(artifact_paths.report, _report(summary))
    writer.write_json(artifact_paths.summary, summary)
    return TriageRunResult(summary=summary, decisions=ordered_decisions)


__all__ = [
    "PipelineConfig",
    "PipelineDependencies",
    "PipelineInputError",
    "run_triage",
]
