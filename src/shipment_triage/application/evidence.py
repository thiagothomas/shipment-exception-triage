"""Build bounded, allowlisted classifier evidence from trusted domain values."""

import hashlib
import json

from shipment_triage.domain.classification import (
    EvidenceEvent,
    EvidencePack,
    EvidenceTracking,
    EvidenceTrackingScan,
    EvidenceTrigger,
)
from shipment_triage.domain.enrichment import EnrichmentResult, EnrichmentStatus
from shipment_triage.domain.timelines import ShipmentTimeline
from shipment_triage.domain.triggers import TriggerEvaluation

_EMPTY_HASH = "0" * 64


def _tracking_evidence(enrichment: EnrichmentResult) -> EvidenceTracking | None:
    if enrichment.status is not EnrichmentStatus.VALID or enrichment.detail is None:
        return None
    detail = enrichment.detail
    return EvidenceTracking(
        current_status=detail.current_status,
        status_reason=detail.status_reason,
        last_event_time=detail.last_event_time,
        promised_delivery_date=detail.promised_delivery_date,
        estimated_delivery_date=detail.estimated_delivery_date,
        exception_notes=detail.exception_notes,
        scans=tuple(
            EvidenceTrackingScan(
                ref=f"tracking:scan:{index}",
                time=scan.time,
                status=scan.status,
                city=scan.city,
                state=scan.state,
            )
            for index, scan in enumerate(detail.scan_history)
        ),
    )


def build_evidence_pack(
    timeline: ShipmentTimeline,
    trigger: TriggerEvaluation,
    enrichment: EnrichmentResult,
) -> EvidencePack:
    """Create the exact redacted JSON-compatible evidence sent to a classifier."""

    if timeline.shipment_id != trigger.shipment_id:
        raise ValueError("timeline and trigger shipment IDs must match")
    if enrichment.detail is not None and enrichment.detail.shipment_id != timeline.shipment_id:
        raise ValueError("enrichment shipment ID must match timeline")

    events = tuple(
        EvidenceEvent(
            ref=event.event_id,
            occurred_at=event.occurred_at,
            status=event.status,
            raw_status=event.raw_status,
            description=event.description,
            city=event.location.city if event.location else None,
            state=event.location.state if event.location else None,
            promised_date=event.promised_date,
        )
        for event in timeline.events
    )
    triggers = tuple(
        EvidenceTrigger(
            ref=f"trigger:{fact.rule.value}",
            rule=fact.rule,
            matched=fact.matched,
            observed=fact.observed,
        )
        for fact in trigger.facts
    )
    tracking = _tracking_evidence(enrichment)
    tracking_refs = (
        (
            "tracking:current_status",
            "tracking:status_reason",
            "tracking:last_event_time",
            "tracking:exception_notes",
            *(scan.ref for scan in tracking.scans),
        )
        if tracking is not None
        else ()
    )
    allowed_refs = tuple(
        dict.fromkeys(
            (
                *(event.ref for event in events),
                *(fact.ref for fact in triggers if fact.matched),
                *tracking_refs,
            )
        )
    )
    issue_codes = tuple(
        sorted(
            {
                *(issue.code for issue in timeline.data_quality_issues),
                *(issue.code for issue in enrichment.data_quality_issues),
                *(issue.code for event in timeline.events for issue in event.data_quality_issues),
            }
        )
    )
    provisional = EvidencePack(
        evidence_hash=_EMPTY_HASH,
        shipment_id=timeline.shipment_id,
        carrier=timeline.carrier,
        terminal_state=timeline.terminal_state,
        data_completeness=enrichment.data_completeness,
        events=events,
        triggers=triggers,
        tracking=tracking,
        data_quality_issue_codes=issue_codes,
        allowed_evidence_refs=allowed_refs,
    )
    canonical = json.dumps(
        provisional.model_dump(mode="json", exclude={"evidence_hash"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return provisional.model_copy(update={"evidence_hash": hashlib.sha256(canonical).hexdigest()})


__all__ = ["build_evidence_pack"]
