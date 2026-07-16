"""Deterministic classification fallback over the same evidence contract."""

from collections.abc import Iterator, Sequence

from shipment_triage.domain.classification import (
    Classification,
    ClassificationResult,
    ClassificationSource,
    EvidencePack,
    ProblemCategory,
    RecommendedAction,
    Severity,
)
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.timelines import TerminalState
from shipment_triage.domain.triggers import TriggerRule

_PROMPT_VERSION = "triage-v1"
_SCHEMA_VERSION = "1"


def _trigger_ref(pack: EvidencePack, rule: TriggerRule) -> str | None:
    return next(
        (trigger.ref for trigger in pack.triggers if trigger.rule is rule and trigger.matched),
        None,
    )


def _text_evidence(pack: EvidencePack) -> Iterator[tuple[str, str]]:
    if pack.tracking is not None:
        if pack.tracking.status_reason:
            yield "tracking:status_reason", pack.tracking.status_reason
        if pack.tracking.exception_notes:
            yield "tracking:exception_notes", pack.tracking.exception_notes
        yield "tracking:current_status", pack.tracking.current_status
        for scan in reversed(pack.tracking.scans):
            yield scan.ref, scan.status
    for event in reversed(pack.events):
        yield event.ref, f"{event.raw_status} {event.description or ''}"


def _matching_text_ref(
    pack: EvidencePack,
    *,
    all_terms: tuple[str, ...],
    any_terms: tuple[str, ...] = (),
) -> str | None:
    for ref, value in _text_evidence(pack):
        searchable = value.lower()
        if all(term in searchable for term in all_terms) and (
            not any_terms or any(term in searchable for term in any_terms)
        ):
            return ref
    return None


def _status_ref(pack: EvidencePack, status: CanonicalStatus) -> str | None:
    return next((event.ref for event in reversed(pack.events) if event.status is status), None)


def _result(
    pack: EvidencePack,
    *,
    category: ProblemCategory,
    severity: Severity,
    action: RecommendedAction,
    evidence_ref: str,
    rationale: str,
) -> Classification:
    return Classification(
        shipment_id=pack.shipment_id,
        category=category,
        severity=severity,
        recommended_action=action,
        confidence=1.0,
        rationale=rationale,
        evidence_refs=(evidence_ref,),
    )


def _classification(pack: EvidencePack) -> Classification:
    if pack.terminal_state is TerminalState.CONFLICTED:
        return _result(
            pack,
            category=ProblemCategory.TERMINAL_STATUS_CONFLICT,
            severity=Severity.CRITICAL,
            action=RecommendedAction.MANUAL_REVIEW,
            evidence_ref=(
                _trigger_ref(pack, TriggerRule.TERMINAL_STATUS_CONFLICT) or pack.events[-1].ref
            ),
            rationale="Delivered and non-delivered terminal evidence conflicts.",
        )

    damage_ref = _status_ref(pack, CanonicalStatus.DAMAGED) or _matching_text_ref(
        pack,
        all_terms=("damage",),
    )
    if damage_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.DAMAGED_IN_TRANSIT,
            severity=Severity.HIGH,
            action=RecommendedAction.FILE_CLAIM_INVESTIGATION,
            evidence_ref=damage_ref,
            rationale="Carrier evidence explicitly reports in-transit damage.",
        )

    missed_ref = _status_ref(
        pack,
        CanonicalStatus.MISSED_APPOINTMENT,
    ) or _matching_text_ref(pack, all_terms=("missed", "appointment"))
    if missed_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT,
            severity=Severity.HIGH,
            action=RecommendedAction.ESCALATE_TO_CARRIER,
            evidence_ref=missed_ref,
            rationale="Carrier evidence reports a missed delivery appointment.",
        )

    weather_ref = _matching_text_ref(pack, all_terms=("weather",))
    if weather_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.CARRIER_DELAY_WEATHER,
            severity=Severity.HIGH,
            action=RecommendedAction.ESCALATE_TO_CARRIER,
            evidence_ref=weather_ref,
            rationale="Carrier evidence attributes the delay to weather.",
        )

    mechanical_ref = _matching_text_ref(pack, all_terms=("mechanical",))
    if mechanical_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.CARRIER_DELAY_MECHANICAL,
            severity=Severity.HIGH,
            action=RecommendedAction.ESCALATE_TO_CARRIER,
            evidence_ref=mechanical_ref,
            rationale="Carrier evidence attributes the delay to a mechanical issue.",
        )

    consignee_ref = _matching_text_ref(
        pack,
        all_terms=("consignee",),
        any_terms=("closed", "unavailable"),
    )
    if consignee_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.HELD_CONSIGNEE_UNAVAILABLE,
            severity=Severity.MEDIUM,
            action=RecommendedAction.CONTACT_CONSIGNEE,
            evidence_ref=consignee_ref,
            rationale="Carrier evidence indicates the consignee is unavailable.",
        )

    stall_ref = _trigger_ref(pack, TriggerRule.STALLED)
    if stall_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.STALLED_NO_SCANS,
            severity=(
                Severity.HIGH if _trigger_ref(pack, TriggerRule.PAST_PROMISE) else Severity.MEDIUM
            ),
            action=RecommendedAction.ESCALATE_TO_CARRIER,
            evidence_ref=stall_ref,
            rationale="The shipment has no recent carrier scan beyond the configured threshold.",
        )

    promise_ref = _trigger_ref(pack, TriggerRule.PAST_PROMISE)
    if promise_ref is not None:
        return _result(
            pack,
            category=ProblemCategory.SLA_BREACH_LATE,
            severity=Severity.HIGH,
            action=RecommendedAction.ESCALATE_TO_CARRIER,
            evidence_ref=promise_ref,
            rationale="The shipment remains unresolved after its promised delivery date.",
        )

    return _result(
        pack,
        category=ProblemCategory.OTHER_EXCEPTION,
        severity=Severity.MEDIUM,
        action=RecommendedAction.MANUAL_REVIEW,
        evidence_ref=_trigger_ref(pack, TriggerRule.EXCEPTION_STATUS) or pack.events[-1].ref,
        rationale="Carrier evidence indicates an exception without a more specific safe mapping.",
    )


class RuleBasedClassifier:
    """Classify every pack locally when the provider cannot be trusted or reached."""

    def classify_batch(self, packs: Sequence[EvidencePack]) -> tuple[ClassificationResult, ...]:
        return tuple(
            ClassificationResult(
                provider_output=None,
                effective=_classification(pack),
                source=ClassificationSource.FALLBACK_RULES,
                provider=None,
                model=None,
                prompt_version=_PROMPT_VERSION,
                schema_version=_SCHEMA_VERSION,
                evidence_hash=pack.evidence_hash,
                attempts=(),
            )
            for pack in packs
        )


__all__ = ["RuleBasedClassifier"]
