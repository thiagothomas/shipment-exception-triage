"""Deterministic classification fallback over the same evidence contract."""

from collections.abc import Sequence

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


def _first_event_ref(pack: EvidencePack, *terms: str) -> str:
    for event in reversed(pack.events):
        searchable = f"{event.raw_status} {event.description or ''}".lower()
        if any(term in searchable for term in terms):
            return event.ref
    return pack.events[-1].ref


def _text(pack: EvidencePack) -> str:
    event_text = " ".join(f"{event.raw_status} {event.description or ''}" for event in pack.events)
    tracking_text = ""
    if pack.tracking is not None:
        tracking_text = " ".join(
            (
                pack.tracking.current_status,
                pack.tracking.status_reason or "",
                pack.tracking.exception_notes or "",
            )
        )
    return f"{event_text} {tracking_text}".lower()


def _classification(pack: EvidencePack) -> Classification:
    searchable = _text(pack)
    refs: tuple[str, ...]

    if pack.terminal_state is TerminalState.CONFLICTED:
        category = ProblemCategory.TERMINAL_STATUS_CONFLICT
        severity = Severity.CRITICAL
        action = RecommendedAction.MANUAL_REVIEW
        refs = (_trigger_ref(pack, TriggerRule.TERMINAL_STATUS_CONFLICT) or pack.events[-1].ref,)
        rationale = "Delivered and non-delivered terminal evidence conflicts."
    elif any(event.status is CanonicalStatus.DAMAGED for event in pack.events):
        category = ProblemCategory.DAMAGED_IN_TRANSIT
        severity = Severity.HIGH
        action = RecommendedAction.FILE_CLAIM_INVESTIGATION
        refs = (_first_event_ref(pack, "damage"),)
        rationale = "Carrier evidence explicitly reports in-transit damage."
    elif "missed" in searchable and "appointment" in searchable:
        category = ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT
        severity = Severity.HIGH
        action = RecommendedAction.ESCALATE_TO_CARRIER
        refs = (_first_event_ref(pack, "missed"),)
        rationale = "Carrier evidence reports a missed delivery appointment."
    elif "weather" in searchable:
        category = ProblemCategory.CARRIER_DELAY_WEATHER
        severity = Severity.HIGH
        action = RecommendedAction.ESCALATE_TO_CARRIER
        refs = (_first_event_ref(pack, "weather"),)
        rationale = "Carrier evidence attributes the delay to weather."
    elif "mechanical" in searchable:
        category = ProblemCategory.CARRIER_DELAY_MECHANICAL
        severity = Severity.HIGH
        action = RecommendedAction.ESCALATE_TO_CARRIER
        refs = (_first_event_ref(pack, "mechanical"),)
        rationale = "Carrier evidence attributes the delay to a mechanical issue."
    elif "consignee" in searchable and ("closed" in searchable or "unavailable" in searchable):
        category = ProblemCategory.HELD_CONSIGNEE_UNAVAILABLE
        severity = Severity.MEDIUM
        action = RecommendedAction.CONTACT_CONSIGNEE
        refs = (_first_event_ref(pack, "consignee", "closed", "unavailable"),)
        rationale = "Carrier evidence indicates the consignee is unavailable."
    elif (stall_ref := _trigger_ref(pack, TriggerRule.STALLED)) is not None:
        category = ProblemCategory.STALLED_NO_SCANS
        severity = (
            Severity.HIGH if _trigger_ref(pack, TriggerRule.PAST_PROMISE) else Severity.MEDIUM
        )
        action = RecommendedAction.ESCALATE_TO_CARRIER
        refs = (stall_ref,)
        rationale = "The shipment has no recent carrier scan beyond the configured threshold."
    elif (promise_ref := _trigger_ref(pack, TriggerRule.PAST_PROMISE)) is not None:
        category = ProblemCategory.SLA_BREACH_LATE
        severity = Severity.HIGH
        action = RecommendedAction.ESCALATE_TO_CARRIER
        refs = (promise_ref,)
        rationale = "The shipment remains unresolved after its promised delivery date."
    else:
        category = ProblemCategory.OTHER_EXCEPTION
        severity = Severity.MEDIUM
        action = RecommendedAction.MANUAL_REVIEW
        refs = (_trigger_ref(pack, TriggerRule.EXCEPTION_STATUS) or pack.events[-1].ref,)
        rationale = "Carrier evidence indicates an exception without a more specific safe mapping."

    return Classification(
        shipment_id=pack.shipment_id,
        category=category,
        severity=severity,
        recommended_action=action,
        confidence=1.0,
        rationale=rationale,
        evidence_refs=refs,
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
