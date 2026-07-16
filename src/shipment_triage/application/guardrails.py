"""Semantic validation that constrains provider classification authority."""

from shipment_triage.domain.classification import (
    SEVERITY_ORDER,
    ClassificationResult,
    EvidencePack,
    GuardrailOverride,
    ProblemCategory,
    RecommendedAction,
    Severity,
)
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.timelines import TerminalState
from shipment_triage.domain.triggers import TriggerRule


def _trigger_ref(pack: EvidencePack, rule: TriggerRule) -> str | None:
    return next(
        (trigger.ref for trigger in pack.triggers if trigger.rule is rule and trigger.matched),
        None,
    )


def apply_guardrails(pack: EvidencePack, result: ClassificationResult) -> ClassificationResult:
    """Preserve provider output while deriving a safe effective classification."""

    if result.effective.shipment_id != pack.shipment_id:
        raise ValueError("classification and evidence shipment IDs must match")
    if result.evidence_hash != pack.evidence_hash:
        raise ValueError("classification evidence hash does not match evidence pack")

    effective = result.effective
    overrides = list(result.overrides)
    if not set(effective.evidence_refs) <= set(pack.allowed_evidence_refs):
        effective = effective.model_copy(
            update={
                "recommended_action": RecommendedAction.MANUAL_REVIEW,
                "evidence_refs": (pack.allowed_evidence_refs[0],),
                "rationale": "Invalid provider evidence references require manual review.",
            }
        )
        overrides.append(
            GuardrailOverride(
                code="INVALID_EVIDENCE_REFERENCE",
                message="Provider references were outside the evidence allowlist.",
            )
        )

    if pack.terminal_state is TerminalState.CONFLICTED:
        effective = effective.model_copy(
            update={
                "category": ProblemCategory.TERMINAL_STATUS_CONFLICT,
                "severity": Severity.CRITICAL,
                "recommended_action": RecommendedAction.MANUAL_REVIEW,
                "rationale": "Conflicting terminal states require human resolution.",
                "evidence_refs": (
                    _trigger_ref(pack, TriggerRule.TERMINAL_STATUS_CONFLICT) or pack.events[-1].ref,
                ),
            }
        )
        overrides.append(
            GuardrailOverride(
                code="TERMINAL_CONFLICT_OVERRIDE",
                message="Terminal conflict forces manual review and prohibits EDI.",
            )
        )
    elif any(event.status is CanonicalStatus.DAMAGED for event in pack.events):
        severity = (
            effective.severity
            if SEVERITY_ORDER[effective.severity] >= SEVERITY_ORDER[Severity.HIGH]
            else Severity.HIGH
        )
        action = effective.recommended_action
        if action not in {
            RecommendedAction.FILE_CLAIM_INVESTIGATION,
            RecommendedAction.MANUAL_REVIEW,
        }:
            action = RecommendedAction.FILE_CLAIM_INVESTIGATION
        if (
            effective.category is not ProblemCategory.DAMAGED_IN_TRANSIT
            or severity is not effective.severity
            or action is not effective.recommended_action
        ):
            effective = effective.model_copy(
                update={
                    "category": ProblemCategory.DAMAGED_IN_TRANSIT,
                    "severity": severity,
                    "recommended_action": action,
                    "rationale": "Explicit damage evidence requires claim investigation.",
                }
            )
            overrides.append(
                GuardrailOverride(
                    code="DAMAGE_POLICY_FLOOR",
                    message="Damage forces a high-severity claim or manual-review action.",
                )
            )
    elif _trigger_ref(pack, TriggerRule.UNKNOWN_STATUS) is not None:
        effective = effective.model_copy(
            update={
                "recommended_action": RecommendedAction.MANUAL_REVIEW,
                "rationale": "An unmapped current carrier state requires manual review.",
                "evidence_refs": (_trigger_ref(pack, TriggerRule.UNKNOWN_STATUS),),
            }
        )
        overrides.append(
            GuardrailOverride(
                code="UNKNOWN_STATUS_OVERRIDE",
                message="Unknown current status prohibits automatic escalation.",
            )
        )

    return result.model_copy(update={"effective": effective, "overrides": tuple(overrides)})


__all__ = ["apply_guardrails"]
