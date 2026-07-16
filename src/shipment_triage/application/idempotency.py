"""Stable business keys for logical escalation-draft idempotency."""

import hashlib
import json

from shipment_triage.domain.escalation import EscalationDraft


def compute_decision_key(
    draft: EscalationDraft,
    *,
    escalation_policy_version: str,
    edi_profile_version: str,
) -> str:
    if not escalation_policy_version or not edi_profile_version:
        raise ValueError("decision-key policy versions are required")
    canonical = json.dumps(
        {
            "draft": draft.model_dump(mode="json"),
            "edi_profile_version": edi_profile_version,
            "escalation_policy_version": escalation_policy_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


__all__ = ["compute_decision_key"]
