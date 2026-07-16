from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shipment_triage.adapters.edi214 import (
    Edi214Renderer,
    EdiRenderError,
    EdiValidationError,
    Exercise214Validator,
)
from shipment_triage.domain.classification import ProblemCategory
from shipment_triage.domain.enrichment import DataCompleteness
from shipment_triage.domain.escalation import (
    EdiControlNumbers,
    EscalationCause,
    EscalationDraft,
)
from shipment_triage.domain.policy import VerificationState
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.triggers import TriggerRule

GOLDENS = Path(__file__).parent / "golden"


def _golden(name: str) -> bytes:
    return (GOLDENS / name).read_text(encoding="ascii").strip().encode("ascii")


def _draft(*, enriched: bool = True) -> EscalationDraft:
    return EscalationDraft(
        shipment_id="SHP-00003",
        carrier_shipment_id="SHP-00003",
        scac="ESTE",
        bol_number="BOL123" if enriched else None,
        po_number="PO456" if enriched else None,
        prepared_at=datetime(2026, 6, 30, 11, 0, tzinfo=UTC),
        event_at=datetime(2026, 6, 29, 4, 0, tzinfo=UTC),
        actual_status=CanonicalStatus.DEPARTED_FACILITY,
        cause=EscalationCause.WEATHER,
        city="Charlotte",
        state="NC",
        category=ProblemCategory.CARRIER_DELAY_WEATHER,
        trigger_rule=TriggerRule.PAST_PROMISE,
        idle_hours=31,
        promised_date=date(2026, 6, 24),
        data_completeness=(DataCompleteness.ENRICHED if enriched else DataCompleteness.FEED_ONLY),
        verification_state=(
            VerificationState.READY_FOR_HUMAN_REVIEW
            if enriched
            else VerificationState.DRAFT_UNVERIFIED
        ),
    )


def test_enriched_draft_matches_golden_and_profile() -> None:
    payload = Edi214Renderer().render(_draft(), EdiControlNumbers(isa=1, gs=1, st=1))

    assert payload == _golden("edi214_enriched.edi")
    Exercise214Validator().validate(payload)
    assert len(payload.decode("ascii").split("~", maxsplit=1)[0]) == 105


def test_feed_only_draft_matches_golden_and_marks_human_review() -> None:
    payload = Edi214Renderer().render(
        _draft(enriched=False),
        EdiControlNumbers(isa=2, gs=2, st=2),
    )

    assert payload == _golden("edi214_feed_only.edi")
    Exercise214Validator().validate(payload)
    assert b"K1*DATA FEED ONLY*HUMAN REVIEW REQUIRED~" in payload


def test_untrusted_text_is_transliterated_without_delimiter_injection() -> None:
    draft = _draft().model_copy(
        update={
            "bol_number": "BOL*123~>",
            "po_number": "PO*456~>",
            "city": "Sao~Paulo*Centro>",
        }
    )

    payload = Edi214Renderer().render(draft, EdiControlNumbers(isa=3, gs=3, st=3))

    Exercise214Validator().validate(payload)
    assert b"B10*SHP-00003*BOL 123*ESTE~" in payload
    assert b"L11*PO 456*PO~" in payload
    assert b"MS1*SAO PAULO CENTRO*NC~" in payload


def test_unmappable_actual_status_is_rejected_instead_of_inventing_a_code() -> None:
    draft = _draft().model_copy(update={"actual_status": CanonicalStatus.DELAYED})

    with pytest.raises(EdiRenderError, match="no AT7 status mapping"):
        Edi214Renderer().render(draft, EdiControlNumbers(isa=4, gs=4, st=4))


def test_validator_rejects_component_delimiter_inside_data() -> None:
    payload = _golden("edi214_enriched.edi").replace(b"BOL123", b"BOL>23")

    with pytest.raises(EdiValidationError, match="component delimiter"):
        Exercise214Validator().validate(payload)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ((b"SE*9*0001~", b"SE*8*0001~"), "SE count"),
        ((b"GE*1*1~", b"GE*1*9~"), "GS/GE control"),
        ((b"IEA*1*000000001~", b"IEA*1*000000009~"), "ISA/IEA control"),
    ],
)
def test_validator_detects_envelope_corruption(
    replacement: tuple[bytes, bytes],
    message: str,
) -> None:
    payload = _golden("edi214_enriched.edi").replace(*replacement)

    with pytest.raises(EdiValidationError, match=message):
        Exercise214Validator().validate(payload)
