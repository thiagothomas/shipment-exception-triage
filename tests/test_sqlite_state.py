import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from shipment_triage.adapters.sqlite_state import (
    FinalizationConflict,
    ReservationConflict,
    SqliteEscalationStore,
)
from shipment_triage.application.idempotency import compute_decision_key
from shipment_triage.domain.classification import ProblemCategory
from shipment_triage.domain.enrichment import DataCompleteness
from shipment_triage.domain.escalation import EscalationCause, EscalationDraft, ReservationState
from shipment_triage.domain.policy import VerificationState
from shipment_triage.domain.statuses import CanonicalStatus
from shipment_triage.domain.triggers import TriggerRule


def _draft() -> EscalationDraft:
    return EscalationDraft(
        shipment_id="SHP-00003",
        carrier_shipment_id="SHP-00003",
        scac="ESTE",
        bol_number="BOL123",
        po_number="PO456",
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
        data_completeness=DataCompleteness.ENRICHED,
        verification_state=VerificationState.READY_FOR_HUMAN_REVIEW,
    )


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_decision_key_is_stable_and_versioned() -> None:
    first = compute_decision_key(
        _draft(),
        escalation_policy_version="1",
        edi_profile_version="exercise-generic-4010-v1",
    )
    repeated = compute_decision_key(
        _draft(),
        escalation_policy_version="1",
        edi_profile_version="exercise-generic-4010-v1",
    )
    changed = compute_decision_key(
        _draft(),
        escalation_policy_version="2",
        edi_profile_version="exercise-generic-4010-v1",
    )

    assert first == repeated
    assert first != changed


def test_reserve_or_get_reuses_controls_and_persists_across_instances(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    store = SqliteEscalationStore(database)

    first = store.reserve_or_get(
        _key("same"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )
    repeated = SqliteEscalationStore(database).reserve_or_get(
        _key("same"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )
    next_reservation = store.reserve_or_get(
        _key("next"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )

    assert first == repeated
    assert first.controls.isa == 1
    assert next_reservation.controls.isa == 2
    assert first.state is ReservationState.PENDING


def test_finalize_is_idempotent_but_refuses_mismatched_artifacts(tmp_path: Path) -> None:
    store = SqliteEscalationStore(tmp_path / "state.sqlite3")
    reservation = store.reserve_or_get(
        _key("finalize"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )

    finalized = store.finalize(
        reservation,
        artifact_path="edi/abc123.edi",
        artifact_hash="a" * 64,
    )
    repeated = store.finalize(
        reservation,
        artifact_path="edi/abc123.edi",
        artifact_hash="a" * 64,
    )

    assert finalized == repeated
    assert finalized.state is ReservationState.FINALIZED
    with pytest.raises(FinalizationConflict):
        store.finalize(
            reservation,
            artifact_path="edi/abc123.edi",
            artifact_hash="b" * 64,
        )


def test_same_decision_key_cannot_change_partner_scope(tmp_path: Path) -> None:
    store = SqliteEscalationStore(tmp_path / "state.sqlite3")
    decision_key = _key("scope")
    store.reserve_or_get(
        decision_key,
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )

    with pytest.raises(ReservationConflict):
        store.reserve_or_get(
            decision_key,
            profile="exercise-generic-4010-v1",
            sender_id="SHIPOPS",
            receiver_id="CARRIER-UPSN",
        )


def test_concurrent_reservations_allocate_unique_controls(tmp_path: Path) -> None:
    store = SqliteEscalationStore(tmp_path / "state.sqlite3")

    def reserve(index: int) -> int:
        return store.reserve_or_get(
            _key(f"concurrent-{index}"),
            profile="exercise-generic-4010-v1",
            sender_id="SHIPOPS",
            receiver_id="CARRIER-ESTE",
        ).controls.isa

    with ThreadPoolExecutor(max_workers=8) as executor:
        controls = tuple(executor.map(reserve, range(8)))

    assert sorted(controls) == list(range(1, 9))


def test_control_sequence_is_scoped_by_receiver(tmp_path: Path) -> None:
    store = SqliteEscalationStore(tmp_path / "state.sqlite3")

    first = store.reserve_or_get(
        _key("este"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-ESTE",
    )
    other = store.reserve_or_get(
        _key("upsn"),
        profile="exercise-generic-4010-v1",
        sender_id="SHIPOPS",
        receiver_id="CARRIER-UPSN",
    )

    assert first.controls.isa == other.controls.isa == 1
