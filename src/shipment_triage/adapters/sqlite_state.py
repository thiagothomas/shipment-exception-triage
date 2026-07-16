"""Transactional SQLite reservations for idempotent escalation drafts."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath

from shipment_triage.domain.escalation import (
    EdiControlNumbers,
    EscalationReservation,
    ReservationState,
)


class ReservationConflict(RuntimeError):
    """A decision key was reused with a different partner scope."""


class FinalizationConflict(RuntimeError):
    """A finalized reservation points at different artifact bytes."""


class ControlNumberExhausted(RuntimeError):
    """The nine-digit exercise control-number space is exhausted."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS escalation_drafts (
    decision_key TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    receiver_id TEXT NOT NULL,
    isa_control INTEGER NOT NULL CHECK (isa_control BETWEEN 1 AND 999999999),
    gs_control INTEGER NOT NULL CHECK (gs_control BETWEEN 1 AND 999999999),
    st_control INTEGER NOT NULL CHECK (st_control BETWEEN 1 AND 999999999),
    state TEXT NOT NULL CHECK (state IN ('PENDING', 'FINALIZED')),
    artifact_path TEXT,
    artifact_hash TEXT,
    UNIQUE (profile, sender_id, receiver_id, isa_control)
)
"""


def _safe_artifact_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError("artifact path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must stay below the run directory")
    return path.as_posix()


def _reservation(row: sqlite3.Row) -> EscalationReservation:
    return EscalationReservation(
        decision_key=row["decision_key"],
        profile=row["profile"],
        sender_id=row["sender_id"],
        receiver_id=row["receiver_id"],
        controls=EdiControlNumbers(
            isa=row["isa_control"],
            gs=row["gs_control"],
            st=row["st_control"],
        ),
        state=ReservationState(row["state"]),
        artifact_path=row["artifact_path"],
        artifact_hash=row["artifact_hash"],
    )


class SqliteEscalationStore:
    """Reserve control numbers once per logical draft using one SQLite table."""

    def __init__(self, database: str | Path, *, timeout_seconds: float = 5.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("SQLite timeout must be positive")
        self._database = Path(database)
        self._timeout_seconds = timeout_seconds
        self._database.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._database.is_symlink():
            raise ValueError("SQLite state path cannot be a symbolic link")
        with closing(self._connect()) as connection:
            connection.execute(_SCHEMA)
        self._database.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database,
            timeout=self._timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}")
        return connection

    def reserve_or_get(
        self,
        decision_key: str,
        *,
        profile: str,
        sender_id: str,
        receiver_id: str,
    ) -> EscalationReservation:
        request = EscalationReservation(
            decision_key=decision_key,
            profile=profile,
            sender_id=sender_id,
            receiver_id=receiver_id,
            controls=EdiControlNumbers(isa=1, gs=1, st=1),
            state=ReservationState.PENDING,
        )
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM escalation_drafts WHERE decision_key = ?",
                    (request.decision_key,),
                ).fetchone()
                if row is not None:
                    existing = _reservation(row)
                    if (
                        existing.profile,
                        existing.sender_id,
                        existing.receiver_id,
                    ) != (request.profile, request.sender_id, request.receiver_id):
                        raise ReservationConflict(
                            "decision key already exists under a different partner scope"
                        )
                    connection.commit()
                    return existing

                maximum = connection.execute(
                    """
                    SELECT COALESCE(MAX(isa_control), 0)
                    FROM escalation_drafts
                    WHERE profile = ? AND sender_id = ? AND receiver_id = ?
                    """,
                    (request.profile, request.sender_id, request.receiver_id),
                ).fetchone()[0]
                control = int(maximum) + 1
                if control > 999_999_999:
                    raise ControlNumberExhausted("EDI control-number space is exhausted")
                connection.execute(
                    """
                    INSERT INTO escalation_drafts (
                        decision_key, profile, sender_id, receiver_id,
                        isa_control, gs_control, st_control, state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING')
                    """,
                    (
                        request.decision_key,
                        request.profile,
                        request.sender_id,
                        request.receiver_id,
                        control,
                        control,
                        control,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM escalation_drafts WHERE decision_key = ?",
                    (request.decision_key,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("reserved escalation row could not be read back")
                connection.commit()
                return _reservation(row)
            except Exception:
                connection.rollback()
                raise

    def finalize(
        self,
        reservation: EscalationReservation,
        *,
        artifact_path: str,
        artifact_hash: str,
    ) -> EscalationReservation:
        safe_path = _safe_artifact_path(artifact_path)
        if len(artifact_hash) != 64 or any(
            char not in "0123456789abcdef" for char in artifact_hash
        ):
            raise ValueError("artifact hash must be a lowercase SHA-256 digest")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM escalation_drafts WHERE decision_key = ?",
                    (reservation.decision_key,),
                ).fetchone()
                if row is None:
                    raise ReservationConflict("reservation does not exist")
                existing = _reservation(row)
                if (
                    existing.profile,
                    existing.sender_id,
                    existing.receiver_id,
                    existing.controls,
                ) != (
                    reservation.profile,
                    reservation.sender_id,
                    reservation.receiver_id,
                    reservation.controls,
                ):
                    raise ReservationConflict("reservation details do not match stored values")
                if existing.state is ReservationState.FINALIZED:
                    if (
                        existing.artifact_path == safe_path
                        and existing.artifact_hash == artifact_hash
                    ):
                        connection.commit()
                        return existing
                    raise FinalizationConflict(
                        "reservation is already finalized with a different artifact"
                    )

                connection.execute(
                    """
                    UPDATE escalation_drafts
                    SET state = 'FINALIZED', artifact_path = ?, artifact_hash = ?
                    WHERE decision_key = ? AND state = 'PENDING'
                    """,
                    (safe_path, artifact_hash, reservation.decision_key),
                )
                row = connection.execute(
                    "SELECT * FROM escalation_drafts WHERE decision_key = ?",
                    (reservation.decision_key,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("finalized escalation row could not be read back")
                connection.commit()
                return _reservation(row)
            except Exception:
                connection.rollback()
                raise


__all__ = [
    "ControlNumberExhausted",
    "FinalizationConflict",
    "ReservationConflict",
    "SqliteEscalationStore",
]
