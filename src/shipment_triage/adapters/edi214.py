"""Serializer and independent validator for the generic exercise X12 214 profile."""

from __future__ import annotations

import re
import string
import unicodedata
from datetime import UTC

from shipment_triage.domain.classification import ProblemCategory
from shipment_triage.domain.enrichment import DataCompleteness
from shipment_triage.domain.escalation import (
    EdiControlNumbers,
    EscalationCause,
    EscalationDraft,
)
from shipment_triage.domain.statuses import CanonicalStatus


class EdiRenderError(ValueError):
    """Draft facts cannot be represented truthfully by the exercise profile."""


class EdiValidationError(ValueError):
    """Rendered bytes violate the declared exercise profile."""


class Exercise214Profile:
    name = "exercise-generic-4010-v1"
    sender_id = "SHIPOPS"
    interchange_version = "00401"
    group_version = "004010"
    usage_indicator = "T"
    element_separator = "*"
    component_separator = ">"
    segment_terminator = "~"
    time_code = "UT"

    @staticmethod
    def receiver_id(scac: str) -> str:
        return f"CARRIER-{scac}"


_STATUS_CODES = {
    CanonicalStatus.PICKED_UP: "AF",
    CanonicalStatus.IN_TRANSIT: "X6",
    CanonicalStatus.ARRIVED_FACILITY: "X4",
    CanonicalStatus.DEPARTED_FACILITY: "P1",
    CanonicalStatus.OUT_FOR_DELIVERY: "X6",
    CanonicalStatus.DELIVERED: "D1",
    CanonicalStatus.MISSED_APPOINTMENT: "AH",
}
_REASON_CODES = {
    EscalationCause.NONE_REPORTED: "NS",
    EscalationCause.MISSED_APPOINTMENT: "A1",
    EscalationCause.MECHANICAL: "AI",
    EscalationCause.WEATHER: "AO",
}
_CATEGORY_CODES = {
    ProblemCategory.CARRIER_DELAY_WEATHER: "WX",
    ProblemCategory.CARRIER_DELAY_MECHANICAL: "MECH",
    ProblemCategory.DELIVERY_FAILED_MISSED_APPOINTMENT: "MISSED",
    ProblemCategory.STALLED_NO_SCANS: "STALL",
    ProblemCategory.SLA_BREACH_LATE: "LATE",
}
_BASIC_CHARACTERS = frozenset(string.ascii_uppercase + string.digits + " !\"&'()+,-./:;?=")


def _text(value: str, *, maximum: int, field: str, truncate: bool = False) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    translated = "".join(
        char if char.upper() in _BASIC_CHARACTERS else " " for char in ascii_value.upper()
    )
    result = re.sub(r"\s+", " ", translated).strip()
    if not result:
        raise EdiRenderError(f"{field} is empty after X12 transliteration")
    if len(result) > maximum:
        if not truncate:
            raise EdiRenderError(f"{field} exceeds the profile length of {maximum}")
        result = result[:maximum].rstrip()
    return result


def _optional_text(value: str | None, *, maximum: int, field: str) -> str:
    return "" if value is None else _text(value, maximum=maximum, field=field)


def _segment(identifier: str, *elements: str) -> str:
    return Exercise214Profile.element_separator.join((identifier, *elements))


def _remark(first: str, second: str) -> str:
    return _segment(
        "K1",
        _text(first, maximum=30, field="K101", truncate=True),
        _text(second, maximum=30, field="K102", truncate=True),
    )


class Edi214Renderer:
    """Render one stateless, human-review-only 214 draft."""

    profile = Exercise214Profile

    def render(self, draft: EscalationDraft, controls: EdiControlNumbers) -> bytes:
        status_code = _STATUS_CODES.get(draft.actual_status)
        if status_code is None:
            raise EdiRenderError(f"no AT7 status mapping for {draft.actual_status}")
        if (
            draft.cause is EscalationCause.MISSED_APPOINTMENT
            and draft.actual_status is not CanonicalStatus.MISSED_APPOINTMENT
        ):
            raise EdiRenderError("missed-appointment cause requires an attempted-delivery status")
        if (
            draft.cause in {EscalationCause.MECHANICAL, EscalationCause.WEATHER}
            and draft.actual_status is CanonicalStatus.MISSED_APPOINTMENT
        ):
            raise EdiRenderError("delay cause requires a preceding movement status")

        scac = _text(draft.scac, maximum=4, field="B1003")
        sender = _text(self.profile.sender_id, maximum=15, field="ISA06")
        receiver = _text(self.profile.receiver_id(scac), maximum=15, field="ISA08")
        shipment_id = _text(
            draft.carrier_shipment_id,
            maximum=30,
            field="B1001",
        )
        bol = _optional_text(draft.bol_number, maximum=30, field="B1002")
        po_number = _optional_text(draft.po_number, maximum=30, field="L1101")
        prepared_at = draft.prepared_at.astimezone(UTC)
        event_at = draft.event_at.astimezone(UTC)
        isa_control = f"{controls.isa:09d}"
        st_control = f"{controls.st:04d}"

        isa = _segment(
            "ISA",
            "00",
            " " * 10,
            "00",
            " " * 10,
            "ZZ",
            sender.ljust(15),
            "ZZ",
            receiver.ljust(15),
            prepared_at.strftime("%y%m%d"),
            prepared_at.strftime("%H%M"),
            "U",
            self.profile.interchange_version,
            isa_control,
            "0",
            self.profile.usage_indicator,
            self.profile.component_separator,
        )
        if len(isa) != 105:
            raise EdiRenderError("ISA segment is not the required fixed width")

        transaction = [
            _segment("ST", "214", st_control),
            _segment("B10", shipment_id, bol, scac),
        ]
        if po_number:
            transaction.append(_segment("L11", po_number, "PO"))
        transaction.extend(
            (
                _segment("LX", "1"),
                _segment(
                    "AT7",
                    status_code,
                    _REASON_CODES[draft.cause],
                    "",
                    "",
                    event_at.strftime("%Y%m%d"),
                    event_at.strftime("%H%M"),
                    self.profile.time_code,
                ),
            )
        )
        if draft.city is not None:
            city = _text(draft.city, maximum=30, field="MS101")
            state = _optional_text(draft.state, maximum=2, field="MS102")
            transaction.append(_segment("MS1", city, state))

        category_code = _CATEGORY_CODES.get(draft.category)
        if category_code is None:
            raise EdiRenderError(f"no K1 category mapping for {draft.category}")
        transaction.append(
            _remark(
                f"TRIGGER {draft.trigger_rule.value}",
                f"CATEGORY {category_code}",
            )
        )
        if draft.idle_hours is not None or draft.promised_date is not None:
            idle = f"IDLE {draft.idle_hours}H" if draft.idle_hours is not None else "IDLE UNKNOWN"
            promise = (
                f"PROMISE {draft.promised_date:%Y%m%d}"
                if draft.promised_date is not None
                else "PROMISE UNKNOWN"
            )
            transaction.append(_remark(idle, promise))
        if draft.data_completeness is DataCompleteness.FEED_ONLY:
            transaction.append(_remark("DATA FEED ONLY", "HUMAN REVIEW REQUIRED"))
        if sum(segment.startswith("K1*") for segment in transaction) > 10:
            raise EdiRenderError("exercise profile permits at most ten K1 segments")

        segment_count = len(transaction) + 1
        transaction.append(_segment("SE", str(segment_count), st_control))
        segments = [
            isa,
            _segment(
                "GS",
                "QM",
                sender,
                receiver,
                prepared_at.strftime("%Y%m%d"),
                prepared_at.strftime("%H%M"),
                str(controls.gs),
                "X",
                self.profile.group_version,
            ),
            *transaction,
            _segment("GE", "1", str(controls.gs)),
            _segment("IEA", "1", isa_control),
        ]
        rendered = self.profile.segment_terminator.join(segments) + self.profile.segment_terminator
        payload = rendered.encode("ascii")
        Exercise214Validator().validate(payload)
        return payload


class Exercise214Validator:
    """Reparse bytes without consulting renderer state and enforce the profile."""

    profile = Exercise214Profile

    def validate(self, payload: bytes) -> None:
        try:
            text = payload.decode("ascii")
        except UnicodeDecodeError as exc:
            raise EdiValidationError("payload must use ASCII X12 characters") from exc
        if not text.endswith(self.profile.segment_terminator):
            raise EdiValidationError("payload must end with a segment terminator")
        invalid = set(text) - (
            _BASIC_CHARACTERS
            | {
                self.profile.element_separator,
                self.profile.component_separator,
                self.profile.segment_terminator,
            }
        )
        if invalid:
            raise EdiValidationError("payload contains characters outside the exercise repertoire")

        raw_segments = text[:-1].split(self.profile.segment_terminator)
        segments = [segment.split(self.profile.element_separator) for segment in raw_segments]
        names = [segment[0] for segment in segments]
        for segment_index, segment in enumerate(segments):
            for element_index, element in enumerate(segment):
                is_isa_component = segment_index == 0 and element_index == 16
                if self.profile.component_separator in element and not is_isa_component:
                    raise EdiValidationError("data elements cannot contain the component delimiter")
        if len(segments) < 10 or names[:4] != ["ISA", "GS", "ST", "B10"]:
            raise EdiValidationError("required envelope and transaction segments are missing")
        if names[-3:] != ["SE", "GE", "IEA"]:
            raise EdiValidationError("transaction and envelope trailers are missing")
        if any(name == "N1" for name in names):
            raise EdiValidationError("N1 loops are outside the exercise profile")

        isa = segments[0]
        if len(raw_segments[0]) != 105 or len(isa) != 17:
            raise EdiValidationError("ISA segment does not have the fixed profile width")
        isa_lengths = (2, 10, 2, 10, 2, 15, 2, 15, 6, 4, 1, 5, 9, 1, 1, 1)
        if tuple(len(element) for element in isa[1:]) != isa_lengths:
            raise EdiValidationError("ISA elements do not have their fixed profile widths")
        if isa[5] != "ZZ" or isa[7] != "ZZ" or isa[12] != "00401" or isa[15] != "T":
            raise EdiValidationError("ISA profile constants do not match exercise-generic-4010")

        gs, st, b10 = segments[1:4]
        if len(gs) != 9 or gs[1] != "QM" or gs[8] != "004010":
            raise EdiValidationError("GS segment violates the exercise profile")
        if len(st) != 3 or st[1] != "214":
            raise EdiValidationError("ST segment must identify transaction 214")
        if (
            len(b10) != 4
            or not b10[1]
            or len(b10[1]) > 30
            or len(b10[2]) > 30
            or not b10[3]
            or len(b10[3]) > 4
        ):
            raise EdiValidationError("B10 identifiers violate the exercise profile")
        expected_receiver = self.profile.receiver_id(b10[3])
        if (
            isa[6].rstrip() != self.profile.sender_id
            or isa[8].rstrip() != expected_receiver
            or gs[2] != self.profile.sender_id
            or gs[3] != expected_receiver
        ):
            raise EdiValidationError("sender or receiver identifiers violate the exercise profile")

        cursor = 4
        if names[cursor] == "L11":
            l11 = segments[cursor]
            if len(l11) != 3 or not l11[1] or len(l11[1]) > 30 or l11[2] != "PO":
                raise EdiValidationError("L11 segment must contain a PO reference")
            cursor += 1
        if names[cursor : cursor + 2] != ["LX", "AT7"]:
            raise EdiValidationError("LX and AT7 segments are required in profile order")
        if segments[cursor] != ["LX", "1"]:
            raise EdiValidationError("exercise profile supports exactly one LX loop")
        at7 = segments[cursor + 1]
        if (
            len(at7) != 8
            or at7[1] not in _STATUS_CODES.values()
            or at7[2] not in _REASON_CODES.values()
            or at7[3:5] != ["", ""]
            or not re.fullmatch(r"\d{8}", at7[5])
            or not re.fullmatch(r"\d{4}", at7[6])
            or at7[7] != "UT"
        ):
            raise EdiValidationError("AT7 segment violates status, reason, or time rules")
        if (at7[1] == "AH") != (at7[2] == "A1"):
            raise EdiValidationError("missed-appointment status and reason must be paired")
        cursor += 2

        if names[cursor] == "MS1":
            ms1 = segments[cursor]
            if len(ms1) != 3 or not ms1[1] or len(ms1[1]) > 30 or len(ms1[2]) > 2:
                raise EdiValidationError("MS1 location violates the exercise profile")
            cursor += 1
        k1_count = 0
        while cursor < len(segments) and names[cursor] == "K1":
            k1 = segments[cursor]
            if len(k1) != 3 or not k1[1] or len(k1[1]) > 30 or len(k1[2]) > 30:
                raise EdiValidationError("K1 elements must be present and at most 30 characters")
            k1_count += 1
            cursor += 1
        if not 1 <= k1_count <= 10:
            raise EdiValidationError("exercise profile requires one to ten K1 segments")
        if cursor != len(segments) - 3:
            raise EdiValidationError("segments are outside the declared profile order")

        se, ge, iea = segments[-3:]
        if len(se) != 3 or se[2] != st[2]:
            raise EdiValidationError("ST/SE control numbers do not match")
        st_index = names.index("ST")
        se_index = names.index("SE")
        if not se[1].isdigit() or int(se[1]) != se_index - st_index + 1:
            raise EdiValidationError("SE count does not match transaction segment count")
        if len(ge) != 3 or ge[1] != "1" or ge[2] != gs[6]:
            raise EdiValidationError("GS/GE control numbers do not match")
        if len(iea) != 3 or iea[1] != "1" or iea[2] != isa[13]:
            raise EdiValidationError("ISA/IEA control numbers do not match")


__all__ = [
    "Edi214Renderer",
    "EdiRenderError",
    "EdiValidationError",
    "Exercise214Profile",
    "Exercise214Validator",
]
