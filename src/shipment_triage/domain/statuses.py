"""Canonical shipment statuses shared by all carrier adapters."""

from enum import StrEnum


class CanonicalStatus(StrEnum):
    """Carrier-independent shipment states used by deterministic policy."""

    PICKED_UP = "PICKED_UP"
    IN_TRANSIT = "IN_TRANSIT"
    ARRIVED_FACILITY = "ARRIVED_FACILITY"
    DEPARTED_FACILITY = "DEPARTED_FACILITY"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    EXCEPTION = "EXCEPTION"
    HELD = "HELD"
    DELAYED = "DELAYED"
    DAMAGED = "DAMAGED"
    MISSED_APPOINTMENT = "MISSED_APPOINTMENT"
    UNKNOWN = "UNKNOWN"


EXCEPTION_STATUSES = frozenset(
    {
        CanonicalStatus.EXCEPTION,
        CanonicalStatus.HELD,
        CanonicalStatus.DELAYED,
        CanonicalStatus.DAMAGED,
        CanonicalStatus.MISSED_APPOINTMENT,
    }
)

__all__ = ["EXCEPTION_STATUSES", "CanonicalStatus"]
