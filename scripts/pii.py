"""PII redaction helpers.

Primary Account Numbers must never be written to disk, shipped to a third-party
model endpoint, or rendered in a dashboard in the clear. PCI-DSS permits at most
the first six and last four digits; this project only ever needs the last four to
let an investigator tie a report back to a card.
"""

_VISIBLE_SUFFIX = 4


def mask_pan(pan) -> str:
    """'9876543210987654' -> '**** **** **** 7654'."""
    if pan is None:
        return "N/A"
    digits = "".join(ch for ch in str(pan) if ch.isdigit())
    if not digits:
        return "N/A"
    if len(digits) <= _VISIBLE_SUFFIX:
        return "*" * len(digits)
    return f"**** **** **** {digits[-_VISIBLE_SUFFIX:]}"


def mask_name(first, last) -> str:
    """'Alice' 'Smith' -> 'A. Smith'. Keeps the report readable without the full identity."""
    f = str(first).strip() if first else ""
    l = str(last).strip() if last else ""
    initial = f"{f[0]}." if f else ""
    return " ".join(p for p in (initial, l) if p) or "N/A"
