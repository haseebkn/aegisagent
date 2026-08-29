"""Data-minimization and secret-redaction controls for external and log boundaries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


# Only these transaction facts may cross the external narrative-model boundary.
NARRATIVE_ALLOWLIST = frozenset(
    {
        "trans_num",
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "distance_km",
        "night",
        "card_mean_amt",
        "amt_z_card",
        "txns_24h",
        "txns_7d",
        "category_risk",
    }
)

# These direct/quasi-identifiers must never enter the external narrative prompt.
PROHIBITED_EXTERNAL_FIELDS = frozenset(
    {
        "first",
        "last",
        "gender",
        "job",
        "dob",
        "street",
        "city",
        "state",
        "zip",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|pk)_(?:test|live)_[A-Za-z0-9_-]+\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/-]+=*"),
)
_PAN_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")


def minimize_for_narrative(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict allowlist projection for an external narrative provider."""
    return {key: transaction.get(key) for key in NARRATIVE_ALLOWLIST}


def redact_sensitive_text(value: object) -> str:
    """Best-effort defense for error/log messages; never a substitute for no logging."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]",
            text,
        )
    return _PAN_PATTERN.sub("[REDACTED_PAN]", text)


def safe_error_message(exc: BaseException) -> str:
    """Expose an error category and redacted text without traceback or credentials."""
    return f"{type(exc).__name__}: {redact_sensitive_text(exc)}"
