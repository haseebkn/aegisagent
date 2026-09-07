"""Factual grounding checks for generated investigation narrative drafts.

The original guardrail screened a blacklist of hedging words. That is the wrong
control in two directions at once:

  * It is too strict. An STR documents *reasonable grounds to suspect*. Forcing
    "definitive" language pushes the model to assert as established fact things it
    cannot know, which is precisely the failure a compliance reviewer cares about.
  * It is too weak. It says nothing about whether the claims are true. Narratives
    passed the word filter while asserting a travel time between two locations the
    model was never given, and inventing arithmetic ("a velocity spike of 67% above
    the 24-hour rate") that compares a 7-day count against a 24-hour count.

This module checks the thing that actually matters: does every quantity in the
narrative trace back to the payload the model was handed, and does the narrative
avoid asserting facts about data that was never supplied?

The output is advisory and structured. Nothing here decides on its own to discard a
report -- see `docs/str-narrative-design.md` for why that matters.
"""
import re
from datetime import datetime
from dataclasses import dataclass, field

# Quantities in prose: 1,234.56 / $7,500.00 / 84.8% / -85.7476
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

# Small integers used for list numbering, and constants that name the windows and
# frameworks this report is built on (24h, 7d, 5W+H). Matching these against the
# payload would produce noise, not signal.
_STRUCTURAL_PATTERN = re.compile(
    r"(?m)^\s*\d+[.)]\s+|\b(?:24\s*[- ]?\s*h(?:our)?s?|7\s*[- ]?\s*d(?:ay)?s?|5W\+H|model\s+[234])\b",
    re.IGNORECASE,
)

# Assertions about data the pipeline does not carry. The model has no prior
# transaction, no device, no IP, no counterparty and no KYC record -- any claim
# about them is fabricated regardless of how confident it sounds.
_UNAVAILABLE_CLAIMS = [
    (r"\b(last|previous|prior|preceding)\s+(documented\s+)?(transaction|purchase|"
     r"activity|location)\b", "prior-transaction detail was not supplied"),
    (r"\b(physically )?impossible to (traverse|travel|reach)\b",
     "travel-time inference requires a prior location and timestamp that were not supplied"),
    (r"\b(ip address|device (id|fingerprint|identifier)|mac address|user agent)\b",
     "no device or network telemetry was supplied"),
    (r"\b(beneficial owner|kyc|know.your.customer|account opening|onboarding)\b",
     "no KYC or account-origination data was supplied"),
    (r"\b(other|related|linked)\s+account(s)?\b", "no counterparty or linked-account data was supplied"),
    (r"\b(chargeback|dispute|prior report|previously reported|earlier str|earlier sar)\b",
     "no case history was supplied"),
    (r"\b(customer (was|has been) contacted|confirmed with the (customer|cardholder))\b",
     "no customer contact record was supplied"),
]


@dataclass
class GroundingReport:
    grounded: list = field(default_factory=list)
    ungrounded: list = field(default_factory=list)
    unavailable_claims: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded and not self.unavailable_claims

    def summary(self) -> str:
        if self.ok:
            return f"GROUNDED: all {len(self.grounded)} quantities trace to the payload."
        parts = []
        if self.ungrounded:
            parts.append(f"{len(self.ungrounded)} ungrounded quantity/quantities: "
                         + ", ".join(str(v) for v in self.ungrounded[:8]))
        if self.unavailable_claims:
            parts.append(f"{len(self.unavailable_claims)} claim(s) about unsupplied data: "
                         + "; ".join(r for _, r in self.unavailable_claims[:4]))
        return "UNGROUNDED -- " + " | ".join(parts)


def _parse(token: str):
    cleaned = token.replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def allowed_values(txn, p_m2, p_m3, p_m4, meta_score):
    """Every quantity a narrative is entitled to state, given this payload."""
    vals = set()

    def add(v):
        if v is None:
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        vals.add(round(f, 4))
        vals.add(round(f, 2))
        vals.add(round(f, 1))
        vals.add(float(round(f)))
        vals.add(abs(round(f, 2)))

    numeric_fields = [
        'amt', 'distance_km', 'card_mean_amt', 'amt_z_card',
        'txns_24h', 'txns_7d', 'category_risk', 'night',
    ]
    for f in numeric_fields:
        add(txn.get(f))

    # The masked card suffix is the one PAN fragment a narrative may cite.
    cc = txn.get('cc_num')
    if cc is not None:
        digits = "".join(ch for ch in str(cc) if ch.isdigit())
        if len(digits) >= 4:
            add(int(digits[-4:]))

    # Model outputs, as probabilities and as percentages.
    for score in (p_m2, p_m3, p_m4, meta_score):
        add(score)
        add(score * 100)

    # Risk encodings are far more likely to be quoted as percentages.
    for f in ('category_risk',):
        v = txn.get(f)
        if v is not None:
            add(float(v) * 100)

    # Ratios a narrative may legitimately derive from two supplied figures.
    amt, mean_amt = txn.get('amt'), txn.get('card_mean_amt')
    if amt and mean_amt:
        ratio = float(amt) / float(mean_amt)
        add(ratio)
        add(ratio * 100)
        add((ratio - 1.0) * 100)          # "1,442% above the cardholder's mean"
        add(float(amt) - float(mean_amt))
    return vals


def _is_grounded(value, allowed, rel_tol=0, abs_tol=1e-6):
    for a in allowed:
        if abs(value - a) <= max(abs_tol, abs(a) * rel_tol):
            return True
    return False


def check_narrative(narrative, txn, p_m2, p_m3, p_m4, meta_score):
    """Return a GroundingReport for one generated narrative."""
    report = GroundingReport()
    if not isinstance(narrative, str) or not narrative.strip():
        report.unavailable_claims.append(("empty narrative", "no draft content was supplied"))
        return report
    allowed = allowed_values(txn, p_m2, p_m3, p_m4, meta_score)

    # Ignore numbers inside legal citations, which name statutes rather than facts.
    scrubbed = re.sub(r"(section|s\.|subsection|part|paragraph)\s*[\d.()]+",
                      " ", narrative, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\bPCMLTFA[\s,]*[\d.()]*", " PCMLTFA ", scrubbed, flags=re.IGNORECASE)

    # Match ordered dates and times, including the correct AM/PM period. A set of
    # timestamp components falsely accepted swapped month/day and morning/night.
    datetime_pattern = r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b"
    try:
        timestamp = datetime.fromisoformat(str(txn.get("trans_date_trans_time", "")))
    except ValueError:
        timestamp = None
    for match in re.findall(datetime_pattern, scrubbed, re.IGNORECASE):
        grounded = False
        if timestamp is not None:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", match):
                grounded = match == timestamp.date().isoformat()
            else:
                clock = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*([AP]M))?", match, re.IGNORECASE)
                hour, minute = int(clock[1]), int(clock[2])
                period = clock[4]
                valid_hour = 1 <= hour <= 12 if period else 0 <= hour <= 23
                if period:
                    hour = hour % 12 + (12 if period.upper() == "PM" else 0)
                grounded = (
                    valid_hour and hour == timestamp.hour and minute == timestamp.minute
                    and (clock[3] is None or int(clock[3]) == timestamp.second)
                )
        if grounded:
            report.grounded.append(match)
        else:
            report.ungrounded.append(match)
    scrubbed = re.sub(datetime_pattern, " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = _STRUCTURAL_PATTERN.sub(" ", scrubbed)

    for token in _NUMBER_RE.findall(scrubbed):
        value = _parse(token)
        if value is None:
            continue
        if _is_grounded(value, allowed):
            report.grounded.append(token)
        else:
            report.ungrounded.append(token)

    for pattern, reason in _UNAVAILABLE_CLAIMS:
        m = re.search(pattern, narrative, re.IGNORECASE)
        if m:
            report.unavailable_claims.append((m.group(0), reason))

    return report


def correction_prompt(report):
    """Feedback handed back to the model when grounding fails."""
    lines = ["The previous narrative failed factual grounding review. Correct it."]
    if report.ungrounded:
        lines.append(
            "These quantities do not appear in, and cannot be derived from, the "
            f"transaction payload you were given: {', '.join(report.ungrounded[:10])}. "
            "Remove them or replace them with figures from the payload. Do not compute "
            "comparisons between measurements taken over different time windows.")
    if report.unavailable_claims:
        for phrase, reason in report.unavailable_claims[:5]:
            lines.append(f'Remove the claim "{phrase}": {reason}.')
    lines.append(
        "State only what the supplied data supports. Where the data grounds suspicion "
        "without establishing a fact, say so plainly -- an STR documents reasonable "
        "grounds to suspect, not proven conclusions.")
    return "\n".join(lines)
