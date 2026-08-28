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
from dataclasses import dataclass, field

# Quantities in prose: 1,234.56 / $7,500.00 / 84.8% / -85.7476
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

# Small integers used for list numbering, and constants that name the windows and
# frameworks this report is built on (24h, 7d, 5W+H). Matching these against the
# payload would produce noise, not signal.
_STRUCTURAL_VALUES = set(range(0, 11)) | {12, 24, 7, 5, 100, 1000}

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
        'amt', 'distance_km', 'card_mean_amt', 'card_std_amt', 'amt_z_card',
        'amt_over_mean_card', 'txns_24h', 'txns_7d', 'category_risk', 'state_risk',
        'merchant_risk', 'card_txn_cnt', 'lat', 'long', 'merch_lat', 'merch_long',
        'zip', 'city_pop', 'hour', 'day_of_week', 'night', 'is_online', 'log_amt',
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
    for f in ('category_risk', 'state_risk', 'merchant_risk'):
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
    std = txn.get('card_std_amt')
    if amt and mean_amt and std:
        add((float(amt) - float(mean_amt)) / float(std))
    v24, v7 = txn.get('txns_24h'), txn.get('txns_7d')
    if v24 and v7:
        add(float(v7) / float(v24))
        add(float(v7) - float(v24))

    # Date and time components from the transaction timestamp.
    ts = txn.get('trans_date_trans_time')
    if ts is not None:
        for part in re.findall(r"\d+", str(ts)):
            add(int(part))

    return vals


def _is_grounded(value, allowed, rel_tol=0.01, abs_tol=0.01):
    for a in allowed:
        if abs(value - a) <= max(abs_tol, abs(a) * rel_tol):
            return True
    return False


def check_narrative(narrative, txn, p_m2, p_m3, p_m4, meta_score):
    """Return a GroundingReport for one generated narrative."""
    report = GroundingReport()
    allowed = allowed_values(txn, p_m2, p_m3, p_m4, meta_score)

    # Ignore numbers inside legal citations, which name statutes rather than facts.
    scrubbed = re.sub(r"(section|s\.|subsection|part|paragraph)\s*[\d.()]+",
                      " ", narrative, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\bPCMLTFA[\s,]*[\d.()]*", " PCMLTFA ", scrubbed, flags=re.IGNORECASE)

    # Dates and clock times are structural. Verify their components against the
    # payload timestamp, then remove them, so the hyphens in an ISO date are not
    # parsed as minus signs on the following number.
    _DATETIME_RE = r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?"
    ts_parts = {int(x) for x in re.findall(r"\d+", str(txn.get('trans_date_trans_time', '')))}
    # Narratives routinely render a 24-hour timestamp on a 12-hour clock, so 23:04
    # legitimately appears as "11:04 PM".
    ts_parts |= {h % 12 or 12 for h in list(ts_parts) if 0 <= h <= 23}
    for match in re.findall(_DATETIME_RE, scrubbed):
        components = {int(x) for x in re.findall(r"\d+", match)}
        if components <= ts_parts:
            report.grounded.append(match)
        else:
            report.ungrounded.append(match)
    scrubbed = re.sub(_DATETIME_RE, " ", scrubbed)

    for token in _NUMBER_RE.findall(scrubbed):
        value = _parse(token)
        if value is None:
            continue
        if value in _STRUCTURAL_VALUES and float(value).is_integer():
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
