"""Factual grounding checks for generated STR narratives.

The regression cases below are drawn from narratives this pipeline actually
produced -- both the hallucinations that the old word-blacklist let through, and the
legitimate phrasings that early versions of the checker wrongly rejected.
"""
from scripts.grounding import check_narrative, correction_prompt

TXN = {
    'trans_num': 'TXN_TEST',
    'cc_num': 571844099986,
    'amt': 993.55,
    'distance_km': 39.93,
    'card_mean_amt': 64.42,
    'card_std_amt': 107.48,
    'amt_z_card': 8.64,
    'txns_24h': 6,
    'txns_7d': 27,
    'category_risk': 0.015,
    'trans_date_trans_time': '2020-06-30 23:04:32',
}
SCORES = (0.9999, 0.9999, 1.0000, 0.9868)


def check(text):
    return check_narrative(text, TXN, *SCORES)


def test_payload_figures_are_grounded():
    r = check("The transaction totalled $993.55 against a mean of $64.42.")
    assert r.ok, r.summary()
    assert len(r.grounded) >= 2


def test_derived_ratio_is_grounded():
    # 993.55 / 64.42 = 15.42x, and 1442% above the mean. Both are legitimate.
    r = check("The amount is 15.42 times the cardholder mean, 1442% above it.")
    assert r.ok, r.summary()


def test_invented_figure_is_flagged():
    r = check("This represents a velocity spike of 67% above the 24-hour rate.")
    assert not r.ok
    assert any("67" in v for v in r.ungrounded)


def test_travel_time_claim_is_flagged():
    r = check("This distance is physically impossible to traverse between the "
              "cardholder's last documented transaction location and the merchant.")
    assert not r.ok
    assert len(r.unavailable_claims) >= 1


def test_linked_account_claim_is_flagged():
    r = check("Comparison against 3 linked accounts confirms layering.")
    assert not r.ok
    assert any("account" in phrase.lower() for phrase, _ in r.unavailable_claims)


def test_device_telemetry_claim_is_flagged():
    r = check("The originating IP address matches a known proxy range.")
    assert not r.ok


def test_masked_pan_suffix_is_not_a_false_positive():
    r = check("Conducted using credit card ending in 9986.")
    assert r.ok, r.summary()


def test_iso_date_is_not_parsed_as_negative_numbers():
    r = check("The transaction settled on 2020-06-30 at 23:04:32 UTC.")
    assert r.ok, r.summary()


def test_twelve_hour_clock_rendering_is_grounded():
    # 23:04 legitimately renders as 11:04 PM.
    r = check("The transaction occurred at 11:04 PM.")
    assert r.ok, r.summary()


def test_statutory_citation_is_not_treated_as_a_claim():
    r = check("Filed under the PCMLTFA, section 7, per FINTRAC guidance.")
    assert r.ok, r.summary()


def test_list_numbering_is_structural():
    r = check("1. Amount anomaly.\n2. Velocity observation.\n3. Category risk.")
    assert r.ok, r.summary()


def test_correction_prompt_names_the_problems():
    r = check("A spike of 67% versus the cardholder's previous transaction location.")
    prompt = correction_prompt(r)
    assert "67" in prompt
    assert "reasonable grounds to suspect" in prompt.lower()
