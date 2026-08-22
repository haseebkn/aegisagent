"""PAN and name masking. These are the controls that keep cardholder data out of
compliance logs, S3 and the Bedrock prompt, so they are tested directly."""
import pytest

from scripts.pii import mask_name, mask_pan


@pytest.mark.parametrize("pan,expected", [
    (9876543210987654, "**** **** **** 7654"),
    ("571844099986", "**** **** **** 9986"),
    ("4111-1111-1111-1111", "**** **** **** 1111"),
    ("4111 1111 1111 1111", "**** **** **** 1111"),
    (1234, "****"),
    ("12", "**"),
    (None, "N/A"),
    ("", "N/A"),
    ("no-digits-here", "N/A"),
])
def test_mask_pan(pan, expected):
    assert mask_pan(pan) == expected


def test_mask_pan_never_leaks_more_than_last_four():
    pan = "5719283746510293"
    masked = mask_pan(pan)
    assert pan not in masked
    assert pan[:-4] not in masked
    digits = [c for c in masked if c.isdigit()]
    assert len(digits) == 4
    assert "".join(digits) == pan[-4:]


@pytest.mark.parametrize("first,last,expected", [
    ("Alice", "Smith", "A. Smith"),
    ("R", "Erickson", "R. Erickson"),
    (None, "Smith", "Smith"),
    ("Alice", None, "A."),
    (None, None, "N/A"),
])
def test_mask_name(first, last, expected):
    assert mask_name(first, last) == expected
