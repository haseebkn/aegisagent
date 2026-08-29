import io
import json
from types import SimpleNamespace

from scripts import sar_agent
from scripts.privacy import (
    NARRATIVE_ALLOWLIST,
    PROHIBITED_EXTERNAL_FIELDS,
    minimize_for_narrative,
    redact_sensitive_text,
)


class CapturingBedrockClient:
    def __init__(self):
        self.body = None

    def invoke_model(self, *, modelId, body):
        self.body = json.loads(body)
        response = {"content": [{"text": "WHO\nMasked cardholder"}]}
        return {"body": io.BytesIO(json.dumps(response).encode())}


def sensitive_transaction():
    return {
        "trans_num": "txn-private",
        "trans_date_trans_time": "2026-01-01T00:00:00Z",
        "cc_num": "4111111111111111",
        "first": "Alice",
        "last": "Sensitive",
        "gender": "F",
        "job": "Private Occupation",
        "street": "1 Secret Street",
        "city": "Private City",
        "state": "NL",
        "zip": "A1A1A1",
        "lat": 47.5615,
        "long": -52.7126,
        "merchant": "merchant_demo",
        "category": "shopping_net",
        "merch_lat": 48.0,
        "merch_long": -53.0,
        "amt": 100.0,
        "distance_km": 4.0,
        "night": 1,
        "card_mean_amt": 40.0,
        "amt_z_card": 2.1,
        "txns_24h": 3,
        "txns_7d": 7,
        "category_risk": 0.12,
    }


def test_narrative_projection_is_a_strict_allowlist():
    minimized = minimize_for_narrative(sensitive_transaction())
    assert set(minimized) == NARRATIVE_ALLOWLIST
    assert not (set(minimized) & PROHIBITED_EXTERNAL_FIELDS)


def test_bedrock_request_excludes_direct_and_location_identifiers(monkeypatch):
    client = CapturingBedrockClient()
    monkeypatch.setattr(
        sar_agent,
        "check_narrative",
        lambda *args: SimpleNamespace(ok=True, summary=lambda: "grounded"),
    )
    sar_agent.generate_sar_narrative(
        sensitive_transaction(), 0.1, 0.2, 0.3, 0.9, client=client
    )
    serialized = json.dumps(client.body)
    for prohibited_value in (
        "Alice",
        "Sensitive",
        "Private Occupation",
        "1 Secret Street",
        "Private City",
        "A1A1A1",
        "47.5615",
        "-52.7126",
    ):
        assert prohibited_value not in serialized
    assert "1111" in serialized
    assert "4111111111111111" not in serialized


def test_log_redaction_removes_credentials_bearer_tokens_and_pan():
    message = (
        "key sk_live_supersecret Authorization: Bearer abc.def.ghi "
        "card 4111-1111-1111-1111"
    )
    redacted = redact_sensitive_text(message)
    assert "supersecret" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "4111-1111-1111-1111" not in redacted
    assert redacted.count("[REDACTED") >= 3
