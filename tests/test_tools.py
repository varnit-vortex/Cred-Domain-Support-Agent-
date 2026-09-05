# ==============================================================================
# File: tests/test_tools.py
# What this file does in plain English:
# This test suite checks our loan underwriting tools (tools.py).
# It verifies that our escalation score formula correctly combines aging and fraud flags,
# that valid application IDs return complete details, and that invalid IDs return
# a polite error message rather than crashing!
# ==============================================================================

import pytest
from tools import calculate_escalation_score, check_loan_application_status, ESCALATION_THRESHOLD


# Test: Verifies escalation formula calculations against known mathematical cases
def test_escalation_score_formula():
    assert calculate_escalation_score(flagged_for_fraud=False, days_since_created=0) == 0.0000
    assert calculate_escalation_score(flagged_for_fraud=False, days_since_created=15) == 0.2500
    assert calculate_escalation_score(flagged_for_fraud=False, days_since_created=30) == 0.5000
    assert calculate_escalation_score(flagged_for_fraud=True, days_since_created=0) == 0.5000
    assert calculate_escalation_score(flagged_for_fraud=True, days_since_created=15) == 0.7500
    assert calculate_escalation_score(flagged_for_fraud=True, days_since_created=30) == 1.0000


# Test: Verifies lookup returns full record details when an ID exists
def test_valid_loan_lookup():
    res = check_loan_application_status("CRD-APP-1001")
    assert res["found"] is True
    assert res["record_id"] == "CRD-APP-1001"
    assert "status" in res
    assert "loan_amount_inr" in res
    assert "escalation_score" in res
    assert "escalation_recommended" in res


# Test: Verifies lookup handles non-existent IDs gracefully
def test_invalid_loan_lookup():
    res = check_loan_application_status("CRD-APP-9999")
    assert res["found"] is False
    assert "error" in res
