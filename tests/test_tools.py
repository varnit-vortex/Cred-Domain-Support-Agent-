
import pytest
from tools import check_loan_application_status, calculate_escalation_score, ESCALATION_THRESHOLD
from dataset import LOAN_APPLICATIONS


def test_escalation_score_formula():
    # Fraud = False, Days = 0 -> 0.00
    assert calculate_escalation_score(False, 0) == 0.00
    # Fraud = False, Days = 30 -> 0.50
    assert calculate_escalation_score(False, 30) == 0.50
    # Fraud = True, Days = 0 -> 0.50
    assert calculate_escalation_score(True, 0) == 0.50
    # Fraud = True, Days = 30 -> 1.00
    assert calculate_escalation_score(True, 30) == 1.00
    # Fraud = True, Days = 10 -> 0.50 + 0.1667 = 0.6667 (exceeds 0.65 threshold)
    assert calculate_escalation_score(True, 10) > ESCALATION_THRESHOLD


def test_valid_loan_lookup():
    first_app = LOAN_APPLICATIONS[0]
    rec_id = first_app["record_id"]
    res = check_loan_application_status(rec_id)

    assert res["found"] is True
    assert res["record_id"] == rec_id
    assert res["category"] == first_app["category"]
    assert res["status"] == first_app["status"]
    assert res["loan_amount_inr"] == first_app["loan_amount_inr"]
    assert 0.0 <= res["escalation_score"] <= 1.0


def test_invalid_loan_lookup():
    res = check_loan_application_status("CRD-APP-99999")
    assert res["found"] is False
    assert "error" in res
