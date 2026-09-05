# ==============================================================================
# File: tools.py
# What this file does in plain English:
# This file provides the specialized lending tools that our AI agent can call.
# When a customer asks: "What is the status of my loan CRD-APP-1001?", our AI agent
# uses the functions in this file to look up the application in our database,
# calculate an Underwriting Escalation Risk Score, and decide whether a human
# senior underwriter needs to intervene!
# ==============================================================================

from typing import Dict, Any, Optional
from dataset import LOAN_APPLICATIONS

# If an application's calculated risk score is 0.65 or higher, we automatically
# recommend escalating the file to a senior underwriting officer for manual review.
ESCALATION_THRESHOLD = 0.65


# Function: calculate_escalation_score
# What it does:
# This function calculates a risk score between 0.00 and 1.00 based on two things:
# 1. Is there a fraud alert on this file? (adds 0.50 points if True).
# 2. How old is this application? (adds up to 0.50 points based on how close aging is to 30 days).
#
# Parameters:
# - flagged_for_fraud: True if flagged for potential fraud, False otherwise.
# - days_since_created: How many days the loan has been sitting in review.
#
# Returns:
# A floating point number rounded to 4 decimal places (e.g. 0.6500).
def calculate_escalation_score(flagged_for_fraud: bool, days_since_created: int) -> float:
    fraud_component = 0.50 if flagged_for_fraud else 0.00
    recency_component = (min(max(days_since_created, 0), 30) / 30.0) * 0.50
    return round(fraud_component + recency_component, 4)


# Function: check_loan_application_status
# What it does:
# Think of this like the front-desk lookup computer at a bank.
# You type in an application ID (like 'CRD-APP-1001'). The function scans through
# our 50-record dataset. If found, it computes the risk score, checks if it exceeds
# the 0.65 escalation threshold, writes an explanation reason, and returns the full details.
#
# Parameters:
# - record_id: The unique application identifier string to look up.
#
# Returns:
# A dictionary containing either the loan details and escalation assessment,
# or an error dictionary stating that the record was not found.
def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    clean_id = record_id.strip().upper()
    matching_record = None

    for app in LOAN_APPLICATIONS:
        if app["record_id"].upper() == clean_id:
            matching_record = app
            break

    if not matching_record:
        return {
            "found": False,
            "record_id": record_id,
            "error": f"Loan application '{record_id}' was not found in the underwriting database."
        }

    flagged = matching_record["flagged_for_fraud_review"]
    days = matching_record["days_since_created"]
    score = calculate_escalation_score(flagged, days)
    escalation_rec = (score >= ESCALATION_THRESHOLD)

    if escalation_rec:
        if flagged:
            reason = f"Application flagged for fraud review and aging exceeds SLA ({days} days pending; score: {score:.2f} >= {ESCALATION_THRESHOLD})."
        else:
            reason = f"Application pending beyond acceptable underwriting duration ({days} days; score: {score:.2f})."
    else:
        reason = f"Application operating within standard underwriting SLAs (score: {score:.2f} < {ESCALATION_THRESHOLD})."

    return {
        "found": True,
        "record_id": matching_record["record_id"],
        "category": matching_record["category"],
        "status": matching_record["status"],
        "loan_amount_inr": matching_record["loan_amount_inr"],
        "days_since_created": days,
        "flagged_for_fraud_review": flagged,
        "escalation_score": score,
        "escalation_recommended": escalation_rec,
        "escalation_reason": reason
    }


if __name__ == "__main__":
    print("Testing check_loan_application_status:")
    sample_ids = ["CRD-APP-1001", "CRD-APP-1002", "CRD-APP-9999"]
    for sid in sample_ids:
        print(f"\nLookup for {sid}:")
        print(check_loan_application_status(sid))
