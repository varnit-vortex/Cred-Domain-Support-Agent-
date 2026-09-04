
from typing import Dict, Any, Optional
from dataset import LOAN_APPLICATIONS

# Escalation threshold calibrated against dataset distribution:
# Mean days_since_created is 15.0 days; 80th percentile is ~24 days.
# Non-fraud cases max out at 0.50; fraud cases reach >= 0.65 when days_since_created > 9 days.
ESCALATION_THRESHOLD = 0.65


def calculate_escalation_score(flagged_for_fraud: bool, days_since_created: int) -> float:
    fraud_component = 0.50 if flagged_for_fraud else 0.00
    recency_component = (min(max(days_since_created, 0), 30) / 30.0) * 0.50
    return round(fraud_component + recency_component, 4)


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
