
import random
from typing import List, Dict, Any

# Controlled Random Seed
RANDOM_SEED = 42

CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan"
]

STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed"
]

# Realistic INR ranges by loan category
AMOUNT_RANGES = {
    "Personal Loan": (50_000, 1_500_000),
    "Home Loan": (1_500_000, 15_000_000),
    "Auto Loan": (300_000, 3_000_000),
    "Education Loan": (200_000, 5_000_000),
    "Business Loan": (500_000, 10_000_000)
}


def generate_loan_dataset(num_records: int = 50, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []

    # Category and status weighting to ensure full vocabulary representation
    # and satisfy minimum count requirements
    cat_weights = [0.25, 0.20, 0.20, 0.15, 0.20]
    status_weights = [0.20, 0.30, 0.25, 0.15, 0.10]

    for i in range(num_records):
        record_id = f"CRD-APP-{1001 + i}"
        category = rng.choices(CATEGORIES, weights=cat_weights, k=1)[0]
        status = rng.choices(STATUSES, weights=status_weights, k=1)[0]
        
        min_amt, max_amt = AMOUNT_RANGES[category]
        # Round loan amount to nearest ₹10,000 for realistic financial figures
        raw_amt = rng.randint(min_amt, max_amt)
        loan_amount_inr = round(raw_amt, -4)

        days_since_created = rng.randint(0, 30)

        # Base fraud probability is 18%; slightly higher for high-amount loans or recent submissions
        fraud_prob = 0.18
        if category in ["Business Loan", "Personal Loan"] and loan_amount_inr > (max_amt * 0.7):
            fraud_prob = 0.25
        
        flagged_for_fraud_review = rng.random() < fraud_prob

        records.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "loan_amount_inr": loan_amount_inr,
            "days_since_created": days_since_created,
            "flagged_for_fraud_review": flagged_for_fraud_review
        })

    return records


# Generate the static master dataset
LOAN_APPLICATIONS: List[Dict[str, Any]] = generate_loan_dataset(50, RANDOM_SEED)


def validate_and_report_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_records = len(records)
    cat_counts = {cat: 0 for cat in CATEGORIES}
    status_counts = {st: 0 for st in STATUSES}
    fraud_count = 0

    for r in records:
        cat_counts[r["category"]] += 1
        status_counts[r["status"]] += 1
        if r["flagged_for_fraud_review"]:
            fraud_count += 1

    fraud_pct = (fraud_count / total_records) * 100

    print("=" * 60)
    print("LOAN APPLICATION DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Applications Generated: {total_records} (Requirement: >= 40)")
    print("\n[Category Distribution] (Requirement: every category >= 3)")
    for cat, cnt in cat_counts.items():
        print(f"  - {cat:20s}: {cnt:2d} records")
        assert cnt >= 3, f"Category {cat} has fewer than 3 records: {cnt}"

    print("\n[Status Distribution] (Requirement: every status >= 1)")
    for st, cnt in status_counts.items():
        print(f"  - {st:20s}: {cnt:2d} records")
        assert cnt >= 1, f"Status {st} has 0 records"

    print("\n[Fraud Review Breakdown] (Requirement: between 10% and 30%)")
    print(f"  - Flagged for Fraud Review: {fraud_count}/{total_records} ({fraud_pct:.2f}%)")
    assert 10.0 <= fraud_pct <= 30.0, f"Fraud rate {fraud_pct:.2f}% is outside [10%, 30%] range"

    print("\n[Sample Records (First 3)]")
    for r in records[:3]:
        print(f"  {r}")
    print("=" * 60)
    print("Status: ALL DATASET VALIDATION CONSTRAINTS PASSED SUCCESSFULLY\n")

    return {
        "total_records": total_records,
        "category_counts": cat_counts,
        "status_counts": status_counts,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct
    }


if __name__ == "__main__":
    validate_and_report_dataset(LOAN_APPLICATIONS)
