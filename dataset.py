# ==============================================================================
# File: dataset.py
# What this file does in plain English:
# Hey there! Think of this file as our mock banking database factory.
# In a real bank like Cred, loan applications are submitted every day.
# Instead of relying on an external website or downloading data from the internet,
# this file creates 50 realistic loan applications right here in Python.
# We use a fixed random seed (42) so that every time we run the program, we get
# the exact same 50 records - which is super handy for reliable automated testing!
# ==============================================================================

import random
from typing import List, Dict, Any

# We lock in random seed 42 so that our generated data is 100% reproducible every run.
RANDOM_SEED = 42

# The 5 realistic loan categories we support in our banking system
CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan"
]

# The 5 possible lifecycle stages a loan application can be in
STATUSES = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed"
]

# Realistic minimum and maximum loan amounts in Indian Rupees (INR) for each category
AMOUNT_RANGES = {
    "Personal Loan": (50_000, 1_500_000),
    "Home Loan": (1_500_000, 15_000_000),
    "Auto Loan": (300_000, 3_000_000),
    "Education Loan": (200_000, 5_000_000),
    "Business Loan": (500_000, 10_000_000)
}


# Function: generate_loan_dataset
# What it does:
# This function acts like an assembly line creating loan applications.
# It loops 50 times to build 50 dictionaries, assigning each one a unique ID
# (like CRD-APP-1001), a loan type, realistic money amount, aging days,
# and whether it should be flagged for extra fraud inspection.
#
# Parameters:
# - num_records: how many applications to create (defaults to 50)
# - seed: the random seed number for repeatable results (defaults to 42)
#
# Returns:
# A list of dictionaries, where each dictionary represents one loan application.
def generate_loan_dataset(num_records: int = 50, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records = []

    # Category and status weighting to make sure every category gets good representation
    cat_weights = [0.25, 0.20, 0.20, 0.15, 0.20]
    status_weights = [0.20, 0.30, 0.25, 0.15, 0.10]

    for i in range(num_records):
        record_id = f"CRD-APP-{1001 + i}"
        category = rng.choices(CATEGORIES, weights=cat_weights, k=1)[0]
        status = rng.choices(STATUSES, weights=status_weights, k=1)[0]
        
        min_amt, max_amt = AMOUNT_RANGES[category]
        # Round loan amount to the nearest 10,000 INR so the numbers look realistic
        raw_amt = rng.randint(min_amt, max_amt)
        loan_amount_inr = round(raw_amt, -4)

        days_since_created = rng.randint(0, 30)

        # Base fraud probability is 18%; slightly higher for large personal/business loans
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


# Create the master static list of 50 loan applications for the rest of our app to use
LOAN_APPLICATIONS: List[Dict[str, Any]] = generate_loan_dataset(50, RANDOM_SEED)


# Function: validate_and_report_dataset
# What it does:
# This function is like a quality inspector. It takes our generated records,
# counts how many of each category and status we have, checks our fraud percentage,
# and makes sure all the strict capstone requirements are satisfied.
#
# Parameters:
# - records: the list of loan dictionaries to inspect
#
# Returns:
# A dictionary summarizing totals, counts per category, counts per status, and fraud rate.
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
