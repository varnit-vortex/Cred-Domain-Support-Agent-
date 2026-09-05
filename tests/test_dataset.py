# ==============================================================================
# File: tests/test_dataset.py
# What this file does in plain English:
# This test suite verifies our mock dataset generator (dataset.py).
# It checks that our factory produces at least 40 records (we produce 50),
# covers all 5 loan categories, includes all 5 status stages, and keeps the
# fraud review rate strictly within the calibrated 10% to 30% band.
# ==============================================================================

import pytest
from dataset import (
    generate_loan_dataset,
    validate_and_report_dataset,
    LOAN_APPLICATIONS,
    CATEGORIES,
    STATUSES,
    AMOUNT_RANGES
)


# Test: Verifies total dataset size is at least 40 records (we generate 50)
def test_dataset_size():
    assert len(LOAN_APPLICATIONS) >= 40
    assert len(LOAN_APPLICATIONS) == 50


# Test: Verifies all 5 loan categories are represented with at least 3 records each
def test_category_coverage_and_counts():
    counts = {cat: 0 for cat in CATEGORIES}
    for app in LOAN_APPLICATIONS:
        counts[app["category"]] += 1

    for cat, cnt in counts.items():
        assert cnt >= 3, f"Category '{cat}' has fewer than 3 records: {cnt}"


# Test: Verifies all 5 statuses (Submitted, Approved, etc.) appear at least once
def test_status_coverage_and_counts():
    counts = {st: 0 for st in STATUSES}
    for app in LOAN_APPLICATIONS:
        counts[app["status"]] += 1

    for st, cnt in counts.items():
        assert cnt >= 1, f"Status '{st}' has 0 records"


# Test: Verifies fraud review rate stays in the required 10% to 30% band (calibrated 14%)
def test_fraud_review_rate_band():
    total = len(LOAN_APPLICATIONS)
    fraud_count = sum(1 for a in LOAN_APPLICATIONS if a["flagged_for_fraud_review"])
    fraud_pct = (fraud_count / total) * 100

    assert 10.0 <= fraud_pct <= 30.0, f"Fraud rate {fraud_pct:.2f}% is outside [10%, 30%]"
    assert fraud_count == 7
    assert round(fraud_pct, 2) == 14.00


# Test: Verifies loan amounts fall within realistic financial bounds and aging is 0-30 days
def test_loan_amount_ranges_and_recency():
    for app in LOAN_APPLICATIONS:
        cat = app["category"]
        amt = app["loan_amount_inr"]
        min_allowed, max_allowed = AMOUNT_RANGES[cat]
        assert min_allowed <= amt <= max_allowed, f"Amount {amt} out of range for {cat}"
        assert 0 <= app["days_since_created"] <= 30


# Test: Verifies the validation report helper function runs and passes assertions
def test_dataset_validation_function():
    summary = validate_and_report_dataset(LOAN_APPLICATIONS)
    assert summary["total_records"] == 50
    assert summary["fraud_count"] == 7
