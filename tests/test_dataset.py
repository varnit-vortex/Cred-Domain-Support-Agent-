"""
Unit Tests for Loan Application Dataset Generation
Track: Banking & FinTech (Cred)
"""

import pytest
from dataset import (
    LOAN_APPLICATIONS,
    CATEGORIES,
    STATUSES,
    AMOUNT_RANGES,
    validate_and_report_dataset,
    generate_loan_dataset
)


def test_dataset_size():
    assert len(LOAN_APPLICATIONS) >= 40, f"Expected >= 40 records, got {len(LOAN_APPLICATIONS)}"


def test_category_coverage_and_counts():
    cat_counts = {cat: 0 for cat in CATEGORIES}
    for app in LOAN_APPLICATIONS:
        assert app["category"] in CATEGORIES, f"Invalid category: {app['category']}"
        cat_counts[app["category"]] += 1

    for cat, count in cat_counts.items():
        assert count >= 3, f"Category '{cat}' has fewer than 3 records: {count}"


def test_status_coverage_and_counts():
    status_counts = {st: 0 for st in STATUSES}
    for app in LOAN_APPLICATIONS:
        assert app["status"] in STATUSES, f"Invalid status: {app['status']}"
        status_counts[app["status"]] += 1

    for st, count in status_counts.items():
        assert count >= 1, f"Status '{st}' has 0 records"


def test_fraud_review_rate_band():
    fraud_count = sum(1 for app in LOAN_APPLICATIONS if app["flagged_for_fraud_review"])
    fraud_pct = (fraud_count / len(LOAN_APPLICATIONS)) * 100.0
    assert 10.0 <= fraud_pct <= 30.0, f"Fraud review rate {fraud_pct:.2f}% is outside [10%, 30%]"


def test_loan_amount_ranges_and_recency():
    for app in LOAN_APPLICATIONS:
        cat = app["category"]
        min_amt, max_amt = AMOUNT_RANGES[cat]
        assert min_amt <= app["loan_amount_inr"] <= max_amt, f"Loan amount ₹{app['loan_amount_inr']} out of range for {cat}"
        assert 0 <= app["days_since_created"] <= 30, f"Days since created {app['days_since_created']} out of [0, 30]"


def test_dataset_validation_function():
    report = validate_and_report_dataset(LOAN_APPLICATIONS)
    assert report["total_records"] >= 40
    assert 10.0 <= report["fraud_percentage"] <= 30.0
