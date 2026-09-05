# ==============================================================================
# File: tests/test_api.py
# What this file does in plain English:
# This test suite checks our FastAPI web server (server.py).
# It tests our GET /health probe, POST /ask endpoint, POST /loan-status endpoint,
# dynamic document addition, and verifies that PII in request bodies is masked
# before being written to disk logs!
# ==============================================================================

import os
import json
import pytest
from fastapi.testclient import TestClient
from server import app, LOG_FILE


# Fixture: Creates a FastAPI TestClient for simulating HTTP requests
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# Test: Checks GET /health returns 200 OK and healthy status
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["track"] == "Banking & FinTech (Cred)"


# Test: Checks POST /ask answers policy questions and returns valid AgentResponse
def test_ask_endpoint_policy_query(client):
    payload = {
        "query": "What are the KYC documents required for verification?",
        "session_id": "api_test_session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "POLICY_RAG"
    assert "KB-DOC-004" in data["sources"]
    assert "X-Trace-ID" in response.headers


# Test: Checks POST /loan-status returns structured loan details and escalation score
def test_loan_status_endpoint(client):
    payload = {
        "record_id": "CRD-APP-1001",
        "session_id": "api_test_session"
    }
    response = client.post("/loan-status", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["record_id"] == "CRD-APP-1001"
    assert 0.0 <= data["escalation_score"] <= 1.0


# Test: Checks POST /knowledge-base/documents dynamically adds and indexes new policies
def test_add_document_endpoint(client):
    payload = {
        "doc_id": "KB-DOC-TEST-99",
        "title": "FinTech Co-lending Framework",
        "topic": "co_lending_framework",
        "content": "Cred partners with premier scheduled commercial banks to offer syndicated co-lending credit lines."
    }
    response = client.post("/add-document", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["doc_id"] == "KB-DOC-TEST-99"


# Test: Checks that our middleware masks PII before writing trace logs to disk
def test_structured_logging_masks_pii(client):
    # Submit request containing raw PAN and Aadhaar
    raw_pan = "ABCDE9999Z"
    raw_aadhaar = "9999 8888 7777"
    payload = {
        "query": f"My PAN is {raw_pan} and Aadhaar is {raw_aadhaar}. Check status for CRD-APP-1001.",
        "session_id": "pii_log_session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    trace_id = response.headers.get("X-Trace-ID")
    assert trace_id is not None

    # Read log file and verify raw PII was never written to disk in the clear
    assert os.path.exists(LOG_FILE)
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
        matching_log = None
        for line in reversed(log_lines):
            entry = json.loads(line)
            if entry.get("trace_id") == trace_id:
                matching_log = entry
                break

        assert matching_log is not None
        sanitized_text = matching_log.get("sanitized_payload", "")
        # Critical verification: Raw PII NOT in log file
        assert raw_pan not in sanitized_text
        assert raw_aadhaar not in sanitized_text
        assert "[PAN_REDACTED]" in sanitized_text
        assert "[AADHAAR_REDACTED]" in sanitized_text
