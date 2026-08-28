"""
Unit Tests for RAG Core, Chunking, Embeddings, and Evaluations
Track: Banking & FinTech (Cred)
"""

import pytest
from knowledge_base import KNOWLEDGE_BASE_DOCS
from chunking import chunk_fixed_size, chunk_sentence_based, chunk_all_documents
from rag_core import CredRAGCore, calibrate_fallback_threshold
from evaluate_rag import evaluate_strategy, EVAL_QUERIES


def test_knowledge_base_doc_count_and_content():
    assert len(KNOWLEDGE_BASE_DOCS) >= 12
    required_topics = [
        "loan_eligibility_criteria", "emi_calculation_rules", "credit_card_fee_structure",
        "kyc_document_requirements", "fraud_dispute_resolution_process", "account_closure_process",
        "interest_rate_slabs", "prepayment_penalty_rules", "minimum_balance_requirements",
        "credit_score_impact_factors", "joint_account_rules", "nri_account_eligibility"
    ]
    existing_topics = [d["topic"] for d in KNOWLEDGE_BASE_DOCS]
    for req_t in required_topics:
        assert req_t in existing_topics, f"Missing required topic: {req_t}"


def test_chunking_strategies_generation():
    chunked = chunk_all_documents(KNOWLEDGE_BASE_DOCS)
    fixed = chunked["fixed_chunks"]
    sentence = chunked["sentence_chunks"]

    assert len(fixed) > 0
    assert len(sentence) > 0

    # Ensure metadata preserved
    assert "doc_id" in fixed[0]
    assert "doc_id" in sentence[0]
    assert fixed[0]["strategy"] == "fixed_size_overlap"
    assert sentence[0]["strategy"] == "sentence_based"


def test_rag_retrieval_and_threshold_separation():
    rag = CredRAGCore()
    
    # In-scope query
    in_scope_res = rag.retrieve("What is the penalty for bouncing an EMI payment?", strategy="sentence", top_k=1)
    assert len(in_scope_res) == 1
    assert in_scope_res[0]["similarity"] > 0.40
    assert in_scope_res[0]["doc_id"] == "KB-DOC-002"

    # Out-of-scope query
    out_scope_res = rag.retrieve("What is the capital of Australia?", strategy="sentence", top_k=1)
    assert len(out_scope_res) == 1
    assert out_scope_res[0]["similarity"] < 0.30


def test_grounded_generation_and_fallback():
    rag = CredRAGCore()

    # In-scope grounded answer
    in_scope_ans = rag.generate_grounded_answer("What are the rules for joint accounts?", strategy="sentence", threshold=0.31)
    assert in_scope_ans["is_grounded"] is True
    assert in_scope_ans["fallback_triggered"] is False
    assert "KB-DOC-011" in in_scope_ans["sources"]

    # Out-of-scope fallback answer
    out_scope_ans = rag.generate_grounded_answer("How to bake apple pie?", strategy="sentence", threshold=0.31)
    assert out_scope_ans["is_grounded"] is False
    assert out_scope_ans["fallback_triggered"] is True
    assert "I don't know" in out_scope_ans["answer"]


def test_chunking_evaluation_precision_and_recall():
    rag = CredRAGCore()
    eval_sentence = evaluate_strategy(rag, strategy="sentence", top_k=3)
    assert eval_sentence["avg_recall"] >= 0.80
    assert eval_sentence["avg_precision"] >= 0.50
