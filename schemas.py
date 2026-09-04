
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GuardrailStatus(BaseModel):
    passed: bool = Field(description="Whether the query passed guardrail inspection")
    pii_detected: bool = Field(default=False, description="Whether fixed-format PII was detected")
    masked_query: str = Field(description="Sanitized input query with fixed PII masked")
    prompt_injection_detected: bool = Field(default=False, description="Whether prompt injection was detected")
    groundedness_verified: bool = Field(default=True, description="Whether output answer is grounded in retrieved context")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection if blocked")


class LoanApplicationStatusResponse(BaseModel):
    record_id: str = Field(description="Loan application record identifier")
    category: str = Field(description="Loan category")
    status: str = Field(description="Current status of the application")
    loan_amount_inr: int = Field(description="Loan amount in INR")
    days_since_created: int = Field(description="Days since application submission")
    flagged_for_fraud_review: bool = Field(description="Fraud flag indicator")
    escalation_score: float = Field(description="Calculated escalation score between 0.0 and 1.0")
    escalation_recommended: bool = Field(description="Whether application exceeds escalation threshold")
    escalation_reason: str = Field(description="Explanation of escalation rationale")


class AgentResponse(BaseModel):
    query: str = Field(description="Original user query")
    sanitized_query: str = Field(description="PII-masked query")
    intent: str = Field(description="Detected query intent: POLICY_RAG, LOAN_STATUS, GUARDRAIL_BLOCKED, OUT_OF_SCOPE")
    answer: str = Field(description="Final agent response text")
    sources: List[str] = Field(default_factory=list, description="Source document IDs referenced")
    loan_details: Optional[LoanApplicationStatusResponse] = Field(default=None, description="Structured loan details if intent is LOAN_STATUS")
    guardrail_status: GuardrailStatus = Field(description="Input and output guardrail assessment details")
    session_id: Optional[str] = Field(default="default_session", description="Session / thread identifier")
    turn_count: int = Field(default=1, description="Current turn in conversation")
    model_used: Optional[str] = Field(default="MOCK_LLM (Deterministic)", description="Underlying model generator")


class AskRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What are the KYC documents required for a loan?"})
    session_id: Optional[str] = Field(default="session-001", json_schema_extra={"example": "session-001"})
    provider: Optional[str] = Field(default="mock", json_schema_extra={"example": "mock"})
    api_key: Optional[str] = Field(default=None, json_schema_extra={"example": None})


class LoanStatusRequest(BaseModel):
    record_id: str = Field(..., json_schema_extra={"example": "CRD-APP-1001"})
    session_id: Optional[str] = Field(default="session-001", json_schema_extra={"example": "session-001"})


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., json_schema_extra={"example": "KB-DOC-013"})
    title: str = Field(..., json_schema_extra={"example": "Digital Gold Lending Policy"})
    topic: str = Field(..., json_schema_extra={"example": "digital_gold_lending"})
    content: str = Field(..., json_schema_extra={"example": "Cred provides instant loans backed by 24K digital gold vault reserves up to 75% LTV."})
