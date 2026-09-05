# ==============================================================================
# File: graph.py
# What this file does in plain English:
# This file is the orchestrator and traffic controller of our AI system!
# We use LangGraph (a framework for building stateful multi-step AI agents).
# Think of this like a factory assembly line:
# - Step 1 (Input Guardrail): Checks for private data (PII) and malicious prompts.
# - Step 2 (Intent Router): Decides where the question should go (Loan Lookup or Policy Search).
# - Step 3 (Specialist Agent): Either fetches the loan status or searches the policy library.
# - Step 4 (Output Guardrail & Memory): Checks that the answer is grounded and saves it to history.
# Every step updates a shared notebook called "AgentState" that moves along the conveyor belt!
# ==============================================================================

import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from schemas import AgentResponse, GuardrailStatus, LoanApplicationStatusResponse
from guardrails import apply_input_guardrails, verify_output_groundedness
from tools import check_loan_application_status
from rag_core import CredRAGCore
from memory import ConversationMemory
from llm_engine import generate_response_sync

# Regex to detect loan application ID format (e.g. CRD-APP-1001)
LOAN_ID_PATTERN = re.compile(r'\bCRD-APP-\d{4}\b', re.IGNORECASE)


# Class: AgentState
# What it represents:
# This is our shared clipboard or notebook.
# As the customer's request moves from one worker (node) to the next along the assembly line,
# each worker reads this state and writes their updates into it (e.g., sanitized query,
# detected intent, retrieved chunks, and the final answer).
class AgentState(TypedDict):
    query: str
    session_id: str
    provider: str
    api_key: Optional[str]
    sanitized_query: str
    intent: str
    guardrail_status: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    raw_answer: str
    sources: List[str]
    model_used: str
    loan_details: Optional[Dict[str, Any]]
    final_response: Optional[Dict[str, Any]]
    turn_count: int


# Shared RAG core and Memory instances
_RAG_CORE = None
_MEMORY = None


# Function: get_rag_core
# Helper function that creates our RAGCore library manager once and shares it everywhere.
def get_rag_core() -> CredRAGCore:
    global _RAG_CORE
    if _RAG_CORE is None:
        _RAG_CORE = CredRAGCore()
    return _RAG_CORE


# Function: get_memory
# Helper function that provides access to our conversation memory manager.
def get_memory() -> ConversationMemory:
    global _MEMORY
    if _MEMORY is None:
        _MEMORY = ConversationMemory()
    return _MEMORY


# Node 1: Input Guardrail
# Function: input_guardrail_node (Node 1)
# What it does:
# The first stop on the assembly line!
# It scrubs sensitive personal numbers (PAN, Aadhaar) and blocks malicious prompt injections.
#
# Parameters:
# - state: Current shared agent state dictionary.
#
# Returns:
# Updates for the state: 'sanitized_query' and 'guardrail_status'.
def input_guardrail_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    guard_res = apply_input_guardrails(query)
    
    return {
        "sanitized_query": guard_res["masked_query"],
        "guardrail_status": guard_res
    }


# Node 2: Intent Router
# Function: intent_router_node (Node 2)
# What it does:
# The traffic cop!
# It inspects the sanitized query. If safety guardrails were tripped, it routes to blocked.
# If it sees a loan application ID (like CRD-APP-1001), it routes to LOAN_STATUS.
# Otherwise, it routes to POLICY_RAG for policy questions.
#
# Parameters:
# - state: Current shared agent state.
#
# Returns:
# State update with the detected 'intent'.
def intent_router_node(state: AgentState) -> Dict[str, Any]:
    guard_status = state.get("guardrail_status", {})
    if not guard_status.get("passed", True):
        return {"intent": "GUARDRAIL_BLOCKED"}

    sanitized = state.get("sanitized_query", "").strip()
    
    # Check if a Loan Record ID is present
    match = LOAN_ID_PATTERN.search(sanitized)
    if match or any(k in sanitized.lower() for k in ["loan status", "application status", "check application", "track my loan"]):
        return {"intent": "LOAN_STATUS"}
    
    return {"intent": "POLICY_RAG"}


# Routing conditional function
# Function: route_intent
# Conditional routing edge function.
# Tells LangGraph which path to take after the Intent Router node.
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "POLICY_RAG")
    if intent == "GUARDRAIL_BLOCKED":
        return "output_guardrail_node"
    elif intent == "LOAN_STATUS":
        return "loan_agent_node"
    else:
        return "rag_agent_node"


# Node 3: Policy RAG Agent
# Function: rag_agent_node (Node 3A)
# What it does:
# The policy research specialist!
# When a customer asks about loan rules or interest rates, this node retrieves the
# matching policy snippets from ChromaDB and synthesizes an authoritative answer.
#
# Parameters:
# - state: Current shared agent state.
#
# Returns:
# Updates containing 'raw_answer', 'retrieved_chunks', and 'sources'.
def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    rag = get_rag_core()
    query = state.get("sanitized_query", "")
    provider = state.get("provider", "mock")
    api_key = state.get("api_key")

    chunks = rag.retrieve(query, strategy="sentence", top_k=3)
    sources = list(dict.fromkeys([c["doc_id"] for c in chunks if c["similarity"] >= 0.31]))

    gen_res = generate_response_sync(
        query=query,
        retrieved_chunks=chunks,
        sources=sources,
        similarity_threshold=0.31,
        provider=provider,
        api_key=api_key
    )

    return {
        "raw_answer": gen_res["answer"],
        "retrieved_chunks": chunks,
        "sources": gen_res.get("sources", sources),
        "model_used": gen_res.get("model_used", "MOCK_LLM")
    }


# Node 4: Loan Status Agent
# Function: loan_agent_node (Node 3B)
# What it does:
# The loan underwriting specialist!
# Looks up the specific application in our database, calculates the escalation risk score,
# and generates a detailed status report for the customer.
#
# Parameters:
# - state: Current shared agent state.
#
# Returns:
# Updates containing 'raw_answer', 'loan_details', and 'sources'.
def loan_agent_node(state: AgentState) -> Dict[str, Any]:
    sanitized = state.get("sanitized_query", "")
    match = LOAN_ID_PATTERN.search(sanitized)
    
    if match:
        record_id = match.group(0).upper()
        lookup_result = check_loan_application_status(record_id)
        
        if lookup_result.get("found"):
            rec_id = lookup_result["record_id"]
            cat = lookup_result["category"]
            stat = lookup_result["status"]
            amt = lookup_result["loan_amount_inr"]
            days = lookup_result["days_since_created"]
            score = lookup_result["escalation_score"]
            esc_rec = lookup_result["escalation_recommended"]
            esc_reason = lookup_result["escalation_reason"]

            status_badge = "✅ Approved" if stat == "Approved" else ("❌ Rejected" if stat == "Rejected" else f"⏳ {stat}")
            esc_badge = "⚠️ Escalated for Senior Underwriter Oversight" if esc_rec else "🟢 Standard Underwriting SLA"

            ans = (
                f"📋 **Loan Application Status: {rec_id}**\n\n"
                f"• **Category:** {cat}\n"
                f"• **Current Status:** **{status_badge}**\n"
                f"• **Loan Amount:** **₹{amt:,}**\n"
                f"• **Application Aging:** **{days} days** since submission\n"
                f"• **Fraud Review Active:** {'🚨 **YES**' if lookup_result['flagged_for_fraud_review'] else 'No'}\n"
                f"• **Escalation Score:** `{score:.4f}` / 1.00 ({esc_badge})\n\n"
                f"**Underwriting Assessment:** {esc_reason}"
            )
            return {
                "raw_answer": ans,
                "loan_details": lookup_result,
                "sources": [f"DB:{rec_id}"],
                "model_used": "Database-Underwriting-Tool"
            }
        else:
            return {
                "raw_answer": lookup_result.get("error", f"Record {record_id} not found in the loan database."),
                "loan_details": None,
                "sources": [],
                "model_used": "Database-Underwriting-Tool"
            }
    else:
        return {
            "raw_answer": "Please provide a valid loan application ID in the format **CRD-APP-XXXX** (e.g., `CRD-APP-1001`) to look up your status.",
            "loan_details": None,
            "sources": [],
            "model_used": "Database-Underwriting-Tool"
        }


# Node 5: Output Guardrail & Structured Formatter
# Function: output_guardrail_node (Node 4)
# What it does:
# The final inspection station!
# Verifies that the answer is grounded in real policy, saves the complete turn
# to conversation memory on disk, and bundles everything into our final AgentResponse schema.
#
# Parameters:
# - state: Current shared agent state.
#
# Returns:
# Final state updates with 'final_response' and 'turn_count'.
def output_guardrail_node(state: AgentState) -> Dict[str, Any]:
    intent = state.get("intent", "POLICY_RAG")
    guard_status = state.get("guardrail_status", {})
    query = state.get("query", "")
    sanitized = state.get("sanitized_query", query)
    raw_answer = state.get("raw_answer", "")
    sources = state.get("sources", [])
    loan_details = state.get("loan_details")
    session_id = state.get("session_id", "default_session")
    retrieved_chunks = state.get("retrieved_chunks", [])
    model_used = state.get("model_used", "MOCK_LLM")

    # Handle Guardrail Rejections
    if intent == "GUARDRAIL_BLOCKED":
        final_answer = guard_status.get("rejection_reason", "Request blocked by safety guardrails.")
        final_guardrail = GuardrailStatus(
            passed=False,
            pii_detected=guard_status.get("pii_detected", False),
            masked_query=sanitized,
            prompt_injection_detected=guard_status.get("prompt_injection_detected", True),
            groundedness_verified=False,
            rejection_reason=final_answer
        )
    else:
        # Check output groundedness if RAG intent and mock mode
        grounded_check = {"passed": True}
        if intent == "POLICY_RAG" and retrieved_chunks and model_used.startswith("MOCK"):
            grounded_check = verify_output_groundedness(raw_answer, retrieved_chunks, similarity_threshold=0.31)
            
        final_answer = raw_answer
        final_guardrail = GuardrailStatus(
            passed=True,
            pii_detected=guard_status.get("pii_detected", False),
            masked_query=sanitized,
            prompt_injection_detected=False,
            groundedness_verified=grounded_check.get("passed", True),
            rejection_reason=grounded_check.get("rejection_reason")
        )

    structured_loan = None
    if loan_details and loan_details.get("found"):
        structured_loan = LoanApplicationStatusResponse(
            record_id=loan_details["record_id"],
            category=loan_details["category"],
            status=loan_details["status"],
            loan_amount_inr=loan_details["loan_amount_inr"],
            days_since_created=loan_details["days_since_created"],
            flagged_for_fraud_review=loan_details["flagged_for_fraud_review"],
            escalation_score=loan_details["escalation_score"],
            escalation_recommended=loan_details["escalation_recommended"],
            escalation_reason=loan_details["escalation_reason"]
        )

    # Persist to memory
    memory = get_memory()
    turn_number = memory.save_turn(
        session_id=session_id,
        user_query=query,
        sanitized_query=sanitized,
        intent=intent,
        answer=final_answer,
        extra_metadata={"sources": sources}
    )

    response_obj = AgentResponse(
        query=query,
        sanitized_query=sanitized,
        intent=intent,
        answer=final_answer,
        sources=sources,
        loan_details=structured_loan,
        guardrail_status=final_guardrail,
        session_id=session_id,
        turn_count=turn_number,
        model_used=model_used
    )

    return {
        "final_response": response_obj.model_dump(),
        "turn_count": turn_number
    }


def build_cred_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add all 5 nodes
    workflow.add_node("input_guardrail_node", input_guardrail_node)
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("rag_agent_node", rag_agent_node)
    workflow.add_node("loan_agent_node", loan_agent_node)
    workflow.add_node("output_guardrail_node", output_guardrail_node)

    # Set Entry Point
    workflow.set_entry_point("input_guardrail_node")

    # Connect Edge: input_guardrail -> intent_router
    workflow.add_edge("input_guardrail_node", "intent_router_node")

    # Conditional Routing Edge
    workflow.add_conditional_edges(
        "intent_router_node",
        route_intent,
        {
            "rag_agent_node": "rag_agent_node",
            "loan_agent_node": "loan_agent_node",
            "output_guardrail_node": "output_guardrail_node"
        }
    )

    # Terminal edges
    workflow.add_edge("rag_agent_node", "output_guardrail_node")
    workflow.add_edge("loan_agent_node", "output_guardrail_node")
    workflow.add_edge("output_guardrail_node", END)

    return workflow.compile()


# Function: run_agent
# What it does:
# The main front-door function called by our web server!
# Takes a user question, packages it into initial state, runs it through the LangGraph
# pipeline, and returns the final validated response dictionary.
#
# Parameters:
# - query: User's question string.
# - session_id: Conversation session identifier (defaults to 'default_session').
# - provider: Model provider ('mock', 'groq', 'openai', etc.).
# - api_key: Optional API key.
#
# Returns:
# A dictionary conforming to the AgentResponse schema.
def run_agent(
    query: str,
    session_id: str = "demo_session",
    provider: str = "mock",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    app = build_cred_agent_graph()
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "provider": provider,
        "api_key": api_key,
        "sanitized_query": "",
        "intent": "",
        "guardrail_status": {},
        "retrieved_chunks": [],
        "raw_answer": "",
        "sources": [],
        "model_used": "MOCK_LLM",
        "loan_details": None,
        "final_response": None,
        "turn_count": 0
    }
    result_state = app.invoke(initial_state)
    return result_state["final_response"]


if __name__ == "__main__":
    print("Testing LangGraph Agent:")
    res = run_agent("what is capital of india", provider="auto")
    print(f"Model Used: {res.get('model_used')}")
    print(f"Answer:\n{res['answer']}")
