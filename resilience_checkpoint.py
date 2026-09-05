# ==============================================================================
# File: resilience_checkpoint.py
# What this file does in plain English:
# What happens if a server crashes right in the middle of a complex multi-step workflow?
# In traditional systems, all progress is lost and you have to start over from scratch!
# This file demonstrates LangGraph's SQLite Checkpointing:
# After every single step (node), the entire state is safely saved into a local SQLite database.
# If the system is interrupted, it can resume from the exact last saved checkpoint
# without repeating steps that already completed successfully!
# ==============================================================================

import os
import sqlite3
from typing import TypedDict, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from schemas import AgentResponse, GuardrailStatus
from guardrails import apply_input_guardrails
from tools import check_loan_application_status
from rag_core import CredRAGCore

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "checkpoints.sqlite")


# Class: ResilientState
# What it represents:
# The state dictionary tracked across our checkpointer demonstration workflow.
class ResilientState(TypedDict):
    query: str
    session_id: str
    sanitized_query: str
    intent: str
    guardrail_status: Dict[str, Any]
    raw_answer: str
    sources: List[str]
    executed_nodes: List[str]


# Global execution counter tracking node call invocations
NODE_CALL_COUNTERS = {
    "node_1_input_guardrail": 0,
    "node_2_intent_router": 0,
    "node_3_rag_execution": 0,
    "node_4_output_guardrail": 0
}


# Function: node_1_input_guardrail
# Step 1: Sanitizes PII and increments call counter.
def node_1_input_guardrail(state: ResilientState) -> Dict[str, Any]:
    NODE_CALL_COUNTERS["node_1_input_guardrail"] += 1
    call_num = NODE_CALL_COUNTERS["node_1_input_guardrail"]
    print(f"  [EXEC] Node 1 (Input Guardrail) - Invocation #{call_num}")
    
    query = state.get("query", "")
    res = apply_input_guardrails(query)
    
    history = state.get("executed_nodes", []) + [f"node_1 (run #{call_num})"]
    return {
        "sanitized_query": res["masked_query"],
        "guardrail_status": res,
        "executed_nodes": history
    }


# Function: node_2_intent_router
# Step 2: Determines routing intent.
def node_2_intent_router(state: ResilientState) -> Dict[str, Any]:
    NODE_CALL_COUNTERS["node_2_intent_router"] += 1
    call_num = NODE_CALL_COUNTERS["node_2_intent_router"]
    print(f"  [EXEC] Node 2 (Intent Router) - Invocation #{call_num}")
    
    history = state.get("executed_nodes", []) + [f"node_2 (run #{call_num})"]
    return {
        "intent": "POLICY_RAG",
        "executed_nodes": history
    }


# Function: node_3_rag_execution
# Step 3: Executes RAG retrieval.
def node_3_rag_execution(state: ResilientState) -> Dict[str, Any]:
    NODE_CALL_COUNTERS["node_3_rag_execution"] += 1
    call_num = NODE_CALL_COUNTERS["node_3_rag_execution"]
    print(f"  [EXEC] Node 3 (RAG Policy Generation) - Invocation #{call_num}")
    
    rag = CredRAGCore()
    query = state.get("sanitized_query", "")
    gen_result = rag.generate_grounded_answer(query, strategy="sentence", threshold=0.31)
    
    history = state.get("executed_nodes", []) + [f"node_3 (run #{call_num})"]
    return {
        "raw_answer": gen_result["answer"],
        "sources": gen_result.get("sources", []),
        "executed_nodes": history
    }


# Function: node_4_output_guardrail
# Step 4: Final output verification.
def node_4_output_guardrail(state: ResilientState) -> Dict[str, Any]:
    NODE_CALL_COUNTERS["node_4_output_guardrail"] += 1
    call_num = NODE_CALL_COUNTERS["node_4_output_guardrail"]
    print(f"  [EXEC] Node 4 (Output Guardrail & Response Formatter) - Invocation #{call_num}")
    
    history = state.get("executed_nodes", []) + [f"node_4 (run #{call_num})"]
    return {
        "executed_nodes": history
    }


# Function: demonstrate_sqlite_checkpointing
# What it does:
# Demonstrates interruption and resumption:
# Phase 1 runs Nodes 1 & 2 and stops.
# Phase 2 resumes using the same thread ID: Nodes 1 & 2 are skipped, and Nodes 3 & 4 finish!
def demonstrate_sqlite_checkpointing(db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    # Reset counters
    for k in NODE_CALL_COUNTERS:
        NODE_CALL_COUNTERS[k] = 0

    print("=" * 80)
    print("SQLITE CHECKPOINTING & INTERRUPTION RESUMPTION DEMO")
    print("=" * 80)
    print(f"SQLite Checkpoint File: {db_path}")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # Construct Graph
    workflow = StateGraph(ResilientState)
    workflow.add_node("node_1_input_guardrail", node_1_input_guardrail)
    workflow.add_node("node_2_intent_router", node_2_intent_router)
    workflow.add_node("node_3_rag_execution", node_3_rag_execution)
    workflow.add_node("node_4_output_guardrail", node_4_output_guardrail)

    workflow.set_entry_point("node_1_input_guardrail")
    workflow.add_edge("node_1_input_guardrail", "node_2_intent_router")
    workflow.add_edge("node_2_intent_router", "node_3_rag_execution")
    workflow.add_edge("node_3_rag_execution", "node_4_output_guardrail")
    workflow.add_edge("node_4_output_guardrail", END)

    # Compile with breakpoint/interruption before node_3_rag_execution
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["node_3_rag_execution"]
    )

    thread_id = "cred_incident_thread_999"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[PHASE 1] Starting initial execution for thread_id='{thread_id}'...")
    print("Expectation: Node 1 and Node 2 execute; execution pauses before Node 3.")
    
    initial_payload: ResilientState = {
        "query": "My PAN is ABCDE1234F. What are the KYC requirements?",
        "session_id": thread_id,
        "sanitized_query": "",
        "intent": "",
        "guardrail_status": {},
        "raw_answer": "",
        "sources": [],
        "executed_nodes": []
    }

    # Run Phase 1
    state_p1 = app.invoke(initial_payload, config=config)

    # Inspect checkpointed state
    checkpoint_state = app.get_state(config)
    print("\n[CHECKPOINT INSPECTION]")
    print(f"  - Thread ID           : {thread_id}")
    print(f"  - Checkpointed Values : Sanitized Query = '{checkpoint_state.values.get('sanitized_query')}'")
    print(f"  - Executed Nodes Log  : {checkpoint_state.values.get('executed_nodes')}")
    print(f"  - Next Pending Node   : {checkpoint_state.next}")
    print(f"  - Node Call Counters  : {NODE_CALL_COUNTERS}")
    
    assert checkpoint_state.next == ("node_3_rag_execution",), "Checkpoint next node mismatch"
    assert NODE_CALL_COUNTERS["node_1_input_guardrail"] == 1
    assert NODE_CALL_COUNTERS["node_2_intent_router"] == 1
    assert NODE_CALL_COUNTERS["node_3_rag_execution"] == 0
    assert NODE_CALL_COUNTERS["node_4_output_guardrail"] == 0

    print("\n[PHASE 2] Resuming execution from checkpoint using the SAME thread_id...")
    print("Expectation: Node 1 and Node 2 are NOT re-executed; Node 3 and 4 complete.")

    # Resume Phase 2 by invoking with None
    state_p2 = app.invoke(None, config=config)

    print("\n[COMPLETION SUMMARY]")
    print(f"  - Final Executed Nodes History : {state_p2['executed_nodes']}")
    print(f"  - Final Answer                 : {state_p2['raw_answer'][:120]}...")
    print(f"  - Final Node Call Counters     : {NODE_CALL_COUNTERS}")

    assert NODE_CALL_COUNTERS["node_1_input_guardrail"] == 1, "Node 1 was re-executed!"
    assert NODE_CALL_COUNTERS["node_2_intent_router"] == 1, "Node 2 was re-executed!"
    assert NODE_CALL_COUNTERS["node_3_rag_execution"] == 1, "Node 3 did not execute!"
    assert NODE_CALL_COUNTERS["node_4_output_guardrail"] == 1, "Node 4 did not execute!"

    print("\nStatus: SQLITE CHECKPOINTING & RESUME VERIFIED WITH ZERO REDUNDANT EXECUTION\n")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_sqlite_checkpointing()
