# ==============================================================================
# File: memory.py
# What this file does in plain English:
# Imagine chatting with a support agent who forgets what you said 10 seconds ago!
# That would be super frustrating. This file gives our AI agent a persistent memory notebook.
# Every conversation is assigned a unique `session_id`. Each time you send a message
# and the bot responds, this file records that "turn" into a JSON file stored on disk.
# That way, the bot remembers previous context and can handle multi-turn conversations!
# ==============================================================================

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Folder where conversation JSON files are stored on disk
CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")


# Class: ConversationMemory
# What it represents:
# Think of this class as a filing cabinet for customer chats.
# Each customer session gets its own clean JSON file inside `data/conversations/`.
# It provides simple methods to load past turns, save new turns, or clear history.
class ConversationMemory:

    # Method: __init__
    # Sets up the memory manager and makes sure the storage directory exists on disk.
    def __init__(self, storage_dir: str = CONVERSATIONS_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    # Method: _get_file_path
    # Cleans the session ID so it's safe to use as a filename on your computer.
    def _get_file_path(self, session_id: str) -> str:
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ("-", "_")])
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    # Method: load_history
    # Reads the JSON file for a given session and returns the list of all past turns.
    # If the session is brand new, it simply returns an empty list [].
    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._get_file_path(session_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("turns", [])
        except Exception:
            return []

    # Method: save_turn
    # Writes a brand new conversational exchange into the session's JSON file.
    # It records: turn number, current timestamp, what the user asked,
    # what the bot answered, and what intent was detected.
    # Returns the new total turn count number.
    def save_turn(
        self,
        session_id: str,
        user_query: str,
        sanitized_query: str,
        intent: str,
        answer: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        path = self._get_file_path(session_id)
        turns = self.load_history(session_id)
        
        turn_number = len(turns) + 1
        new_turn = {
            "turn_index": turn_number,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "sanitized_query": sanitized_query,
            "intent": intent,
            "answer": answer,
            "metadata": extra_metadata or {}
        }
        turns.append(new_turn)

        payload = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "turn_count": turn_number,
            "turns": turns
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return turn_number

    # Method: clear_history
    # Deletes the session JSON file from disk to reset conversation history.
    def clear_history(self, session_id: str) -> bool:
        path = self._get_file_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    # Method: get_turn_count
    # Returns how many back-and-forth turns have happened in this session so far.
    def get_turn_count(self, session_id: str) -> int:
        return len(self.load_history(session_id))


# Function: demonstrate_memory
# What it does:
# This demo function simulates two users: one having a multi-turn chat,
# and another fresh user. It proves that conversation turns are saved correctly
# and that conversations don't accidentally leak between different users!
def demonstrate_memory() -> None:
    mem = ConversationMemory()

    session_active = "session_customer_101"
    session_fresh = "session_customer_202"

    # Reset any prior state
    mem.clear_history(session_active)
    mem.clear_history(session_fresh)

    print("=" * 65)
    print("CONVERSATION MEMORY DEMONSTRATION")
    print("=" * 65)

    print("\n--- Session 1: Multi-Turn Conversation (Active Session) ---")
    
    # Turn 1
    t1 = mem.save_turn(
        session_id=session_active,
        user_query="Can you check status for my loan application CRD-APP-1001?",
        sanitized_query="Can you check status for my loan application CRD-APP-1001?",
        intent="LOAN_STATUS",
        answer="Loan CRD-APP-1001 is currently Submitted. Escalation score: 0.12.",
        extra_metadata={"record_id": "CRD-APP-1001"}
    )
    print(f"Turn {t1} Saved. History length: {mem.get_turn_count(session_active)}")

    # Turn 2
    t2 = mem.save_turn(
        session_id=session_active,
        user_query="What are the KYC documents required for this loan?",
        sanitized_query="What are the KYC documents required for this loan?",
        intent="POLICY_RAG",
        answer="According to Cred policy (KB-DOC-004): Mandatory KYC requires a valid PAN along with Aadhaar authentication.",
        extra_metadata={"context_referenced": "CRD-APP-1001"}
    )
    print(f"Turn {t2} Saved. History length: {mem.get_turn_count(session_active)}")

    active_history = mem.load_history(session_active)
    print("\nPersisted State for session_customer_101:")
    for turn in active_history:
        print(f"  [Turn {turn['turn_index']}] Query: '{turn['user_query']}' -> Intent: {turn['intent']}")

    print("\n--- Session 2: Fresh Conversation (Brand New Session) ---")
    fresh_history = mem.load_history(session_fresh)
    print(f"Persisted State for session_customer_202: {fresh_history} (Count: {len(fresh_history)})")
    assert len(fresh_history) == 0, "Fresh session unexpectedly contains history"
    print("Verification: State correctly isolated and absent in fresh conversation.")
    print("=" * 65)


if __name__ == "__main__":
    demonstrate_memory()
