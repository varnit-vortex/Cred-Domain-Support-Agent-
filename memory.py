"""
Persistent Conversation Memory Manager
Track: Banking & FinTech (Cred)

Persists multi-turn conversation exchanges to JSON files on disk.
Enables conversation continuity across turns and demonstrates isolation for fresh sessions.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conversations")


class ConversationMemory:
    def __init__(self, storage_dir: str = CONVERSATIONS_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_file_path(self, session_id: str) -> str:
        safe_id = "".join([c for c in session_id if c.isalnum() or c in ("-", "_")])
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Loads the chronological turn history for a session ID.
        """
        path = self._get_file_path(session_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("turns", [])
        except Exception:
            return []

    def save_turn(
        self,
        session_id: str,
        user_query: str,
        sanitized_query: str,
        intent: str,
        answer: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Appends a new turn to the session history on disk.
        Returns the updated turn count.
        """
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

    def clear_history(self, session_id: str) -> bool:
        """
        Resets / deletes the session history.
        """
        path = self._get_file_path(session_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def get_turn_count(self, session_id: str) -> int:
        """
        Returns the current number of completed turns for a session.
        """
        return len(self.load_history(session_id))


def demonstrate_memory() -> None:
    """
    Demonstrates state carried across a multi-turn conversation and a separate fresh session.
    """
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
