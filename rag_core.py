
import os
import shutil
from typing import List, Dict, Any, Tuple
import chromadb
import numpy as np

# Enforce local offline huggingface usage to avoid network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

from knowledge_base import KNOWLEDGE_BASE_DOCS
from chunking import chunk_all_documents

# Persisted ChromaDB directory
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chroma_db")

# Singleton embedding model
_EMBEDDING_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        except Exception:
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


class CredRAGCore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, force_reindex: bool = False):
        self.persist_dir = persist_dir
        self.model = get_embedding_model()
        
        if force_reindex and os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.fixed_collection = self.client.get_or_create_collection(
            name="cred_fixed_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.sentence_collection = self.client.get_or_create_collection(
            name="cred_sentence_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # Index documents if empty
        if self.fixed_collection.count() == 0 or self.sentence_collection.count() == 0:
            self.index_knowledge_base()

    def index_knowledge_base(self) -> None:
        chunked = chunk_all_documents(KNOWLEDGE_BASE_DOCS)
        fixed_chunks = chunked["fixed_chunks"]
        sentence_chunks = chunked["sentence_chunks"]

        # Index fixed-size chunks
        if fixed_chunks:
            texts = [c["text"] for c in fixed_chunks]
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            ids = [c["chunk_id"] for c in fixed_chunks]
            metadatas = [{
                "doc_id": c["doc_id"],
                "title": c["title"],
                "topic": c["topic"],
                "strategy": c["strategy"],
                "chunk_index": c["chunk_index"]
            } for c in fixed_chunks]
            self.fixed_collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        # Index sentence chunks
        if sentence_chunks:
            texts = [c["text"] for c in sentence_chunks]
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            ids = [c["chunk_id"] for c in sentence_chunks]
            metadatas = [{
                "doc_id": c["doc_id"],
                "title": c["title"],
                "topic": c["topic"],
                "strategy": c["strategy"],
                "chunk_index": c["chunk_index"]
            } for c in sentence_chunks]
            self.sentence_collection.upsert(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

    def retrieve(
        self,
        query: str,
        strategy: str = "sentence",
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        collection = self.sentence_collection if strategy == "sentence" else self.fixed_collection
        query_embedding = self.model.encode([query], convert_to_numpy=True).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            ids = results["ids"][0]

            for doc_text, meta, dist, cid in zip(docs, metas, dists, ids):
                # Cosine distance in Chroma is 1 - cosine_similarity
                similarity = max(0.0, min(1.0, 1.0 - dist))
                retrieved.append({
                    "chunk_id": cid,
                    "text": doc_text,
                    "doc_id": meta.get("doc_id", "UNKNOWN"),
                    "title": meta.get("title", ""),
                    "topic": meta.get("topic", ""),
                    "strategy": meta.get("strategy", strategy),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": dist,
                    "similarity": round(similarity, 4)
                })

        return retrieved

    def generate_grounded_answer(
        self,
        query: str,
        strategy: str = "sentence",
        top_k: int = 3,
        threshold: float = 0.40
    ) -> Dict[str, Any]:
        chunks = self.retrieve(query, strategy=strategy, top_k=top_k)
        
        top_similarity = chunks[0]["similarity"] if chunks else 0.0

        # Empirical threshold check
        if not chunks or top_similarity < threshold:
            return {
                "query": query,
                "answer": "I don't know: The requested information is not covered in Cred's lending and banking policy guidelines.",
                "is_grounded": False,
                "fallback_triggered": True,
                "top_similarity": top_similarity,
                "threshold": threshold,
                "retrieved_chunks": chunks,
                "sources": []
            }

        # Grounded synthesis strictly using retrieved policy sentences
        relevant_texts = [c["text"] for c in chunks if c["similarity"] >= threshold]
        sources = list(dict.fromkeys([c["doc_id"] for c in chunks if c["similarity"] >= threshold]))

        answer_body = " ".join(relevant_texts)
        answer = f"According to Cred policy ({', '.join(sources)}): {answer_body}"

        return {
            "query": query,
            "answer": answer,
            "is_grounded": True,
            "fallback_triggered": False,
            "top_similarity": top_similarity,
            "threshold": threshold,
            "retrieved_chunks": chunks,
            "sources": sources
        }


def calibrate_fallback_threshold() -> Dict[str, Any]:
    rag = CredRAGCore(force_reindex=True)

    in_scope_queries = [
        "What is the penalty for bouncing an EMI payment?",
        "What are the KYC documents required for personal loan verification?",
        "What are the rules and penalties for loan prepayment or foreclosure?",
        "What is the minimum balance required in metro branch savings accounts?",
        "How does Cred handle unauthorized fraudulent transactions?"
    ]

    out_of_scope_queries = [
        "What is the traditional recipe for Hyderabadi mutton biryani?",
        "How do I tune an acoustic guitar in open D tuning?",
        "Who won the 2022 FIFA World Cup final in Qatar?"
    ]

    print("=" * 70)
    print("EMPIRICAL FALLBACK THRESHOLD CALIBRATION")
    print("=" * 70)

    print("\n[In-Scope Queries (Policy Relevant)]")
    in_scope_scores = []
    for q in in_scope_queries:
        res = rag.retrieve(q, strategy="sentence", top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        in_scope_scores.append(sim)
        print(f"  - Query: '{q}'")
        print(f"    Top Match: [{res[0]['doc_id']}] Similarity = {sim:.4f}")

    print("\n[Out-of-Scope Queries (Irrelevant / Unrelated)]")
    out_of_scope_scores = []
    for q in out_of_scope_queries:
        res = rag.retrieve(q, strategy="sentence", top_k=1)
        sim = res[0]["similarity"] if res else 0.0
        out_of_scope_scores.append(sim)
        print(f"  - Query: '{q}'")
        print(f"    Top Match: [{res[0]['doc_id']}] Similarity = {sim:.4f}")

    min_in_scope = min(in_scope_scores)
    max_out_of_scope = max(out_of_scope_scores)
    
    # Choose optimal threshold midpoint between highest out-of-scope and lowest in-scope
    calibrated_threshold = round((min_in_scope + max_out_of_scope) / 2.0, 2)
    if calibrated_threshold <= max_out_of_scope:
        calibrated_threshold = round(max_out_of_scope + 0.05, 2)

    print("\n[Empirical Calibration Summary]")
    print(f"  - In-Scope Similarity Range    : [{min(in_scope_scores):.4f}, {max(in_scope_scores):.4f}] (Mean: {sum(in_scope_scores)/len(in_scope_scores):.4f})")
    print(f"  - Out-of-Scope Similarity Range: [{min(out_of_scope_scores):.4f}, {max(out_of_scope_scores):.4f}] (Mean: {sum(out_of_scope_scores)/len(out_of_scope_scores):.4f})")
    print(f"  - Separation Margin            : {min_in_scope - max_out_of_scope:.4f}")
    print(f"  - Calibrated Optimal Threshold : {calibrated_threshold}")
    print("=" * 70)

    # Demonstrate 5 in-scope queries + 1 out-of-scope query triggering fallback
    print("\n[Grounded Generation Demonstration with Calibrated Threshold]")
    demo_queries = [
        "What are the eligibility criteria for a Personal Loan?",
        "What fee is charged if an EMI deduction bounces?",
        "How are credit card annual fees waived?",
        "What is the penalty for foreclosing a floating rate home loan?",
        "What is the difference between NRE and NRO accounts?",
        "What is the capital city of France?"  # Out-of-scope
    ]

    for dq in demo_queries:
        gen = rag.generate_grounded_answer(dq, strategy="sentence", threshold=calibrated_threshold)
        status_tag = "FALLBACK TRIGGERED ('I don't know')" if gen["fallback_triggered"] else "GROUNDED ANSWER"
        print(f"\nQuery: '{dq}'")
        print(f"Status: [{status_tag}] (Top Sim: {gen['top_similarity']:.4f} vs Threshold: {calibrated_threshold})")
        print(f"Answer: {gen['answer']}")

    return {
        "in_scope_scores": in_scope_scores,
        "out_of_scope_scores": out_of_scope_scores,
        "min_in_scope": min_in_scope,
        "max_out_of_scope": max_out_of_scope,
        "calibrated_threshold": calibrated_threshold
    }


if __name__ == "__main__":
    calibrate_fallback_threshold()
