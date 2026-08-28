"""
Chunking Strategies for Cred Knowledge Base
Track: Banking & FinTech (Cred)

Implements two distinct chunking strategies:
1. Fixed-size chunking with overlap (character-based with token/word awareness)
2. Sentence-based chunking (natural linguistic sentence boundaries)
"""

import re
from typing import List, Dict, Any


def chunk_fixed_size(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
    doc_meta: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Splits text into fixed-size character windows with a defined overlap.
    """
    if doc_meta is None:
        doc_meta = {}

    chunks = []
    text_len = len(text)
    start = 0
    chunk_idx = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunk_id = f"{doc_meta.get('doc_id', 'DOC')}_FIXED_{chunk_idx:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "doc_id": doc_meta.get("doc_id", "UNKNOWN"),
                "title": doc_meta.get("title", ""),
                "topic": doc_meta.get("topic", ""),
                "strategy": "fixed_size_overlap",
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": end
            })
            chunk_idx += 1

        if end >= text_len:
            break
        start += (chunk_size - overlap)

    return chunks


def chunk_sentence_based(
    text: str,
    doc_meta: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Splits text into natural linguistic sentences.
    """
    if doc_meta is None:
        doc_meta = {}

    # Regular expression splitting on sentence terminators (. ! ?) followed by whitespace
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    chunk_idx = 0

    for sentence in raw_sentences:
        clean_sentence = sentence.strip()
        if clean_sentence:
            chunk_id = f"{doc_meta.get('doc_id', 'DOC')}_SENT_{chunk_idx:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": clean_sentence,
                "doc_id": doc_meta.get("doc_id", "UNKNOWN"),
                "title": doc_meta.get("title", ""),
                "topic": doc_meta.get("topic", ""),
                "strategy": "sentence_based",
                "chunk_index": chunk_idx
            })
            chunk_idx += 1

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Chunks all input documents across both strategies.
    Returns a dictionary with 'fixed_chunks' and 'sentence_chunks'.
    """
    fixed_chunks: List[Dict[str, Any]] = []
    sentence_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        meta = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "topic": doc["topic"]
        }
        # Fixed size chunks
        f_chunks = chunk_fixed_size(doc["content"], chunk_size=220, overlap=40, doc_meta=meta)
        fixed_chunks.extend(f_chunks)

        # Sentence based chunks
        s_chunks = chunk_sentence_based(doc["content"], doc_meta=meta)
        sentence_chunks.extend(s_chunks)

    return {
        "fixed_chunks": fixed_chunks,
        "sentence_chunks": sentence_chunks
    }


if __name__ == "__main__":
    from knowledge_base import KNOWLEDGE_BASE_DOCS
    res = chunk_all_documents(KNOWLEDGE_BASE_DOCS)
    print(f"Total Fixed-Size Overlap Chunks: {len(res['fixed_chunks'])}")
    print(f"Total Sentence-Based Chunks: {len(res['sentence_chunks'])}")
    print(f"Sample Fixed Chunk: {res['fixed_chunks'][0]}")
    print(f"Sample Sentence Chunk: {res['sentence_chunks'][0]}")
