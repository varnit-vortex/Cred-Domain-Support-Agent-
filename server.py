"""
FastAPI Deployment Backend & Structured PII-Safe JSONL Logging
Track: Banking & FinTech (Cred)

Endpoints:
- GET  / : Interactive Web Dashboard & Chat UI for Google Chrome / Web Browsers
- POST /ask: End-to-end query answering via LangGraph agent
- POST /loan-status: Direct loan application lookup & escalation assessment
- POST /add-document: Dynamic knowledge base document ingestion & indexing
- GET  /health: Service health and system metadata
- GET  /docs: Interactive OpenAPI Swagger documentation
"""

import os
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from schemas import (
    AskRequest,
    LoanStatusRequest,
    AddDocumentRequest,
    AgentResponse,
    LoanApplicationStatusResponse
)
from graph import run_agent, get_rag_core
from tools import check_loan_application_status
from guardrails import mask_fixed_format_pii
from chunking import chunk_fixed_size, chunk_sentence_based

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "api_traces.jsonl")


def append_structured_log(log_entry: Dict[str, Any]) -> None:
    """
    Appends a sanitized structured log entry to the JSON-Lines trace log.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure RAG core initialized
    os.makedirs(LOG_DIR, exist_ok=True)
    get_rag_core()
    yield
    # Shutdown


app = FastAPI(
    title="Cred Domain Support Agent API",
    description="Production-minded lending operations & policy support agent with LangGraph orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # Read body for POST endpoints to mask PII before logging
    body_text = ""
    masked_body_log = ""
    if request.method in ["POST", "PUT", "PATCH"]:
        body_bytes = await request.body()
        body_text = body_bytes.decode("utf-8", errors="ignore")
        # Mask any fixed-format PII in the request body before logging to disk
        masked_body_log, _ = mask_fixed_format_pii(body_text)

        # Restore body for endpoint handler
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Build PII-safe structured JSON-Lines log entry
    log_entry = {
        "trace_id": trace_id,
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "client_ip": request.client.host if request.client else "unknown",
        "sanitized_payload": masked_body_log if masked_body_log else None
    }
    
    # Write to disk
    append_structured_log(log_entry)

    # Attach trace ID header
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Response-Time-MS"] = str(duration_ms)

    return response


# Mount static assets
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def serve_home():
    """
    Serves the modern Cred Domain Support interactive web frontend.
    """
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML)
    return {"message": "Cred Domain Support Agent API. Visit /docs for OpenAPI specifications."}


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint returning system status and configuration.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "track": "Banking & FinTech (Cred)",
        "mode": "MOCK_LLM",
        "version": "1.0.0"
    }


@app.post("/ask", response_model=AgentResponse)
def ask_agent(req: AskRequest) -> AgentResponse:
    """
    Main conversational endpoint: routes through LangGraph StateGraph.
    """
    try:
        response_dict = run_agent(
            query=req.query,
            session_id=req.session_id or "default_session",
            provider=req.provider or "mock",
            api_key=req.api_key
        )
        return AgentResponse(**response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution error: {str(e)}")


@app.post("/loan-status", response_model=LoanApplicationStatusResponse)
def get_loan_status(req: LoanStatusRequest) -> LoanApplicationStatusResponse:
    """
    Direct lookup endpoint for loan application status and escalation score.
    """
    result = check_loan_application_status(req.record_id)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=result.get("error", "Application not found."))

    return LoanApplicationStatusResponse(
        record_id=result["record_id"],
        category=result["category"],
        status=result["status"],
        loan_amount_inr=result["loan_amount_inr"],
        days_since_created=result["days_since_created"],
        flagged_for_fraud_review=result["flagged_for_fraud_review"],
        escalation_score=result["escalation_score"],
        escalation_recommended=result["escalation_recommended"],
        escalation_reason=result["escalation_reason"]
    )


@app.post("/add-document")
def add_knowledge_document(req: AddDocumentRequest) -> Dict[str, Any]:
    """
    Dynamic document ingestion endpoint: chunks and updates ChromaDB vector store.
    """
    rag = get_rag_core()
    meta = {
        "doc_id": req.doc_id,
        "title": req.title,
        "topic": req.topic
    }

    # Generate chunks
    fixed_chunks = chunk_fixed_size(req.content, doc_meta=meta)
    sentence_chunks = chunk_sentence_based(req.content, doc_meta=meta)

    # Embed and index into both collections
    if fixed_chunks:
        texts = [c["text"] for c in fixed_chunks]
        embeddings = rag.model.encode(texts, convert_to_numpy=True).tolist()
        rag.fixed_collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "doc_id": c["doc_id"],
                "title": c["title"],
                "topic": c["topic"],
                "strategy": c["strategy"],
                "chunk_index": c["chunk_index"]
            } for c in fixed_chunks],
            ids=[c["chunk_id"] for c in fixed_chunks]
        )

    if sentence_chunks:
        texts = [c["text"] for c in sentence_chunks]
        embeddings = rag.model.encode(texts, convert_to_numpy=True).tolist()
        rag.sentence_collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "doc_id": c["doc_id"],
                "title": c["title"],
                "topic": c["topic"],
                "strategy": c["strategy"],
                "chunk_index": c["chunk_index"]
            } for c in sentence_chunks],
            ids=[c["chunk_id"] for c in sentence_chunks]
        )

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "fixed_chunks_added": len(fixed_chunks),
        "sentence_chunks_added": len(sentence_chunks),
        "message": f"Document '{req.title}' ({req.doc_id}) successfully indexed into dual ChromaDB collections."
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Cred Domain Support Agent FastAPI server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
