"""
src/api/main.py
FastAPI REST API cho IT Helpdesk KBQA.

Chạy:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.agent import ITHelpdeskAgent
from src.agent.neo4j_query import close_driver
from src.utils.logger import get_logger

from uuid import uuid4
import asyncio

load_dotenv()
logger = get_logger(__name__, log_file="logs/api.log")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")
    close_driver()  # close Neo4j singleton driver


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="IT Helpdesk KBQA API",
    description="Knowledge Base Question Answering for IT Helpdesk",
    version="1.0.0",
    lifespan=lifespan,
)

# allow_origins=["*"] is intentional for demo; restrict to UI origin in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How to fix ERROR_INVALID_HANDLE?",
                "session_id": "user_123"
            }
        }


class QueryResponse(BaseModel):
    question: str
    answer: str
    tool_used: str
    entity: str
    sources: list[str]
    session_id: str
    steps: list[dict] = []
    plan_note: str = ""
    confidence: str = "medium"
    reflection_reason: str = ""
    observations: list[str] = []
    context: str = ""


# ── Session management ────────────────────────────────────────
# LRU cache: mỗi session_id có agent riêng, tối đa MAX_SESSIONS sessions
MAX_SESSIONS = 500
sessions: OrderedDict[str, ITHelpdeskAgent] = OrderedDict()

_session_lock = asyncio.Lock()
async def get_or_create_session(session_id: str) -> ITHelpdeskAgent:
    async with _session_lock:
        if session_id in sessions:
            sessions.move_to_end(session_id)
            return sessions[session_id]
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            del sessions[oldest]
        sessions[session_id] = ITHelpdeskAgent()
        return sessions[session_id]


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Kiểm tra API còn sống không."""
    return {
        "status": "ok",
        "active_sessions": len(sessions),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Nhận câu hỏi IT helpdesk, trả lời dùng KG + Agent.

    - **question**: câu hỏi của user
    - **session_id**: ID session để giữ conversation history
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question không được rỗng")

    try:
        session_agent = await get_or_create_session(request.session_id)
        result = await asyncio.to_thread(session_agent.answer, request.question)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            tool_used=result["tool_used"],
            entity=result["entity"],
            sources=result["sources"],
            session_id=request.session_id,
            steps=result.get("steps", []),
            plan_note=result.get("plan_note", ""),
            confidence=result.get("confidence", "medium"),
            reflection_reason=result.get("reflection_reason", ""),
            observations=result.get("observations", []),
            context=result.get("context", ""),
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    """Xóa session và toàn bộ conversation history."""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session deleted: {session_id}")
        return {"message": f"Session {session_id} đã được xóa"}
    return {"message": f"Session {session_id} không tồn tại"}


@app.get("/sessions")
async def list_sessions():
    """Xem danh sách sessions đang active."""
    return {
        "active_sessions": list(sessions.keys()),
        "total": len(sessions),
    }