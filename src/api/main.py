"""
src/api/main.py
FastAPI REST API cho IT Helpdesk KBQA.

Chạy:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.agent import ITHelpdeskAgent
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/api.log")

# ── Global agent instance ─────────────────────────────────────
agent: ITHelpdeskAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo agent khi startup, cleanup khi shutdown."""
    global agent
    logger.info("Starting up — loading agent...")
    agent = ITHelpdeskAgent()
    logger.info("Agent ready!")
    yield
    logger.info("Shutting down...")


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="IT Helpdesk KBQA API",
    description="Knowledge Base Question Answering for IT Helpdesk",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

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


# ── Session management ────────────────────────────────────────
# Mỗi session_id có agent riêng để giữ conversation history độc lập
sessions: dict[str, ITHelpdeskAgent] = {}


def get_or_create_session(session_id: str) -> ITHelpdeskAgent:
    """Lấy hoặc tạo agent cho session."""
    if session_id not in sessions:
        sessions[session_id] = ITHelpdeskAgent()
        logger.info(f"New session: {session_id}")
    return sessions[session_id]


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Kiểm tra API còn sống không."""
    return {
        "status": "ok",
        "agent_loaded": agent is not None,
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
        session_agent = get_or_create_session(request.session_id)
        result = session_agent.answer(request.question)

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            tool_used=result["tool_used"],
            entity=result["entity"],
            sources=result["sources"],
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    """Reset conversation history của một session."""
    if session_id in sessions:
        sessions[session_id].reset_history()
        logger.info(f"Session reset: {session_id}")
        return {"message": f"Session {session_id} đã được reset"}
    return {"message": f"Session {session_id} không tồn tại"}


@app.get("/sessions")
async def list_sessions():
    """Xem danh sách sessions đang active."""
    return {
        "active_sessions": list(sessions.keys()),
        "total": len(sessions),
    }