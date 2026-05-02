"""
src/ui/app.py
Streamlit UI cho IT Helpdesk KBQA.

Chạy (cần FastAPI đang chạy ở port 8000):
    streamlit run src/ui/app.py
"""
from __future__ import annotations
import os
import uuid
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="IT Helpdesk KBQA",
    page_icon="🖥️",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_tool" not in st.session_state:
    st.session_state.last_tool = ""


# ── Helper ────────────────────────────────────────────────────

def query_api(question: str, session_id: str) -> dict | None:
    """Gọi FastAPI backend."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Không kết nối được API. Hãy chắc chắn FastAPI đang chạy ở port 8000.")
        return None
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None


def reset_session():
    """Reset conversation."""
    try:
        requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
    except Exception:
        pass
    st.session_state.session_id  = str(uuid.uuid4())[:8]
    st.session_state.messages    = []
    st.session_state.last_sources = []
    st.session_state.last_tool    = ""


TOOL_COLORS = {
    "CYPHER":    ("🔍", "#4A90D9"),
    "EMBEDDING": ("🧠", "#7B68EE"),
    "BFS":       ("🕸️", "#48C774"),
    "WEBSEARCH": ("🌐", "#FF8C00"),
}


# ── Layout ────────────────────────────────────────────────────
st.title("🖥️ IT Helpdesk Assistant")
st.caption("Powered by Knowledge Graph + GCN Embedding + Agentic AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 New Conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()

    st.header("📊 Last Query Info")
    if st.session_state.last_tool:
        icon, color = TOOL_COLORS.get(st.session_state.last_tool, ("🔧", "#888"))
        st.markdown(
            f"**Tool used:** {icon} "
            f"<span style='color:{color}'>{st.session_state.last_tool}</span>",
            unsafe_allow_html=True,
        )

    if st.session_state.last_sources:
        st.markdown("**Sources:**")
        for src in st.session_state.last_sources[:3]:
            # Rút gọn URL để hiển thị
            short = src.split("/")[-1][:40] if "/" in src else src[:40]
            st.markdown(f"- [{short}]({src})")

    st.divider()

    st.header("💡 Example Questions")
    examples = [
        "How to fix ERROR_INVALID_HANDLE?",
        "Smart card not working on Windows",
        "Teams meeting keeps dropping",
        "Cannot connect to internet after update",
        "Relationship between VPN and Teams failure",
        "Latest Windows 11 update causing BSOD",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_question = ex
            st.rerun()

    st.divider()
    st.caption("IT Helpdesk KBQA | NLP Project")

# ── Chat area ─────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("tool"):
                icon, color = TOOL_COLORS.get(msg["tool"], ("🔧", "#888"))
                st.caption(
                    f"{icon} Tool: **{msg['tool']}** | "
                    f"Entity: `{msg.get('entity', '')}`"
                )

# ── Input ─────────────────────────────────────────────────────
# Xử lý example question click
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

user_input = st.chat_input("Ask an IT question...")
if user_input:
    question = user_input

# Xử lý câu hỏi
if question:
    # Hiện câu hỏi user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Gọi API và hiện response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_api(question, st.session_state.session_id)

        if result:
            st.markdown(result["answer"])

            # Hiện tool info
            tool = result.get("tool_used", "")
            entity = result.get("entity", "")
            icon, color = TOOL_COLORS.get(tool, ("🔧", "#888"))
            st.caption(f"{icon} Tool: **{tool}** | Entity: `{entity}`")

            # Lưu vào session state
            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["answer"],
                "tool":    tool,
                "entity":  entity,
            })
            st.session_state.last_sources = result.get("sources", [])
            st.session_state.last_tool    = tool

            # Hiện sources nếu có
            if result.get("sources"):
                with st.expander("📚 Sources"):
                    for src in result["sources"]:
                        st.markdown(f"- [{src}]({src})")

            st.rerun()