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

st.set_page_config(
    page_title="IT Helpdesk KBQA",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
.stApp { background: #0f1117; color: #e6edf3; }
.block-container { padding-top: 1.5rem; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 60%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #58a6ff, #bc8cff, #ff7b72);
}
.hero-title {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #bc8cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px 0;
}
.hero-sub { color: #8b949e; font-size: 0.85rem; margin: 0; }

/* Tool badges */
.tool-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.badge-CYPHER    { background: #1f3a5f; color: #79c0ff; border: 1px solid #1f6feb; }
.badge-EMBEDDING { background: #2d1f4a; color: #d2a8ff; border: 1px solid #8957e5; }
.badge-BFS       { background: #0f2d1f; color: #56d364; border: 1px solid #238636; }
.badge-WEBSEARCH { background: #3d2400; color: #f0a800; border: 1px solid #9e6a03; }

/* Reasoning steps */
.steps-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 14px 0 6px 0;
}
.step-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px 10px 46px;
    margin: 5px 0;
    position: relative;
}
.step-num {
    position: absolute;
    left: 12px;
    top: 12px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #21262d;
    border: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.68rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-tool-line { margin-bottom: 4px; }
.step-input {
    font-size: 0.77rem;
    color: #8b949e;
    font-family: monospace;
    margin-bottom: 5px;
}
.step-obs {
    font-size: 0.76rem;
    color: #6e7681;
    line-height: 1.45;
    border-left: 2px solid #30363d;
    padding-left: 8px;
    white-space: pre-wrap;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 9px 13px;
    margin: 5px 0;
}
.stat-label { color: #6e7681; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { color: #e6edf3; font-weight: 600; font-size: 1rem; margin-top: 2px; }
.stat-mono  { color: #e6edf3; font-family: monospace; font-size: 0.78rem; margin-top: 2px; }

/* Source links */
.src-item {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.8rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.src-item a { color: #58a6ff; text-decoration: none; }
.src-item a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
if "session_id"   not in st.session_state:
    st.session_state.session_id   = str(uuid.uuid4())
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "query_count"  not in st.session_state:
    st.session_state.query_count  = 0
if "last_result"  not in st.session_state:
    st.session_state.last_result  = None


# ── Tool metadata ─────────────────────────────────────────────
TOOL_META = {
    "CYPHER":    {"icon": "🔍", "label": "Knowledge Graph",  "badge": "badge-CYPHER"},
    "EMBEDDING": {"icon": "🧠", "label": "Semantic Search",  "badge": "badge-EMBEDDING"},
    "BFS":       {"icon": "🕸️", "label": "Graph Path",       "badge": "badge-BFS"},
    "WEBSEARCH": {"icon": "🌐", "label": "Web Search",       "badge": "badge-WEBSEARCH"},
}


def tool_badge(tool: str) -> str:
    m = TOOL_META.get(tool, {"icon": "🔧", "label": tool, "badge": ""})
    return f'<span class="tool-badge {m["badge"]}">{m["icon"]} {tool}</span>'


# ── API helpers ───────────────────────────────────────────────
def query_api(question: str, session_id: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": session_id},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None


def reset_session():
    try:
        requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
    except Exception:
        pass
    st.session_state.session_id  = str(uuid.uuid4())
    st.session_state.messages    = []
    st.session_state.query_count = 0
    st.session_state.last_result = None


# ── Render helpers ────────────────────────────────────────────
def render_steps(steps: list[dict]):
    if not steps:
        return
    st.markdown('<div class="steps-header">⚙ Reasoning Steps</div>', unsafe_allow_html=True)
    for s in steps:
        tool = s.get("tool", "")
        num  = s.get("step", "")
        inp  = s.get("input", "")
        obs  = s.get("observation", "")[:280]
        st.markdown(
            f'<div class="step-card">'
            f'  <div class="step-num">{num}</div>'
            f'  <div class="step-tool-line">{tool_badge(tool)}</div>'
            f'  <div class="step-input">→ {inp}</div>'
            f'  <div class="step-obs">{obs}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_meta(tool: str, entity: str):
    parts = []
    if tool:
        parts.append(tool_badge(tool))
    if entity:
        parts.append(
            f'<span style="font-size:0.8rem;color:#8b949e;margin-left:8px;">'
            f'entity: <code style="background:#21262d;padding:1px 5px;border-radius:4px;">'
            f'{entity}</code></span>'
        )
    if parts:
        st.markdown(" ".join(parts), unsafe_allow_html=True)


def render_sources(sources: list[str]):
    if not sources:
        return
    with st.expander("📚 Sources", expanded=False):
        for src in sources[:5]:
            short = (src.split("/")[-1] or src)[:60]
            st.markdown(
                f'<div class="src-item"><a href="{src}" target="_blank">🔗 {short}</a></div>',
                unsafe_allow_html=True,
            )


# ── Hero header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🖥️ IT Helpdesk Assistant</div>
  <div class="hero-sub">Knowledge Graph &nbsp;·&nbsp; GCN Embedding &nbsp;·&nbsp; ReAct Agent &nbsp;·&nbsp; Groq LLaMA 3</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Session")
    sid_short = st.session_state.session_id[:20] + "…"
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Session ID</div>'
        f'<div class="stat-mono">{sid_short}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Queries this session</div>'
        f'<div class="stat-value">{st.session_state.query_count}</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_result:
        lr     = st.session_state.last_result
        ltool  = lr.get("tool_used", "")
        lent   = lr.get("entity", "—")
        lmeta  = TOOL_META.get(ltool, {"icon": "🔧", "label": ltool, "badge": ""})
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Last tool</div>'
            f'<div style="margin-top:4px;">{tool_badge(ltool)}</div></div>',
            unsafe_allow_html=True,
        )
        if lent and lent != "—":
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">Extracted entity</div>'
                f'<div class="stat-mono">{lent}</div></div>',
                unsafe_allow_html=True,
            )

    if st.button("🔄 New Conversation", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()

    st.markdown("### 💡 Try These")
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

    st.markdown("### 🔧 Tools")
    for tk, tm in TOOL_META.items():
        st.markdown(
            f'<div class="stat-card" style="padding:7px 12px;">'
            f'{tool_badge(tk)}'
            f'<span style="font-size:0.76rem;color:#8b949e;margin-left:8px;">{tm["label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("IT Helpdesk KBQA · NLP Project")


# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_steps(msg.get("steps", []))
            render_meta(msg.get("tool", ""), msg.get("entity", ""))
            render_sources(msg.get("sources", []))


# ── Input ─────────────────────────────────────────────────────
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

user_input = st.chat_input("Ask an IT question…")
if user_input:
    question = user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            result = query_api(question, st.session_state.session_id)

        if result:
            answer  = result["answer"]
            steps   = result.get("steps", [])
            tool    = result.get("tool_used", "")
            entity  = result.get("entity", "")
            sources = result.get("sources", [])

            st.markdown(answer)
            render_steps(steps)
            render_meta(tool, entity)
            render_sources(sources)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "tool":    tool,
                "entity":  entity,
                "sources": sources,
                "steps":   steps,
            })
            st.session_state.query_count += 1
            st.session_state.last_result  = result
