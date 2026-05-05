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
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {
    background: #0a0e14;
    color: #c5cdd9;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.block-container { padding-top: 1.2rem; max-width: 900px; }

/* ── Hero header ────────────────────────────────────────── */
.hero-header {
    background: linear-gradient(145deg, #111820 0%, #0d1117 50%, #0f1923 100%);
    border: 1px solid rgba(56, 68, 85, 0.6);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #2f81f7, #a371f7, #f778ba, #f0883e);
    border-radius: 16px 16px 0 0;
}
.hero-header::after {
    content: "";
    position: absolute;
    top: -50%; right: -20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(47,129,247,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f778ba 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.hero-sub {
    color: #6e7681;
    font-size: 0.82rem;
    margin: 0;
    letter-spacing: 0.2px;
}
.hero-pills {
    display: flex;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    background: rgba(56,68,85,0.3);
    border: 1px solid rgba(56,68,85,0.5);
    color: #8b949e;
    letter-spacing: 0.3px;
}

/* ── Tool badges ────────────────────────────────────────── */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
.badge-CYPHER {
    background: rgba(31,58,95,0.6);
    color: #79c0ff;
    border: 1px solid rgba(31,111,235,0.4);
}
.badge-EMBEDDING {
    background: rgba(45,31,74,0.6);
    color: #d2a8ff;
    border: 1px solid rgba(137,87,229,0.4);
}
.badge-BFS {
    background: rgba(15,45,31,0.6);
    color: #56d364;
    border: 1px solid rgba(35,134,54,0.4);
}
.badge-WEBSEARCH {
    background: rgba(61,36,0,0.6);
    color: #f0a800;
    border: 1px solid rgba(158,106,3,0.4);
}

/* ── Reasoning steps (collapsible via <details>) ───────── */
details.steps-container {
    margin-top: 12px;
    border: 1px solid rgba(48,54,61,0.6);
    border-radius: 12px;
    overflow: hidden;
    background: rgba(13,17,23,0.5);
}
details.steps-container summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: rgba(22,27,34,0.8);
    cursor: pointer;
    user-select: none;
    color: #8b949e;
    transition: background 0.2s ease;
    list-style: none;
    outline: none;
}
details.steps-container summary::-webkit-details-marker { display: none; }
details.steps-container summary::marker { display: none; content: ""; }
details.steps-container summary:hover {
    background: rgba(33,38,45,0.9);
}
.steps-toggle-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8b949e;
}
.steps-toggle-right {
    display: flex;
    align-items: center;
    gap: 6px;
}
.steps-count {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(56,68,85,0.4);
    color: #6e7681;
}
.steps-chevron {
    font-size: 0.7rem;
    color: #6e7681;
    transition: transform 0.3s ease;
}
details.steps-container[open] .steps-chevron {
    transform: rotate(90deg);
}
.steps-inner {
    padding: 6px 12px 12px 12px;
}

/* ── Step cards ──────────────────────────────────────────── */
.step-card {
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.5);
    border-radius: 10px;
    padding: 12px 16px 12px 50px;
    margin: 6px 0;
    position: relative;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.step-card:hover {
    border-color: rgba(48,54,61,0.8);
    background: rgba(22,27,34,0.9);
}
.step-num {
    position: absolute;
    left: 14px;
    top: 14px;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(47,129,247,0.15), rgba(163,113,247,0.15));
    border: 1px solid rgba(47,129,247,0.3);
    color: #58a6ff;
    font-size: 0.68rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
}
.step-tool-line {
    margin-bottom: 6px;
}
.step-input {
    font-size: 0.76rem;
    color: #c5cdd9;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
    padding: 4px 8px;
    background: rgba(56,68,85,0.2);
    border-radius: 6px;
    display: inline-block;
}
.step-input-arrow {
    color: #6e7681;
    margin-right: 4px;
}
.step-obs {
    font-size: 0.74rem;
    color: #6e7681;
    line-height: 1.5;
    border-left: 2px solid rgba(47,129,247,0.3);
    padding-left: 10px;
    white-space: pre-wrap;
    margin-top: 4px;
}

/* ── Answer meta bar ────────────────────────────────────── */
.meta-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    padding: 8px 14px;
    background: rgba(22,27,34,0.5);
    border: 1px solid rgba(48,54,61,0.4);
    border-radius: 10px;
    flex-wrap: wrap;
}
.meta-entity {
    font-size: 0.78rem;
    color: #8b949e;
}
.meta-entity code {
    background: rgba(56,68,85,0.4);
    padding: 2px 7px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #c5cdd9;
}
.meta-divider {
    width: 1px;
    height: 16px;
    background: rgba(48,54,61,0.6);
}

/* ── Source links ────────────────────────────────────────── */
.src-item {
    background: rgba(22,27,34,0.6);
    border: 1px solid rgba(48,54,61,0.5);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.78rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    transition: border-color 0.2s ease;
}
.src-item:hover {
    border-color: rgba(47,129,247,0.4);
}
.src-item a {
    color: #58a6ff;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 6px;
}
.src-item a:hover { text-decoration: underline; }
.src-icon {
    font-size: 0.72rem;
    opacity: 0.7;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0c1017 !important;
    border-right: 1px solid rgba(33,38,45,0.8);
}
.stat-card {
    background: rgba(22,27,34,0.7);
    border: 1px solid rgba(48,54,61,0.5);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    transition: border-color 0.2s ease;
}
.stat-card:hover {
    border-color: rgba(48,54,61,0.8);
}
.stat-label {
    color: #6e7681;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 500;
}
.stat-value {
    color: #e6edf3;
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 3px;
}
.stat-mono {
    color: #c5cdd9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    margin-top: 3px;
}

/* ── Sidebar example buttons ────────────────────────────── */
[data-testid="stSidebar"] button {
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
}

/* ── Chat styling ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
}
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
    "BFS":       {"icon": "🕸️", "label": "Graph Traversal",  "badge": "badge-BFS"},
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
def render_steps(steps: list[dict], msg_idx: int = 0):
    """Render reasoning steps as a collapsible section using HTML5 <details>."""
    if not steps:
        return

    # Build step cards HTML
    step_cards = ""
    tools_used = []
    for s in steps:
        tool = s.get("tool", "")
        num  = s.get("step", "")
        inp  = s.get("input", "")
        obs  = s.get("observation", "")[:300]
        if tool and tool not in tools_used:
            tools_used.append(tool)

        step_cards += (
            f'<div class="step-card">'
            f'  <div class="step-num">{num}</div>'
            f'  <div class="step-tool-line">{tool_badge(tool)}</div>'
            f'  <div class="step-input"><span class="step-input-arrow">→</span> {inp}</div>'
            f'  <div class="step-obs">{obs}</div>'
            f'</div>'
        )

    # Tool summary pills for the toggle bar
    tool_pills = " ".join(tool_badge(t) for t in tools_used)

    st.markdown(
        f'<details class="steps-container">'
        f'  <summary>'
        f'    <div class="steps-toggle-left">'
        f'      ⚙ Reasoning Steps'
        f'    </div>'
        f'    <div class="steps-toggle-right">'
        f'      {tool_pills}'
        f'      <span class="steps-count">{len(steps)} step{"s" if len(steps) > 1 else ""}</span>'
        f'      <span class="steps-chevron">▶</span>'
        f'    </div>'
        f'  </summary>'
        f'  <div class="steps-inner">'
        f'    {step_cards}'
        f'  </div>'
        f'</details>',
        unsafe_allow_html=True,
    )


def render_meta(tool: str, entity: str):
    """Render answer metadata bar (tool + entity)."""
    if not tool and not entity:
        return
    parts = []
    if tool:
        parts.append(tool_badge(tool))
    if entity:
        if tool:
            parts.append('<span class="meta-divider"></span>')
        parts.append(
            f'<span class="meta-entity">'
            f'Entity: <code>{entity}</code>'
            f'</span>'
        )
    st.markdown(
        f'<div class="meta-bar">{" ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]):
    """Render source links as a collapsible expander."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for src in sources[:5]:
            # Extract readable slug from URL
            slug = src.rstrip("/").split("/")[-1].replace("-", " ").title()[:55]
            st.markdown(
                f'<div class="src-item">'
                f'<a href="{src}" target="_blank">'
                f'<span class="src-icon">📄</span> {slug}'
                f'</a></div>',
                unsafe_allow_html=True,
            )


# ── Hero header ───────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🖥️ IT Helpdesk Assistant</div>
  <div class="hero-sub">Intelligent troubleshooting powered by Knowledge Graphs and Agentic AI</div>
  <div class="hero-pills">
    <span class="hero-pill">🕸️ Knowledge Graph</span>
    <span class="hero-pill">🧠 GCN Embedding</span>
    <span class="hero-pill">🤖 ReAct Agent</span>
    <span class="hero-pill">⚡ Groq LLaMA 3.1</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ Session")
    sid_short = st.session_state.session_id[:16] + "…"
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Session ID</div>'
        f'<div class="stat-mono">{sid_short}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Queries</div>'
        f'<div class="stat-value">{st.session_state.query_count}</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_result:
        lr     = st.session_state.last_result
        ltool  = lr.get("tool_used", "")
        lent   = lr.get("entity", "—")
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">Last Tool</div>'
            f'<div style="margin-top:5px;">{tool_badge(ltool)}</div></div>',
            unsafe_allow_html=True,
        )
        if lent and lent != "—":
            st.markdown(
                f'<div class="stat-card"><div class="stat-label">Entity</div>'
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

    st.markdown("### 🔧 Available Tools")
    for tk, tm in TOOL_META.items():
        st.markdown(
            f'<div class="stat-card" style="padding:8px 13px;">'
            f'{tool_badge(tk)}'
            f'<span style="font-size:0.74rem;color:#6e7681;margin-left:10px;">'
            f'{tm["label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("IT Helpdesk KBQA · NLP for Enterprise · VNU-HCMUS")


# ── Chat history ──────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_steps(msg.get("steps", []), msg_idx=i)
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
