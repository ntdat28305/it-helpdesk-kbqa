"""Sprint 2 agent upgrade tests."""
import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

# ── Stub heavy optional deps ──────────────────────────────────
for _m in ["groq", "rapidfuzz", "rapidfuzz.process", "dotenv",
           "neo4j", "tavily"]:
    sys.modules.setdefault(_m, MagicMock())

_st_mock = MagicMock()
_st_instance = MagicMock()
_st_instance.encode = MagicMock(return_value=np.array([0.9, 0.1, 0.1]))
_st_mock.SentenceTransformer = MagicMock(return_value=_st_instance)
sys.modules["sentence_transformers"] = _st_mock

sys.modules.setdefault("src.agent.neo4j_query", MagicMock())
sys.modules.setdefault("src.agent.prompts", MagicMock())
sys.modules.setdefault("src.utils.logger", MagicMock())


# ── S2.1 global_search ────────────────────────────────────────

def test_forced_tool_overview_routes_global():
    from src.agent.agent import _forced_tool
    assert _forced_tool("What are the main IT topics covered?") == "global_search"

def test_forced_tool_themes_routes_global():
    from src.agent.agent import _forced_tool
    assert _forced_tool("Give me an overview of MDM themes") == "global_search"

def test_forced_tool_summarize_routes_global():
    from src.agent.agent import _forced_tool
    assert _forced_tool("Summarize the IT knowledge base topics") == "global_search"

def test_forced_tool_summarize_false_positive():
    """'summarize my issue' must NOT route to global_search."""
    from src.agent.agent import _forced_tool
    result = _forced_tool("I can't summarize my issue with Teams")
    assert result != "global_search", f"False positive: got {result!r}"

def test_global_search_in_tools_array():
    from src.agent.agent import TOOLS
    names = [t["function"]["name"] for t in TOOLS]
    assert "global_search" in names

def test_global_search_tool_description_has_overview():
    from src.agent.agent import TOOLS
    desc = next(t["function"]["description"] for t in TOOLS
                if t["function"]["name"] == "global_search")
    assert "overview" in desc.lower()

def test_global_search_returns_summaries():
    from src.agent.agent import global_search

    communities = {
        "0": {"summary": "Azure AD authentication and sign-in issues.", "nodes": ["Azure AD"]},
        "1": {"summary": "Intune device enrollment and MDM management.", "nodes": ["Intune"]},
        "2": {"summary": "Teams meetings and audio problems.",           "nodes": ["Teams"]},
    }
    comm_embs = np.array([
        [0.1, 0.9, 0.1],
        [0.9, 0.1, 0.1],  # closest to query
        [0.1, 0.1, 0.9],
    ], dtype=np.float32)
    _st_instance.encode.return_value = np.array([0.9, 0.1, 0.1])

    resources = {
        "communities":    communities,
        "community_embs": comm_embs,
        "community_ids":  ["0", "1", "2"],
        "retriever":      _st_instance,
    }
    result = global_search("overview of MDM", resources, top_k=2)
    assert "Intune" in result or "MDM" in result or "enrollment" in result.lower()

def test_global_search_returns_no_data_message_when_no_embs():
    from src.agent.agent import global_search
    result = global_search("overview", {})
    assert "No community data" in result or "not" in result.lower()

# ── S2.2 merged preprocess ────────────────────────────────────

def test_preprocess_prompt_has_both_fields():
    import importlib, sys
    sys.modules.pop("src.agent.prompts", None)
    import src.agent.prompts as p
    assert "is_ambiguous" in p.PREPROCESS_PROMPT
    assert "topic_changed" in p.PREPROCESS_PROMPT

def test_preprocess_prompt_has_prev_current_placeholders():
    import src.agent.prompts as p
    assert "{prev}" in p.PREPROCESS_PROMPT
    assert "{current}" in p.PREPROCESS_PROMPT

# ── S2.3 reflexion memory ─────────────────────────────────────

def test_save_reflection_lesson_creates_file(tmp_path):
    from src.agent import agent as ag
    orig = ag.REFLEXION_MEMORY_FILE
    ag.REFLEXION_MEMORY_FILE = tmp_path / "reflexion_memory.jsonl"
    try:
        ag.save_reflection_lesson("Teams fails on VPN", "EMBEDDING", "Answer was too vague")
        lines = ag.REFLEXION_MEMORY_FILE.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["question"] == "Teams fails on VPN"
        assert entry["lesson"] == "Answer was too vague"
    finally:
        ag.REFLEXION_MEMORY_FILE = orig

def test_load_reflection_lessons_returns_empty_when_no_file(tmp_path):
    from src.agent import agent as ag
    orig = ag.REFLEXION_MEMORY_FILE
    ag.REFLEXION_MEMORY_FILE = tmp_path / "nonexistent.jsonl"
    try:
        result = ag.load_reflection_lessons("anything", {})
        assert result == []
    finally:
        ag.REFLEXION_MEMORY_FILE = orig

def test_load_reflection_lessons_semantic_match(tmp_path):
    from src.agent import agent as ag
    orig = ag.REFLEXION_MEMORY_FILE
    ag.REFLEXION_MEMORY_FILE = tmp_path / "reflexion_memory.jsonl"
    try:
        entries = [
            {"question": "Teams drops on VPN", "lesson": "Check VPN split tunneling",
             "tool_used": "BFS", "ts": "2026-01-01"},
            {"question": "Printer not found", "lesson": "Check printer driver",
             "tool_used": "EMBEDDING", "ts": "2026-01-02"},
        ]
        ag.REFLEXION_MEMORY_FILE.write_text(
            "\n".join(json.dumps(e) for e in entries), encoding="utf-8"
        )
        mock_retriever = MagicMock()
        mock_retriever.encode.side_effect = [
            np.array([1.0, 0.0]),
            np.array([[0.95, 0.05],
                      [0.05, 0.95]]),
        ]
        resources = {"retriever": mock_retriever}
        result = ag.load_reflection_lessons("VPN Teams issue", resources, top_k=1)
        assert isinstance(result, list)
    finally:
        ag.REFLEXION_MEMORY_FILE = orig


# ── S2.4 query rewriter ───────────────────────────────────────

def test_rewrite_prompt_exists():
    import sys
    sys.modules.pop("src.agent.prompts", None)
    import src.agent.prompts as p
    assert hasattr(p, "REWRITE_PROMPT")
    assert len(p.REWRITE_PROMPT) > 20

def test_rewrite_prompt_has_query_placeholder():
    import sys
    sys.modules.pop("src.agent.prompts", None)
    import src.agent.prompts as p
    assert "{query}" in p.REWRITE_PROMPT
