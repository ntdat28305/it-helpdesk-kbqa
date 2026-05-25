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
