"""Unit tests for agent tool pre-routing logic."""
import sys
from unittest.mock import MagicMock

# Stub out heavy dependencies before importing agent module
sys.modules.setdefault("src.agent.neo4j_query", MagicMock())
sys.modules.setdefault("src.agent.prompts", MagicMock())
sys.modules.setdefault("src.utils.logger", MagicMock())
# Stub groq and other optional deps
for _m in ["groq", "rapidfuzz", "rapidfuzz.process", "dotenv"]:
    sys.modules.setdefault(_m, MagicMock())

from src.agent.agent import _forced_tool


# ── cypher_search cases ───────────────────────────────────────

def test_hex_error_code_routes_cypher():
    assert _forced_tool("Auto MDM Enroll Failed (0x80180031)") == "cypher_search"

def test_named_error_routes_cypher():
    assert _forced_tool("ERROR_INVALID_HANDLE when opening app") == "cypher_search"

def test_kb_number_routes_cypher():
    assert _forced_tool("KB5034441 fails to install") == "cypher_search"


# ── web_search cases ──────────────────────────────────────────

def test_build_number_routes_websearch():
    assert _forced_tool("Windows 11 24H2 BSOD after update") == "web_search"

def test_latest_keyword_routes_websearch():
    assert _forced_tool("latest patch for 23H2") == "web_search"


# ── bfs_search cases ──────────────────────────────────────────

def test_what_causes_routes_bfs():
    assert _forced_tool("What causes Teams to fail when VPN is on?") == "bfs_search"

def test_how_does_routes_bfs():
    assert _forced_tool("How does Intune relate to Azure AD?") == "bfs_search"


# ── embedding_search default ──────────────────────────────────

def test_vague_symptom_routes_embedding():
    assert _forced_tool("My device is not showing in Intune portal") == "embedding_search"

def test_product_name_alone_routes_embedding():
    assert _forced_tool("Intune application transfer from MaaS360") == "embedding_search"

def test_teams_issue_routes_embedding():
    assert _forced_tool("Teams consent issue in tenant") == "embedding_search"

def test_setup_question_routes_embedding():
    assert _forced_tool("How to set up MDM enrollment for Windows") == "embedding_search"


# ── SYSTEM_PROMPT content checks ──────────────────────────────

def test_system_prompt_blocks_cypher_for_product_names():
    from src.agent.agent import SYSTEM_PROMPT
    assert "DO NOT use for product names alone" in SYSTEM_PROMPT

def test_system_prompt_blocks_websearch_as_fallback():
    from src.agent.agent import SYSTEM_PROMPT
    assert "DO NOT use as a fallback" in SYSTEM_PROMPT

def test_system_prompt_has_embedding_default_rule():
    from src.agent.agent import SYSTEM_PROMPT
    assert "embedding_search" in SYSTEM_PROMPT
    assert "DEFAULT" in SYSTEM_PROMPT

def test_cypher_tool_description_has_negative_constraint():
    from src.agent.agent import TOOLS
    cypher_desc = next(
        t["function"]["description"] for t in TOOLS
        if t["function"]["name"] == "cypher_search"
    )
    assert "Do NOT use for product names alone" in cypher_desc

def test_websearch_tool_description_has_no_fallback_constraint():
    from src.agent.agent import TOOLS
    web_desc = next(
        t["function"]["description"] for t in TOOLS
        if t["function"]["name"] == "web_search"
    )
    assert "Do NOT use as a fallback" in web_desc
