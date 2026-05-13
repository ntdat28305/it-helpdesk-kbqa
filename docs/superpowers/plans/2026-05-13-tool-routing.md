# Tool Routing Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve agent tool accuracy from 60% to ≥75% by tightening pre-routing logic and rewriting SYSTEM_PROMPT/tool descriptions in `src/agent/agent.py`.

**Architecture:** Two changes in the same file — (1) `_forced_tool()` returns `"embedding_search"` as hard default instead of `None`, and adds `_RE_BFS` for relationship questions; (2) `SYSTEM_PROMPT` rewritten with explicit negative constraints ("DO NOT use cypher for product names", "DO NOT use web_search as fallback") and updated tool descriptions.

**Tech Stack:** Python 3.12, regex, Groq LLM function-calling, pytest

---

## File Map

| File | Change |
|------|--------|
| `src/agent/agent.py` | Add `_RE_BFS`, update `_forced_tool()`, rewrite `SYSTEM_PROMPT`, update 2 tool descriptions |
| `tests/test_agent_routing.py` | Create — unit tests for `_forced_tool` and prompt content |

---

## Task 1: Tests + implement `_forced_tool` and `_RE_BFS`

**Files:**
- Create: `tests/test_agent_routing.py`
- Modify: `src/agent/agent.py:39-62`

- [ ] **Step 1: Write failing tests**

Create `tests/test_agent_routing.py`:

```python
"""Unit tests for agent tool pre-routing logic."""
import sys
from unittest.mock import MagicMock

# Stub out heavy dependencies before importing agent module
sys.modules.setdefault("src.agent.neo4j_query", MagicMock())
sys.modules.setdefault("src.agent.prompts", MagicMock())
sys.modules.setdefault("src.utils.logger", MagicMock())
# Stub groq and other optional deps
for _m in ["groq", "rapidfuzz", "rapidfuzz.process", "numpy", "dotenv"]:
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
```

- [ ] **Step 2: Run tests — verify they FAIL**

```
pytest tests/test_agent_routing.py -v
```

Expected: `test_what_causes_routes_bfs` and `test_how_does_routes_bfs` fail with AttributeError (`_RE_BFS` not defined). Other embedding tests may also fail because current `_forced_tool` returns `None` not `"embedding_search"`.

- [ ] **Step 3: Add `_RE_BFS` and update `_forced_tool` in agent.py**

In `src/agent/agent.py`, find the regex section (around line 39) and replace:

```python
# ── Regex patterns for pre-routing ────────────────────────────
_RE_ERROR_CODE = re.compile(
    r'\b(0x[0-9A-Fa-f]+|ERROR_\w+|KB\d{6,7}|HRESULT\b)',
    re.IGNORECASE,
)
_RE_WEBSEARCH = re.compile(
    r'\b(latest|newest|recent|update|patch|release|version)\b.*\b(24H2|23H2|2[0-9]{3})\b'
    r'|\b(24H2|23H2)\b'
    r'|\b(20[2-9][0-9])\b',
    re.IGNORECASE,
)
_RE_FOLLOWUP = re.compile(
    r'\b(it|this|that|they|them|the (issue|problem|error|fix|solution)|'
    r'what about|any other|elaborate|more detail|next step|how do i fix|same)\b',
    re.IGNORECASE,
)


def _forced_tool(question: str) -> str | None:
    if _RE_ERROR_CODE.search(question):
        return "cypher_search"
    if _RE_WEBSEARCH.search(question):
        return "web_search"
    return None
```

With:

```python
# ── Regex patterns for pre-routing ────────────────────────────
_RE_ERROR_CODE = re.compile(
    r'\b(0x[0-9A-Fa-f]+|ERROR_\w+|KB\d{6,7}|HRESULT\b)',
    re.IGNORECASE,
)
_RE_WEBSEARCH = re.compile(
    r'\b(latest|newest|recent|update|patch|release|version)\b.*\b(24H2|23H2|2[0-9]{3})\b'
    r'|\b(24H2|23H2)\b'
    r'|\b(20[2-9][0-9])\b',
    re.IGNORECASE,
)
_RE_FOLLOWUP = re.compile(
    r'\b(it|this|that|they|them|the (issue|problem|error|fix|solution)|'
    r'what about|any other|elaborate|more detail|next step|how do i fix|same)\b',
    re.IGNORECASE,
)
_RE_BFS = re.compile(
    r'\b(what causes|why does|how does|relationship between|'
    r'connection between|when .+ and .+)\b',
    re.IGNORECASE,
)


def _forced_tool(question: str) -> str:
    if _RE_ERROR_CODE.search(question):
        return "cypher_search"
    if _RE_WEBSEARCH.search(question):
        return "web_search"
    if _RE_BFS.search(question):
        return "bfs_search"
    return "embedding_search"
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_agent_routing.py -v
```

Expected output:
```
tests/test_agent_routing.py::test_hex_error_code_routes_cypher PASSED
tests/test_agent_routing.py::test_named_error_routes_cypher PASSED
tests/test_agent_routing.py::test_kb_number_routes_cypher PASSED
tests/test_agent_routing.py::test_build_number_routes_websearch PASSED
tests/test_agent_routing.py::test_latest_keyword_routes_websearch PASSED
tests/test_agent_routing.py::test_what_causes_routes_bfs PASSED
tests/test_agent_routing.py::test_how_does_routes_bfs PASSED
tests/test_agent_routing.py::test_vague_symptom_routes_embedding PASSED
tests/test_agent_routing.py::test_product_name_alone_routes_embedding PASSED
tests/test_agent_routing.py::test_teams_issue_routes_embedding PASSED
tests/test_agent_routing.py::test_setup_question_routes_embedding PASSED
11 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/agent/agent.py tests/test_agent_routing.py
git commit -m "feat: extend pre-routing with BFS detection and embedding default"
```

---

## Task 2: Rewrite SYSTEM_PROMPT and tool descriptions

**Files:**
- Modify: `src/agent/agent.py:67-184`
- Modify: `tests/test_agent_routing.py` (add prompt content tests)

- [ ] **Step 6: Write failing prompt content tests**

Append to `tests/test_agent_routing.py`:

```python
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
```

- [ ] **Step 7: Run tests — verify 5 new tests FAIL**

```
pytest tests/test_agent_routing.py -v -k "prompt or tool_description"
```

Expected: 5 FAILED (phrases not yet in SYSTEM_PROMPT/descriptions).

- [ ] **Step 8: Replace SYSTEM_PROMPT in agent.py**

Find and replace the entire `SYSTEM_PROMPT` string (lines ~67-89) with:

```python
SYSTEM_PROMPT = """You are an IT helpdesk assistant with access to a knowledge graph and web search.

TOOL SELECTION — apply the FIRST matching rule:

1. cypher_search — ONLY when question contains an EXPLICIT error code:
   - Hex codes: 0x80004005, 0xC000021A, 0x80180031
   - Named errors: ERROR_INVALID_HANDLE, ERROR_ACCESS_DENIED
   - KB articles: KB5034441, KB123456
   - HRESULT codes
   DO NOT use for product names alone (Intune, Teams, Azure AD, Windows).

2. bfs_search — ONLY when question asks the relationship between TWO named entities:
   - "What causes Teams to fail when VPN is on?"
   - "How does Intune relate to Azure AD?"

3. web_search — ONLY when question explicitly mentions:
   - A specific OS build number: 24H2, 23H2, 22H2
   - Words like "latest", "newest", "recent patch", "current update"
   DO NOT use as a fallback when the knowledge graph returns no results.
   DO NOT use for general troubleshooting questions.

4. embedding_search — DEFAULT for everything else:
   - Vague symptoms: "not working", "keeps failing", "can't connect"
   - Setup/config issues: "how to set up", "can't enroll", "won't install"
   - Product questions without error codes: "Intune enrollment issue", "Teams consent issue"
   - General IT troubleshooting of any kind

Examples:
- "Intune application transfer from MaaS360" → embedding_search
- "Device not showing in Intune portal" → embedding_search
- "Teams consent issue in tenant" → embedding_search
- "Can't connect to Wi-Fi after update" → embedding_search
- "Auto MDM Enroll Failed (0x80180031)" → cypher_search
- "ERROR_INVALID_HANDLE when opening app" → cypher_search
- "KB5034441 fails to install" → cypher_search
- "Windows 11 24H2 BSOD latest patch" → web_search
- "What causes Teams to fail when VPN is on?" → bfs_search

Rules:
1. Call ONE tool per step — pick the most relevant one first.
2. If a tool returns no results, retry with embedding_search using a rephrased query — do NOT switch to web_search.
3. Once you have enough information, provide a concise, actionable answer with steps.
4. Do NOT keep calling tools if you already have a good answer."""
```

- [ ] **Step 9: Update cypher_search tool description in TOOLS list**

Find the `cypher_search` function entry in `TOOLS` (around line 98) and replace its `"description"` value:

```python
"description": (
    "Search the IT knowledge graph for a specific entity, error code, or product. "
    "ONLY use for explicit error codes (0x..., ERROR_XXX, KB numbers, HRESULT) "
    "or exact component names paired with an error code. "
    "Do NOT use for product names alone (e.g. 'Intune', 'Teams', 'Azure AD') "
    "without an accompanying error code."
),
```

- [ ] **Step 10: Update web_search tool description in TOOLS list**

Find the `web_search` function entry in `TOOLS` (around line 170) and replace its `"description"` value:

```python
"description": (
    "Search the web for recent IT issues, latest Windows updates, or version-specific problems. "
    "ONLY use when the question explicitly mentions a specific OS build (24H2, 23H2, 22H2) "
    "or words like 'latest', 'recent', 'newest patch', 'current update'. "
    "Do NOT use as a fallback when the knowledge graph returns no results — "
    "use embedding_search with a rephrased query instead. "
    "Do NOT use for general troubleshooting or vague symptoms."
),
```

- [ ] **Step 11: Run all tests — verify all 16 pass**

```
pytest tests/test_agent_routing.py tests/test_match_articles.py tests/test_clean_matches.py -v
```

```
pytest tests/ -v
```

Expected: **28 passed** (11 routing + 5 prompt content + 9 match_articles + 3 clean_matches)

- [ ] **Step 12: Commit**

```bash
git add src/agent/agent.py tests/test_agent_routing.py
git commit -m "feat: rewrite SYSTEM_PROMPT and tool descriptions to fix tool routing"
```
