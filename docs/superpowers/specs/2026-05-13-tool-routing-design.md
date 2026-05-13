# Tool Routing Improvement — Design Spec

**Date:** 2026-05-13
**Status:** Approved

## Problem

Evaluation results show Tool Accuracy at 60%:
- `EMBEDDING→CYPHER`: 22 cases — agent uses `cypher_search` for product-name questions (Intune, Teams) without error codes
- `EMBEDDING→WEBSEARCH`: 9 cases — agent uses `web_search` as fallback when KG returns no results

Root cause: LLM tool selection in `_react_loop` is insufficiently constrained by the current `SYSTEM_PROMPT` and tool descriptions.

## Goal

- Tool Accuracy ≥ 75%
- EMBEDDING→CYPHER ≤ 5
- EMBEDDING→WEBSEARCH ≤ 3
- Only `src/agent/agent.py` changes — no other files touched

## Design

### 1. Pre-routing Logic (`_forced_tool`)

Extend `_forced_tool()` to return `"embedding_search"` as the hard default instead of `None`.

Add `_RE_BFS` regex to detect relationship questions.

Priority order:
1. Error code pattern → `cypher_search`
2. Version/recency pattern → `web_search`
3. Relationship pattern (two entities) → `bfs_search`
4. Everything else → `embedding_search` (new default)

```python
_RE_BFS = re.compile(
    r'\b(what causes|why does|how does|relationship between|'
    r'connection between|when .+ and .+)\b',
    re.IGNORECASE,
)

def _forced_tool(question: str) -> str | None:
    if _RE_ERROR_CODE.search(question):
        return "cypher_search"
    if _RE_WEBSEARCH.search(question):
        return "web_search"
    if _RE_BFS.search(question):
        return "bfs_search"
    return "embedding_search"
```

`tool_choice` is forced only at step 0. Steps 1+ remain `"auto"` — agent can escalate if needed.

### 2. SYSTEM_PROMPT Rewrite

```
TOOL SELECTION — apply the FIRST matching rule:

1. cypher_search — ONLY when question contains an EXPLICIT error code:
   • Hex: 0x80004005, 0xC000021A
   • Named: ERROR_INVALID_HANDLE, ERROR_ACCESS_DENIED
   • KB: KB5034441
   • HRESULT
   DO NOT use for product names alone (Intune, Teams, Azure AD).

2. bfs_search — ONLY for relationship between TWO named entities.

3. web_search — ONLY for explicit build numbers (24H2, 23H2) or
   "latest"/"recent"/"newest patch".
   DO NOT use as fallback when KG returns no results.

4. embedding_search — DEFAULT for everything else.

Rule 2 (fallback): if tool returns no results → retry embedding_search
with rephrased query. Do NOT switch to web_search.
```

Examples added matching actual testset patterns (Intune vague, Teams consent, MDM error codes).

### 3. Tool Description Updates

**cypher_search:** Add "Do NOT use for product names alone without an error code."

**web_search:** Add "Do NOT use as a fallback when the knowledge graph returns no results — use embedding_search with a rephrased query instead."

## Files Changed

| File | Change |
|------|--------|
| `src/agent/agent.py` | `_forced_tool()`, `_RE_BFS`, `SYSTEM_PROMPT`, `TOOLS[cypher_search].description`, `TOOLS[web_search].description` |

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Tool Accuracy | 60% | ≥ 75% |
| EMBEDDING→CYPHER | 22 | ≤ 5 |
| EMBEDDING→WEBSEARCH | 9 | ≤ 3 |
