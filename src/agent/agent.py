"""
src/agent/agent.py
ReAct Agent cho IT Helpdesk KBQA.

Dùng Groq function-calling để chạy vòng lặp Thought→Action→Observation
thật sự thay vì single-pass router.

Chạy thử:
    python -m src.agent.agent
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rapidfuzz import process as fuzz_process

from src.agent.neo4j_query import cypher_search, bfs_search, get_community_context
from src.agent.prompts import (
    IS_AMBIGUOUS_PROMPT,
    TOPIC_CHANGE_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
)
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/agent.log")

MAX_STEPS = 4
EMBEDDING_FUZZY_THRESHOLD = 55  # rapidfuzz score 0–100; lowered to improve match rate for vague queries
MAX_HISTORY_MESSAGES = 40       # hard cap before oldest messages are dropped

# ── Regex patterns for pre-routing ────────────────────────────
_RE_ERROR_CODE = re.compile(
    r'\b(0x[0-9A-Fa-f]+|ERROR_\w+|KB\d{6,7}|HRESULT\b)',
    re.IGNORECASE,
)
_RE_WEBSEARCH = re.compile(
    r'\b(latest|newest|recent|update|patch|release|version)\b.*\b(24H2|23H2|2[0-9]{3})\b'
    r'|\b(24H2|23H2)\b'   # Windows build codes thì ok match thẳng
    r'|\b(20[2-9][0-9])\b',  # standalone year (2020–2029) → likely version-specific query
    re.IGNORECASE,
)
_RE_FOLLOWUP = re.compile(
    r'\b(it|this|that|they|them|the (issue|problem|error|fix|solution)|'
    r'what about|any other|elaborate|more detail|next step|how do i fix|same)\b',
    re.IGNORECASE,
)


def _forced_tool(question: str) -> str | None:
    """Return Groq tool name to force on step 0, or None to let LLM decide."""
    if _RE_ERROR_CODE.search(question):
        return "cypher_search"
    if _RE_WEBSEARCH.search(question):
        return "web_search"
    return None

# ── ReAct system prompt ───────────────────────────────────────

SYSTEM_PROMPT = """You are an IT helpdesk assistant with access to a knowledge graph and web search.

Choose the right tool based on the question:
- cypher_search: specific error codes (0x..., ERROR_XXX, KB numbers) or exact product names
- embedding_search: vague symptoms, general issues, "not working", "keeps crashing"
- bfs_search: relationship between two specific IT entities ("what causes X when Y")
- web_search: recent updates, latest known issues, specific version numbers

Rules:
1. Call ONE tool per step — pick the most relevant one first.
2. If a tool returns no results, try a different tool or rephrase the entity.
3. Once you have enough information, provide a concise, actionable answer with steps.
4. Do NOT keep calling tools if you already have a good answer."""

# ── Groq tool schemas ─────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cypher_search",
            "description": (
                "Search the IT knowledge graph for a specific entity, error code, or product. "
                "Best for: error codes (0x..., ERROR_XXX, KB numbers), exact product/component names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": (
                            "IT entity, error code, or product name "
                            "(e.g. 'ERROR_INVALID_HANDLE', 'Teams', 'KB5034441')"
                        ),
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "embedding_search",
            "description": (
                "Find semantically similar IT concepts using GCN embeddings, then fetch related articles. "
                "Best for: vague symptoms, general issues, descriptive problems."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": (
                            "IT concept or symptom "
                            "(e.g. 'Smart Card', 'Network Connection', 'Teams Meeting Failure')"
                        ),
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bfs_search",
            "description": (
                "Find the relationship path between two IT entities in the knowledge graph. "
                "Best for: 'what causes X when Y', 'how does X relate to Y'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity1": {
                        "type": "string",
                        "description": "First IT entity (e.g. 'VPN Connection')",
                    },
                    "entity2": {
                        "type": "string",
                        "description": "Second IT entity (e.g. 'Teams Meeting Failure')",
                    },
                },
                "required": ["entity1", "entity2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for recent IT issues, latest Windows updates, or version-specific problems. "
                "Best for: recent updates, current known issues, specific version numbers. "
                "Do NOT use for general symptoms or vague troubleshooting — use embedding_search for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'Windows 11 24H2 BSOD after update 2024')",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_NAME_MAP = {
    "cypher_search":    "CYPHER",
    "embedding_search": "EMBEDDING",
    "bfs_search":       "BFS",
    "web_search":       "WEBSEARCH",
}


# ── Load resources ────────────────────────────────────────────

def load_resources() -> dict:
    resources = {}

    emb_path  = Path("models/embeddings/node_embeddings.npy")
    idx_path  = Path("models/embeddings/idx_to_name.json")
    name_path = Path("models/embeddings/name_to_idx.json")

    if emb_path.exists():
        emb = np.load(emb_path)
        resources["embeddings"] = emb / (
            np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        )
        resources["idx2name"] = json.loads(idx_path.read_text(encoding="utf-8"))
        resources["name2idx"] = {
            v: int(k) for k, v in resources["idx2name"].items()
        }
        logger.info(f"Loaded embeddings: {emb.shape}")

    comm_path = Path("data/community_summaries.json")
    if comm_path.exists():
        resources["communities"] = json.loads(
            comm_path.read_text(encoding="utf-8")
        )
        logger.info(f"Loaded {len(resources['communities'])} communities")

    sem_path       = Path("models/embeddings/node_semantic_embeddings.npy")
    retriever_path = Path("models/retriever")
    if sem_path.exists() and retriever_path.exists():
        from sentence_transformers import SentenceTransformer as _ST
        resources["node_semantic_emb"] = np.load(sem_path)
        resources["retriever"] = _ST(str(retriever_path))
        logger.info(f"Loaded finetuned retriever: {retriever_path}")
    else:
        logger.warning(
            "Finetuned retriever khong tim thay -- fallback sang fuzzy match. "
            "Chay finetune_retriever.py va train_gcn de kich hoat semantic search."
        )

    return resources


# ── Groq plain LLM call (used for pre-processing checks) ─────

_MODEL_FAST = "llama-3.1-8b-instant"
_MODEL_STRONG = "llama-3.3-70b-versatile"


def llm_call(
    client: Groq,
    prompt: str,
    max_tokens: int = 512,
    history: list[dict] | None = None,
    model: str = _MODEL_FAST,
) -> str:
    try:
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return None


# ── Tool implementations ──────────────────────────────────────

def embedding_search(
    query_entity: str,
    resources: dict,
    top_k: int = 5,
) -> list[str]:
    if "embeddings" not in resources:
        return []

    embeddings = resources["embeddings"]
    idx2name   = resources["idx2name"]
    name2idx   = resources["name2idx"]
    node_names = list(name2idx.keys())

    if "retriever" in resources and "node_semantic_emb" in resources:
        query_vec       = resources["retriever"].encode(query_entity, normalize_embeddings=True)
        semantic_scores = resources["node_semantic_emb"] @ query_vec
        matched_idx     = int(np.argmax(semantic_scores))
        matched_name    = idx2name[str(matched_idx)]
        logger.info(
            f"Semantic match: '{query_entity}' -> '{matched_name}' "
            f"(score={semantic_scores[matched_idx]:.3f})"
        )
    else:
        match_result = fuzz_process.extractOne(query_entity, node_names)
        if not match_result or match_result[1] < EMBEDDING_FUZZY_THRESHOLD:
            logger.warning(f"Khong tim duoc node gan voi: {query_entity}")
            return []
        matched_name = match_result[0]
        matched_idx  = name2idx[matched_name]
        logger.info(f"Fuzzy match: '{query_entity}' -> '{matched_name}'")

    query_vec = embeddings[matched_idx]
    scores    = embeddings @ query_vec
    scores[matched_idx] = -1

    top_ids = np.argsort(scores)[::-1][:top_k]
    return [idx2name[str(tid)] for tid in top_ids]


def web_search(query: str) -> dict:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp   = client.search(query=query, max_results=3, include_answer=True)
        results = []
        urls    = []
        if resp.get("answer"):
            results.append(f"Quick Answer: {resp['answer']}")
        for r in resp.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:300]}")
            urls.append(r["url"])
        return {"text": "\n".join(results), "urls": urls}
    except Exception as e:
        logger.warning(f"Web search error: {e}")
        return {"text": "", "urls": []}


# ── Main Agent ────────────────────────────────────────────────

class ITHelpdeskAgent:
    def __init__(self):
        self.client    = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
        self.resources = load_resources()
        self.history: list[dict] = []
        logger.info("Agent initialized")

    def reset_history(self):
        self.history = []
        logger.info("Conversation history reset")

    def _detect_topic_change(self, question: str) -> bool:
        if len(self.history) < 2:
            return False
        prev_question = self.history[-2]["content"]
        result = llm_call(
            self.client,
            TOPIC_CHANGE_PROMPT.format(prev=prev_question, current=question),
            max_tokens=5,
        )
        changed = bool(result) and result.lower().strip().startswith("yes")
        logger.info(f"Topic change: '{prev_question[:40]}' → '{question[:40]}' = {changed}")
        return changed

    def _plan(self, question: str) -> str:
        """Generate a lightweight planning note before the ReAct loop."""
        try:
            note = llm_call(
                self.client,
                PLAN_PROMPT.format(question=question),
                max_tokens=80,
            )
            result = (note or "").strip()
            logger.info(f"Plan: {result[:100]}")
            return result
        except Exception:
            return ""

    def _reflect(
        self,
        question: str,
        answer_text: str,
        num_sources: int,
        tool_used: str,
    ) -> dict:
        """Self-evaluate the generated answer; return confidence metadata."""
        _default = {"is_sufficient": True, "confidence": "medium", "reason": "Reflection parse failed"}
        if not answer_text:
            return {"is_sufficient": False, "confidence": "low", "reason": "Empty answer"}
        try:
            raw = llm_call(
                self.client,
                REFLECT_PROMPT.format(
                    question=question,
                    answer=answer_text[:600],
                    num_sources=num_sources,
                    tools_used=tool_used,
                ),
                max_tokens=80,
                model=_MODEL_STRONG,
            )
            if not raw:
                return _default
            # Strip markdown fences if present
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            return {
                "is_sufficient": bool(parsed.get("is_sufficient", True)),
                "confidence":    str(parsed.get("confidence", "medium")),
                "reason":        str(parsed.get("reason", "")),
            }
        except Exception as e:
            logger.warning(f"Reflection parse error: {e}")
            return _default

    def _execute_tool(
        self, fn_name: str, args: dict
    ) -> tuple[str, list[str], str]:
        """Dispatch a tool call; return (observation, sources, entity)."""
        sources: list[str] = []

        if fn_name == "cypher_search":
            entity = args.get("entity", "")
            data   = cypher_search(entity)
            parts  = []
            if data.get("relations"):
                rels = "\n".join(
                    f"  {r['src']} --[{r['rel']}]--> {r['tgt']}"
                    for r in data["relations"][:10]
                )
                parts.append(f"Knowledge Graph Relations:\n{rels}")
            if data.get("articles"):
                arts = "\n".join(
                    f"  - {a['title']} ({a['url']})"
                    for a in data["articles"][:5]
                )
                parts.append(f"Related Articles:\n{arts}")
                sources = [a["url"] for a in data["articles"][:5]]
            obs = "\n".join(parts) if parts else "No results found in knowledge graph."
            return obs, sources, entity

        if fn_name == "embedding_search":
            entity  = args.get("entity", "")
            similar = embedding_search(entity, self.resources, top_k=10)
            art_lines: list[str] = []
            for node in similar:
                node_data = cypher_search(node)
                for art in node_data.get("articles", []):
                    if art["url"] not in sources:
                        sources.append(art["url"])
                        art_lines.append(f"  - {art['title']} ({art['url']})")
            if similar:
                obs_parts = [f"Similar IT concepts: {', '.join(similar[:5])}"]
                if art_lines:
                    obs_parts.append("Related Articles:\n" + "\n".join(art_lines[:8]))
                obs = "\n".join(obs_parts)
            else:
                obs = "No similar entities found."
            return obs, sources, entity

        if fn_name == "bfs_search":
            e1 = args.get("entity1", "")
            e2 = args.get("entity2", "")
            data = bfs_search(e1, e2)
            parts = []
            if isinstance(data, dict) and data.get("path"):
                for p in data["path"][:2]:
                    parts.append("Path: " + " → ".join(p["nodes"]))
            for ent in [e1, e2]:
                node_data = cypher_search(ent)
                for art in node_data.get("articles", []):
                    if art["url"] not in sources:
                        sources.append(art["url"])
            obs = "\n".join(parts) if parts else f"No path found between '{e1}' and '{e2}'."
            return obs, sources, e1

        if fn_name == "web_search":
            query  = args.get("query", "")
            result = web_search(query)
            sources.extend(result.get("urls", []))
            obs = result.get("text") or "No web results found."
            return obs, sources, ""

        return "Unknown tool.", [], ""

    def _react_loop(
        self, question: str, history_context: list[dict], plan_note: str = ""
    ) -> dict:
        """Run Thought→Action→Observation loop via Groq function calling."""
        _plan_note = plan_note.strip()
        system_content = (
            SYSTEM_PROMPT + f"\n\n[Planning note: {_plan_note}]"
            if _plan_note
            else SYSTEM_PROMPT
        )
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            *history_context,
            {"role": "user", "content": question},
        ]

        tool_used       = "WEBSEARCH"
        tool_first_used = ""   # tracks the first tool chosen (routing decision)
        entity          = ""
        sources: list[str]      = []
        observations: list[str] = []
        steps:   list[dict]     = []
        answer_text  = ""
        global_step  = 0  # increments per tool call or thought step

        used_tool_inputs: set[tuple] = set()

        first_tool = _forced_tool(question)

        for step in range(MAX_STEPS):
            logger.info(f"ReAct step {step + 1}/{MAX_STEPS}")
            # Force the first tool call when regex gives high-confidence routing
            tool_choice = (
                {"type": "function", "function": {"name": first_tool}}
                if step == 0 and first_tool
                else "auto"
            )
            try:
                resp = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    temperature=0,
                    max_tokens=768,
                )
            except Exception as e:
                logger.error(f"LLM call failed at step {step + 1}: {e}")
                break

            msg = resp.choices[0].message

            if not msg.tool_calls:
                answer_text = (msg.content or "").strip()
                if answer_text:
                    global_step += 1
                    steps.append({
                        "step":        global_step,
                        "tool":        "thought",
                        "input":       "",
                        "observation": answer_text[:300],
                    })
                logger.info(f"Final answer at step {step + 1}")
                break

            # Only execute the first tool call per LLM turn to avoid parallel-call sprawl
            first_tc = msg.tool_calls[0]
            messages.append({
                "role":    "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id":   first_tc.id,
                        "type": "function",
                        "function": {
                            "name":      first_tc.function.name,
                            "arguments": first_tc.function.arguments,
                        },
                    }
                ],
            })

            for tc in [first_tc]:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info(f"Tool call: {fn_name}({args})")
                _dedup_key = (fn_name, json.dumps(args, sort_keys=True))
                if _dedup_key in used_tool_inputs:
                    obs, step_sources, step_entity = (
                        "[Skipped: same tool+input already used in this query]",
                        [],
                        "",
                    )
                    logger.info("Tool dedup: skipped duplicate call")
                else:
                    used_tool_inputs.add(_dedup_key)
                    obs, step_sources, step_entity = self._execute_tool(fn_name, args)

                tool_used = _TOOL_NAME_MAP.get(fn_name, "WEBSEARCH")
                if not tool_first_used:
                    tool_first_used = tool_used
                if step_entity:
                    entity = step_entity
                for s in step_sources:
                    if s not in sources:
                        sources.append(s)
                observations.append(obs)
                logger.info(f"Observation: {obs[:120]}")

                global_step += 1
                step_input = (
                    args.get("entity")
                    or args.get("query")
                    or f"{args.get('entity1', '')} ↔ {args.get('entity2', '')}"
                )
                steps.append({
                    "step":        global_step,
                    "tool":        _TOOL_NAME_MAP.get(fn_name, fn_name),
                    "input":       step_input,
                    "observation": obs[:300],
                })

                _EMPTY_OBS = (
                    "No results found",
                    "No similar entities",
                    "No path found",
                    "No web results",
                )
                obs_content = obs
                if any(obs.startswith(m) for m in _EMPTY_OBS):
                    obs_content = (
                        obs + "\n\n[Hint: previous search returned no results. "
                        "Try rephrasing the entity with broader or more specific terms. "
                        "Only switch to web_search if the question is about a recent update or version-specific issue.]"
                    )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      obs_content,
                })

        # Synthesize if loop exhausted without a direct answer
        if not answer_text:
            if observations:
                synthesis = "\n\n---\n\n".join(observations)
                answer_text = llm_call(
                    self.client,
                    f"Based on the findings below, give a concise actionable answer.\n\n"
                    f"Question: {question}\n\nFindings:\n{synthesis}",
                    max_tokens=512,
                    model=_MODEL_STRONG,
                ) or ""
            if not answer_text:
                answer_text = (
                    "I couldn't find relevant information. "
                    "Please try rephrasing or contact IT support."
                )

        # Community context for the context field
        community_summary = ""
        if "communities" in self.resources and entity:
            community_summary = get_community_context(
                entity, self.resources["communities"]
            )

        _EMPTY_MARKERS = (
            "No results found",
            "No similar entities",
            "No path found",
            "No web results",
        )
        useful_obs = [
            o for o in observations
            if not any(o.startswith(m) for m in _EMPTY_MARKERS)
        ]
        context_parts = []
        if community_summary:
            context_parts.append(f"Related IT Domain: {community_summary}")
        context_parts.extend(useful_obs if useful_obs else observations)
        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found."

        return {
            "question":     question,
            "tool_used":    tool_first_used or tool_used,
            "entity":       entity,
            "answer":       answer_text,
            "sources":      sources,
            "context":      context,
            "steps":        steps,
            "observations": observations,
        }

    def answer(self, question: str) -> dict:
        logger.info(f"Question: {question}")

        # Guard: ambiguous follow-up with no prior context → ask for clarification
        # Regex is a fast pre-filter; LLM confirms before we reject, because
        # \bit\b also matches "it" when the question already names a clear subject
        # (e.g. "my wifi is error, how to fix it?" — NOT ambiguous).
        if not self.history and _RE_FOLLOWUP.search(question):
            _amb_check = llm_call(
                self.client,
                IS_AMBIGUOUS_PROMPT.format(question=question),
                max_tokens=5,
            )
            _truly_ambiguous = bool(_amb_check) and _amb_check.lower().strip().startswith("yes")
        else:
            _truly_ambiguous = False

        if _truly_ambiguous:
            clarification = (
                "Could you provide more details? "
                "For example, what device, error code, or issue are you experiencing?"
            )
            logger.info("Ambiguous question with empty history → returning clarification")
            return {
                "question":          question,
                "tool_used":         "",
                "entity":            "",
                "answer":            clarification,
                "sources":           [],
                "context":           "",
                "steps":             [],
                "observations":      [],
                "plan_note":         "",
                "confidence":        "low",
                "reflection_reason": "Ambiguous question with no prior context.",
            }

        # Topic-change + ambiguity pre-processing
        if self.history:
            # Fast regex pre-check: obvious follow-up → skip LLM call
            if _RE_FOLLOWUP.search(question):
                ambiguous = True
            else:
                _amb = llm_call(
                    self.client,
                    IS_AMBIGUOUS_PROMPT.format(question=question),
                    max_tokens=5,
                )
                ambiguous = bool(_amb) and _amb.lower().strip().startswith("yes")

            if not ambiguous and self._detect_topic_change(question):
                logger.info("Topic changed → auto reset history")
                self.reset_history()

        history_context = self.history[-10:] if self.history else []
        plan_note = self._plan(question)
        result = self._react_loop(question, history_context, plan_note)

        reflection = self._reflect(
            question,
            result["answer"],
            len(result["sources"]),
            result["tool_used"],
        )
        final_confidence = reflection["confidence"]
        final_reason     = reflection["reason"]

        if not reflection["is_sufficient"] and result.get("observations"):
            synthesis = "\n\n---\n\n".join(result["observations"])
            hint = reflection["reason"]
            new_answer = llm_call(
                self.client,
                f"Based on the findings below, give a concise actionable answer.\n\n"
                f"Question: {question}\n\nFindings:\n{synthesis}\n\n"
                f"Note: A previous answer was flagged as insufficient because: {hint}\n"
                f"Make sure to address this specifically.",
                max_tokens=512,
                model=_MODEL_STRONG,
            )
            if new_answer:
                result["answer"] = new_answer
                final_confidence = "medium"
                final_reason     = f"[Re-synthesized] {hint}"
                logger.info("Reflection triggered re-synthesis")
            else:
                logger.warning("Re-synthesis returned None; keeping original answer")

        result["plan_note"]         = plan_note
        result["confidence"]        = final_confidence
        result["reflection_reason"] = final_reason

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": result["answer"]})
        if len(self.history) >= MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

        logger.info(f"Confidence: {result['confidence']} | Reason: {result['reflection_reason'][:80]}")
        logger.info(f"Answer: {result['answer'][:100]}...")
        return result


# ── CLI test ──────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ITHelpdeskAgent()

    print("\n" + "=" * 60)
    print("TEST: ReAct loop — multi-turn conversation")

    conv = [
        "My Teams meeting keeps dropping",
        "What about VPN? Could that be the cause?",
        "How do I fix it?",
        "How to fix error 0x80070005 Access Denied?",
        "What is the relationship between VPN and Teams failure?",
        "Latest Windows 11 24H2 update causing BSOD",
    ]

    for q in conv:
        print(f"\nUser: {q}")
        result = agent.answer(q)
        print(f"Tool: {result['tool_used']} | Entity: {result['entity']}")
        print(f"Agent: {result['answer'][:200]}")
