"""
src/agent/agent.py
ReAct Agent cho IT Helpdesk KBQA.

Fix:
1. BFS Sources — lấy articles từ cả 2 entities
2. Entity extraction — phát hiện câu mơ hồ dùng IS_AMBIGUOUS_PROMPT
3. Fallback mechanism — nếu tool rỗng → WEBSEARCH
4. Conversation history — nhớ context từ câu trước
5. Topic change detection — tự reset history khi topic thay đổi

Chạy thử:
    python -m src.agent.agent
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import result

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from rapidfuzz import process as fuzz_process

from src.agent.neo4j_query import cypher_search, bfs_search, get_community_context
from src.agent.prompts import (
    ROUTER_PROMPT,
    ENTITY_EXTRACT_PROMPT,
    ANSWER_PROMPT,
    BFS_ENTITY_PROMPT,
    IS_AMBIGUOUS_PROMPT,
    TOPIC_CHANGE_PROMPT,
)
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/agent.log")


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
            v: int(k)
            for k, v in resources["idx2name"].items()
        }
        logger.info(f"Loaded embeddings: {emb.shape}")

    comm_path = Path("data/community_summaries.json")
    if comm_path.exists():
        resources["communities"] = json.loads(
            comm_path.read_text(encoding="utf-8")
        )
        logger.info(f"Loaded {len(resources['communities'])} communities")

    return resources


# ── Groq LLM call ─────────────────────────────────────────────

def llm_call(
    client: Groq,
    prompt: str,
    max_tokens: int = 512,
    history: list[dict] | None = None,
) -> str:
    try:
        messages = history.copy() if history else []
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return ""


# ── Tool 2: Embedding search ──────────────────────────────────

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

    match_result = fuzz_process.extractOne(query_entity, node_names)
    if not match_result or match_result[1] < 50:
        logger.warning(f"Không tìm được node gần với: {query_entity}")
        return []

    matched_name = match_result[0]
    matched_idx  = name2idx[matched_name]
    logger.info(f"Fuzzy match: '{query_entity}' → '{matched_name}'")

    query_vec = embeddings[matched_idx]
    scores    = embeddings @ query_vec
    scores[matched_idx] = -1

    top_ids = np.argsort(scores)[::-1][:top_k]
    return [idx2name[str(tid)] for tid in top_ids]


# ── Tool 4: Web Search ────────────────────────────────────────

def web_search(query: str) -> dict:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        resp   = client.search(
            query=query,
            max_results=3,
            include_answer=True,
        )
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


# ── Fallback check ────────────────────────────────────────────

def is_empty_result(tool: str, tool_result) -> bool:
    if tool_result is None:
        return True
    if tool == "CYPHER":
        return not tool_result.get("relations") and not tool_result.get("articles")
    if tool == "EMBEDDING":
        return len(tool_result) == 0
    if tool == "BFS":
        return not tool_result.get("path")
    if tool == "WEBSEARCH":
        return not tool_result.get("text")
    return True


# ── Format context ────────────────────────────────────────────

def format_context(
    tool_used: str,
    tool_result,
    community_summary: str = "",
) -> str:
    parts = []

    if community_summary:
        parts.append(f"Related IT Domain: {community_summary}")

    if tool_used == "CYPHER":
        data = tool_result
        if data.get("relations"):
            rels = "\n".join(
                f"  {r['src']} --[{r['rel']}]--> {r['tgt']}"
                for r in data["relations"][:10]
            )
            parts.append(f"Knowledge Graph Relations:\n{rels}")
        if data.get("articles"):
            arts = "\n".join(
                f"  - {a['title']} ({a['url']})"
                for a in data["articles"][:3]
            )
            parts.append(f"Related Articles:\n{arts}")

    elif tool_used == "EMBEDDING":
        nodes = tool_result
        if nodes:
            parts.append(
                f"Similar IT concepts and components: {', '.join(nodes[:8])}\n"
                f"These are the most relevant IT entities related to your question."
            )

    elif tool_used == "BFS":
        data = tool_result
        if isinstance(data, dict) and data.get("path"):
            for p in data["path"][:2]:
                path_str = " → ".join(p["nodes"])
                parts.append(f"Relationship path: {path_str}")

    elif tool_used == "WEBSEARCH":
        text = tool_result.get("text", "") if isinstance(tool_result, dict) else ""
        if text:
            parts.append(
                f"Web Search Results:\n{text}\n"
                f"Use these results to provide a specific, actionable answer."
            )

    return "\n\n".join(parts) if parts else "No relevant information found."


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
        """
        Phát hiện topic thay đổi so với câu hỏi trước.
        Trả về True nếu topic khác → cần reset history.
        """
        if not self.history:
            return False
        prev_question = self.history[-2]["content"]
        result = llm_call(
        self.client,
        TOPIC_CHANGE_PROMPT.format(
            prev=prev_question,
            current=question,
            ),
            max_tokens=5,
        ).lower().strip()
        changed = result.startswith("yes")
        logger.info(f"Topic change: '{prev_question[:40]}' → '{question[:40]}' = {changed}")
        return changed

    def answer(self, question: str) -> dict:
        logger.info(f"Question: {question}")

        # Fix 5: Tự động phát hiện topic change → reset history
        # Fix 5: Tự động phát hiện topic change → reset history
# Chỉ check khi câu hỏi KHÔNG mơ hồ
        if self.history:
            ambiguous_pre = llm_call(
                self.client,
                IS_AMBIGUOUS_PROMPT.format(question=question),
                max_tokens=5,
            ).lower().strip().startswith("yes")


            if not ambiguous_pre:
                topic_changed = self._detect_topic_change(question)
                if topic_changed:
                    logger.info("Topic changed → auto reset history")
                    self.reset_history()

        # Bước 1: Chọn tool
        tool = llm_call(
            self.client,
            ROUTER_PROMPT.format(question=question),
            max_tokens=10,
            history=self.history[-4:] if self.history else None,
        ).upper().strip()

        valid_tools = {"CYPHER", "EMBEDDING", "BFS", "WEBSEARCH"}
        if tool not in valid_tools:
            tool = "EMBEDDING"
        logger.info(f"Tool selected: {tool}")

        # Bước 2: Phát hiện câu mơ hồ → dùng history cho entity extraction
        is_ambiguous = False
        if self.history:
            is_ambiguous = llm_call(
                self.client,
                IS_AMBIGUOUS_PROMPT.format(question=question),
                max_tokens=5,
            ).lower().strip().startswith("yes")
            logger.info(f"Ambiguous: {is_ambiguous}")

        entity = llm_call(
            self.client,
            ENTITY_EXTRACT_PROMPT.format(question=question),
            max_tokens=20,
            history=self.history[-2:] if is_ambiguous else None,
        ).strip()
        logger.info(f"Entity extracted: {entity}")

        # Bước 3: Gọi tool
        tool_result = None
        sources     = []

        if tool == "CYPHER":
            tool_result = cypher_search(entity)
            sources = [a["url"] for a in tool_result.get("articles", [])]

        elif tool == "EMBEDDING":
            similar_nodes = embedding_search(entity, self.resources)
            tool_result   = similar_nodes
            for node in similar_nodes[:3]:
                node_data = cypher_search(node)
                for art in node_data.get("articles", []):
                    if art["url"] not in sources:
                        sources.append(art["url"])

        elif tool == "BFS":
            two_entities = llm_call(
                self.client,
                BFS_ENTITY_PROMPT.format(question=question),
                max_tokens=30,
            )
            parts = two_entities.split("|")
            e1 = parts[0].strip() if len(parts) > 0 else entity
            e2 = parts[1].strip() if len(parts) > 1 else ""
            if e1:
                entity = e1.replace("Entities:", "").replace("entities:", "").strip()
                for prefix in ["Entity:", "entity:", "1.", "2."]:
                    entity = entity.replace(prefix, "").strip()
            if e2:
                tool_result = bfs_search(e1, e2)
                for ent in [e1, e2]:
                    node_data = cypher_search(ent)
                    for art in node_data.get("articles", []):
                        if art["url"] not in sources:
                            sources.append(art["url"])
                    similar = embedding_search(ent, self.resources, top_k=2)
                    for node in similar:
                        node_data = cypher_search(node)
                        for art in node_data.get("articles", []):
                            if art["url"] not in sources:
                                sources.append(art["url"])
            else:
                tool_result = cypher_search(entity)
                tool = "CYPHER"

        elif tool == "WEBSEARCH":
            tool_result = web_search(question)
            sources.extend(tool_result.get("urls", []))

        # Fix 3: Fallback sang WEBSEARCH nếu rỗng
        if is_empty_result(tool, tool_result) and tool != "WEBSEARCH":
            logger.info(f"Tool {tool} rỗng → fallback WEBSEARCH")
            tool_result = web_search(question)
            sources.extend(tool_result.get("urls", []))
            tool = "WEBSEARCH"

        # Bước 4: Community context
        community_summary = ""
        if "communities" in self.resources:
            community_summary = get_community_context(
                entity, self.resources["communities"]
            )

        # Bước 5: Format context
        context = format_context(tool, tool_result, community_summary)

        # Bước 6: Sinh câu trả lời
        answer = llm_call(
            self.client,
            ANSWER_PROMPT.format(question=question, context=context),
            max_tokens=512,
            history=self.history[-4:] if self.history else None,
        )

        # Cập nhật history
        self.history.append({"role": "user",     "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Giới hạn 10 turns
        if len(self.history) > 20:
            self.history = self.history[-20:]
        display_entity = e1 if tool == "BFS" and 'e1' in dir() else entity

        result = {
            "question":  question,
            "tool_used": tool,
            "entity":    display_entity,
            "answer":    answer,
            "sources":   sources,
            "context":   context,
        }

        logger.info(f"Answer: {answer[:100]}...")
        return result


# ── CLI test ──────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ITHelpdeskAgent()

    print("\n" + "="*60)
    print("TEST: Topic change detection + Conversation")

    conv = [
    # ── Tool: EMBEDDING — câu mơ hồ ──
    "My Teams meeting keeps dropping",
    "What about VPN? Could that be the cause?",
    "How do I fix it?",

    # ── Tool: CYPHER — error code cụ thể ──
    "How to fix error 0x80070005 Access Denied?",
    "What does KB5034441 fix?",

    # ── Tool: BFS — quan hệ 2 entity ──
    "What is the relationship between VPN and Teams failure?",
    "How does Windows Update affect network connection?",

    # ── Tool: WEBSEARCH — recent/version cụ thể ──
    "Latest Windows 11 24H2 update causing BSOD",
    "Windows 11 KB5032288 network issues 2024",
]

    for q in conv:
        print(f"\nUser: {q}")
        result = agent.answer(q)
        print(f"Tool: {result['tool_used']} | Entity: {result['entity']}")
        print(f"Agent: {result['answer'][:150]}")