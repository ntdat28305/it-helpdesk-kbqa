"""
src/agent/neo4j_query.py
Các hàm query Neo4j cho Agent tools.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from src.utils.logger import get_logger
logger = get_logger(__name__)

load_dotenv()

_driver = None

COMMUNITY_SIMILARITY_THRESHOLD = 0.25


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
        )
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ── Tool 1: Cypher query ──────────────────────────────────────

def cypher_search(entity_name: str) -> dict:
    """
    Tìm thông tin về một entity trong KG.
    Trả về: relations, articles liên quan.
    """
    if not entity_name or not entity_name.strip():
        return {"entity": entity_name, "relations": [], "articles": []}

    # Normalize để tăng recall khi LLM extract tên sai dạng
    name_norm = entity_name.strip().lower()

    driver = get_driver()
    results = {"entity": entity_name, "relations": [], "articles": []}
    try:
        with driver.session() as session:
            rows = session.run("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS $name
                WITH e
                OPTIONAL MATCH (e)-[r]-(e2:Entity)
                OPTIONAL MATCH (a:Article)-[:MENTIONS]->(e)
                RETURN
                collect(DISTINCT CASE WHEN r IS NOT NULL
                    THEN {src: e.name, rel: type(r), tgt: e2.name} END)[..20] AS relations,
                collect(DISTINCT CASE WHEN a IS NOT NULL
                    THEN {title: a.title, url: a.url} END)[..5]  AS articles
            """, name=name_norm)
            row = rows.single()
            if row:
                results["relations"] = [x for x in (row["relations"] or []) if x]
                results["articles"]  = [x for x in (row["articles"]  or []) if x]
    except Exception as e:
        logger.error(f"Neo4j cypher_search error: {e}")
    return results


# ── Tool 3: BFS multi-hop ─────────────────────────────────────

def bfs_search(entity1: str, entity2: str) -> dict:
    """
    Tìm đường đi ngắn nhất giữa 2 entities trong KG.
    Dùng CONTAINS để match linh hoạt hơn.
    """
    if not entity1 or not entity2:
        return {"entity1": entity1, "entity2": entity2, "path": []}

    driver = get_driver()
    results = {"entity1": entity1, "entity2": entity2, "path": []}

    # Ưu tiên: exact match > prefix match > contains
    resolve_query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(e.name)
        RETURN e.name AS name
        ORDER BY
          CASE WHEN toLower(e.name) = toLower($name) THEN 0
               WHEN toLower(e.name) STARTS WITH toLower($name) THEN 1
               ELSE 2 END,
          size(e.name) ASC
        LIMIT 1
    """

    try:
        with driver.session() as session:
            n1 = session.run(resolve_query, name=entity1).single()
            n2 = session.run(resolve_query, name=entity2).single()

            if not n1 or not n2:
                logger.warning(f"Không tìm thấy node: {entity1} hoặc {entity2}")
                return results

            name1 = n1["name"]
            name2 = n2["name"]
            logger.info(f"BFS: '{entity1}' → '{name1}' | '{entity2}' → '{name2}'")

            if name1 == name2:
                logger.warning(f"BFS: cả 2 entity resolve về cùng node '{name1}', bỏ qua")
                return results

            path = session.run("""
                MATCH (a:Entity {name: $n1}), (b:Entity {name: $n2})
                MATCH p = allShortestPaths((a)-[*..4]-(b))
                RETURN [n in nodes(p) | n.name] AS path_nodes,
                    length(p) AS path_length
                LIMIT 3
            """, n1=name1, n2=name2)

            for r in path:
                clean_nodes = [n for n in r["path_nodes"] if n is not None]
                if clean_nodes:
                    results["path"].append({
                        "nodes": clean_nodes,
                        "length": r["path_length"],
                    })
    except Exception as e:
        logger.error(f"Neo4j bfs_search error: {e}")
    return results


# ── Community search ──────────────────────────────────────────

# Cache: maps entity_lower → best_community_id; built lazily on first call
_community_index: dict[str, str] | None = None
_community_summaries: dict | None = None


def _build_community_index(communities: dict) -> dict[str, str]:
    """Build entity_lower → comm_id lookup once per communities dict."""
    index: dict[str, str] = {}
    for comm_id, info in communities.items():
        for node in info.get("nodes", []):
            index[node.lower()] = comm_id
    return index


def get_community_context(
    entity_name: str,
    communities: dict,
    community_embs=None,
    community_ids: list | None = None,
    retriever=None,
) -> str:
    """Find community summary for entity via embedding similarity (preferred)
    or substring matching (fallback when embeddings not available)."""
    if not entity_name or not communities:
        return ""

    # ── Embedding path (preferred) ────────────────────────────
    if community_embs is not None and community_ids is not None and retriever is not None:
        import numpy as np
        query_vec = retriever.encode(entity_name, normalize_embeddings=True)
        scores = community_embs @ query_vec
        best_idx = int(np.argmax(scores))
        if scores[best_idx] >= COMMUNITY_SIMILARITY_THRESHOLD:
            return communities[community_ids[best_idx]].get("summary", "")
        # Score too low — fall through to substring matching

    # ── Substring fallback ────────────────────────────────────
    global _community_index, _community_summaries

    if _community_index is None or communities is not _community_summaries:
        _community_index    = _build_community_index(communities)
        _community_summaries = communities

    entity_lower = entity_name.lower()
    best_comm_id = None
    best_count   = 0

    for node_lower, comm_id in _community_index.items():
        if entity_lower in node_lower or node_lower in entity_lower:
            info  = communities[comm_id]
            count = sum(
                1 for n in info.get("nodes", [])
                if entity_lower in n.lower() or n.lower() in entity_lower
            )
            if count > best_count:
                best_count   = count
                best_comm_id = comm_id

    if best_comm_id is None:
        return ""
    return communities[best_comm_id].get("summary", "")