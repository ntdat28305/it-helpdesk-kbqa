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


def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
    )


# ── Tool 1: Cypher query ──────────────────────────────────────

def cypher_search(entity_name: str) -> dict:
    """
    Tìm thông tin về một entity trong KG.
    Trả về: relations, articles liên quan.
    """
    driver = get_driver()
    results = {"entity": entity_name, "relations": [], "articles": []}

    with driver.session() as session:
        # Tìm relations của entity
        rels = session.run("""
            MATCH (e:Entity)-[r]-(e2:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN e.name AS src, type(r) AS rel, e2.name AS tgt
            LIMIT 20
        """, name=entity_name)
        results["relations"] = [
            {"src": r["src"], "rel": r["rel"], "tgt": r["tgt"]}
            for r in rels
        ]

        # Tìm articles đề cập entity
        arts = session.run("""
            MATCH (a:Article)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
            RETURN DISTINCT a.title AS title, a.url AS url
            LIMIT 5
        """, name=entity_name)
        results["articles"] = [
            {"title": r["title"], "url": r["url"]}
            for r in arts
        ]

    driver.close()
    return results


# ── Tool 3: BFS multi-hop ─────────────────────────────────────

def bfs_search(entity1: str, entity2: str) -> dict:
    """
    Tìm đường đi ngắn nhất giữa 2 entities trong KG.
    Dùng CONTAINS để match linh hoạt hơn.
    """
    driver = get_driver()
    results = {"entity1": entity1, "entity2": entity2, "path": []}

    with driver.session() as session:
        # Tìm tên node thực khớp nhất với entity1 và entity2
        n1 = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(e.name)
            RETURN e.name AS name
            ORDER BY size(e.name) ASC
            LIMIT 1
        """, name=entity1).single()

        n2 = session.run("""
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(e.name)
            RETURN e.name AS name
            ORDER BY size(e.name) ASC
            LIMIT 1
        """, name=entity2).single()

        if not n1 or not n2:
            logger.warning(f"Không tìm thấy node: {entity1} hoặc {entity2}")
            return results

        name1 = n1["name"]
        name2 = n2["name"]
        logger.info(f"BFS: '{entity1}' → '{name1}' | '{entity2}' → '{name2}'")

        path = session.run("""
            MATCH (a:Entity {name: $n1}), (b:Entity {name: $n2})
            MATCH p = shortestPath((a)-[*..6]-(b))
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
    driver.close()
    return results


# ── Community search ──────────────────────────────────────────

def get_community_context(entity_name: str, communities: dict) -> str:
    """
    Tìm community summary liên quan đến entity.
    """
    entity_lower = entity_name.lower()
    best_summary = ""
    best_count   = 0

    for comm_id, info in communities.items():
        nodes = [n.lower() for n in info.get("nodes", [])]
        count = sum(1 for n in nodes if entity_lower in n or n in entity_lower)
        if count > best_count:
            best_count   = count
            best_summary = info.get("summary", "")

    return best_summary