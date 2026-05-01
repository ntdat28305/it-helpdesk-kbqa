"""
src/kg_build/kg_loader.py
Load entities và relations từ data/processed vào Neo4j.

Chạy:
    python -m src.kg_build.kg_loader --limit 5 --dry-run
    python -m src.kg_build.kg_loader
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/kg_loader.log")

PROCESSED_DIR = Path("data/processed")


# ── Neo4j connection ──────────────────────────────────────────

def get_driver():
    uri      = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not password:
        raise ValueError("Thiếu NEO4J_URI hoặc NEO4J_PASSWORD trong .env")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    logger.info("Kết nối Neo4j thành công")
    return driver


# ── Load một file vào Neo4j ───────────────────────────────────

def load_file(session, data: dict) -> tuple[int, int]:
    """
    Load entities và relations từ một file vào Neo4j.
    Dùng MERGE để tránh trùng lặp.
    Trả về (số entities, số relations) đã load.
    """
    metadata  = data.get("metadata", {})
    entities  = data.get("entities", [])
    relations = data.get("relations", [])

    article_id = metadata.get("article_id", "unknown")
    title      = metadata.get("title", "")
    url        = metadata.get("url", "")
    category   = metadata.get("category", "")

    # Tạo Article node
    session.run("""
        MERGE (a:Article {article_id: $article_id})
        SET a.title    = $title,
            a.url      = $url,
            a.category = $category
    """, article_id=article_id, title=title, url=url, category=category)

    # Tạo Entity nodes
    entity_count = 0
    for ent in entities:
        name = ent.get("name", "").strip()
        etype = ent.get("type", "Concept").strip()
        if not name:
            continue
        session.run("""
            MERGE (e:Entity {name: $name})
            SET e.type = $type
        """, name=name, type=etype)

        # Link entity với article
        session.run("""
            MATCH (a:Article {article_id: $article_id})
            MATCH (e:Entity {name: $name})
            MERGE (a)-[:MENTIONS]->(e)
        """, article_id=article_id, name=name)

        entity_count += 1

    # Tạo Relations
    relation_count = 0
    for rel in relations:
        source   = rel.get("source", "").strip()
        relation = rel.get("relation", "RELATED_TO").strip().upper()
        target   = rel.get("target", "").strip()

        if not source or not target:
            continue

        # Chỉ tạo relation nếu cả 2 entity đã tồn tại
        session.run(f"""
            MATCH (a:Entity {{name: $source}})
            MATCH (b:Entity {{name: $target}})
            MERGE (a)-[:{relation}]->(b)
        """, source=source, target=target)

        relation_count += 1

    return entity_count, relation_count


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False):
    files = sorted(PROCESSED_DIR.rglob("*.json"))

    if limit:
        files = files[:limit]

    total = len(files)
    logger.info(f"Load {total} files vào Neo4j...")

    if dry_run:
        for i, f in enumerate(files, 1):
            data = json.loads(f.read_text(encoding="utf-8"))
            logger.info(
                f"[{i}/{total}] {f.name} | "
                f"entities={len(data.get('entities', []))} | "
                f"relations={len(data.get('relations', []))}"
            )
        logger.info("=== DRY RUN xong ===")
        return

    driver = get_driver()
    total_ent, total_rel = 0, 0
    errors = 0

    with driver.session() as session:
        for i, filepath in enumerate(files, 1):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                ent, rel = load_file(session, data)
                total_ent += ent
                total_rel += rel
                logger.info(f"[{i}/{total}] {filepath.name} | +{ent} entities, +{rel} relations")
            except Exception as e:
                logger.error(f"Lỗi load {filepath.name}: {e}")
                errors += 1

    driver.close()
    logger.info(f"=== Xong: {total_ent} entities, {total_rel} relations, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)