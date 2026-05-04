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
import re
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
    _ensure_schema(driver)
    return driver


def _ensure_schema(driver):
    """Tạo constraints và indexes nếu chưa tồn tại."""
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT article_id IF NOT EXISTS "
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE"
        )
        session.run(
            "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]"
        )
    logger.info("Schema (constraints + indexes) đã sẵn sàng")


# ── Load một file vào Neo4j ───────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normalize entity name: strip + collapse whitespace để tránh duplicate."""
    return " ".join(name.strip().split())


def _load_file_tx(tx, data: dict) -> tuple[int, int]:
    """Transaction function — được gọi trong execute_write để đảm bảo atomicity."""
    metadata  = data.get("metadata", {})
    entities  = data.get("entities", [])
    relations = data.get("relations", [])

    article_id = metadata.get("article_id", "unknown")
    title      = metadata.get("title", "")
    url        = metadata.get("url", "")
    category   = metadata.get("category", "")

    # Tạo Article node
    tx.run("""
        MERGE (a:Article {article_id: $article_id})
        SET a.title    = $title,
            a.url      = $url,
            a.category = $category
    """, article_id=article_id, title=title, url=url, category=category)

    # Batch UNWIND: 1 query cho toàn bộ entities + MENTIONS links
    valid_entities = [
        {"name": _normalize_name(e.get("name", "")), "type": e.get("type", "Concept").strip()}
        for e in entities
        if e.get("name", "").strip()
    ]
    if valid_entities:
        tx.run("""
            UNWIND $entities AS ent
            MERGE (e:Entity {name: ent.name})
            SET e.type = ent.type
            WITH e, ent
            MATCH (a:Article {article_id: $article_id})
            MERGE (a)-[:MENTIONS]->(e)
        """, entities=valid_entities, article_id=article_id)

    entity_count = len(valid_entities)

    # Tạo Relations
    relation_count = 0
    for rel in relations:
        source   = _normalize_name(rel.get("source", ""))
        relation = rel.get("relation", "RELATED_TO").strip().upper()
        target   = _normalize_name(rel.get("target", ""))

        if not source or not target:
            continue

        # Chỉ cho phép relation type hợp lệ (tránh Cypher injection)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", relation):
            logger.warning(f"Relation type không hợp lệ '{relation}', dùng RELATED_TO")
            relation = "RELATED_TO"

        # Chỉ tạo relation nếu cả 2 entity đã tồn tại
        result = tx.run(f"""
            MATCH (a:Entity {{name: $source}})
            MATCH (b:Entity {{name: $target}})
            MERGE (a)-[:{relation}]->(b)
            RETURN 1 AS created
        """, source=source, target=target)
        if not result.single():
            logger.warning(f"Relation bỏ qua (entity không tìm thấy): '{source}' -[{relation}]-> '{target}'")
            continue

        relation_count += 1

    return entity_count, relation_count


def load_file(session, data: dict) -> tuple[int, int]:
    """
    Load entities và relations từ một file vào Neo4j trong 1 atomic transaction.
    Trả về (số entities, số relations) đã load.
    """
    return session.execute_write(_load_file_tx, data)


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