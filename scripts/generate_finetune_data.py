"""
scripts/generate_finetune_data.py
Tạo training pairs cho finetune bi-encoder retriever.

Nguồn:
  1. data/test_set.json        — 49 (question, article) labeled pairs
  2. Neo4j KG                  — (entity_name, article) pairs qua [:MENTIONS]
  3. BM25 hard negatives       — top BM25 results (trừ correct article)

Chạy:
    python scripts/generate_finetune_data.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TEST_SET_FILE = Path("data/test_set.json")
RAW_DIR       = Path("data/raw")
OUTPUT_FILE   = Path("data/finetune_pairs.jsonl")
MAX_TEXT_CHARS = 512  # cắt article text để tránh quá dài


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def load_articles() -> dict[str, dict]:
    """Trả về {article_id: {text, url}}."""
    articles: dict[str, dict] = {}
    for f in RAW_DIR.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            aid  = data["metadata"]["article_id"]
            articles[aid] = {
                "text": data["plain_text"][:MAX_TEXT_CHARS],
                "url":  data["metadata"]["url"],
            }
        except Exception:
            pass
    return articles


def build_bm25(articles: dict[str, dict]):
    from rank_bm25 import BM25Okapi
    ids    = list(articles.keys())
    corpus = [_tokenize(articles[a]["text"]) for a in ids]
    return BM25Okapi(corpus), ids


def mine_hard_negatives(
    query: str,
    correct_id: str,
    bm25,
    ids: list[str],
    top_n: int = 2,
) -> list[str]:
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:10] if ids[i] != correct_id][:top_n]


def get_kg_pairs() -> list[dict]:
    """Query Neo4j: trả về [{entity, url}]."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
    )
    pairs: list[dict] = []
    try:
        with driver.session() as session:
            results = session.run(
                "MATCH (a:Article)-[:MENTIONS]->(e:Entity) "
                "RETURN e.name AS entity, a.url AS url LIMIT 500"
            )
            pairs = [{"entity": r["entity"], "url": r["url"]} for r in results]
    finally:
        driver.close()
    return pairs


def main():
    print("=== generate_finetune_data.py ===")

    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    articles = load_articles()
    print(f"Articles loaded from data/raw/: {len(articles)}")

    pairs: list[dict] = []
    bm25 = ids = None

    if articles:
        bm25, ids = build_bm25(articles)

    # ── Nguồn 1: test_set.json ──────────────────────────────────
    for qa in test_set:
        aid = qa["article_id"]
        if aid not in articles:
            continue
        pair: dict = {
            "query":      qa["question"],
            "positive":   articles[aid]["text"],
            "article_id": aid,
        }
        if bm25 is not None:
            negs = mine_hard_negatives(qa["question"], aid, bm25, ids)
            if negs:
                pair["negative"] = articles[negs[0]]["text"]
        pairs.append(pair)

    print(f"Pairs from test_set: {len(pairs)}")

    # ── Nguồn 2: KG-based pairs ─────────────────────────────────
    if articles:
        url_to_id = {v["url"]: k for k, v in articles.items()}
        try:
            kg_pairs = get_kg_pairs()
            print(f"KG pairs from Neo4j: {len(kg_pairs)}")
            added = 0
            for kp in kg_pairs:
                aid = url_to_id.get(kp["url"])
                if not aid:
                    continue
                pair = {
                    "query":      kp["entity"],
                    "positive":   articles[aid]["text"],
                    "article_id": aid,
                }
                if bm25 is not None:
                    negs = mine_hard_negatives(kp["entity"], aid, bm25, ids, top_n=1)
                    if negs:
                        pair["negative"] = articles[negs[0]]["text"]
                pairs.append(pair)
                added += 1
            print(f"KG pairs added: {added}")
        except Exception as e:
            print(f"Neo4j unavailable ({e}) — skipping KG pairs")

    # ── Ghi output ───────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with_neg = sum(1 for p in pairs if "negative" in p)
    print(f"\nOutput: {len(pairs)} pairs -> {OUTPUT_FILE}")
    print(f"  voi hard negative: {with_neg}")
    print(f"  khong co negative: {len(pairs) - with_neg}")


if __name__ == "__main__":
    main()
