"""
scripts/evaluate.py
Đánh giá Agent vs BM25 baseline trên test set.

Chạy:
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

TEST_SET_FILE = Path("data/test_set.json")
RAW_DIR       = Path("data/raw")
API_URL       = "http://localhost:8000"


# ── Load data ─────────────────────────────────────────────────

def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))


def load_all_articles() -> list[dict]:
    """Load tất cả raw articles làm corpus cho BM25."""
    articles = []
    for f in RAW_DIR.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            articles.append({
                "article_id": data["metadata"]["article_id"],
                "title":      data["metadata"]["title"],
                "text":       data["plain_text"],
                "url":        data["metadata"]["url"],
            })
        except Exception:
            pass
    return articles


# ── BM25 Baseline ─────────────────────────────────────────────

def build_bm25(articles: list[dict]) -> tuple[BM25Okapi, list[dict]]:
    """Build BM25 index từ tất cả articles."""
    corpus = [
        (a["title"] + " " + a["text"]).lower().split()
        for a in articles
    ]
    bm25 = BM25Okapi(corpus)
    return bm25, articles


def bm25_search(
    bm25: BM25Okapi,
    articles: list[dict],
    query: str,
    top_k: int = 5,
) -> list[str]:
    """Tìm top-k articles bằng BM25."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [articles[i]["article_id"] for i in top_ids]


# ── Agent search ──────────────────────────────────────────────

def agent_search(question: str, session_id: str = "eval") -> list[str]:
    """Gọi Agent API, lấy sources."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract article_id từ URLs
        sources = data.get("sources", [])
        article_ids = []
        for url in sources:
            # URL cuối là article slug = article_id
            slug = url.rstrip("/").split("/")[-1]
            article_ids.append(slug)
        return article_ids
    except Exception as e:
        print(f"  API error: {e}")
        return []


# ── Metrics ───────────────────────────────────────────────────

def hit_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    """Hit@K: 1 nếu relevant trong top-k, 0 nếu không."""
    return 1.0 if relevant in retrieved[:k] else 0.0


def reciprocal_rank(retrieved: list[str], relevant: str) -> float:
    """Reciprocal Rank cho MRR."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id == relevant:
            return 1.0 / i
    return 0.0


# ── Evaluation pipeline ───────────────────────────────────────

def evaluate():
    print("Loading test set...")
    test_set = load_test_set()
    print(f"Test set: {len(test_set)} questions")

    print("Loading articles for BM25...")
    articles = load_all_articles()
    print(f"Corpus: {len(articles)} articles")

    print("Building BM25 index...")
    bm25, articles = build_bm25(articles)

    # Metrics
    bm25_hit1  = bm25_hit5  = bm25_mrr  = 0.0
    agent_hit1 = agent_hit5 = agent_mrr = 0.0
    total = len(test_set)

    print(f"\nEvaluating {total} questions...\n")

    for i, qa in enumerate(test_set, 1):
        question   = qa["question"]
        article_id = qa["article_id"]

        print(f"[{i}/{total}] {question[:60]}...")

        # BM25
        bm25_results = bm25_search(bm25, articles, question, top_k=5)
        bm25_hit1  += hit_at_k(bm25_results, article_id, 1)
        bm25_hit5  += hit_at_k(bm25_results, article_id, 5)
        bm25_mrr   += reciprocal_rank(bm25_results, article_id)

        # Agent
        agent_results = agent_search(question, session_id=f"eval_{i}")
        agent_hit1  += hit_at_k(agent_results, article_id, 1)
        agent_hit5  += hit_at_k(agent_results, article_id, 5)
        agent_mrr   += reciprocal_rank(agent_results, article_id)

        print(f"  BM25: {bm25_results[:3]} | Agent: {agent_results[:3]}")
        time.sleep(2)

    # Tổng kết
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<15} {'BM25':>10} {'Agent':>10} {'Improvement':>12}")
    print("-"*50)

    metrics = [
        ("Hit@1",  bm25_hit1/total,  agent_hit1/total),
        ("Hit@5",  bm25_hit5/total,  agent_hit5/total),
        ("MRR",    bm25_mrr/total,   agent_mrr/total),
    ]

    for name, bm25_score, agent_score in metrics:
        improvement = ((agent_score - bm25_score) / max(bm25_score, 0.001)) * 100
        print(f"{name:<15} {bm25_score:>10.3f} {agent_score:>10.3f} {improvement:>+11.1f}%")

    print("="*60)

    # Lưu kết quả
    results = {
        "total_questions": total,
        "bm25": {
            "hit@1": round(bm25_hit1/total, 3),
            "hit@5": round(bm25_hit5/total, 3),
            "mrr":   round(bm25_mrr/total, 3),
        },
        "agent": {
            "hit@1": round(agent_hit1/total, 3),
            "hit@5": round(agent_hit5/total, 3),
            "mrr":   round(agent_mrr/total, 3),
        },
    }
    Path("data/eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to data/eval_results.json")


if __name__ == "__main__":
    evaluate()