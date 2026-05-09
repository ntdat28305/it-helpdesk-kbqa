"""
scripts/evaluate.py
Đánh giá Agent vs BM25 baseline trên test set.

Metrics:
  - Hit@1, Hit@5, MRR  — retrieval (có tìm đúng article không)
  - ROUGE-L             — answer quality (so sánh với reference answer)

Chạy:
    python scripts/evaluate.py
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv()

TEST_SET_FILE = Path("data/test_set.json")
RAW_DIR       = Path("data/raw")
API_URL       = os.environ.get("API_URL", "http://localhost:8000")

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


# ── Load data ─────────────────────────────────────────────────

def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))


def load_all_articles() -> list[dict]:
    """Load raw articles làm corpus BM25. Trả về [] nếu data/raw/ trống."""
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

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def build_bm25(articles: list[dict]):
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(a["title"] + " " + a["text"]) for a in articles]
    return BM25Okapi(corpus), articles


def bm25_search(bm25, articles: list[dict], query: str, top_k: int = 5) -> list[str]:
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [articles[i]["article_id"] for i in top_ids]


# ── Agent ─────────────────────────────────────────────────────

def agent_search(
    question: str,
    url_to_id: dict,
    session_id: str,
) -> dict:
    """Query the agent API. Returns dict: article_ids, tool_used, answer, latency, steps_count."""
    start = time.time()
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": session_id},
            timeout=90,
        )
        resp.raise_for_status()
        data        = resp.json()
        latency     = time.time() - start
        tool_used   = data.get("tool_used", "UNKNOWN")
        answer      = data.get("answer", "")
        sources     = data.get("sources", [])
        steps_count = len(data.get("steps", []) or [])
        article_ids = [
            url_to_id.get(url, url.rstrip("/").split("/")[-1])
            for url in sources
        ]
        return {
            "article_ids": article_ids,
            "tool_used":   tool_used,
            "answer":      answer,
            "latency":     latency,
            "steps_count": steps_count,
        }
    except Exception as e:
        print(f"  API error: {e}")
        return {
            "article_ids": [],
            "tool_used":   "ERROR",
            "answer":      "",
            "latency":     time.time() - start,
            "steps_count": 0,
        }


# ── Metrics ───────────────────────────────────────────────────

def hit_at_k(retrieved: list[str], relevant: str, k: int) -> float:
    return 1.0 if relevant in retrieved[:k] else 0.0


def reciprocal_rank(retrieved: list[str], relevant: str) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id == relevant:
            return 1.0 / i
    return 0.0


def rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    return _rouge.score(hypothesis, reference)["rougeL"].fmeasure


JUDGE_PROMPT = """\
You are an IT support expert evaluating answer quality.

Question: {question}
Reference Answer: {reference}
Agent Answer: {agent_answer}

Rate the agent's answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat correct but missing key information
4 = Mostly correct with minor issues
5 = Fully correct and comprehensive

Output ONLY the number (1-5):"""


def llm_judge(question: str, reference: str, agent_answer: str) -> int | None:
    """Score agent answer 1-5 using Groq LLM. Returns None on empty input or API failure."""
    if not agent_answer or not reference:
        return None
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question,
                reference=reference,
                agent_answer=agent_answer,
            )}],
            temperature=0,
            max_tokens=5,
        )
        score = int(resp.choices[0].message.content.strip()[0])
        return min(max(score, 1), 5)
    except Exception:
        return None


def compute_bert_score(predictions: list[str], references: list[str]) -> float | None:
    """Compute BERT-Score F1 (roberta-large) over non-empty prediction/reference pairs.
    Returns None if bert-score is not installed."""
    pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not pairs:
        return None
    try:
        from bert_score import score as bert_score_fn
        preds, refs = zip(*pairs)
        _, _, F1 = bert_score_fn(
            list(preds),
            list(refs),
            lang="en",
            model_type="roberta-large",
            verbose=False,
        )
        return F1.mean().item()
    except ImportError:
        print("  WARNING: bert-score not installed — skipping BERT-Score")
        return None


# ── Evaluation pipeline ───────────────────────────────────────

def evaluate():
    print("Loading test set...")
    test_set = load_test_set()
    n        = len(test_set)
    print(f"Test set: {n} questions")

    # Category distribution
    cat_dist: dict[str, int] = defaultdict(int)
    for qa in test_set:
        cat_dist[qa.get("category", "?")] += 1
    print("Category distribution: " + ", ".join(f"{c}={v}" for c, v in sorted(cat_dist.items())))

    # BM25 setup — optional, skip gracefully if no raw data
    bm25_available = False
    bm25 = bm25_articles = None
    print("\nLoading articles for BM25...")
    articles = load_all_articles()
    if articles:
        print(f"Corpus: {len(articles)} articles")
        bm25, bm25_articles = build_bm25(articles)
        url_to_id = {a["url"]: a["article_id"] for a in articles if a.get("url")}
        bm25_available = True
    else:
        print("WARNING: data/raw/ empty — skipping BM25 baseline")
        url_to_id = {}

    # Unique run ID để tránh session contamination khi chạy nhiều lần
    run_id = uuid.uuid4().hex[:8]
    print(f"\nRun ID: {run_id}")

    # Accumulators
    bm25_h1 = bm25_h5 = bm25_mrr = 0.0
    ag_h1 = ag_h5 = ag_mrr = ag_rl = 0.0
    tool_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "h5": 0.0, "mrr": 0.0, "rl": 0.0, "count": 0}
    )
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "mrr": 0.0, "rl": 0.0, "count": 0}
    )

    print(f"\nEvaluating {n} questions...\n")

    for i, qa in enumerate(test_set, 1):
        question   = qa["question"]
        article_id = qa["article_id"]
        ref_answer = qa.get("answer", "")
        category   = qa.get("category", "?")

        print(f"[{i:02d}/{n}] {question[:65]}...")

        # BM25
        if bm25_available:
            bm25_results = bm25_search(bm25, bm25_articles, question, top_k=5)
            bm25_h1  += hit_at_k(bm25_results, article_id, 1)
            bm25_h5  += hit_at_k(bm25_results, article_id, 5)
            bm25_mrr += reciprocal_rank(bm25_results, article_id)

        # Agent — session ID gắn run_id để không bị contaminate
        session_id = f"eval_{run_id}_{i}"
        ag_results, tool_used, ag_answer = agent_search(question, url_to_id, session_id)

        h1 = hit_at_k(ag_results, article_id, 1)
        h5 = hit_at_k(ag_results, article_id, 5)
        rr = reciprocal_rank(ag_results, article_id)
        rl = rouge_l(ag_answer, ref_answer)

        ag_h1  += h1
        ag_h5  += h5
        ag_mrr += rr
        ag_rl  += rl

        tool_stats[tool_used]["h1"]    += h1
        tool_stats[tool_used]["h5"]    += h5
        tool_stats[tool_used]["mrr"]   += rr
        tool_stats[tool_used]["rl"]    += rl
        tool_stats[tool_used]["count"] += 1

        cat_stats[category]["h1"]    += h1
        cat_stats[category]["mrr"]   += rr
        cat_stats[category]["rl"]    += rl
        cat_stats[category]["count"] += 1

        rl_str = f"ROUGE-L={rl:.2f}"
        print(f"  [{tool_used}] Hit@1={h1:.0f}  MRR={rr:.2f}  {rl_str}  | {ag_results[:2]}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    if bm25_available:
        print(f"\n{'Metric':<12} {'BM25':>8} {'Agent':>8} {'Delta':>8}  ({'%'})")
        print("-" * 50)
        metrics = [
            ("Hit@1",   bm25_h1/n,  ag_h1/n),
            ("Hit@5",   bm25_h5/n,  ag_h5/n),
            ("MRR",     bm25_mrr/n, ag_mrr/n),
        ]
        for name, b, a in metrics:
            d    = a - b
            pct  = (d / b * 100) if b > 0 else float("nan")
            pcts = f"{pct:+.1f}%" if b > 0 else "N/A"
            print(f"{name:<12} {b:>8.3f} {a:>8.3f} {d:>+8.3f}  ({pcts})")
    else:
        print(f"\n{'Metric':<12} {'Agent':>8}")
        print("-" * 24)
        for name, val in [("Hit@1", ag_h1/n), ("Hit@5", ag_h5/n), ("MRR", ag_mrr/n)]:
            print(f"{name:<12} {val:>8.3f}")

    print(f"\n{'ROUGE-L':<12} {'—':>8} {ag_rl/n:>8.3f}   (answer quality vs reference)")

    # ── Per-tool ──────────────────────────────────────────────
    if tool_stats:
        print("\n" + "-" * 65)
        print("PER-TOOL BREAKDOWN")
        print(f"{'Tool':<12} {'N':>4} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} {'ROUGE-L':>9}")
        print("-" * 50)
        for tool, s in sorted(tool_stats.items()):
            c = s["count"]
            print(
                f"{tool:<12} {c:>4} "
                f"{s['h1']/c:>7.3f} {s['h5']/c:>7.3f} "
                f"{s['mrr']/c:>7.3f} {s['rl']/c:>9.3f}"
            )

    # ── Per-category ──────────────────────────────────────────
    if len(cat_stats) > 1:
        print("\n" + "-" * 65)
        print("PER-CATEGORY BREAKDOWN")
        print(f"{'Category':<14} {'N':>4} {'Hit@1':>7} {'MRR':>7} {'ROUGE-L':>9}")
        print("-" * 46)
        for cat, s in sorted(cat_stats.items()):
            c = s["count"]
            print(
                f"{cat:<14} {c:>4} "
                f"{s['h1']/c:>7.3f} {s['mrr']/c:>7.3f} "
                f"{s['rl']/c:>9.3f}"
            )

    print("=" * 65)

    # ── Save ──────────────────────────────────────────────────
    results: dict = {
        "run_id":           run_id,
        "total_questions":  n,
        "agent": {
            "hit@1":   round(ag_h1/n,  3),
            "hit@5":   round(ag_h5/n,  3),
            "mrr":     round(ag_mrr/n, 3),
            "rouge_l": round(ag_rl/n,  3),
        },
        "per_tool": {
            tool: {
                "count":   s["count"],
                "hit@1":   round(s["h1"]  / s["count"], 3),
                "hit@5":   round(s["h5"]  / s["count"], 3),
                "mrr":     round(s["mrr"] / s["count"], 3),
                "rouge_l": round(s["rl"]  / s["count"], 3),
            }
            for tool, s in tool_stats.items()
        },
        "per_category": {
            cat: {
                "count":   s["count"],
                "hit@1":   round(s["h1"]  / s["count"], 3),
                "mrr":     round(s["mrr"] / s["count"], 3),
                "rouge_l": round(s["rl"]  / s["count"], 3),
            }
            for cat, s in cat_stats.items()
        },
    }
    if bm25_available:
        results["bm25"] = {
            "hit@1": round(bm25_h1/n,  3),
            "hit@5": round(bm25_h5/n,  3),
            "mrr":   round(bm25_mrr/n, 3),
        }

    out = Path("data/eval_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    evaluate()
