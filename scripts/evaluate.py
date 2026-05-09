"""
scripts/evaluate.py
Đánh giá Agent vs BM25 baseline trên test set.

Metrics:
  - Hit@1, Hit@5, MRR  — retrieval (có tìm đúng article không)
  - ROUGE-L             — answer quality (so sánh với reference answer)
  - LLM-as-Judge        — answer correctness via Groq (llama-3.3-70b-versatile)
  - BERT-Score F1       — semantic similarity (roberta-large)
  - Tool Accuracy       — tỷ lệ agent chọn đúng tool (requires expected_tool in test_set)
  - Latency p50/p95     — performance monitoring
  - Avg/Max Steps       — agent behavior analysis

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


# Lazy singleton Groq client to avoid recreating connections per evaluation
_groq_client: object = None


def _get_groq_client():
    """Get or create cached Groq client."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
    return _groq_client


def llm_judge(question: str, reference: str, agent_answer: str) -> int | None:
    """Score agent answer 1-5 using Groq LLM. Returns None on empty input or API failure."""
    if not agent_answer or not reference:
        return None
    try:
        client = _get_groq_client()
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
        content = resp.choices[0].message.content.strip()
        m = re.search(r"[1-5]", content)
        return int(m.group()) if m else None
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


def compute_tool_accuracy(results: list[dict], test_set: list[dict]) -> dict:
    """Compare tool_used vs expected_tool. Skips entries without expected_tool field."""
    correct   = 0
    total     = 0
    confusion: dict[tuple[str, str], int] = {}

    for qa, result in zip(test_set, results):
        expected = qa.get("expected_tool")
        if not expected:
            continue
        actual = result.get("tool_used", "")
        total += 1
        if actual == expected:
            correct += 1
        key = (expected, actual)
        confusion[key] = confusion.get(key, 0) + 1

    return {
        "accuracy":  correct / total if total else 0.0,
        "correct":   correct,
        "total":     total,
        "confusion": {f"{e}->{a}": c for (e, a), c in sorted(confusion.items())},
    }


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

    if not os.getenv("GROQ_API_KEY_1"):
        print("WARNING: GROQ_API_KEY_1 not set — LLM-Judge will be skipped (all scores = N/A)")

    # Unique run ID để tránh session contamination khi chạy nhiều lần
    run_id = uuid.uuid4().hex[:8]
    print(f"\nRun ID: {run_id}")

    # Accumulators
    bm25_h1 = bm25_h5 = bm25_mrr = 0.0
    ag_h1 = ag_h5 = ag_mrr = ag_rl = 0.0
    latencies:       list[float] = []
    step_counts:     list[int]   = []
    judge_scores:    list[int]   = []
    all_predictions: list[str]   = []
    all_references:  list[str]   = []
    all_results:     list[dict]  = []

    tool_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "h5": 0.0, "mrr": 0.0, "rl": 0.0, "count": 0,
                 "judge_sum": 0.0, "judge_n": 0}
    )
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "mrr": 0.0, "rl": 0.0, "count": 0,
                 "judge_sum": 0.0, "judge_n": 0}
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
        result     = agent_search(question, url_to_id, session_id)
        time.sleep(3)  # rate limit buffer between Groq API bursts

        ag_results  = result["article_ids"]
        tool_used   = result["tool_used"]
        ag_answer   = result["answer"]

        latencies.append(result["latency"])
        step_counts.append(result["steps_count"])
        all_predictions.append(ag_answer)
        all_references.append(ref_answer)
        all_results.append(result)

        h1 = hit_at_k(ag_results, article_id, 1)
        h5 = hit_at_k(ag_results, article_id, 5)
        rr = reciprocal_rank(ag_results, article_id)
        rl = rouge_l(ag_answer, ref_answer)

        ag_h1  += h1
        ag_h5  += h5
        ag_mrr += rr
        ag_rl  += rl

        # LLM-as-Judge
        judge_score = llm_judge(question, ref_answer, ag_answer)
        if judge_score is not None:
            judge_scores.append(judge_score)
            tool_stats[tool_used]["judge_sum"] += judge_score
            tool_stats[tool_used]["judge_n"]   += 1
            cat_stats[category]["judge_sum"]   += judge_score
            cat_stats[category]["judge_n"]     += 1

        tool_stats[tool_used]["h1"]    += h1
        tool_stats[tool_used]["h5"]    += h5
        tool_stats[tool_used]["mrr"]   += rr
        tool_stats[tool_used]["rl"]    += rl
        tool_stats[tool_used]["count"] += 1

        cat_stats[category]["h1"]    += h1
        cat_stats[category]["mrr"]   += rr
        cat_stats[category]["rl"]    += rl
        cat_stats[category]["count"] += 1

        judge_str = f"Judge={judge_score}" if judge_score is not None else "Judge=N/A"
        print(
            f"  [{tool_used}] Hit@1={h1:.0f}  MRR={rr:.2f}  "
            f"ROUGE-L={rl:.2f}  {judge_str}  {result['latency']:.1f}s"
        )

    # ── Post-loop metrics ────────────────────────────────────────
    bert_f1  = compute_bert_score(all_predictions, all_references)
    tool_acc = compute_tool_accuracy(all_results, test_set)

    latencies_s = sorted(latencies)
    n_lat   = len(latencies_s)
    lat_p50 = latencies_s[n_lat // 2]          if n_lat else 0.0
    lat_p95 = latencies_s[int(n_lat * 0.95)]   if n_lat else 0.0
    lat_avg = sum(latencies_s) / n_lat          if n_lat else 0.0

    avg_steps  = sum(step_counts) / len(step_counts) if step_counts else 0.0
    max_steps  = max(step_counts)                    if step_counts else 0
    judge_avg  = sum(judge_scores) / len(judge_scores) if judge_scores else None
    judge_n    = len(judge_scores)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    if bm25_available:
        print(f"\n{'Metric':<12} {'BM25':>8} {'Agent':>8} {'Delta':>8}")
        print("-" * 44)
        for name, b, a in [
            ("Hit@1",  bm25_h1/n,  ag_h1/n),
            ("Hit@5",  bm25_h5/n,  ag_h5/n),
            ("MRR",    bm25_mrr/n, ag_mrr/n),
        ]:
            d    = a - b
            pct  = (d / b * 100) if b > 0 else float("nan")
            pcts = f"{pct:+.1f}%" if b > 0 else "N/A"
            print(f"{name:<12} {b:>8.3f} {a:>8.3f} {d:>+8.3f}  ({pcts})")
    else:
        print(f"\n{'Metric':<12} {'Agent':>8}")
        print("-" * 24)
        for name, val in [("Hit@1", ag_h1/n), ("Hit@5", ag_h5/n), ("MRR", ag_mrr/n)]:
            print(f"{name:<12} {val:>8.3f}")

    print(f"\n--- Answer Quality ---")
    print(f"  {'ROUGE-L':<18} {ag_rl/n:.3f}")
    bert_str  = f"{bert_f1:.3f}" if bert_f1 is not None else "N/A (not installed)"
    judge_str = f"{judge_avg:.2f}/5.0  (n={judge_n}/{n} valid)" if judge_avg is not None else "N/A"
    print(f"  {'BERT-Score F1':<18} {bert_str}")
    print(f"  {'LLM-Judge avg':<18} {judge_str}")

    print(f"\n--- Agent Quality ---")
    if tool_acc["total"] > 0:
        print(f"  Tool Accuracy:  {tool_acc['correct']}/{tool_acc['total']} = {tool_acc['accuracy']:.1%}")
    else:
        print(f"  Tool Accuracy:  N/A (no expected_tool annotations)")
    print(f"  Avg Steps:      {avg_steps:.1f}  (max: {max_steps})")

    print(f"\n--- Performance ---")
    print(f"  Latency p50:    {lat_p50:.2f}s")
    print(f"  Latency p95:    {lat_p95:.2f}s")
    print(f"  Latency avg:    {lat_avg:.2f}s")

    # ── Tool Accuracy confusion ───────────────────────────────
    if tool_acc["confusion"]:
        print("\n--- Tool Accuracy Confusion ---")
        for pair, count in sorted(tool_acc["confusion"].items()):
            mark = "✓" if pair.split("->")[0] == pair.split("->")[1] else "✗"
            print(f"  {pair:<28} {count:>3}  {mark}")

    # ── Per-tool ──────────────────────────────────────────────
    if tool_stats:
        print("\n--- Per-Tool Breakdown ---")
        print(f"{'Tool':<12} {'N':>4} {'Hit@1':>7} {'Hit@5':>7} {'MRR':>7} {'ROUGE-L':>9} {'Judge':>7}")
        print("-" * 60)
        for tool, s in sorted(tool_stats.items()):
            c = s["count"]
            j = f"{s['judge_sum']/s['judge_n']:.2f}" if s["judge_n"] else "  N/A"
            print(
                f"{tool:<12} {c:>4} "
                f"{s['h1']/c:>7.3f} {s['h5']/c:>7.3f} "
                f"{s['mrr']/c:>7.3f} {s['rl']/c:>9.3f} {j:>7}"
            )

    # ── Per-category ──────────────────────────────────────────
    if len(cat_stats) > 1:
        print("\n--- Per-Category Breakdown ---")
        print(f"{'Category':<14} {'N':>4} {'Hit@1':>7} {'MRR':>7} {'ROUGE-L':>9} {'Judge':>7}")
        print("-" * 54)
        for cat, s in sorted(cat_stats.items()):
            c = s["count"]
            j = f"{s['judge_sum']/s['judge_n']:.2f}" if s["judge_n"] else "  N/A"
            print(
                f"{cat:<14} {c:>4} "
                f"{s['h1']/c:>7.3f} {s['mrr']/c:>7.3f} "
                f"{s['rl']/c:>9.3f} {j:>7}"
            )

    print("=" * 65)

    # ── Save ──────────────────────────────────────────────────
    results: dict = {
        "run_id":          run_id,
        "total_questions": n,
        "agent": {
            "hit@1":             round(ag_h1/n,  3),
            "hit@5":             round(ag_h5/n,  3),
            "mrr":               round(ag_mrr/n, 3),
            "rouge_l":           round(ag_rl/n,  3),
            "bert_score_f1":     round(bert_f1, 3) if bert_f1 is not None else None,
            "llm_judge_avg":     round(judge_avg, 3) if judge_avg is not None else None,
            "llm_judge_n_valid": judge_n,
            "tool_accuracy":     round(tool_acc["accuracy"], 3),
            "tool_accuracy_n":   tool_acc["total"],
            "latency_p50":       round(lat_p50, 3),
            "latency_p95":       round(lat_p95, 3),
            "latency_avg":       round(lat_avg, 3),
            "avg_steps":         round(avg_steps, 2),
            "max_steps":         max_steps,
        },
        "tool_accuracy_confusion": tool_acc["confusion"],
        "per_tool": {
            tool: {
                "count":     s["count"],
                "hit@1":     round(s["h1"]  / s["count"], 3),
                "hit@5":     round(s["h5"]  / s["count"], 3),
                "mrr":       round(s["mrr"] / s["count"], 3),
                "rouge_l":   round(s["rl"]  / s["count"], 3),
                "llm_judge": round(s["judge_sum"] / s["judge_n"], 3) if s["judge_n"] else None,
            }
            for tool, s in tool_stats.items()
        },
        "per_category": {
            cat: {
                "count":     s["count"],
                "hit@1":     round(s["h1"]  / s["count"], 3),
                "mrr":       round(s["mrr"] / s["count"], 3),
                "rouge_l":   round(s["rl"]  / s["count"], 3),
                "llm_judge": round(s["judge_sum"] / s["judge_n"], 3) if s["judge_n"] else None,
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
