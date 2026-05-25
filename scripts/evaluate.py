"""
scripts/evaluate.py
Đánh giá Agent vs BM25 baseline trên test set.

Metrics:
  - Hit@1, Hit@5, MRR     — retrieval (chỉ tính cho câu có matched_article_id)
  - ROUGE-L               — answer quality (so sánh với reference answer)
  - Keyword Accuracy      — % keywords quan trọng xuất hiện trong answer
  - LLM-as-Judge          — answer correctness via Groq (llama-3.3-70b-versatile)
  - BERT-Score F1         — semantic similarity (roberta-large)
  - Tool Accuracy         — tỷ lệ agent chọn đúng tool
  - Latency p50/p95       — performance monitoring
  - Avg/Max Steps         — agent behavior analysis

Hỗ trợ 2 loại test set:
  - test_set.json cũ: article_id map về KG → tính đủ tất cả metrics
  - qa_testset.json mới (từ Microsoft Q&A): article_id = "qa_{id}"
    → bỏ qua Hit@K/MRR nếu matched_article_id rỗng
    → vẫn tính ROUGE-L, Keyword Accuracy, LLM-Judge đầy đủ

Chạy:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-set data/qa_testset.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from rouge_score import rouge_scorer

load_dotenv()

JUDGE_MODEL   = "llama-3.3-70b-versatile"

TEST_SET_FILE = Path("data/test_set.json")
RAW_DIR       = Path("data/raw")
API_URL       = os.environ.get("API_URL", "http://localhost:8000")

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


# ── Load data ─────────────────────────────────────────────────

def load_test_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def bm25_answer(bm25, articles: list[dict], query: str, top_k: int = 3) -> str:
    """Generate a BM25 baseline 'answer' by concatenating top-k article excerpts."""
    tokens  = _tokenize(query)
    scores  = bm25.get_scores(tokens)
    top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    parts   = []
    for idx in top_ids:
        art     = articles[idx]
        snippet = art["text"][:200].replace("\n", " ")
        parts.append(f"{art['title']}: {snippet}")
    return "\n".join(parts) if parts else ""


# ── Agent ─────────────────────────────────────────────────────

def agent_search(
    question: str,
    url_to_id: dict,
    session_id: str,
) -> dict:
    """Query the agent API."""
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


def keyword_accuracy(answer: str, keywords: list[str]) -> float:
    """
    Tỷ lệ keywords quan trọng xuất hiện trong answer.
    Phù hợp cho IT helpdesk vì answer phải chứa đúng error codes,
    product names, technical terms.
    """
    if not keywords or not answer:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def get_article_id(qa: dict) -> tuple[str, bool]:
    """
    Lấy article_id để dùng cho retrieval evaluation.
    Returns: (article_id, can_eval_retrieval)
    - can_eval_retrieval = False nếu câu từ Q&A chưa được map về KG
    """
    matched = qa.get("matched_article_id", "")
    if matched:
        return matched, True
    article_id = qa["article_id"]
    # article_id = "qa_{num}" → chưa map về KG → không tính retrieval
    can_eval = not article_id.startswith("qa_")
    return article_id, can_eval


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

_groq_client: object = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY_2") or os.getenv("GROQ_API_KEY_1")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


FAITHFULNESS_PROMPT = """\
Given this IT helpdesk context and agent answer, count how many factual claims in the answer are directly supported by the context.

Context: {context}

Answer: {answer}

Count the total number of factual claims in the answer, then count how many are supported by the context.
Return ONLY valid JSON, no markdown:
{{"supported": 3, "total": 4}}"""


def faithfulness_score(answer: str, context: str) -> float | None:
    """RAGAS-style faithfulness: fraction of answer claims supported by context."""
    if not answer or not context:
        return None
    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(
                context=context[:1200],
                answer=answer[:600],
            )}],
            temperature=0,
            max_tokens=40,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed    = json.loads(raw)
        supported = int(parsed.get("supported", 0))
        total     = int(parsed.get("total", 0))
        return supported / total if total > 0 else None
    except Exception:
        return None


ANSWER_RELEVANCY_PROMPT = """\
Given this IT helpdesk answer, generate exactly 3 questions that this answer would be a good response to.
Return ONLY a JSON array of 3 strings.

Answer: {answer}

Questions (JSON array of 3):"""

_st_model: object = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def answer_relevancy_score(question: str, answer: str) -> float | None:
    """RAGAS-style answer relevancy: cosine sim of reverse-generated questions to original."""
    if not answer:
        return None
    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": ANSWER_RELEVANCY_PROMPT.format(
                answer=answer[:600]
            )}],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        gen_questions = json.loads(raw)
        if not isinstance(gen_questions, list) or not gen_questions:
            return None
    except Exception:
        return None

    try:
        st       = _get_st_model()
        orig_vec = st.encode(question, normalize_embeddings=True)
        gen_vecs = st.encode(gen_questions, normalize_embeddings=True)
        scores   = gen_vecs @ orig_vec
        return float(np.mean(scores))
    except Exception:
        return None


PAIRWISE_PROMPT = """\
You are an expert IT support evaluator comparing two answers to an IT helpdesk question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer better addresses the IT question? Consider: accuracy, completeness, actionability, specificity.
Reply ONLY with exactly one of: A, B, or Tie"""


def pairwise_judge(
    question: str, answer_a: str, answer_b: str
) -> str | None:
    """Returns 'A', 'B', or 'Tie'. None if either answer is missing.

    Randomizes answer order to counteract LLM position bias.
    """
    if not answer_a or not answer_b:
        return None
    import random
    flipped = random.random() < 0.5
    a_sent  = answer_b if flipped else answer_a
    b_sent  = answer_a if flipped else answer_b
    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": PAIRWISE_PROMPT.format(
                question=question,
                answer_a=a_sent[:500],
                answer_b=b_sent[:500],
            )}],
            temperature=0,
            max_tokens=10,
        )
        verdict = resp.choices[0].message.content.strip()
        if "Tie" in verdict or "tie" in verdict:
            return "Tie"
        if verdict.startswith("A"):
            return "B" if flipped else "A"
        if verdict.startswith("B"):
            return "A" if flipped else "B"
        return None
    except Exception:
        return None


def generate_annotation_template(
    test_set: list[dict],
    out_path: "Path",
    n: int = 30,
    api_url: str = "http://localhost:8000",
) -> None:
    """Generate human annotation template JSON with real agent answers pre-filled.

    Calls the agent API for each question. Leaves human_score (null) and
    human_notes ('') for the user to fill in manually.
    Requires the FastAPI backend to be running.
    """
    sample = test_set[:n] if len(test_set) >= n else test_set
    run_id = uuid.uuid4().hex[:6]

    template = []
    for i, qa in enumerate(sample, 1):
        question = qa["question"]
        print(f"  [{i:02d}/{len(sample)}] {question[:60]}...")
        agent_answer = ""
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={"question": question, "session_id": f"annotation_{run_id}_{i}"},
                timeout=90,
            )
            resp.raise_for_status()
            raw = resp.json().get("answer", "")
            agent_answer = raw if isinstance(raw, str) else ""
        except Exception as e:
            print(f"    WARNING: API call failed — {e}. agent_answer will be empty.")

        template.append({
            "question":         question,
            "category":         qa.get("category", ""),
            "expected_tool":    qa.get("expected_tool", ""),
            "reference_answer": qa.get("ground_truth_answer") or qa.get("answer", ""),
            "agent_answer":     agent_answer,
            "human_score":      None,   # USER FILLS: 1-5 (1=wrong, 5=perfect)
            "human_notes":      "",     # USER FILLS: optional comments
        })
        time.sleep(2)   # 2s buffer between API calls to avoid rate limits

    out_path = Path(out_path)
    out_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    filled = sum(1 for r in template if r["agent_answer"])
    print(f"Annotation template saved → {out_path}  ({len(template)} questions, {filled} with agent answers)")
    print("Fill in 'human_score' (1-5) and 'human_notes' for each question.")


def llm_judge(question: str, reference: str, agent_answer: str) -> int | None:
    """Score agent answer 1-5 using Groq."""
    if not agent_answer or not reference:
        return None
    try:
        client = _get_groq_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
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
    pairs = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not pairs:
        return None
    try:
        from bert_score import score as bert_score_fn
        preds, refs = zip(*pairs)
        _, _, F1 = bert_score_fn(
            list(preds), list(refs),
            lang="en", model_type="roberta-large", verbose=False,
        )
        return F1.mean().item()
    except ImportError:
        print("  WARNING: bert-score not installed — skipping BERT-Score")
        return None


def compute_tool_accuracy(results: list[dict], test_set: list[dict]) -> dict:
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

def evaluate(test_set_path: Path = TEST_SET_FILE):
    print("Loading test set...")
    test_set = load_test_set(test_set_path)
    n        = len(test_set)
    print(f"Test set: {n} questions  [{test_set_path}]")

    # Phân tích test set
    cat_dist: dict[str, int] = defaultdict(int)
    tool_dist: dict[str, int] = defaultdict(int)
    n_from_qa    = sum(1 for qa in test_set if qa.get("source") == "microsoft_qa")
    n_retrieval  = sum(1 for qa in test_set if not get_article_id(qa)[0].startswith("qa_"))

    for qa in test_set:
        cat_dist[qa.get("category", "?")] += 1
        tool_dist[qa.get("expected_tool", "?")] += 1

    print(f"Source: {n - n_from_qa} from Microsoft Learn, {n_from_qa} from Microsoft Q&A")
    print(f"Retrieval-evaluable: {n_retrieval}/{n} (rest skipped for Hit@K/MRR)")
    print("Category: " + ", ".join(f"{c}={v}" for c, v in sorted(cat_dist.items())))
    print("Expected tool: " + ", ".join(f"{t}={v}" for t, v in sorted(tool_dist.items())))

    # BM25 setup
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
        print("WARNING: GROQ_API_KEY_1 not set — LLM-Judge will be skipped")

    run_id = uuid.uuid4().hex[:8]
    print(f"\nRun ID: {run_id}")

    # Accumulators
    bm25_h1 = bm25_h5 = bm25_mrr = 0.0
    ag_h1 = ag_h5 = ag_mrr = ag_rl = ag_kw = 0.0
    n_retrieval_done  = 0   # số câu thực sự tính retrieval
    latencies:       list[float] = []
    step_counts:     list[int]   = []
    judge_scores:    list[int]   = []
    all_predictions: list[str]   = []
    all_references:  list[str]   = []
    all_results:     list[dict]  = []
    faith_scores:   list[float] = []
    ar_scores:      list[float] = []
    pairwise_wins   = 0
    pairwise_ties   = 0
    pairwise_total  = 0

    tool_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "h5": 0.0, "mrr": 0.0, "rl": 0.0, "kw": 0.0,
                 "count": 0, "retrieval_count": 0,
                 "judge_sum": 0.0, "judge_n": 0}
    )
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"h1": 0.0, "mrr": 0.0, "rl": 0.0, "kw": 0.0,
                 "count": 0, "retrieval_count": 0,
                 "judge_sum": 0.0, "judge_n": 0}
    )

    print(f"\nEvaluating {n} questions...\n")

    for i, qa in enumerate(test_set, 1):
        question                  = qa["question"]
        article_id, can_eval_ret  = get_article_id(qa)
        # Dùng ground_truth_answer nếu có (đã fill thủ công), fallback về answer
        ref_answer  = qa.get("ground_truth_answer") or qa.get("answer", "")
        category    = qa.get("category", "?")
        source      = qa.get("source", "learn")
        kw_list     = qa.get("answer_keywords", [])
        src_label   = "QA" if source == "microsoft_qa" else "MS"

        print(f"[{i:02d}/{n}] [{src_label}] {question[:60]}...")

        # BM25 (chỉ tính nếu có thể eval retrieval)
        if bm25_available and can_eval_ret:
            bm25_results = bm25_search(bm25, bm25_articles, question, top_k=5)
            bm25_h1  += hit_at_k(bm25_results, article_id, 1)
            bm25_h5  += hit_at_k(bm25_results, article_id, 5)
            bm25_mrr += reciprocal_rank(bm25_results, article_id)

        # Agent
        session_id = f"eval_{run_id}_{i}"
        q_start    = time.time()
        result     = agent_search(question, url_to_id, session_id)
        elapsed    = time.time() - q_start
        time.sleep(max(0, 15 - elapsed))

        ag_results  = result["article_ids"]
        tool_used   = result["tool_used"]
        ag_answer   = result["answer"]

        latencies.append(result["latency"])
        step_counts.append(result["steps_count"])
        all_predictions.append(ag_answer)
        all_references.append(ref_answer)
        all_results.append(result)

        # Retrieval metrics — chỉ tính khi có article_id hợp lệ
        if can_eval_ret:
            h1 = hit_at_k(ag_results, article_id, 1)
            h5 = hit_at_k(ag_results, article_id, 5)
            rr = reciprocal_rank(ag_results, article_id)
            ag_h1  += h1
            ag_h5  += h5
            ag_mrr += rr
            n_retrieval_done += 1
            tool_stats[tool_used]["h1"]              += h1
            tool_stats[tool_used]["h5"]              += h5
            tool_stats[tool_used]["mrr"]             += rr
            tool_stats[tool_used]["retrieval_count"] += 1
            cat_stats[category]["h1"]                += h1
            cat_stats[category]["mrr"]               += rr
            cat_stats[category]["retrieval_count"]   += 1
        else:
            h1 = h5 = rr = None

        # Answer quality metrics — tính cho tất cả
        rl       = rouge_l(ag_answer, ref_answer)
        kw_score = keyword_accuracy(ag_answer, kw_list)
        ag_rl   += rl
        ag_kw   += kw_score

        tool_stats[tool_used]["rl"]    += rl
        tool_stats[tool_used]["kw"]    += kw_score
        tool_stats[tool_used]["count"] += 1
        cat_stats[category]["rl"]      += rl
        cat_stats[category]["kw"]      += kw_score
        cat_stats[category]["count"]   += 1

        # LLM-as-Judge
        judge_score = llm_judge(question, ref_answer, ag_answer)
        if judge_score is not None:
            judge_scores.append(judge_score)
            tool_stats[tool_used]["judge_sum"] += judge_score
            tool_stats[tool_used]["judge_n"]   += 1
            cat_stats[category]["judge_sum"]   += judge_score
            cat_stats[category]["judge_n"]     += 1

        # Faithfulness
        q_context = result.get("context", "")
        faith = faithfulness_score(ag_answer, q_context)
        if faith is not None:
            faith_scores.append(faith)

        # Answer relevancy
        ar = answer_relevancy_score(question, ag_answer)
        if ar is not None:
            ar_scores.append(ar)

        # Pairwise: agent vs BM25
        if bm25_available and ag_answer:
            bm25_ans = bm25_answer(bm25, bm25_articles, question, top_k=3)
            if bm25_ans:
                verdict = pairwise_judge(question, ag_answer, bm25_ans)
                if verdict == "A":
                    pairwise_wins  += 1
                    pairwise_total += 1
                elif verdict == "B":
                    pairwise_total += 1
                elif verdict == "Tie":
                    pairwise_ties  += 1
                    pairwise_total += 1

        h1_str = f"{h1:.0f}" if h1 is not None else "N/A"
        rr_str = f"{rr:.2f}" if rr is not None else "N/A"
        j_str  = str(judge_score) if judge_score is not None else "N/A"
        print(
            f"  [{tool_used}][{src_label}] "
            f"Hit@1={h1_str}  MRR={rr_str}  "
            f"ROUGE-L={rl:.2f}  KW={kw_score:.2f}  "
            f"Judge={j_str}  {result['latency']:.1f}s"
        )

    # ── Post-loop ─────────────────────────────────────────────
    bert_f1  = compute_bert_score(all_predictions, all_references)
    tool_acc = compute_tool_accuracy(all_results, test_set)

    latencies_s = sorted(latencies)
    n_lat    = len(latencies_s)
    lat_p50  = latencies_s[n_lat // 2]        if n_lat else 0.0
    lat_p95  = latencies_s[int(n_lat * 0.95)] if n_lat else 0.0
    lat_avg  = sum(latencies_s) / n_lat        if n_lat else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0.0
    max_steps = max(step_counts)                    if step_counts else 0
    judge_avg = sum(judge_scores) / len(judge_scores) if judge_scores else None
    judge_n   = len(judge_scores)

    denom_ret = max(n_retrieval_done, 1)  # denominator cho retrieval metrics

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total: {n} questions  |  Retrieval-evaluable: {n_retrieval_done}  |  Q&A source: {n_from_qa}")

    if bm25_available and n_retrieval_done > 0:
        print(f"\n{'Metric':<12} {'BM25':>8} {'Agent':>8} {'Delta':>8}")
        print("-" * 44)
        for name, b, a in [
            ("Hit@1",  bm25_h1/denom_ret,  ag_h1/denom_ret),
            ("Hit@5",  bm25_h5/denom_ret,  ag_h5/denom_ret),
            ("MRR",    bm25_mrr/denom_ret, ag_mrr/denom_ret),
        ]:
            d   = a - b
            pct = (d / b * 100) if b > 0 else float("nan")
            pcts = f"{pct:+.1f}%" if b > 0 else "N/A"
            print(f"{name:<12} {b:>8.3f} {a:>8.3f} {d:>+8.3f}  ({pcts})")
    elif n_retrieval_done > 0:
        print(f"\n{'Metric':<12} {'Agent':>8}")
        print("-" * 24)
        for name, val in [
            ("Hit@1", ag_h1/denom_ret),
            ("Hit@5", ag_h5/denom_ret),
            ("MRR",   ag_mrr/denom_ret),
        ]:
            print(f"{name:<12} {val:>8.3f}")
    else:
        print("\n⚠ No retrieval metrics (all questions from Q&A source without matched_article_id)")
        print("  → Fill 'matched_article_id' in test set to enable Hit@K/MRR")

    print(f"\n--- Answer Quality (all {n} questions) ---")
    print(f"  {'ROUGE-L':<20} {ag_rl/n:.3f}")
    print(f"  {'Keyword Accuracy':<20} {ag_kw/n:.3f}")
    bert_str  = f"{bert_f1:.3f}" if bert_f1 is not None else "N/A (not installed)"
    judge_str = f"{judge_avg:.2f}/5.0  (n={judge_n}/{n} valid)" if judge_avg is not None else "N/A"
    print(f"  {'BERT-Score F1':<20} {bert_str}")
    print(f"  {'LLM-Judge avg':<20} {judge_str}")
    faith_str = f"{sum(faith_scores)/len(faith_scores):.3f}  (n={len(faith_scores)})" if faith_scores else "N/A"
    ar_str    = f"{sum(ar_scores)/len(ar_scores):.3f}  (n={len(ar_scores)})" if ar_scores else "N/A"
    print(f"  {'Faithfulness':<20} {faith_str}")
    print(f"  {'Answer Relevancy':<20} {ar_str}")
    if pairwise_total > 0:
        pairwise_losses = pairwise_total - pairwise_wins - pairwise_ties
        win_rate = pairwise_wins / pairwise_total
        print(f"\n--- Pairwise: Agent vs BM25 (n={pairwise_total}) ---")
        print(f"  Agent wins:   {pairwise_wins}  ({win_rate:.1%})")
        print(f"  Ties:         {pairwise_ties}")
        print(f"  BM25 wins:    {pairwise_losses}")

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

    if tool_acc["confusion"]:
        print("\n--- Tool Accuracy Confusion ---")
        for pair, count in sorted(tool_acc["confusion"].items()):
            mark = "✓" if pair.split("->")[0] == pair.split("->")[1] else "✗"
            print(f"  {pair:<28} {count:>3}  {mark}")

    if tool_stats:
        print("\n--- Per-Tool Breakdown ---")
        print(f"{'Tool':<12} {'N':>4} {'Hit@1':>7} {'MRR':>7} {'ROUGE-L':>9} {'KW-Acc':>8} {'Judge':>7}")
        print("-" * 65)
        for tool, s in sorted(tool_stats.items()):
            c  = s["count"]
            rc = s["retrieval_count"]
            h1_s = f"{s['h1']/rc:.3f}" if rc else "  N/A"
            mr_s = f"{s['mrr']/rc:.3f}" if rc else "  N/A"
            j    = f"{s['judge_sum']/s['judge_n']:.2f}" if s["judge_n"] else "  N/A"
            print(
                f"{tool:<12} {c:>4} "
                f"{h1_s:>7} {mr_s:>7} "
                f"{s['rl']/c:>9.3f} {s['kw']/c:>8.3f} {j:>7}"
            )

    if len(cat_stats) > 1:
        print("\n--- Per-Category Breakdown ---")
        print(f"{'Category':<14} {'N':>4} {'Hit@1':>7} {'MRR':>7} {'ROUGE-L':>9} {'KW-Acc':>8} {'Judge':>7}")
        print("-" * 65)
        for cat, s in sorted(cat_stats.items()):
            c  = s["count"]
            rc = s["retrieval_count"]
            h1_s = f"{s['h1']/rc:.3f}" if rc else "  N/A"
            mr_s = f"{s['mrr']/rc:.3f}" if rc else "  N/A"
            j    = f"{s['judge_sum']/s['judge_n']:.2f}" if s["judge_n"] else "  N/A"
            print(
                f"{cat:<14} {c:>4} "
                f"{h1_s:>7} {mr_s:>7} "
                f"{s['rl']/c:>9.3f} {s['kw']/c:>8.3f} {j:>7}"
            )

    print("=" * 70)

    # ── Save results ──────────────────────────────────────────
    results: dict = {
        "run_id":              run_id,
        "test_set_path":       str(test_set_path),
        "total_questions":     n,
        "retrieval_evaluable": n_retrieval_done,
        "qa_source_count":     n_from_qa,
        "agent": {
            "hit@1":              round(ag_h1/denom_ret,  3) if n_retrieval_done else None,
            "hit@5":              round(ag_h5/denom_ret,  3) if n_retrieval_done else None,
            "mrr":                round(ag_mrr/denom_ret, 3) if n_retrieval_done else None,
            "rouge_l":            round(ag_rl/n,  3),
            "keyword_accuracy":   round(ag_kw/n,  3),
            "bert_score_f1":      round(bert_f1, 3) if bert_f1 is not None else None,
            "llm_judge_avg":      round(judge_avg, 3) if judge_avg is not None else None,
            "llm_judge_n_valid":  judge_n,
            "tool_accuracy":      round(tool_acc["accuracy"], 3),
            "tool_accuracy_n":    tool_acc["total"],
            "latency_p50":        round(lat_p50, 3),
            "latency_p95":        round(lat_p95, 3),
            "latency_avg":        round(lat_avg, 3),
            "avg_steps":          round(avg_steps, 2),
            "max_steps":          max_steps,
            "faithfulness":       round(sum(faith_scores)/len(faith_scores), 3) if faith_scores else None,
            "answer_relevancy":   round(sum(ar_scores)/len(ar_scores), 3) if ar_scores else None,
            "pairwise_win_rate":  round(pairwise_wins/pairwise_total, 3) if pairwise_total else None,
            "pairwise_wins":      pairwise_wins,
            "pairwise_ties":      pairwise_ties,
            "pairwise_total":     pairwise_total,
        },
        "bm25": {
            "hit@1": round(bm25_h1/denom_ret,  3),
            "hit@5": round(bm25_h5/denom_ret,  3),
            "mrr":   round(bm25_mrr/denom_ret, 3),
        } if bm25_available and n_retrieval_done else None,
        "tool_accuracy_confusion": tool_acc["confusion"],
        "per_tool": {
            tool: {
                "count":          s["count"],
                "retrieval_count": s["retrieval_count"],
                "hit@1":          round(s["h1"]/s["retrieval_count"],  3) if s["retrieval_count"] else None,
                "mrr":            round(s["mrr"]/s["retrieval_count"], 3) if s["retrieval_count"] else None,
                "rouge_l":        round(s["rl"]/s["count"],  3),
                "keyword_accuracy": round(s["kw"]/s["count"], 3),
                "llm_judge":      round(s["judge_sum"]/s["judge_n"], 3) if s["judge_n"] else None,
            }
            for tool, s in tool_stats.items()
        },
        "per_category": {
            cat: {
                "count":          s["count"],
                "hit@1":          round(s["h1"]/s["retrieval_count"],  3) if s["retrieval_count"] else None,
                "mrr":            round(s["mrr"]/s["retrieval_count"], 3) if s["retrieval_count"] else None,
                "rouge_l":        round(s["rl"]/s["count"],  3),
                "keyword_accuracy": round(s["kw"]/s["count"], 3),
                "llm_judge":      round(s["judge_sum"]/s["judge_n"], 3) if s["judge_n"] else None,
            }
            for cat, s in cat_stats.items()
        },
    }

    out = Path("data/eval_results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out}")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KBQA agent")
    parser.add_argument(
        "--test-set", type=Path, default=TEST_SET_FILE,
        help=f"Path to test set JSON (default: {TEST_SET_FILE})"
    )
    parser.add_argument(
        "--generate-annotation", action="store_true",
        help="Generate human annotation template JSON (requires backend running)"
    )
    parser.add_argument(
        "--annotation-n", type=int, default=30,
        help="Number of questions for annotation template (default: 30)"
    )
    args = parser.parse_args()

    if args.generate_annotation:
        test_set = load_test_set(args.test_set)
        out = Path("data/human_annotation_template.json")
        generate_annotation_template(test_set, out, n=args.annotation_n)
    else:
        evaluate(test_set_path=args.test_set)