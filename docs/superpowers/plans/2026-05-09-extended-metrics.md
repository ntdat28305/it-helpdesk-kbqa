# Extended Evaluation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LLM-as-Judge, BERT-Score, Tool Accuracy, Latency, and Steps Count metrics to `scripts/evaluate.py`, extending the existing pipeline in-place.

**Architecture:** Extend `evaluate.py` directly — no new files or modules. `agent_search()` return type changes from a 3-tuple to a dict to carry latency and steps count. LLM-Judge runs per-question inside the loop; BERT-Score and Tool Accuracy run once after the loop in batch. `data/test_set.json` gets a new `expected_tool` field per entry.

**Tech Stack:** `groq` (already installed), `bert-score>=0.3.13` (new), `time` (stdlib), `data/test_set.json` annotation (manual via script).

> **Note on Tool Accuracy:** All 49 test questions are plain-language symptom descriptions — none contain explicit error codes, KB article numbers, or relationship queries in the question text. Therefore all 49 get `expected_tool = "EMBEDDING"` per the annotation priority rules. Tool Accuracy with this test set measures "how often does the agent choose EMBEDDING?" — a valid routing-quality signal that will be more discriminating when the test set is extended with questions that include error codes.

---

## Files

| File | Change |
|---|---|
| `requirements.txt` | Add `bert-score>=0.3.13` |
| `data/test_set.json` | Add `"expected_tool": "EMBEDDING"` to all 49 entries |
| `scripts/evaluate.py` | All code changes: new imports, new functions, refactored loop, updated summary |

---

## Task 1: Add bert-score to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, find the `# ── Evaluation ────` section (currently lines 43-45) and add `bert-score`:

```
# ── Evaluation ────────────────────────────────────────────────
rank-bm25>=0.2.2
rouge-score>=0.1.2
bert-score>=0.3.13
```

- [ ] **Step 2: Verify the line is present**

Run:
```bash
python -c "import importlib.util; print('ok' if importlib.util.find_spec('bert_score') else 'need install')"
```

If output is `need install`, run `pip install bert-score>=0.3.13`.

Expected after install: `ok`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add bert-score to requirements"
```

---

## Task 2: Annotate test_set.json with expected_tool

**Files:**
- Modify: `data/test_set.json`

- [ ] **Step 1: Run the annotation script**

From the repo root, run this one-off script (do not save it — paste directly into a terminal Python session or run as a temp file):

```python
import json
from pathlib import Path

path = Path("data/test_set.json")
data = json.loads(path.read_text(encoding="utf-8"))
for item in data:
    item["expected_tool"] = "EMBEDDING"
path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Annotated {len(data)} entries -> all EMBEDDING")
```

Expected output: `Annotated 49 entries -> all EMBEDDING`

- [ ] **Step 2: Verify structure**

```bash
python -c "
import json
data = json.load(open('data/test_set.json', encoding='utf-8'))
missing = [i for i, q in enumerate(data) if 'expected_tool' not in q]
print(f'Missing: {missing}' if missing else f'All {len(data)} entries have expected_tool')
"
```

Expected: `All 49 entries have expected_tool`

- [ ] **Step 3: Commit**

```bash
git add data/test_set.json
git commit -m "feat: annotate test_set with expected_tool (all EMBEDDING)"
```

---

## Task 3: Refactor agent_search() to return dict

**Files:**
- Modify: `scripts/evaluate.py:80-104`

- [ ] **Step 1: Add `import time` at the top of evaluate.py**

Find the existing imports block (around line 12-18) and add `import time`:

```python
from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
```

- [ ] **Step 2: Replace the agent_search() function**

Replace the entire `agent_search` function (lines 80-104) with:

```python
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
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/evaluate.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate.py
git commit -m "refactor: agent_search returns dict with latency and steps_count"
```

---

## Task 4: Add llm_judge() function

**Files:**
- Modify: `scripts/evaluate.py` — add after the `rouge_l` function (after line ~123)

- [ ] **Step 1: Add JUDGE_PROMPT constant and llm_judge() function**

Insert the following block immediately after the `rouge_l` function definition (before the `# ── Evaluation pipeline` comment):

```python
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
```

- [ ] **Step 2: Smoke test**

With the FastAPI server running on port 8000, run:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from scripts.evaluate import llm_judge
score = llm_judge(
    'How do I fix Teams not loading?',
    'Add Teams URLs to trusted sites in browser settings.',
    'Try clearing the Teams cache and restarting the app.',
)
print(f'Score: {score}')
assert score is None or 1 <= score <= 5, 'score out of range'
print('ok')
"
```

Expected: `Score: <1-5>` then `ok` (exact score may vary).

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add llm_judge function for answer quality scoring"
```

---

## Task 5: Add compute_bert_score() function

**Files:**
- Modify: `scripts/evaluate.py` — add after `llm_judge()`

- [ ] **Step 1: Add the function**

Insert immediately after the `llm_judge` function:

```python
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
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from scripts.evaluate import compute_bert_score
f1 = compute_bert_score(['restart your PC'], ['reboot the computer'])
print(f'BERT-Score F1: {f1:.3f}')
assert f1 is None or 0.0 <= f1 <= 1.0
print('ok')
"
```

Expected: `BERT-Score F1: 0.8xx` (should be high — semantic similarity) then `ok`.

> First run downloads roberta-large (~500MB). Subsequent runs use HuggingFace cache.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add compute_bert_score function"
```

---

## Task 6: Add compute_tool_accuracy() function

**Files:**
- Modify: `scripts/evaluate.py` — add after `compute_bert_score()`

- [ ] **Step 1: Add the function**

Insert immediately after the `compute_bert_score` function:

```python
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
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from scripts.evaluate import compute_tool_accuracy
results  = [{'tool_used': 'EMBEDDING'}, {'tool_used': 'CYPHER'}, {'tool_used': 'EMBEDDING'}]
test_set = [
    {'expected_tool': 'EMBEDDING'},
    {'expected_tool': 'EMBEDDING'},
    {'expected_tool': 'EMBEDDING'},
]
out = compute_tool_accuracy(results, test_set)
assert out['correct'] == 2, f'expected 2 correct, got {out[\"correct\"]}'
assert out['total']   == 3
print(f'accuracy={out[\"accuracy\"]:.2f} correct={out[\"correct\"]}/{out[\"total\"]}')
print('ok')
"
```

Expected: `accuracy=0.67 correct=2/3` then `ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add compute_tool_accuracy function"
```

---

## Task 7: Update evaluate() loop — new accumulators and per-question collection

**Files:**
- Modify: `scripts/evaluate.py:128-212` (the `evaluate()` function)

This task updates `evaluate()` in two parts: (A) the accumulators before the loop, and (B) the loop body.

- [ ] **Step 1: Update the accumulators block (before the loop)**

Find the `# Accumulators` comment (around line 159) and replace the entire block from there through the `cat_stats` defaultdict definition with:

```python
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
```

- [ ] **Step 2: Update the loop body**

Find the loop body that starts with `# Agent — session ID...` (around line 185) and replace from the `session_id = ...` line through the `cat_stats[category]...` block with:

```python
        # Agent — session ID gắn run_id để không bị contaminate
        session_id = f"eval_{run_id}_{i}"
        result     = agent_search(question, url_to_id, session_id)

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
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('scripts/evaluate.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: collect latency, steps, judge scores, predictions in evaluate loop"
```

---

## Task 8: Update summary output and JSON save

**Files:**
- Modify: `scripts/evaluate.py` — the `# ── Summary` section onward

- [ ] **Step 1: Add post-loop metric computation**

Find the `# ── Summary ───` comment (around line 213) and insert the following block **before** it (i.e., after the loop ends and before the summary prints):

```python
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
```

- [ ] **Step 2: Replace the summary print block**

Find the existing `# ── Summary ───` block and replace everything from the `print("\n" + "=" * 65)` line through the end of the per-category section (before `# ── Save ──`) with:

```python
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
```

- [ ] **Step 3: Update the JSON save block**

Find `results: dict = {` (the save block, around line 270) and replace the entire `results` dict construction with:

```python
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
```

- [ ] **Step 4: Verify final syntax**

```bash
python -c "import ast; ast.parse(open('scripts/evaluate.py').read()); print('syntax ok')"
```

Expected: `syntax ok`

- [ ] **Step 5: Run a dry-run with 2 questions to confirm pipeline works end-to-end**

With the FastAPI server running (`uvicorn src.api.main:app --reload --port 8000`), temporarily edit the test set slice in the loop to only run 2 items (for speed), or run:

```bash
python -c "
import json
from pathlib import Path

# Temporarily patch test set to 2 questions
original = json.loads(Path('data/test_set.json').read_text(encoding='utf-8'))
Path('data/test_set_backup.json').write_text(
    json.dumps(original, indent=2, ensure_ascii=False), encoding='utf-8'
)
Path('data/test_set.json').write_text(
    json.dumps(original[:2], indent=2, ensure_ascii=False), encoding='utf-8'
)
print('Patched to 2 questions for dry run')
"
python scripts/evaluate.py
# Restore
python -c "
import json
from pathlib import Path
data = json.loads(Path('data/test_set_backup.json').read_text(encoding='utf-8'))
Path('data/test_set.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
Path('data/test_set_backup.json').unlink()
print('Restored test_set.json')
"
```

Expected output includes all new sections:
```
--- Answer Quality ---
  ROUGE-L            0.xxx
  BERT-Score F1      0.xxx
  LLM-Judge avg      x.xx/5.0  (n=2/2 valid)

--- Agent Quality ---
  Tool Accuracy:  x/2 = xx.x%
  Avg Steps:      x.x  (max: x)

--- Performance ---
  Latency p50:    x.xxs
  ...
```

- [ ] **Step 6: Run full evaluation**

```bash
python scripts/evaluate.py
```

Expected: completes in ~5-8 minutes (49 LLM-Judge calls + BERT-Score batch download on first run).

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add LLM-Judge, BERT-Score, Tool Accuracy, Latency, Steps to evaluate.py"
```
