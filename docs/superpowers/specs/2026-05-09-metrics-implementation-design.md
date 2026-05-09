# Design: Extended Evaluation Metrics for IT Helpdesk KBQA

**Date:** 2026-05-09  
**Status:** Approved  
**Scope:** `scripts/evaluate.py`, `data/test_set.json`

---

## 1. Goals

Add 5 metrics to the evaluation pipeline to support both report/thesis reporting and agent debugging:

| Metric | Purpose |
|---|---|
| LLM-as-Judge | Semantic answer correctness (replaces ROUGE-L as primary quality metric) |
| BERT-Score F1 | Semantic similarity (complements ROUGE-L) |
| Tool Accuracy | Routing quality — which tool agent chose vs expected |
| Latency p50/p95 | Performance monitoring |
| Avg Steps | Agent behavior analysis |

---

## 2. Approach

**Extend `evaluate.py` in-place** (Approach A). File goes from ~313 to ~500 lines. No new files, no new modules. Single command to run as before:

```bash
python scripts/evaluate.py
```

---

## 3. Architecture

### 3.1 Changes to `agent_search()` return type

Current:
```python
tuple[list[str], str, str]  # (article_ids, tool_used, answer)
```

New — return a dict for forward compatibility:
```python
dict with keys: article_ids, tool_used, answer, latency, steps_count
```

Latency is measured by wrapping the `requests.post` call with `time.time()`. Steps count comes from `len(data.get("steps", []))` in the API response.

### 3.2 Loop changes (per-question)

In the main `evaluate()` loop, after getting the agent result:
1. Record `latency` from the result dict
2. Record `steps_count` from the result dict
3. Call `llm_judge(question, ref_answer, ag_answer)` → append score if not None
4. Append `ag_answer` and `ref_answer` to batch lists for BERT-Score

### 3.3 Post-loop (batch)

After the loop completes:
1. `compute_bert_score(all_predictions, all_references)` — skips entries where answer is empty
2. `compute_tool_accuracy(all_results, test_set)` — skips entries without `expected_tool`
3. Print full summary with new sections

### 3.4 New functions to add

```
llm_judge(question, reference, agent_answer) -> int | None
compute_bert_score(predictions, references) -> float  # returns F1 only
compute_tool_accuracy(results, test_set) -> dict
```

---

## 4. LLM-as-Judge

- **Model:** `llama-3.3-70b-versatile` via Groq (`GROQ_API_KEY_1`)
- **Temperature:** 0, `max_tokens=5`
- **Scale:** 1–5 integer
- **Parse fail:** return `None`, exclude from avg (not 0 — avoids biasing results)
- **Scope:** added to per-tool and per-category breakdowns

Prompt (unchanged from metrics_analysis.md):
```
You are an IT support expert evaluating answer quality.
Question: {question}
Reference Answer: {reference}
Agent Answer: {agent_answer}
Rate the agent's answer on a scale of 1-5:
1=Completely wrong, 2=Partially relevant, 3=Somewhat correct,
4=Mostly correct, 5=Fully correct and comprehensive
Output ONLY the number (1-5):
```

---

## 5. BERT-Score

- **Model:** `roberta-large` (not `deberta-xlarge-mnli` — ~60% lighter, sufficient accuracy)
- **Language:** `en`
- **Report:** F1 only
- **Skips:** entries where `ag_answer == ""` (API error)
- **First run:** downloads ~500MB model, cached thereafter
- **Dependency:** `bert-score` added to `requirements.txt`

---

## 6. Tool Accuracy

### 6.1 Annotation rules for `test_set.json`

Add `"expected_tool"` field to each of the 49 entries. Apply rules in **priority order** (first match wins — mirrors agent's regex pre-routing logic):

| Priority | Question pattern | expected_tool |
|---|---|---|
| 1 | Contains error code (0x…, ERROR_*, KB\d{6,7}, HRESULT) | `CYPHER` |
| 2 | Asks relationship between 2 entities ("difference between", "how does X relate to Y") | `BFS` |
| 3 | Asks about latest/recent versions or time-sensitive info ("latest", "24H2", "recent") | `WEBSEARCH` |
| 4 | General symptom / how-to / troubleshooting (default) | `EMBEDDING` |

### 6.2 Code

- Backward compatible: entries without `expected_tool` are skipped silently
- Outputs overall accuracy + confusion matrix (expected → actual, count)
- Stored in `data/eval_results.json` under `"tool_accuracy"` key

---

## 7. Latency & Steps Count

- **Latency:** `time.time()` wrap around `requests.post` in `agent_search()`; report p50, p95, avg
- **Steps:** `len(data.get("steps", []))` from API response; report avg and max
- No changes to API or agent required

---

## 8. Output Format

```
================================================================
IT HELPDESK KBQA — EVALUATION REPORT
================================================================
Test set: 49 questions, 4 categories  |  Run: <run_id>

--- Retrieval ---
  Metric         BM25    Agent    Delta
  Hit@1         0.347    0.122   -0.225
  Hit@5         0.469    0.224   -0.245
  MRR           0.398    0.181   -0.217

--- Answer Quality ---
  ROUGE-L:          0.121
  BERT-Score F1:    0.xxx
  LLM-Judge (avg):  x.xx / 5.0  (n=47/49 valid)

--- Agent Quality ---
  Tool Accuracy:    xx/49 = xx.x%
  Avg Steps:        x.x  (max: x)

--- Performance ---
  Latency p50:      x.xx s
  Latency p95:      x.xx s

--- Tool Accuracy Confusion ---
  CYPHER→CYPHER:         x ✓
  CYPHER→EMBEDDING:      x ✗
  EMBEDDING→EMBEDDING:   x ✓
  ...

--- Per-Tool Breakdown ---
  Tool         N  Hit@1  Hit@5   MRR  ROUGE-L  LLM-Judge
  CYPHER       x  x.xxx  x.xxx  x.xxx  x.xxx    x.xx
  ...

--- Per-Category Breakdown ---
  ...
================================================================
```

---

## 9. `data/eval_results.json` schema additions

```json
{
  "agent": {
    "hit@1": 0.122,
    "hit@5": 0.224,
    "mrr": 0.181,
    "rouge_l": 0.121,
    "bert_score_f1": 0.0,
    "llm_judge_avg": 0.0,
    "llm_judge_n_valid": 0,
    "tool_accuracy": 0.0,
    "tool_accuracy_n": 0,
    "latency_p50": 0.0,
    "latency_p95": 0.0,
    "latency_avg": 0.0,
    "avg_steps": 0.0,
    "max_steps": 0
  }
}
```

---

## 10. Dependencies

Add to `requirements.txt`:
```
bert-score>=0.3.13
```

`groq` is already used by the agent — no new package needed for LLM-Judge.

---

## 11. Out of Scope

- Source Recall (multi-ground-truth per question) — requires re-annotating test set
- Streaming evaluation output / progress bar — nice to have, not needed
- Separate evaluate_v2.py — rejected in favor of in-place extension
