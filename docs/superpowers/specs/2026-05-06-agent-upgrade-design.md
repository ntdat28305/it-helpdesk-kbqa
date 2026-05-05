# Agent Upgrade Design — 5 Agentic Improvements
**Date:** 2026-05-06  
**Status:** Approved  
**Scope:** `src/agent/agent.py`, `src/agent/prompts.py`

---

## Summary

Nâng cấp `ITHelpdeskAgent` từ ReAct thuần (7.5/10) lên Advanced Agentic bằng cách thêm 5 cải tiến:

1. **Planning step** — lightweight thinking note trước ReAct loop
2. **Reflection step** — self-evaluation sau khi có answer
3. **Confidence score** — exposed trong response JSON
4. **Tool dedup** — tránh gọi cùng tool+input 2 lần
5. **Thought logging** — log `msg.content` khi LLM không gọi tool

---

## Architecture

### Approach: Method injection

Giữ nguyên `_react_loop()`. Thêm 2 method mới vào class, gọi từ `answer()`:

```
answer(question)
  │
  ├─ [hiện có] ambiguity check + topic change detection
  │
  ├─ [MỚI] _plan(question) → plan_note: str
  │
  ├─ _react_loop(question, history_context, plan_note)   ← nhận plan_note
  │   ├─ [MỚI] plan_note inject vào system prompt
  │   ├─ [MỚI] used_tool_inputs: set() để dedup
  │   └─ [MỚI] thought logging khi msg.content tồn tại
  │
  ├─ [MỚI] _reflect(question, answer_text, observations)
  │         → is_sufficient: bool, confidence: str, reason: str
  │         → nếu !is_sufficient: trigger re-synthesis từ observations
  │
  └─ return { ...result, plan_note, confidence, reflection_reason }
```

---

## Components

### 1. `_plan(question: str) -> str`

- **1 LLM call** với `PLAN_PROMPT`
- Output: plain text 1–2 câu, lưu vào `plan_note`
- `plan_note` được append vào system message của `_react_loop()` dưới dạng:
  `"[Planning note: {plan_note}]"`
- Nếu LLM call fail → `plan_note = ""` (graceful degradation, không crash)

### 2. `_react_loop()` — thay đổi nhỏ

**Tool dedup:**
```python
used_tool_inputs: set[tuple] = set()
key = (fn_name, str(tool_args))
if key in used_tool_inputs:
    obs = "[Skipped: same tool+input already used in this query]"
else:
    used_tool_inputs.add(key)
    obs = _execute_tool(...)
```

**Thought logging:**
```python
if msg.content and not msg.tool_calls:
    steps.append({
        "step": global_step,
        "tool": "thought",
        "input": "",
        "observation": msg.content[:300],
    })
```

`_react_loop()` nhận thêm param `plan_note: str = ""`. Signature cũ vẫn tương thích nhờ default value.

`_react_loop()` return thêm `observations: list[str]` (hiện tại là local var) để `answer()` truyền vào `_reflect()`. Return tuple: `(answer_text, steps, tool_used, entity, sources, observations)`.

### 3. `_reflect(question, answer_text, observations) -> dict`

- **1 LLM call** với `REFLECT_PROMPT`
- Parse JSON response: `{"is_sufficient": bool, "confidence": str, "reason": str}`
- Nếu `is_sufficient=False` và có `observations` → re-run synthesis (dùng code synthesis hiện có ở line 514-528)
- Nếu JSON parse fail → default `{"confidence": "medium", "is_sufficient": True, "reason": ""}`

### 4. Response schema

```python
# Thêm vào dict hiện tại:
{
    "plan_note": "This is a network connectivity issue...",   # str
    "confidence": "high",                                     # "low" | "medium" | "high"
    "reflection_reason": "Answer includes specific steps..."  # str
}
```

---

## Prompts mới (src/agent/prompts.py)

### `PLAN_PROMPT`

```python
PLAN_PROMPT = """You are an IT helpdesk assistant. Briefly analyze what the user needs.
In 1-2 sentences, describe: what type of problem this is and what information would best answer it.
Be concise — this is an internal thinking note, not shown to the user.

Question: {question}
Plan:"""
```

### `REFLECT_PROMPT`

```python
REFLECT_PROMPT = """Evaluate this IT helpdesk answer.

Question: {question}
Answer: {answer}
Sources used: {num_sources} source(s), tools: {tools_used}

Reply with ONLY valid JSON, no markdown:
{{"is_sufficient": true/false, "confidence": "low"/"medium"/"high", "reason": "one sentence"}}

Criteria:
- is_sufficient: does the answer actually address the question?
- confidence: high=clear answer with sources, medium=partial info, low=generic/no sources"""
```

---

## Error Handling

| Scenario | Behavior |
|---|---|
| `_plan()` LLM call fail / timeout | `plan_note = ""`, tiếp tục bình thường |
| `_reflect()` LLM call fail | default `confidence="medium"`, `is_sufficient=True` |
| `_reflect()` JSON parse fail | same default above |
| Tool dedup hit | obs = skip message, loop tiếp tục |

---

## Files thay đổi

| File | Thay đổi |
|---|---|
| `src/agent/prompts.py` | Thêm `PLAN_PROMPT`, `REFLECT_PROMPT` |
| `src/agent/agent.py` | Thêm `_plan()`, `_reflect()`, sửa `_react_loop()` signature + body, sửa `answer()` |

Không thay đổi API schema, Dockerfile, hay các module khác.

---

## Latency estimate

| Phase | Hiện tại | Sau nâng cấp |
|---|---|---|
| Preprocessing | ~1–2 calls | ~1–2 calls |
| Planning | — | +1 call |
| ReAct loop | ~2–4 calls | ~2–4 calls |
| Reflection | — | +1 call |
| **Tổng** | **~3–6 calls (~1.5–3s)** | **~5–8 calls (~2.5–4s)** |
