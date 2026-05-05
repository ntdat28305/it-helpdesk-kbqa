# Agent Upgrade — 5 Agentic Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nâng cấp `ITHelpdeskAgent` bằng cách thêm Planning step, Reflection step, Confidence score, Tool dedup, và Thought logging — không thay đổi API schema hay các module khác.

**Architecture:** Dùng method injection pattern — thêm `_plan()` và `_reflect()` vào class, gọi từ `answer()`. `_react_loop()` nhận thêm `plan_note` param và được sửa nhỏ để dedup + thought logging + return `observations`. Không thay đổi bất kỳ file nào ngoài `src/agent/prompts.py` và `src/agent/agent.py`.

**Tech Stack:** Python, Groq SDK (`llama-3.1-8b-instant`), `json` stdlib (để parse reflection output).

---

## File Map

| File | Thay đổi |
|---|---|
| `src/agent/prompts.py` | Thêm `PLAN_PROMPT`, `REFLECT_PROMPT` |
| `src/agent/agent.py` | Update import; thêm `_plan()`, `_reflect()`; sửa `_react_loop()` signature + body; sửa `answer()` |

---

## Task 1: Thêm PLAN_PROMPT và REFLECT_PROMPT vào prompts.py

**Files:**
- Modify: `src/agent/prompts.py`

- [ ] **Step 1: Append hai prompt vào cuối file `src/agent/prompts.py`**

Thêm vào cuối file (sau `TOPIC_CHANGE_PROMPT`):

```python
# ── Prompt planning (lightweight thinking note) ───────────────

PLAN_PROMPT = """You are an IT helpdesk assistant. Briefly analyze what the user needs.
In 1-2 sentences, describe: what type of problem this is and what information would best answer it.
Be concise — this is an internal thinking note, not shown to the user.

Question: {question}
Plan:"""


# ── Prompt reflection (self-evaluation) ──────────────────────

REFLECT_PROMPT = """Evaluate this IT helpdesk answer.

Question: {question}
Answer: {answer}
Sources used: {num_sources} source(s), tools: {tools_used}

Reply with ONLY valid JSON, no markdown:
{{"is_sufficient": true, "confidence": "high", "reason": "one sentence"}}

Confidence criteria:
- "high": clear, actionable answer with specific steps and sources
- "medium": partial info, generic steps, or only 1 weak source
- "low": no sources, vague answer, or answer does not address the question

is_sufficient: false only when the answer is completely off-topic or empty."""
```

- [ ] **Step 2: Verify file trông đúng**

```bash
python -c "from src.agent.prompts import PLAN_PROMPT, REFLECT_PROMPT; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/agent/prompts.py
git commit -m "feat: add PLAN_PROMPT and REFLECT_PROMPT for planning and reflection steps"
```

---

## Task 2: Update import trong agent.py

**Files:**
- Modify: `src/agent/agent.py` (line 24)

- [ ] **Step 1: Sửa dòng import ở line 24**

Đổi từ:
```python
from src.agent.prompts import IS_AMBIGUOUS_PROMPT, TOPIC_CHANGE_PROMPT
```

Thành:
```python
from src.agent.prompts import (
    IS_AMBIGUOUS_PROMPT,
    TOPIC_CHANGE_PROMPT,
    PLAN_PROMPT,
    REFLECT_PROMPT,
)
```

- [ ] **Step 2: Verify import hoạt động**

```bash
python -c "from src.agent.agent import ITHelpdeskAgent; print('Import OK')"
```

Expected output: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: import PLAN_PROMPT and REFLECT_PROMPT in agent.py"
```

---

## Task 3: Thêm method `_plan()` vào class `ITHelpdeskAgent`

**Files:**
- Modify: `src/agent/agent.py` — thêm method sau `_detect_topic_change()` (sau line 326)

- [ ] **Step 1: Thêm method `_plan()` vào class, ngay sau `_detect_topic_change()`**

Thêm đoạn code này sau closing của `_detect_topic_change()` (sau line 326), trước `_execute_tool()`:

```python
    def _plan(self, question: str) -> str:
        """Generate a lightweight planning note before the ReAct loop."""
        try:
            note = llm_call(
                self.client,
                PLAN_PROMPT.format(question=question),
                max_tokens=80,
            )
            result = (note or "").strip()
            logger.info(f"Plan: {result[:100]}")
            return result
        except Exception:
            return ""
```

- [ ] **Step 2: Verify class load không lỗi**

```bash
python -c "from src.agent.agent import ITHelpdeskAgent; a = ITHelpdeskAgent(); print('_plan' in dir(a))"
```

Expected output: `True`

- [ ] **Step 3: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: add _plan() method for lightweight pre-loop planning note"
```

---

## Task 4: Thêm method `_reflect()` vào class `ITHelpdeskAgent`

**Files:**
- Modify: `src/agent/agent.py` — thêm method sau `_plan()` (trước `_execute_tool()`)

- [ ] **Step 1: Thêm method `_reflect()` vào class, ngay sau `_plan()`**

```python
    def _reflect(
        self,
        question: str,
        answer_text: str,
        num_sources: int,
        tool_used: str,
    ) -> dict:
        """Self-evaluate the generated answer; return confidence metadata."""
        _default = {"is_sufficient": True, "confidence": "medium", "reason": ""}
        if not answer_text:
            return {"is_sufficient": False, "confidence": "low", "reason": "Empty answer"}
        try:
            raw = llm_call(
                self.client,
                REFLECT_PROMPT.format(
                    question=question,
                    answer=answer_text[:600],
                    num_sources=num_sources,
                    tools_used=tool_used,
                ),
                max_tokens=80,
            )
            if not raw:
                return _default
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            return {
                "is_sufficient": bool(parsed.get("is_sufficient", True)),
                "confidence":    str(parsed.get("confidence", "medium")),
                "reason":        str(parsed.get("reason", "")),
            }
        except Exception as e:
            logger.warning(f"Reflection parse error: {e}")
            return _default
```

- [ ] **Step 2: Verify method tồn tại**

```bash
python -c "from src.agent.agent import ITHelpdeskAgent; a = ITHelpdeskAgent(); print('_reflect' in dir(a))"
```

Expected output: `True`

- [ ] **Step 3: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: add _reflect() method for answer self-evaluation and confidence scoring"
```

---

## Task 5: Sửa `_react_loop()` — thêm plan_note param, tool dedup, thought logging, return observations

**Files:**
- Modify: `src/agent/agent.py` — `_react_loop()` (lines 398–561)

- [ ] **Step 1: Đổi signature của `_react_loop()` (line 398–400)**

Từ:
```python
    def _react_loop(
        self, question: str, history_context: list[dict]
    ) -> dict:
```

Thành:
```python
    def _react_loop(
        self, question: str, history_context: list[dict], plan_note: str = ""
    ) -> dict:
```

- [ ] **Step 2: Inject `plan_note` vào system message (lines 402–406)**

Từ:
```python
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history_context,
            {"role": "user", "content": question},
        ]
```

Thành:
```python
        system_content = (
            SYSTEM_PROMPT + f"\n\n[Planning note: {plan_note}]"
            if plan_note
            else SYSTEM_PROMPT
        )
        messages: list[dict] = [
            {"role": "system", "content": system_content},
            *history_context,
            {"role": "user", "content": question},
        ]
```

- [ ] **Step 3: Thêm `used_tool_inputs` set ngay sau khai báo biến (sau line 414)**

Tìm đoạn:
```python
        first_tool = _forced_tool(question)
```

Thêm vào TRƯỚC dòng đó:
```python
        used_tool_inputs: set[tuple] = set()

        first_tool = _forced_tool(question)
```

- [ ] **Step 4: Thêm thought logging trong branch `if not msg.tool_calls:` (line 442–445)**

Từ:
```python
            if not msg.tool_calls:
                answer_text = (msg.content or "").strip()
                logger.info(f"Final answer at step {step + 1}")
                break
```

Thành:
```python
            if not msg.tool_calls:
                answer_text = (msg.content or "").strip()
                if answer_text:
                    global_step += 1
                    steps.append({
                        "step":        global_step,
                        "tool":        "thought",
                        "input":       "",
                        "observation": answer_text[:300],
                    })
                logger.info(f"Final answer at step {step + 1}")
                break
```

- [ ] **Step 5: Thêm tool dedup check trước `_execute_tool()` (line 472)**

Tìm đoạn:
```python
                logger.info(f"Tool call: {fn_name}({args})")
                obs, step_sources, step_entity = self._execute_tool(fn_name, args)
```

Thay bằng:
```python
                logger.info(f"Tool call: {fn_name}({args})")
                _dedup_key = (fn_name, json.dumps(args, sort_keys=True))
                if _dedup_key in used_tool_inputs:
                    obs, step_sources, step_entity = (
                        "[Skipped: same tool+input already used in this query]",
                        [],
                        "",
                    )
                    logger.info("Tool dedup: skipped duplicate call")
                else:
                    used_tool_inputs.add(_dedup_key)
                    obs, step_sources, step_entity = self._execute_tool(fn_name, args)
```

- [ ] **Step 6: Thêm `observations` vào return dict (line 553–561)**

Từ:
```python
        return {
            "question":  question,
            "tool_used": tool_used,
            "entity":    entity,
            "answer":    answer_text,
            "sources":   sources,
            "context":   context,
            "steps":     steps,
        }
```

Thành:
```python
        return {
            "question":     question,
            "tool_used":    tool_used,
            "entity":       entity,
            "answer":       answer_text,
            "sources":      sources,
            "context":      context,
            "steps":        steps,
            "observations": observations,
        }
```

- [ ] **Step 7: Verify syntax OK**

```bash
python -c "from src.agent.agent import ITHelpdeskAgent; print('OK')"
```

Expected output: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: add plan_note injection, tool dedup, thought logging, return observations in _react_loop()"
```

---

## Task 6: Sửa `answer()` để gọi `_plan()`, `_reflect()`, và merge kết quả

**Files:**
- Modify: `src/agent/agent.py` — `answer()` (lines 563–592)

- [ ] **Step 1: Thêm `_plan()` call trong `answer()` trước `_react_loop()`**

Tìm đoạn (line 583–584):
```python
        history_context = self.history[-10:] if self.history else []
        result = self._react_loop(question, history_context)
```

Thay bằng:
```python
        history_context = self.history[-10:] if self.history else []
        plan_note = self._plan(question)
        result = self._react_loop(question, history_context, plan_note)
```

- [ ] **Step 2: Thêm `_reflect()` call và merge fields, sau `_react_loop()` và trước history append**

Tìm đoạn (line 586–589):
```python
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": result["answer"]})
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]
```

Thêm TRƯỚC đoạn đó:
```python
        reflection = self._reflect(
            question,
            result["answer"],
            len(result["sources"]),
            result["tool_used"],
        )
        if not reflection["is_sufficient"] and result.get("observations"):
            synthesis = "\n\n---\n\n".join(result["observations"])
            new_answer = llm_call(
                self.client,
                f"Based on the findings below, give a concise actionable answer.\n\n"
                f"Question: {question}\n\nFindings:\n{synthesis}",
                max_tokens=512,
            )
            if new_answer:
                result["answer"] = new_answer
                logger.info("Reflection triggered re-synthesis")

        result["plan_note"]          = plan_note
        result["confidence"]         = reflection["confidence"]
        result["reflection_reason"]  = reflection["reason"]
```

- [ ] **Step 3: Verify syntax OK**

```bash
python -c "from src.agent.agent import ITHelpdeskAgent; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: wire _plan() and _reflect() into answer(), add plan_note/confidence/reflection_reason to response"
```

---

## Task 7: Smoke test end-to-end

**Files:**
- No file changes — chỉ chạy test

- [ ] **Step 1: Chạy CLI test tích hợp sẵn**

```bash
python -m src.agent.agent
```

Expected: không crash, output mỗi query có `Tool: ... | Entity: ...` và answer text. Không cần thấy `plan_note`/`confidence` ở đây vì CLI test chỉ print `result['answer']`.

- [ ] **Step 2: Chạy quick sanity check in Python**

```bash
python -c "
from src.agent.agent import ITHelpdeskAgent
agent = ITHelpdeskAgent()
r = agent.answer('My Teams meeting keeps dropping')
print('Keys:', sorted(r.keys()))
print('Confidence:', r.get('confidence'))
print('Plan:', r.get('plan_note', '')[:80])
"
```

Expected output có đủ keys:
```
Keys: ['answer', 'confidence', 'context', 'entity', 'observations', 'plan_note', 'question', 'reflection_reason', 'sources', 'steps', 'tool_used']
Confidence: high   # hoặc medium/low
Plan: This is a...
```

- [ ] **Step 3: Verify response JSON từ API (nếu FastAPI đang chạy)**

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Teams meeting drops","session_id":"test-smoke"}' \
  | python -m json.tool | grep -E '"confidence"|"plan_note"|"reflection_reason"'
```

Expected: 3 fields xuất hiện trong response.

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "docs: update agentic_audit.md status after upgrade (planning, reflection, dedup, thought logging)"
```

---

## Tóm tắt thay đổi

| Cải tiến | Task | File | LOC thêm |
|---|---|---|---|
| PLAN_PROMPT + REFLECT_PROMPT | 1 | prompts.py | ~25 |
| `_plan()` method | 3 | agent.py | ~12 |
| `_reflect()` method | 4 | agent.py | ~30 |
| Tool dedup trong `_react_loop()` | 5 | agent.py | ~10 |
| Thought logging trong `_react_loop()` | 5 | agent.py | ~7 |
| `plan_note` injection, return `observations` | 5 | agent.py | ~10 |
| Wire `_plan()` + `_reflect()` vào `answer()` | 6 | agent.py | ~18 |
| **Tổng** | | | **~112 LOC** |
