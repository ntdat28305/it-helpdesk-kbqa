# 🤖 Agentic AI Audit — IT Helpdesk KBQA

> **Đánh giá:** Hệ thống có chuẩn agentic không?  
> **Phạm vi:** `src/agent/agent.py`, `neo4j_query.py`, `prompts.py`, `main.py`

---

## Tổng điểm: 7.5 / 10 — ✅ Agentic, có chỗ cần cải thiện

```
██████████████████████░░░░░ 7.5/10
```

---

## 📋 Checklist 10 tiêu chí Agentic AI

| # | Tiêu chí | Trạng thái | Điểm |
|---|---|---|---|
| 1 | ReAct Loop (Thought → Action → Observation) | ✅ Đạt | 9/10 |
| 2 | Tool Autonomy (LLM tự chọn tool) | ⚠️ Đạt có điều kiện | 7/10 |
| 3 | Multi-Step Reasoning (>1 bước) | ✅ Đạt | 8/10 |
| 4 | Self-Correction (thử lại khi thất bại) | ✅ Đạt | 8/10 |
| 5 | Memory & Context (multi-turn) | ✅ Đạt | 9/10 |
| 6 | Observation-Driven Decisions | ✅ Đạt | 8/10 |
| 7 | Answer Synthesis (tổng hợp cuối) | ✅ Đạt | 7/10 |
| 8 | Planning / Goal Decomposition | ❌ Thiếu | 4/10 |
| 9 | Reflection / Self-Evaluation | ❌ Thiếu | 3/10 |
| 10 | Transparency (reasoning steps) | ✅ Đạt | 9/10 |

---

## 🔍 Phân tích chi tiết

### 1. ✅ ReAct Loop — 9/10

**Verdict:** Chuẩn ReAct, dùng native function-calling thay vì parse text.

[agent.py:398-512](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L398-L512) — `_react_loop()`

```
Step 1: LLM → tool_calls → execute → observation → append to messages
Step 2: LLM (now has observation) → tool_calls or final answer
...up to MAX_STEPS=4
```

**Điểm mạnh:**
- Dùng Groq native `tool_calls` (không phải hack parse text)
- `parallel_tool_calls=False` — gọi 1 tool/step, đúng pattern ReAct
- Observations được append vào `messages` → LLM thấy toàn bộ lịch sử reasoning
- `tool_call_id` mapping chính xác

**Nhận xét:** Đây là ReAct chuẩn theo paper gốc. Rất tốt.

---

### 2. ⚠️ Tool Autonomy — 7/10

**Verdict:** LLM có quyền chọn tool, NHƯNG regex pre-routing override ở step 0.

[agent.py:50-56](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L50-L56) — `_forced_tool()`

```python
def _forced_tool(question: str) -> str | None:
    if _RE_ERROR_CODE.search(question):
        return "cypher_search"     # ← LLM bị override
    if _RE_WEBSEARCH.search(question):
        return "web_search"        # ← LLM bị override
    return None                    # ← LLM tự quyết
```

[agent.py:421-425](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L421-L425) — Force logic

```python
tool_choice = (
    {"type": "function", "function": {"name": first_tool}}
    if step == 0 and first_tool
    else "auto"
)
```

**Vấn đề:**
- Regex `_RE_ERROR_CODE` match `0x...`, `ERROR_XXX`, `KB...` → ép `cypher_search`
- Regex `_RE_WEBSEARCH` match `latest`, `2024`, `24H2` → ép `web_search`
- **CHỈ ở step 0**, steps sau vẫn `"auto"` → nên không quá nghiêm trọng

> [!NOTE]
> Đây là trade-off hợp lý: regex pre-routing giúp latency tốt hơn (giảm 1 bước LLM reasoning), nhưng giảm tính autonomous. Trong thực tế production, pattern này rất phổ biến (xem LangChain Conditional Routing). **Không phải anti-pattern**, nhưng nên document rõ là "heuristic hint" chứ không phải override cứng.

---

### 3. ✅ Multi-Step Reasoning — 8/10

**Verdict:** Agent CÓ THỂ chạy tới 4 bước, observations tích lũy cross-step.

```python
MAX_STEPS = 4  # line 30
```

**Điểm mạnh:**
- Mỗi step: LLM nhận toàn bộ messages (system + history + user + tool results)
- Observations tích lũy: `observations.append(obs)` → LLM thấy kết quả từ steps trước
- Empty observation → hint được inject: `"[Hint: previous tool returned no results...]"` (line 503-507)

**Điểm yếu:**
- `MAX_STEPS = 4` hơi thấp — nếu 3 tools trả empty, chỉ còn 1 step cuối
- Không có logic "stop early nếu confidence đủ cao" — phụ thuộc LLM tự quyết stop (đúng nhưng uncontrolled)

---

### 4. ✅ Self-Correction — 8/10

**Verdict:** Khi tool trả empty, agent nhận hint và thử tool khác.

[agent.py:496-507](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L496-L507)

```python
_EMPTY_OBS = (
    "No results found",
    "No similar entities",
    "No path found",
    "No web results",
)
if any(obs.startswith(m) for m in _EMPTY_OBS):
    obs_content = (
        obs + "\n\n[Hint: previous tool returned no results. "
        "Try a different tool or a broader / differently phrased entity.]"
    )
```

**Điểm mạnh:**
- Hint injection → nudge LLM thử tool khác
- System prompt cũng nói: "If a tool returns no results, try a different tool or rephrase"
- Fallback synthesis ở cuối loop (line 514-528) nếu loop exhausted

**Điểm yếu:**
- Không có retry cùng tool với entity khác (entity rephrasing phụ thuộc LLM)
- Không track đã dùng tool nào → có thể gọi lại cùng tool với cùng entity

---

### 5. ✅ Memory & Context — 9/10

**Verdict:** Multi-turn conversation rất tốt, có cả topic detection và ambiguity resolution.

[agent.py:563-592](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L563-L592) — `answer()`

| Feature | Mechanism | Code |
|---|---|---|
| Conversation history | `self.history` (list of user/assistant messages) | line 308 |
| History cap | 40 messages, trim oldest | line 588-589 |
| Follow-up detection | Regex `_RE_FOLLOWUP` + LLM `IS_AMBIGUOUS_PROMPT` | line 569-577 |
| Topic change | LLM `TOPIC_CHANGE_PROMPT` → auto reset | line 579-581 |
| Context window | Last 10 messages sent to ReAct loop | line 583 |
| Session management | Per-session agent via LRU OrderedDict (500 max) | `main.py:78-93` |

**Điểm mạnh:**
- 2-layer detection: regex (fast) → LLM (precise) cho ambiguity
- Topic change phát hiện tự động → reset history → tránh contamination
- Session isolation: mỗi `session_id` có `ITHelpdeskAgent` riêng

---

### 6. ✅ Observation-Driven Decisions — 8/10

**Verdict:** LLM nhận observations và quyết định bước tiếp theo dựa trên kết quả.

Observations được append vào `messages` dưới vai trò `"tool"`:
```python
messages.append({
    "role":         "tool",
    "tool_call_id": tc.id,
    "content":      obs_content,
})
```

→ Ở step tiếp theo, LLM thấy observation và quyết định:
- Gọi tool khác (nếu chưa đủ info)
- Trả lời (nếu đã đủ info → `not msg.tool_calls` → break)

**Điểm yếu:**
- LLM không explicitly output "Thought" text — Groq function-calling mode thường skip `content` khi có `tool_calls`
- → Không thể log reasoning chain rõ ràng (chỉ log tool calls, không log suy nghĩ)

---

### 7. ✅ Answer Synthesis — 7/10

**Verdict:** Có fallback synthesis, nhưng thiếu structured synthesis prompt.

[agent.py:514-528](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L514-L528)

```python
if not answer_text:
    if observations:
        synthesis = "\n\n---\n\n".join(observations)
        answer_text = llm_call(
            self.client,
            f"Based on the findings below, give a concise actionable answer.\n\n"
            f"Question: {question}\n\nFindings:\n{synthesis}",
            max_tokens=512,
        ) or ""
```

**Điểm mạnh:**
- Có synthesis step khi loop exhausted
- Community context enrichment (line 530-551)

**Điểm yếu:**
- Synthesis prompt quá đơn giản — không có instruction về format, sources attribution, confidence level
- Khi LLM tự stop (trả answer ở step 2-3), KHÔNG qua synthesis — answer quality phụ thuộc hoàn toàn vào LLM inline
- `tool_used` chỉ track tool cuối cùng, không phải tool chính → misleading trong eval

---

### 8. ❌ Planning / Goal Decomposition — 4/10

**Verdict:** KHÔNG có planning step. Agent reactive, không proactive.

**Thiếu gì:**
- Không có "plan before execute" step — agent nhận question và immediately chọn tool
- Không decompose câu hỏi phức tạp thành sub-questions
- Ví dụ: *"Compare the causes of BSOD on Windows 11 vs Windows 10 after update"* → agent sẽ chỉ gọi 1 tool, không biết cần 2 sub-queries

> [!IMPORTANT]
> Đây là điểm khác biệt lớn nhất giữa ReAct agent (reactive) và Plan-and-Execute agent (proactive). Hệ thống hiện tại là **ReAct thuần** — đủ tốt cho single-entity questions, nhưng yếu ở complex multi-entity reasoning.

**Cải thiện:**
- Thêm planning step: LLM phân tích question → output sub-tasks → execute tuần tự
- Hoặc dùng pattern "Plan-and-Solve" (Wang et al., 2023)

---

### 9. ❌ Reflection / Self-Evaluation — 3/10

**Verdict:** KHÔNG có reflection step. Agent không đánh giá chất lượng answer của mình.

**Thiếu gì:**
- Không có "self-critique" step sau khi sinh answer
- Không có confidence score
- Không có "Is this answer actually helpful?" check
- Hint injection (line 503-507) là dạng reflection nhẹ, nhưng chỉ ở level "tool returned empty"

> [!NOTE]
> Reflection (Reflexion pattern, Shinn et al. 2023) sẽ giúp agent tự đánh giá: *"Answer này có thực sự trả lời đúng câu hỏi không? Có cần tìm thêm không?"* Tuy nhiên, trade-off là thêm 1 LLM call → tăng latency.

---

### 10. ✅ Transparency — 9/10

**Verdict:** Agent expose reasoning steps đầy đủ cho frontend.

```python
steps.append({
    "step":        global_step,
    "tool":        _TOOL_NAME_MAP.get(fn_name, fn_name),
    "input":       step_input,
    "observation": obs[:300],
})
```

- Response trả về `steps[]` → UI render từng bước reasoning
- `tool_used`, `entity`, `sources` đều exposed
- Community context trong `context` field
- Streamlit UI render `render_steps()` với badge từng tool

---

## 📊 So sánh với các Agentic Patterns

| Pattern | Paper/Framework | Hệ thống hiện tại | Trạng thái |
|---|---|---|---|
| **ReAct** | Yao et al. 2022 | ✅ Implemented via function-calling loop | Chuẩn |
| **Tool Use** | Schick et al. 2023 (Toolformer) | ✅ 4 tools, native function-calling | Chuẩn |
| **Multi-Turn** | — | ✅ History + topic detection + ambiguity | Tốt |
| **Plan-and-Solve** | Wang et al. 2023 | ❌ Không có planning step | Thiếu |
| **Reflexion** | Shinn et al. 2023 | ❌ Không có self-reflection | Thiếu |
| **Chain-of-Thought** | Wei et al. 2022 | ⚠️ Implicit (Groq function-calling skip thoughts) | Partial |
| **Self-Ask** | Press et al. 2022 | ⚠️ Hint injection là dạng nhẹ | Partial |

---

## 🎯 Kết luận

### Hệ thống ĐÃ CHUẨN AGENTIC ở mức ReAct Agent

**Đạt 7 / 10 tiêu chí cốt lõi:**
1. ✅ ReAct loop thật sự (không fake single-pass)
2. ✅ LLM tự chọn tool (với heuristic hint hợp lý)
3. ✅ Multi-step reasoning (≤4 steps, observations tích lũy)
4. ✅ Self-correction (empty → hint → retry)
5. ✅ Multi-turn memory (history + topic detection)
6. ✅ Observation-driven decisions
7. ✅ Full transparency (steps exposed to UI)

### Nếu muốn nâng cấp lên "Advanced Agentic":

| Priority | Feature | Effort | Impact |
|---|---|---|---|
| 🟡 Medium | **Planning step** — decompose complex queries | ~50 LOC | Cải thiện multi-entity questions |
| 🟡 Medium | **Confidence score** — LLM tự đánh giá answer | ~30 LOC | Giúp user biết khi nào tin agent |
| 🟢 Low | **Tool dedup** — track đã dùng tool nào | ~10 LOC | Tránh gọi lại cùng tool+entity |
| 🟢 Low | **Explicit thought logging** — extract `content` from tool_calls response | ~5 LOC | Improve transparency |
| 🔴 High | **Reflection step** — self-critique before returning | ~40 LOC + 1 LLM call | Tăng answer quality, tăng latency |

> [!TIP]
> Với scope academic project, hệ thống hiện tại **đủ chuẩn agentic** để present. Planning và Reflection là advanced patterns thường thấy ở production-grade agents (AutoGPT, CrewAI, OpenAI Assistants). Nếu có thời gian, thêm planning step sẽ là improvement có ROI cao nhất.
