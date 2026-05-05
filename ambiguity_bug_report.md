# 🐛 Bug Report: Ambiguous Query Hallucination

## Mô tả

Khi user hỏi câu mơ hồ (ví dụ `"how do i fix it"`) mà **chưa có lịch sử hội thoại**, agent LLM **tự bịa ra entity** không liên quan rồi trả lời sai hoàn toàn.

---

## Tái hiện

**Bước 1:** Mở session mới (chưa hỏi gì)  
**Bước 2:** Hỏi `"how do i fix it"`  
**Kết quả:**

```log
Question: how do i fix it
Tool call: embedding_search({'entity': 'my laptop keeps freezing'})   ← BỊA
Tool call: cypher_search({'entity': 'profile installation failed...'}) ← BỊA
Tool call: web_search({'query': 'profile installation failed...'})     ← BỊA
Answer: The issue with your laptop freezing can be caused by...        ← SAI
```

**Kỳ vọng:** Agent nên hỏi lại user để làm rõ vấn đề.

---

## Nguyên nhân gốc

[agent.py:567-583](file:///q:/OneDrive%20-%20VNU-HCMUS/NLPDoanhNghiep/Project/it-helpdesk-kbqa/src/agent/agent.py#L567-L583)

```python
if self.history:           # ← history rỗng → SKIP toàn bộ
    if _RE_FOLLOWUP.search(question):
        ambiguous = True
    else:
        ...
    if not ambiguous and self._detect_topic_change(question):
        self.reset_history()

history_context = self.history[-10:]  # = []
result = self._react_loop(question, history_context)  # LLM chỉ nhận "how do i fix it"
```

**Lỗ hổng:** Ambiguity check chỉ chạy khi `self.history` **có dữ liệu**. Khi history rỗng, câu hỏi mơ hồ đi thẳng vào ReAct loop → LLM không có context → hallucinate entity.

---

## Phân tích ảnh hưởng

| Tình huống | History | Kết quả hiện tại | Đúng/Sai |
|---|---|---|---|
| Câu hỏi rõ ràng, session mới | Rỗng | Agent trả lời bình thường | ✅ |
| Câu hỏi rõ ràng, đang chat | Có | Agent trả lời bình thường | ✅ |
| Follow-up (có "it", "this"), đang chat | Có | Giữ history, trả lời đúng context | ✅ |
| Đổi topic, đang chat | Có | Reset history, trả lời topic mới | ✅ |
| **Follow-up, session mới** | **Rỗng** | **LLM bịa entity, trả lời sai** | ❌ |

---

## Giải pháp đề xuất

### Option A: Trả lời hỏi lại (đơn giản, an toàn)

Thêm check trước block `if self.history` trong method `answer()`:

```python
def answer(self, question: str) -> dict:
    # ── Guard: câu mơ hồ khi chưa có history ──
    if not self.history and _RE_FOLLOWUP.search(question):
        return {
            "question": question,
            "tool_used": "",
            "entity": "",
            "answer": "Could you provide more details? "
                      "For example, what device, error code, or issue "
                      "are you experiencing?",
            "sources": [],
            "context": "",
            "steps": [],
        }

    # ── Code hiện tại giữ nguyên ──
    if self.history:
        ...
```

**Ưu điểm:** Đơn giản, không hallucinate, UX tự nhiên.  
**Nhược điểm:** User phải hỏi lại.

---

### Option B: Thử xử lý bằng LLM (phức tạp hơn)

Cho LLM cố gắng "suy đoán" ý user trước khi từ chối:

```python
if not self.history and _RE_FOLLOWUP.search(question):
    clarified = llm_call(
        self.client,
        f"The user asked: '{question}' without any prior context. "
        f"If you can reasonably infer what they mean, rewrite as a "
        f"clear standalone IT question. If not, respond with UNCLEAR.",
        max_tokens=50,
    )
    if "UNCLEAR" in clarified.upper():
        return {"answer": "Could you provide more details?", ...}
    else:
        question = clarified  # dùng câu hỏi đã rewrite
```

**Ưu điểm:** Không bắt user hỏi lại nếu câu hỏi đoán được.  
**Nhược điểm:** LLM 8B có thể suy đoán sai → vẫn hallucinate.

---

### Option C: Nâng model ReAct lên 70B (bổ trợ)

Đổi model trong ReAct loop để giảm hallucination:

```python
# agent.py:428
model="llama-3.3-70b-versatile"  # thay vì llama-3.1-8b-instant
```

**Ưu điểm:** Cải thiện toàn diện (tool calling, coreference, reasoning).  
**Nhược điểm:** Không fix được case history rỗng + câu mơ hồ. Cần kết hợp với Option A hoặc B.

---

## Khuyến nghị

> **Dùng Option A + Option C:**
> - Option A: guard chắc chắn cho edge case history rỗng (~5 dòng code)
> - Option C: nâng model cho ReAct → cải thiện chất lượng tổng thể (~1 dòng code)

### Mức độ ưu tiên

| Fix | Effort | Impact | Priority |
|---|---|---|---|
| Option A (guard clause) | ~5 LOC | Fix bug trực tiếp | 🔴 Cao |
| Option C (đổi model 70B) | ~1 LOC | Cải thiện tổng thể | 🟡 Trung bình |
| Option B (LLM rewrite) | ~15 LOC | Edge case improvement | 🟢 Thấp |
