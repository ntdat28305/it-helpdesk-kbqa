# Project Audit Report

> Ngày audit: 2026-05-06  
> Phạm vi: toàn bộ codebase (agent, API, UI, KG build, embedding, Docker)

---

## Mức độ ưu tiên

| ID | File | Vấn đề | Mức độ |
|----|------|---------|--------|
| B1 | `src/api/main.py:125-133` | Response không truyền `plan_note`, `confidence`, `reflection_reason` | 🔴 Bug |
| B2 | `src/agent/agent.py:360` | `_reflect()` default `is_sufficient=True` → re-synthesis không bao giờ trigger khi parse lỗi | 🔴 Bug |
| B3 | `src/api/main.py:55` | Default `session_id="default"` → nhiều user dùng chung 1 session | 🔴 Bug |
| B4 | `src/agent/neo4j_query.py:51,99` | `cypher_search` và `bfs_search` không có try-except → Neo4j down crash API | 🟠 Bug |
| B5 | `src/kg_build/summarizer.py:93,105` | `x[1]["nodes"]` và `comm_data["nodes"]` không guard → `KeyError` nếu dict thiếu key | 🟠 Bug |
| D1 | `src/embedding/train_gcn.py:35-37` | Thiếu CUDA seed → kết quả không reproducible khi chạy GPU | 🟡 Design |
| D2 | `src/agent/agent.py:724-725` | History truncation chạy sau khi append → thực tế cho phép `MAX+2` messages | 🟡 Design |
| D3 | `src/api/main.py:82-96` | `OrderedDict` session không có lock → race condition khi concurrent requests | 🟡 Design |
| D4 | `src/agent/agent.py:40-46` | `_RE_WEBSEARCH` match năm `2024`/`2025` quá rộng → force web search sai | 🟡 Design |
| N1 | `src/embedding/train_gcn.py:237` | Chỉ save weights cuối, không save best epoch | 🟢 Minor |
| N2 | `docker-compose.yml:17` vs `Dockerfile.api:12` | Health check syntax không nhất quán | 🟢 Minor |

---

## Chi tiết từng vấn đề

### B1 — Response bị mất 3 fields `plan_note` / `confidence` / `reflection_reason`

**File:** [src/api/main.py:125-133](src/api/main.py)

```python
# Hiện tại — thiếu 3 fields
return QueryResponse(
    question=result["question"],
    answer=result["answer"],
    tool_used=result["tool_used"],
    entity=result["entity"],
    sources=result["sources"],
    session_id=request.session_id,
    steps=result.get("steps", []),
    # ← plan_note, confidence, reflection_reason KHÔNG được truyền vào
)
```

`QueryResponse` đã khai báo 3 field này (lines 74-76) nhưng chúng luôn trả về giá trị default (`""`, `"medium"`, `""`). Agent tính toán xong rồi bỏ đi.

**Fix:**
```python
return QueryResponse(
    question=result["question"],
    answer=result["answer"],
    tool_used=result["tool_used"],
    entity=result["entity"],
    sources=result["sources"],
    session_id=request.session_id,
    steps=result.get("steps", []),
    plan_note=result.get("plan_note", ""),
    confidence=result.get("confidence", "medium"),
    reflection_reason=result.get("reflection_reason", ""),
)
```

---

### B2 — `_reflect()` không bao giờ trigger re-synthesis khi LLM trả JSON lỗi

**File:** [src/agent/agent.py:360](src/agent/agent.py)

```python
_default = {"is_sufficient": True, "confidence": "medium", "reason": ""}
# ...
except Exception:
    return _default  # is_sufficient=True → re-synthesis bị bỏ qua
```

Khi `json.loads()` fail (LLM trả markdown, text thường, v.v.), hàm trả `_default` với `is_sufficient=True`. Điều kiện check ở line 696:

```python
if not reflection["is_sufficient"] and result.get("observations"):
    # → KHÔNG BAO GIỜ vào đây khi parse lỗi
```

Hệ quả: re-synthesis chỉ hoạt động khi LLM trả đúng JSON format. Với 8B model, đây là edge case thường xuyên xảy ra.

**Fix:**
```python
_default = {"is_sufficient": False, "confidence": "medium", "reason": "Reflection parse failed"}
```

---

### B3 — Default `session_id="default"` gây session bleed

**File:** [src/api/main.py:55](src/api/main.py)

```python
class QueryRequest(BaseModel):
    session_id: Optional[str] = "default"
```

Mọi client không truyền `session_id` (ví dụ: test bằng curl, Postman) đều dùng chung agent `"default"`. Lịch sử hội thoại bị lẫn giữa các người dùng.

UI Streamlit luôn generate UUID nên không ảnh hưởng trong demo bình thường. Nhưng nếu có nhiều người test API trực tiếp (evaluation script, tester khác), lịch sử sẽ corrupt.

**Fix:**
```python
from uuid import uuid4

class QueryRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
```

---

### B4 — Neo4j exceptions không được bắt → 500 unhandled khi Neo4j down

**File:** [src/agent/neo4j_query.py:51](src/agent/neo4j_query.py) và [line 99](src/agent/neo4j_query.py)

Cả `cypher_search()` và `bfs_search()` gọi `driver.session()` và `session.run()` mà không có try-except. Nếu Neo4j AuraDB mất kết nối hoặc timeout, exception sẽ propagate qua API thành HTTP 500 với stack trace.

**Fix — ví dụ cho `cypher_search`:**
```python
try:
    with driver.session() as session:
        rows = session.run("""...""", name=name_norm)
        row = rows.single()
        if row:
            results["relations"] = [x for x in (row["relations"] or []) if x]
            results["articles"]  = [x for x in (row["articles"]  or []) if x]
except Exception as e:
    logger.error(f"Neo4j cypher_search error: {e}")
return results
```

Áp dụng tương tự cho `bfs_search()`.

---

### B5 — `KeyError` tiềm ẩn trong `summarizer.py`

**File:** [src/kg_build/summarizer.py:93,105](src/kg_build/summarizer.py)

```python
# line 93 — sort dùng x[1]["nodes"] mà không guard
sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]["nodes"]), ...)

# line 105 — tương tự
nodes = comm_data["nodes"]
```

Đoạn chuẩn hoá format (lines 86-89) kiểm tra `isinstance(v, dict)` nhưng không kiểm tra dict có key `"nodes"` không. Nếu `communities.json` bị corrupt hoặc có format lạ, cả 2 dòng này sẽ crash với `KeyError`.

**Fix:**
```python
communities = {
    k: (v if isinstance(v, dict) and "nodes" in v else {"nodes": v if isinstance(v, list) else [], "summary": ""})
    for k, v in communities.items()
}
```

---

### D1 — CUDA seed thiếu → không reproducible trên GPU

**File:** [src/embedding/train_gcn.py:35-37](src/embedding/train_gcn.py)

```python
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
# ← thiếu CUDA seed
```

Nếu máy có GPU và PyTorch dùng CUDA, kết quả embedding sẽ khác mỗi lần train.

**Fix:**
```python
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
```

---

### D2 — History truncation cho phép `MAX_HISTORY_MESSAGES + 2`

**File:** [src/agent/agent.py:722-725](src/agent/agent.py)

```python
self.history.append({"role": "user",      "content": question})   # +1
self.history.append({"role": "assistant",  "content": result["answer"]})  # +2
if len(self.history) > MAX_HISTORY_MESSAGES:  # check SAU khi append
    self.history = self.history[-MAX_HISTORY_MESSAGES:]
```

Khi `len == MAX`, vòng sau append 2 → `len == MAX+2` → truncate xuống MAX. Hiệu quả là history dao động giữa `MAX` và `MAX+2` thay vì ổn định ở `MAX`.

**Fix (thay `>` bằng `>=`):**
```python
if len(self.history) >= MAX_HISTORY_MESSAGES:
    self.history = self.history[-MAX_HISTORY_MESSAGES:]
```

---

### D3 — Race condition trên `sessions` OrderedDict

**File:** [src/api/main.py:85-96](src/api/main.py)

FastAPI chạy trên asyncio; nếu có nhiều requests đồng thời vào `get_or_create_session()`, 2 coroutines có thể đồng thời check `session_id not in sessions` → tạo 2 agent cho cùng 1 session_id. Với `MAX_SESSIONS=500` và deployment thật, đây là risk thật sự.

**Fix:**
```python
import asyncio
_session_lock = asyncio.Lock()

async def get_or_create_session(session_id: str) -> ITHelpdeskAgent:
    async with _session_lock:
        if session_id in sessions:
            sessions.move_to_end(session_id)
            return sessions[session_id]
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            del sessions[oldest]
        sessions[session_id] = ITHelpdeskAgent()
        return sessions[session_id]
```

Và đổi endpoint thành `async def query(request)` gọi `await get_or_create_session(...)`.

---

### D4 — `_RE_WEBSEARCH` match năm quá rộng

**File:** [src/agent/agent.py:40-46](src/agent/agent.py)

Pattern hiện tại có thể bao gồm `2024`, `2025` để detect câu hỏi về Windows update mới. Hệ quả: câu hỏi như *"password expires in 2025"* hay *"error since 2024"* bị force sang WEBSEARCH thay vì query KG.

**Fix:** Thêm context keyword bắt buộc đi kèm với năm:
```python
_RE_WEBSEARCH = re.compile(
    r'\b(latest|newest|recent|update|patch|release|version)\b.*\b(24H2|23H2|2[0-9]{3})\b'
    r'|\b(24H2|23H2)\b',  # Windows build codes thì ok match thẳng
    re.IGNORECASE,
)
```

---

### N1 — Chỉ save weights cuối, không save best epoch

**File:** [src/embedding/train_gcn.py](src/embedding/train_gcn.py)

Nếu model overfit ở epoch cuối, `best_weights` (được tính toán trong vòng lặp) không được save ra disk. Chỉ có `state_dict()` tại thời điểm kết thúc train được lưu.

**Fix:** Thêm `torch.save(best_weights, CHECKPOINT_DIR / "best_weights.pt")` ngay sau khi update `best_val_loss`.

---

### N2 — Health check syntax không nhất quán

`docker-compose.yml:17` dùng CMD array format, `Dockerfile.api` dùng shell form với `|| exit 1`. Không gây lỗi runtime nhưng nên đồng nhất.

---

## Tóm tắt công việc

| Thứ tự | ID | Effort | Ghi chú |
|--------|----|--------|---------|
| 1 | B1 | ~3 dòng | Thêm 3 field vào `QueryResponse(...)` |
| 2 | B2 | ~1 dòng | Đổi `True` → `False` trong `_default` |
| 3 | B3 | ~3 dòng | Dùng `uuid4()` làm default session_id |
| 4 | B4 | ~10 dòng | Wrap Neo4j calls trong try-except |
| 5 | B5 | ~5 dòng | Guard `"nodes"` key trong summarizer |
| 6 | D1 | ~3 dòng | Thêm CUDA seed vào train_gcn.py |
| 7 | D2 | ~1 dòng | Đổi `>` thành `>=` trong history check |
| 8 | D3 | ~10 dòng | Thêm asyncio Lock cho session dict |
| 9 | D4 | ~5 dòng | Narrow regex pattern cho WEBSEARCH |
