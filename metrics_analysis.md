# 📊 Phân tích & Đề xuất Metrics — IT Helpdesk KBQA

---

## 1. Metrics hiện có

Hệ thống đánh giá hiện tại nằm trong `scripts/evaluate.py`, sử dụng 4 metrics:

### 1.1 Hit@K (K=1, K=5)

**Ý nghĩa:** Bài viết đúng (ground truth) có nằm trong top-K kết quả trả về không?

**Công thức:**

```
Hit@K = 1 nếu article đúng ∈ top-K results, ngược lại = 0
Hit@K trung bình = Σ Hit@K / N
```

**Cách tính trong code:**
- Agent trả về `sources` (danh sách URL)
- So sánh với `article_id` trong `test_set.json`
- Nếu URL chứa `article_id` → hit

**Kết quả hiện tại (n=49):**

| | Agent | BM25 |
|---|---|---|
| Hit@1 | 0.122 | 0.347 |
| Hit@5 | 0.224 | 0.469 |

### 1.2 MRR (Mean Reciprocal Rank)

**Ý nghĩa:** Bài viết đúng nằm ở vị trí bao nhiêu? Càng đầu càng tốt.

**Công thức:**

```
RR = 1/rank (rank = vị trí đầu tiên của article đúng trong results)
MRR = Σ RR / N
```

**Ví dụ:**
- Article đúng ở vị trí 1 → RR = 1/1 = 1.0
- Article đúng ở vị trí 3 → RR = 1/3 = 0.33
- Article đúng không có trong results → RR = 0

**Kết quả hiện tại:** Agent MRR = 0.181, BM25 MRR = 0.398

### 1.3 ROUGE-L

**Ý nghĩa:** Câu trả lời của agent overlap bao nhiêu với reference answer (đo bằng Longest Common Subsequence)?

**Công thức:**

```
LCS = Longest Common Subsequence(agent_answer, reference_answer)
Precision = LCS / len(agent_answer)
Recall    = LCS / len(reference_answer)
ROUGE-L   = F1 = 2 × P × R / (P + R)
```

**Kết quả hiện tại:** Agent ROUGE-L = 0.121

---

## 2. Vấn đề với Metrics hiện tại

| Vấn đề | Giải thích | Ảnh hưởng |
|---|---|---|
| **Retrieval metrics đo sai thứ** | Hit@K đo "tìm đúng article gốc" nhưng agent có thể trả lời đúng qua web search hoặc multi-hop mà không cần đúng article | Agent bị đánh giá thấp hơn thực tế |
| **ROUGE-L chỉ đo lexical overlap** | Agent nói "restart your PC" vs reference nói "reboot the computer" → ROUGE-L = 0 dù cùng ý | Không phản ánh chất lượng thực |
| **Reference answer quá ngắn** | Test set sinh bởi LLM, mỗi answer chỉ 1-3 câu → agent trả lời dài hơn bị penalize | ROUGE-L bị kéo xuống giả tạo |
| **Thiếu đo answer correctness** | Không biết câu trả lời có ĐÚNG không, chỉ biết nó GIỐNG reference bao nhiêu | Metric chính yếu nhất |
| **Thiếu đo tool routing** | Không biết agent chọn tool có hợp lý không | Mất insight quan trọng |
| **1 article = 1 ground truth** | Một câu hỏi có thể liên quan nhiều articles | Hit@K bị đánh giá sai |

---

## 3. Metrics đề xuất & Cách triển khai

### 3.1 LLM-as-Judge (Answer Correctness) — Ưu tiên #1

**Mục đích:** Dùng LLM đánh giá câu trả lời của agent có đúng không (1-5 điểm).

**Tại sao cần:** ROUGE-L chỉ đo từ trùng nhau, không đo ý nghĩa. LLM-as-Judge đo semantic correctness.

**Cách triển khai trong `evaluate.py`:**

```python
JUDGE_PROMPT = """You are an IT support expert evaluating answer quality.

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


def llm_judge(question: str, reference: str, agent_answer: str) -> int:
    """Dùng Groq LLM chấm điểm câu trả lời."""
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # dùng model lớn để judge
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                reference=reference,
                agent_answer=agent_answer,
            )
        }],
        temperature=0,
        max_tokens=5,
    )

    try:
        score = int(resp.choices[0].message.content.strip()[0])
        return min(max(score, 1), 5)
    except (ValueError, IndexError):
        return 0  # parse fail
```

**Tích hợp vào evaluate.py:**

```python
# Trong vòng lặp đánh giá từng câu hỏi:
judge_score = llm_judge(
    question=qa["question"],
    reference=qa["answer"],
    agent_answer=result["answer"],
)
judge_scores.append(judge_score)

# Tổng hợp:
avg_judge = sum(judge_scores) / len(judge_scores)
print(f"LLM-as-Judge (avg): {avg_judge:.2f} / 5.0")
```

**Output kỳ vọng:**

```
=== Evaluation Results ===
  Hit@1:          0.122
  Hit@5:          0.224
  MRR:            0.181
  ROUGE-L:        0.121
  LLM-Judge:      3.45 / 5.0    ← MỚI
```

**Chi phí:** 49 câu × 1 LLM call = 49 calls → ~2 phút trên Groq free tier.

---

### 3.2 BERT-Score — Ưu tiên #2

**Mục đích:** Đo semantic similarity giữa agent answer và reference answer bằng BERT embeddings (thay vì n-gram như ROUGE).

**Tại sao cần:** "restart your PC" vs "reboot the computer" → ROUGE-L ≈ 0, BERT-Score ≈ 0.9.

**Cách triển khai:**

```python
# Thêm vào requirements.txt:
# bert-score==0.3.13

from bert_score import score as bert_score_fn

def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    """Tính BERT-Score cho danh sách predictions vs references."""
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",  # model chính xác nhất
        verbose=False,
    )
    return {
        "precision": P.mean().item(),
        "recall":    R.mean().item(),
        "f1":        F1.mean().item(),
    }
```

**Tích hợp vào evaluate.py:**

```python
# Thu thập tất cả predictions và references:
all_predictions = []
all_references  = []

for qa in test_set:
    result = query_api(qa["question"])
    all_predictions.append(result["answer"])
    all_references.append(qa["answer"])

# Tính BERT-Score (batch):
bert_scores = compute_bert_score(all_predictions, all_references)
print(f"BERT-Score F1: {bert_scores['f1']:.3f}")
```

**Output kỳ vọng:**

```
  ROUGE-L:        0.121
  BERT-Score F1:   0.72     ← cao hơn ROUGE-L nhiều (đo semantic)
```

**Lưu ý:** Lần chạy đầu sẽ tải model DeBERTa (~1.5GB). Các lần sau dùng cache.

---

### 3.3 Tool Accuracy — Ưu tiên #3

**Mục đích:** Đo tỷ lệ agent chọn đúng tool cho từng loại câu hỏi.

**Tại sao cần:** Biết tool nào đang hoạt động tốt, tool nào cần cải thiện.

**Cách triển khai:**

**Bước 1:** Thêm field `expected_tool` vào `data/test_set.json`:

```json
{
  "question": "How to fix error 0x80070005?",
  "answer": "...",
  "article_id": "...",
  "category": "DeviceMgmt",
  "expected_tool": "CYPHER"       // ← THÊM MỚI
}
```

Quy tắc gán `expected_tool`:

| Loại câu hỏi | expected_tool |
|---|---|
| Có mã lỗi (0x..., ERROR_XXX, KB...) | CYPHER |
| Triệu chứng mơ hồ ("not working", "keeps crashing") | EMBEDDING |
| Quan hệ giữa 2 thực thể | BFS |
| Liên quan phiên bản/thời gian gần đây | WEBSEARCH |

**Bước 2:** Đo trong evaluate.py:

```python
def compute_tool_accuracy(results: list[dict], test_set: list[dict]) -> dict:
    """So sánh tool agent chọn vs expected_tool."""
    correct = 0
    total   = 0
    confusion = {}  # {(expected, actual): count}

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
        "accuracy": correct / total if total else 0,
        "total": total,
        "correct": correct,
        "confusion": confusion,
    }
```

**Output kỳ vọng:**

```
=== Tool Accuracy ===
  Overall: 28/49 = 57.1%

  Confusion Matrix:
  Expected → Actual     Count
  CYPHER → CYPHER         4    ✅
  CYPHER → EMBEDDING      2    ❌
  EMBEDDING → EMBEDDING  18    ✅
  EMBEDDING → WEBSEARCH   6    ❌
  BFS → BFS               3    ✅
  BFS → EMBEDDING         5    ❌
  WEBSEARCH → WEBSEARCH   3    ✅
  ...
```

---

### 3.4 Latency Metrics (Bổ sung)

**Cách triển khai:**

```python
import time

latencies = []

for qa in test_set:
    start = time.time()
    result = query_api(qa["question"])
    elapsed = time.time() - start
    latencies.append(elapsed)

latencies.sort()
n = len(latencies)
print(f"Latency p50: {latencies[n//2]:.2f}s")
print(f"Latency p95: {latencies[int(n*0.95)]:.2f}s")
print(f"Latency avg: {sum(latencies)/n:.2f}s")
```

---

### 3.5 Steps Count (Bổ sung)

**Cách triển khai:**

```python
step_counts = [len(result.get("steps", [])) for result in all_results]
avg_steps = sum(step_counts) / len(step_counts)
print(f"Avg steps per query: {avg_steps:.1f}")
print(f"Max steps: {max(step_counts)}")
print(f"1-step queries: {step_counts.count(1)} / {len(step_counts)}")
```

---

## 4. Tổng hợp Framework Metrics đề xuất

| Nhóm | Metric hiện có | Metric đề xuất thêm | Ưu tiên |
|---|---|---|---|
| **Retrieval** | Hit@1, Hit@5, MRR | Source Recall | Thấp |
| **Answer Quality** | ROUGE-L | **LLM-as-Judge**, **BERT-Score** | 🔴 Cao |
| **Agent Quality** | (không có) | **Tool Accuracy**, Steps Count | 🟡 Trung bình |
| **Performance** | (không có) | Latency p50/p95 | 🟢 Thấp |

### Thứ tự triển khai khuyến nghị:

```
Bước 1: LLM-as-Judge (~30 LOC, 2 phút chạy)
        → Metric quan trọng nhất, thay thế ROUGE-L làm chính
        
Bước 2: Tool Accuracy (~20 LOC + annotate test_set)
        → Insight về routing quality
        
Bước 3: BERT-Score (~10 LOC + pip install bert-score)
        → Bổ sung cho ROUGE-L, không thay thế
        
Bước 4: Latency + Steps Count (~10 LOC)
        → Performance monitoring
```

---

## 5. Bảng kết quả kỳ vọng sau khi thêm metrics

```
=== IT Helpdesk KBQA — Evaluation Report ===
Test set: 49 questions, 4 categories

--- Retrieval ---
  Hit@1:            0.122
  Hit@5:            0.224
  MRR:              0.181

--- Answer Quality ---
  ROUGE-L:          0.121
  BERT-Score F1:    ~0.72     ← semantic (cao hơn ROUGE-L nhiều)
  LLM-Judge (avg):  ~3.4/5   ← correctness thực tế

--- Agent Quality ---
  Tool Accuracy:    ~57%
  Avg Steps:        ~2.1

--- Performance ---
  Latency p50:      ~3.2s
  Latency p95:      ~8.5s

--- Baseline (BM25) ---
  Hit@1: 0.347 | Hit@5: 0.469 | MRR: 0.398
```
