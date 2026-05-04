# Thiết kế: Finetune Bi-encoder Retriever cho EMBEDDING Search

**Ngày:** 2026-05-04  
**Trạng thái:** Đã duyệt

## Vấn đề

EMBEDDING search hiện xử lý 61% queries (30/49) nhưng chỉ đạt Hit@1 6.7%.
Nguyên nhân gốc: `rapidfuzz` match chuỗi ký tự giữa entity được LLM trích xuất với tên node — không có hiểu biết về ngữ nghĩa.
BM25 baseline hiện đang thắng agent rõ rệt: 34.7% vs 6.1% Hit@1.

## Mục tiêu

Finetune `msmarco-MiniLM-L6-cos-v5` thành retriever chuyên domain IT helpdesk, dùng MNRL + hard negatives từ BM25.
Tích hợp model finetuned vào hai điểm:
1. Encoding node features trong `train_gcn.py` (features 384d ban đầu cho GCN tốt hơn)
2. Encoding query trong `agent.py` (thay fuzzy match bằng semantic nearest-neighbor)

Tiêu chí thành công: Agent Hit@1 vượt BM25 baseline (>34.7%) sau khi tích hợp.

## Kiến trúc

### Trước

```
Entity được LLM trích xuất (chuỗi ký tự)
  → rapidfuzz string match → tên node khớp nhất
  → GCN embedding[matched_idx] (32d) làm query_vec
  → cosine(query_vec, tất cả GCN embeddings 32d)
  → top-k nodes → lấy articles
```

### Sau

```
Entity được LLM trích xuất (chuỗi ký tự)
  → finetuned_model.encode() → query_vec (384d, L2-normalized)
  → cosine(query_vec, node_semantic_embeddings 384d)  ← tính sẵn lúc train GCN
  → matched_idx (node gần nhất về ngữ nghĩa)
  → GCN embedding[matched_idx] (32d) làm query_vec
  → cosine(query_vec, tất cả GCN embeddings 32d)
  → top-k nodes → lấy articles
```

GCN traversal theo cấu trúc đồ thị được giữ nguyên. Chỉ thay đổi bước chọn anchor node (fuzzy → semantic).

## Các thành phần

### 1. `scripts/generate_finetune_data.py` (mới)

Tạo `data/finetune_pairs.jsonl` với các trường: `query`, `positive`, `negative`.

**Nguồn dữ liệu:**
- `data/test_set.json` — 49 cặp `(question, article_id)` → map sang `(question, article_text)` qua `data/raw/`
- KG-based pairs — query Neo4j lấy `(Entity)-[:MENTIONS]->(Article)`, dùng tên entity làm query và nội dung article làm positive; sinh thêm ~200-400 cặp
- Hard negatives — với mỗi query, BM25 top-5 loại trừ article đúng

**Ước tính output:** ~300-600 training pairs.

### 2. `scripts/finetune_retriever.py` (mới)

Finetune model retriever.

- Base model: `msmarco-MiniLM-L6-cos-v5` (384d, đã pretrain trên MS MARCO retrieval)
- Loss: `MultipleNegativesRankingLoss` (sentence-transformers)
- Format training: `InputExample(texts=[query, positive, hard_negative])`
- Evaluator: `InformationRetrievalEvaluator` trên `test_set.json` — báo cáo NDCG@10 mỗi epoch
- Epochs: 15, batch size: 32, warmup: 10% số bước
- Output: `models/retriever/` (định dạng HuggingFace SentenceTransformer)

### 3. `src/embedding/train_gcn.py` (sửa)

Hai thay đổi:
1. Swap model path: load `models/retriever/` nếu tồn tại, fallback về `all-MiniLM-L6-v2`
2. Sau khi encode tên node, lưu thêm `models/embeddings/node_semantic_embeddings.npy` (shape: `[num_nodes, 384]`, L2-normalized) cạnh GCN embeddings

### 4. `src/agent/agent.py` (sửa)

`load_resources()`: load thêm `node_semantic_embeddings.npy` và `SentenceTransformer("models/retriever/")` vào resources dict.

`embedding_search()`: thay block fuzzy match:
```python
# Xóa:
match_result = fuzz_process.extractOne(query_entity, node_names)
if not match_result or match_result[1] < EMBEDDING_FUZZY_THRESHOLD: return []
matched_name = match_result[0]
matched_idx = name2idx[matched_name]

# Thay bằng:
query_vec = resources["retriever"].encode(query_entity, normalize_embeddings=True)
semantic_scores = resources["node_semantic_emb"] @ query_vec
matched_idx = int(np.argmax(semantic_scores))
```

Import `rapidfuzz` và hằng số `EMBEDDING_FUZZY_THRESHOLD` có thể xóa nếu không có chỗ nào khác dùng.

## Files & Artifacts mới

| Đường dẫn | Mô tả |
|---|---|
| `scripts/generate_finetune_data.py` | Tạo training pairs |
| `scripts/finetune_retriever.py` | Finetune model |
| `data/finetune_pairs.jsonl` | Training data (gitignored) |
| `models/retriever/` | Weights model đã finetune (commit vào repo như các model khác) |
| `models/embeddings/node_semantic_embeddings.npy` | Node encodings 384d tính sẵn |

## Thứ tự thực thi

```
1. python scripts/generate_finetune_data.py        # → data/finetune_pairs.jsonl
2. python scripts/finetune_retriever.py             # → models/retriever/
3. python -m src.embedding.train_gcn                # → node_semantic_embeddings.npy + GCN mới
4. uvicorn src.api.main:app --reload --port 8000    # restart API
5. python scripts/evaluate.py                       # so sánh before/after
```

## Rủi ro

- **Ít dữ liệu labeled:** 49 cặp là ít. KG-based pairs bổ sung thêm nhưng chất lượng phụ thuộc vào độ chính xác của KG. Hard negatives giúp mỗi cặp training có giá trị hơn.
- `data/raw/` phải có sẵn trên máy để mine BM25 hard negatives và tạo KG-based pairs. Nếu không có (do bị gitignore), fallback về 49 cặp từ test_set.json — vẫn dùng được nhưng yếu hơn.
- Sau khi sửa `train_gcn.py`, phải chạy lại để sinh `node_semantic_embeddings.npy` trước khi agent có thể dùng semantic search. Chạy agent code mới với GCN weights cũ sẽ lỗi.
