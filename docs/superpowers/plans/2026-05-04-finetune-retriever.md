# Finetune Bi-encoder Retriever — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finetune `msmarco-MiniLM-L6-cos-v5` làm IT helpdesk retriever, tích hợp vào EMBEDDING search để thay thế fuzzy match bằng semantic nearest-neighbor, nâng Agent Hit@1 từ 6.1% lên vượt BM25 baseline 34.7%.

**Architecture:** Finetune sentence-transformer với MNRL loss + BM25 hard negatives. Model finetuned được dùng 2 lần: (1) encode node names làm input features cho GCN, (2) encode query lúc search thay fuzzy match. Node semantic embeddings (384d) được lưu sẵn để tránh re-encode mỗi query.

**Tech Stack:** `sentence-transformers>=2.7.0`, `rank-bm25`, `neo4j`, `numpy`, `torch`, `torch-geometric`

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `scripts/generate_finetune_data.py` | Tạo mới | Tạo training pairs từ test_set + KG + BM25 hard negatives |
| `scripts/finetune_retriever.py` | Tạo mới | Finetune model, lưu vào `models/retriever/` |
| `src/embedding/train_gcn.py` | Sửa | Swap model path, lưu thêm `node_semantic_embeddings.npy` |
| `src/agent/agent.py` | Sửa | Load retriever, thay fuzzy match bằng semantic search |
| `requirements.txt` | Sửa | Bỏ comment các dòng sentence-transformers, torch |
| `data/finetune_pairs.jsonl` | Tạo mới (gitignored) | Training data |
| `models/retriever/` | Tạo mới | Finetuned model weights |
| `models/embeddings/node_semantic_embeddings.npy` | Tạo mới | Pre-computed 384d node encodings |

---

## Task 1: Cập nhật dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Bỏ comment các dòng GCN/sentence-transformers trong `requirements.txt`**

Sửa phần cuối file từ:
```
# ── GCN Training (not required for serving) ──────────────────
# torch>=2.1.0
# torch-geometric>=2.4.0
# sentence-transformers>=2.7.0
# graspologic>=3.4.0
```
Thành:
```
# ── GCN Training & Retriever Finetuning ──────────────────────
torch>=2.1.0
torch-geometric>=2.4.0
sentence-transformers>=2.7.0
```
(Bỏ `graspologic` — không dùng trong codebase)

- [ ] **Cài dependencies**

```bash
pip install torch>=2.1.0 torch-geometric>=2.4.0 "sentence-transformers>=2.7.0"
```

Expected output: `Successfully installed sentence-transformers-...`

- [ ] **Verify cài đặt thành công**

```bash
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

Expected: `OK`

- [ ] **Commit**

```bash
git add requirements.txt
git commit -m "deps: enable sentence-transformers and torch for retriever finetuning"
```

---

## Task 2: Script tạo training data

**Files:**
- Create: `scripts/generate_finetune_data.py`

- [ ] **Tạo file `scripts/generate_finetune_data.py`**

```python
"""
scripts/generate_finetune_data.py
Tạo training pairs cho finetune bi-encoder retriever.

Nguồn:
  1. data/test_set.json        — 49 (question, article) labeled pairs
  2. Neo4j KG                  — (entity_name, article) pairs qua [:MENTIONS]
  3. BM25 hard negatives       — top BM25 results (trừ correct article)

Chạy:
    python scripts/generate_finetune_data.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TEST_SET_FILE = Path("data/test_set.json")
RAW_DIR       = Path("data/raw")
OUTPUT_FILE   = Path("data/finetune_pairs.jsonl")
MAX_TEXT_CHARS = 512  # cắt article text để tránh quá dài


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def load_articles() -> dict[str, dict]:
    """Trả về {article_id: {text, url}}."""
    articles: dict[str, dict] = {}
    for f in RAW_DIR.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            aid  = data["metadata"]["article_id"]
            articles[aid] = {
                "text": data["plain_text"][:MAX_TEXT_CHARS],
                "url":  data["metadata"]["url"],
            }
        except Exception:
            pass
    return articles


def build_bm25(articles: dict[str, dict]):
    from rank_bm25 import BM25Okapi
    ids    = list(articles.keys())
    corpus = [_tokenize(articles[a]["text"]) for a in ids]
    return BM25Okapi(corpus), ids


def mine_hard_negatives(
    query: str,
    correct_id: str,
    bm25,
    ids: list[str],
    top_n: int = 2,
) -> list[str]:
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:10] if ids[i] != correct_id][:top_n]


def get_kg_pairs() -> list[dict]:
    """Query Neo4j: trả về [{entity, url}]."""
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
    )
    pairs: list[dict] = []
    try:
        with driver.session() as session:
            results = session.run(
                "MATCH (e:Entity)-[:MENTIONS]->(a:Article) "
                "RETURN e.name AS entity, a.url AS url LIMIT 500"
            )
            pairs = [{"entity": r["entity"], "url": r["url"]} for r in results]
    finally:
        driver.close()
    return pairs


def main():
    print("=== generate_finetune_data.py ===")

    test_set = json.loads(TEST_SET_FILE.read_text(encoding="utf-8"))
    articles = load_articles()
    print(f"Articles loaded from data/raw/: {len(articles)}")

    pairs: list[dict] = []
    bm25 = ids = None

    if articles:
        bm25, ids = build_bm25(articles)

    # ── Nguồn 1: test_set.json ──────────────────────────────────
    for qa in test_set:
        aid = qa["article_id"]
        if aid not in articles:
            continue
        pair: dict = {
            "query":      qa["question"],
            "positive":   articles[aid]["text"],
            "article_id": aid,
        }
        if bm25 is not None:
            negs = mine_hard_negatives(qa["question"], aid, bm25, ids)
            if negs:
                pair["negative"] = articles[negs[0]]["text"]
        pairs.append(pair)

    print(f"Pairs from test_set: {len(pairs)}")

    # ── Nguồn 2: KG-based pairs ─────────────────────────────────
    if articles:
        url_to_id = {v["url"]: k for k, v in articles.items()}
        try:
            kg_pairs = get_kg_pairs()
            print(f"KG pairs from Neo4j: {len(kg_pairs)}")
            added = 0
            for kp in kg_pairs:
                aid = url_to_id.get(kp["url"])
                if not aid:
                    continue
                pair = {
                    "query":      kp["entity"],
                    "positive":   articles[aid]["text"],
                    "article_id": aid,
                }
                if bm25 is not None:
                    negs = mine_hard_negatives(kp["entity"], aid, bm25, ids, top_n=1)
                    if negs:
                        pair["negative"] = articles[negs[0]]["text"]
                pairs.append(pair)
                added += 1
            print(f"KG pairs added: {added}")
        except Exception as e:
            print(f"Neo4j unavailable ({e}) — skipping KG pairs")

    # ── Ghi output ───────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with_neg = sum(1 for p in pairs if "negative" in p)
    print(f"\nOutput: {len(pairs)} pairs → {OUTPUT_FILE}")
    print(f"  với hard negative: {with_neg}")
    print(f"  không có negative: {len(pairs) - with_neg}")


if __name__ == "__main__":
    main()
```

- [ ] **Chạy script và verify output**

```bash
python scripts/generate_finetune_data.py
```

Expected output (ví dụ):
```
Articles loaded from data/raw/: 378
Pairs from test_set: 42
KG pairs from Neo4j: 312
KG pairs added: 198
Output: 240 pairs → data/finetune_pairs.jsonl
  với hard negative: 230
  không có negative: 10
```

Nếu `data/raw/` trống:
```
Articles loaded from data/raw/: 0
Pairs from test_set: 0   ← sẽ 0 vì không map được article_id
```
→ Trong trường hợp này cần chạy scraper trước: `python -m src.ingestion.scraper`

- [ ] **Kiểm tra file output**

```bash
python -c "
from pathlib import Path
import json
lines = Path('data/finetune_pairs.jsonl').read_text(encoding='utf-8').strip().split('\n')
print(f'Total pairs: {len(lines)}')
p = json.loads(lines[0])
print('Keys:', list(p.keys()))
print('Query sample:', p['query'][:80])
print('Positive sample:', p['positive'][:80])
print('Has negative:', 'negative' in p)
"
```

Expected: Keys chứa `query`, `positive`, và `negative` (với hầu hết pairs).

- [ ] **Commit**

```bash
git add scripts/generate_finetune_data.py
git commit -m "feat: add script to generate retriever finetuning data"
```

---

## Task 3: Script finetune retriever

**Files:**
- Create: `scripts/finetune_retriever.py`

- [ ] **Tạo file `scripts/finetune_retriever.py`**

```python
"""
scripts/finetune_retriever.py
Finetune msmarco-MiniLM-L6-cos-v5 làm IT helpdesk bi-encoder retriever.

Chạy:
    python scripts/finetune_retriever.py

Output: models/retriever/  (HuggingFace SentenceTransformer format)
"""
from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
)
from torch.utils.data import DataLoader

PAIRS_FILE  = Path("data/finetune_pairs.jsonl")
OUTPUT_DIR  = Path("models/retriever")
BASE_MODEL  = "msmarco-MiniLM-L6-cos-v5"
EPOCHS      = 15
BATCH_SIZE  = 32


def load_pairs() -> list[dict]:
    pairs = []
    with PAIRS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_evaluator() -> evaluation.InformationRetrievalEvaluator | None:
    """Xây evaluator từ test_set.json + data/raw/ articles."""
    test_path = Path("data/test_set.json")
    raw_dir   = Path("data/raw")
    if not test_path.exists() or not raw_dir.exists():
        return None

    test_set = json.loads(test_path.read_text(encoding="utf-8"))

    # Load article texts
    articles: dict[str, str] = {}
    for f in raw_dir.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            aid  = data["metadata"]["article_id"]
            articles[aid] = data["plain_text"][:512]
        except Exception:
            pass

    if not articles:
        return None

    queries:       dict[str, str]       = {}
    corpus:        dict[str, str]       = {}
    relevant_docs: dict[str, set[str]]  = {}

    for i, qa in enumerate(test_set):
        aid = qa["article_id"]
        if aid not in articles:
            continue
        qid = str(i)
        queries[qid]       = qa["question"]
        corpus[aid]        = articles[aid]
        relevant_docs[qid] = {aid}

    if not queries:
        return None

    print(f"Evaluator: {len(queries)} queries, {len(corpus)} docs")
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="it-helpdesk",
        show_progress_bar=False,
    )


def main():
    print(f"=== finetune_retriever.py ===")
    print(f"Base model : {BASE_MODEL}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")

    model = SentenceTransformer(BASE_MODEL)

    pairs = load_pairs()
    print(f"Training pairs: {len(pairs)}")

    train_examples = [
        InputExample(
            texts=(
                [p["query"], p["positive"], p["negative"]]
                if "negative" in p
                else [p["query"], p["positive"]]
            )
        )
        for p in pairs
    ]

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=BATCH_SIZE
    )
    train_loss   = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(train_dataloader) * EPOCHS * 0.1))
    print(f"Warmup steps: {warmup_steps}")

    evaluator = build_evaluator()
    if evaluator:
        print("InformationRetrievalEvaluator sẽ chạy sau mỗi epoch")
    else:
        print("WARNING: Không build được evaluator (thiếu data/raw/) — train không có eval")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),  # eval sau mỗi epoch
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
        save_best_model=True,
    )

    print(f"\nModel saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Chạy finetune**

```bash
python scripts/finetune_retriever.py
```

Training với A100 ước tính ~10-20 phút. Expected output cuối:
```
Epoch: 100%|████| 15/15
Model saved → models/retriever
```

Nếu có evaluator, log sẽ in NDCG@10 sau mỗi epoch — verify con số tăng dần.

- [ ] **Verify model đã lưu**

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('models/retriever')
v = m.encode('Teams meeting keeps dropping', normalize_embeddings=True)
print('Embedding shape:', v.shape)
print('Norm:', round(float((v**2).sum()**0.5), 4))  # phải ~1.0
"
```

Expected:
```
Embedding shape: (384,)
Norm: 1.0
```

- [ ] **Commit**

```bash
git add scripts/finetune_retriever.py models/retriever/
git commit -m "feat: add finetune script and finetuned retriever model"
```

---

## Task 4: Sửa train_gcn.py — swap model + lưu semantic embeddings

**Files:**
- Modify: `src/embedding/train_gcn.py:83-88` (phần load SentenceTransformer)
- Modify: `src/embedding/train_gcn.py:247-264` (hàm `save_results`)

- [ ] **Sửa phần load SentenceTransformer trong `prepare_data()`**

Tìm đoạn (dòng ~83-88):
```python
    # Random features thay vì one-hot — tránh overfitting
    logger.info("Encoding node names bằng sentence-transformers...")
    st_model  = SentenceTransformer("all-MiniLM-L6-v2")
    node_names = [node_mapping[nid] for nid in node_ids]
    embeddings = st_model.encode(node_names, show_progress_bar=True)
    features   = torch.tensor(embeddings, dtype=torch.float)
```

Thay bằng:
```python
    # Random features thay vì one-hot — tránh overfitting
    logger.info("Encoding node names bằng sentence-transformers...")
    _retriever_path = Path("models/retriever")
    _model_name = str(_retriever_path) if _retriever_path.exists() else "all-MiniLM-L6-v2"
    logger.info(f"Dùng model: {_model_name}")
    st_model   = SentenceTransformer(_model_name)
    node_names = [node_mapping[nid] for nid in node_ids]
    embeddings = st_model.encode(node_names, show_progress_bar=True, normalize_embeddings=True)
    features   = torch.tensor(embeddings, dtype=torch.float)
```

- [ ] **Sửa `save_results()` để lưu thêm node_semantic_embeddings.npy**

Tìm hàm `save_results` (dòng ~247):
```python
def save_results(model, embeddings, idx2name, name2idx):
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    emb_np = embeddings.detach().numpy()
    np.save(EMBEDDING_DIR / "node_embeddings.npy", emb_np)
```

Thêm parameter `node_features` và lưu file mới:
```python
def save_results(model, embeddings, idx2name, name2idx, node_features: np.ndarray | None = None):
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    emb_np = embeddings.detach().numpy()
    np.save(EMBEDDING_DIR / "node_embeddings.npy", emb_np)

    if node_features is not None:
        # L2-normalize để dot product = cosine similarity lúc query
        norms = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
        sem_emb = node_features / norms
        np.save(EMBEDDING_DIR / "node_semantic_embeddings.npy", sem_emb)
        logger.info(f"Saved semantic embeddings: {sem_emb.shape}")
```

- [ ] **Sửa hàm `run()` để truyền `node_features` vào `save_results()`**

Tìm hàm `run()`:
```python
def run():
    node_mapping, edge_list = get_graph_data()
    train_data, val_data, idx2name, name2idx = prepare_data(node_mapping, edge_list)
    model, embeddings = train(train_data, val_data)
    save_results(model, embeddings, idx2name, name2idx)
    logger.info("=== Train GCN xong ===")
```

Thay bằng:
```python
def run():
    node_mapping, edge_list = get_graph_data()
    train_data, val_data, idx2name, name2idx = prepare_data(node_mapping, edge_list)
    model, embeddings = train(train_data, val_data)
    # train_data.x là node features 384d từ sentence-transformer (numpy array đã được convert sang tensor)
    node_features = train_data.x.numpy()
    save_results(model, embeddings, idx2name, name2idx, node_features=node_features)
    logger.info("=== Train GCN xong ===")
```

- [ ] **Chạy lại train_gcn để sinh files mới**

```bash
python -m src.embedding.train_gcn
```

Expected output cuối:
```
Saved embeddings: (N, 32)
Saved semantic embeddings: (N, 384)
Saved weights: models/gcn_checkpoint/gcn_weights.pt
=== Train GCN xong ===
```

- [ ] **Verify file mới tồn tại và đúng shape**

```bash
python -c "
import numpy as np
sem = np.load('models/embeddings/node_semantic_embeddings.npy')
gcn = np.load('models/embeddings/node_embeddings.npy')
print('Semantic embeddings shape:', sem.shape)   # phải (N, 384)
print('GCN embeddings shape     :', gcn.shape)   # phải (N, 32)
# Verify L2-normalized
norms = (sem**2).sum(axis=1)**0.5
print('Norm min/max:', round(norms.min(),4), round(norms.max(),4))  # phải ~1.0 / ~1.0
"
```

- [ ] **Commit**

```bash
git add src/embedding/train_gcn.py models/embeddings/node_semantic_embeddings.npy models/embeddings/node_embeddings.npy models/gcn_checkpoint/
git commit -m "feat: swap to finetuned retriever in GCN, save node_semantic_embeddings"
```

---

## Task 5: Sửa agent.py — thay fuzzy match bằng semantic search

**Files:**
- Modify: `src/agent/agent.py:179-204` (hàm `load_resources`)
- Modify: `src/agent/agent.py:232-259` (hàm `embedding_search`)

- [ ] **Sửa `load_resources()` để load retriever + semantic embeddings**

Tìm cuối hàm `load_resources()` (sau khi load communities, trước `return resources`):
```python
    comm_path = Path("data/community_summaries.json")
    if comm_path.exists():
        resources["communities"] = json.loads(
            comm_path.read_text(encoding="utf-8")
        )
        logger.info(f"Loaded {len(resources['communities'])} communities")

    return resources
```

Thêm block load retriever vào giữa (trước `return resources`):
```python
    comm_path = Path("data/community_summaries.json")
    if comm_path.exists():
        resources["communities"] = json.loads(
            comm_path.read_text(encoding="utf-8")
        )
        logger.info(f"Loaded {len(resources['communities'])} communities")

    sem_path      = Path("models/embeddings/node_semantic_embeddings.npy")
    retriever_path = Path("models/retriever")
    if sem_path.exists() and retriever_path.exists():
        from sentence_transformers import SentenceTransformer as _ST
        resources["node_semantic_emb"] = np.load(sem_path)
        resources["retriever"] = _ST(str(retriever_path))
        logger.info(f"Loaded finetuned retriever: {retriever_path}")
    else:
        logger.warning(
            "Finetuned retriever không tìm thấy — fallback sang fuzzy match. "
            "Chạy finetune_retriever.py và train_gcn để kích hoạt semantic search."
        )

    return resources
```

- [ ] **Sửa `embedding_search()` để dùng semantic search khi có retriever**

Tìm block fuzzy match trong `embedding_search()`:
```python
    match_result = fuzz_process.extractOne(query_entity, node_names)
    if not match_result or match_result[1] < EMBEDDING_FUZZY_THRESHOLD:
        logger.warning(f"Không tìm được node gần với: {query_entity}")
        return []

    matched_name = match_result[0]
    matched_idx  = name2idx[matched_name]
    logger.info(f"Fuzzy match: '{query_entity}' → '{matched_name}'")
```

Thay bằng:
```python
    if "retriever" in resources and "node_semantic_emb" in resources:
        query_vec      = resources["retriever"].encode(query_entity, normalize_embeddings=True)
        semantic_scores = resources["node_semantic_emb"] @ query_vec
        matched_idx    = int(np.argmax(semantic_scores))
        matched_name   = idx2name[str(matched_idx)]
        logger.info(f"Semantic match: '{query_entity}' → '{matched_name}' "
                    f"(score={semantic_scores[matched_idx]:.3f})")
    else:
        match_result = fuzz_process.extractOne(query_entity, node_names)
        if not match_result or match_result[1] < EMBEDDING_FUZZY_THRESHOLD:
            logger.warning(f"Không tìm được node gần với: {query_entity}")
            return []
        matched_name = match_result[0]
        matched_idx  = name2idx[matched_name]
        logger.info(f"Fuzzy match: '{query_entity}' → '{matched_name}'")
```

- [ ] **Verify agent load thành công**

```bash
python -c "
from src.agent.agent import ITHelpdeskAgent
agent = ITHelpdeskAgent()
print('retriever loaded:', 'retriever' in agent.resources)
print('semantic_emb loaded:', 'node_semantic_emb' in agent.resources)
"
```

Expected:
```
retriever loaded: True
semantic_emb loaded: True
```

- [ ] **Test quick smoke test với câu hỏi thực**

```bash
python -c "
from src.agent.agent import ITHelpdeskAgent
agent = ITHelpdeskAgent()
r = agent.answer('My Teams meeting keeps dropping')
print('Tool used:', r['tool_used'])
print('Entity:', r['entity'])
print('Answer:', r['answer'][:150])
print('Sources:', r['sources'][:2])
"
```

Expected: Tool used = EMBEDDING, entity không rỗng, answer có nội dung.

- [ ] **Commit**

```bash
git add src/agent/agent.py
git commit -m "feat: replace fuzzy match with semantic search in embedding_search"
```

---

## Task 6: Đánh giá before/after

**Files:**
- Read: `data/eval_results.json` (baseline đã có: Hit@1=0.061)

- [ ] **Khởi động API**

```bash
uvicorn src.api.main:app --reload --port 8000
```

Đợi log: `Application startup complete.`

- [ ] **Chạy evaluation (terminal mới)**

```bash
python scripts/evaluate.py
```

Expected: Hit@1 agent vượt 0.347 (BM25 baseline). Nếu chưa đạt, xem per-tool breakdown — EMBEDDING Hit@1 phải tăng rõ rệt so với 0.067 cũ.

- [ ] **Commit kết quả evaluation**

```bash
git add data/eval_results.json
git commit -m "eval: post-finetune evaluation results"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Data generation (Task 2) ✓, Finetune script (Task 3) ✓, train_gcn swap model + save 384d (Task 4) ✓, agent fuzzy→semantic (Task 5) ✓, evaluation (Task 6) ✓
- [x] **Placeholders:** Không có TBD/TODO/similar-to-task-N
- [x] **Type consistency:** `node_features: np.ndarray` trả về từ `train_data.x.numpy()` → đúng với parameter type trong `save_results()`. `resources["retriever"]` là `SentenceTransformer` → `.encode()` trả về `np.ndarray` → `@` với `node_semantic_emb` hợp lệ.
- [x] **Fallback:** Fuzzy match vẫn còn làm fallback trong `embedding_search()` → không break nếu chạy agent trước khi finetune xong.
- [x] **Thứ tự thực thi:** Task 2 → 3 → 4 → 5 → 6 — mỗi task phụ thuộc output của task trước, không có circular dependency.
