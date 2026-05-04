# Runbook: Việc cần làm sau khi code đã được sửa

> Tài liệu này liệt kê **các script / bước cần chạy lại** sau khi toàn bộ REVIEW.md đã được sửa.
> Đọc từ trên xuống — các bước phụ thuộc nhau theo thứ tự.

---

## 0. Cài đặt môi trường (chỉ cần làm 1 lần)

```powershell
# Tạo venv và cài deps
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Copy và điền .env
cp .env.example .env
# Sửa .env: GROQ_API_KEY_1, NEO4J_URI, NEO4J_PASSWORD, TAVILY_API_KEY
```

---

## 1. Áp dụng schema Neo4j (PHẢI chạy — 1 lần)

> **Lý do:** `kg_loader.py` đã được sửa để tạo CONSTRAINT và FULLTEXT INDEX.
> Cần chạy 1 lần để áp các constraint lên Neo4j AuraDB đang có sẵn.
> Script sẽ tự bỏ qua nếu không có file JSON trong `data/processed/`.

```powershell
python -m src.kg_build.kg_loader
```

**Kiểm tra:**
```powershell
python -c "
from neo4j import GraphDatabase; import os; from dotenv import load_dotenv
load_dotenv()
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with d.session() as s: print([r['name'] for r in s.run('SHOW INDEXES')])
"
```
Phải thấy `entity_name_ft` và các constraint trong output.

---

## 2. Regenerate communities (NÊN chạy lại)

> **Lý do:** `community.py` đã sửa để dùng **undirected edges** và `trials=10` (ổn định hơn).
> Nếu không chạy lại, `data/communities.json` cũ vẫn dùng được nhưng community assignments có thể kém tối ưu.

```powershell
python -m src.kg_build.community
python -m src.kg_build.summarizer
```

Output: `data/communities.json`, `data/community_summaries.json` (ghi đè file cũ).

---

## 3. Chạy lại API và UI (local)

```powershell
# Terminal 1 — FastAPI
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run src/ui/app.py
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## 4. Smoke test nhanh

```powershell
# Kiểm tra empty entity guard
python -c "
from src.agent.neo4j_query import cypher_search
r = cypher_search('')
assert r['relations'] == [] and r['articles'] == [], 'FAIL: empty entity guard'
print('OK: empty entity guard')
"

# Kiểm tra import agent
python -c "from src.agent.agent import ITHelpdeskAgent; print('OK: agent import')"
```

---

## 5. Rebuild dữ liệu từ đầu (chỉ khi cần khởi tạo lại KG)

> Chỉ cần làm nếu Neo4j database bị xóa hoặc muốn cập nhật data.
> **Thứ tự phải đúng:**

```powershell
# Bước 1: Discover URLs
python scripts/discover_urls.py

# Bước 2: Scrape articles
python -m src.ingestion.scraper

# Bước 3: Extract entities (checkpoint tự động tại data/checkpoint.json)
python -m src.kg_build.entity_extractor

# Bước 4: Load vào Neo4j (apply schema + load data)
python -m src.kg_build.kg_loader

# Bước 5: Community detection
python -m src.kg_build.community
python -m src.kg_build.summarizer
```

> ⚠️ **Checkpoint thay đổi:** checkpoint cũ `data/checkpoint.txt` không còn dùng được.
> Checkpoint mới là `data/checkpoint.json`. Nếu đã chạy entity_extractor trước đây,
> xóa `data/checkpoint.txt` và bắt đầu lại hoặc tạo `data/checkpoint.json` thủ công:
> ```powershell
> # Tạo checkpoint.json từ checkpoint.txt cũ (nếu có)
> python -c "
> import json; from pathlib import Path
> old = Path('data/checkpoint.txt')
> if old.exists():
>     ids = [l.strip() for l in old.read_text().splitlines() if l.strip()]
>     Path('data/checkpoint.json').write_text(json.dumps({i: 'success' for i in ids}, indent=2))
>     print(f'Converted {len(ids)} entries')
> "
> ```

---

## 6. Re-train GCN (chỉ nếu KG thay đổi)

> Chỉ cần nếu đã load lại data mới vào Neo4j.
> Cần cài thêm training deps (hiện đang comment trong requirements.txt):

```powershell
pip install torch>=2.1.0 torch-geometric>=2.4.0 sentence-transformers>=2.7.0 graspologic>=3.4.0
python -m src.embedding.train_gcn
```

Output: `models/embeddings/node_embeddings.npy`, `idx_to_name.json`, `name_to_idx.json`

> ⚠️ Sau khi train xong phải **restart FastAPI** vì embeddings được load vào memory khi khởi động.

---

## 7. Regenerate test set (chỉ nếu muốn đánh giá lại)

> `generate_testset.py` đã sửa để stratified sample, seed cố định, và dedup câu hỏi.

```powershell
# Tạo test set mới với seed cố định (reproducible)
python scripts/generate_testset.py --limit 50 --seed 42

# Chạy evaluation (cần FastAPI đang chạy ở port 8000)
python scripts/evaluate.py
```

Output: `data/test_set.json`, `data/eval_results.json`

---

## 8. Docker Compose (deployment)

> Dockerfile.ui đã được tối ưu: chỉ cài `requirements-ui.txt` (~2 packages) thay vì full requirements.
> Image UI sẽ nhỏ hơn đáng kể.

```powershell
# Build và chạy
docker-compose up --build

# Chỉ rebuild 1 service
docker-compose build api
docker-compose up -d
```

---

## Tổng kết: cần chạy gì sau khi pull code mới

| Bước | Cần chạy? | Lý do |
|------|-----------|-------|
| `pip install -r requirements.txt` | ✅ | Đảm bảo deps mới nhất |
| `python -m src.kg_build.kg_loader` | ✅ | Apply schema Neo4j |
| `python -m src.kg_build.community` + `summarizer` | 🟡 Nên | Edge direction fix |
| Restart FastAPI + UI | ✅ | Load code mới |
| Re-train GCN | ❌ Không cần | Embeddings không thay đổi |
| Regenerate test set | ❌ Không cần | Chỉ khi muốn eval lại |
| `docker-compose up --build` | 🟡 Nếu dùng Docker | Dockerfile.ui thay đổi |
