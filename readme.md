# IT Helpdesk KBQA System

Knowledge Base Question Answering system cho IT Helpdesk, sử dụng Microsoft Learn Troubleshooting documentation, Knowledge Graph Neo4j, GCN Embedding và Agentic AI.

## Kiến trúc hệ thống

```
Microsoft Learn Docs
        ↓ scraper
   data/raw/*.json
        ↓ entity extraction (Groq LLM)
  Neo4j Knowledge Graph
        ↓ GCN Embedding
  models/embeddings/
        ↓
  ReAct Agent (4 tools)
  ├── CYPHER      → exact entity search
  ├── EMBEDDING   → semantic similarity
  ├── BFS         → multi-hop reasoning
  └── WEBSEARCH   → web fallback
        ↓
  FastAPI REST API → Streamlit UI
```

## Cài đặt môi trường

### Yêu cầu
- Python 3.11+
- Neo4j AuraDB account (free tier)
- Groq API key (free)
- Tavily API key (free)

### Cài đặt

```bash
git clone https://github.com/ntdat28305/it-helpdesk-kbqa.git
cd it-helpdesk-kbqa

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
```

### Cấu hình

```bash
cp .env.example .env
# Điền các giá trị vào .env:
# GROQ_API_KEY_1=...
# NEO4J_URI=neo4j+s://...
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=...
# TAVILY_API_KEY=...
```

## Data Pipeline

### Bước 1: Discover URLs

```bash
python scripts/discover_urls.py
```

Output: `data/discovered_urls.json` — 378 URLs từ 4 categories.

### Bước 2: Scrape articles

```bash
# Test trước
python -m src.ingestion.scraper --limit 50 --dry-run

# Scrape toàn bộ
python -m src.ingestion.scraper
```

Output: `data/raw/{category}/*.json`

### Bước 3: Extract entities

```bash
# Test trước
python -m src.kg_build.entity_extractor --limit 5 --dry-run

# Chạy toàn bộ
python -m src.kg_build.entity_extractor
```

Output: `data/processed/*.json`

### Bước 4: Load vào Neo4j

```bash
python -m src.kg_build.kg_loader --limit 5
python -m src.kg_build.kg_loader
```

### Bước 5: Community detection + summaries

```bash
python -m src.kg_build.community
python -m src.kg_build.summarizer
```

Output: `data/communities.json`, `data/community_summaries.json`

## Training

### Train GCN Embedding

```bash
python -m src.embedding.train_gcn
```

Output:
- `models/embeddings/node_embeddings.npy` — (3266, 32) node vectors
- `models/embeddings/idx_to_name.json` — index to node name mapping
- `models/embeddings/name_to_idx.json` — node name to index mapping
- `models/gcn_checkpoint/gcn_weights.pt` — model weights

## Chạy inference

### Option 1: Local

```bash
# Terminal 1: FastAPI
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Streamlit UI
streamlit run src/ui/app.py
```

Truy cập:
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

### Option 2: Docker Compose

```bash
docker-compose up --build
```

### API Usage

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How to fix ERROR_INVALID_HANDLE?", "session_id": "user_1"}'

# Health check
curl http://localhost:8000/health

# Reset session
curl -X DELETE http://localhost:8000/session/user_1
```

Response example:
```json
{
  "question": "How to fix ERROR_INVALID_HANDLE?",
  "answer": "To fix ERROR_INVALID_HANDLE...",
  "tool_used": "CYPHER",
  "entity": "ERROR_INVALID_HANDLE",
  "sources": ["https://learn.microsoft.com/..."],
  "session_id": "user_1"
}
```

## Evaluation

```bash
# Tạo test set (50 câu hỏi ngôn ngữ tự nhiên)
python scripts/generate_testset.py --limit 50

# Chạy evaluation (cần FastAPI đang chạy)
python scripts/evaluate.py
```

Kết quả so sánh Agent vs BM25 baseline:

| Metric | BM25  | Agent | Improvement |
|--------|-------|-------|-------------|
| Hit@1  | 0.060 | 0.180 | +200.0%     |
| Hit@5  | 0.260 | 0.260 | +0.0%       |
| MRR    | 0.115 | 0.217 | +87.9%      |

Agent vượt BM25 ở Hit@1 (+200%) và MRR (+87.9%) trên natural language test set.

## Cấu trúc thư mục

```
it-helpdesk-kbqa/
├── src/
│   ├── ingestion/          # scraper.py, cleaner.py
│   ├── kg_build/           # entity_extractor.py, kg_loader.py, community.py, summarizer.py
│   ├── embedding/          # train_gcn.py
│   ├── agent/              # agent.py, neo4j_query.py, prompts.py
│   ├── api/                # main.py (FastAPI)
│   ├── ui/                 # app.py (Streamlit)
│   └── utils/              # logger.py
├── data/
│   ├── raw/                # scraped articles (gitignored)
│   ├── processed/          # extracted entities (gitignored)
│   ├── discovered_urls.json
│   ├── communities.json
│   ├── community_summaries.json
│   └── test_set.json
├── models/
│   ├── embeddings/         # GCN node embeddings
│   └── gcn_checkpoint/     # model weights
├── configs/
│   └── scraper.yaml
├── scripts/
│   ├── discover_urls.py
│   ├── generate_testset.py
│   └── evaluate.py
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Deployment

Hệ thống được đóng gói bằng Docker Compose gồm 2 services:
- `api`: FastAPI model service (port 8000)
- `ui`: Streamlit frontend (port 8501)

Neo4j dùng AuraDB cloud (không cần deploy local).

## Nhóm phát triển

| Vai trò         | Trách nhiệm                              |
|-----------------|------------------------------------------|
| A — Data & Infra | Scraper, data pipeline, Docker, deployment |
| B — KG & Neo4j  | Schema, Neo4j loader, GCN training, Streamlit UI |
| C — Agent & LLM | Entity extraction, ReAct agent, 4 tools  |
| D — Eval & Report | Test set, metrics, report, slides       |