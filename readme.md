<div align="center">

# 🖥️ IT Helpdesk KBQA

### Knowledge Graph–Powered Question Answering for IT Support

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

*An end-to-end agentic AI system that transforms Microsoft Learn troubleshooting articles into a structured Knowledge Graph, learns semantic representations via Graph Convolutional Networks, and answers IT helpdesk questions through a multi-tool ReAct agent.*

---

[**Architecture**](#-architecture) · [**Quick Start**](#-quick-start) · [**Data Pipeline**](#-data-pipeline) · [**Agent Tools**](#-agent--react-loop) · [**Evaluation**](#-evaluation) · [**API Reference**](#-api-reference)

</div>

---

## ✨ Highlights

| Feature | Description |
|---|---|
| 🕸️ **Knowledge Graph** | 3,266 IT entities & 7,000+ relationships stored in Neo4j AuraDB |
| 🧠 **GCN Embeddings** | Graph Autoencoder produces 32-dim structural embeddings for semantic node similarity |
| 🤖 **ReAct Agent** | Groq function-calling drives a Thought → Action → Observation loop across 4 specialized tools |
| 🔍 **Multi-Tool Retrieval** | Cypher exact search · GCN semantic search · BFS path reasoning · Tavily web fallback |
| 🏘️ **Community Intelligence** | Leiden algorithm clusters entities into topic communities, LLM-summarized for contextual grounding |
| 🌐 **Finetuned Retriever** | Optional sentence-transformers retriever trained on domain-specific pairs for improved entity matching |
| 💬 **Multi-Turn Dialogue** | Session-aware conversation with automatic topic-change detection and ambiguity resolution |
| 📊 **Rigorous Evaluation** | Hit@K, MRR, ROUGE-L metrics with per-tool and per-category breakdowns vs BM25 baseline |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  Microsoft Learn Docs ──► discover_urls.py ──► 378 article URLs     │
│          │                                                          │
│          ▼                                                          │
│  scraper.py ──────────► data/raw/{category}/*.json                  │
│          │               (structured articles with symptoms,        │
│          │                causes, resolution steps, error codes)     │
│          ▼                                                          │
│  entity_extractor.py ──► data/processed/*.json                      │
│  (Groq LLaMA 3.1)        (entities + relations via LLM extraction) │
│          │                                                          │
│          ▼                                                          │
│  kg_loader.py ──────────► Neo4j AuraDB                              │
│                           ├── :Entity nodes (3,266)                 │
│                           ├── :Article nodes                        │
│                           ├── CAUSES / FIXES / AFFECTS / ...        │
│                           └── MENTIONS (Article → Entity)           │
│          │                                                          │
│          ├── community.py ──► Leiden clustering ──► communities.json │
│          ├── summarizer.py ──► LLM summaries ──► community_summ.json│
│          └── train_gcn.py ──► GAE training ──► node_embeddings.npy  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE ENGINE                              │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────┐                │
│  │          ReAct Agent (ITHelpdeskAgent)           │                │
│  │  ┌───────────────────────────────────────────┐  │                │
│  │  │  Pre-routing: regex → forced tool hint    │  │                │
│  │  │  Ambiguity check → follow-up detection    │  │                │
│  │  │  Topic change → auto history reset        │  │                │
│  │  └───────────────────────────────────────────┘  │                │
│  │       │                                         │                │
│  │       ▼  Groq Function-Calling Loop (≤4 steps)  │                │
│  │  ┌──────────┬──────────┬──────────┬──────────┐  │                │
│  │  │ 🔍       │ 🧠       │ 🕸️       │ 🌐       │  │                │
│  │  │ CYPHER   │EMBEDDING │  BFS     │WEBSEARCH │  │                │
│  │  │ Exact KG │ GCN sim  │ Path     │ Tavily   │  │                │
│  │  │ search   │ search   │ reason   │ fallback │  │                │
│  │  └──────────┴──────────┴──────────┴──────────┘  │                │
│  │       │                                         │                │
│  │       ▼                                         │                │
│  │  Community context enrichment                   │                │
│  │  Answer synthesis + source attribution          │                │
│  └─────────────────────────────────────────────────┘                │
│       │                                                             │
│       ▼                                                             │
│  FastAPI REST API ──────► Streamlit Chat UI                         │
│  (port 8000)              (port 8501)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Purpose | Where to get |
|---|---|---|
| **Python 3.11+** | Runtime | [python.org](https://python.org) |
| **Neo4j AuraDB** | Graph database (free tier) | [console.neo4j.io](https://console.neo4j.io) |
| **Groq API Key** | LLM inference (free) | [console.groq.com](https://console.groq.com) |
| **Tavily API Key** | Web search fallback (free) | [app.tavily.com](https://app.tavily.com) |

### Installation

```bash
# Clone the repository
git clone https://github.com/ntdat28305/it-helpdesk-kbqa.git
cd it-helpdesk-kbqa

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Groq — entity extraction rotates through all keys; agent uses key 1
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...          # optional, for rate-limit rotation
GROQ_API_KEY_3=gsk_...          # optional

# Neo4j AuraDB
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Tavily web search
TAVILY_API_KEY=tvly-...
```

### Run (Local)

```bash
# Terminal 1 — API server
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Chat UI
streamlit run src/ui/app.py
```

| Service | URL |
|---|---|
| 💬 Chat UI | http://localhost:8501 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |
| ❤️ Health Check | http://localhost:8000/health |

### Run (Docker)

```bash
docker-compose up --build
```

> **Note:** Neo4j AuraDB is a cloud service — no local Neo4j instance needed.  
> Make sure `data/communities.json`, `data/community_summaries.json`, and `models/` exist before running the API. See [Data Pipeline](#-data-pipeline) if building from scratch.

---

## 📦 Data Pipeline

The pipeline transforms raw Microsoft Learn documentation into a queryable Knowledge Graph in 5 stages:

```
 ① Discover    ② Scrape       ③ Extract       ④ Load          ⑤ Enrich
────────────  ────────────  ──────────────  ────────────  ────────────────
  URLs          Articles      Entities        Neo4j KG      Communities
  (378)         (raw JSON)    (LLM-based)     (3,266 E)     + GCN Embed
```

### ① Discover URLs

```bash
python scripts/discover_urls.py
```

Crawls Microsoft Learn Table-of-Contents JSON endpoints to discover troubleshooting article URLs across 4 IT domains.

| Category | Source | Articles |
|---|---|---|
| Network | Windows Client & Server | 100 |
| Teams | Microsoft Teams | 78 |
| Identity | Microsoft Entra | 100 |
| DeviceMgmt | Intune & ConfigMgr | 100 |

**Output:** `data/discovered_urls.json`

### ② Scrape Articles

```bash
# Dry run — preview without saving
python -m src.ingestion.scraper --limit 50 --dry-run

# Full scrape
python -m src.ingestion.scraper
```

Extracts structured content from each article: title, plain text, headings, symptoms, causes, resolution steps, and error codes (e.g., `0x80070005`, `ERROR_INVALID_HANDLE`, `KB5034441`).

**Output:** `data/raw/{category}/*.json`

### ③ Extract Entities & Relations

```bash
# Dry run
python -m src.kg_build.entity_extractor --limit 5 --dry-run

# Full extraction
python -m src.kg_build.entity_extractor
```

Uses **Groq LLaMA 3.1-8B** to extract IT entities (`Error`, `Product`, `Fix`, `Symptom`, `Concept`) and typed relations (`CAUSES`, `FIXES`, `AFFECTS`, `REQUIRES`, `RELATED_TO`) from each article. Supports multi-key rotation to handle rate limits.

**Output:** `data/processed/*.json`

### ④ Load into Neo4j

```bash
# Test with a few files
python -m src.kg_build.kg_loader --limit 5

# Load all
python -m src.kg_build.kg_loader
```

Creates Entity/Article nodes with `UNIQUE` constraints and full-text indexing. Relations are validated against a whitelist (`[A-Z][A-Z0-9_]*`) to prevent Cypher injection.

### ⑤ Community Detection + GCN Training

```bash
# Detect communities (Leiden algorithm)
python -m src.kg_build.community

# Summarize each community (LLM)
python -m src.kg_build.summarizer

# Train Graph Autoencoder
python -m src.embedding.train_gcn
```

| Component | Algorithm | Output |
|---|---|---|
| Community Detection | [Leiden](https://github.com/microsoft/graspologic) with 10 trials | `data/communities.json` |
| Community Summaries | Groq LLM (2–3 sentence description per cluster) | `data/community_summaries.json` |
| GCN Embeddings | 2-layer GAE (input→64→32) with dropout + early stopping | `models/embeddings/node_embeddings.npy` |

The **Graph Autoencoder (GAE)** uses sentence-transformer features as input (not one-hot), trains with negative sampling + BCEWithLogitsLoss, and applies early stopping on a 90/10 train/val split.

---

## 🤖 Agent & ReAct Loop

The core of the system is `ITHelpdeskAgent` — a multi-step reasoning agent that uses **Groq function-calling** to implement the [ReAct](https://arxiv.org/abs/2210.03629) paradigm:

```
User Question
     │
     ├──► Regex pre-routing (error codes → CYPHER, time refs → WEBSEARCH)
     ├──► Ambiguity detection (pronouns, follow-ups)
     ├──► Topic change detection (auto-reset history)
     │
     ▼
  ┌─── ReAct Loop (max 4 steps) ────────────────────────┐
  │  Thought: LLM decides which tool to call             │
  │  Action:  Execute tool, get observation              │
  │  Loop:    If enough info → synthesize answer         │
  │           If not → try different tool / rephrase      │
  └──────────────────────────────────────────────────────┘
     │
     ▼
  Answer + Sources + Reasoning Steps
```

### Tool Suite

| Tool | Trigger | What it does |
|---|---|---|
| 🔍 **CYPHER** | Error codes (`0x...`, `ERROR_XXX`, `KB...`) | Exact entity lookup in Neo4j with Cypher `CONTAINS` matching |
| 🧠 **EMBEDDING** | Vague symptoms ("not working", "keeps crashing") | GCN cosine similarity → top-K similar nodes → fetch related articles |
| 🕸️ **BFS** | Relationship queries ("what causes X when Y") | `allShortestPaths` with max 4 hops between two entities |
| 🌐 **WEBSEARCH** | Recent/versioned issues ("24H2", "latest update") | Tavily API with `include_answer=True` for quick summaries |

### Smart Features

- **Fuzzy matching** via `rapidfuzz` (threshold 75) when the finetuned retriever is unavailable
- **Finetuned retriever** (optional): `sentence-transformers` model trained on domain-specific entity pairs for improved semantic matching
- **Community context enrichment**: Appends relevant community summaries to the agent's response context
- **Session memory**: Maintains up to 40 messages per session with LRU eviction (max 500 concurrent sessions)

---

## 📊 Evaluation

### Test Set Generation

```bash
# Generate 50 natural language questions from raw articles (stratified by category)
python scripts/generate_testset.py --limit 50
```

Questions are designed to mimic real user complaints (e.g., *"My computer keeps crashing after the latest update"*) rather than technical queries.

### Run Evaluation

```bash
# Requires FastAPI to be running
python scripts/evaluate.py
```

### Results (n=49)

<table>
<tr>
<th></th>
<th align="center">Metric</th>
<th align="center">BM25 Baseline</th>
<th align="center">ReAct Agent</th>
</tr>
<tr><td>📍</td><td><b>Hit@1</b></td><td align="center">0.347</td><td align="center">0.122</td></tr>
<tr><td>📍</td><td><b>Hit@5</b></td><td align="center">0.469</td><td align="center">0.224</td></tr>
<tr><td>📈</td><td><b>MRR</b></td><td align="center">0.398</td><td align="center">0.181</td></tr>
<tr><td>📝</td><td><b>ROUGE-L</b></td><td align="center">—</td><td align="center">0.121</td></tr>
</table>

### Per-Tool Performance

| Tool | N | Hit@1 | Hit@5 | MRR | ROUGE-L |
|---|---|---|---|---|---|
| 🔍 CYPHER | 4 | 0.250 | 0.250 | 0.250 | 0.116 |
| 🧠 EMBEDDING | 24 | 0.125 | 0.292 | 0.208 | 0.121 |
| 🕸️ BFS | 8 | 0.125 | 0.250 | 0.182 | 0.114 |
| 🌐 WEBSEARCH | 13 | 0.077 | 0.077 | 0.110 | 0.126 |

### Per-Category Performance

| Category | N | Hit@1 | MRR | ROUGE-L |
|---|---|---|---|---|
| Network | 11 | 0.182 | 0.251 | 0.125 |
| DeviceMgmt | 12 | 0.167 | 0.206 | 0.120 |
| Teams | 12 | 0.083 | 0.206 | 0.112 |
| Identity | 12 | 0.083 | 0.088 | 0.122 |

> **Note:** The agent is optimized for *answer quality* and *multi-turn conversation*, not pure retrieval recall. BM25 operates on the full article corpus directly, while the agent retrieves through a Knowledge Graph with LLM-based entity extraction as an intermediary — a fundamentally different retrieval paradigm.

---

## 📡 API Reference

### `POST /query`

Submit an IT helpdesk question.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How to fix ERROR_INVALID_HANDLE?",
    "session_id": "user_1"
  }'
```

**Response:**
```json
{
  "question": "How to fix ERROR_INVALID_HANDLE?",
  "answer": "To fix ERROR_INVALID_HANDLE, try these steps: ...",
  "tool_used": "CYPHER",
  "entity": "ERROR_INVALID_HANDLE",
  "sources": ["https://learn.microsoft.com/..."],
  "session_id": "user_1",
  "steps": [
    {
      "step": 1,
      "tool": "CYPHER",
      "input": "ERROR_INVALID_HANDLE",
      "observation": "Knowledge Graph Relations: ..."
    }
  ]
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status": "ok", "active_sessions": 3}
```

### `DELETE /session/{session_id}`

Reset a conversation session and clear its history.

```bash
curl -X DELETE http://localhost:8000/session/user_1
```

### `GET /sessions`

List all active sessions.

```bash
curl http://localhost:8000/sessions
# → {"active_sessions": ["user_1", "user_2"], "total": 2}
```

---

## 📁 Project Structure

```
it-helpdesk-kbqa/
│
├── src/
│   ├── ingestion/
│   │   └── scraper.py              # Microsoft Learn article scraper
│   │
│   ├── kg_build/
│   │   ├── entity_extractor.py     # LLM-based entity & relation extraction
│   │   ├── kg_loader.py            # Neo4j graph loader with schema validation
│   │   ├── community.py            # Leiden community detection
│   │   └── summarizer.py           # LLM community summarization
│   │
│   ├── embedding/
│   │   └── train_gcn.py            # Graph Autoencoder (GCN → 32-dim embeddings)
│   │
│   ├── agent/
│   │   ├── agent.py                # ReAct agent with 4-tool function calling
│   │   ├── neo4j_query.py          # Cypher/BFS query functions
│   │   └── prompts.py              # Prompt templates (routing, extraction, etc.)
│   │
│   ├── api/
│   │   └── main.py                 # FastAPI REST endpoints
│   │
│   ├── ui/
│   │   └── app.py                  # Streamlit chat interface
│   │
│   └── utils/
│       └── logger.py               # Structured logging utility
│
├── scripts/
│   ├── discover_urls.py            # Microsoft Learn URL discovery
│   ├── generate_testset.py         # LLM-generated test set (stratified)
│   ├── evaluate.py                 # Hit@K, MRR, ROUGE-L evaluation
│   ├── generate_finetune_data.py   # Finetune data generation
│   └── finetune_retriever.py       # Sentence-transformer finetuning
│
├── data/
│   ├── raw/                        # Scraped articles (gitignored)
│   ├── processed/                  # Extracted entities (gitignored)
│   ├── discovered_urls.json        # 378 article URLs
│   ├── communities.json            # Leiden community assignments
│   ├── community_summaries.json    # LLM summaries per community
│   ├── test_set.json               # 50-question evaluation set
│   └── eval_results.json           # Latest evaluation metrics
│
├── models/
│   ├── embeddings/                 # GCN node embeddings (32-dim)
│   ├── gcn_checkpoint/             # GAE model weights
│   └── retriever/                  # Finetuned sentence-transformer (optional)
│
├── configs/
│   └── scraper.yaml                # Scraper configuration
│
├── docker-compose.yml              # 2-service deployment (API + UI)
├── Dockerfile.api                  # FastAPI container
├── Dockerfile.ui                   # Streamlit container
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment template
```

---

## 🐳 Deployment

The system is containerized via Docker Compose with **2 services**:

| Service | Container | Port | Description |
|---|---|---|---|
| `api` | `kbqa_api` | 8000 | FastAPI + Agent + Neo4j client |
| `ui` | `kbqa_ui` | 8501 | Streamlit chat frontend |

```bash
docker-compose up --build
```

- The `ui` service waits for `api` to be healthy before starting (`depends_on: condition: service_healthy`)
- Neo4j AuraDB is a managed cloud service — no local database container needed
- Logs are mounted to `./logs` for persistence

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq Cloud — LLaMA 3.1 8B Instant |
| **Graph Database** | Neo4j AuraDB (free tier) |
| **Graph ML** | PyTorch Geometric — GCN Autoencoder |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2 / finetuned) |
| **Community Detection** | graspologic — Leiden algorithm |
| **Entity Matching** | rapidfuzz (fuzzy) / finetuned retriever (semantic) |
| **API** | FastAPI 0.110 + Uvicorn |
| **Frontend** | Streamlit with custom CSS dark theme |
| **Web Search** | Tavily API |
| **Evaluation** | rank-bm25, rouge-score |
| **Deployment** | Docker Compose (2 services) |
| **Web Scraping** | BeautifulSoup4 + lxml |

---

## 👥 Team

| Role | Responsibilities |
|---|---|
| **A — Data & Infrastructure** | Web scraper, data pipeline, Docker deployment |
| **B — Knowledge Graph** | Neo4j schema, graph loader, GCN training, Streamlit UI |
| **C — Agent & LLM** | Entity extraction, ReAct agent, 4-tool integration |
| **D — Evaluation & Report** | Test set generation, metrics, academic report & slides |

---

## 📄 License

This project was developed as part of the **NLP for Enterprise** course at **VNU-HCMUS** (University of Science, Ho Chi Minh City).

---

<div align="center">

*Built with ❤️ using Knowledge Graphs, Graph Neural Networks, and Agentic AI*

</div>