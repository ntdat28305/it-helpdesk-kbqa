<div align="center">

# IT Helpdesk KBQA

### Knowledge Graph–Powered Question Answering for IT Support

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

*An end-to-end agentic AI system that transforms Microsoft Learn troubleshooting articles into a structured Knowledge Graph, learns semantic node representations via Graph Convolutional Networks, and answers IT helpdesk questions through a multi-tool ReAct agent with self-reflection.*

---

[**Architecture**](#-architecture) · [**Quick Start**](#-quick-start) · [**Data Pipeline**](#-data-pipeline) · [**Agent**](#-agent--react-loop) · [**Evaluation**](#-evaluation) · [**API Reference**](#-api-reference)

</div>

---

## Highlights

| Feature | Description |
|---|---|
| **Knowledge Graph** | 3,266 IT entities and 7,000+ typed relationships stored in Neo4j AuraDB |
| **GCN Embeddings** | Graph Autoencoder (384 → 64 → 32 dim) trained on structural co-occurrence with L2-normalized outputs |
| **ReAct Agent** | Groq function-calling drives a Thought → Action → Observation loop with up to 4 steps per query |
| **4-Tool Retrieval** | Cypher exact search · GCN semantic search · BFS path reasoning · Tavily web fallback |
| **Dual LLM** | LLaMA 3.3 70B Versatile for reasoning & synthesis; LLaMA 3.1 8B Instant for fast pre-processing |
| **Community Intelligence** | Leiden algorithm clusters entities into topic communities, LLM-summarized for contextual grounding |
| **Finetuned Retriever** | Optional sentence-transformers model trained on domain-specific entity pairs |
| **Multi-Turn Dialogue** | Session-aware conversations with automatic topic-change detection and ambiguity resolution |
| **Self-Reflection** | Agent evaluates its own answer confidence and re-synthesizes if insufficient |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                              │
│                                                                     │
│  Microsoft Learn Docs                                               │
│        │                                                            │
│        ▼                                                            │
│  discover_urls.py ──────────────► data/discovered_urls.json         │
│        │                          (378 URLs across 4 IT domains)    │
│        ▼                                                            │
│  scraper.py ────────────────────► data/raw/{category}/*.json        │
│        │                          (title, symptoms, causes,         │
│        │                           resolutions, error codes)        │
│        ▼                                                            │
│  entity_extractor.py ───────────► data/processed/*.json             │
│  (Groq LLaMA 3.1 8B)              (entities + typed relations)      │
│        │                                                            │
│        ▼                                                            │
│  kg_loader.py ──────────────────► Neo4j AuraDB                      │
│                                   ├── :Entity (3,266 nodes)         │
│                                   ├── :Article nodes                │
│                                   ├── CAUSES / FIXES / AFFECTS /    │
│                                   │   REQUIRES / RELATED_TO         │
│                                   └── MENTIONS (Article → Entity)   │
│        │                                                            │
│        ├── community.py ────────► data/communities.json             │
│        │   (Leiden algorithm)                                       │
│        ├── summarizer.py ───────► data/community_summaries.json     │
│        │   (LLM per cluster)                                        │
│        └── train_gcn.py ────────► models/embeddings/                │
│            (Graph Autoencoder)    node_embeddings.npy (32-dim)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE ENGINE                            │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ├──► Regex pre-routing (error codes → CYPHER, versions → WEB) │
│       ├──► Ambiguity detection (follow-up resolution)               │
│       ├──► Topic-change detection (auto history reset)              │
│       ├──► Lightweight planning note (8B model)                     │
│       │                                                             │
│       ▼  LLaMA 3.3 70B · Function-Calling · ≤4 steps                │
│  ┌──────────┬──────────┬──────────┬──────────┐                      │
│  │ CYPHER   │EMBEDDING │   BFS    │WEBSEARCH │                      │
│  │ Exact KG │ GCN sim  │  Path    │  Tavily  │                      │
│  │ search   │ top-K    │ 4-hop    │ fallback │                      │
│  └──────────┴──────────┴──────────┴──────────┘                      │
│       │                                                             │
│       ├──► Community context enrichment                             │
│       ├──► Answer synthesis (70B model)                             │
│       └──► Self-reflection → re-synthesize if low confidence        │
│                                                                     │
│  FastAPI (port 8000) ──────────► Streamlit UI (port 8501)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

| Requirement | Purpose | Where to get |
|---|---|---|
| **Python 3.11+** | Runtime | [python.org](https://python.org) |
| **Neo4j AuraDB** | Graph database (free tier) | [console.neo4j.io](https://console.neo4j.io) |
| **Groq API Key** | LLM inference (free tier) | [console.groq.com](https://console.groq.com) |
| **Tavily API Key** | Web search fallback (free tier) | [app.tavily.com](https://app.tavily.com) |

### Installation

```bash
git clone https://github.com/ntdat28305/it-helpdesk-kbqa.git
cd it-helpdesk-kbqa

python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Groq — extraction rotates through all keys; agent inference uses key 1
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...   # optional — for rate-limit rotation during pipeline
GROQ_API_KEY_3=gsk_...   # optional

# Neo4j AuraDB
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Tavily web search
TAVILY_API_KEY=tvly-...
```

### Run (Local)

```bash
# Terminal 1 — start the API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — start the UI
streamlit run src/ui/app.py
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

### Run (Docker)

```bash
docker-compose up --build
```

> **Note:** `data/communities.json`, `data/community_summaries.json`, and the `models/` directory must exist before running the API. Neo4j AuraDB is a cloud service — no local database needed. See [Data Pipeline](#-data-pipeline) if you need to build from scratch.

---

## Data Pipeline

The pipeline converts raw Microsoft Learn documentation into a queryable Knowledge Graph in 5 sequential stages:

```
 ① Discover    ② Scrape       ③ Extract       ④ Load         ⑤ Enrich
────────────  ────────────  ──────────────  ───────────  ──────────────────
  378 URLs      Raw JSON      Entities &      Neo4j KG     Communities +
                              Relations       3,266 E       GCN Embeddings
```

### ① Discover URLs

```bash
python scripts/discover_urls.py
```

Crawls Microsoft Learn Table-of-Contents JSON endpoints to discover troubleshooting article URLs.

| Category | Source | Articles |
|---|---|---|
| Network | Windows Client & Server | 100 |
| Teams | Microsoft Teams | 78 |
| Identity | Microsoft Entra | 100 |
| DeviceMgmt | Intune & Configuration Manager | 100 |

**Output:** `data/discovered_urls.json`

### ② Scrape Articles

```bash
python -m src.ingestion.scraper --limit 50 --dry-run   # preview
python -m src.ingestion.scraper                         # full run
```

Extracts structured fields from each article: title, plain text, headings, symptoms, causes, resolution steps, and error codes (`0x80070005`, `ERROR_INVALID_HANDLE`, `KB5034441`, etc.). Rate-limited with configurable delays (1.5–3 s, 3 retries, caching enabled).

**Output:** `data/raw/{category}/*.json`

### ③ Extract Entities & Relations

```bash
python -m src.kg_build.entity_extractor --limit 5 --dry-run   # preview
python -m src.kg_build.entity_extractor                        # full run
```

Uses **Groq LLaMA 3.1 8B Instant** to extract typed entities and relations from each article. Supports multi-key rotation to handle rate limits (`GROQ_API_KEY_1` … `GROQ_API_KEY_N`). Checkpoint-based — safe to interrupt and resume.

| Entity types | Relation types |
|---|---|
| `Error`, `Product`, `Fix`, `Symptom`, `Concept` | `CAUSES`, `FIXES`, `AFFECTS`, `REQUIRES`, `RELATED_TO` |

**Output:** `data/processed/*.json`

### ④ Load into Neo4j

```bash
python -m src.kg_build.kg_loader --limit 5   # test
python -m src.kg_build.kg_loader              # full load
```

Creates `UNIQUE` constraints on `Entity.name` and `Article.article_id`, adds a fulltext index on `Entity.name`, and validates all relation types against a whitelist pattern (`[A-Z][A-Z0-9_]*`) to prevent Cypher injection.

### ⑤ Enrich: Communities + GCN

```bash
# Step 5a — Leiden community detection
python -m src.kg_build.community

# Step 5b — LLM summaries per community
python -m src.kg_build.summarizer

# Step 5c — Train Graph Autoencoder
python -m src.embedding.train_gcn
```

| Component | Algorithm | Output |
|---|---|---|
| Community detection | [Leiden](https://github.com/microsoft/graspologic) (10 trials) | `data/communities.json` |
| Community summaries | Groq LLM (2–3 sentence per cluster) | `data/community_summaries.json` |
| GCN node embeddings | Graph Autoencoder: 384 → 64 → 32 dim | `models/embeddings/node_embeddings.npy` |

The **Graph Autoencoder (GAE)** uses `all-MiniLM-L6-v2` sentence-transformer features (384-dim) as initial node features. Architecture: 2-layer GCNConv (64 hidden, 32 output) with dropout 0.3, trained with BCEWithLogitsLoss + negative sampling on a 90/10 split, early stopping patience 20. Embeddings are L2-normalized for cosine similarity via dot product.

### Optional: Finetune the Retriever

```bash
# Generate domain entity pairs for contrastive training
python scripts/generate_finetune_data.py

# Finetune sentence-transformers on those pairs
python scripts/finetune_retriever.py   # → models/retriever/
```

When `models/retriever/` exists, the agent uses semantic (cosine) matching for entity lookup instead of RapidFuzz fuzzy matching, improving recall on domain-specific terminology.

---

## Agent & ReAct Loop

The core of the system is `ITHelpdeskAgent` in [src/agent/agent.py](src/agent/agent.py). Every query follows a fixed pipeline with **two Groq models** at different cost/speed tradeoffs:

| Stage | Model | Purpose |
|---|---|---|
| Pre-processing | LLaMA 3.1 8B Instant | Ambiguity check, topic-change detection, planning |
| ReAct loop | LLaMA 3.3 70B Versatile | Function-calling, tool selection, reasoning |
| Synthesis & reflection | LLaMA 3.3 70B Versatile | Answer generation, confidence scoring, re-synthesis |

### Query Flow

```
User Question
     │
     ├─ Regex pre-routing  (_forced_tool)
     │     0x... / ERROR_* / KB\d+            → force CYPHER
     │     "what causes" / "how does" / etc.  → force BFS
     │     24H2 / latest / recent             → force WEBSEARCH
     │     everything else                    → force EMBEDDING (default)
     │
     ├─ Ambiguity detection  (8B)
     │     "how do I fix it?" with prior context → resolve pronouns from history
     │
     ├─ Topic-change detection  (8B)
     │     New topic detected → auto-reset conversation history
     │
     ├─ Lightweight planning note  (8B)
     │     1-2 sentence strategy hint passed to the ReAct loop
     │
     ▼
  ┌─── ReAct Loop — max 4 steps ─────────────────────────────────┐
  │  Model: LLaMA 3.3 70B · temperature=0 · function-calling     │
  │                                                              │
  │  Thought: which tool best answers this question?             │
  │  Action:  call tool, get observation                         │
  │  Repeat:  if insufficient → try different tool or rephrase   │
  │  Early exit: stop once answer is clear                       │
  └──────────────────────────────────────────────────────────────┘
     │
     ├─ Community context enrichment
     │     substring-match entity against community summaries → append domain context
     │
     ├─ Answer synthesis  (70B)
     │
     └─ Self-reflection  (70B)
           Evaluate answer quality → confidence: high / medium / low
           If low confidence → re-synthesize with targeted hint
```

### Tool Suite

| Tool | Trigger | Mechanism |
|---|---|---|
| **CYPHER** | Error codes (`0x...`, `ERROR_XXX`, `KB\d+`), exact product names | `CONTAINS toLower(...)` match on `Entity.name` → up to 20 relations + 5 articles |
| **EMBEDDING** | Vague symptoms ("not working", "keeps crashing") | RapidFuzz fuzzy match (threshold 75) or finetuned retriever → GCN cosine top-K → CYPHER on top 3 nodes |
| **BFS** | Relational queries ("what causes X when Y") | `allShortestPaths((a)-[*..4]-(b))` with 3-tier entity fallback (exact → prefix → contains) |
| **WEBSEARCH** | Recent/version-specific issues (`24H2`, `latest update`) | Tavily API · max 3 results · `include_answer=True` for quick summary |

### Smart Behaviors

- **Deduplication**: skips repeated `(tool, args)` pairs within the same query
- **Empty-result fallback**: if any KG tool returns empty results, automatically re-runs as WEBSEARCH
- **Fuzzy entity matching**: RapidFuzz (score ≥ 75) when no finetuned retriever is available
- **Session memory**: up to 40 messages per session (LRU eviction); last 4 turns passed to answer generation; max 500 concurrent sessions

---

## Evaluation

### Test Set

The evaluation uses **97 real user questions** scraped from Microsoft Q&A (`data/qa_testset.json`), stratified across 4 IT categories (Identity=35, Teams=24, DeviceMgmt=20, Network=18).

**Regenerate the test set:**
```bash
python scripts/scrape_qa.py --limit 100       # → data/qa_testset_raw.json
python scripts/match_articles.py              # hybrid BM25 + MiniLM matching
python scripts/clean_matches.py               # → data/qa_testset.json
```

`match_articles.py` uses hybrid scoring (alpha=0.6 BM25 + 0.4 cosine similarity via `all-MiniLM-L6-v2`). Embeddings are cached at `data/.cache/article_embeddings.npz`.

### Run Evaluation

```bash
# Requires FastAPI to be running on port 8000
PYTHONUTF8=1 python scripts/evaluate.py --test-set data/qa_testset.json
```

Metrics: **Hit@1**, **Hit@5**, **MRR**, **ROUGE-L**, **Keyword Accuracy**, **LLM-Judge** (Groq `GROQ_KEY_2`), **Tool Accuracy**, **Latency p50/p95**. Compared against a BM25 baseline.

### Results (n = 97, Microsoft Q&A)

<table>
<tr>
  <th>Metric</th>
  <th align="center">BM25 Baseline</th>
  <th align="center">ReAct Agent</th>
</tr>
<tr><td><b>Hit@1</b></td><td align="center">0.247</td><td align="center">0.031</td></tr>
<tr><td><b>Hit@5</b></td><td align="center">0.536</td><td align="center">0.062</td></tr>
<tr><td><b>MRR</b></td><td align="center">0.348</td><td align="center">0.052</td></tr>
<tr><td><b>ROUGE-L</b></td><td align="center">—</td><td align="center">0.148</td></tr>
<tr><td><b>Keyword Accuracy</b></td><td align="center">—</td><td align="center">0.361</td></tr>
<tr><td><b>LLM-Judge</b></td><td align="center">—</td><td align="center">3.17 / 5.0</td></tr>
<tr><td><b>Answer Relevancy</b></td><td align="center">—</td><td align="center">0.757</td></tr>
<tr><td><b>Pairwise Win Rate</b></td><td align="center">—</td><td align="center"><b>100%</b> (95/95 vs BM25)</td></tr>
<tr><td><b>Tool Accuracy</b></td><td align="center">—</td><td align="center"><b>74.2%</b></td></tr>
<tr><td><b>Latency p50</b></td><td align="center">—</td><td align="center">34.1s</td></tr>
<tr><td><b>Latency p95</b></td><td align="center">—</td><td align="center">50.9s</td></tr>
<tr><td><b>Avg Steps</b></td><td align="center">—</td><td align="center">3.1</td></tr>
</table>

> **Note on Hit@K:** The Q&A test set uses `matched_article_id` assigned by hybrid similarity matching (not human annotation), so Hit@K reflects KG coverage rather than answer correctness. LLM-Judge, Answer Relevancy, and Pairwise Win Rate are the more meaningful quality metrics for this dataset.

### Tool Accuracy Detail

Tool accuracy improved from **60% → 74.2%** after rewriting `SYSTEM_PROMPT` with explicit `DO NOT` constraints and extending `_forced_tool()` with BFS detection and embedding as hard default.

| Routing (expected → actual) | Count | Status |
|---|---|---|
| EMBEDDING → EMBEDDING | 59 | ✓ correct |
| CYPHER → CYPHER | 13 | ✓ correct |
| CYPHER → EMBEDDING | 16 | ✗ misrouted |
| WEBSEARCH → EMBEDDING | 5 | ✗ misrouted |
| BFS → EMBEDDING | 2 | ✗ misrouted |
| CYPHER → WEBSEARCH | 1 | ✗ misrouted |
| EMBEDDING → WEBSEARCH | 1 | ✗ misrouted |

The largest misroute bucket is CYPHER→EMBEDDING (16 cases): questions about specific error codes that the agent routed to embedding search instead of Cypher exact-match.

### Per-Category Performance

| Category | N | Hit@1 | MRR | ROUGE-L | LLM-Judge |
|---|---|---|---|---|---|
| Identity | 35 | 0.029 | 0.046 | 0.156 | 3.00 |
| Teams | 24 | 0.042 | 0.046 | 0.147 | 3.63 |
| Network | 18 | 0.000 | 0.011 | 0.140 | 3.06 |
| DeviceMgmt | 20 | 0.050 | 0.108 | 0.146 | 3.00 |

---

## API Reference

Base URL: `http://localhost:8000`

### `POST /query`

Submit an IT helpdesk question. `session_id` is auto-generated (UUID4) if omitted.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How to fix ERROR_INVALID_HANDLE?", "session_id": "user_1"}'
```

**Response:**

```json
{
  "question": "How to fix ERROR_INVALID_HANDLE?",
  "answer": "To fix ERROR_INVALID_HANDLE, check the handle lifecycle...",
  "tool_used": "CYPHER",
  "entity": "ERROR_INVALID_HANDLE",
  "sources": ["https://learn.microsoft.com/en-us/troubleshoot/..."],
  "session_id": "user_1",
  "steps": [
    {
      "step": 1,
      "tool": "cypher_search",
      "input": "ERROR_INVALID_HANDLE",
      "observation": "Relations: ERROR_INVALID_HANDLE -[CAUSES]-> Application Crash ..."
    }
  ],
  "plan_note": "Error code detected — start with CYPHER search.",
  "confidence": "high",
  "reflection_reason": "Answer directly addresses the error with actionable steps."
}
```

### `GET /health`

```bash
curl http://localhost:8000/health
# → {"status": "ok", "active_sessions": 3}
```

### `GET /sessions`

```bash
curl http://localhost:8000/sessions
# → {"active_sessions": ["user_1", "user_2"], "total": 2}
```

### `DELETE /session/{session_id}`

Reset a conversation session and clear its history.

```bash
curl -X DELETE http://localhost:8000/session/user_1
# → {"message": "Session user_1 đã được xóa"}
```

---

## Project Structure

```
it-helpdesk-kbqa/
│
├── src/
│   ├── ingestion/
│   │   └── scraper.py              # Microsoft Learn article scraper (BS4 + lxml)
│   │
│   ├── kg_build/
│   │   ├── entity_extractor.py     # LLM entity/relation extraction + key rotation
│   │   ├── kg_loader.py            # Neo4j batch loader with schema validation
│   │   ├── community.py            # Leiden community detection (graspologic)
│   │   └── summarizer.py           # LLM summarization per community
│   │
│   ├── embedding/
│   │   └── train_gcn.py            # Graph Autoencoder (384 → 64 → 32 dim)
│   │
│   ├── agent/
│   │   ├── agent.py                # ReAct agent — 4 tools, dual LLM, self-reflection
│   │   ├── neo4j_query.py          # Cypher / BFS / community query functions
│   │   └── prompts.py              # All LLM prompt templates
│   │
│   ├── api/
│   │   └── main.py                 # FastAPI REST API + LRU session management
│   │
│   ├── ui/
│   │   └── app.py                  # Streamlit chat UI (dark theme, custom CSS)
│   │
│   └── utils/
│       └── logger.py               # Structured logging to file + console
│
├── scripts/
│   ├── discover_urls.py            # Microsoft Learn URL discovery
│   ├── scrape_qa.py                # Microsoft Q&A scraper → real user questions
│   ├── match_articles.py           # Hybrid BM25+MiniLM article matching for test set
│   ├── clean_matches.py            # Filter + re-rank matched articles → qa_testset.json
│   ├── generate_testset.py         # LLM-generated stratified test set (legacy)
│   ├── evaluate.py                 # Hit@K, MRR, ROUGE-L, Tool Accuracy, LLM-Judge
│   ├── generate_finetune_data.py   # Domain entity pairs for contrastive training
│   └── finetune_retriever.py       # Sentence-transformer finetuning
│
├── tests/
│   ├── test_agent_routing.py       # Unit tests: _forced_tool routing + SYSTEM_PROMPT content
│   ├── test_match_articles.py      # Unit tests: hybrid scoring functions
│   └── test_clean_matches.py       # Unit tests: clean() threshold filtering
│
├── data/
│   ├── raw/                        # Scraped articles (gitignored)
│   ├── processed/                  # Extracted entities (gitignored)
│   ├── .cache/                     # article_embeddings.npz (MiniLM cache, gitignored)
│   ├── discovered_urls.json        # 378 article URLs
│   ├── communities.json            # Leiden community assignments
│   ├── community_summaries.json    # LLM summaries per community (required at runtime)
│   ├── qa_testset_raw.json         # raw Microsoft Q&A questions (gitignored, regenerate via scrape_qa.py)
│   ├── qa_testset.json             # 97-question evaluation set (with matched_article_id)
│   └── eval_results.json           # Latest evaluation metrics
│
├── models/
│   ├── embeddings/                 # GCN node embeddings (32-dim, L2-normalized)
│   │   ├── node_embeddings.npy
│   │   ├── idx_to_name.json
│   │   ├── name_to_idx.json
│   │   └── node_semantic_embeddings.npy   # optional — finetuned retriever (gitignored, 4.8MB)
│   ├── gcn_checkpoint/             # GAE model weights (best checkpoint)
│   └── retriever/                  # Finetuned sentence-transformer (optional)
│
├── configs/
│   └── scraper.yaml                # Scraper settings (delays, retries, cache)
│
├── docker-compose.yml              # 2-service orchestration (api + ui)
├── Dockerfile.api                  # FastAPI container (python:3.11-slim)
├── Dockerfile.ui                   # Streamlit container
├── requirements.txt                # Full dependency set (~50 packages)
├── requirements-ui.txt             # UI-only minimal dependencies
└── .env.example                    # Environment variable template
```

---

## Deployment

Two Docker services orchestrated by Compose:

| Service | Container | Port | Description |
|---|---|---|---|
| `api` | `kbqa_api` | 8000 | FastAPI + ReAct Agent + Neo4j driver |
| `ui` | `kbqa_ui` | 8501 | Streamlit chat frontend |

```bash
docker-compose up --build
```

- The `ui` service waits for `api` to pass its health check before starting (`depends_on: condition: service_healthy`)
- Neo4j AuraDB is managed cloud — no local database container needed
- `models/embeddings/node_embeddings.npy`, `idx_to_name.json`, `name_to_idx.json` ship with the repo; large optional files (`model.safetensors`, `node_semantic_embeddings.npy`) are gitignored and must be regenerated locally if needed

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM — Reasoning** | Groq Cloud · LLaMA 3.3 70B Versatile (function-calling, synthesis, reflection) |
| **LLM — Utility** | Groq Cloud · LLaMA 3.1 8B Instant (ambiguity check, topic change, planning) |
| **Graph Database** | Neo4j AuraDB (cloud, free tier) |
| **Graph ML** | PyTorch Geometric · Graph Autoencoder (GCNConv 384→64→32) |
| **Embeddings** | sentence-transformers · all-MiniLM-L6-v2 (base) / finetuned domain model |
| **Community Detection** | graspologic · Leiden algorithm |
| **Entity Matching** | rapidfuzz (fuzzy, threshold 75) / finetuned retriever (semantic) |
| **API** | FastAPI 0.110 + Uvicorn |
| **Frontend** | Streamlit 1.35+ with custom CSS dark theme |
| **Web Search** | Tavily API |
| **Web Scraping** | BeautifulSoup4 4.14 + lxml 6.1 |
| **Evaluation** | rank-bm25 · rouge-score |
| **Deployment** | Docker Compose (2 services) |

---

## Team

| Role | Responsibilities |
|---|---|
| **Data & Infrastructure** | Web scraper, data pipeline, Docker deployment |
| **Knowledge Graph** | Neo4j schema, graph loader, GCN training, Streamlit UI |
| **Agent & LLM** | Entity extraction, ReAct agent, 4-tool integration, self-reflection |
| **Evaluation & Report** | Test set generation, metrics, academic report & slides |

---

## License

Developed as part of the **NLP for Enterprise** course at **VNU-HCMUS** (University of Science, Ho Chi Minh City).

---

<div align="center">

*Built with Knowledge Graphs, Graph Neural Networks, and Agentic AI*

</div>
