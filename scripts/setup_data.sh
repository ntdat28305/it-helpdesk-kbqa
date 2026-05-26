#!/bin/bash
# scripts/setup_data.sh
# Run this after cloning the repo to regenerate all data from scratch.
# Requires: .env configured with GROQ_API_KEY_1, NEO4J_*, TAVILY_API_KEY

set -e

echo "=== Step 1: Discover Microsoft Learn URLs ==="
python scripts/discover_urls.py

echo "=== Step 2: Scrape articles ==="
python -m src.ingestion.scraper

echo "=== Step 3: Extract entities and relations (LLM) ==="
python -m src.kg_build.entity_extractor

echo "=== Step 4: Load into Neo4j ==="
python -m src.kg_build.kg_loader

echo "=== Step 5a: Leiden community detection ==="
python -m src.kg_build.community

echo "=== Step 5b: LLM community summaries ==="
python -m src.kg_build.summarizer

echo "=== Step 5c: Train Graph Autoencoder ==="
python -m src.embedding.train_gcn

echo ""
echo "Done. Data pipeline complete. System is ready to run."
echo "  Start API : uvicorn src.api.main:app --reload --port 8000"
echo "  Start UI  : streamlit run src/ui/app.py"
