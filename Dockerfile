FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY data/communities.json ./data/communities.json
COPY data/community_summaries.json ./data/community_summaries.json
COPY models/ ./models/

RUN mkdir -p logs

EXPOSE 8000
EXPOSE 8501