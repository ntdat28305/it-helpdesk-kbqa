# ── Base image ────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Cài system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer
COPY requirements.txt .

# Cài Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/communities.json ./data/communities.json
COPY data/community_summaries.json ./data/community_summaries.json

# Copy models (embeddings + weights)
COPY models/ ./models/

# Tạo thư mục logs
RUN mkdir -p logs

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default command — chạy FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]