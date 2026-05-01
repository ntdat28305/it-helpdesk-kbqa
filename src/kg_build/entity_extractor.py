"""
src/kg_build/entity_extractor.py
Extract entities và relations từ raw articles dùng Groq LLM.

Chạy:
    python -m src.kg_build.entity_extractor --limit 5 --dry-run
    python -m src.kg_build.entity_extractor
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/entity_extractor.log")

RAW_DIR       = Path("data/raw")
CHECKPOINT    = Path("data/checkpoint.txt")
OUTPUT_DIR    = Path("data/processed")

# ── Groq key rotation ─────────────────────────────────────────

def get_api_keys() -> list[str]:
    keys = []
    for i in range(1, 10):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k:
            keys.append(k)
    if not keys:
        raise ValueError("Không tìm thấy GROQ_API_KEY_1 trong .env")
    logger.info(f"Loaded {len(keys)} API keys")
    return keys


class GroqRotator:
    """Xoay vòng API keys tự động khi hết token."""

    def __init__(self, keys: list[str]):
        self.keys    = keys
        self.index   = 0
        self.client  = Groq(api_key=self.keys[0])
        logger.info(f"Dùng key #{self.index + 1}")

    def next_key(self):
        self.index = (self.index + 1) % len(self.keys)
        self.client = Groq(api_key=self.keys[self.index])
        logger.warning(f"Chuyển sang key #{self.index + 1}")

    def call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    logger.warning(f"Rate limit key #{self.index + 1} — chuyển key")
                    self.next_key()
                    time.sleep(3)
                elif "tokens" in err and "exceeded" in err:
                    logger.warning(f"Token limit key #{self.index + 1} — chuyển key")
                    self.next_key()
                    time.sleep(3)
                else:
                    logger.error(f"Lỗi Groq: {e}")
                    time.sleep(2)
        return ""


# ── Checkpoint ────────────────────────────────────────────────

def load_checkpoint() -> set[str]:
    if not CHECKPOINT.exists():
        return set()
    done = set(CHECKPOINT.read_text(encoding="utf-8").splitlines())
    logger.info(f"Checkpoint: {len(done)} files đã xử lý")
    return done


def save_checkpoint(file_id: str):
    with open(CHECKPOINT, "a", encoding="utf-8") as f:
        f.write(file_id + "\n")


# ── Prompt ────────────────────────────────────────────────────

PROMPT_TEMPLATE = """Extract IT troubleshooting entities and relationships from this article.

Article title: {title}
Content: {text}

Return JSON only, no explanation:
{{
  "entities": [
    {{"name": "entity name", "type": "Error|Product|Fix|Symptom|Concept"}}
  ],
  "relations": [
    {{"source": "entity1", "relation": "CAUSES|FIXES|AFFECTS|REQUIRES|RELATED_TO", "target": "entity2"}}
  ]
}}

Rules:
- Entity names must be concise (max 5 words)
- Only extract IT-relevant entities
- Relations must connect entities from the entities list
- Return valid JSON only"""


def extract(groq: GroqRotator, title: str, text: str) -> dict:
    # Giới hạn text để không vượt token limit
    text = text[:2000]
    prompt = PROMPT_TEMPLATE.format(title=title, text=text)

    raw = groq.call(prompt)
    if not raw:
        return {}

    # Parse JSON từ response
    try:
        # Tìm JSON trong response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        logger.warning("Không parse được JSON từ response")
        return {}


# ── Save processed ────────────────────────────────────────────

def save_processed(file_id: str, metadata: dict, result: dict):
    out = OUTPUT_DIR / file_id
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": metadata,
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
    }
    out.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False):
    keys     = get_api_keys()
    groq     = GroqRotator(keys)
    done     = load_checkpoint()

    # Lấy tất cả raw files
    all_files = sorted(RAW_DIR.rglob("*.json"))
    pending   = [f for f in all_files if f.name not in done]

    if limit:
        pending = pending[:limit]

    total = len(pending)
    logger.info(f"Cần xử lý: {total} files (đã xong: {len(done)})")

    success, errors = 0, 0

    for i, filepath in enumerate(pending, 1):
        file_id = filepath.name
        logger.info(f"[{i}/{total}] {filepath.parent.name}/{file_id}")

        try:
            data     = json.loads(filepath.read_text(encoding="utf-8"))
            title    = data["metadata"]["title"]
            text     = data["plain_text"]
            metadata = data["metadata"]
        except Exception as e:
            logger.error(f"Lỗi đọc file {file_id}: {e}")
            errors += 1
            continue

        result = extract(groq, title, text)

        if not result:
            logger.warning(f"Không extract được: {file_id}")
            errors += 1
            continue

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        if dry_run:
            logger.info(
                f"  [DRY RUN] entities={len(entities)} | "
                f"relations={len(relations)} | "
                f"sample={entities[:2]}"
            )
        else:
            save_processed(file_id, metadata, result)
            save_checkpoint(file_id)
            success += 1
            logger.info(
                f"  OK: {len(entities)} entities, "
                f"{len(relations)} relations"
            )

        time.sleep(2)

    logger.info(f"=== Xong: {success} success, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)