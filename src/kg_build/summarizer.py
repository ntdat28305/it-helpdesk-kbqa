"""
src/kg_build/summarizer.py
Tóm tắt từng community bằng Groq LLM.

Chạy:
    python -m src.kg_build.summarizer --limit 5 --dry-run
    python -m src.kg_build.summarizer
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
logger = get_logger(__name__, log_file="logs/summarizer.log")

COMMUNITIES_FILE  = Path("data/communities.json")
OUTPUT_FILE       = Path("data/community_summaries.json")


# ── Groq client ───────────────────────────────────────────────

def get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY_1")
    if not key:
        raise ValueError("Thiếu GROQ_API_KEY_1 trong .env")
    return Groq(api_key=key)


# ── Summarize một community ───────────────────────────────────

PROMPT = """You are an IT knowledge base assistant.
Below is a list of IT-related entities that belong to the same knowledge cluster.
Write a concise 2-3 sentence summary describing what this cluster is about.
Focus on the main IT topic, common issues, and technologies involved.

Entities:
{entities}

Summary:"""


def summarize_community(
    client: Groq,
    community_id: str,
    nodes: list[str],
) -> str:
    """Tóm tắt một community bằng Groq."""

    # Chỉ dùng tối đa 30 nodes để tránh vượt token limit
    sample = nodes[:30]
    entities_text = "\n".join(f"- {n}" for n in sample)

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": PROMPT.format(entities=entities_text)
            }],
            temperature=0,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Lỗi summarize community {community_id}: {e}")
        return ""


# ── Pipeline ──────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False):
    # Load communities
    communities = json.loads(COMMUNITIES_FILE.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(communities)} communities")

    # Chuẩn hoá: hỗ trợ cả format cũ {id: [nodes]} lẫn mới {id: {"nodes": [...]}}
    communities = {
        k: (v if isinstance(v, dict) and "nodes" in v else {"nodes": v if isinstance(v, list) else [], "summary": ""})
        for k, v in communities.items()
    }

    sorted_comms = sorted(
        communities.items(),
        key=lambda x: len(x[1]["nodes"]),
        reverse=True,
    )

    if limit:
        sorted_comms = sorted_comms[:limit]

    client   = get_groq_client()
    summaries = {}
    errors   = 0

    for i, (comm_id, comm_data) in enumerate(sorted_comms, 1):
        nodes = comm_data["nodes"]
        logger.info(
            f"[{i}/{len(sorted_comms)}] "
            f"Community {comm_id} ({len(nodes)} nodes)"
        )

        if dry_run:
            logger.info(
                f"  [DRY RUN] Sample nodes: {nodes[:5]}"
            )
            summaries[comm_id] = {
                "nodes": nodes,
                "summary": "[DRY RUN]",
                "size": len(nodes),
            }
            continue

        summary = summarize_community(client, comm_id, nodes)

        if not summary:
            errors += 1
            continue

        summaries[comm_id] = {
            "nodes":   nodes,
            "summary": summary,
            "size":    len(nodes),
        }

        logger.info(f"  Summary: {summary[:80]}...")
        time.sleep(2)  # tránh rate limit

    # Lưu
    OUTPUT_FILE.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved: {OUTPUT_FILE}")
    logger.info(f"=== Xong: {len(summaries)} summaries, {errors} errors ===")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run(limit=args.limit, dry_run=args.dry_run)