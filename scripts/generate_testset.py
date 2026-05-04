"""
scripts/generate_testset.py
Tự động tạo test set từ raw articles dùng Groq LLM.

Chạy:
    python scripts/generate_testset.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.kg_build.entity_extractor import GroqRotator, get_api_keys

load_dotenv()

RAW_DIR     = Path("data/raw")
OUTPUT_FILE = Path("data/test_set.json")

PROMPT = """You are an IT helpdesk expert. Based on this IT troubleshooting article, generate ONE question-answer pair.

Article title: {title}
Article content: {text}

Requirements:
- Question must sound like a real user complaint, NOT a technical query
- Use everyday language, NOT technical terms from the article
- Do NOT copy phrases directly from the article title or content
- Examples of good questions:
  * "My computer keeps crashing after the latest update, what should I do?"
  * "I can't log into my work account this morning"
  * "The printer stopped working suddenly"
- Answer must be specific, actionable, based on the article
- Answer should be 1-3 sentences maximum

Return JSON only, no explanation:
{{
  "question": "...",
  "answer": "...",
  "article_id": "{article_id}",
  "category": "{category}"
}}"""


def _truncate_at_sentence(text: str, max_chars: int = 1500) -> str:
    """Cắt text ở ranh giới câu."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_boundary = max(truncated.rfind(". "), truncated.rfind(".\n"))
    return truncated[:last_boundary + 1] if last_boundary > max_chars * 0.6 else truncated


def generate_qa(groq: GroqRotator, article: dict) -> dict | None:
    metadata   = article.get("metadata", {})
    title      = metadata.get("title", "")
    text       = _truncate_at_sentence(article.get("plain_text", ""), max_chars=1500)
    article_id = metadata.get("article_id", "")
    category   = metadata.get("category", "")

    if not title or not text:
        return None

    raw = groq.call(PROMPT.format(
        title=title, text=text,
        article_id=article_id, category=category,
    ))
    if not raw:
        return None

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(raw[start:end])
    except Exception:
        return None


def stratified_sample(all_files: list[Path], limit: int) -> list[Path]:
    """Sample đều theo category thực sự."""
    by_cat: dict[str, list[Path]] = defaultdict(list)
    for f in all_files:
        by_cat[f.parent.name].append(f)

    cats = list(by_cat.keys())
    per_cat = max(1, limit // len(cats))
    selected: list[Path] = []

    for cat in cats:
        files = by_cat[cat]
        selected.extend(random.sample(files, min(per_cat, len(files))))

    # Nếu vẫn thiếu (do rounding), bổ sung ngẫu nhiên từ tất cả
    if len(selected) < limit:
        remaining = [f for f in all_files if f not in set(selected)]
        selected.extend(random.sample(remaining, min(limit - len(selected), len(remaining))))

    return selected[:limit]


def run(limit: int = 50, seed: int = 42):
    random.seed(seed)
    keys = get_api_keys()
    groq = GroqRotator(keys)

    all_files = [
        f for f in sorted(RAW_DIR.rglob("*.json"))
        if ".cache" not in str(f)
    ]
    print(f"Total raw files: {len(all_files)}")

    selected = stratified_sample(all_files, limit)
    print(f"Selected {len(selected)} files (stratified by category, seed={seed})")

    # Log phân phối
    dist: dict[str, int] = defaultdict(int)
    for f in selected:
        dist[f.parent.name] += 1
    for cat, n in sorted(dist.items()):
        print(f"  {cat}: {n}")

    test_set: list[dict] = []
    seen_questions: set[str] = set()
    errors = 0

    for i, filepath in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {filepath.parent.name}/{filepath.name}")
        try:
            article = json.loads(filepath.read_text(encoding="utf-8"))
            qa = generate_qa(groq, article)
            if qa:
                q_norm = qa["question"].strip().lower()
                if q_norm in seen_questions:
                    print(f"  SKIP duplicate: {qa['question'][:60]}...")
                    errors += 1
                else:
                    seen_questions.add(q_norm)
                    test_set.append(qa)
                    print(f"  Q: {qa['question'][:60]}...")
            else:
                errors += 1
        except Exception as e:
            print(f"  Error: {e}")
            errors += 1

    OUTPUT_FILE.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone: {len(test_set)} QA pairs -> {OUTPUT_FILE}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    run(args.limit, args.seed)
