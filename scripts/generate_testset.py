"""
scripts/generate_testset.py
Tự động tạo test set từ processed articles dùng Groq LLM.

Chạy:
    python scripts/generate_testset.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sympy import limit

load_dotenv()

PROCESSED_DIR = Path("data/raw")
OUTPUT_FILE   = Path("data/test_set.json")

PROMPT = """You are an IT helpdesk expert. Based on this IT troubleshooting article, generate ONE question-answer pair.

Article title: {title}
Article content: {text}

Requirements:
- Question must be something an IT helpdesk user would actually ask
- Answer must be specific and based on the article content
- Answer should be 1-3 sentences maximum

Return JSON only:
{{
  "question": "...",
  "answer": "...",
  "article_id": "{article_id}",
  "category": "{category}"
}}"""


def generate_qa(client: Groq, article: dict) -> dict | None:
    metadata = article.get("metadata", {})
    title      = metadata.get("title", "")
    text       = article.get("plain_text", "")[:1500]
    article_id = metadata.get("article_id", "")
    category   = metadata.get("category", "")

    if not title or not text:
        return None

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    title=title,
                    text=text,
                    article_id=article_id,
                    category=category,
                )
            }],
            temperature=0.3,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        return json.loads(raw[start:end])

    except Exception as e:
        print(f"Error: {e}")
        return None


def run(limit: int = 50):
    client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))

    # Bỏ qua file cache
    all_files = [
        f for f in sorted(PROCESSED_DIR.rglob("*.json"))
        if ".cache" not in str(f)
    ]
    print(f"Total processed files: {len(all_files)}")

    # Sample đều từ các categories
    selected = all_files[:limit]

    test_set = []
    errors   = 0

    for i, filepath in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {filepath.parent.name}/{filepath.name}")
        try:
            article = json.loads(filepath.read_text(encoding="utf-8"))
            qa = generate_qa(client, article)
            if qa:
                test_set.append(qa)
                print(f"  Q: {qa['question'][:60]}...")
            else:
                errors += 1
        except Exception as e:
            import traceback
            print(f"  Error: {e}")
            traceback.print_exc()
            errors += 1

        time.sleep(2)  # rate limit

    # Lưu
    OUTPUT_FILE.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone: {len(test_set)} QA pairs saved to {OUTPUT_FILE}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    run(args.limit)