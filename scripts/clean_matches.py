"""
scripts/clean_matches.py
Lọc và re-rank matched_article_id dựa trên hybrid_score.

Dùng:
    python scripts/clean_matches.py
    python scripts/clean_matches.py --threshold 0.40
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter

INPUT_FILE  = Path("data/qa_testset_matched.json")
OUTPUT_FILE = Path("data/qa_testset.json")
DEFAULT_THRESHOLD = 0.35

# Stratified sample targets when --balanced is used
BALANCED_TARGETS = {
    "EMBEDDING": 60,
    "CYPHER":    30,
    "BFS":       20,
    "WEBSEARCH": 10,
}


def title_keyword_overlap(question: str, title: str) -> float:
    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "how", "what",
        "why", "when", "where", "can", "could", "would", "should",
        "have", "has", "had", "do", "does", "did", "will", "not",
        "from", "with", "for", "and", "or", "but", "in", "on", "at",
        "to", "of", "my", "your", "our", "their", "its", "i", "we",
    }
    q_words = set(re.findall(r"\w+", question.lower())) - STOP
    t_words = set(re.findall(r"\w+", title.lower()))
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def stratified_sample(data: list[dict], targets: dict[str, int]) -> list[dict]:
    """Sample data with controlled per-tool distribution.

    Takes up to `targets[tool]` items per tool. If a bucket has fewer items
    than its target, all items in that bucket are kept (no oversampling).
    Final list is shuffled so tool types are interleaved.
    """
    import random
    by_tool: dict[str, list[dict]] = {}
    for item in data:
        tool = item.get("expected_tool", "EMBEDDING")
        by_tool.setdefault(tool, []).append(item)

    result = []
    for tool, target in targets.items():
        pool = by_tool.get(tool, [])
        random.shuffle(pool)
        sampled = pool[:target]
        result.extend(sampled)
        available = len(pool)
        print(f"  {tool:<12} target={target}  available={available}  sampled={len(sampled)}")

    # Report any tool types in data that weren't in targets
    for tool, pool in by_tool.items():
        if tool not in targets:
            print(f"  {tool:<12} (not in targets, skipped {len(pool)} items)")

    random.shuffle(result)
    return result


def clean(
    threshold: float = DEFAULT_THRESHOLD,
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    balanced: int | None = None,
) -> None:
    data = json.loads(input_file.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} entries from {input_file}")

    cleared = swapped = kept = 0

    for qa in data:
        # Prefer hybrid_candidates; fall back to bm25_candidates for old files
        candidates = qa.get("hybrid_candidates") or qa.get("bm25_candidates", [])

        if not candidates:
            qa["matched_article_id"] = ""
            qa["article_id"]         = f"qa_{qa['id']}"
            cleared += 1
            continue

        top = candidates[0]
        score_field = "hybrid_score" if "hybrid_score" in top else "score"

        if top[score_field] < threshold:
            qa["matched_article_id"] = ""
            qa["article_id"]         = f"qa_{qa['id']}"
            cleared += 1
            continue

        # Re-rank by title keyword overlap
        question = qa["question"] + " " + qa.get("question_body", "")
        scored = sorted(
            candidates,
            key=lambda c: c.get("hybrid_score", c.get("score", 0)) * 0.7
                          + title_keyword_overlap(question, c["title"]) * 0.3,
            reverse=True,
        )
        best = scored[0]

        if best["article_id"] != top["article_id"]:
            qa["matched_article_id"] = best["article_id"]
            qa["article_id"]         = best["article_id"]
            swapped += 1
        else:
            kept += 1

    total   = len(data)
    n_valid = sum(1 for d in data if d.get("matched_article_id"))
    print(f"\nResults:")
    print(f"  Kept    : {kept}")
    print(f"  Swapped : {swapped}")
    print(f"  Cleared : {cleared}  (score < {threshold})")
    print(f"  Valid   : {n_valid}/{total} ({n_valid/total:.0%})")

    cat_valid = Counter(d["category"] for d in data if d.get("matched_article_id"))
    cat_total = Counter(d["category"] for d in data)
    print("\nPer-category:")
    for cat in sorted(cat_total):
        v, t = cat_valid.get(cat, 0), cat_total[cat]
        print(f"  {cat:<14} {v}/{t} ({v/t:.0%})")

    if balanced is not None:
        print(f"\nStratified sampling (total target ≈ {balanced}):")
        # Scale targets proportionally to `balanced`
        total_target = sum(BALANCED_TARGETS.values())
        scaled = {
            tool: max(1, round(t * balanced / total_target))
            for tool, t in BALANCED_TARGETS.items()
        }
        data = stratified_sample(data, scaled)
        print(f"  Final sample: {len(data)} items")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean matched article IDs using hybrid score")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--input",     type=Path,  default=INPUT_FILE)
    parser.add_argument("--output",    type=Path,  default=OUTPUT_FILE)
    parser.add_argument(
        "--balanced", type=int, default=None,
        help="Stratified sample to N items (e.g. 120). "
             "Default ratio EMBEDDING:CYPHER:BFS:WEBSEARCH = 60:30:20:10.",
    )
    args = parser.parse_args()
    clean(
        threshold=args.threshold,
        input_file=args.input,
        output_file=args.output,
        balanced=args.balanced,
    )
