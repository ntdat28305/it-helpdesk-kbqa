# Hybrid Article Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `match_articles.py` dùng Hybrid BM25 + Sentence-transformer và cải thiện `clean_matches.py` để tăng coverage `matched_article_id` từ 58% lên ≥ 75%.

**Architecture:** BM25 lấy lexical candidates, `all-MiniLM-L6-v2` tính cosine similarity, kết hợp bằng weighted sum (`hybrid_score = 0.6×bm25_norm + 0.4×cosine`). Article embeddings được pre-compute và cache tại `data/.cache/article_embeddings.npz`. `clean_matches.py` lọc bằng `hybrid_score` (threshold 0.35) thay vì BM25 score thô.

**Tech Stack:** Python 3.12, `rank-bm25==0.2.2`, `sentence-transformers==5.4.1`, `numpy==1.26.4`, `pytest`

---

## File Map

| File | Thay đổi |
|------|----------|
| `scripts/match_articles.py` | Rewrite toàn bộ — hybrid pipeline |
| `scripts/clean_matches.py` | Update threshold field + default output path |
| `tests/test_match_articles.py` | Tạo mới — unit tests cho hybrid functions |
| `tests/test_clean_matches.py` | Tạo mới — unit tests cho clean logic |
| `requirements.txt` | Thêm `sentence-transformers`, `rank-bm25` |
| `data/.cache/` | Generated at runtime, không commit |

---

## Task 1: Unit tests cho hybrid scoring functions

**Files:**
- Create: `tests/test_match_articles.py`

- [ ] **Step 1: Viết failing tests**

Tạo file `tests/test_match_articles.py`:

```python
"""Unit tests for hybrid scoring functions in match_articles.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ── tokenize ──────────────────────────────────────────────────

def test_tokenize_basic():
    from scripts.match_articles import tokenize
    result = tokenize("Windows Error 0x80004005 fix")
    assert "windows" in result
    assert "error" in result
    assert "fix" in result

def test_tokenize_filters_short():
    from scripts.match_articles import tokenize
    result = tokenize("a to in the windows")
    assert "a" not in result
    assert "to" not in result
    assert "in" not in result
    assert "the" not in result
    assert "windows" in result


# ── bm25_normalize ────────────────────────────────────────────

def test_bm25_normalize_max_is_one():
    from scripts.match_articles import bm25_normalize
    scores = np.array([10.0, 5.0, 0.0, 20.0])
    norm = bm25_normalize(scores)
    assert norm.max() == pytest.approx(1.0)
    assert norm.min() == pytest.approx(0.0)
    assert norm[3] == pytest.approx(1.0)

def test_bm25_normalize_all_zeros():
    from scripts.match_articles import bm25_normalize
    scores = np.array([0.0, 0.0, 0.0])
    norm = bm25_normalize(scores)
    # All zeros stays all zeros, no division by zero
    assert all(norm == pytest.approx(0.0))


# ── hybrid_combine ────────────────────────────────────────────

def test_hybrid_combine_alpha_one_is_bm25():
    from scripts.match_articles import hybrid_combine
    bm25_norm = np.array([0.8, 0.5, 0.2])
    cosine    = np.array([0.1, 0.9, 0.3])
    result = hybrid_combine(bm25_norm, cosine, alpha=1.0)
    np.testing.assert_allclose(result, bm25_norm)

def test_hybrid_combine_alpha_zero_is_cosine():
    from scripts.match_articles import hybrid_combine
    bm25_norm = np.array([0.8, 0.5, 0.2])
    cosine    = np.array([0.1, 0.9, 0.3])
    result = hybrid_combine(bm25_norm, cosine, alpha=0.0)
    np.testing.assert_allclose(result, cosine)

def test_hybrid_combine_weighted():
    from scripts.match_articles import hybrid_combine
    bm25_norm = np.array([1.0, 0.0])
    cosine    = np.array([0.0, 1.0])
    result = hybrid_combine(bm25_norm, cosine, alpha=0.6)
    assert result[0] == pytest.approx(0.6)
    assert result[1] == pytest.approx(0.4)


# ── top_k_candidates ──────────────────────────────────────────

def test_top_k_candidates_returns_k():
    from scripts.match_articles import top_k_candidates
    articles = [
        {"article_id": f"art-{i}", "title": f"Title {i}", "category": "Network"}
        for i in range(5)
    ]
    bm25_scores   = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    cosine_scores = np.array([0.9, 0.1, 0.5, 0.8, 0.3])
    hybrid_scores = np.array([0.9, 0.5, 0.6, 0.7, 0.4])

    result = top_k_candidates(articles, bm25_scores, cosine_scores, hybrid_scores, top_k=3)
    assert len(result) == 3
    # First should be index 0 (hybrid=0.9)
    assert result[0]["article_id"] == "art-0"
    assert "bm25_score"   in result[0]
    assert "cosine_score" in result[0]
    assert "hybrid_score" in result[0]

def test_top_k_candidates_sorted_by_hybrid():
    from scripts.match_articles import top_k_candidates
    articles = [
        {"article_id": f"art-{i}", "title": f"T{i}", "category": "Teams"}
        for i in range(3)
    ]
    bm25_scores   = np.array([1.0, 2.0, 3.0])
    cosine_scores = np.array([0.9, 0.5, 0.1])
    hybrid_scores = np.array([0.95, 0.6, 0.2])

    result = top_k_candidates(articles, bm25_scores, cosine_scores, hybrid_scores, top_k=3)
    assert result[0]["hybrid_score"] >= result[1]["hybrid_score"]
    assert result[1]["hybrid_score"] >= result[2]["hybrid_score"]
```

- [ ] **Step 2: Run tests — pastikan semua FAIL**

```
pytest tests/test_match_articles.py -v
```

Expected output: `ImportError` hoặc `ModuleNotFoundError` vì `match_articles` chưa có các functions mới.

---

## Task 2: Implement hybrid scoring functions

**Files:**
- Modify: `scripts/match_articles.py`

- [ ] **Step 3: Rewrite match_articles.py với pure functions**

Thay thế toàn bộ nội dung `scripts/match_articles.py`:

```python
"""
scripts/match_articles.py
Hybrid BM25 + Sentence-transformer matching để fill matched_article_id.

Dùng:
    python scripts/match_articles.py
    python scripts/match_articles.py --alpha 0.5 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

INPUT_FILE  = Path("data/qa_testset.json")
RAW_DIR     = Path("data/raw")
OUTPUT_FILE = Path("data/qa_testset_matched.json")
CACHE_FILE  = Path("data/.cache/article_embeddings.npz")
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_ALPHA = 0.6
DEFAULT_TOP_K = 3


# ── Pure scoring functions (unit-testable) ─────────────────────

def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]


def bm25_normalize(scores: np.ndarray) -> np.ndarray:
    max_score = scores.max()
    return scores / (max_score + 1e-9)


def hybrid_combine(bm25_norm: np.ndarray, cosine: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * bm25_norm + (1.0 - alpha) * cosine


def top_k_candidates(
    articles: list[dict],
    bm25_scores: np.ndarray,
    cosine_scores: np.ndarray,
    hybrid_scores: np.ndarray,
    top_k: int,
) -> list[dict]:
    top_ids = np.argsort(hybrid_scores)[::-1][:top_k]
    return [
        {
            "article_id":   articles[i]["article_id"],
            "title":        articles[i]["title"],
            "category":     articles[i]["category"],
            "bm25_score":   round(float(bm25_scores[i]),   3),
            "cosine_score": round(float(cosine_scores[i]), 4),
            "hybrid_score": round(float(hybrid_scores[i]), 4),
        }
        for i in top_ids
    ]


# ── Corpus loading ─────────────────────────────────────────────

def load_articles() -> list[dict]:
    articles = []
    for f in RAW_DIR.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data["metadata"]
            articles.append({
                "article_id": meta["article_id"],
                "title":      meta["title"],
                "category":   meta["category"],
                "text":       data["plain_text"],
                "url":        meta["url"],
            })
        except Exception:
            pass
    print(f"Loaded {len(articles)} articles from corpus")
    return articles


# ── BM25 ──────────────────────────────────────────────────────

def build_bm25(articles: list[dict]):
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(a["title"] + " " + a["text"]) for a in articles]
    return BM25Okapi(corpus)


# ── Embedding cache ────────────────────────────────────────────

def build_or_load_embeddings(
    articles: list[dict],
    cache_file: Path,
    model_name: str,
) -> np.ndarray:
    current_ids = [a["article_id"] for a in articles]
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=False)
        if data["article_ids"].tolist() == current_ids:
            print(f"Loaded embeddings from cache ({len(articles)} articles)")
            return data["embeddings"]
        print("Cache stale (article list changed) — rebuilding...")

    print(f"Building embeddings for {len(articles)} articles (model: {model_name})...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [a["title"] + " " + a["text"][:512] for a in articles]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        embeddings=embeddings,
        article_ids=np.array(current_ids),
    )
    print(f"Saved cache → {cache_file}")
    return embeddings


# ── Keyword extraction ─────────────────────────────────────────

def extract_better_keywords(answer: str) -> list[str]:
    kws = []
    kws += re.findall(r"0x[0-9A-Fa-f]{4,8}", answer)
    kws += re.findall(r"\bKB\d{6,7}\b", answer)
    kws += re.findall(r"\bERROR_[A-Z_]+\b", answer)
    kws += re.findall(r"\b[0-9]{8,10}\b", answer)
    kws += re.findall(r"\b[A-Z][a-z]+-[A-Z][a-zA-Z]+\b", answer)
    NOISE = {"The", "This", "Your", "Our", "Use", "Try", "Check",
             "If", "In", "On", "For", "To", "And", "Or", "But",
             "When", "Then", "Also", "Note", "See", "Make"}
    for w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", answer):
        if w not in NOISE and len(w) > 4:
            kws.append(w)
    seen, result = set(), []
    for kw in kws:
        kl = kw.lower()
        if kl not in seen:
            seen.add(kl)
            result.append(kw)
        if len(result) >= 8:
            break
    return result


# ── Main pipeline ──────────────────────────────────────────────

def run(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
    embed_model: str = EMBED_MODEL,
) -> None:
    print("=" * 60)
    print("Hybrid BM25 + Embedding: Match QA → KG articles")
    print("=" * 60)

    test_set = json.loads(input_file.read_text(encoding="utf-8"))
    articles = load_articles()
    if not articles:
        print("ERROR: No articles found in data/raw/ — run scraper first")
        return

    print("Building BM25 index...")
    bm25 = build_bm25(articles)

    use_embeddings = False
    article_embeddings = None
    embed_model_obj = None
    try:
        article_embeddings = build_or_load_embeddings(articles, CACHE_FILE, embed_model)
        from sentence_transformers import SentenceTransformer
        embed_model_obj = SentenceTransformer(embed_model)
        use_embeddings = True
    except ImportError:
        print("WARNING: sentence-transformers not installed — falling back to BM25-only")

    matched_count = 0

    for i, qa in enumerate(test_set):
        question = qa["question"]
        body     = qa.get("question_body", "")
        answer   = qa.get("answer", "")
        query_text = question + " " + body

        # BM25 scores for all articles
        tokens      = tokenize(query_text)
        bm25_raw    = np.array(bm25.get_scores(tokens))
        bm25_norm   = bm25_normalize(bm25_raw)

        # Cosine scores
        if use_embeddings:
            query_vec     = embed_model_obj.encode(query_text, normalize_embeddings=True)
            cosine_scores = article_embeddings @ query_vec
        else:
            cosine_scores = np.zeros(len(articles))

        hybrid_scores = hybrid_combine(bm25_norm, cosine_scores, alpha=alpha)
        candidates    = top_k_candidates(articles, bm25_raw, cosine_scores, hybrid_scores, top_k=top_k)

        # Fill matched_article_id only if not already set
        if not qa.get("matched_article_id") and candidates:
            best = candidates[0]
            qa["matched_article_id"] = best["article_id"]
            qa["article_id"]         = best["article_id"]
            matched_count += 1

        # New field: hybrid_candidates
        qa["hybrid_candidates"] = candidates

        # Backward-compat: keep bm25_candidates for evaluate.py
        qa["bm25_candidates"] = [
            {
                "article_id": c["article_id"],
                "title":      c["title"],
                "score":      c["bm25_score"],
                "category":   c["category"],
            }
            for c in candidates
        ]

        if answer:
            qa["answer_keywords"] = extract_better_keywords(answer)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_set)}] processed...")

    print(f"\n{'='*60}")
    print(f"Matched: {matched_count}/{len(test_set)}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved → {output_file}")
    print("""
Next: python scripts/clean_matches.py
      (outputs data/qa_testset.json ready for evaluate.py)
""")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid BM25+Embedding article matching")
    parser.add_argument("--input",   type=Path,  default=INPUT_FILE)
    parser.add_argument("--output",  type=Path,  default=OUTPUT_FILE)
    parser.add_argument("--top-k",   type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--alpha",   type=float, default=DEFAULT_ALPHA,
                        help="BM25 weight (0-1). Default 0.6.")
    parser.add_argument("--model",   type=str,   default=EMBED_MODEL)
    args = parser.parse_args()
    run(
        input_file=args.input,
        output_file=args.output,
        top_k=args.top_k,
        alpha=args.alpha,
        embed_model=args.model,
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```
pytest tests/test_match_articles.py -v
```

Expected output:
```
tests/test_match_articles.py::test_tokenize_basic PASSED
tests/test_match_articles.py::test_tokenize_filters_short PASSED
tests/test_match_articles.py::test_bm25_normalize_max_is_one PASSED
tests/test_match_articles.py::test_bm25_normalize_all_zeros PASSED
tests/test_match_articles.py::test_hybrid_combine_alpha_one_is_bm25 PASSED
tests/test_match_articles.py::test_hybrid_combine_alpha_zero_is_cosine PASSED
tests/test_match_articles.py::test_hybrid_combine_weighted PASSED
tests/test_match_articles.py::test_top_k_candidates_returns_k PASSED
tests/test_match_articles.py::test_top_k_candidates_sorted_by_hybrid PASSED
9 passed
```

- [ ] **Step 5: Commit**

```bash
git add scripts/match_articles.py tests/test_match_articles.py
git commit -m "feat: rewrite match_articles with hybrid BM25+embedding scoring"
```

---

## Task 3: Unit tests và update clean_matches.py

**Files:**
- Create: `tests/test_clean_matches.py`
- Modify: `scripts/clean_matches.py`

- [ ] **Step 6: Viết failing tests cho clean_matches**

Tạo file `tests/test_clean_matches.py`:

```python
"""Unit tests for clean_matches.py."""
import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_entry(id_, hybrid_score, bm25_score, matched="art-1"):
    return {
        "id": id_,
        "question": f"How to fix issue {id_}",
        "question_body": "",
        "answer": "Some answer",
        "article_id": matched,
        "category": "Network",
        "source": "microsoft_qa",
        "answer_keywords": [],
        "ground_truth_answer": "",
        "matched_article_id": matched,
        "hybrid_candidates": [
            {
                "article_id":   matched,
                "title":        f"Troubleshoot {id_}",
                "category":     "Network",
                "bm25_score":   bm25_score,
                "cosine_score": 0.5,
                "hybrid_score": hybrid_score,
            }
        ],
        "bm25_candidates": [
            {"article_id": matched, "title": f"Troubleshoot {id_}",
             "score": bm25_score, "category": "Network"}
        ],
    }


def test_clean_clears_low_hybrid_score():
    from scripts.clean_matches import clean
    data = [
        _make_entry("1", hybrid_score=0.20, bm25_score=30.0),  # below threshold
        _make_entry("2", hybrid_score=0.50, bm25_score=60.0),  # above threshold
    ]
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "matched.json"
        out = Path(tmp) / "clean.json"
        inp.write_text(json.dumps(data), encoding="utf-8")
        clean(threshold=0.35, input_file=inp, output_file=out)
        result = json.loads(out.read_text(encoding="utf-8"))

    assert result[0]["matched_article_id"] == ""   # cleared
    assert result[1]["matched_article_id"] == "art-1"  # kept


def test_clean_no_candidates_clears():
    from scripts.clean_matches import clean
    entry = _make_entry("3", 0.0, 0.0)
    entry["hybrid_candidates"] = []
    entry["bm25_candidates"]   = []
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "matched.json"
        out = Path(tmp) / "clean.json"
        inp.write_text(json.dumps([entry]), encoding="utf-8")
        clean(threshold=0.35, input_file=inp, output_file=out)
        result = json.loads(out.read_text(encoding="utf-8"))

    assert result[0]["matched_article_id"] == ""


def test_clean_output_has_all_entries():
    from scripts.clean_matches import clean
    data = [_make_entry(str(i), 0.6, 80.0) for i in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "matched.json"
        out = Path(tmp) / "clean.json"
        inp.write_text(json.dumps(data), encoding="utf-8")
        clean(threshold=0.35, input_file=inp, output_file=out)
        result = json.loads(out.read_text(encoding="utf-8"))

    assert len(result) == 5
```

- [ ] **Step 7: Run tests — verify they FAIL**

```
pytest tests/test_clean_matches.py -v
```

Expected: FAIL — `clean()` chưa nhận `input_file`/`output_file` params và dùng `bm25_score` thay `hybrid_score`.

- [ ] **Step 8: Rewrite clean_matches.py**

Thay thế toàn bộ nội dung `scripts/clean_matches.py`:

```python
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


def clean(
    threshold: float = DEFAULT_THRESHOLD,
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
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
                          + title_keyword_overlap(question, c["title"]) * 100 * 0.3,
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved → {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean matched article IDs using hybrid score")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--input",     type=Path,  default=INPUT_FILE)
    parser.add_argument("--output",    type=Path,  default=OUTPUT_FILE)
    args = parser.parse_args()
    clean(threshold=args.threshold, input_file=args.input, output_file=args.output)
```

- [ ] **Step 9: Run tests — verify they pass**

```
pytest tests/test_clean_matches.py -v
```

Expected output:
```
tests/test_clean_matches.py::test_clean_clears_low_hybrid_score PASSED
tests/test_clean_matches.py::test_clean_no_candidates_clears PASSED
tests/test_clean_matches.py::test_clean_output_has_all_entries PASSED
3 passed
```

- [ ] **Step 10: Commit**

```bash
git add scripts/clean_matches.py tests/test_clean_matches.py
git commit -m "feat: update clean_matches to use hybrid_score threshold"
```

---

## Task 4: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 11: Thêm dependencies**

`requirements.txt` hiện ở encoding UTF-16, rewrite bằng UTF-8:

```python
# Chạy lệnh này một lần để fix encoding + thêm packages
python -c "
from pathlib import Path
pkgs = [
    'rouge-score',
    'rank-bm25',
    'sentence-transformers',
    'numpy',
    'requests',
    'python-dotenv',
    'groq',
    'bert-score',
    'rapidfuzz',
    'torch',
    'transformers',
]
Path('requirements.txt').write_text('\n'.join(pkgs) + '\n', encoding='utf-8')
print('Done')
"
```

> **Lưu ý:** Chỉ thêm package thực sự dùng trong project. Kiểm tra `pip list` nếu không chắc một package đã được cài chưa.

- [ ] **Step 12: Commit**

```bash
git add requirements.txt
git commit -m "chore: fix requirements.txt encoding and add sentence-transformers, rank-bm25"
```

---

## Task 5: End-to-end smoke test + verify coverage

- [ ] **Step 13: Chạy toàn bộ pipeline**

Đảm bảo server FastAPI **không cần chạy** cho bước này.

```bash
python scripts/match_articles.py
```

Expected output (dạng):
```
============================...
Hybrid BM25 + Embedding: Match QA → KG articles
============================...
Loaded 378 articles from corpus
Building BM25 index...
Building embeddings for 378 articles (model: all-MiniLM-L6-v2)...
  [10/100] processed...
  ...
  [100/100] processed...
Matched: 100/100
Saved → data/qa_testset_matched.json
```

- [ ] **Step 14: Chạy clean_matches**

```bash
python scripts/clean_matches.py
```

Expected output:
```
Loaded 100 entries
Results:
  Valid   : XX/100 (XX%)      ← phải ≥ 75
Per-category:
  DeviceMgmt     XX/30
  Identity       XX/25
  Network        XX/20
  Teams          XX/25        ← phải ≥ 13/25 (52%)
Saved → data/qa_testset.json
```

Nếu coverage < 75%: thử `--threshold 0.30` để hạ ngưỡng.
Nếu Teams < 50%: thử `--alpha 0.4` trong match_articles để tăng trọng số semantic.

- [ ] **Step 15: Chạy full test suite**

```
pytest tests/ -v
```

Expected: tất cả 12 tests PASS.

- [ ] **Step 16: Commit kết quả**

```bash
git add data/qa_testset.json data/qa_testset_matched.json
git commit -m "data: regenerate testset with hybrid article matching"
```

---

## Checklist nhanh sau hoàn thành

- [ ] `data/qa_testset.json` coverage ≥ 75%
- [ ] Teams coverage ≥ 50%
- [ ] `evaluate.py` vẫn chạy được không lỗi (`bm25_candidates` backward-compat)
- [ ] `data/.cache/` được thêm vào `.gitignore`:
  ```bash
  echo "data/.cache/" >> .gitignore
  git add .gitignore && git commit -m "chore: ignore embedding cache dir"
  ```
