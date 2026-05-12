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

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "has", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "its", "now", "put", "say", "she", "too",
    "use", "was", "who", "why", "yet", "had", "let", "did", "may",
    "any", "two",
}


def tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"\w+", text.lower())
        if len(t) > 2 and t not in _STOPWORDS
    ]


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
    model=None,  # optional pre-built SentenceTransformer
) -> np.ndarray:
    current_ids = [a["article_id"] for a in articles]
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as data:
            if data["article_ids"].tolist() == current_ids:
                print(f"Loaded embeddings from cache ({len(articles)} articles)")
                return data["embeddings"].copy()
        print("Cache stale (article list changed) — rebuilding...")

    print(f"Building embeddings for {len(articles)} articles (model: {model_name})...")
    if model is None:
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
        from sentence_transformers import SentenceTransformer
        embed_model_obj = SentenceTransformer(embed_model)
        article_embeddings = build_or_load_embeddings(articles, CACHE_FILE, embed_model, model=embed_model_obj)
        use_embeddings = True
    except ImportError:
        print("WARNING: sentence-transformers not installed — falling back to BM25-only")

    matched_count = 0

    for i, qa in enumerate(test_set):
        question = qa["question"]
        body     = qa.get("question_body", "")
        answer   = qa.get("answer", "")
        query_text = question + " " + body

        tokens      = tokenize(query_text)
        bm25_raw    = np.array(bm25.get_scores(tokens))
        bm25_norm   = bm25_normalize(bm25_raw)

        if use_embeddings:
            query_vec     = embed_model_obj.encode(query_text, normalize_embeddings=True)
            cosine_scores = article_embeddings @ query_vec
        else:
            cosine_scores = np.zeros(len(articles))

        hybrid_scores = hybrid_combine(bm25_norm, cosine_scores, alpha=alpha)
        candidates    = top_k_candidates(articles, bm25_raw, cosine_scores, hybrid_scores, top_k=top_k)

        if not qa.get("matched_article_id") and candidates:
            best = candidates[0]
            qa["matched_article_id"] = best["article_id"]
            qa["article_id"]         = best["article_id"]
            matched_count += 1

        qa["hybrid_candidates"] = candidates
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
