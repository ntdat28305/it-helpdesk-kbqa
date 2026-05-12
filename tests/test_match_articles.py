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
    np.testing.assert_allclose(norm, 0.0)


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
