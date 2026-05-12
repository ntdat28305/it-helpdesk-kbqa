# Hybrid Article Matching Pipeline — Design Spec

**Date:** 2026-05-13  
**Status:** Approved  

## Problem

`qa_testset.json` chứa 100 câu hỏi từ Microsoft Q&A. Mỗi câu cần `matched_article_id` trỏ về đúng article trong KG (378 articles) để tính Hit@1/MRR trong `evaluate.py`.

Pipeline hiện tại (BM25-only) cho coverage 58% sau threshold, Teams chỉ 28%. BM25 bỏ sót các câu vague/semantic vì chỉ dựa trên lexical overlap.

## Goal

- Coverage ≥ 75% (có `matched_article_id` hợp lệ)
- Teams coverage ≥ 50%
- Giữ backward-compatible với `evaluate.py`

## Architecture & Data Flow

```
qa_testset.json (100 Q&A)
        ↓
  match_articles.py  (rewrite)
    ├── Load 378 articles từ data/raw/
    ├── Build BM25Okapi index
    ├── Pre-compute article embeddings (MiniLM) → cache data/.cache/article_embeddings.npz
    ├── Encode mỗi question (question + question_body)
    ├── BM25 top-20 candidates (lexical)
    ├── Cosine similarity trên toàn bộ 378 articles (dense)
    ├── Hybrid score: 0.6×bm25_norm + 0.4×cosine
    └── Lưu top-3 candidates + hybrid_score
        ↓
  qa_testset_matched.json
        ↓
  clean_matches.py  (cải thiện)
    ├── Threshold trên hybrid_score (default 0.35)
    ├── Title keyword overlap re-rank (giữ nguyên)
    └── Output → data/qa_testset.json (mặc định)
        ↓
  qa_testset.json  (ground truth sạch cho evaluate.py)
```

## Component Details

### match_articles.py (rewrite)

**Embedding cache**  
Pre-compute embeddings cho 378 articles một lần, cache vào `data/.cache/article_embeddings.npz`.  
Cache tự invalidate nếu số articles thay đổi.

**Hybrid scoring**
```python
query_vec   = MiniLM.encode(question + " " + question_body)  # L2-normalized
cosine[i]   = dot(query_vec, article_vec[i])
bm25_norm[i] = bm25_score[i] / (max_bm25 + 1e-9)
hybrid[i]   = alpha * bm25_norm[i] + (1 - alpha) * cosine[i]
# default alpha = 0.6
```

**Output fields (mỗi entry)**
- `hybrid_candidates`: top-3, mỗi cái có `article_id`, `title`, `bm25_score`, `cosine_score`, `hybrid_score`
- `bm25_candidates`: giữ lại (backward-compatible với evaluate.py)
- `matched_article_id`: best hybrid candidate

**CLI flags**: `--input`, `--output`, `--top-k`, `--alpha`

### clean_matches.py (cải thiện)

- Threshold trên `hybrid_score` (default `0.35`) thay vì `bm25_score` (cũ: `45.0`)
- Re-rank title keyword overlap chạy trên `hybrid_candidates`
- `--output` mặc định `data/qa_testset.json` (không cần copy thủ công)
- `--threshold` và `--alpha` flags

### Error Handling

| Tình huống | Xử lý |
|---|---|
| `sentence-transformers` chưa install | Warning + fallback BM25-only |
| Cache stale (số articles đổi) | Tự động re-build cache |
| Không có candidate trên threshold | `matched_article_id = ""`, `article_id = "qa_{id}"` |

## Files Changed

| File | Thay đổi |
|---|---|
| `scripts/match_articles.py` | Rewrite — Hybrid BM25+MiniLM |
| `scripts/clean_matches.py` | Update threshold field + output path |
| `data/.cache/article_embeddings.npz` | Generated (không commit) |

## Success Criteria

| Metric | Hiện tại | Mục tiêu |
|---|---|---|
| Coverage (có matched_article_id) | 58% | ≥ 75% |
| Teams coverage | 28% | ≥ 50% |
| Script chạy không cần server | ✓ | ✓ |
| Backward-compatible với evaluate.py | ✓ | ✓ |
