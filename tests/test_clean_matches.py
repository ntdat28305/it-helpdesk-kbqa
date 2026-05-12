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
