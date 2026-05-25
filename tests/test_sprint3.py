"""Sprint 3 evaluation overhaul tests."""
import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

# ── Stub heavy optional deps ──────────────────────────────────
for _m in ["groq", "dotenv", "rank_bm25", "rouge_score", "bert_score",
           "requests", "sentence_transformers"]:
    sys.modules.setdefault(_m, MagicMock())


# ── S3.1 Faithfulness ─────────────────────────────────────────

def test_faithfulness_prompt_exists():
    import scripts.evaluate as ev
    assert hasattr(ev, "FAITHFULNESS_PROMPT")
    assert "{answer}" in ev.FAITHFULNESS_PROMPT
    assert "{context}" in ev.FAITHFULNESS_PROMPT

def test_faithfulness_returns_float_in_range():
    import scripts.evaluate as ev

    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.choices[0].message.content = '{"supported": 3, "total": 4}'
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(ev, "_get_groq_client", return_value=mock_client):
        score = ev.faithfulness_score(
            answer="Install the Teams client and re-authenticate.",
            context="Teams requires fresh authentication after password change.",
        )
    assert score is not None
    assert 0.0 <= score <= 1.0

def test_faithfulness_returns_none_when_empty_answer():
    import scripts.evaluate as ev
    score = ev.faithfulness_score(answer="", context="some context")
    assert score is None

def test_faithfulness_returns_none_when_empty_context():
    import scripts.evaluate as ev
    score = ev.faithfulness_score(answer="some answer", context="")
    assert score is None


# ── S3.2 Answer Relevancy ─────────────────────────────────────

def test_answer_relevancy_prompt_exists():
    import scripts.evaluate as ev
    assert hasattr(ev, "ANSWER_RELEVANCY_PROMPT")
    assert "{answer}" in ev.ANSWER_RELEVANCY_PROMPT

def test_answer_relevancy_returns_float():
    import scripts.evaluate as ev

    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.choices[0].message.content = '["How do I fix Teams audio?", "Why does Teams drop calls?", "What causes Teams failures?"]'
    mock_client.chat.completions.create.return_value = mock_resp

    # Mock sentence-transformers model
    mock_st = MagicMock()
    mock_st.encode.side_effect = [
        np.array([0.9, 0.1]),                                           # call 1: original question, shape (2,)
        np.array([[0.8, 0.2], [0.7, 0.3], [0.85, 0.15]]),              # call 2: gen questions, shape (3,2)
    ]

    with patch.object(ev, "_get_groq_client", return_value=mock_client):
        with patch.object(ev, "_get_st_model", return_value=mock_st):
            score = ev.answer_relevancy_score(
                question="Why does Teams keep dropping?",
                answer="Teams drops due to VPN split tunneling issues. Disable split tunneling.",
            )
    assert score is not None
    assert 0.0 <= score <= 1.0

def test_answer_relevancy_returns_none_when_empty_answer():
    import scripts.evaluate as ev
    score = ev.answer_relevancy_score(question="any question", answer="")
    assert score is None


# ── S3.3 Pairwise eval ────────────────────────────────────────

def test_pairwise_prompt_exists():
    import scripts.evaluate as ev
    assert hasattr(ev, "PAIRWISE_PROMPT")
    assert "{answer_a}" in ev.PAIRWISE_PROMPT
    assert "{answer_b}" in ev.PAIRWISE_PROMPT
    assert "{question}" in ev.PAIRWISE_PROMPT

def test_pairwise_judge_returns_valid_winner():
    import scripts.evaluate as ev

    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.choices[0].message.content = "A"
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(ev, "_get_groq_client", return_value=mock_client):
        winner = ev.pairwise_judge(
            question="How do I fix Teams audio?",
            answer_a="Check your audio drivers and re-enroll device.",
            answer_b="Microsoft Teams audio issue.",
        )
    assert winner in ("A", "B", "Tie"), f"Expected A/B/Tie, got {winner!r}"

def test_pairwise_judge_returns_tie():
    import scripts.evaluate as ev

    mock_client = MagicMock()
    mock_resp   = MagicMock()
    mock_resp.choices[0].message.content = "Tie"
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.object(ev, "_get_groq_client", return_value=mock_client):
        winner = ev.pairwise_judge("question", "answer a", "answer b")
    assert winner in ("A", "B", "Tie")

def test_pairwise_judge_returns_none_on_missing_answers():
    import scripts.evaluate as ev
    result = ev.pairwise_judge("q", "", "answer b")
    assert result is None
