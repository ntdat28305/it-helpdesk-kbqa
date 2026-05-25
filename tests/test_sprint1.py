"""Sprint 1 critical fix tests."""
import sys, json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np

# ── Stub heavy optional deps ──────────────────────────────────
for _m in ["groq", "rapidfuzz", "rapidfuzz.process", "dotenv",
           "neo4j", "tavily"]:
    sys.modules.setdefault(_m, MagicMock())

# Stub sentence_transformers
_st_mock = MagicMock()
_st_instance = MagicMock()
_st_instance.encode = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
_st_mock.SentenceTransformer = MagicMock(return_value=_st_instance)
sys.modules["sentence_transformers"] = _st_mock

# Stub src dependencies
sys.modules.setdefault("src.agent.neo4j_query", MagicMock())
sys.modules.setdefault("src.agent.prompts", MagicMock())
sys.modules.setdefault("src.utils.logger", MagicMock())


# ─────────────────────────────────────────────────────────────
# S1.1 — GCN default path
# ─────────────────────────────────────────────────────────────

def test_embedding_search_uses_finetuned_retriever_when_present():
    """embedding_search() uses finetuned retriever when both retriever + semantic emb present."""
    from src.agent.agent import embedding_search

    emb = np.eye(5, dtype=np.float32)
    sem = np.eye(5, dtype=np.float32)
    mock_retriever = MagicMock()
    mock_retriever.encode.return_value = np.array([1., 0., 0., 0., 0.])

    resources = {
        "embeddings":        emb,
        "idx2name":          {str(i): f"Node{i}" for i in range(5)},
        "name2idx":          {f"Node{i}": i for i in range(5)},
        "node_semantic_emb": sem,
        "retriever":         mock_retriever,
    }

    result = embedding_search("Node0", resources, top_k=3)
    assert isinstance(result, list)
    mock_retriever.encode.assert_called_once()


def test_embedding_search_uses_base_encoder_when_no_finetuned():
    """embedding_search() uses base_encoder (not fuzzy) when only base_encoder present."""
    from src.agent.agent import embedding_search

    emb = np.eye(5, dtype=np.float32)
    sem = np.eye(5, dtype=np.float32)
    mock_base = MagicMock()
    mock_base.encode.return_value = np.array([1., 0., 0., 0., 0.])

    resources = {
        "embeddings":        emb,
        "idx2name":          {str(i): f"Node{i}" for i in range(5)},
        "name2idx":          {f"Node{i}": i for i in range(5)},
        "node_semantic_emb": sem,
        "base_encoder":      mock_base,
        # No "retriever" key — should use base_encoder
    }

    result = embedding_search("Node0", resources, top_k=3)
    assert isinstance(result, list)
    mock_base.encode.assert_called_once()


def test_load_resources_loads_sem_emb_independently_of_retriever():
    """node_semantic_emb should be loaded even without models/retriever/."""
    _st_mock.SentenceTransformer.reset_mock()

    # Patch Path so sem_path.exists() = True, retriever_path.exists() = False
    import src.agent.agent as ag

    real_np_load = np.load
    dummy_arr = np.ones((5, 384), dtype=np.float32)

    def fake_path(p):
        m = MagicMock(spec=Path)
        p_str = str(p)
        if "node_semantic" in p_str:
            m.exists.return_value = True
            m.__str__ = lambda s: p_str
        elif "retriever" in p_str:
            m.exists.return_value = False
        elif "node_embeddings" in p_str and "semantic" not in p_str:
            m.exists.return_value = False
        elif "community" in p_str:
            m.exists.return_value = False
        else:
            m.exists.return_value = False
        return m

    with patch.object(ag, "Path", side_effect=fake_path):
        with patch("numpy.load", return_value=dummy_arr):
            resources = ag.load_resources()

    # sem emb should be loaded
    assert "node_semantic_emb" in resources
    # base_encoder should be loaded (retriever absent)
    assert "base_encoder" in resources or "retriever" in resources
    # SentenceTransformer called with base model
    calls_str = str(_st_mock.SentenceTransformer.call_args_list)
    assert "all-MiniLM-L6-v2" in calls_str

# ─────────────────────────────────────────────────────────────
# S1.2 — Community embedding lookup
# ─────────────────────────────────────────────────────────────

import importlib
import contextlib

@contextlib.contextmanager
def _real_neo4j_query():
    """Temporarily replace the neo4j_query stub with the real module."""
    stub = sys.modules.get("src.agent.neo4j_query")
    # Remove stub so importlib loads the real module
    sys.modules.pop("src.agent.neo4j_query", None)
    real_mod = importlib.import_module("src.agent.neo4j_query")
    try:
        yield real_mod
    finally:
        # Restore original stub so S1.1 tests remain isolated
        if stub is not None:
            sys.modules["src.agent.neo4j_query"] = stub
        else:
            sys.modules.pop("src.agent.neo4j_query", None)


def test_get_community_context_embedding_path():
    """get_community_context should use cosine sim when embeddings provided."""
    import numpy as np

    with _real_neo4j_query() as nq:
        get_community_context = nq.get_community_context

        communities = {
            "0": {"summary": "Azure AD authentication and sign-in issues", "nodes": ["Azure AD"]},
            "1": {"summary": "Intune device enrollment and management",     "nodes": ["Intune"]},
            "2": {"summary": "Teams meeting and audio problems",            "nodes": ["Teams"]},
        }

        comm_embs = np.array([
            [0.1, 0.9, 0.1],  # community 0 — Azure AD
            [0.9, 0.1, 0.1],  # community 1 — Intune (should win)
            [0.1, 0.1, 0.9],  # community 2 — Teams
        ], dtype=np.float32)
        community_ids = ["0", "1", "2"]

        retriever_mock = MagicMock()
        retriever_mock.encode.return_value = np.array([0.9, 0.1, 0.1])  # "Intune" → matches community 1

        result = get_community_context(
            "Intune",
            communities,
            community_embs=comm_embs,
            community_ids=community_ids,
            retriever=retriever_mock,
        )
        assert "Intune" in result or "device" in result.lower() or "enrollment" in result.lower(), \
            f"Expected Intune community summary, got: {result!r}"


def test_get_community_context_fallback_when_no_embeddings():
    """get_community_context should fall back to substring when embeddings absent."""
    with _real_neo4j_query() as nq:
        get_community_context = nq.get_community_context

        communities = {
            "0": {"summary": "Azure AD authentication issues", "nodes": ["Azure AD", "Sign-in"]},
        }
        result = get_community_context("Azure AD", communities)
        assert "Azure AD" in result or "authentication" in result.lower(), \
            f"Fallback expected Azure AD match, got: {result!r}"


def test_get_community_context_returns_empty_below_threshold():
    """get_community_context falls back to substring when embedding score < threshold;
    returns empty string when substring also finds no match."""
    import numpy as np

    with _real_neo4j_query() as nq:
        get_community_context = nq.get_community_context

        communities = {
            "0": {"summary": "Azure AD authentication", "nodes": ["Azure AD"]},
        }
        comm_embs = np.array([[0.1, 0.9, 0.1]], dtype=np.float32)
        community_ids = ["0"]

        retriever_mock = MagicMock()
        # Low similarity vector — should score below 0.25 threshold
        retriever_mock.encode.return_value = np.array([0.0, 0.0, 0.1])

        result = get_community_context(
            "random unrelated query",
            communities,
            community_embs=comm_embs,
            community_ids=community_ids,
            retriever=retriever_mock,
        )
        assert result == "", f"Expected empty string for low-score match, got: {result!r}"


# ─────────────────────────────────────────────────────────────
# S1.4 — LLM judge model upgrade
# ─────────────────────────────────────────────────────────────

def test_judge_model_is_70b():
    """evaluate.py must use llama-3.3-70b-versatile as the judge model."""
    import importlib.util, sys
    # Stub groq so the module imports without a real API key
    sys.modules.setdefault("groq", __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock())
    spec = importlib.util.spec_from_file_location(
        "evaluate", "scripts/evaluate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "JUDGE_MODEL"), "JUDGE_MODEL constant not found in evaluate.py"
    assert mod.JUDGE_MODEL == "llama-3.3-70b-versatile", \
        f"Expected llama-3.3-70b-versatile, got {mod.JUDGE_MODEL!r}"
