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
