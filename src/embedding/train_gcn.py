"""
src/embedding/train_gcn.py
Train Graph Autoencoder trên Neo4j KG, lưu node embeddings.

Chạy:
    python -m src.embedding.train_gcn
"""
from __future__ import annotations

import json
import os
from pyexpat import features
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.utils.logger import get_logger
from sentence_transformers import SentenceTransformer

load_dotenv()
logger = get_logger(__name__, log_file="logs/train_gcn.log")

EMBEDDING_DIR  = Path("models/embeddings")
CHECKPOINT_DIR = Path("models/gcn_checkpoint")

# Fix random seed để kết quả reproducible
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ── Lấy data từ Neo4j ────────────────────────────────────────

def get_graph_data() -> tuple[dict, list]:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
    )
    with driver.session() as session:
        nodes = session.run(
            "MATCH (n:Entity) RETURN elementId(n) AS node_id, n.name AS name"
        )
        node_mapping = {r["node_id"]: r["name"] for r in nodes}

        edges = session.run("""
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN elementId(a) AS src, elementId(b) AS tgt
        """)
        edge_list = [(r["src"], r["tgt"]) for r in edges]

    driver.close()
    logger.info(f"Nodes: {len(node_mapping)}, Edges: {len(edge_list)}")
    return node_mapping, edge_list


# ── Chuẩn bị data ────────────────────────────────────────────

def prepare_data(
    node_mapping: dict,
    edge_list: list,
    val_ratio: float = 0.1,
) -> tuple[Data, Data, dict, dict]:
    """
    Chuẩn bị train/val data.
    Thay one-hot bằng random features để tránh overfitting.
    """
    node_ids = list(node_mapping.keys())
    id2idx   = {nid: i for i, nid in enumerate(node_ids)}
    idx2name = {i: node_mapping[nid] for i, nid in enumerate(node_ids)}
    name2idx = {v: k for k, v in idx2name.items()}

    num_nodes = len(node_ids)

    # Random features thay vì one-hot — tránh overfitting
    logger.info("Encoding node names bằng sentence-transformers...")
    st_model  = SentenceTransformer("all-MiniLM-L6-v2")
    node_names = [node_mapping[nid] for nid in node_ids]
    embeddings = st_model.encode(node_names, show_progress_bar=True)
    features   = torch.tensor(embeddings, dtype=torch.float)
    logger.info(f"Features shape: {features.shape}")

    # Edge index
    valid_edges = [
        (id2idx[s], id2idx[t])
        for s, t in edge_list
        if s in id2idx and t in id2idx
    ]

    # Shuffle và split train/val
    random.shuffle(valid_edges)
    split = int(len(valid_edges) * (1 - val_ratio))
    train_edges = valid_edges[:split]
    val_edges   = valid_edges[split:]

    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
    val_edge_index   = torch.tensor(val_edges,   dtype=torch.long).t().contiguous()

    train_data = Data(x=features, edge_index=train_edge_index)
    val_data   = Data(x=features, edge_index=val_edge_index)

    logger.info(
        f"Prepared: {num_nodes} nodes | "
        f"train edges: {len(train_edges)} | "
        f"val edges: {len(val_edges)}"
    )
    return train_data, val_data, idx2name, name2idx


# ── Model ─────────────────────────────────────────────────────

class GAE(nn.Module):
    """Graph Autoencoder với Dropout để tránh overfitting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, embedding_dim)
        self.dropout  = nn.Dropout(p=dropout)

    def encode(self, x, edge_index):
        x = F.relu(self.encoder1(x, edge_index))
        x = self.dropout(x)
        return self.encoder2(x, edge_index)

    def decode(self, z, edge_index):
        src, tgt = edge_index
        return (z[src] * z[tgt]).sum(dim=1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z, self.decode(z, edge_index)


# ── Train ─────────────────────────────────────────────────────

def train(
    train_data: Data,
    val_data: Data,
    hidden_dim: int = 64,
    embedding_dim: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 20,
) -> tuple[GAE, torch.Tensor]:
    """
    Train GAE với early stopping.
    patience: dừng nếu val_loss không giảm sau N epochs.
    """
    input_dim = train_data.x.size(1)
    model     = GAE(input_dim, hidden_dim, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    logger.info(f"Train: {epochs} epochs | patience={patience} | input_dim={input_dim}")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        z, recon_pos = model(train_data.x, train_data.edge_index)

# Tạo negative edges ngẫu nhiên
        num_nodes = train_data.x.size(0)
        num_neg   = train_data.edge_index.size(1)
        neg_src   = torch.randint(0, num_nodes, (num_neg,))
        neg_tgt   = torch.randint(0, num_nodes, (num_neg,))
        neg_edge  = torch.stack([neg_src, neg_tgt])

# Decode negative edges
        neg_recon = model.decode(z, neg_edge)

# Loss: positive=1, negative=0
        pos_loss = criterion(recon_pos, torch.ones(recon_pos.size(0)))
        neg_loss = criterion(neg_recon, torch.zeros(neg_recon.size(0)))
        loss     = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        # Validate
        # Validate
        model.eval()
        with torch.no_grad():
            z_val, val_recon_pos = model(train_data.x, val_data.edge_index)

    # Negative edges cho val
            num_neg_val = val_data.edge_index.size(1)
            neg_src_val = torch.randint(0, num_nodes, (num_neg_val,))
            neg_tgt_val = torch.randint(0, num_nodes, (num_neg_val,))
            neg_edge_val = torch.stack([neg_src_val, neg_tgt_val])
            val_recon_neg = model.decode(z_val, neg_edge_val)

            val_loss = (
                criterion(val_recon_pos, torch.ones(val_recon_pos.size(0))) +
                criterion(val_recon_neg, torch.zeros(val_recon_neg.size(0)))
            )

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss  = val_loss.item()
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"Early stopping tại epoch {epoch}")
                break

    # Load best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # Lấy embeddings
    model.eval()
    with torch.no_grad():
        embeddings, _ = model(train_data.x, train_data.edge_index)

    logger.info(f"Best val loss: {best_val_loss:.4f}")
    return model, embeddings


# ── Lưu kết quả ──────────────────────────────────────────────

def save_results(model, embeddings, idx2name, name2idx):
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    emb_np = embeddings.detach().numpy()
    np.save(EMBEDDING_DIR / "node_embeddings.npy", emb_np)

    (EMBEDDING_DIR / "idx_to_name.json").write_text(
        json.dumps(idx2name, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EMBEDDING_DIR / "name_to_idx.json").write_text(
        json.dumps(name2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.save(model.state_dict(), CHECKPOINT_DIR / "gcn_weights.pt")

    logger.info(f"Saved embeddings: {emb_np.shape}")
    logger.info(f"Saved weights: {CHECKPOINT_DIR / 'gcn_weights.pt'}")


# ── Main ──────────────────────────────────────────────────────

def run():
    node_mapping, edge_list = get_graph_data()
    train_data, val_data, idx2name, name2idx = prepare_data(node_mapping, edge_list)
    model, embeddings = train(train_data, val_data)
    save_results(model, embeddings, idx2name, name2idx)
    logger.info("=== Train GCN xong ===")


if __name__ == "__main__":
    run()