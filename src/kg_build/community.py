"""
src/kg_build/community.py
Phát hiện communities từ Neo4j KG dùng Leiden algorithm.

Chạy:
    python -m src.kg_build.community
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from graspologic.partition import leiden
from neo4j import GraphDatabase

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__, log_file="logs/community.log")

OUTPUT_FILE = Path("data/communities.json")


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


# ── Detect communities ────────────────────────────────────────

def detect_communities(
    node_mapping: dict,
    edge_list: list,
) -> dict[int, list[str]]:
    """
    Chạy Leiden algorithm để phát hiện communities.

    Returns:
        dict {community_id: [node_names]}
    """
    # Re-index nodes về 0..N-1
    node_ids = list(node_mapping.keys())
    id2idx   = {nid: i for i, nid in enumerate(node_ids)}
    idx2name = {i: node_mapping[nid] for i, nid in enumerate(node_ids)}

    # Tạo edge list với index
    edges = [
        (id2idx[s], id2idx[t], 1.0)
        for s, t in edge_list
        if s in id2idx and t in id2idx
    ]

    if not edges:
        logger.error("Không có edges hợp lệ")
        return {}

    # Chạy Leiden
    logger.info("Chạy Leiden algorithm...")
    partition = leiden(edges, trials=3)

    # Group nodes theo community
    communities: dict[int, list[str]] = {}
    for node_idx, community_id in partition.items():
        name = idx2name.get(node_idx, f"node_{node_idx}")
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(name)

    logger.info(f"Tìm được {len(communities)} communities")

    # Log thống kê
    sizes = [len(v) for v in communities.values()]
    logger.info(
        f"Community size: "
        f"min={min(sizes)}, "
        f"max={max(sizes)}, "
        f"avg={sum(sizes)/len(sizes):.1f}"
    )

    return communities


# ── Lưu kết quả ──────────────────────────────────────────────

def save_communities(communities: dict[int, list[str]]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Convert key sang string cho JSON
    data = {str(k): v for k, v in communities.items()}
    OUTPUT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved: {OUTPUT_FILE}")


# ── Preview ───────────────────────────────────────────────────

def preview_communities(communities: dict[int, list[str]], top_n: int = 5):
    """In preview các community lớn nhất."""
    sorted_comms = sorted(
        communities.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    print(f"\n=== Top {top_n} communities lớn nhất ===")
    for comm_id, nodes in sorted_comms[:top_n]:
        print(f"\nCommunity {comm_id} ({len(nodes)} nodes):")
        for n in nodes[:8]:
            print(f"  - {n}")
        if len(nodes) > 8:
            print(f"  ... và {len(nodes)-8} nodes khác")


# ── Main ──────────────────────────────────────────────────────

def run():
    node_mapping, edge_list = get_graph_data()
    communities = detect_communities(node_mapping, edge_list)
    save_communities(communities)
    preview_communities(communities)
    logger.info("=== Community detection xong ===")


if __name__ == "__main__":
    run()