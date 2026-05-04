"""
scripts/finetune_retriever.py
Finetune msmarco-MiniLM-L6-cos-v5 lam IT helpdesk bi-encoder retriever.

Chay:
    python scripts/finetune_retriever.py

Output: models/retriever/  (HuggingFace SentenceTransformer format)
"""
from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    evaluation,
    losses,
)
from torch.utils.data import DataLoader

PAIRS_FILE  = Path("data/finetune_pairs.jsonl")
OUTPUT_DIR  = Path("models/retriever")
BASE_MODEL  = "msmarco-MiniLM-L6-cos-v5"
EPOCHS      = 10
BATCH_SIZE  = 128  # A100 80GB — batch lớn = nhiều in-batch negatives hơn cho MNRL


def load_pairs() -> list[dict]:
    pairs = []
    with PAIRS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_evaluator() -> evaluation.InformationRetrievalEvaluator | None:
    """Xay evaluator tu test_set.json + data/raw/ articles."""
    test_path = Path("data/test_set.json")
    raw_dir   = Path("data/raw")
    if not test_path.exists() or not raw_dir.exists():
        return None

    test_set = json.loads(test_path.read_text(encoding="utf-8"))

    articles: dict[str, str] = {}
    for f in raw_dir.rglob("*.json"):
        if ".cache" in str(f):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            aid  = data["metadata"]["article_id"]
            articles[aid] = data["plain_text"][:512]
        except Exception:
            pass

    if not articles:
        return None

    queries:       dict[str, str]       = {}
    corpus:        dict[str, str]       = {}
    relevant_docs: dict[str, set[str]]  = {}

    for i, qa in enumerate(test_set):
        aid = qa["article_id"]
        if aid not in articles:
            continue
        qid = str(i)
        queries[qid]       = qa["question"]
        corpus[aid]        = articles[aid]
        relevant_docs[qid] = {aid}

    if not queries:
        return None

    print(f"Evaluator: {len(queries)} queries, {len(corpus)} docs")
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="it-helpdesk",
        show_progress_bar=False,
    )


def main():
    print("=== finetune_retriever.py ===")
    print(f"Base model : {BASE_MODEL}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")

    model = SentenceTransformer(BASE_MODEL)

    pairs = load_pairs()
    print(f"Training pairs: {len(pairs)}")

    train_examples = [
        InputExample(
            texts=(
                [p["query"], p["positive"], p["negative"]]
                if "negative" in p
                else [p["query"], p["positive"]]
            )
        )
        for p in pairs
    ]

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=BATCH_SIZE
    )
    train_loss   = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, int(len(train_dataloader) * EPOCHS * 0.1))
    print(f"Warmup steps: {warmup_steps}")

    evaluator = build_evaluator()
    if evaluator:
        print("InformationRetrievalEvaluator se chay sau moi epoch")
    else:
        print("WARNING: Khong build duoc evaluator -- train khong co eval")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_kwargs: dict = dict(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(OUTPUT_DIR),
        show_progress_bar=True,
        bf16=True,  # A100 native bf16 — nhanh hơn fp32, không mất accuracy
    )
    if evaluator is not None:
        fit_kwargs["evaluator"]        = evaluator
        fit_kwargs["evaluation_steps"] = len(train_dataloader)
        fit_kwargs["save_best_model"]  = True

    model.fit(**fit_kwargs)

    print(f"\nModel saved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
