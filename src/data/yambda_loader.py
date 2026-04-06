"""Download and prepare Yambda embeddings for SID experiments."""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_yambda_embeddings(config: dict, dry_run: bool = False):
    """Load Yambda embeddings, either from HuggingFace or synthetic (dry-run).

    Args:
        config: Full config dict with 'data' and 'dry_run' sections.
        dry_run: If True, generate random unit-sphere embeddings.

    Returns:
        item_ids: np.ndarray of uint32 shape (N,)
        embeddings: np.ndarray of float32 shape (N, D)
    """
    if dry_run:
        n_items = config["dry_run"]["n_items"]
        embed_dim = config["dry_run"]["embed_dim"]
        rng = np.random.RandomState(config["data"]["seed"])
        embeddings = rng.randn(n_items, embed_dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        item_ids = np.arange(n_items, dtype=np.uint32)
        print(f"[dry-run] Generated {n_items} random embeddings of dim {embed_dim}")
        return item_ids, embeddings

    from datasets import load_dataset

    data_cfg = config["data"]
    source = data_cfg["source"]
    embed_column = data_cfg["embed_column"]
    n_items = data_cfg["n_items"]
    seed = data_cfg["seed"]

    print(f"Loading embeddings from '{source}' (data_files='embeddings.parquet')...")
    ds = load_dataset(source, data_files="embeddings.parquet", split="train")

    # Shuffle and take subset
    ds = ds.shuffle(seed=seed)
    if n_items < len(ds):
        ds = ds.select(range(n_items))
        print(f"Selected {n_items} items from {len(ds)} total")

    print(f"Extracting embeddings (column='{embed_column}')...")
    embeddings_list = []
    item_ids_list = []
    for i, row in enumerate(tqdm(ds, total=len(ds), desc="Reading embeddings")):
        embeddings_list.append(row[embed_column])
        item_ids_list.append(row.get("item_id", i))

    embeddings = np.array(embeddings_list, dtype=np.float32)
    item_ids = np.array(item_ids_list, dtype=np.uint32)
    print(f"Loaded {len(item_ids)} embeddings of dim {embeddings.shape[1]}")
    return item_ids, embeddings


def save_embeddings(item_ids: np.ndarray, embeddings: np.ndarray, path: str):
    """Save embeddings to parquet with item_id + flattened embed columns.

    Args:
        item_ids: shape (N,)
        embeddings: shape (N, D)
        path: output parquet file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"item_id": item_ids}
    for d in range(embeddings.shape[1]):
        data[f"embed_{d}"] = embeddings[:, d]

    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} embeddings to {path}")


def load_embeddings(path: str):
    """Load embeddings from parquet.

    Returns:
        item_ids: np.ndarray of uint32 shape (N,)
        embeddings: np.ndarray of float32 shape (N, D)
    """
    df = pd.read_parquet(path)
    item_ids = df["item_id"].values.astype(np.uint32)
    embed_cols = sorted(
        [c for c in df.columns if c.startswith("embed_")],
        key=lambda c: int(c.split("_")[1]),
    )
    embeddings = df[embed_cols].values.astype(np.float32)
    print(f"Loaded {len(item_ids)} embeddings of dim {embeddings.shape[1]} from {path}")
    return item_ids, embeddings
