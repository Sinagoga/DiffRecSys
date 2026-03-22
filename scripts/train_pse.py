#!/usr/bin/env python3
"""Train PSE/OPQ tokenizer on embeddings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from src.data.yambda_loader import load_embeddings
from src.tokenizers.pse_opq import PSETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train PSE/OPQ")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dry_run = args.dry_run or config["dry_run"]["enabled"]

    # Load embeddings
    embed_path = config["data"]["output_path"]
    if dry_run:
        embed_path = embed_path.replace(".parquet", "_dryrun.parquet")

    item_ids, embeddings = load_embeddings(embed_path)

    # Configure model
    pse_cfg = config["pse_opq"]
    n_digit = pse_cfg["n_digit"]
    codebook_size = pse_cfg["codebook_size"]
    pca_dim = pse_cfg["pca_dim"]

    if dry_run:
        n_digit = min(4, n_digit)
        codebook_size = min(32, codebook_size)
        pca_dim = min(32, pca_dim)

    model = PSETokenizer(
        n_digit=n_digit,
        codebook_size=codebook_size,
        pca_dim=pca_dim,
        disable_opq=pse_cfg["disable_opq"],
        use_gpu=pse_cfg["use_gpu"],
    )

    # Train
    model.fit(embeddings)

    # Encode
    sids = model.encode(embeddings)
    print(f"SIDs shape: {sids.shape}")

    # Save SIDs
    sids_path = pse_cfg["sids_path"]
    if dry_run:
        sids_path = sids_path.replace(".parquet", "_dryrun.parquet")

    sids_df = pd.DataFrame({"item_id": item_ids})
    for i in range(sids.shape[1]):
        sids_df[f"sid_{i}"] = sids[:, i]

    Path(sids_path).parent.mkdir(parents=True, exist_ok=True)
    sids_df.to_parquet(sids_path, index=False)
    print(f"SIDs saved to {sids_path}")

    # Save index
    index_path = pse_cfg["index_path"]
    if dry_run:
        index_path = index_path.replace(".faiss", "_dryrun.faiss")
    model.save(index_path)


if __name__ == "__main__":
    main()
