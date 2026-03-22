#!/usr/bin/env python3
"""Train RQ-KMeans tokenizer on embeddings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from src.data.yambda_loader import load_embeddings
from src.tokenizers.rq_kmeans import RQKMeans


def main():
    parser = argparse.ArgumentParser(description="Train RQ-KMeans")
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
    rq_cfg = config["rq_kmeans"]
    codebook_width = rq_cfg["codebook_width"]
    num_hierarchies = rq_cfg["num_hierarchies"]

    if dry_run:
        codebook_width = min(32, codebook_width)
        num_hierarchies = min(4, num_hierarchies)

    model = RQKMeans(
        num_hierarchies=num_hierarchies,
        codebook_width=codebook_width,
        normalize_residuals=rq_cfg["normalize_residuals"],
        max_iter=rq_cfg["max_iter"] if not dry_run else 10,
        mini_batch_size=rq_cfg["mini_batch_size"],
        device=rq_cfg["device"],
    )

    # Train
    model.fit(embeddings)

    # Encode
    sids = model.encode(embeddings)
    print(f"SIDs shape: {sids.shape}")

    # Save SIDs
    sids_path = rq_cfg["sids_path"]
    if dry_run:
        sids_path = sids_path.replace(".parquet", "_dryrun.parquet")

    sids_df = pd.DataFrame({"item_id": item_ids})
    for i in range(sids.shape[1]):
        sids_df[f"sid_{i}"] = sids[:, i]

    Path(sids_path).parent.mkdir(parents=True, exist_ok=True)
    sids_df.to_parquet(sids_path, index=False)
    print(f"SIDs saved to {sids_path}")

    # Save checkpoint
    ckpt_path = rq_cfg["checkpoint_path"]
    if dry_run:
        ckpt_path = ckpt_path.replace(".pt", "_dryrun.pt")
    model.save(ckpt_path)


if __name__ == "__main__":
    main()
