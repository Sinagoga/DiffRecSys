#!/usr/bin/env python3
"""Compare RQ-KMeans and PSE/OPQ tokenizers."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from src.data.yambda_loader import load_embeddings
from src.tokenizers.rq_kmeans import RQKMeans
from src.tokenizers.pse_opq import PSETokenizer
from src.evaluation.metrics import summarize_metrics, plot_comparison


def load_sids(path: str) -> tuple:
    """Load SIDs from parquet. Returns (item_ids, sids_array)."""
    df = pd.read_parquet(path)
    item_ids = df["item_id"].values
    sid_cols = sorted(
        [c for c in df.columns if c.startswith("sid_")],
        key=lambda c: int(c.split("_")[1]),
    )
    sids = df[sid_cols].values.astype(np.int32)
    return item_ids, sids


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dry_run = args.dry_run or config["dry_run"]["enabled"]
    suffix = "_dryrun" if dry_run else ""

    # Load embeddings
    embed_path = config["data"]["output_path"].replace(".parquet", f"{suffix}.parquet")
    item_ids, embeddings = load_embeddings(embed_path)

    # Load RQ-KMeans results
    rq_sids_path = config["rq_kmeans"]["sids_path"].replace(".parquet", f"{suffix}.parquet")
    _, rq_sids = load_sids(rq_sids_path)

    rq_ckpt_path = config["rq_kmeans"]["checkpoint_path"].replace(".pt", f"{suffix}.pt")
    rq_model = RQKMeans.load(rq_ckpt_path)
    rq_reconstructed = rq_model.reconstruct(embeddings)

    rq_codebook_width = min(32, config["rq_kmeans"]["codebook_width"]) if dry_run \
        else config["rq_kmeans"]["codebook_width"]

    # Load PSE results
    pse_sids_path = config["pse_opq"]["sids_path"].replace(".parquet", f"{suffix}.parquet")
    _, pse_sids = load_sids(pse_sids_path)

    pse_index_path = config["pse_opq"]["index_path"].replace(".faiss", f"{suffix}.faiss")
    pse_model = PSETokenizer.load(pse_index_path)
    pse_reconstructed = pse_model.reconstruct(embeddings)

    pse_codebook_size = min(32, config["pse_opq"]["codebook_size"]) if dry_run \
        else config["pse_opq"]["codebook_size"]

    # Compute metrics
    results_rq = summarize_metrics("RQ-KMeans", rq_sids, embeddings,
                                   rq_reconstructed, rq_codebook_width)
    results_pse = summarize_metrics("PSE/OPQ", pse_sids, embeddings,
                                    pse_reconstructed, pse_codebook_size)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for key in ["reconstruction_mse", "codebook_utilization_mean",
                "collision_rate", "entropy_mean"]:
        print(f"{key:>30s}: RQ={results_rq[key]:.6f}  PSE={results_pse[key]:.6f}")
    print("=" * 60)

    # Save results
    output_dir = config["evaluation"]["output_dir"]
    if dry_run:
        output_dir = output_dir + "_dryrun"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump({"rq_kmeans": results_rq, "pse_opq": results_pse}, f, indent=2)

    # Generate plots
    plot_comparison(results_rq, results_pse, output_dir,
                    config["evaluation"]["plot_format"])

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
