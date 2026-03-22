#!/usr/bin/env python3
"""Download and prepare Yambda embeddings."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.data.yambda_loader import load_yambda_embeddings, save_embeddings


def main():
    parser = argparse.ArgumentParser(description="Prepare Yambda embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dry_run = args.dry_run or config["dry_run"]["enabled"]
    item_ids, embeddings = load_yambda_embeddings(config, dry_run=dry_run)

    output_path = config["data"]["output_path"]
    if dry_run:
        output_path = output_path.replace(".parquet", "_dryrun.parquet")

    save_embeddings(item_ids, embeddings, output_path)
    print(f"Done. Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
