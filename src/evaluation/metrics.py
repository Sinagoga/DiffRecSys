"""Evaluation metrics for SID assignment comparison."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Mean squared error between original and reconstructed embeddings."""
    return float(np.mean((X - X_hat) ** 2))


def codebook_utilization(sids: np.ndarray, codebook_width: int) -> np.ndarray:
    """Fraction of codebook entries used at each level.

    Args:
        sids: int32 array of shape (N, num_levels)
        codebook_width: number of codes per level

    Returns:
        float array of shape (num_levels,) with utilization in [0, 1]
    """
    num_levels = sids.shape[1]
    util = np.zeros(num_levels)
    for level in range(num_levels):
        unique_codes = len(np.unique(sids[:, level]))
        util[level] = unique_codes / codebook_width
    return util


def collision_rate(sids: np.ndarray) -> float:
    """Fraction of items that share their full SID with another item.

    Returns:
        float in [0, 1]
    """
    N = sids.shape[0]
    # Convert each row to a tuple for hashing
    sid_tuples = [tuple(row) for row in sids]
    unique_count = len(set(sid_tuples))
    return 1.0 - unique_count / N


def entropy_per_level(sids: np.ndarray, codebook_width: int) -> np.ndarray:
    """Shannon entropy per level, normalized by max entropy.

    Args:
        sids: int32 array of shape (N, num_levels)
        codebook_width: number of codes per level

    Returns:
        float array of shape (num_levels,) with normalized entropy in [0, 1]
    """
    num_levels = sids.shape[1]
    max_entropy = np.log2(codebook_width)
    entropies = np.zeros(num_levels)

    for level in range(num_levels):
        codes = sids[:, level]
        counts = np.bincount(codes, minlength=codebook_width)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        entropies[level] = entropy / max_entropy if max_entropy > 0 else 0.0

    return entropies


def summarize_metrics(name: str, sids: np.ndarray, X: np.ndarray,
                      X_hat: np.ndarray, codebook_width: int) -> dict:
    """Compute all metrics for a single method.

    Returns:
        dict with metric name -> value
    """
    mse = reconstruction_mse(X, X_hat)
    util = codebook_utilization(sids, codebook_width)
    coll = collision_rate(sids)
    ent = entropy_per_level(sids, codebook_width)

    return {
        "name": name,
        "reconstruction_mse": mse,
        "codebook_utilization_mean": float(np.mean(util)),
        "codebook_utilization_per_level": util.tolist(),
        "collision_rate": coll,
        "entropy_mean": float(np.mean(ent)),
        "entropy_per_level": ent.tolist(),
    }


def plot_comparison(results_rq: dict, results_pse: dict, output_dir: str,
                    plot_format: str = "png"):
    """Generate 4 comparison plots.

    1. Codebook utilization per level
    2. Entropy per level
    3. Reconstruction MSE (bar chart)
    4. Summary metrics (bar chart)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_levels = len(results_rq["codebook_utilization_per_level"])
    levels = np.arange(num_levels)

    # 1. Codebook utilization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(levels - 0.2, results_rq["codebook_utilization_per_level"],
           0.4, label="RQ-KMeans", color="steelblue")
    ax.bar(levels + 0.2, results_pse["codebook_utilization_per_level"],
           0.4, label="PSE/OPQ", color="coral")
    ax.set_xlabel("Level")
    ax.set_ylabel("Codebook Utilization")
    ax.set_title("Codebook Utilization per Level")
    ax.set_xticks(levels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"codebook_utilization.{plot_format}", dpi=150)
    plt.close(fig)

    # 2. Entropy per level
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(levels - 0.2, results_rq["entropy_per_level"],
           0.4, label="RQ-KMeans", color="steelblue")
    ax.bar(levels + 0.2, results_pse["entropy_per_level"],
           0.4, label="PSE/OPQ", color="coral")
    ax.set_xlabel("Level")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("Code Entropy per Level")
    ax.set_xticks(levels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"entropy.{plot_format}", dpi=150)
    plt.close(fig)

    # 3. Reconstruction MSE
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ["RQ-KMeans", "PSE/OPQ"]
    mses = [results_rq["reconstruction_mse"], results_pse["reconstruction_mse"]]
    bars = ax.bar(methods, mses, color=["steelblue", "coral"])
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Error")
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.6f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / f"reconstruction_mse.{plot_format}", dpi=150)
    plt.close(fig)

    # 4. Summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("Avg Utilization", "codebook_utilization_mean"),
        ("Avg Entropy", "entropy_mean"),
        ("Collision Rate", "collision_rate"),
    ]
    for ax, (title, key) in zip(axes, metrics):
        vals = [results_rq[key], results_pse[key]]
        bars = ax.bar(methods, vals, color=["steelblue", "coral"])
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"summary.{plot_format}", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")
