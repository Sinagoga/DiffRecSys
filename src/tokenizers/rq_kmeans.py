"""RQ-KMeans: Residual Quantization with K-Means (pure PyTorch).

Implements the hierarchical/sequential SID assignment from GRID.
Each level fits K-Means on residuals from the previous level.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MiniBatchKMeans:
    """Single-level Mini-Batch K-Means in PyTorch."""

    def __init__(self, n_clusters: int = 512, max_iter: int = 100,
                 mini_batch_size: int = 10000, device: str = "cpu"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.mini_batch_size = mini_batch_size
        self.device = device
        self._centroids = None
        self._counts = None

    def _kmeans_plus_plus_init(self, X: torch.Tensor) -> torch.Tensor:
        """KMeans++ initialization on a subsample."""
        n = min(len(X), self.mini_batch_size)
        indices = torch.randperm(len(X))[:n]
        X_sub = X[indices].to(self.device)

        centroids = torch.empty(self.n_clusters, X.shape[1], device=self.device)
        idx = torch.randint(0, n, (1,)).item()
        centroids[0] = X_sub[idx]

        for k in range(1, self.n_clusters):
            dists = torch.cdist(X_sub, centroids[:k]).min(dim=1).values  # (n,)
            probs = dists ** 2
            probs = probs / probs.sum()
            chosen = torch.multinomial(probs, 1).item()
            centroids[k] = X_sub[chosen]

        return centroids

    def fit(self, X: torch.Tensor):
        """Fit K-Means on X using mini-batch updates.

        Args:
            X: float32 tensor of shape (N, D)
        """
        N = len(X)
        self._centroids = self._kmeans_plus_plus_init(X)
        self._counts = torch.zeros(self.n_clusters, device=self.device)

        for iteration in range(self.max_iter):
            # Sample a mini-batch
            indices = torch.randperm(N)[:self.mini_batch_size]
            batch = X[indices].to(self.device)

            # Assign to nearest centroid
            dists = torch.cdist(batch, self._centroids)  # (B, K)
            assignments = dists.argmin(dim=1)  # (B,)

            # Online centroid update
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() == 0:
                    continue
                count = mask.sum().float()
                self._counts[k] += count
                eta = count / self._counts[k]
                self._centroids[k] = (1 - eta) * self._centroids[k] + eta * batch[mask].mean(dim=0)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Assign each point to nearest centroid.

        Args:
            X: float32 tensor of shape (N, D)

        Returns:
            int64 tensor of shape (N,)
        """
        # Process in chunks to avoid OOM
        chunk_size = 50000
        assignments = []
        for start in range(0, len(X), chunk_size):
            batch = X[start:start + chunk_size].to(self.device)
            dists = torch.cdist(batch, self._centroids)
            assignments.append(dists.argmin(dim=1).cpu())
        return torch.cat(assignments)

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids


class RQKMeans:
    """Residual Quantization with K-Means (multi-level).

    Matches the GRID paper approach: at each level, optionally normalize
    residuals, fit K-Means, subtract selected centroids to get new residuals.
    """

    def __init__(self, num_hierarchies: int = 10, codebook_width: int = 512,
                 normalize_residuals: bool = True, max_iter: int = 100,
                 mini_batch_size: int = 10000, device: str = "auto"):
        self.num_hierarchies = num_hierarchies
        self.codebook_width = codebook_width
        self.normalize_residuals = normalize_residuals
        self.max_iter = max_iter
        self.mini_batch_size = mini_batch_size
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.kmeans_levels = []

    def fit(self, X: np.ndarray):
        """Fit RQ-KMeans on embeddings.

        Args:
            X: float32 array of shape (N, D)
        """
        residuals = torch.from_numpy(X).float()
        self.kmeans_levels = []

        for level in tqdm(range(self.num_hierarchies), desc="RQ-KMeans levels"):
            input_data = residuals.clone()
            if self.normalize_residuals:
                input_data = F.normalize(input_data, dim=1)

            kmeans = MiniBatchKMeans(
                n_clusters=self.codebook_width,
                max_iter=self.max_iter,
                mini_batch_size=self.mini_batch_size,
                device=self.device,
            )
            kmeans.fit(input_data)
            self.kmeans_levels.append(kmeans)

            # Compute assignments and new residuals
            assignments = kmeans.predict(input_data)
            selected = kmeans.centroids[assignments]  # (N, D)
            residuals = input_data - selected.cpu()

        print(f"RQ-KMeans fitted: {self.num_hierarchies} levels, "
              f"codebook_width={self.codebook_width}")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode embeddings into SID codes.

        Args:
            X: float32 array of shape (N, D)

        Returns:
            int32 array of shape (N, num_hierarchies)
        """
        residuals = torch.from_numpy(X).float()
        all_assignments = []

        for level in range(self.num_hierarchies):
            input_data = residuals.clone()
            if self.normalize_residuals:
                input_data = F.normalize(input_data, dim=1)

            kmeans = self.kmeans_levels[level]
            assignments = kmeans.predict(input_data)
            all_assignments.append(assignments.numpy())

            selected = kmeans.centroids[assignments]
            residuals = input_data - selected.cpu()

        return np.stack(all_assignments, axis=1).astype(np.int32)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct embeddings from SID codes.

        Note: with normalize_residuals=True, exact reconstruction is not
        straightforward because normalization is lossy. We approximate by
        summing centroids (which is what the model sees).

        Args:
            X: float32 array of shape (N, D)

        Returns:
            float32 array of shape (N, D)
        """
        codes = self.encode(X)
        N, D = X.shape
        reconstructed = torch.zeros(N, D)

        for level in range(self.num_hierarchies):
            kmeans = self.kmeans_levels[level]
            level_codes = torch.from_numpy(codes[:, level]).long()
            reconstructed += kmeans.centroids[level_codes].cpu()

        return reconstructed.numpy()

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "num_hierarchies": self.num_hierarchies,
            "codebook_width": self.codebook_width,
            "normalize_residuals": self.normalize_residuals,
            "centroids": [km.centroids.cpu() for km in self.kmeans_levels],
            "counts": [km._counts.cpu() for km in self.kmeans_levels],
        }
        torch.save(checkpoint, path)
        print(f"RQ-KMeans saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            num_hierarchies=checkpoint["num_hierarchies"],
            codebook_width=checkpoint["codebook_width"],
            normalize_residuals=checkpoint["normalize_residuals"],
            device=device,
        )
        for i in range(checkpoint["num_hierarchies"]):
            km = MiniBatchKMeans(
                n_clusters=checkpoint["codebook_width"],
                device=device,
            )
            km._centroids = checkpoint["centroids"][i].to(device)
            km._counts = checkpoint["counts"][i].to(device)
            model.kmeans_levels.append(km)
        print(f"RQ-KMeans loaded from {path}")
        return model
