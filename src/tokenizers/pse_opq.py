"""PSE/OPQ: Parallel Subspace Encoding via FAISS OPQ/PQ.

Implements the parallel subspace SID assignment from DiffGRM.
Uses FAISS index factory with OPQ rotation + Product Quantization.
"""

import math
import numpy as np
import faiss
from pathlib import Path
from sklearn.decomposition import PCA


class PSETokenizer:
    """Parallel Subspace Encoding using FAISS OPQ/PQ.

    Each item is encoded as n_digit independent codes, one per subspace.
    Optionally applies OPQ rotation for better subspace alignment.
    """

    def __init__(self, n_digit: int = 10, codebook_size: int = 512,
                 pca_dim: int = 256, disable_opq: bool = False,
                 use_gpu: str | bool = "auto"):
        self.n_digit = n_digit
        self.codebook_size = codebook_size
        self.n_bits = int(math.log2(codebook_size))
        self.pca_dim = pca_dim
        self.disable_opq = disable_opq
        if use_gpu == "auto":
            self.use_gpu = faiss.get_num_gpus() > 0 and self.n_bits == 8
        else:
            self.use_gpu = bool(use_gpu)
        if self.use_gpu and self.n_bits != 8:
            print(f"Warning: FAISS GPU only supports PQ8, but n_bits={self.n_bits}. Falling back to CPU.")
            self.use_gpu = False
        self.index = None
        self.pca = None
        self._trained_dim = None

    def _build_index(self, dim: int) -> faiss.Index:
        """Build FAISS index using index factory."""
        if self.disable_opq:
            factory_str = f"IVF1,PQ{self.n_digit}x{self.n_bits}"
        else:
            factory_str = f"OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{self.n_bits}"

        index = faiss.index_factory(dim, factory_str)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def fit(self, X: np.ndarray):
        """Train the PSE tokenizer on embeddings.

        Args:
            X: float32 array of shape (N, D)
        """
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Optional PCA dimensionality reduction
        if self.pca_dim and self.pca_dim < X.shape[1]:
            print(f"Applying PCA: {X.shape[1]} -> {self.pca_dim}")
            self.pca = PCA(n_components=self.pca_dim, random_state=42)
            X = self.pca.fit_transform(X).astype(np.float32)
            X = np.ascontiguousarray(X)

        self._trained_dim = X.shape[1]
        self.index = self._build_index(X.shape[1])

        print(f"Training FAISS index (n_digit={self.n_digit}, "
              f"codebook_size={self.codebook_size}, dim={X.shape[1]})...")
        self.index.train(X)
        self.index.add(X)

        # Enable direct map for reconstruction
        ivf = faiss.extract_index_ivf(self.index)
        ivf.make_direct_map()

        print(f"Index trained and populated with {X.shape[0]} vectors")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Extract PQ codes for each vector.

        Args:
            X: float32 array of shape (N, D)

        Returns:
            int32 array of shape (N, n_digit)
        """
        X = np.ascontiguousarray(X, dtype=np.float32)

        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
            X = np.ascontiguousarray(X)

        # Get the underlying IVF index to access inverted lists
        index_cpu = self.index
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)

        ivf_index = faiss.extract_index_ivf(index_cpu)
        invlists = ivf_index.invlists
        list_size = invlists.list_size(0)

        # Get raw codes from the single IVF list (IVF1 = all vectors in list 0)
        code_size = invlists.code_size
        codes_ptr = invlists.get_codes(0)
        codes_raw = faiss.rev_swig_ptr(codes_ptr, list_size * code_size)
        codes_raw = np.array(codes_raw, dtype=np.uint8).reshape(list_size, code_size)

        # Parse codes: each code is n_bits wide, packed in bytes
        codes = self._parse_pq_codes(codes_raw)
        return codes.astype(np.int32)

    def _parse_pq_codes(self, raw_codes: np.ndarray) -> np.ndarray:
        """Parse packed PQ codes into per-subspace integers (vectorized).

        Args:
            raw_codes: uint8 array of shape (N, code_size)

        Returns:
            int array of shape (N, n_digit)
        """
        N = raw_codes.shape[0]
        n_bits = self.n_bits

        if n_bits == 8:
            return raw_codes[:, :self.n_digit].astype(np.int32)

        # Vectorized bit extraction for arbitrary n_bits
        # Unpack all bytes to individual bits: (N, code_size*8)
        bits = np.unpackbits(raw_codes, axis=1, bitorder='little')

        codes = np.zeros((N, self.n_digit), dtype=np.int32)
        for d in range(self.n_digit):
            start_bit = d * n_bits
            # Extract n_bits for this digit across all N vectors at once
            digit_bits = bits[:, start_bit:start_bit + n_bits]  # (N, n_bits)
            # Convert bits to integer: bit 0 is LSB
            powers = (1 << np.arange(n_bits, dtype=np.int32))  # (n_bits,)
            codes[:, d] = digit_bits.astype(np.int32) @ powers

        return codes

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct embeddings from PQ codes.

        Args:
            X: float32 array of shape (N, D) — original embeddings (used to
               identify vectors by index since they were added to the index).

        Returns:
            float32 array of shape (N, D_original)
        """
        N = X.shape[0]
        index_cpu = self.index
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)

        reconstructed = np.zeros((N, self._trained_dim), dtype=np.float32)
        for i in range(N):
            reconstructed[i] = index_cpu.reconstruct(i)

        # Inverse PCA if applied
        if self.pca is not None:
            reconstructed = self.pca.inverse_transform(reconstructed).astype(np.float32)

        return reconstructed

    def save(self, path: str):
        """Save FAISS index and PCA to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index_cpu = self.index
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)

        faiss.write_index(index_cpu, str(path))

        if self.pca is not None:
            import pickle
            pca_path = path.parent / "pca.pkl"
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)

        # Save metadata
        import json
        meta = {
            "n_digit": self.n_digit,
            "codebook_size": self.codebook_size,
            "pca_dim": self.pca_dim,
            "disable_opq": self.disable_opq,
            "trained_dim": self._trained_dim,
        }
        meta_path = path.parent / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"PSE index saved to {path}")

    @classmethod
    def load(cls, path: str, use_gpu: bool = False):
        """Load FAISS index and PCA from disk."""
        path = Path(path)

        import json
        meta_path = path.parent / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        model = cls(
            n_digit=meta["n_digit"],
            codebook_size=meta["codebook_size"],
            pca_dim=meta["pca_dim"],
            disable_opq=meta["disable_opq"],
            use_gpu=use_gpu,
        )
        model._trained_dim = meta["trained_dim"]
        model.index = faiss.read_index(str(path))

        # Re-enable direct map for reconstruction
        ivf = faiss.extract_index_ivf(model.index)
        ivf.make_direct_map()

        if use_gpu:
            res = faiss.StandardGpuResources()
            model.index = faiss.index_cpu_to_gpu(res, 0, model.index)

        pca_path = path.parent / "pca.pkl"
        if pca_path.exists():
            import pickle
            with open(pca_path, "rb") as f:
                model.pca = pickle.load(f)

        print(f"PSE index loaded from {path}")
        return model
