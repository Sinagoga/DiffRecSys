import os
import numpy as np
import json

import faiss

from src.tokenizers.sid_tokenizer_base import SIDTokenizerBase


class RQKMeansTokenizer(SIDTokenizerBase):
    """Residual Quantization (KMeans) tokenizer."""

    def _init_index_factory(self):
        self.sid_quantizer = 'rq_kmeans'
        self.index_factory = f'RQKMEANS{self.n_digit}x{self.n_codebook_bits}'

    def _get_quant_tag_extra(self) -> str:
        # Include seed + iterations to avoid conflicts across different runs
        return f'_seed{self.config.get("rq_kmeans_seed", 1234)}_it{self.config.get("rq_kmeans_niters", 20)}'

    def _prepare_sentence_embeddings(self, dataset, raw_path: str, pca_path: str):
        """Prepare sentence embeddings for RQ-KMeans (raw only, no PCA)."""
        if os.path.exists(raw_path):
            self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
            return np.fromfile(raw_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim']
            )

        self.log(f'[TOKENIZER] Encoding sentence embeddings (RAW, no PCA for RQ-KMeans)...')
        return self._encode_sent_emb(dataset, raw_path)

    def _generate_semantic_ids(self, sent_embs, sem_ids_path, train_mask):
        """Generate semantic IDs using Residual Quantization (KMeans)."""

        d = sent_embs.shape[1]
        K = self.codebook_size
        niter = int(self.config.get('rq_kmeans_niters', 20))
        seed = int(self.config.get('rq_kmeans_seed', 1234))

        # Initialize residuals as the original vectors
        residuals = sent_embs.copy().astype(np.float32, copy=False)
        codes_all = np.zeros((sent_embs.shape[0], self.n_digit), dtype=np.int64)

        for stage in range(self.n_digit):
            kmeans = faiss.Kmeans(d=d, k=K, niter=niter, verbose=False, seed=seed + stage)
            kmeans.train(residuals[train_mask])
            # In current Faiss Python, Kmeans.centroids is already a numpy array
            centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            assert centroids.shape == (K, d), f"centroids shape {centroids.shape} != {(K, d)}"

            # Assign nearest centroids for all samples
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            D, I = index.search(residuals, 1)  # I: [N, 1]
            codes_all[:, stage] = I[:, 0].astype(np.int64)

            # Update residuals
            residuals = residuals - centroids[I[:, 0]]

        # Convert to dict
        item2sem_ids = {}
        for i in range(codes_all.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes_all[i].tolist())

        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
