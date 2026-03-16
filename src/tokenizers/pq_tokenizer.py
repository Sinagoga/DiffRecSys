import json
import os
import numpy as np

from sklearn.decomposition import PCA

import faiss

from src.tokenizers.sid_tokenizer_base import SIDTokenizerBase


class PQTokenizer(SIDTokenizerBase):
    """PQ/OPQ-based tokenizer."""

    def _init_index_factory(self):
        self.sid_quantizer = 'opq_pq'
        use_opq = not self.config.get('disable_opq', False)
        if use_opq:
            self.index_factory = f'OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{self.n_codebook_bits}'
        else:
            self.index_factory = f'IVF1,PQ{self.n_digit}x{self.n_codebook_bits}'

    def _prepare_sentence_embeddings(self, dataset, raw_path: str, pca_path: str):
        """Prepare sentence embeddings for PQ/OPQ quantization."""
        # opq_pq: allows PCA (same behavior as previous implementation)
        if self.config['sent_emb_pca'] > 0 and os.path.exists(pca_path):
            self.log(f'[TOKENIZER] Loading PCA-ed sentence embeddings from {pca_path}...')
            return np.fromfile(pca_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_pca']
            )

        if os.path.exists(raw_path):
            self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
            raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim']
            )
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')

                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                training_item_mask = self._get_items_for_training(dataset)
                pca.fit(raw_embs[training_item_mask])
                sent_embs = pca.transform(raw_embs)
                sent_embs = sent_embs.astype(np.float32, copy=False)
                if self.config.get('normalize_after_pca', True):
                    norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                    sent_embs = sent_embs / norms
                sent_embs.tofile(pca_path)
                return sent_embs
            return raw_embs

        # Otherwise, encode from scratch
        self.log(f'[TOKENIZER] Encoding sentence embeddings...')
        raw_embs = self._encode_sent_emb(dataset, raw_path)
        if self.config['sent_emb_pca'] > 0:
            self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')

            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            training_item_mask = self._get_items_for_training(dataset)
            pca.fit(raw_embs[training_item_mask])
            sent_embs = pca.transform(raw_embs)
            sent_embs = sent_embs.astype(np.float32, copy=False)
            if self.config.get('normalize_after_pca', True):
                norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                sent_embs = sent_embs / norms
            sent_embs.tofile(pca_path)
            return sent_embs
        return raw_embs

    def _generate_semantic_ids(self, sent_embs, sem_ids_path, train_mask):
        """Generate semantic IDs using OPQ/PQ."""

        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')

        # Build index
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        # Handle IndexPreTransform vs non-PreTransform
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        # Extract codes and ids, keeping their order aligned
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        # Decode PQ codes
        faiss_sem_ids = []
        n_bytes = invlists.code_size
        for u8code in pq_codes_u8:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)

        # Align item ordering using ids
        item2sem_ids = {}
        for pos, iid0 in enumerate(ids):
            item = self.id2item[int(iid0) + 1]
            item2sem_ids[item] = tuple(int(v) for v in faiss_sem_ids[pos])

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
