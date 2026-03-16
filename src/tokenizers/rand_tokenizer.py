# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import json

from src.tokenizers.sid_tokenizer_base import SIDTokenizerBase


class RandTokenizer(SIDTokenizerBase):
    """Random mapping tokenizer (no embedding quantization)."""

    def _init_index_factory(self):
        self.sid_quantizer = 'none'
        self.index_factory = f'RAND{self.n_digit}x{self.n_codebook_bits}'

    def _get_quant_tag_extra(self) -> str:
        # Include random seed in tag so different runs don't conflict
        return f'_seed{self.config.get("sid_random_seed", 12345)}'

    def _prepare_sentence_embeddings(self, dataset, raw_path: str, pca_path: str):
        """Random mapping does not require embeddings."""
        return None

    def _generate_semantic_ids(self, sent_embs, sem_ids_path, train_mask):
        """Generate semantic IDs randomly."""
        rng = np.random.default_rng(self.config.get('sid_random_seed', 12345))
        item2sem_ids = {}
        for i in range(1, self.dataset.n_items):
            item = self.id2item[i]
            codes = rng.integers(low=0, high=self.codebook_size, size=self.n_digit, endpoint=False, dtype=np.int64)
            item2sem_ids[item] = tuple(int(c) for c in codes.tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
