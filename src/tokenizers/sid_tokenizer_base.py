from collections import defaultdict
from typing import Optional
import os
import math
import json
import pickle
import logging

import numpy as np

import torch

from sentence_transformers import SentenceTransformer

from src.datasets.base_dataset import BaseDataset as AbstractDataset
from src.tokenizers.abstract_tokenizer import AbstractTokenizer

logger = logging.getLogger(__name__)


class SIDTokenizerBase(AbstractTokenizer):
    """Base class for SID-based tokenizers (PQ / RQ-KMeans / Random)."""

    def __init__(
            self,
            config: dict
        ):
        # Provide defaults to avoid KeyError
        config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        config.setdefault('num_proc', 1)

        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])

        # Initialize base class (stores config and logger)
        super(SIDTokenizerBase, self).__init__(config)

        # Special tokens - simplify token ID assignment
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.mask_token = -1  # MASK token used for inference; not in vocab
        self.sid_offset = 3  # SID tokens start from 3

        # Initialize quantizer-specific configuration (index factory, tags, etc.)
        self._init_index_factory()
        logger.info(f'[TOKENIZER] Index factory: {getattr(self, "index_factory", None)}')

    # -------------------------------------------------------------------------
    # Methods to be implemented by subclasses
    # -------------------------------------------------------------------------
    def _init_index_factory(self):
        """Configure self.index_factory and any quantizer-specific settings."""
        raise NotImplementedError

    def _get_quant_tag_extra(self) -> str:
        """Return extra suffix for the quantization tag (seed/iters etc.)."""
        return ''

    def _prepare_sentence_embeddings(
        self,
        dataset: AbstractDataset,
        raw_path: str,
        pca_path: str,
    ) -> Optional[np.ndarray]:
        """Prepare sentence embeddings required by the quantizer.

        Returns:
            np.ndarray or None: embeddings array (N, D) or None if not needed.
        """
        raise NotImplementedError

    def _generate_semantic_ids(
        self,
        sent_embs: Optional[np.ndarray],
        sem_ids_path: str,
        train_mask: np.ndarray,
    ):
        """Generate semantic IDs and save to sem_ids_path."""
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared helper utilities
    # -------------------------------------------------------------------------
    @property
    def n_digit(self):
        return self.config['n_digit']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.n_digit  # [BOS] + n_digit SID tokens

    @property
    def vocab_size(self) -> int:
        return self.sid_offset + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens
    
    def digit_vocab_pos(self, digit: int | torch.Tensor) -> int:
        """Calculate the starting token ID for a given digit position."""
        return self.sid_offset + digit * self.codebook_size

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str) -> np.ndarray:
        """Encode sentence embeddings using a Hugging Face SentenceTransformer and normalize vectors."""
        meta_sentences = []
        for v in dataset.id_mapping['id2item']:  # Skip item_id=0 (PAD)
            if v != "[PAD]":
                meta_sentences.append(dataset.item2meta.get(v, ""))

        # Supports any HF model id (e.g., Alibaba-NLP/gte-large-en-v1.5 or BAAI/bge-large-en-v1.5)
        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(model_id, trust_remote_code=True).to(self.config['device'])

        # Encode directly (GTE/BGE do not require prefixes) and perform L2 normalization
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )

        # Save per model basename to avoid conflicts between different models
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """Get items used for training."""
        items_for_training = set()

        for element in dataset:
            item_seq = element['history']
            if isinstance(item_seq, (list, tuple)):
                items_for_training.update(item_seq)
            else:
                items_for_training.add(item_seq)

        item2id = dataset.id_mapping['item2id']

        # Ensure mask size matches sentence embeddings
        # sent_embs contains items with item_id in [1, n_items-1]
        n_sent_embs = len(item2id) - 1  # Matches range(1, dataset.n_items) in _encode_sent_emb
        logger.info(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        logger.info(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')

        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = item2id[item]
            if 1 <= item_id < len(item2id):  # Ensure item_id is in valid range
                mask[item_id - 1] = True  # Convert to 0-based index

        logger.info(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
        return mask

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """Convert semantic IDs to tokens."""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # Fix: reintroduce offsets to avoid collisions with PAD/BOS
            # Add the corresponding offset to each digit's codebook ID
            tokens = [t + self.sid_offset + d * self.codebook_size 
                     for d, t in enumerate(tokens)]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def fit(self, dataset: AbstractDataset):
        """Initialize tokenizer and generate/load mappings."""
        self.dataset = dataset
        self.item2id = dataset.id_mapping['item2id']
        self.id2item = dataset.id_mapping['id2item']

        # Build cache path - use class name + category

        # If dataset has a category attribute, include it in the path
        if hasattr(dataset, 'category') and dataset.category:
            cache_dir = os.path.join(
                dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset.__class__.__name__, 'processed'
            )

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        # Load semantic IDs (include PCA dim and quantizer tag in filename to avoid config conflicts)
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        quant_tag = self.index_factory + self._get_quant_tag_extra()
        sem_ids_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}.sem_ids'
        )

        # 🚀 New: check whether to force regenerate quantization results
        force_regenerate = self.config.get('force_regenerate_opq', False)

        # Two embedding files: raw and PCA versions, to avoid naming ambiguity/conflicts
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        raw_path = os.path.join(
            cache_dir,
            f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb'
        )
        pca_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}.sent_emb'
        )

        # Prepare sentence embeddings if the quantizer needs them; none mode doesn't
        sent_embs = self._prepare_sentence_embeddings(dataset, raw_path, pca_path)

        # 🚀 Generate or load quantization results
        if force_regenerate or not os.path.exists(sem_ids_path):
            if force_regenerate:
                logger.info(f'[TOKENIZER] Force regenerating quantization results ({self.index_factory})...')
            else:
                logger.info(f'[TOKENIZER] Quantization results not found, generating ({self.index_factory})...')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_ids(sent_embs, sem_ids_path, training_item_mask)
        else:
            logger.info(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')

        logger.info(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        # 🚀 Mapping filenames: reuse the previously built quant_tag
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        fwd_path = os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy')
        inv_path = os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl')

        # 🚀 Fix #1: handle mapping file consistency
        if force_regenerate:
            # When force regenerating, ignore old files so the logic below will re-save them
            fwd_exists = inv_exists = False
            logger.info(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)

        if fwd_exists and inv_exists:
            # ---------- ① Files exist ----------
            logger.info(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')

            # Reconstruct item2tokens mapping
            item_id2tokens = np.load(fwd_path)
            item2tokens = {}
            for iid, toks in enumerate(item_id2tokens):
                if iid == 0:  # Skip PAD row (all zeros)
                    continue
                item2tokens[self.id2item[iid]] = tuple(toks.tolist())

            # Load inverted index
            with open(inv_path, 'rb') as f:
                self.tokens2item = pickle.load(f)

            logger.info(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            # ---------- ② Files absent or force regenerate; need to regenerate ----------
            if force_regenerate:
                logger.info(f'[TOKENIZER] Force regenerate enabled, generating new mappings')
            else:
                logger.info(f'[TOKENIZER] No existing mappings found for {self.n_digit}-digit, will generate new ones')

            # Whether files are missing or force regenerate is enabled, save new item2tokens
            self.item2tokens = item2tokens
            self.tokens2item = self._create_reverse_mapping()
            self._save_mappings()  # Only written to disk when creating new mappings

        # ---- ③ Always attach mapping to instance then return ----
        # Note: in the "files exist" branch, self.item2tokens must be set
        if not hasattr(self, 'item2tokens'):
            self.item2tokens = item2tokens
        return item2tokens

    def _create_reverse_mapping(self):
        """Create a reverse mapping for inference."""
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.item2id[item]
            tokens2item[tuple(tokens)] = item_id
        return tokens2item

    def _save_mappings(self):
        """Save mapping files."""
        # Build cache path - fix: use class name and category

        # If dataset has a category attribute, include it in the path
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(
                self.dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', self.dataset.__class__.__name__, 'processed'
            )

        os.makedirs(cache_dir, exist_ok=True)

        # 🚀 Filenames include: model + PCA + quantizer tag (+ seed/iters) + n_digit, avoiding config conflicts
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        quant_tag = self.index_factory + self._get_quant_tag_extra()
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'

        # Save forward index: item_id → SID tokens
        item_id2tokens = np.zeros((len(self.item2id), self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)

        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)

        # Save inverted index: SID tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)

        logger.info(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        logger.info(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """Encode user history sequence and return a padding mask."""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        history_sid = torch.full(
            (max(len(item_seq), max_len), self.n_digit),
            fill_value=-1,
            dtype=torch.long
        )

        for i, item in enumerate(item_seq):
            if item in self.item2tokens:
                history_sid[i] = torch.tensor(self.item2tokens[item], dtype=torch.long) - \
                    self.digit_vocab_pos(torch.arange(self.n_digit))

        history_mask = (history_sid != -1).any(dim=-1)

        return history_sid, history_mask  # Return lists so datasets.map can tensorize automatically

    def encode_decoder_input(self, target_item):
        """Encode decoder input - consistent with RPG_ED."""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4 token IDs (with offsets)

            # Convert token IDs to codebook IDs
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)

            # decoder input and labels are both codebook IDs
            return codebook_tokens  # [cb0, cb1, cb2, cb3]

        # Unknown item
        return [self.mask_token] * self.n_digit  # length n_digit

    def codebooks_to_item_id(self, cb_ids):
        """Convert a codebook ID sequence to an item_id, validating length."""
        if len(cb_ids) != self.n_digit:
            return None

        # Convert codebook IDs to token IDs
        token_ids = [
            cb_ids[d] + self.sid_offset + d * self.codebook_size
            for d in range(self.n_digit)
        ]

        # Lookup the corresponding item_id
        return self.tokens2item.get(tuple(token_ids))

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenize function - fixes data leakage issues."""
        item_seq = example['history']  # Python list
        target_item = example['target']  # raw string

        history_sid, history_mask = self.encode_history(item_seq)
        codebook_tokens = self.encode_decoder_input(target_item)

        if split == 'train':
            # Encode decoder input during training
            additive = {
                'history_sid': history_sid,  # list
                'history_mask': history_mask,  # list
                'decoder_input_ids': codebook_tokens,  # list
                'decoder_labels': codebook_tokens  # list
            }
        else:
            # Produce ground truth labels for validation/testing
            additive = {
                'history_sid': history_sid,  # list
                'history_mask': history_mask,  # list
                'labels': codebook_tokens  # new: ground truth label sequence
            }

        example.update(additive)
        
        return example

    def tokenize(self, batch, split: str):
        """Tokenize the dataset."""
        return [
            self.tokenize_function(example, split=split)
            for example in batch
        ]

    # ====== New: SID→items mapping helpers ======
    def _sid_tokens_to_cb_tuple(self, tokens):
        """Convert offset SID tokens (length n_digit) to a codebook index tuple (each 0..K-1).

        Example: [sid_offset + 0*K + a, sid_offset + 1*K + b, ...] → (a, b, ...)
        """
        assert len(tokens) == self.n_digit
        cb = []
        for d, tok in enumerate(tokens):
            cb.append(int(tok) - (self.sid_offset + d * self.codebook_size))
        return tuple(cb)

    def _build_cb2items_map(self):
        """Build an inverted index from SID combinations to items based on self.item2tokens.

        Note: supports one-to-many mappings (conflicts are kept as lists).
        """
        cb2items = defaultdict(list)
        for item, toks in self.item2tokens.items():
            cb = self._sid_tokens_to_cb_tuple(toks)
            cb2items[cb].append(item)
        return cb2items

    @property
    def cb2items(self):
        """Lazy cache: build SID→items mapping on first access and cache to _cb2items."""
        if not hasattr(self, "_cb2items") or self._cb2items is None:
            self._cb2items = self._build_cb2items_map()
        return self._cb2items

    def cb_tuple_to_item_ids(self, cb):
        """Given a codebook tuple, return the corresponding list of item_ids (stable insertion order)."""
        items = self.cb2items.get(cb, [])
        out = []
        for it in items:
            iid = self.item2id.get(it, 0)
            if iid > 0:
                out.append(iid)
        return out

    def save(self, path):
        """Save tokenizer state to a file."""
        state = {
            'item2tokens': self.item2tokens,
            'tokens2item': self.tokens2item,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f'[TOKENIZER] Saved tokenizer state to {path}')

    def load(self, path):
        """Load tokenizer state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.item2tokens = state['item2tokens']
        self.tokens2item = state['tokens2item']
        self.config = state['config']
        logger.info(f'[TOKENIZER] Loaded tokenizer state from {path}')
