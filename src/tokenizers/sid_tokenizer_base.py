from collections import defaultdict
import os
import math
import json
import pickle
from typing import Dict, List, Any

import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from src.datasets.base_dataset import BaseDataset as AbstractDataset
from src.tokenizers.abstract_tokenizer import AbstractTokenizer


def stack_to_tensor(seq, dtype=None):
    """Utility: stack all elements into a tensor; cast dtype if provided."""
    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        return out.to(dtype) if dtype is not None else out
    return torch.tensor(seq, dtype=(dtype if dtype is not None else torch.long))


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function used during training.

    Args:
        batch: A list of dicts containing the following fields:
            - history_sid: history SID sequence [seq_len, n_digit]
            - history_mask: history mask [seq_len]
            - decoder_input_ids: decoder input [n_digit]
            - decoder_labels: decoder labels [n_digit]

    Returns:
        A dict with batched tensors.
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),                      # [B, S, n_digit]
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),  # [B, S]
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),          # [B, n_digit]
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),                # [B, n_digit]
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function used during validation.

    Args:
        batch: A list of dicts containing the following fields:
            - history_sid: history SID sequence [seq_len, n_digit]
            - history_mask: history mask [seq_len]
            - labels: ground truth label sequence [n_digit]

    Returns:
        A dict with batched tensors.
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'labels': stack_to_tensor([b['labels'] for b in batch]),
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function used during testing (same as validation)."""
    return collate_fn_val(batch)


class SIDTokenizerBase(AbstractTokenizer):
    """Base class for SID-based tokenizers (PQ / RQ-KMeans / Random)."""

    def __init__(self, config: dict):
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
        self.log(f'[TOKENIZER] Index factory: {getattr(self, "index_factory", None)}')

        # Create reverse mapping for inference (if not already created)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()

        # Set collate functions
        self.collate_fn = {
            'train': collate_fn_train,
            'val': collate_fn_val,
            'test': collate_fn_test
        }

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
    ) -> np.ndarray | None:
        """Prepare sentence embeddings required by the quantizer.

        Returns:
            np.ndarray or None: embeddings array (N, D) or None if not needed.
        """
        raise NotImplementedError

    def _generate_semantic_ids(
        self,
        sent_embs: np.ndarray | None,
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
        return 3 + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str) -> np.ndarray:
        """Encode sentence embeddings using a Hugging Face SentenceTransformer and normalize vectors."""
        assert self.config['metadata'] == 'sentence', \
            'SIDTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

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

        # Trigger dataset splitting (if not already done)
        split_data = dataset.split()

        # Collect all items from the training split
        if 'train' in split_data:
            train_dataset = split_data['train']
            # train_dataset is a Hugging Face Dataset object
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                # Iterate over all item_seq entries
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)

        # Ensure mask size matches sentence embeddings
        # sent_embs contains items with item_id in [1, n_items-1]
        n_sent_embs = dataset.n_items - 1  # Matches range(1, dataset.n_items) in _encode_sent_emb
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        self.log(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')

        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:  # Ensure item_id is in valid range
                mask[item_id - 1] = True  # Convert to 0-based index

        self.log(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
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
        self.item2id = dataset.item2id
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
                self.log(f'[TOKENIZER] Force regenerating quantization results ({getattr(self, "sid_quantizer", "")})...')
            else:
                self.log(f'[TOKENIZER] Quantization results not found, generating ({getattr(self, "sid_quantizer", "")})...')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_ids(sent_embs, sem_ids_path, training_item_mask)
        else:
            self.log(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
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
            self.log(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)

        if fwd_exists and inv_exists:
            # ---------- ① Files exist ----------
            self.log(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')

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

            self.log(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            # ---------- ② Files absent or force regenerate; need to regenerate ----------
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerate enabled, generating new mappings')
            else:
                self.log(f'[TOKENIZER] No existing mappings found for {self.n_digit}-digit, will generate new ones')

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
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)

        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)

        # Save inverted index: SID tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)

        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        self.log(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """Encode user history sequence."""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        history_sid = []
        for item in item_seq:
            if item in self.item2tokens:
                # Convert offset token IDs to codebook IDs (0..K-1)
                tokens = list(self.item2tokens[item])  # offset token IDs
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
            else:
                # Unknown items are padded with PAD (use -1 as sentinel to avoid confusion with codebook_id=0)
                history_sid.append([-1] * self.n_digit)

        # Pad to fixed length
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)

        return history_sid  # Return lists so datasets.map can tensorize automatically

    def encode_history_with_mask(self, item_seq, max_len=None):
        """Encode user history sequence and return a padding mask."""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]

        history_sid = []
        history_mask = []  # True=valid position, False=PAD position

        for item in item_seq:
            if item in self.item2tokens:
                # Convert offset token IDs to codebook IDs (0..K-1)
                tokens = list(self.item2tokens[item])  # offset token IDs
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
                history_mask.append(True)  # valid position
            else:
                # Unknown items are padded with PAD (use -1 as sentinel to avoid confusion with codebook_id=0)
                history_sid.append([-1] * self.n_digit)
                history_mask.append(False)  # PAD position

        # Pad to fixed length
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
            history_mask.append(False)  # PAD position

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
            decoder_input = codebook_tokens  # [cb0, cb1, cb2, cb3]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3]
        else:
            # Unknown item
            decoder_input = [self.pad_token] * self.n_digit  # length n_digit
            decoder_labels = [self.pad_token] * self.n_digit  # length n_digit

        return decoder_input, decoder_labels

    def decode_tokens_to_item(self, tokens):
        """Decode a token sequence to an item ID."""
        if len(tokens) != self.n_digit:
            return None

        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

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

        history_sid, history_mask = self.encode_history_with_mask(item_seq)
        decoder_input, decoder_labels = self.encode_decoder_input(target_item)

        if split == 'train':
            # Encode decoder input during training
            additive = {
                'history_sid': history_sid,  # list
                'history_mask': history_mask,  # list
                'decoder_input_ids': decoder_input,  # list
                'decoder_labels': decoder_labels  # list
            }
        else:
            # Produce ground truth labels for validation/testing
            additive = {
                'history_sid': history_sid,  # list
                'history_mask': history_mask,  # list
                'labels': decoder_labels  # new: ground truth label sequence
            }

        example.update(additive)

    def tokenize(self, dataset: AbstractDataset):
        """Tokenize the dataset."""
        dataset._index = [
            self.tokenize_function(example, split=dataset.split_name)
            for example in dataset._index
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
        self.log(f'[TOKENIZER] Saved tokenizer state to {path}')

    def load(self, path):
        """Load tokenizer state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.item2tokens = state['item2tokens']
        self.tokens2item = state['tokens2item']
        self.config = state['config']
        self.log(f'[TOKENIZER] Loaded tokenizer state from {path}')
