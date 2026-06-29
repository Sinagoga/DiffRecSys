from typing import List, Dict, Any

import torch


def stack_to_tensor(seq, dtype=None):
    """Utility: stack all elements into a tensor; cast dtype if provided."""
    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
    else:
        out = torch.tensor(seq, dtype=dtype)
    return out


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function used during training.

    Args:
        batch: A list of dicts containing the following fields:
            - history_sid: history SID sequence [seq_len, n_digit]
            - history_mask: history mask [seq_len]
            optional:
            - decoder_input_ids: decoder input [n_digit]
            - decoder_labels: decoder labels [n_digit]
            - labels: ground truth label sequence [n_digit]

    Returns:
        A dict with batched tensors.
    """
    out = {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),                      # [B, S, n_digit]
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),  # [B, S]
    }
    if 'decoder_input_ids' in batch[0]:
        out['decoder_input_ids'] = stack_to_tensor([b['decoder_input_ids'] for b in batch])     # [B, n_digit]
    if 'decoder_labels' in batch[0]:
        out['decoder_labels'] = stack_to_tensor([b['decoder_labels'] for b in batch])           # [B, n_digit]
    if 'labels' in batch[0]:
        out['labels'] = stack_to_tensor([b['labels'] for b in batch])                           # [B, n_digit]
    return out
