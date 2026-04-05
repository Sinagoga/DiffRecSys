from typing import List, Dict, Any

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_and_mask_sequences(sequences, padding_value=0):
    """
    Pad sequences to the same length and create a mask.

    Args:
        sequences (list[Tensor]): list of tensors, each of shape (seq_len, feature_dim).
        padding_value (float): value to use for padding.
    Returns:
        padded_sequences (Tensor): tensor of shape (batch_size, max_seq_len, feature_dim), containing the padded sequences.
        mask (Tensor): tensor of shape (batch_size, max_seq_len), containing 1 for valid positions and 0 for padded positions.
    """

    # Pad sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)

    # Create mask
    mask = (padded_sequences != padding_value).any(dim=-1).float()

    return padded_sequences, mask


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


# TODO: This version should works faster, but have not been tested yet.
def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}
    
    histories = [torch.tensor(item["history"], dtype=torch.long) for item in dataset_items]
    batch["history"], batch["history_mask"] = pad_and_mask_sequences(histories, padding_value=0)

    if "decoder_input_ids" in dataset_items[0]:
        batch["decoder_input_ids"] = torch.stack(
            [torch.tensor(item["decoder_input_ids"], dtype=torch.long) for item in dataset_items]
        )
    if "decoder_labels" in dataset_items[0]:
        batch["decoder_labels"] = torch.stack(
            [torch.tensor(item["decoder_labels"], dtype=torch.long) for item in dataset_items]
        )

    return batch
