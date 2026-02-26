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
