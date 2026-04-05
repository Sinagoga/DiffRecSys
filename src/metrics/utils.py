import torch

def calculate_pos_index(preds: torch.Tensor, labels: torch.Tensor, maxk: int) -> torch.Tensor:
    """
    Calculate a boolean tensor indicating whether the predictions are correct for each example in the batch.
    Args:
        preds (torch.Tensor): A tensor of shape (batch_size, maxk) containing the predicted token IDs for each example.
        labels (torch.Tensor): A tensor of shape (batch_size, seq_len) containing the true token IDs for each example.
        maxk (int): The maximum number of predictions to consider for each example.
    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size, maxk) where True indicates a correct prediction.
    """
    labels = labels.unsqueeze(1)
    preds = preds[:, :maxk]
    pos_index = (preds == labels).all(dim=2)
    return pos_index

def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculate Recall@K for a batch of predictions.
    Args:
        pos_index (torch.Tensor): A boolean tensor of shape (batch_size, maxk) where True indicates a correct prediction.
        k (int): The value of K for Recall@K.
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the Recall@K for each example in the batch.
    """

    return pos_index[:, :k].any(dim=1)

def ndcg_at_k(pos_index: torch.Tensor, k: int, use_only_first_hit: bool = False) -> torch.Tensor:
    """
    Calculate NDCG@K for a batch of predictions.
    Args:
        pos_index (torch.Tensor): A boolean tensor of shape (batch_size, maxk) where True indicates a correct prediction.
        k (int): The value of K for NDCG@K.
        use_only_first_hit (bool): Whether to only consider the first hit for NDCG calculation.
    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the NDCG@K for each example in the batch.
    Note:
        This implementation assumes that there is at most one relevant item per example.
    """
    B, maxk = pos_index.shape

    ranks = torch.arange(maxk, device=pos_index.device)
    dcg_weights = 1.0 / torch.log2(ranks + 2)

    if not use_only_first_hit:
        # Normal colculation of DCG for one relevant item (idcg = 1)
        dcg_weights = torch.where(pos_index, dcg_weights, 0)
        dcg_scores = dcg_weights[:, :k].sum(dim=1)
    else:
        # If we only consider the first hit, we need to find the position of the first hit and assign the corresponding DCG weight to it.
        valid_mask = pos_index[:, :k].any(dim=1)
        first_hit_positions = pos_index.int().argmax(dim=1)
        dcg_scores = torch.where(valid_mask, dcg_weights[first_hit_positions], 0.0)
    
    return dcg_scores
