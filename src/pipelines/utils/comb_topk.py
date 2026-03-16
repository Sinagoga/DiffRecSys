import torch
from typing import Tuple


@torch.no_grad()
def _topk_2d_sum(s: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given two columns of candidate scores, compute pairwise sums and take the global top-k.

    Args:
        s: [B, X, Y], where X/Y are the candidate counts for the two columns
        k: number of top combinations to return

    Returns:
        values: [B, k]
        ix: [B, k], indices for the first column
        iy: [B, k], indices for the second column
    """
    B, X, Y = s.shape
    flat = s.reshape(B, -1)
    top_k = min(k, X * Y)
    vals, idx = torch.topk(flat, k=top_k, dim=-1)
    ix = idx // Y
    iy = idx % Y
    return vals, ix, iy


@torch.no_grad()
def combine_remaining_topk(
    per_digit_logp: torch.Tensor,
    topK_final: int,
    per_digit_topL: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine per-digit log-probabilities for r remaining positions using a top-k strategy:
    - First keep top-L candidates per digit
    - Iteratively merge columns pairwise, truncating to `topK_final` after each merge
      until all columns are merged.

    Args:
        per_digit_logp: [B, r, V], where r is the number of remaining columns and V is codebook size
        topK_final: number of candidates to keep after final combination
        per_digit_topL: number of top candidates to keep per digit (<= V)

    Returns:
        comb_vals: [B, topK_final] combined log-probabilities
        comb_tokens: [B, topK_final, r] selected codebook ids for each column
    """
    B, r, V = per_digit_logp.shape
    assert r >= 1
    L = min(per_digit_topL, V)

    # Take top-L candidates per column
    vals = []
    ids = []
    for i in range(r):
        v, idx = torch.topk(per_digit_logp[:, i], k=L, dim=-1)  # [B, L]
        vals.append(v)
        ids.append(idx)  # codebook ids

    # Iteratively merge columns pairwise
    cur_vals = vals[0]  # [B, L]
    cur_ids = ids[0].unsqueeze(-1)  # [B, L, 1]
    for i in range(1, r):
        s = cur_vals.unsqueeze(2) + vals[i].unsqueeze(1)  # [B, L, L]
        v, ix, iy = _topk_2d_sum(s, k=min(topK_final, L * L))  # [B, K]

        # Backtrack selected token ids
        prev = torch.gather(cur_ids, dim=1, index=ix.unsqueeze(-1).expand(-1, -1, cur_ids.shape[-1]))  # [B, K, i]
        tok_i = torch.gather(ids[i], dim=1, index=iy)  # [B, K]
        cur_ids = torch.cat([prev, tok_i.unsqueeze(-1)], dim=-1)  # [B, K, i+1]
        cur_vals = v

    return cur_vals, cur_ids
