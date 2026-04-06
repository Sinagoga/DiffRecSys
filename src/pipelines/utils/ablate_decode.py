import torch
import torch.nn.functional as F

from src.models.abstract_model import AbstractModel
from src.tokenizers.abstract_tokenizer import AbstractTokenizer

from src.pipelines.utils.comb_topk import combine_remaining_topk


@torch.no_grad()
def decode_ablate_confidence(
    model: AbstractModel,
    encoder_hidden: torch.Tensor,
    tokenizer: AbstractTokenizer,
    steps: int,
    n_return_sequences: int,

    current_split: str = None,
    vectorized_beam_search: dict = {},
    ablate_decode_config: dict = {},
) -> torch.Tensor:
    """
    Confidence-driven ablation decoding with 1/2/3 steps:
        - steps==1: single forward with full mask; combine all columns at once into K full sequences
        - steps==2: determine 1 column first (K parent branches); then combine the remaining columns in one pass
        - steps==3: determine 2 columns (two passes); finally combine the remaining columns in one pass

    Returns:
            codebook id sequences of shape [B, top_k_final, n_digit]
    """
    device = encoder_hidden.device
    B = encoder_hidden.size(0)
    n_digit = model.n_digit
    VOC = model.codebook_size

    # Read shared beam search configuration
    beam_cfg = vectorized_beam_search
    split = current_split or 'val'

    def _as_int(d, k, default):
        try:
            return int(d.get(k, default))
        except Exception:
            return int(default)

    base = beam_cfg.get(split, beam_cfg)
    BEAM_ACT = _as_int(base, 'beam_act', 128)
    TOP_K_FINAL_CFG = _as_int(beam_cfg, 'top_k_final', n_return_sequences)
    TOP_K_FINAL = min(TOP_K_FINAL_CFG, n_return_sequences)

    neg_key = 'neg_inf_fp16' if encoder_hidden.dtype == torch.float16 else 'neg_inf_fp32'
    NEG_INF = float(beam_cfg.get(neg_key, -10000.0 if 'fp16' in neg_key else -1.0e9))

    # Ablation-specific overrides
    ab_cfg = ablate_decode_config
    if 'beam' in ab_cfg and isinstance(ab_cfg['beam'], dict) and split in ab_cfg['beam']:
        BEAM_ACT = int(ab_cfg['beam'][split].get('beam_act', BEAM_ACT))
    per_digit_topL = int(ab_cfg.get('per_digit_topk') or BEAM_ACT)

    # Use internal sentinel -1 to mark "unfilled" positions and avoid conflict with codebook=0
    MASK_ID = -1

    # Step 0: single forward with full mask
    mask_positions = torch.ones(B, n_digit, device=device)
    batch0 = {
        'decoder_input_ids': torch.zeros(B, n_digit, device=device, dtype=torch.long),
        'encoder_hidden': encoder_hidden,
        'mask_positions': mask_positions
    }
    out0_logits, _ = model.decode(batch0, digit=None, use_cache=False)
    logp0 = F.log_softmax(out0_logits, dim=-1)  # [B, n_digit, VOC]

    # steps==1: directly combine all columns
    if steps <= 1:
        comb_vals, comb_tok = combine_remaining_topk(
            per_digit_logp=logp0,
            topK_final=BEAM_ACT,
            per_digit_topL=per_digit_topL,
        )
        seqs = _post_select(comb_tok, comb_vals, tokenizer, TOP_K_FINAL)
        return seqs

    # Common beam containers
    beam_ids = torch.full((B, BEAM_ACT, n_digit), MASK_ID, dtype=torch.long, device=device)
    beam_lp = torch.full((B, BEAM_ACT), NEG_INF, dtype=logp0.dtype, device=device)

    # First pass: determine 1 column (global top-K)
    flat0 = logp0.view(B, -1)
    best0, idx0 = torch.topk(flat0, k=BEAM_ACT, dim=-1)
    d0 = idx0 // VOC
    t0 = idx0 % VOC
    batch_idx = torch.arange(B, device=device).unsqueeze(1)
    beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0)
    beam_ids[batch_idx, beam_idx, d0] = t0
    beam_lp[:, :] = best0

    if steps == 2:
        return _final_combine(model, encoder_hidden, tokenizer, beam_ids, beam_lp,
                              TOP_K_FINAL, BEAM_ACT, per_digit_topL)

    # Second pass: determine another column (2 columns fixed in total)
    mask_pos = (beam_ids == MASK_ID).float()
    dec_in = torch.clamp(beam_ids, min=0).view(-1, n_digit)
    mp_flat = mask_pos.view(-1, n_digit)
    batch1 = {
        'decoder_input_ids': dec_in,
        'encoder_hidden': encoder_hidden.unsqueeze(1).repeat(1, BEAM_ACT, 1, 1).view(-1, encoder_hidden.size(1), encoder_hidden.size(2)),
        'mask_positions': mp_flat
    }
    out1_logits, _ = model.decode(batch1, digit=None, use_cache=False)
    lp1 = F.log_softmax(out1_logits, dim=-1).view(B, BEAM_ACT, n_digit, VOC)

    masked = lp1 + (1.0 - mask_pos.unsqueeze(-1)) * NEG_INF
    cand = beam_lp.unsqueeze(-1).unsqueeze(-1) + masked  # [B,K,D,V]
    flat = cand.view(B, -1)
    best1, idx1 = torch.topk(flat, k=BEAM_ACT, dim=-1)
    parent = idx1 // (n_digit * VOC)
    remain = idx1 % (n_digit * VOC)
    d1 = remain // VOC
    t1 = remain % VOC

    new_ids = beam_ids[batch_idx, parent].clone()
    new_ids.scatter_(2, d1.unsqueeze(-1), t1.unsqueeze(-1))
    beam_ids = new_ids
    beam_lp = best1

    return _final_combine(model, encoder_hidden, tokenizer, beam_ids, beam_lp,
                          TOP_K_FINAL, BEAM_ACT, per_digit_topL)


@torch.no_grad()
def _final_combine(
    model: AbstractModel,
    encoder_hidden,
    tokenizer: AbstractTokenizer,
    beam_ids,
    beam_lp,
    top_k_final: int,
    beam_act: int,
    per_digit_topL: int,
) -> torch.Tensor:
    """
    Given parent beams (with some columns fixed), forward once to obtain log-probs
    for the remaining columns, combine the remaining columns in one pass, add parent
    beam scores, and then take top-K from the combined candidates.
    """
    device = encoder_hidden.device
    B, K, D = beam_ids.shape
    MASK_ID = -1

    mask_pos = (beam_ids == MASK_ID).float()
    dec_in = torch.clamp(beam_ids, min=0).view(-1, D)
    mp_flat = mask_pos.view(-1, D)
    batch = {
        'decoder_input_ids': dec_in,
        'encoder_hidden': encoder_hidden.unsqueeze(1).repeat(1, K, 1, 1).view(-1, encoder_hidden.size(1), encoder_hidden.size(2)),
        'mask_positions': mp_flat
    }
    out_logits, _ = model.decode(batch, digit=None, use_cache=False)
    lp = F.log_softmax(out_logits, dim=-1).view(B, K, D, -1)  # [B,K,D,V]

    r_mask = mask_pos.bool()  # [B,K,D]
    bb = B * K
    V = lp.size(-1)

    # Assemble into [bb, D, V]
    per_bb = torch.stack([lp[:, :, d, :].reshape(bb, V) for d in range(D)], dim=1)  # [bb, D, V]

    # Move remaining-to-fill columns to the front and truncate to r columns
    mask_flat = r_mask.view(bb, D)                  # [bb, D]
    r_per = mask_flat.sum(dim=1)
    assert float(r_per.min().item()) == float(r_per.max().item()), "remaining-column count must be equal across beams"
    r = int(r_per[0].item())

    order = torch.argsort(mask_flat.float(), dim=1, descending=True)           # [bb, D]
    gather_index = order.unsqueeze(-1).expand(-1, -1, V)                       # [bb, D, V]
    per_bb = per_bb.gather(dim=1, index=gather_index)[:, :r, :]                # [bb, r, V]

    # Combine remaining columns
    comb_vals, comb_tok = combine_remaining_topk(per_bb, topK_final=beam_act, per_digit_topL=per_digit_topL)  # [bb,K], [bb,K,r]
    parent_lp = beam_lp.reshape(bb).unsqueeze(1)
    total_lp = (parent_lp + comb_vals).reshape(B, K * beam_act)
    best, besti = torch.topk(total_lp, k=beam_act, dim=-1)

    # Fill selected tokens back into the full sequences
    final = []
    for b in range(B):
        for kidx in range(beam_act):
            parent = (besti[b, kidx] // beam_act).item()
            sid = beam_ids[b, parent].clone()
            maskb = r_mask[b, parent].clone()
            combos = comb_tok.view(B, K, beam_act, -1)[b, parent]
            chosen = combos[besti[b, kidx] % beam_act]
            it = 0
            for d in range(D):
                if maskb[d]:
                    sid[d] = chosen[it]
                    it += 1
            final.append(sid)
    final = torch.stack(final, dim=0).reshape(B, beam_act, D)

    return _post_select(final, best, tokenizer, top_k_final)


def _post_select(seqs: torch.Tensor, scores: torch.Tensor, tokenizer, top_k_final: int) -> torch.Tensor:
    """
    Simple deduplication + legality filtering + keep top `top_k_final`.
    seqs: [B, K, D]
    scores: [B, K]
    """
    B, K, D = seqs.shape
    out = []
    for b in range(B):
        cand = seqs[b]
        lp = scores[b]
        order = torch.argsort(lp, descending=True)
        uniq = []
        for idx in order:
            s = cand[idx]
            legal = tokenizer.codebooks_to_item_id(s.tolist()) is not None
            if not legal:
                continue
            if not any(torch.equal(s, u) for u in uniq):
                uniq.append(s)
            if len(uniq) >= top_k_final:
                break
        if not uniq:
            uniq = [cand[order[0]]]
        while len(uniq) < top_k_final:
            uniq.append(uniq[-1])
        out.append(torch.stack(uniq))
    return torch.stack(out, dim=0)
