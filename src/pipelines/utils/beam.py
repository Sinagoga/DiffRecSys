# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


from src.models.abstract_model import AbstractModel
from src.tokenizers.abstract_tokenizer import AbstractTokenizer


def _beam_step_select(mode,
                      logp_matrix,      # [B, act, n_digit*VOC]
                      cur_beam_logp,    # [B, act]
                      beam_ids,         # [B, act, n_digit]  (parent)
                      n_digit, VOC, beam_act,
                      rand_cfg):
    """
    Unified single-step branching selection logic

    Args:
        mode: "confidence" or "random"
        logp_matrix: log-prob matrix for the current step [B, act, n_digit*VOC]
        cur_beam_logp: current beams' log-probabilities [B, act]
        beam_ids: current beam token sequences [B, act, n_digit]
        n_digit: number of digit positions
        VOC: vocabulary size
        beam_act: number of active beams
        rand_cfg: sampling configuration dict

    Returns:
        next_lp: next-step log-probabilities [B, act]
        next_ids: next-step token sequences [B, act, n_digit]
    """
    B = logp_matrix.size(0)

    if mode == "confidence":
        # confidence mode: choose the highest-probability paths
        cand_lp  = cur_beam_logp.unsqueeze(-1) + logp_matrix      # logP
        flat_lp  = cand_lp.view(B, -1)
        best_lp, flat_idx = torch.topk(flat_lp, k=beam_act)       # [B, act]
    else:   # "random"
        # random mode: use temperature and top-p/top-k sampling
        temperature = rand_cfg.get("temperature", 1.0)
        logits = (cur_beam_logp.unsqueeze(-1) + logp_matrix) / temperature      # [B, act, *]

        # top-k truncation
        top_k = rand_cfg.get("top_k")
        if top_k is not None:
            kth_vals, _ = logits.topk(top_k, dim=-1)
            min_valid   = kth_vals[..., -1:].detach()
            logits      = torch.where(logits < min_valid, logits.new_full((), -1e9), logits)

        # top-p (nucleus) sampling
        top_p = rand_cfg.get("top_p")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            # Remove tokens that push cumulative prob over the threshold (keep at least one)
            sorted_indices_to_remove = cumsum_probs > top_p
            # Force-keep the first position (avoid removing all tokens)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Restore boolean mask to original ordering
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = torch.softmax(logits, dim=-1)                   # probabilities
        flat_prob = probs.view(B, -1)
        # Save RNG state to avoid contaminating the global RNG
        original_state = torch.get_rng_state()
        try:
            # Fix RNG seed (optional)
            seed = rand_cfg.get("seed")
            if seed is not None:
                torch.manual_seed(seed)

            flat_idx = torch.multinomial(flat_prob, beam_act, replacement=False)  # [B, act]
            idx_rows = torch.arange(B, device=flat_idx.device).unsqueeze(1)
            best_lp  = logits.view(B, -1)[idx_rows, flat_idx]        # corresponding log probability
        finally:
            # restore RNG state
            torch.set_rng_state(original_state)
    # -------------------------------------------------------------------------

    parent   = flat_idx // (n_digit * VOC)
    remain   = flat_idx %  (n_digit * VOC)
    d_pos    = remain // VOC
    tok      = remain %  VOC

    batch_idx = torch.arange(B, device=beam_ids.device).unsqueeze(1)
    next_ids  = beam_ids[batch_idx, parent].clone()
    next_ids.scatter_(2, d_pos.unsqueeze(-1), tok.unsqueeze(-1))
    return best_lp, next_ids


def expand_cross_kv_for_beams(initial_kv_cache, beam_size):
    """
    Copy the cross-attention KV from the first step to each beam, keeping self-KV as None.
    This allows DecoderBlock self-attention KV to continue accumulating while avoiding
    recomputation of cross-attention KV.

    Args:
        initial_kv_cache: initial KV cache
        beam_size: beam size

    Returns:
        expanded KV cache
    """
    if initial_kv_cache is None:
        return None

    expanded = []
    for layer_cache in initial_kv_cache:
        if layer_cache is None:
            expanded.append(None)
            continue

        self_kv, cross_kv = layer_cache        # self_kv present only in the first step; later steps accumulate via cache
        if cross_kv is not None:
            k, v = cross_kv                    # [B, S, d]
            k = k.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *k.shape[1:])
            v = v.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *v.shape[1:])
            cross_kv = (k, v)
        # set self_kv to None to avoid broadcasting first-step decoder tokens
        expanded.append((None, cross_kv))
    return expanded



def iterative_mask_decode(
        model: AbstractModel,
        encoder_hidden: torch.Tensor,
        n_return_sequences: int = 1,
        tokenizer: AbstractTokenizer = None,
        mode: str = "confidence",
        rand_cfg: dict = None,
        config: dict = {},
    ):
    """
    Vectorized iterative mask-filling decoding that removes Python-loop bottlenecks.

    Args:
        model: DIFF_GRM model
        encoder_hidden: encoder outputs [B, seq_len, emb_dim]
        n_return_sequences: number of sequences to return (may be capped by top_k_final)
        tokenizer: tokenizer instance
        mode: "confidence" or "random"
        rand_cfg: sampling configuration dict

    Returns:
        generated_sequences: [B, top_k_final, n_digit]
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size
    
    # Load vectorized beam search parameters from config (supports split-specific settings)
    if 'vectorized_beam_search' not in config:
        # require configuration to be present (no fallback)
        raise ValueError("Missing 'vectorized_beam_search' configuration in config")
    
    beam_config = config['vectorized_beam_search']

    # get current split (default: 'val')
    split = config.get("current_split", "val")   # "val" / "test"

    # check for split-specific configuration (three supported formats)
    if split in beam_config:                           # check split-specific first
        BEAM_ACT = int(beam_config[split]["beam_act"])
        BEAM_MAX = int(beam_config[split]["beam_max"])
    elif isinstance(beam_config.get("beam_act"), dict): # support alternate format: beam_act is a dict
        BEAM_ACT = int(beam_config["beam_act"].get(split,
                                                    beam_config["beam_act"]["val"]))
        BEAM_MAX = int(beam_config["beam_max"].get(split,
                                                    beam_config["beam_max"]["val"]))
    else:                                              # fallback to global settings
        BEAM_ACT = int(beam_config["beam_act"])
        BEAM_MAX = int(beam_config["beam_max"])

    TOP_K_FINAL = min(int(beam_config['top_k_final']), n_return_sequences)
    # Ensure NEG_INF values are floats to avoid YAML string issues
    NEG_INF_FP32 = float(beam_config['neg_inf_fp32'])
    NEG_INF_FP16 = float(beam_config['neg_inf_fp16'])
    # ensure BEAM_ACT does not exceed BEAM_MAX
    assert BEAM_ACT <= BEAM_MAX, "beam_act should not exceed beam_max"
    
    # ---------- ① Parse beam size (special handling for random mode) ----------
    if mode == "random":
        # If random_beam specifies beam_act/beam_max, override them
        rb_cfg = config.get("random_beam", {})
        BEAM_ACT = int(rb_cfg.get("beam_act", BEAM_ACT))
        BEAM_MAX = int(rb_cfg.get("beam_max", BEAM_MAX))
        # Ensure BEAM_ACT does not exceed BEAM_MAX
        assert BEAM_ACT <= BEAM_MAX, "random_beam.beam_act should not exceed random_beam.beam_max"
    
    # ---------- ② Randomize column order once (random mode only) ----------
    decode_order = None
    if mode == "random":
        # 🚀 Fix: save current RNG state to avoid contaminating training
        original_state = torch.get_rng_state()
        try:
            seed = config.get("random_beam", {}).get("seed")
            if seed is not None:
                torch.manual_seed(seed)
            decode_order = torch.randperm(n_digit).tolist()      # e.g. [1,5,3,7,0,2,6,4]
            if batch_size == 1:  # only print for single-sample to avoid multi-worker spam
                print(f"[RANDOM_BEAM] 🎲 Decode order: {decode_order}")
        finally:
            # 🚀 Restore original RNG state
            torch.set_rng_state(original_state)
    
    # Constants
    MASK_ID = tokenizer.mask_token if tokenizer is not None else -1
    VOC = codebook_size
    
    # Reduce log noise
    if batch_size == 1:  # only print for single-sample to avoid multi-worker spam
        print(f"[VECTORIZED_BEAM] 🚀 Using optimized beam search:")
        print(f"[VECTORIZED_BEAM] BEAM_ACT: {BEAM_ACT}, BEAM_MAX: {BEAM_MAX}, TOP_K_FINAL: {TOP_K_FINAL}")
    
    # Step 0: All-mask prediction to obtain probabilities for all positions
    with torch.no_grad():
        # Build mask_positions: 1 indicates masked
        mask_positions = torch.ones(batch_size, n_digit, device=device)
        
        # Build batch
        batch_dict = {
            'decoder_input_ids': torch.zeros(batch_size, n_digit, device=device, dtype=torch.long),
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions
        }
        
        # Forward pass - enable KV cache to speed subsequent inference
        all_logits, initial_kv_cache = model.decode(batch_dict, digit=None, use_cache=True) # [B, n_digit, codebook_size]; save initial KV cache
        
        # Compute log probabilities
        all_log_probs = F.log_softmax(all_logits, dim=-1)  # [B, n_digit, codebook_size]
        
        if mode == "random":
            # === random mode: look at the first column only ===
            first_col = decode_order[0]
            probs_col = all_log_probs[:, first_col, :]          # [B, VOC]
            top_k_probs, top_k_idx = torch.topk(probs_col, k=BEAM_ACT, dim=-1)  # [B, BEAM_ACT]
            
            # Parse positions and tokens
            first_col_tensor = torch.full((batch_size, BEAM_ACT), first_col, device=device, dtype=torch.long)
            first_token = top_k_idx
        else:
            # === confidence mode: global top-k ===
            # Concatenate probabilities from all positions: [B, n_digit * codebook_size]
            flattened_log_probs = all_log_probs.view(batch_size, -1)
            
            # Take top BEAM_ACT candidates
            top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=BEAM_ACT)
            
            # Parse positions and tokens
            first_col_tensor = top_k_indices // VOC      # which digit [B, BEAM_ACT]
            first_token = top_k_indices % VOC     # ID within codebook [B, BEAM_ACT]
        
        # 🚀 Pre-allocate fixed-size beam tensors (key optimization)
        beam_ids = torch.full((batch_size, BEAM_MAX, n_digit), MASK_ID, 
                             dtype=torch.long, device=device)
        
        # Determine NEG_INF value
        NEG_INF = NEG_INF_FP16 if top_k_probs.dtype == torch.float16 else NEG_INF_FP32
        beam_logp = torch.full((batch_size, BEAM_MAX), NEG_INF, 
                              dtype=top_k_probs.dtype, device=device)
        
        # Fill results for the first step
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
        beam_indices = torch.arange(BEAM_ACT, device=device).unsqueeze(0)     # [1, BEAM_ACT]
        
        beam_ids[batch_indices, beam_indices, first_col_tensor] = first_token
        beam_logp[:, :BEAM_ACT] = top_k_probs
        
        # 🚀 Fix: expand to BEAM_MAX to ensure adequate capacity
        encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, BEAM_MAX, 1, 1)
        encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))
        
        # After Step 0, produce a broadcasted cache for reuse
        kv_cache_for_act = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)
        kv_cache_final = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)  # for final step
    
    # Steps 1-2: Vectorized beam expansion (eliminate Python loops)
    if mode == "random":
        # === random mode: loop according to decode_order ===
        for step, cur_col in enumerate(decode_order[1:], 1):
            with torch.no_grad():
                # Use only the first BEAM_ACT active beams
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]
                
                # Build mask_positions for current state
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]
                
                # Reshape into decoder input format
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                
                # 🚀 Use pre-generated KV cache to enable real cache reuse
                expanded_kv_cache = kv_cache_for_act
                
                # Build batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # only use the first BEAM_ACT portion
                    'mask_positions': mask_pos_flat
                }
                
                # Forward pass
                all_logits, _ = model.decode(batch_dict, digit=None, past_key_values=expanded_kv_cache, use_cache=True) # [B*BEAM_ACT, n_digit, codebook_size]
                
                # Reshape into beam dimension
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
                
                # 🚀 Vectorized mask handling (core optimization)
                all_log_probs = F.log_softmax(all_logits, dim=-1)
                
                # Only consider masked positions
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF
                
                # === random mode: only look at current column ===
                logits = masked_log_probs[:, :, cur_col, :]                     # [B, BEAM_ACT, VOC]
                
                joint_lp = logits + active_beam_logp.unsqueeze(-1)              # [B, BEAM_ACT, VOC]
                flat_lp  = joint_lp.view(batch_size, -1)                        # [B, BEAM_ACT*VOC]
                best_lp, flat_idx = torch.topk(flat_lp, k=BEAM_ACT)            # ← top-k, no sampling
                
                # Parse indices
                parent_beam_ids = flat_idx // VOC                               # [B, BEAM_ACT]
                token_ids = flat_idx % VOC                                      # [B, BEAM_ACT]
                
                # Update beams
                batch_range = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
                new_beam_ids = active_beam_ids[batch_range, parent_beam_ids]        # [B, BEAM_ACT, n_digit]
                new_beam_ids.scatter_(2, torch.full((batch_size, BEAM_ACT), cur_col, device=device, dtype=torch.long).unsqueeze(-1), token_ids.unsqueeze(-1))
                
                # Update beam state
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_lp
                
                # Clear invalid beams (maintain BEAM_MAX size)
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    else:
        # === confidence mode: original logic ===
        for step in range(1, n_digit - 1):
            with torch.no_grad():
                # Use only the first BEAM_ACT active beams
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]
                
                # Build mask_positions for current state
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]
                
                # Reshape into decoder input format
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                
                # 🚀 Use pre-generated KV cache to enable real cache reuse
                expanded_kv_cache = kv_cache_for_act
                
                # Build batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # only use the first BEAM_ACT portion
                    'mask_positions': mask_pos_flat
                }
                
                # Forward pass
                all_logits, _ = model.decode(batch_dict, digit=None, past_key_values=expanded_kv_cache, use_cache=True)  # [B*BEAM_ACT, n_digit, codebook_size]
                
                # Reshape into beam dimension
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
                
                # 🚀 Vectorized mask handling (core optimization)
                all_log_probs = F.log_softmax(all_logits, dim=-1)
                
                # Only consider masked positions
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF
                
                # Concatenate all candidate possibilities: [B, BEAM_ACT, n_digit * codebook_size]
                flattened_log_probs = masked_log_probs.view(batch_size, BEAM_ACT, -1)
                
                # 🚀 Use unified branching selection logic
                best_logprobs, new_beam_ids = _beam_step_select(
                    mode=mode,
                    logp_matrix=flattened_log_probs,          # [B, act, n_digit*VOC]
                    cur_beam_logp=active_beam_logp,           # [B, act]
                    beam_ids=active_beam_ids,                 # [B, act, n_digit]
                    n_digit=n_digit, VOC=VOC, beam_act=BEAM_ACT,
                    rand_cfg=rand_cfg or {}
                )
                
                # Update beam state
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_logprobs
                
                # Clear invalid beams (maintain BEAM_MAX size)
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    
    # Final step: fill last position and choose top-K
    with torch.no_grad():
        if mode == "random":
            # === random mode: positions already filled via loop, use current results ===
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            final_beam_logp = beam_logp[:, :BEAM_ACT]
        else:
            # === confidence mode: need to fill the last position ===
            # Only handle the first BEAM_ACT beams
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            active_beam_logp = beam_logp[:, :BEAM_ACT]

            # Find the last MASK position for each beam
            mask_positions = (active_beam_ids == MASK_ID).float()

            # Build decoder input
            decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
            mask_pos_flat = mask_positions.view(-1, n_digit)

            # Use pre-generated KV cache for the final step
            final_expanded_kv_cache = kv_cache_final

            batch_dict = {
                'decoder_input_ids': decoder_input,
                'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # only use the first BEAM_ACT portion
                'mask_positions': mask_pos_flat
            }

            # Get logits for all positions
            all_logits, _ = model.decode(batch_dict, digit=None, past_key_values=final_expanded_kv_cache, use_cache=True)  # [B*BEAM_ACT, n_digit, codebook_size]

            # Reshape and compute log probabilities
            all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
            all_log_probs = F.log_softmax(all_logits, dim=-1)

            # Find the final position each beam needs to fill
            last_mask_pos = torch.argmax(mask_positions.float(), dim=-1)  # [B, BEAM_ACT]

            # Select the best token for the corresponding position for each beam
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, BEAM_ACT)
            beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0).expand(batch_size, -1)

            final_logits = all_log_probs[batch_idx, beam_idx, last_mask_pos]  # [B, BEAM_ACT, codebook_size]
            best_token_logprobs, best_tokens = torch.max(final_logits, dim=-1)  # [B, BEAM_ACT]

            # Update the final token
            active_beam_ids.scatter_(2, last_mask_pos.unsqueeze(-1), best_tokens.unsqueeze(-1))
            final_beam_logp = active_beam_logp + best_token_logprobs
        
        # 🚀 Flexible deduplication strategies
        dedup_strategy = "simple"  # default to simple deduplication
        if 'dedup_strategy' in config:
            dedup_strategy = config['dedup_strategy']
        
        if dedup_strategy == "none":
            # Strategy 1: No deduplication, directly select top-K
            top_logprobs, top_indices = torch.topk(final_beam_logp, k=min(TOP_K_FINAL, BEAM_ACT), dim=-1)
            batch_range = torch.arange(batch_size, device=device).unsqueeze(1)
            final_sequences = active_beam_ids[batch_range, top_indices]  # [B, TOP_K_FINAL, n_digit]

            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} sequences (no deduplication)")
                
        elif dedup_strategy == "simple":
            # Strategy 2: Simple dedup + legality check (improved)
            # ① requires tokenizer passed in
            assert tokenizer is not None, "tokenizer is required for legality check"
            
            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]
                
                # Sort by probability, then simple dedup + legality check
                sorted_indices = torch.argsort(batch_logprobs, descending=True)
                unique_sequences = []
                
                for idx in sorted_indices:
                    seq = batch_sequences[idx]
                    # --------- Added: legality check ----------
                    is_legal = tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                    if not is_legal:
                        continue  # skip illegal sequences
                    # ------------------------------------
                    is_duplicate = any(torch.equal(seq, existing) for existing in unique_sequences)
                    if not is_duplicate:
                        unique_sequences.append(seq)
                        if len(unique_sequences) >= TOP_K_FINAL:
                            break
                            
                # Fill missing slots (ensure filled sequences are legal)
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # If a legal sequence exists, repeat the last one
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # If no legal sequence, find a legal filler
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # If still none found, use the first one (may be illegal but better than crashing)
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])
                
                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)
            
            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (simple deduplication + legality check)")
                
        else:  # weighted
            # Strategy 3: Probability-weighted dedup + legality check (improved)
            # ① requires tokenizer passed in
            assert tokenizer is not None, "tokenizer is required for legality check"
            
            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]
                
                # Build mapping from sequence to probability, accumulate duplicates (only legal sequences)
                seq_to_logprob = {}
                for i in range(BEAM_ACT):
                    seq_tuple = tuple(batch_sequences[i].cpu().tolist())
                    # --------- Added: legality check ----------
                    is_legal = tokenizer.codebooks_to_item_id(list(seq_tuple)) is not None
                    if not is_legal:
                        continue  # skip illegal sequences
                    # ------------------------------------
                    if seq_tuple in seq_to_logprob:
                        # Duplicate sequences: accumulate probabilities with log-sum-exp (more stable)
                        seq_to_logprob[seq_tuple] = torch.logaddexp(
                            seq_to_logprob[seq_tuple], 
                            batch_logprobs[i]
                        )
                    else:
                        seq_to_logprob[seq_tuple] = batch_logprobs[i]
                
                # Sort by accumulated probability
                sorted_items = sorted(seq_to_logprob.items(), 
                                    key=lambda x: x[1].item(), reverse=True)
                
                # Select top TOP_K_FINAL unique sequences (already sorted by weighted probability)
                unique_sequences = []
                for seq_tuple, _ in sorted_items[:TOP_K_FINAL]:
                    seq_tensor = torch.tensor(seq_tuple, device=device, dtype=torch.long)
                    unique_sequences.append(seq_tensor)
                
                # Fill missing slots (ensure filled sequences are legal)
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # If a legal sequence exists, repeat the last one
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # If no legal sequence, find a legal filler
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # If still none found, use the first one (may be illegal but better than crashing)
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])
                
                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)
            
            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (probability-weighted deduplication + legality check)")
    
    # ------- Compute statistics for the current batch -------
    if tokenizer is not None:  # no longer limited to batch_size==1
        # Fix legality ratio calculation: use number of sequences as denominator, not tokens
        total_seqs = final_sequences.numel() // n_digit
        legal_final = sum(tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                          for seq in final_sequences.view(-1, n_digit))
        final_legal_ratio = legal_final / total_seqs

        # Fix duplicate ratio calculation: use correct formula
        unique_seqs = len({tuple(seq.tolist()) for seq in final_sequences.view(-1, n_digit)})
        duplicate_ratio = 1 - unique_seqs / total_seqs

        # Return statistics for evaluator use instead of printing directly
        return final_sequences, final_legal_ratio, duplicate_ratio
    # --------------------------------
    
    return final_sequences


def fast_beam_search_for_eval(
        model,
        encoder_hidden,
        beam_size=10,
        tokenizer=None,
        mode="confidence",
        rand_cfg=None,
        config: dict = {},
    ):
    """
    Fast vectorized beam search for evaluation.
    Uses the same strategy as TensorFlow: first 3 steps use fixed 512 beams, then take top-K.

    Args:
        model: DIFF_GRM model
        encoder_hidden: Encoder outputs [batch_size, seq_len, hidden_dim]
        beam_size: final beam size (may be overridden by TOP_K_FINAL)
        max_len: maximum generation length
        tokenizer: Tokenizer
        mode: "confidence" or "random"
        rand_cfg: random sampling config dict

    Returns:
        torch.Tensor: generated token sequences [batch_size, TOP_K_FINAL, max_len]
    """
    # Directly call the vectorized iterative_mask_decode
    result = iterative_mask_decode(
        model=model,
        encoder_hidden=encoder_hidden,
        n_return_sequences=beam_size,
        tokenizer=tokenizer,
        mode=mode,
        rand_cfg=rand_cfg or {},
        config=config
    )
    
    # Handle return: may be tuple (sequences + stats) or sequences only
    if isinstance(result, tuple):
        return result[0]  # return sequences only
    else:
        return result


 