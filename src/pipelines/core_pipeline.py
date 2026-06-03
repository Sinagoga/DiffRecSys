import torch
import math
from src.pipelines.diffusion_pipeline import DiffusionPipeline

class CoRePipeline(DiffusionPipeline):
    def generate(self, batch, n_return_sequences=1, mode="core"):
        encoder_hidden = self.model.encode(batch)
        
        if n_return_sequences > 1:
            encoder_hidden = torch.repeat_interleave(encoder_hidden, n_return_sequences, dim=0)
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = torch.repeat_interleave(batch[k], n_return_sequences, dim=0)
                    
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device
        
        core_cfg = self.config.get('core_remasking', {})
        N = core_cfg.get('diffusion_steps', 128)
        gamma_s = core_cfg.get('gamma_s', 0.25)
        gamma_e = core_cfg.get('gamma_e', 0.75)
        E = core_cfg.get('revision_interval', 8)
        m = core_cfg.get('candidate_size', 32)
        k_rm = core_cfg.get('remasking_limit', 1)
        
        L = self.model.n_digit
        mask_token_id = self.tokenizer.mask_token
        
        y = torch.full((batch_size, L), mask_token_id, dtype=torch.long, device=device)
        
        for t in range(1, N + 1):
            decode_batch = {
                'decoder_input_ids': y,
                'encoder_hidden': encoder_hidden,
                'mask_positions': (y == mask_token_id)
            }
            outputs = self.model.decode(decode_batch)
            logits = outputs[0]
            probs = torch.softmax(logits, dim=-1)
            
            k_t = max(1, math.ceil(L / N)) 
            
            in_window = (gamma_s <= t / N < gamma_e)
            is_revision_step = (t % E == 0)
            
            if in_window and is_revision_step and k_rm > 0:
                is_unmasked = (y != mask_token_id)
                
                top2_v, _ = torch.topk(probs, 2, dim=-1)
                margins = top2_v[:, :, 0] - top2_v[:, :, 1]
                margins[~is_unmasked] = float('inf')
                
                S_t_indices = []
                for b_idx in range(batch_size):
                    num_unmasked = is_unmasked[b_idx].sum().item()
                    actual_m = min(m, num_unmasked)
                    if actual_m > 0:
                        _, min_margin_idx = torch.topk(margins[b_idx], actual_m, largest=False)
                        S_t_indices.append(min_margin_idx)
                    else:
                        S_t_indices.append(torch.empty(0, dtype=torch.long, device=device))
                
                y_tilde = y.clone()
                for b_idx in range(batch_size):
                    y_tilde[b_idx, S_t_indices[b_idx]] = mask_token_id
                
                decode_batch_tilde = {
                    'decoder_input_ids': y_tilde,
                    'encoder_hidden': encoder_hidden,
                    'mask_positions': (y_tilde == mask_token_id)
                }
                outputs_tilde = self.model.decode(decode_batch_tilde)
                logits_tilde = outputs_tilde[0]
                probs_tilde = torch.softmax(logits_tilde, dim=-1)
                
                for b_idx in range(batch_size):
                    b_S_t = S_t_indices[b_idx]
                    if len(b_S_t) == 0:
                        continue
                        
                    original_tokens = y[b_idx, b_S_t]
                    prob_i = probs_tilde[b_idx, b_S_t].gather(1, original_tokens.unsqueeze(1)).squeeze(1)
                    instability = -torch.log(prob_i + 1e-9)
                    
                    actual_k_rm = min(k_rm, len(b_S_t))
                    _, max_inst_idx = torch.topk(instability, actual_k_rm)
                    I_t = b_S_t[max_inst_idx]
                    
                    new_tokens = torch.argmax(probs_tilde[b_idx, I_t], dim=-1)
                    y[b_idx, I_t] = new_tokens
            
            for b_idx in range(batch_size):
                b_masked = (y[b_idx] == mask_token_id)
                if not b_masked.any():
                    continue
                    
                b_mask_probs, b_mask_preds = torch.max(probs[b_idx], dim=-1)
                b_mask_probs[~b_masked] = -1.0
                
                actual_k_t = min(k_t, b_masked.sum().item())
                _, topk_idx = torch.topk(b_mask_probs, actual_k_t)
                
                y[b_idx, topk_idx] = b_mask_preds[topk_idx]
                
        return y