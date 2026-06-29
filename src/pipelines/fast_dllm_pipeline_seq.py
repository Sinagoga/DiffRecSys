import torch
from src.pipelines.diffusion_pipeline import DiffusionPipeline

class FastDLLMPipeline(DiffusionPipeline):
    def generate(self, batch, n_return_sequences=1, mode="fast_dllm"):
        encoder_hidden = self.model.encode(batch)
        
        if n_return_sequences > 1:
            encoder_hidden = torch.repeat_interleave(encoder_hidden, n_return_sequences, dim=0)
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = torch.repeat_interleave(batch[k], n_return_sequences, dim=0)
                    
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device
        
        fast_dllm_cfg = self.config.get('fast_dllm', {})
        L = self.config.get('answer_length', 128)
        B = fast_dllm_cfg.get('block_size', 32)
        K = (L + B - 1) // B
        T = fast_dllm_cfg.get('steps_per_block', 10)
        strategy = fast_dllm_cfg.get('strategy', 'threshold')
        tau = fast_dllm_cfg.get('threshold', 0.9)
        f = fast_dllm_cfg.get('factor', 0.5)
        use_cache = fast_dllm_cfg.get('use_cache', True)
        use_dual_cache = fast_dllm_cfg.get('use_dual_cache', False)

        # mask_token_id = self.tokenizer.mask_token_id
        mask_token_id = self.tokenizer.mask_token

        prompt = batch.get("prompt_ids", torch.empty((batch_size, 0), dtype=torch.long, device=device))
        prompt_len = prompt.size(1)
        
        masks = torch.full((batch_size, L), mask_token_id, dtype=torch.long, device=device)
        x = torch.cat([prompt, masks], dim=1)
        
        past_key_values = None

        for k in range(1, K + 1):
            s = prompt_len + (k - 1) * B
            e = min(prompt_len + k * B, x.size(1))
            
            for t in range(1, T + 1):
                if use_cache:
                    model_input = x[:, :e] if not use_dual_cache else x

                    batch = {
                        'decoder_input_ids': model_input,
                        'encoder_hidden': encoder_hidden,
                        'mask_positions': (model_input == mask_token_id).nonzero(as_tuple=True)[1].view(batch_size, -1)
                    }

                    outputs = self.model.decode(
                        batch,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    block_logits = logits[:, s:e, :]
                else:
                    batch = {
                        'decoder_input_ids': x,
                        'encoder_hidden': encoder_hidden,
                        'mask_positions': (x == mask_token_id).nonzero(as_tuple=True)[1].view(batch_size, -1)
                    }

                    outputs = self.model.decode(batch)
                    block_logits = outputs.logits[:, s:e, :]

                block_x = x[:, s:e]
                is_masked = (block_x == mask_token_id)
                
                probs = torch.softmax(block_logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)
                
                for b_idx in range(batch_size):
                    b_masked_indices = is_masked[b_idx].nonzero(as_tuple=True)[0]
                    if len(b_masked_indices) == 0:
                        continue
                        
                    b_conf = confidences[b_idx, b_masked_indices]
                    b_preds = preds[b_idx, b_masked_indices]
                    
                    unmask_mask = torch.zeros_like(b_conf, dtype=torch.bool)
                    
                    if strategy == "threshold":
                        unmask_mask = b_conf >= tau
                        if not unmask_mask.any():
                            unmask_mask[torch.argmax(b_conf)] = True
                            
                    elif strategy == "factor":
                        sorted_conf, sorted_idx = torch.sort(b_conf, descending=True)
                        n = 0
                        for i in range(len(sorted_conf)):
                            if (i + 2) * (1 - sorted_conf[i].item()) < f:
                                n = i + 1
                            else:
                                break
                        n = max(n, 1)
                        unmask_mask[sorted_idx[:n]] = True
                    
                    tokens_to_unmask = b_masked_indices[unmask_mask]
                    x[b_idx, s + tokens_to_unmask] = b_preds[unmask_mask]
                
                if not (x[:, s:e] == mask_token_id).any():
                    break

        return x[:, prompt_len:]