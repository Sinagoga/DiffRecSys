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
        
        L = self.model.n_digit
        mask_token_id = self.tokenizer.mask_token
        
        x = torch.full((batch_size, L), mask_token_id, dtype=torch.long, device=device)
        
        fast_dllm_cfg = self.config.get('fast_dllm', {})
        T = fast_dllm_cfg.get('steps_per_block', 10)
        strategy = fast_dllm_cfg.get('strategy', 'threshold')
        tau = fast_dllm_cfg.get('threshold', 0.9)
        f = fast_dllm_cfg.get('factor', 0.5)
        use_cache = fast_dllm_cfg.get('use_cache', False)

        past_key_values = None

        for t in range(1, T + 1):
            decode_batch = {
                'decoder_input_ids': x,
                'encoder_hidden': encoder_hidden,
                'mask_positions': (x == mask_token_id) 
            }

            if use_cache:
                outputs = self.model.decode(                # [logits, present_key_values]
                    decode_batch,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs[1]
            else:
                outputs = self.model.decode(decode_batch)   # [logits, present_key_values]
                
            logits = outputs[0]
            
            is_masked = (x == mask_token_id)
            probs = torch.softmax(logits, dim=-1)
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
                x[b_idx, tokens_to_unmask] = b_preds[unmask_mask]
            
            if not (x == mask_token_id).any():
                break

        return x
    
    # def training_step(self, batch, batch_idx):
    #     loss = self.model.calculate_loss(batch)
    #     self.log("train_loss", loss, prog_bar=True, sync_dist=True)

    #     if self.train_metrics:
    #         preds = self.model(batch)
    #         for metric in self.train_metrics.values():
    #             metric.update(preds=preds, **batch)

    #     return loss

    # def training_step(self, batch, batch_idx):
    #     loss = self.model.calculate_loss(batch)
    #     self.log("train_loss", loss, prog_bar=True, sync_dist=True)

    #     # Экономим 95% вычислений: считаем метрики обучения только для каждого 20-го батча
    #     if self.train_metrics and batch_idx % 20 == 0:
    #         preds = self.model(batch)  # Тяжелый вызов генерации
    #         for metric in self.train_metrics.values():
    #             metric.update(preds=preds, **batch)

    #     return loss

    def training_step(self, batch, batch_idx):
        loss = self.model.calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        if self.train_metrics:
            # preds = self.model(batch)
            if self.train_metrics and batch_idx % 20 == 0:
                preds = self.generate(batch, n_return_sequences=self.config.get('n_return_sequences_eval', 10))

                for metric in self.train_metrics.values():
                    # metric.update(preds=preds, **batch)
                    metric.update(preds=preds, labels=batch['decoder_labels'])

        return loss
    
    def on_train_epoch_end(self):
        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                value = metric.compute()
                self.log(f"train_{key}", value, prog_bar=True, sync_dist=True)
                metric.reset()