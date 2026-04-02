import torch
import lightning as L

from src.models.abstract_model import AbstractModel
from src.tokenizers.abstract_tokenizer import AbstractTokenizer

from pipelines.utils.ablate_decode import decode_ablate_confidence
from pipelines.utils.beam import fast_beam_search_for_eval


class DiffusionPipeline(L.LightningModule):
    def __init__(
            self,
            model: AbstractModel,
            tokenizer: AbstractTokenizer,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            evaluator,
            **config,
        ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.config = config

        # When ablation is enabled, automatically inject confidence_s1/s2/s3 modes to ensure evaluation runs three modes
        if self.config.get('ablate_decode', {}).get('enabled', False):
            modes = self.config.get('beam_search_modes', [])

            base = 0
            if 'confidence' in modes:
                base = modes.index('confidence')

            to_add = ['confidence_s1', 'confidence_s2', 'confidence_s3']
            for m in reversed(to_add):
                if m not in modes:
                    modes.insert(base, m)

            self.config['beam_search_modes'] = modes

    def training_step(self, batch, batch_idx):
        loss = self.model.calculate_loss(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds = self.generate(batch, n_return_sequences=self.evaluator.maxk)
        results = self.evaluator.calculate_metrics(preds, batch['labels'])
        
        for key, value in results.items():
            self.log(f"val_{key}", value, prog_bar=True)
        
        return results

    def generate(self, batch, n_return_sequences=1, mode="confidence"):

        # Ensure evaluation mode for inference (disable dropout)
        # obtain encoder outputs
        encoder_hidden = self.model.encode(batch)

        ablate_decode_config = self.config.get('ablate_decode', {})

        # routing: ablation 1/2/3 steps (confidence-only)
        if mode.startswith("confidence_s") and bool(ablate_decode_config.get('enabled', False)):
            try:
                steps = int(mode.split("confidence_s")[-1])
            except Exception:
                steps = int(ablate_decode_config.get('steps_default', 3))
            if steps < 4:
                return decode_ablate_confidence(
                    model=self.model,
                    encoder_hidden=encoder_hidden,
                    tokenizer=self.tokenizer,
                    steps=steps,
                    n_return_sequences=n_return_sequences,
                    vectorized_beam_search=self.config.get('vectorized_beam_search', {}),
                    ablate_decode_config=ablate_decode_config,
                )
            
        # fallback: unknown mode
        if mode not in ("confidence", "random"):
            mode = "confidence"

        # use original 4-step decoding
        return fast_beam_search_for_eval(
            model=self.model,
            encoder_hidden=encoder_hidden,
            beam_size=n_return_sequences,
            tokenizer=self.tokenizer,
            mode="confidence",
            rand_cfg=self.config.get("random_beam", {}),
            config=self.config
        )

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
