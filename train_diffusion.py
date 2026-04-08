import warnings

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_datasets, get_dataloaders, move_batch_transforms_to_device
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    """
    Main script for diffusion training. Loads a previously trained tokenizer,
    instantiates model/pipeline/trainer, and runs training/evaluation.

    Args:
        config (DictConfig): hydra experiment config.
    """
    L.seed_everything(config.global_setings.seed)

    project_config = OmegaConf.to_container(config)

    setup_saving_and_logging(config)

    tokenizer = instantiate(config.tokenizer)
    
    # setup data_loader instances
    # batch_transforms should be put on device
    datasets = get_datasets(config)
    
    tokenizer.load(config.global_setings.tokenizer_state_path)

    dataloaders = get_dataloaders(config, datasets, tokenization=tokenizer.tokenize)

    # build model architecture, then print to console
    model = instantiate(
        config.model, 
        vocab_size=tokenizer.vocab_size,
        sid_offset=tokenizer.sid_offset,
        num_classes=config.get("num_classes", 1000)
    )

    # Apply model transforms
    for transform_config in config.global_setings.get("model_transforms", []):
        instantiate(transform_config, model)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            # use text_encoder in metrics
            metrics[metric_type].append(
                instantiate(metric_config)
            )

    optimizer = instantiate(config.training_pipeline.optimizer, model.parameters())
    scheduler = instantiate(config.training_pipeline.scheduler, optimizer)
    training_pipeline = instantiate(
        config.training_pipeline,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    )

    if config.get("transforms", {}).get("batch_transforms", None) is not None:
        batch_transforms = instantiate(config.transforms.batch_transforms)
        training_pipeline.batch_transforms = move_batch_transforms_to_device(
            batch_transforms, 
            config.trainer.get("accelerator", "cuda")
        )

    
    trainer = instantiate(config.trainer)

    for logger in trainer.loggers:
        logger.log_hyperparams(project_config)

    trainer.fit(
        model=training_pipeline,
        train_dataloaders=dataloaders.get("train"),
        val_dataloaders=dataloaders.get("val")
    )

    for analyzer_config in config.get("analyzers", []):
        analyzer = instantiate(analyzer_config.analyzer)
        analyzer.analyze_and_visualize(
            save_path=config.trainer.save_dir,
            model=trainer.model,
        )


if __name__ == "__main__":
    main()
