import os
import warnings

import hydra
import lightning as L
from hydra.utils import instantiate

from src.datasets.data_utils import get_datasets
from src.utils.init_utils import setup_saving_and_logging

@hydra.main(version_base=None, config_path="config", config_name="train")

def main(config):
    """
    Main script for tokenizer training.
    Fits tokenizer on train split and saves serialized tokenizer state.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # L.seed_everything(config.global_setings.seed)

    setup_saving_and_logging(config)

    datasets = get_datasets(config)


if __name__ == "__main__":
    main()