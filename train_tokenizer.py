import os
import warnings

import hydra
import lightning as L
from hydra.utils import instantiate

from src.datasets.data_utils import get_datasets
from src.utils.init_utils import setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def _get_tokenizer_state_path(config):
    tokenizer_state_path = config.get("tokenizer_state_path")
    if tokenizer_state_path is not None:
        return tokenizer_state_path
    return os.path.join(config.global_setings.save_dir, "tokenizer_state.pkl")


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    """
    Main script for tokenizer training.
    Fits tokenizer on train split and saves serialized tokenizer state.

    Args:
        config (DictConfig): hydra experiment config.
    """
    L.seed_everything(config.global_setings.seed)

    setup_saving_and_logging(config)

    tokenizer = instantiate(config.tokenizer)
    datasets = get_datasets(config)

    train_dataset = datasets.get("train")
    if train_dataset is None:
        raise ValueError("Training dataset not found for tokenizer training.")

    tokenizer.fit(train_dataset)

    tokenizer_state_path = _get_tokenizer_state_path(config)
    tokenizer_state_dir = os.path.dirname(tokenizer_state_path)
    if tokenizer_state_dir:
        os.makedirs(tokenizer_state_dir, exist_ok=True)

    tokenizer.save(tokenizer_state_path)


if __name__ == "__main__":
    main()
