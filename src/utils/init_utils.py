import logging
import os
import random
import secrets
import shutil
import string
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Args:
        length (int): length of a string.
    Returns:
        run_id (str): base-36 string with an experiment id.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def saving_init(save_dir, config):
    """
    Initialize saving by getting run_id.

    Args:
        save_dir (Path): path to the directory to log everything:
            logs, checkpoints, config, etc.
        config (DictConfig): hydra config for the current experiment.
    """
    for i, logger_config in enumerate(config.trainer.logger):
        run_id = None
        run_id_param_name = None

        for param_name in ["run_id", "id"]:
            if param_name in logger_config:
                run_id_param_name = param_name
                break
        else:
            continue

        if save_dir.exists():
            if config.global_setings.get("resume_from") is not None:
                saved_config = OmegaConf.load(save_dir / "config.yaml")
                logger_config = saved_config.trainer.logger[i]
                run_id = logger_config.get(run_id_param_name, None)
                print(f"Resuming training from run {run_id}...")
            elif config.global_setings.override:
                print(f"Overriding save directory '{save_dir}'...")
                shutil.rmtree(str(save_dir))
            elif not config.trainer.override:
                raise ValueError(
                    "Save directory exists. Change the name or set override=True"
                )

        if run_id is None:
            run_id = generate_id()

        OmegaConf.set_struct(config, False)
        config.trainer.logger[i][run_id_param_name] = run_id
        OmegaConf.set_struct(config, True)

    save_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, save_dir / "config.yaml")


def setup_saving_and_logging(config):
    """
    Initialize the logger, writer, and saving directory.
    The saving directory is defined by the run_name and save_dir
    arguments of config.writer and config.trainer, respectfully.

    Args:
        config (DictConfig): hydra config for the current experiment.
    Returns:
        logger (Logger): logger that logs output.
    """
    append_mode = config.global_setings.get("resume_from") is not None

    saving_init(Path(config.global_setings.save_dir), config)
    setup_logging(Path(config.global_setings.save_dir), append=append_mode)
