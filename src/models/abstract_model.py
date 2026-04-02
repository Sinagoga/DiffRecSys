from typing import Tuple


import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    def __init__(
        self,
        config: dict,
    ):
        super(AbstractModel, self).__init__()

        self.config = config

    def calculate_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError('calculate_loss method must be implemented.')

    def encode(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError('predict method must be implemented.')
    
    def decode(self, batch: dict, digit=None, past_key_values=None, use_cache=False) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('decode method must be implemented.')