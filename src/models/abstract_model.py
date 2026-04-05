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
        """
        Calculate the loss for the given batch.
        
        Args:
            batch (dict): A dictionary containing the input data and labels.

        Returns:
            torch.Tensor: The calculated loss for the batch.
        """

        raise NotImplementedError('calculate_loss method must be implemented.')

    def encode(self, batch: dict) -> torch.Tensor:
        """
        Encode the input batch and return the encoded representation.

        Purpose of this method is to encode context or a history to save up computation time during beam search.

        Args:
            batch (dict): A dictionary containing the input data.

        Returns:
            torch.Tensor: The encoded representation of the input batch.
        """
        raise NotImplementedError('predict method must be implemented.')
    
    def decode(self, batch: dict, digit=None, past_key_values=None, use_cache=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the input batch and return the decoded representation and updated past key values.

        Args:
            batch (dict): A dictionary containing the input data. Should contain at least 'encoder_hidden' from this model and 'mask_positions'.
            digit (torch.Tensor, optional): The digit to decode.
            past_key_values (Tuple[torch.Tensor, torch.Tensor], optional): The past key values for caching.
            use_cache (bool): Whether to use caching.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The decoded representation of the input batch and the updated past key values.
        """

        raise NotImplementedError('decode method must be implemented.')