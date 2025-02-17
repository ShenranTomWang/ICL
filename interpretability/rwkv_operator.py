from transformers import AutoTokenizer
import torch
from typing import Callable
from .operator import Operator

class RWKVOperator(Operator):
    def __init__(self, tokenizer: AutoTokenizer, model):
        self.model = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.device = model.device
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> list[torch.Tensor]:
        """
        TODO: implement this method
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            list[torch.Tensor]: list of tensors (n_layers, seqlen, hidden_size)
        """
        raise NotImplementedError("Method not implemented")
    
    def store_cache(self, cache: tuple[tuple[torch.Tensor]], path: str) -> None:
        """
        Store cache to path
        Args:
            cache (tuple[tuple[torch.Tensor]])
            path (str)
        """
        raise NotImplementedError("Method not implemented")