from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
import os
import logging

class RWKVOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.device = device
        super().__init__(tokenizer, model)
    
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
            torch.Tensor: tensor of shape (n_inputs, n_layers, n_heads, head_dim_k, head_dim_v)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).state
            kv = cache[1]
            cache = torch.movedim(kv[..., layers], -1, 0).squeeze(1)     # (n_layers, n_heads, head_dim_k, head_dim_v)
            cache = cache.cpu()
            cache = activation_callback(cache)
            return cache
        
        caches = []
        for input in inputs:
            cache = extract_single_line(input)
            caches.append(cache)
        caches = torch.stack(caches, dim=0)
        
        return caches
    
    def store_cache(self, cache: torch.Tensor, path: str) -> None:
        """
        Store cache to path
        Args:
            cache (torch.Tensor)
            path (str)
        """
        logger = logging.getLogger(__name__)
        if path.endswith(".pt"):
            path = path[:-3]
        path = f"{path}_kv.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(cache, path)
        logger.info(f"Saved activations to {path}")