from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .operator import Operator
import os, logging
from typing import Callable

def callback(cache: tuple[tuple[torch.Tensor]]) -> tuple[torch.Tensor]:
    ks, vs = [], []
    for k, v in cache:
        ks.append(k[0, ...])
        vs.append(v[0, ...])
    ks = torch.stack(ks, dim=0)
    vs = torch.stack(vs, dim=0)
    return ks, vs

class TransformerOperator(Operator):
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
        super().__init__(tokenizer, model)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = callback) -> list[torch.Tensor]:
        """
        Extract kv cache at specified layers
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            tuple[torch.Tensor]: tuple of 2 tensors (n_layers, seqlen, hidden_size)
        """

        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).past_key_values
            cache = [cache[layer] for layer in layers]
            cache = activation_callback(cache)
            return cache
        
        caches = []
        for input in inputs:
            cache = extract_single_line(input)
            caches.append(cache)
        
        return caches

    def store_cache(self, kvs: tuple[tuple[torch.Tensor]], path: str) -> None:
        """
        Store cache to specified path
        Args:
            cache (tuple[tuple[torch.Tensor]]): cache
            path (str): path to store cache
        """
        logger = logging.getLogger(__name__)
        k, v = kvs[0][0], kvs[0][1]     # (n, n_heads, seqlen, headdim)
        if path.endswith(".pt"):
            path = path[:-3]
        out_path = path + "_k.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(k, out_path)
        out_path = path + "_v.pt"
        torch.save(v, out_path)
        logger.info(f"Saved activations to {out_path}")