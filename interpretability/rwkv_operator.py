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
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            tuple[torch.Tensor]: (n_inputs, n_layers, hidden_size), (n_inputs, n_layers, n_heads, head_dim, head_dim), (n_inputs, n_layers, hidden_size)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).state
            x, kv, ffn = cache[0], cache[1], cache[2]
            x = torch.movedim(x[..., layers], -1, 0).squeeze(1).cpu()       # (n_layers, hidden_size)
            kv = torch.movedim(kv[..., layers], -1, 0).squeeze(1).cpu()     # (n_layers, n_heads, head_dim, head_dim)
            ffn = torch.movedim(ffn[..., layers], -1, 0).squeeze(1).cpu()   # (n_layers, hidden_size)
            cache = (x, kv, ffn)
            cache = activation_callback(cache)
            return cache
        
        xs, kvs, ffns = [], [], []
        for input in inputs:
            x, kv, ffn = extract_single_line(input)
            xs.append(x)
            kvs.append(kv)
            ffns.append(ffn)
        xs = torch.stack(xs, dim=0)
        kvs = torch.stack(kvs, dim=0)
        ffns = torch.stack(ffns, dim=0)
        
        return xs, kvs, ffns
    
    def store_cache(self, cache: tuple[torch.Tensor], path: str) -> None:
        """
        Store cache to path
        Args:
            cache (tuple[torch.Tensor]): (n_inputs, n_layers, hidden_size), (n_inputs, n_layers, n_heads, head_dim, head_dim), (n_inputs, n_layers, hidden_size)
            path (str)
        """
        logger = logging.getLogger(__name__)
        xs, kvs, ffns = cache
        if path.endswith(".pt"):
            path = path[:-3]
            
        out_path = f"{path}_x.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(cache, out_path)
        
        out_path = f"{path}_kv.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(cache, out_path)
        
        out_path = f"{path}_ffn.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(cache, out_path)
        
        logger.info(f"Saved activations to {path}")