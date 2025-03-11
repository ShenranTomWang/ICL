import shutup; shutup.please()
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
        self.dtype = dtype
        self.device = device
        super().__init__(tokenizer, model)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of x (n_inputs, n_layers, hidden_size), list of kv (n_inputs, n_layers, n_heads, head_dim, head_dim),
                    list of ffn (n_inputs, n_layers, hidden_size)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).state
            x, kv, ffn = cache[0], cache[1], cache[2]
            x = torch.movedim(x[..., layers], -1, 0).squeeze(1)       # (n_layers, hidden_size)
            kv = torch.movedim(kv[..., layers], -1, 0).squeeze(1)     # (n_layers, n_heads, head_dim, head_dim)
            ffn = torch.movedim(ffn[..., layers], -1, 0).squeeze(1)   # (n_layers, hidden_size)
            cache = (x, kv, ffn)
            cache = activation_callback(cache)
            return cache
        
        xs, kvs, ffns = [], [], []
        for input in inputs:
            x, kv, ffn = extract_single_line(input)
            xs.append(x)
            kvs.append(kv)
            ffns.append(ffn)
        
        return xs, kvs, ffns
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str) -> None:
        """
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]]): list of x (n_inputs, n_layers, hidden_size), 
                    list of kv (n_inputs, n_layers, n_heads, head_dim, head_dim), list of ffn (n_inputs, n_layers, hidden_size)
            path (str)
        """
        logger = logging.getLogger(__name__)
        xs, kvs, ffns = cache
        if path.endswith(".pt"):
            path = path[:-3]
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        for i, (x, kv, ffn) in enumerate(zip(xs, kvs, ffns)):
            torch.save(x, f"{path}_x_{i}.pt")
            torch.save(kv, f"{path}_kv_{i}.pt")
            torch.save(ffn, f"{path}_ffn_{i}.pt")
        
        logger.info(f"Saved activations to {path}")
        
    def load_cache(self, dir: str, split: str, index: int) -> tuple:
        xs_path = os.path.join(dir, f"{split}_x_{index}.pt")
        kvs_path = os.path.join(dir, f"{split}_kv_{index}.pt")
        ffns_path = os.path.join(dir, f"{split}_ffn_{index}.pt")
        xs = torch.load(xs_path, map_location=self.device).to(self.dtype)
        kvs = torch.load(kvs_path, map_location=self.device).to(self.dtype)
        ffns = torch.load(ffns_path, map_location=self.device).to(self.dtype)
        return xs, kvs, ffns
    
    def cache2kwargs(self, cache: tuple[torch.Tensor], keep_x: bool = True, keep_kv: bool = True, keep_ffn: bool = True, **kwargs) -> dict:
        """
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            keep_x (bool, optional): whether to use x cache. Defaults to True.
            keep_kv (bool, optional): whether to use kv cache. Defaults to True.
            keep_ffn (bool, optional): whether to use ffn cache. Defaults to True.

        Returns:
            dict: kwargs
        """
        x, kv, ffn = cache
        if keep_kv:
            kv = torch.movedim(kv, 0, -1).unsqueeze(0)
        if keep_x:
            x = torch.movedim(x, 0, -1).unsqueeze(0)
        if keep_ffn:
            ffn = torch.movedim(ffn, 0, -1).unsqueeze(0)
        kwargs = {"use_cache": True, "state": [x, kv, ffn]}
        return kwargs