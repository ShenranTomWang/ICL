import shutup; shutup.please()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
import os, logging, warnings

class RWKVOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        warnings.warn("RWKVOperator is not fully implemented, please use with caution")
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        super().__init__(tokenizer, model, device, dtype)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        TODO: should return a cache object
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of x (n_inputs, n_layers, hidden_size), list of kv (n_inputs, n_layers, n_heads, head_dim, head_dim),
                    list of ffn (n_inputs, n_layers, hidden_size)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).state
            x, kv, ffn = cache[0], cache[1], cache[2]
            x = torch.movedim(x, -1, 0).squeeze(1)       # (n_layers, hidden_size)
            kv = torch.movedim(kv, -1, 0).squeeze(1)     # (n_layers, n_heads, head_dim, head_dim)
            ffn = torch.movedim(ffn, -1, 0).squeeze(1)   # (n_layers, hidden_size)
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
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str, fname: str = "") -> None:
        """
        TODO: should take a cache object and save
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]]): list of x (n_inputs, n_layers, hidden_size), 
                    list of kv (n_inputs, n_layers, n_heads, head_dim, head_dim), list of ffn (n_inputs, n_layers, hidden_size)
            path (str)
            fname (str, optional): special filename, Defaults to "".
        """
        logger = logging.getLogger(__name__)
        xs, kvs, ffns = cache
        if path.endswith(".pt"):
            path = path[:-3]
        os.makedirs(os.path.dirname(path), exist_ok=True)
            
        for i, (x, kv, ffn) in enumerate(zip(xs, kvs, ffns)):
            torch.save(x, f"{path}_cache_{fname}_x_{i}.pt")
            torch.save(kv, f"{path}_cache_{fname}_kv_{i}.pt")
            torch.save(ffn, f"{path}_cache_{fname}_ffn_{i}.pt")
        
        logger.info(f"Saved activations to {path}")
        
    def load_cache(self, dir: str, split: str, index: int, fname: str = "") -> tuple:
        """
        TODO: should load into a cache object
        Load cache from specified directory
        Args:
            dir (str): directory to load cache from
            split (str): cache split
            index (int): index of cache
            fname (str, optional): special filename. Defaults to "".

        Returns:
            tuple
        """
        xs_path = os.path.join(dir, f"{split}_cache_{fname}_x_{index}.pt")
        kvs_path = os.path.join(dir, f"{split}_cache_{fname}_kv_{index}.pt")
        ffns_path = os.path.join(dir, f"{split}_cache_ffn_{index}.pt")
        xs = torch.load(xs_path, map_location=self.device).to(self.dtype)
        kvs = torch.load(kvs_path, map_location=self.device).to(self.dtype)
        ffns = torch.load(ffns_path, map_location=self.device).to(self.dtype)
        return xs, kvs, ffns
    
    def cache2kwargs(self, cache: tuple[torch.Tensor], keep_x: bool = True, keep_kv: bool = True, keep_ffn: bool = True, **kwargs) -> dict:
        """
        TODO: should take a cache object and return kwargs
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