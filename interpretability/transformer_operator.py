import shutup; shutup.please()
from transformers import AutoTokenizer
from interpretability.models.qwen2 import Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
import torch
from .operator import Operator
import os, logging
from typing import Callable

class TransformerOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        super().__init__(tokenizer, model, device, dtype)
        
    def get_cache_instance(self):
        cache = DynamicCache()
        return cache
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, activation_callback: Callable = lambda x: x) -> DynamicCache:
        """
        Extract kv cache
        Args:
            inputs (list): list of inputs
            activation_callback (function(DynamicCache), optional): callback function for cache, applied to all cache from all layers
        Returns:
            DynamicCache: cache
        """

        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).past_key_values
            cache = activation_callback(cache)
            return cache
        
        caches = []
        for input in inputs:
            cache = extract_single_line(input)
            caches.append(cache)
        
        return caches

    def store_cache(self, caches: list[DynamicCache], path: str) -> None:
        """
        Store cache to specified path
        Args:
            cache (list[DynamicCache]): cache
            path (str): path to store cache
        """
        logger = logging.getLogger(__name__)
        if path.endswith(".pt"):
            path = path[:-3]
        for i, cache in enumerate(caches):
            k = cache.key_cache                      # [(batch_size, n_kv_heads, seqlen, headdim) * n_layers]
            v = cache.value_cache                    # [(batch_size, n_kv_heads, seqlen, headdim) * n_layers]
            k = torch.stack(k, dim=0).squeeze(1)     # (n_layers, n_kv_heads, seqlen, headdim)
            v = torch.stack(v, dim=0).squeeze(1)     # (n_layers, n_kv_heads, seqlen, headdim)
            out_path = path + f"_k_{i}.pt"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(k, out_path)
            out_path = path + "_v_{i}.pt"
            torch.save(v, out_path)
            logger.info(f"Saved activations to {out_path}")
            
    def load_cache(self, dir: str, split: str, index: int) -> DynamicCache:
        """
        Load cache from specified directory
        Args:
            dir (str)
            split (str): one of demo, test and train
            index (int)

        Returns:
            DynamicCache: cache
        """
        cache = self.get_cache_instance()
        k_path = os.path.join(dir, f"{split}_cache_k_{index}.pt")
        v_path = os.path.join(dir, f"{split}_cache_v_{index}.pt")
        k = torch.load(k_path, map_location=self.device).to(self.dtype)
        v = torch.load(v_path, map_location=self.device).to(self.dtype)
        k = [k[i: i + 1, ...] for i in range(k.shape[0])]
        v = [v[i: i + 1, ...] for i in range(v.shape[0])]
        for i, (k_, v_) in enumerate(zip(k, v)):
            cache.update(k_, v_, i)
        return cache
    
    def cache2kwargs(
        self,
        cache: DynamicCache,
        demo_length: int,
        **kwargs
    ) -> dict:
        """
        TODO: Qwen2.5 currently do not support caching, so do not use this method
        Convert cache to kwargs
        Args:
            cache (DynamicCache)
            demo_length (int): length of demo
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        raise NotImplementedError("Qwen2.5 currently do not support caching, so do not use this method")
        return {
            "use_cache": True,
            "past_key_values": cache,
            "cache_position": torch.tensor([demo_length], device=self.device, dtype=self.dtype)
        }