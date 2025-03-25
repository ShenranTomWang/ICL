import shutup; shutup.please()
from transformers import AutoTokenizer
from interpretability.models.qwen2 import Qwen2ForCausalLM
from interpretability.attention_outputs import SelfAttentionOutput
from interpretability.hooks import add_mean_hybrid
from transformers.cache_utils import DynamicCache
import torch
from .operator import Operator
import os, logging
from typing import Callable

class TransformerOperator(Operator):    
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
        
    def extract_attention_outputs(self, inputs: list[str], activation_callback = lambda x: x) -> SelfAttentionOutput:
        """
        Extract internal representations at of attention outputs
        Args:
            inputs (list): list of inputs
            activation_callback (SelfAttentionOutput): callback function applied to all attention outputs from all layers
        Returns:
            SelfAttentionOutput: attention outputs
        """
        attn_outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            all_attn, attn_output = self.model(**tokenized, output_attentions=True).attentions
            attn_output = SelfAttentionOutput(all_attn, attn_output)
            attn_output = activation_callback(attn_output)
            attn_outputs.append(attn_output)
        return attn_outputs
    
    def attention2kwargs(
        self,
        attention: SelfAttentionOutput,
        attention_intervention_fn: Callable = add_mean_hybrid,
        layers: list[int] = None,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            attention (SelfAttentionOutput)
            attention_intervention_fn (Callable): intervention function for attention, defaults to add_mean_hybrid
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        _, attn_outputs = attention
        params = ()
        for layer in self.ALL_LAYERS:
            attn = attn_outputs[layer] if layer in layers else None
            params += ((attention_intervention_fn, attn),)
        return {"attention_overrides": params}
        
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