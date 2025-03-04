from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import torch
from .operator import Operator
import os, logging
from typing import Callable

class TransformerOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        super().__init__(tokenizer, model)
        
    def get_cache_instance(self):
        cache = DynamicCache()
        return cache
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        Extract kv cache at specified layers
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback (function(tuple[torch.Tensor]), optional): callback function for cache, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: tuple of 2 list of tensors kv of shape (n_layers, n_heads, seqlen, headdim)
        """

        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).past_key_values
            k = cache.key_cache
            v = cache.value_cache
            k = [k[layer] for layer in layers]
            v = [v[layer] for layer in layers]
            k = torch.stack(k, dim=0).squeeze(1)     # (n_layers, n_heads, seqlen, headdim)
            v = torch.stack(v, dim=0).squeeze(1)     # (n_layers, n_heads, seqlen, headdim)
            cache = (k, v)
            k, v = activation_callback(cache)
            return k, v
        
        ks, vs = [], []
        for input in inputs:
            k, v = extract_single_line(input)
            ks.append(k)
            vs.append(v)
        
        return ks, vs

    def store_cache(self, kvs: tuple[list[torch.Tensor]], path: str) -> None:
        """
        Store cache to specified path
        Args:
            cache (tuple[tuple[torch.Tensor]]): cache
            path (str): path to store cache
        """
        logger = logging.getLogger(__name__)
        ks, vs = kvs[0], kvs[1]       # (n, n_layers, n_heads, seqlen, headdim)
        if path.endswith(".pt"):
            path = path[:-3]
        for i, (k, v) in enumerate(zip(ks, vs)):
            out_path = path + f"_k_{i}.pt"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(k, out_path)
            out_path = path + "_v_{i}.pt"
            torch.save(v, out_path)
            logger.info(f"Saved activations to {out_path}")
            
    def load_cache(self, dir: str, split: str, index: int) -> tuple:
        """
        Load cache from specified directory
        Args:
            dir (str)
            split (str): one of demo, test and train
            index (int)

        Returns:
            tuple[torch.Tensor]: k, v
        """
        k_path = os.path.join(dir, f"{split}_cache_k_{index}.pt")
        v_path = os.path.join(dir, f"{split}_cache_v_{index}.pt")
        k = torch.load(k_path, map_location=self.device).to(self.dtype)
        v = torch.load(v_path, map_location=self.device).to(self.dtype)
        return k, v
    
    def cache2kwargs(
        self,
        cache: tuple[torch.Tensor],
        layers: list[int] = None,
        keep_kv: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            kv_layers (list[int]): list of layers kv cache maps to
            layers (list[int], optional): list of layers to use cache, if None, use all layers. Defaults to None.
            keep_kv (bool, optional): whether to keep kv cache. Defaults to True.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        cache_instance = self.get_cache_instance()
        if layers is None:
            layers = self.ALL_LAYERS
        k, v = cache
        k = [k[i: i + 1, ...] for i in range(k.shape[0])]
        v = [v[i: i + 1, ...] for i in range(v.shape[0])]
        k_list, v_list = [], []
        if keep_kv:
            for layer in range(self.model.config.num_hidden_layers):
                if layer in layers:
                    k_list.append(k[layer])
                    v_list.append(v[layer])
                else:
                    k_list.append(torch.zeros((1, 0)))
                    v_list.append(torch.zeros((1, 0)))
        cache_instance.key_cache = k_list
        cache_instance.value_cache = v_list
        # TODO: left off here last time
        kwargs = {"past_key_values": cache_instance}
        return kwargs