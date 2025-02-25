from transformers import AutoTokenizer
from interpretability.models.hymba import HymbaForCausalLM
import torch
from typing import Callable
from .operator import Operator
import logging, os

class HymbaOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = HymbaForCausalLM.from_pretrained(path).to(device).to(dtype)
        super().__init__(tokenizer, model)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        TODO: should use built-in cache instance
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for k, v, ssm_state, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of k (n_layers, n_heads, seqlen, k_head_dim), list of v (n_layers, n_heads, seqlen, v_head_dim),
                list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size)
        """
        layer_type = ["h" for _ in range(self.model.config.num_hidden_layers)]
        cache = self.model.get_cache_instance(1, self.device, self.dtype, layer_type)
        ks, vs, ssm_states = [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            _ = self.model(**tokenized, use_cache=True, past_key_values=cache)
            k = cache.key_cache
            v = cache.value_cache
            ssm_state = cache.ssm_states
            k = [k[layer] for layer in layers if len(k[layer].shape) == 4]
            v = [v[layer] for layer in layers if len(v[layer].shape) == 4]
            ssm_state = [ssm_state[layer] for layer in layers]
            k = torch.stack(k, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, k_head_dim)
            v = torch.stack(v, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, v_head_dim)
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            cache = (k, v, ssm_state)
            k, v, ssm_state = activation_callback(cache)
            ks.append(k)
            vs.append(v)
            ssm_states.append(ssm_state)
        
        return ks, vs, ssm_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str) -> None:
        """
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]])
            path (str)
        """
        logger = logging.getLogger(__name__)
        ks, vs, ssm_states = cache
        if path.endswith(".pt"):
            path = path[:-3]
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        for i, (k, v, ssm_state) in enumerate(zip(ks, vs, ssm_states)):
            torch.save(k, f"{path}_k_{i}.pt")
            torch.save(v, f"{path}_v_{i}.pt")
            torch.save(ssm_state, f"{path}_ssm_state_{i}.pt")
        logger.info(f"Stored cache to {path}")