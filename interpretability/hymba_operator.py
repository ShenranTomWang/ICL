from transformers import AutoTokenizer
from interpretability.models.hymba import HymbaForCausalLM
import torch
from typing import Callable
from .operator import Operator

class HymbaOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = HymbaForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        super().__init__(tokenizer, model)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> list[torch.Tensor]:
        """
        TODO: should use built-in cache instance
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for cache, applied to all cache from all layers
        Returns:
            list[torch.Tensor]: attention (n_inputs, n_layers, n_heads, seqlen, seqlen), ssm_states (n_layers, ssm_intermediate_size, ssm_state_size)
        """
        layer_type = ["h" for _ in range(self.model.config.num_hidden_layers)]
        cache = self.model.get_cache_instance(1, self.device, self.dtype, layer_type)
        attns, ssm_states = [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            output = self.model(**tokenized, use_cache=True, past_key_values=cache)
            import pdb; pdb.set_trace()
            attn = output.attentions
            ssm_state = output.ssm_states
            attn = [attn[layer] for layer in layers]
            ssm_state = [ssm_state[layer] for layer in layers]
            attn = torch.stack(attn, dim=0).squeeze(1)              # (n_layers, n_heads, seqlen, seqlen)
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            cache = (attn, ssm_state)
            attn, ssm_state = activation_callback(cache)
            attns.append(attn)
            ssm_states.append(ssm_state)
        
        attns = torch.stack(attns, dim=0)               # (n_inputs, n_layers, n_heads, seqlen, seqlen)
        ssm_states = torch.stack(ssm_states, dim=0)     # (n_inputs, n_layers, ssm_intermediate_size, ssm_state_size)
        return attns, ssm_states
    
    def store_cache(self, cache: tuple[tuple[torch.Tensor]], path: str) -> None:
        """
        Store cache to path
        Args:
            cache (tuple[tuple[torch.Tensor]])
            path (str)
        """
        raise NotImplementedError("Method not implemented")