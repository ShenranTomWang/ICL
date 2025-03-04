from transformers import AutoTokenizer, AutoModelForCausalLM
from interpretability.models.zamba import ZambaHybridDynamicCache
import torch
from typing import Callable
from .operator import Operator
import logging, os

class ZambaOperator(Operator):
    
    KV_LAYERS = [5, 11, 17, 23, 29, 35]
    
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        super().__init__(tokenizer, model)
    
    def get_cache_instance(self):
        cache = ZambaHybridDynamicCache(self.model.config, 1, dtype=self.dtype, device=self.device)
        return cache
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for k, v, ssm_state, conv_state, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of k (n_layers, n_heads, seqlen, k_head_dim), list of v (n_layers, n_heads, seqlen, v_head_dim),
            list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size), list of conv_states (n_layers, conv_intermediate_size, conv_state_size)
        """
        ks, vs, ssm_states, conv_states = [], [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).past_key_values
            k = cache.key_cache
            v = cache.value_cache
            ssm_state = cache.ssm_states
            conv_state = cache.conv_states
            k = [k[layer] for layer in layers if len(k[layer].shape) == 4]
            v = [v[layer] for layer in layers if len(v[layer].shape) == 4]
            ssm_state = [ssm_state[layer] for layer in layers]
            conv_state = [conv_state[layer] for layer in layers]
            k = torch.stack(k, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, k_head_dim)
            v = torch.stack(v, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, v_head_dim)
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            conv_state = torch.stack(conv_state, dim=0).squeeze(1)  # (n_layers, conv_intermediate_size, conv_state_size)
            cache = (k, v, ssm_state, conv_state)
            ssm_state = activation_callback(cache)
            ks.append(k)
            vs.append(v)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)
        
        return ks, vs, ssm_states, conv_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str) -> None:
        """
        TODO: modify this method
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]])
            path (str)
        """
        logger = logging.getLogger(__name__)
        ks, vs, ssm_states, conv_states = cache
        if path.endswith(".pt"):
            path = path[:-3]
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, (k, v, ssm_state, conv_state) in enumerate(zip(ks, vs, ssm_states, conv_states)):
            torch.save(k, f"{path}_k_{i}.pt")
            torch.save(v, f"{path}_v_{i}.pt")
            torch.save(ssm_state, f"{path}_ssm_state_{i}.pt")
            torch.save(conv_state, f"{path}_conv_state_{i}.pt")
        logger.info(f"Stored cache to {path}")
        
    def load_cache(self, dir: str, split: str, index: int) -> tuple[torch.Tensor]:
        """
        Load cache from specified directory
        Args:
            dir (str)
            split (str): one of demo, test and train
            index (int)

        Returns:
            tuple[torch.Tensor]: ssm_state
        """
        k_path = os.path.join(dir, f"{split}_cache_k_{index}.pt")
        v_path = os.path.join(dir, f"{split}_cache_v_{index}.pt")
        ssm_state_path = os.path.join(dir, f"{split}_cache_ssm_state_{index}.pt")
        conv_state_path = os.path.join(dir, f"{split}_cache_conv_state_{index}.pt")
        k = torch.load(k_path)
        v = torch.load(v_path)
        ssm_state = torch.load(ssm_state_path, map_location=self.device).to(self.dtype)
        conv_state = torch.load(conv_state_path, map_location=self.device).to(self.dtype)
        return k, v, ssm_state, conv_state
    
    def cache2kwargs(
        self,
        cache: tuple[torch.Tensor],
        kv_layers: list[int] = KV_LAYERS,
        layers: list[int] = None,
        keep_kv: bool = True,
        keep_ssm: bool = True,
        keep_conv: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            layers (list[int], optional): list of layers to use cache, if None, use all layers. Defaults to None.
            keep_ssm (bool, optional): whether to keep ssm cache. Defaults to True.
            keep_conv (bool, optional): whether to keep conv cache. Defaults to True.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        k, v, ssm_state, conv_state = cache
        cache_instance = self.get_cache_instance()
        k = [k[i: i + 1, ...] for i in range(k.shape[0])]
        v = [v[i: i + 1, ...] for i in range(v.shape[0])]
        ssm_state = [ssm_state[i: i + 1, ...] for i in range(ssm_state.shape[0])]
        conv_state = [conv_state[i: i + 1, ...] for i in range(conv_state.shape[0])]
        k_list, v_list = [], []
        if keep_kv:
            for layer in range(self.model.config.num_hidden_layers):
                if layer in kv_layers and layer in layers:
                    k_list.append(k[0])
                    v_list.append(v[0])
                else:
                    k_list.append(torch.zeros((1, 0)))
                    v_list.append(torch.zeros((1, 0)))
            cache_instance.key_cache = k_list
            cache_instance.value_cache = v_list
        if keep_ssm:
            cache_instance.ssm_states = [ssm_state[layer] for layer in layers]
        if keep_conv:
            cache_instance.conv_states = [conv_state[layer] for layer in layers]
        kwargs = {"use_cache": True, "past_key_values": cache_instance}
        return kwargs
        