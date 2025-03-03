from transformers import AutoTokenizer, ZambaForCausalLM
from interpretability.models.zamba import ZambaHybridDynamicCache
import torch
from typing import Callable
from .operator import Operator
import logging, os

class ZambaOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = ZambaForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.n_layers)]
        super().__init__(tokenizer, model)
    
    def get_cache_instance(self):
        # TODO: implement this method
        cache = ZambaHybridDynamicCache(self.model.config, dtype=self.dtype, device=self.device)
        return cache
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        TODO: modify this method
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for k, v, ssm_state, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size), list of conv_states (n_layers, conv_intermediate_size, conv_state_size)
        """
        cache = self.get_cache_instance()
        ssm_states, conv_states = [], [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            _ = self.model(**tokenized, use_cache=True, cache_params=cache)
            ssm_state = cache.ssm_states
            conv_state = cache.conv_states
            ssm_state = [ssm_state[layer] for layer in layers]
            conv_state = [conv_state[layer] for layer in layers]
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            conv_state = torch.stack(conv_state, dim=0).squeeze(1)  # (n_layers, conv_intermediate_size, conv_state_size)
            cache = (ssm_state, conv_state)
            ssm_state = activation_callback(cache)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)
        
        return ssm_states, conv_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str) -> None:
        """
        TODO: modify this method
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]])
            path (str)
        """
        logger = logging.getLogger(__name__)
        ssm_states, conv_states = cache
        if path.endswith(".pt"):
            path = path[:-3]
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, (ssm_state, conv_state) in enumerate(zip(ssm_states, conv_states)):
            torch.save(ssm_state, f"{path}_ssm_state_{i}.pt")
            torch.save(conv_state, f"{path}_conv_state_{i}.pt")
        logger.info(f"Stored cache to {path}")
        
    def load_cache(self, dir: str, split: str, index: int) -> tuple[torch.Tensor]:
        """
        TODO: modify this method
        Load cache from specified directory
        Args:
            dir (str)
            split (str): one of demo, test and train
            index (int)

        Returns:
            tuple[torch.Tensor]: ssm_state
        """
        ssm_state_path = os.path.join(dir, f"{split}_cache_ssm_state_{index}.pt")
        conv_state_path = os.path.join(dir, f"{split}_cache_conv_state_{index}.pt")
        ssm_state = torch.load(ssm_state_path)
        conv_state = torch.load(conv_state_path)
        return ssm_state, conv_state
    
    def cache2kwargs(self, cache: tuple[torch.Tensor], layers: list[int] = None, keep_ssm: bool = True, keep_conv: bool = True, **kwargs) -> dict:
        """
        TODO: modify this method
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
        ssm_state, conv_state = cache
        cache_instance = self.get_cache_instance()
        ssm_state = [ssm_state[i: i + 1, ...] for i in range(ssm_state.shape[0])]
        if keep_ssm:
            cache_instance.ssm_states = [ssm_state[layer] for layer in layers]
        if keep_conv:
            cache_instance.conv_states = [conv_state[layer] for layer in layers]
        kwargs = {"use_cache": True, "cache_params": cache_instance}
        return kwargs
        