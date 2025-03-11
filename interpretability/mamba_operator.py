import shutup; shutup.please()
from transformers import AutoTokenizer
from interpretability.models.mamba import MambaCache, MambaForCausalLM
import torch
from typing import Callable
from .operator import Operator
import logging, os

class MambaOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = MambaForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.n_layer)]
        super().__init__(tokenizer, model)
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for ssm_state, conv_state applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size), list of conv_states (n_layers, conv_intermediate_size, conv_state_size)
        """
        ssm_states, conv_states = [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).cache_params     # TODO: left off here, need to check cache args
            ssm_state = cache.ssm_states
            conv_state = cache.conv_states
            ssm_state = [ssm_state[layer] for layer in layers]
            conv_state = [conv_state[layer] for layer in layers]
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            conv_state = torch.stack(conv_state, dim=0).squeeze(1)  # (n_layers, conv_intermediate_size, conv_state_size)
            cache = (ssm_state, conv_state)
            ssm_state, conv_state = activation_callback(cache)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)
        
        return ssm_states, conv_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str) -> None:
        """
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
        ssm_state = torch.load(ssm_state_path, map_location=self.device).to(self.dtype)
        conv_state = torch.load(conv_state_path, map_location=self.device).to(self.dtype)
        return ssm_state, conv_state
    
    def cache2kwargs(
        self,
        cache: tuple[torch.Tensor],
        demo_length: int,
        layers: list[int] = None,
        keep_ssm: bool = True,
        keep_conv: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            demo_length (int): length of demo, tokenized
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
        ssm_state = ssm_state.unsqueeze(1)      # (n_layers, batch_size=1, ssm_intermediate_size, ssm_state_size)
        conv_state = conv_state.unsqueeze(1)    # (n_layers, batch_size=1, conv_intermediate_size, conv_state_size)
        ssm_state = [ssm_state[i] for i in self.ALL_LAYERS]
        conv_state = [conv_state[i] for i in self.ALL_LAYERS]
        ssm_states, conv_states = None, None
        if keep_ssm:
            i = 0
            ssm_states = []
            for layer in self.ALL_LAYERS:
                if layer in layers:
                    ssm_state_layer = ssm_state[i]
                    i += 1
                else:
                    ssm_state_layer = torch.zeros_like(ssm_state[0])
                torch._dynamo.mark_static_address(ssm_state_layer)
                ssm_states.append(ssm_state_layer)
        if keep_conv:
            i = 0
            conv_states = []
            for layer in self.ALL_LAYERS:
                if layer in layers:
                    conv_state_layer = conv_state[i]
                    i += 1
                else:
                    conv_state_layer = torch.zeros_like(conv_state[0])
                torch._dynamo.mark_static_address(conv_state_layer)
                conv_states.append(conv_state_layer)
        cache_instance = MambaCache(self.model.config, 1, dtype=self.dtype, device=self.device, ssm_states=ssm_states, conv_states=conv_states)
        cache_kwargs = {"use_cache": True, "cache_params": cache_instance, "cache_position": torch.tensor([demo_length], device=self.device)}
        return cache_kwargs
        
        
    def prepare_cache_kwargs_for_inputs(self, cache_kwargs: dict, input_length: int) -> dict:
        """
        Prepare cache kwargs for inputs
        Args:
            cache_kwargs (dict)
            input_length (int): length of input, tokenized
        Returns:
            dict: cache_kwargs
        """
        outputs = {}
        cache_kwargs = self.model._update_model_kwargs_for_generation(outputs, cache_kwargs, num_new_tokens=input_length)
        return cache_kwargs