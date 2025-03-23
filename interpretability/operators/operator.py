import shutup; shutup.please()
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import Cache, MambaCache
import torch
from typing import Callable
from interpretability.attention_outputs import AttentionOutput
from abc import ABC, abstractmethod
import os, logging

def layer_callback(resid: torch.Tensor) -> torch.Tensor:
    return resid[0, -1, :]

class Operator(ABC):
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.DeviceObjType, dtype: torch.dtype, tl_model: HookedTransformer = None):
        self.model = model if tl_model == None else tl_model
        self.tokenizer: AutoTokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.transformer_lens = tl_model != None
        
    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        self.forward(text, **kwargs)
    
    def forward(self, text: str, **kwargs) -> torch.Tensor:
        tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        output = self.model(**tokenized, **kwargs)
        return output

    @torch.inference_mode()
    def extract(self, inputs: list, stream: str, layers: list, activation_callback: Callable) -> list[torch.Tensor]:
        """
        Extract internal representations at specified layers of stream using transformer_lens
        Args:
            inputs (list): list of inputs
            stream (str): one of supported streams by TransformerLens
            layers (list): list of layer indices
            activation_callback (function(torch.Tensor, HookPoint)): callback function to extract activations, applied to activation values at each layer
        Returns:
            list[torch.Tensor]: list of tensors originally (n_layers, seqlen, hidden_size), but processed by activation_callback
        """
        if not self.transformer_lens:
            raise NotImplementedError("This method is only supported with transformer_lens models")
        def extract_single_line(input: str) -> torch.Tensor:
            activation = []
            def get_act(value: torch.Tensor, hook: HookPoint):
                value = activation_callback(value, hook)
                activation.append(value)

            hooks = []
            for layer in layers:
                hooks.append((get_act_name(stream, layer=layer), get_act))
                
            tokenized = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True)
            input_ids = tokenized.input_ids.to(self.model.device)
            attention_mask = tokenized.attention_mask.to(self.model.device)
            self.model.run_with_hooks(input_ids=input_ids, attention_mask=attention_mask, fwd_hooks=hooks, return_type=None)
            
            activation = torch.stack(activation, dim=0)
            return activation
        
        activations = []
        for input in inputs:
            activation = extract_single_line(input)
            activations.append(activation)
        
        return activations
    
    @torch.inference_mode()
    def extract_resid(self, inputs: list, layers: list, layer_callback: Callable = layer_callback, all_callback: Callable = lambda x: x) -> torch.Tensor:
        """
        Extract internal representations at specified layers of residual stream using vanilla transformers implementation
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            layer_callback (function(torch.Tensor)): callback function to extracted activations, applied to activation values at each layer
            all_callback (function(torch.Tensor)): callback function all extracted activations corresponding to each input
        Returns:
            torch.Tensor: (n_inputs, n_layers, hidden_size)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            output = self.model(**tokenized, output_hidden_states=True)
            activation = output.hidden_states
            activation = [activation[layer] for layer in layers]
            activation = [layer_callback(act) for act in activation]
            activation = torch.stack(activation, dim=0).cpu()
            return activation
        
        activations = []
        for input in inputs:
            activation = extract_single_line(input)
            activations.append(activation)
        activations = torch.stack(activations, dim=0)
        activations = all_callback(activations)
        
        return activations
    
    @abstractmethod
    def get_attention_mean(self, attn: AttentionOutput) -> AttentionOutput:
        """
        Get mean attention values along sequence dimension
        Args:
            attn (AttentionOutput): attention values
        Returns:
            AttentionOutput: mean attention values
        """
        pass
    
    @abstractmethod
    def extract_attention_outputs(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[AttentionOutput]:
        """
        Extract attentions at specified layers of attention stream
        Args:
            inputs (list): list of inputs
            activation_callback (function(torch.Tensor)): callback function to extracted activations, applied to activation values at each layer
        Returns:
            list[AttentionOutput]: attention outputs
        """
        pass
    
    @abstractmethod
    def store_attention_outputs(self, attention_outputs: list[AttentionOutput], path: str, fname: str = "") -> None:
        """
        Store attention outputs to specified path
        Args:
            attention_outputs (list[AttentionOutput]): list of attention outputs
            path (str): path to store attention outputs
            fname (str): special filename
        """
        pass
    
    @abstractmethod
    def load_attention_outputs(self, dir: str, split: str, index: int, fname: str = "") -> AttentionOutput:
        """
        Load attention outputs from specified directory
        Args:
            dir (str): directory to load attention outputs
            split (str): split of attention outputs
            index (int): index of attention outputs
            fname (str): special filename
        Returns:
            AttentionOutput: attention outputs
        """
        pass
    
    @abstractmethod
    def attention2kwargs(self, attention: AttentionOutput, **kwargs: dict) -> dict:
        """
        Convert attention to kwargs for forward pass
        Args:
            attention (AttentionOutput): attention
            kwargs (dict): kwargs
        Returns:
            dict: kwargs
        """
        pass
    
    @abstractmethod
    def extract_cache(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> Cache | MambaCache:
        """
        Extract cache
        Args:
            inputs (list): list of inputs
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            cache (Cache | MambaCache): transformers cache
        """
        pass
    
    @abstractmethod
    def store_cache(self, cache: Cache | MambaCache, path: str, fname: str = "") -> None:
        """
        Store cache to specified path
        Args:
            cache (Cache | MambaCache): cache
            path (str): path to store cache
            fname (str): special filename
        """
        pass
    
    @abstractmethod
    def load_cache(self, dir: str, split: str, index: int, fname: str = "") -> Cache | MambaCache:
        """
        Load cache from specified directory
        Args:
            dir (str): directory to load cache
            split (str): split of cache
            index (int): index of cache
            fname (str): special filename
        Returns:
            Cache | MambaCache: cache
        """
        pass
    
    
    @abstractmethod
    def cache2kwargs(self, cache: Cache | MambaCache, **kwargs: dict) -> dict:
        """
        Convert cache to kwargs for forward pass
        Args:
            cache (Cache | MambaCache): cache
            kwargs (dict): kwargs
        Returns:
            dict: kwargs
        """
        pass
    
    def prepare_cache_kwargs_for_inputs(self, cache_kwargs: dict, input_length: int) -> dict:
        """
        Prepare cache kwargs for inputs
        Args:
            cache_kwargs (dict)
            input_length (int): length of input, tokenized
        Returns:
            dict: cache_kwargs
        """
        return cache_kwargs
    
    def store_resid(self, activation: torch.Tensor, path: str) -> None:
        """
        Store residual to specified path
        Args:
            activation (torch.Tensor): residual
            path (str): path to store residual
        """
        logger = logging.getLogger(__name__)
        path = path + ".pt" if not path.endswith(".pt") else path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(activation, path)
        logger.info(f"Saved activations to {path}")