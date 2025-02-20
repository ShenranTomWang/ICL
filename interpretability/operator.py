from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable
from abc import ABC, abstractmethod
import os, logging

def layer_callback(resid: torch.Tensor) -> torch.Tensor:
    return resid[0, -1, :]

class Operator(ABC):
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, tl_model: HookedTransformer = None):
        self.model = model if tl_model == None else tl_model
        self.tokenizer: AutoTokenizer = tokenizer
        self.device = model.device
        self.transformer_lens = tl_model != None

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
    def extract_resid(self, inputs: list, layers: list, layer_callback: Callable = layer_callback, all_callback: Callable = lambda x: x) -> list[torch.Tensor]:
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
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> torch.Tensor:
        """
        Extract cache at specified layers
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor])): callback function for cache, applied to all cache from all layers
        Returns:
            torch.Tensor
        """
        pass
    
    @abstractmethod
    def store_cache(self, cache: tuple[tuple[torch.Tensor]], path: str) -> None:
        """
        Store cache to specified path
        Args:
            cache (tuple[tuple[torch.Tensor]]): cache
            path (str): path to store cache
        """
        pass
    
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