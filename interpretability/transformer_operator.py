from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from transformers import AutoTokenizer
import torch
from typing import Callable

class TransformerOperator:
    def __init__(self, tokenizer: AutoTokenizer, model):
        self.model = model
        self.tokenizer: AutoTokenizer = tokenizer
        self.device = model.device
        
    def extract_tf(self, inputs: list, layers: list, activation_callback: Callable) -> list[torch.Tensor]:
        """
        Extract internal representations at specified layers of residual stream using vanilla transformers implementation
        Args:
            inputs (list): list of inputs
            stream (str): one of supported streams by TransformerLens
            layers (list): list of layer indices
            activation_callback (function(torch.Tensor)): callback function to extract activations, applied to activation values at each layer
        Returns:
            list[torch.Tensor]: list of tensors (n_layers, seqlen, hidden_size)
        """
        def extract_single_line(input: str) -> torch.Tensor:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            output = self.model(**tokenized, output_hidden_states=True)
            activation = output.hidden_states
            activation = [activation[layer] for layer in layers]
            activation = [activation_callback(act) for act in activation]
            activation = torch.stack(activation, dim=0)
            return activation
        
        activations = []
        for input in inputs:
            activation = extract_single_line(input)
            activations.append(activation)
        
        return activations

    def extract_tl(self, inputs: list, stream: str, layers: list, activation_callback: Callable) -> list[torch.Tensor]:
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