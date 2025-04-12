import shutup; shutup.please()
from interpretability.attention_managers import SelfAttentionManager
from interpretability.hooks import add_mean_hybrid
from transformers import AutoModelForCausalLM
from interpretability.tokenizers import Tokenizer
import torch
from .operator import Operator
from typing import Callable

class TransformerOperator(Operator):    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: Tokenizer, device: torch.DeviceObjType, dtype: torch.dtype):
        super().__init__(tokenizer=tokenizer, model=model, device=device, dtype=dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
        
    def extract_attention_outputs(self, inputs: list[str], activation_callback = lambda x: x) -> SelfAttentionManager:
        """
        Extract internal representations at of attention outputs
        Args:
            inputs (list): list of inputs
            activation_callback (SelfAttentionManager): callback function applied to all attention outputs from all layers
        Returns:
            SelfAttentionManager: attention outputs
        """
        attn_outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            all_attn, attn_output = self.model(**tokenized, output_attentions=True).attentions
            attn_output = SelfAttentionManager(all_attn, attn_output)
            attn_output = activation_callback(attn_output)
            attn_outputs.append(attn_output)
        return attn_outputs
    
    def attention2kwargs(
        self,
        attention: SelfAttentionManager,
        attention_intervention_fn: Callable = add_mean_hybrid,
        layers: list[int] = None,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            attention (SelfAttentionManager)
            attention_intervention_fn (Callable): intervention function for attention, defaults to add_mean_hybrid
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        _, attn_outputs = attention
        params = ()
        for layer in self.ALL_LAYERS:
            attn = attn_outputs[layer] if layer in layers else None
            params += ((attention_intervention_fn, attn, kwargs),)
        return {"attention_overrides": params}
