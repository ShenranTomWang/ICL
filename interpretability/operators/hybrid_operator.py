import shutup; shutup.please()
from transformers import AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
from interpretability.attention_outputs import HybridAttentionOutput
from interpretability.hooks import add_mean_hybrid
from interpretability.tokenizers import Tokenizer
from abc import ABC

class HybridOperator(Operator, ABC):
    
    def __init__(self, tokenizer: Tokenizer, model: AutoModelForCausalLM, device: torch.DeviceObjType, dtype: torch.dtype):
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
    
    @torch.inference_mode()
    def extract_attention_outputs(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[HybridAttentionOutput]:
        """
        Extract attentions from the model
        Args:
            inputs (_type_): list of inputs (string)
            activation_callback (Callable, optional): callback function to process attentions. Defaults to lambda x: x

        Returns:
            list[HybridAttentionOutput]: attentions (self-attention and scan)
        """
        outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            output = self.model(**tokenized, output_attentions=True)
            # n_layers * (batch_size, n_heads, pad_len + seqlen, pad_len + seqlen), (batch_size, pad_len + seqlen, attn_channels), (batch_size, pad_len + seqlen, attn_channels)
            all_attn, attn_output, scan_output = output.attentions
            all_attn, attn_output, scan_output = list(all_attn), list(attn_output), list(scan_output)
            hybrid_output = HybridAttentionOutput(all_attn, attn_output, scan_output, "cpu")
            hybrid_output = activation_callback(hybrid_output)
            outputs.append(hybrid_output)
        return outputs

    def attention2kwargs(
        self,
        attention: HybridAttentionOutput,
        attention_intervention_fn: Callable = add_mean_hybrid,
        scan_intervention_fn: Callable = add_mean_hybrid,
        layers: list[int] = None,
        keep_scan: bool = True,
        keep_attention: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            attention (HybridAttentionOutput)
            attention_intervention_fn (Callable, optional): intervention function for attention, defaults to add_mean_hybrid
            scan_intervention_fn (Callable, optional): intervention function for scan, defaults to add_mean_hybrid
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            keep_scan (bool, optional): whether to keep scan outputs. Defaults to True.
            keep_attention (bool, optional): whether to keep attention outputs. Defaults to True.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = set(self.ALL_LAYERS)
        _, attn_outputs, scan_outputs = attention
        params = ()
        for layer in self.ALL_LAYERS:
            attn = attn_outputs[layer] if keep_attention and layer in layers else None
            scan = scan_outputs[layer] if keep_scan and layer in layers else None
            params += ((attention_intervention_fn, attn, scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
