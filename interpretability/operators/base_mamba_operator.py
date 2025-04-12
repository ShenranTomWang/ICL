import shutup; shutup.please()
import torch
from typing import Callable
from transformers import AutoModelForCausalLM
from .operator import Operator
from interpretability.tokenizers import Tokenizer
from interpretability.attention_managers import ScanManager
from interpretability.hooks import add_mean_scan

class BaseMambaOperator(Operator):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: Tokenizer, device: torch.DeviceObjType, dtype: torch.dtype):
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self) -> Callable:
        return add_mean_scan
        
    def extract_attention_outputs(self, inputs, activation_callback = lambda x: x) -> list[ScanManager]:
        """
        Extract internal representations at of attention outputs
        Args:
            inputs (list): list of inputs
            activation_callback (function(torch.Tensor)): callback function applied to all attention outputs from all layers
        Returns:
            list[ScanManager]: list of ScanOutputs
        """
        attention_outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            scan_outputs = self.model(**tokenized, output_attentions=True).attentions
            scan_outputs = list(scan_outputs)
            scan_outputs = ScanManager(scan_outputs)
            scan_outputs = activation_callback(scan_outputs)
            attention_outputs.append(scan_outputs)
        return attention_outputs
    
    def attention2kwargs(
        self,
        scan_outputs: ScanManager,
        scan_intervention_fn: Callable = add_mean_scan,
        layers: list[int] = None,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            scan_outputs (ScanManager): intervention values
            scan_intervention_fn (Callable): intervention function for scan, defaults to add_mean_scan
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        params = ()
        for layer in self.ALL_LAYERS:
            scan = scan_outputs[layer] if layer in layers else None
            params += ((scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
