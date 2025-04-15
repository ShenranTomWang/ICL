import shutup; shutup.please()
from interpretability.models.mamba import MambaForCausalLM
from .base_mamba_operator import BaseMambaOperator
import torch
from transformers import AutoTokenizer
from interpretability.tokenizers import StandardTokenizer
from interpretability.attention_managers import MambaScanManager
from interpretability.hooks import add_mean_scan_mamba

class MambaOperator(BaseMambaOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = MambaForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = 1
        super().__init__(model, StandardTokenizer(tokenizer), device, dtype, n_layers, n_heads)
        
    def get_attention_add_mean_hook(self):
        return add_mean_scan_mamba
    
    @torch.inference_mode()
    def extract_attention_managers(self, inputs, activation_callback = lambda x: x) -> list[MambaScanManager]:
        """
        Extract internal representations at of attention outputs
        Args:
            inputs (list): list of inputs
            activation_callback (function(torch.Tensor)): callback function applied to all attention outputs from all layers
        Returns:
            list[MambaScanManager]: list of ScanOutputs
        """
        attention_outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            scan_outputs = self.model(**tokenized, output_attentions=True).attentions
            scan_outputs = list(scan_outputs)
            scan_outputs = MambaScanManager(scan_outputs)
            scan_outputs = activation_callback(scan_outputs)
            attention_outputs.append(scan_outputs)
        return attention_outputs