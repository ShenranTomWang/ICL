import shutup; shutup.please()
from interpretability.models.mamba2 import Mamba2ForCausalLM
from .base_mamba_operator import BaseMambaOperator
import torch
from transformers import AutoTokenizer
from interpretability.tokenizers import StandardTokenizer
from interpretability.attention_managers import Mamba2ScanManager, AttentionManager
from interpretability.fv_maps import ScanFVMap
from interpretability.hooks import fv_replace_head_generic, fv_remove_head_generic
from interpretability.hooks import add_mean_hybrid
from typing import Callable

class Mamba2Operator(BaseMambaOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = Mamba2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_heads
        super().__init__(model, StandardTokenizer(tokenizer), device, dtype, n_layers, n_heads)
        
    def _get_add_mean_hook(self) -> Callable:
        return add_mean_hybrid
    
    def get_fv_remove_head_scan_hook(self):
        return fv_remove_head_generic
    
    @torch.inference_mode()
    def generate_AIE_map(
        self,
        steer: list[AttentionManager],
        inputs: list[list[str]],
        label_ids: list[torch.Tensor],
        scan_intervention_fn: Callable = fv_replace_head_generic,
        return_F1: bool = False,
        **kwargs
    ) -> ScanFVMap:
        """
        Generate AIE map from attention outputs
        Args:
            steer (list[AttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
            scan_intervention_fn (Callable, optional): intervention function for scan, defaults to fv_replace_head_generic
            return_F1 (bool, optional): whether to return F1 score, defaults to False to return AIE
            kwargs: additional arguments, not used
        Returns:
            TransformerFVMap: AIE map
        """
        original_logits = []
        for inputs_task in inputs:
            task_logits = []
            for input_task in inputs_task:
                logit = self.forward(input_task).logits[:, -1, :].to("cpu")
                task_logits.append(logit)
            task_logits = torch.cat(task_logits, dim=0)
            original_logits.append(task_logits)
        
        scan_map = torch.zeros((self.n_layers, self.n_heads))
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_fv_logits = []
                for i, (attn, inputs_task) in enumerate(zip(steer, inputs)):
                    attn_kwargs = self.attention2kwargs(
                        attn,
                        layers=[layer],
                        scan_intervention_fn=scan_intervention_fn,
                        head=head,
                        heads=[{layer: {"stream": "scan", "head": head}}],
                        ablation_type="mean_ablation"
                    )
                    task_fv_logits = []
                    for input_task in inputs_task:
                        logit_fv = self.forward(input_task, **attn_kwargs).logits[:, -1, :].to("cpu")
                        task_fv_logits.append(logit_fv)
                    task_fv_logits = torch.cat(task_fv_logits, dim=0)
                    head_fv_logits.append(task_fv_logits)
                if return_F1:
                    head_AIE = self.compute_F1(head_fv_logits, original_logits, label_ids)
                else:
                    head_AIE = self.compute_AIE(head_fv_logits, original_logits, label_ids)
                scan_map[layer, head] = head_AIE
        return ScanFVMap(scan_map, self.dtype)
        
    @torch.inference_mode()
    def extract_attention_managers(self, inputs, activation_callback = lambda x: x) -> list[Mamba2ScanManager]:
        """
        Extract internal representations at of attention outputs
        Args:
            inputs (list): list of inputs
            activation_callback (function(torch.Tensor)): callback function applied to all attention outputs from all layers
        Returns:
            list[Mamba2ScanManager]: list of ScanOutputs
        """
        attention_outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            scan_outputs = self.model(**tokenized, output_attentions=True).attentions
            scan_outputs = list(scan_outputs)
            scan_outputs = Mamba2ScanManager(scan_outputs)
            scan_outputs = activation_callback(scan_outputs)
            attention_outputs.append(scan_outputs)
        return attention_outputs