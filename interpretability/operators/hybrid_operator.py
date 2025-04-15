import shutup; shutup.please()
from transformers import AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
from interpretability.attention_managers import HybridAttentionManager
from interpretability.fv_maps import HybridFVMap
from interpretability.hooks import add_mean_hybrid
from interpretability.tokenizers import Tokenizer
from abc import ABC

class HybridOperator(Operator, ABC):
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: AutoModelForCausalLM,
        device: torch.DeviceObjType,
        dtype: torch.dtype,
        n_layers: int,
        attn_layers: list[int],
        scan_layers: list[int],
        n_attn_heads: int,
        n_scan_heads: int
    ):
        self.n_layers = n_layers
        self.attn_layers = attn_layers
        self.n_attn_layers = len(attn_layers)
        self.n_scan_layers = len(scan_layers)
        self.scan_layers = scan_layers
        self.n_attn_heads = n_attn_heads
        self.n_scan_heads = n_scan_heads
        self.ALL_LAYERS = [i for i in range(n_layers)]
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
    
    @torch.inference_mode()
    def extract_attention_managers(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[HybridAttentionManager]:
        """
        Extract attentions from the model
        Args:
            inputs (_type_): list of inputs (string)
            activation_callback (Callable, optional): callback function to process attentions. Defaults to lambda x: x

        Returns:
            list[HybridAttentionManager]: attentions (self-attention and scan)
        """
        outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            output = self.model(**tokenized, output_attentions=True)
            # n_layers * (batch_size, n_heads, pad_len + seqlen, pad_len + seqlen), (batch_size, pad_len + seqlen, attn_channels), (batch_size, pad_len + seqlen, attn_channels)
            all_attn, attn_output, scan_output = output.attentions
            all_attn, attn_output, scan_output = list(all_attn), list(attn_output), list(scan_output)
            hybrid_output = HybridAttentionManager(all_attn, attn_output, scan_output, "cpu")
            hybrid_output = activation_callback(hybrid_output)
            outputs.append(hybrid_output)
        return outputs
    
    @torch.inference_mode()
    def generate_AIE_map(self, steer: list[HybridAttentionManager], inputs: list[list[str]], label_ids: list[torch.Tensor]) -> HybridFVMap:
        """
        Generate AIE map from attention outputs
        Args:
            steer (list[HybridAttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
        Returns:
            TransformerFVMap: AIE map
        """
        attn_map = torch.empty((self.n_attn_layers, self.n_attn_heads))
        scan_map = torch.empty((self.n_scan_layers, self.n_scan_heads))
        for layer in self.attn_layers:
            for head in range(self.n_attn_heads):
                head_logits, head_fv_logits = [], []
                for i, attn in enumerate(steer):
                    attn_kwargs = self.attention2kwargs(attn, layers=[layer], last_k=1, heads=[head], keep_scan=False)
                    inputs_task = inputs[i]
                    task_logits, task_fv_logits = [], []
                    for input in inputs_task:
                        logit = self.forward(input).logits[:, -1, :]
                        logit_fv = self.forward(input, **attn_kwargs).logits[:, -1, :]
                        task_logits.append(logit)
                        task_fv_logits.append(logit_fv)
                    task_logits = torch.stack(task_logits, dim=0)
                    task_fv_logits = torch.stack(task_fv_logits, dim=0)
                    head_logits.append(task_logits)
                    head_fv_logits.append(task_fv_logits)
                head_AIE = self.compute_AIE(head_fv_logits, head_logits, label_ids)
                attn_map[layer, head] = head_AIE
        for layer in self.scan_layers:
            for head in range(self.n_scan_heads):
                head_logits, head_fv_logits = [], []
                for i, attn in enumerate(steer):
                    attn_kwargs = self.attention2kwargs(attn, layers=[layer], last_k=1, heads=[head], keep_attention=False)
                    inputs_task = inputs[i]
                    task_logits, task_fv_logits = [], []
                    for input in inputs_task:
                        logit = self.forward(input).logits[:, -1, :]
                        logit_fv = self.forward(input, **attn_kwargs).logits[:, -1, :]
                        task_logits.append(logit)
                        task_fv_logits.append(logit_fv)
                    task_logits = torch.stack(task_logits, dim=0)
                    task_fv_logits = torch.stack(task_fv_logits, dim=0)
                    head_logits.append(task_logits)
                    head_fv_logits.append(task_fv_logits)
                head_AIE = self.compute_AIE(head_fv_logits, head_logits, label_ids)
                scan_map[layer, head] = head_AIE
        return HybridFVMap(attn_map, scan_map, self.attn_layers, self.scan_layers, self.dtype)

    def attention2kwargs(
        self,
        attention: HybridAttentionManager,
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
            attention (HybridAttentionManager)
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
            if keep_attention and layer in layers:
                attn = attn_outputs[layer]
            if keep_scan and layer in layers:
                scan = scan_outputs[layer]
            params += ((attention_intervention_fn, attn, scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
