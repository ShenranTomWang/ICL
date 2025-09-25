import shutup; shutup.please()
from transformers import AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
from interpretability.attention_managers import HybridAttentionManager
from interpretability.fv_maps import HybridFVMap
from interpretability.hooks import add_mean_hybrid, fv_replace_head_generic, fv_remove_head_generic, fv_replace_head_mamba
from interpretability.tokenizers import Tokenizer
from abc import ABC, abstractmethod

class HybridOperator(Operator, ABC):
    """
    Operator class for hybrid models. This is not an abstract class, but it is not meant to be used directly.
    Inherit this class to create an operator for a specific hybrid model.
    """
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
        self.attn_layers = attn_layers
        self.n_attn_layers = len(attn_layers)
        self.n_scan_layers = len(scan_layers)
        self.scan_layers = scan_layers
        self.n_attn_heads = n_attn_heads
        self.n_scan_heads = n_scan_heads
        n_total_heads = self.n_attn_layers * n_attn_heads + self.n_scan_layers * n_scan_heads
        super().__init__(tokenizer, model, device, dtype, n_layers, n_total_heads)
        
    def _get_attention_add_mean_hook(self) -> Callable:
        return add_mean_hybrid
    
    @abstractmethod
    def _get_scan_add_mean_hook(self) -> Callable:
        pass
    
    @abstractmethod
    def _get_fv_replace_head_scan_hook(self) -> Callable:
        pass
    
    @abstractmethod
    def _get_attention_manager_class(self) -> type[HybridAttentionManager]:
        pass
    
    def get_fv_remove_head_attn_hook(self):
        return fv_remove_head_generic
    
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
            hybrid_output = self._get_attention_manager_class()(all_attns=all_attn, attn_outputs=attn_output, scan_outputs=scan_output, device="cpu")
            hybrid_output = activation_callback(hybrid_output)
            outputs.append(hybrid_output)
        return outputs
    
    @torch.inference_mode()
    def generate_AIE_map(
        self,
        steer: list[HybridAttentionManager],
        inputs: list[list[str]],
        label_ids: list[torch.Tensor],
        attn_intervention_fn: Callable = fv_replace_head_generic,
        scan_intervention_fn: Callable = None,
        return_F1: bool = False,
        **kwargs
    ) -> HybridFVMap:
        """
        Generate AIE map from attention outputs
        Args:
            steer (list[HybridAttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
            attn_intervention_fn (Callable, optional): intervention function for attention, defaults to fv_replace_head_generic
            scan_intervention_fn (Callable, optional): intervention function for scan, defaults to  None for fv_replace_head_scan_hook
            return_F1 (bool, optional): whether to return F1 score, defaults to False to return AIE
            kwargs: additional arguments, not used
        Returns:
            TransformerFVMap: AIE map
        """
        if scan_intervention_fn is None:
            scan_intervention_fn = self._get_fv_replace_head_scan_hook()
        original_logits = []
        for inputs_task in inputs:
            task_logits = []
            for input_task in inputs_task:
                logit = self.forward(input_task).logits[:, -1, :].to("cpu")
                task_logits.append(logit)
            task_logits = torch.cat(task_logits, dim=0)
            original_logits.append(task_logits)
        
        attn_map = torch.zeros((self.n_attn_layers, self.n_attn_heads))
        scan_map = torch.zeros((self.n_scan_layers, self.n_scan_heads))
        for layer_idx, layer in enumerate(self.attn_layers):
            for head in range(self.n_attn_heads):
                head_fv_logits = []
                for i, (attn, inputs_task) in enumerate(zip(steer, inputs)):
                    attn_kwargs = self.attention2kwargs(
                        attn,
                        layers=[layer],
                        keep_scan=False,
                        attention_intervention_fn=attn_intervention_fn,
                        head=head,
                        heads=[{layer_idx: {"stream": "attn", "head": head}}],
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
                attn_map[layer_idx, head] = head_AIE
        for layer_idx, layer in enumerate(self.scan_layers):
            for head in range(self.n_scan_heads):
                head_fv_logits = []
                for i, (attn, inputs_task) in enumerate(zip(steer, inputs)):
                    attn_kwargs = self.attention2kwargs(
                        attn,
                        layers=[layer],
                        keep_attention=False,
                        scan_intervention_fn=scan_intervention_fn,
                        head=head,
                        heads=[{layer_idx: {"stream": "scan", "head": head}}],
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
                scan_map[layer_idx, head] = head_AIE
        return HybridFVMap(attn_map, scan_map, self.attn_layers, self.scan_layers, self.dtype)

    def attention2kwargs(
        self,
        attention: HybridAttentionManager | None,
        attention_intervention_fn: Callable = None,
        scan_intervention_fn: Callable = None,
        layers: list[int] = None,
        keep_scan: bool = True,
        keep_attention: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            attention (HybridAttentionManager | None)
            attention_intervention_fn (Callable, optional): intervention function for attention, defaults to None for add mean hook of each model
            scan_intervention_fn (Callable, optional): intervention function for scan, defaults to  None for add mean hook of each model
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            keep_scan (bool, optional): whether to keep scan outputs. Defaults to True.
            keep_attention (bool, optional): whether to keep attention outputs. Defaults to True.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if attention_intervention_fn is None:
            attention_intervention_fn = self._get_attention_add_mean_hook()
        if scan_intervention_fn is None:
            scan_intervention_fn = self._get_scan_add_mean_hook()
        if layers is None:
            layers = set(self.ALL_LAYERS)
        _, attn_outputs, scan_outputs = attention if attention else (None, None, None)
        params = ()
        for layer in self.ALL_LAYERS:
            attn = attn_outputs[layer] if keep_attention and attn_outputs and layer in layers else None
            scan = scan_outputs[layer] if keep_scan and scan_outputs and layer in layers else None
            params += ((attention_intervention_fn, attn, scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
