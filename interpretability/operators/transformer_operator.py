import shutup; shutup.please()
from interpretability.attention_managers import SelfAttentionManager
from interpretability.fv_maps import TransformerFVMap
from interpretability.hooks import add_mean_hybrid, fv_replace_head_generic
from transformers import AutoModelForCausalLM
from interpretability.tokenizers import Tokenizer
import torch
from .operator import Operator
from typing import Callable

class TransformerOperator(Operator):
    """
    Operator class for transformer models. This is not an abstract class, but it is not meant to be used directly.
    Inherit this class to create an operator for a specific transformer model.
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: Tokenizer,
        device: torch.DeviceObjType,
        dtype: torch.dtype,
        n_layers: int,
        n_heads: int
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ALL_LAYERS = [i for i in range(n_layers)]
        super().__init__(tokenizer=tokenizer, model=model, device=device, dtype=dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
        
    @torch.inference_mode()
    def extract_attention_managers(self, inputs: list[str], activation_callback = lambda x: x) -> SelfAttentionManager:
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
            attn_output = SelfAttentionManager(all_attn, attn_output, "cpu")
            attn_output = activation_callback(attn_output)
            attn_outputs.append(attn_output)
        return attn_outputs
    
    @torch.inference_mode()
    def generate_AIE_map(self, steer: list[SelfAttentionManager], inputs: list[list[str]], label_ids: list[torch.Tensor]) -> TransformerFVMap:
        """
        Generate AIE map from attention outputs for a list of inputs
        Args:
            steer (list[SelfAttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
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
        
        attn_map = torch.empty((self.n_layers, self.n_heads))
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_fv_logits = []
                for i, (attn, inputs_task) in enumerate(zip(steer, inputs)):
                    attn_kwargs = self.attention2kwargs(attn, layers=[layer], attention_intervention_fn=fv_replace_head_generic, head=head)
                    task_fv_logits = []
                    for input_task in inputs_task:
                        logit_fv = self.forward(input_task, **attn_kwargs).logits[:, -1, :].to("cpu")
                        task_fv_logits.append(logit_fv)
                    task_fv_logits = torch.cat(task_fv_logits, dim=0)
                    head_fv_logits.append(task_fv_logits)
                head_AIE = self.compute_AIE(head_fv_logits, original_logits, label_ids)
                attn_map[layer, head] = head_AIE
        return TransformerFVMap(attn_map, self.dtype)
                        
    
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
