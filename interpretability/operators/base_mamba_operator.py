import shutup; shutup.please()
import torch
from typing import Callable
from transformers import AutoModelForCausalLM
from .operator import Operator
from interpretability.tokenizers import Tokenizer
from interpretability.attention_managers import ScanManager
from interpretability.fv_maps import ScanFVMap
from interpretability.hooks import add_mean_scan

class BaseMambaOperator(Operator):
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
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self) -> Callable:
        return add_mean_scan
        
    def extract_attention_managers(self, inputs, activation_callback = lambda x: x) -> list[ScanManager]:
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
    
    def generate_AIE_map(self, steer: list[ScanManager], inputs: list[list[str]], label_ids: list[torch.Tensor]) -> ScanFVMap:
        """
        Generate AIE map from attention outputs
        Args:
            steer (list[ScanManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
        Returns:
            TransformerFVMap: AIE map
        """
        scan_map = torch.empty((self.n_layers, self.n_heads))
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_logits, head_fv_logits = [], []
                for i, attn in enumerate(steer):
                    attn_kwargs = self.attention2kwargs(attn, layers=[layer], last_k=1, heads=[head])
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
        return ScanFVMap(scan_map, self.dtype)
    
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
