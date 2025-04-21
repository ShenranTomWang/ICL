import shutup; shutup.please()
import torch
from typing import Callable
from transformers import AutoModelForCausalLM
from .operator import Operator
from interpretability.tokenizers import Tokenizer
from interpretability.attention_managers import AttentionManager
from interpretability.fv_maps import ScanFVMap

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
    
    @torch.inference_mode()
    def generate_AIE_map(self, steer: list[AttentionManager], inputs: list[list[str]], label_ids: list[torch.Tensor]) -> ScanFVMap:
        """
        Generate AIE map from attention outputs
        Args:
            steer (list[AttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
        Returns:
            TransformerFVMap: AIE map
        """
        original_logits = []
        for inputs_task in inputs:
            task_logits = []
            for input in inputs_task:
                logit = self.forward(input).logits[:, -1, :].to("cpu")
                task_logits.append(logit)
            task_logits = torch.cat(task_logits, dim=0)
            original_logits.append(task_logits)
        
        scan_map = torch.empty((self.n_layers, self.n_heads))
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                head_fv_logits = []
                for i, attn in enumerate(steer):
                    attn_kwargs = self.attention2kwargs(attn, layers=[layer], last_k=1, heads=[head])
                    inputs_task = inputs[i]
                    task_fv_logits = []
                    for input in inputs_task:
                        logit_fv = self.forward(input, **attn_kwargs).logits[:, -1, :].to("cpu")
                        task_fv_logits.append(logit_fv)
                    task_fv_logits = torch.cat(task_fv_logits, dim=0)
                    head_fv_logits.append(task_fv_logits)
                head_AIE = self.compute_AIE(head_fv_logits, original_logits, label_ids)
                scan_map[layer, head] = head_AIE
        return ScanFVMap(scan_map, self.dtype)
    
    def attention2kwargs(
        self,
        scan_outputs: AttentionManager,
        scan_intervention_fn: Callable = None,
        layers: list[int] = None,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            scan_outputs (AttentionManager): intervention values
            scan_intervention_fn (Callable): intervention function for scan, defaults to None for using default
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        if scan_intervention_fn is None:
            scan_intervention_fn = self.get_attention_add_mean_hook()
        params = ()
        scan_output = scan_outputs.scan_outputs
        for layer in self.ALL_LAYERS:
            scan = scan_output[layer] if layer in layers else None
            params += ((scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
