import shutup; shutup.please()
import torch
from typing import Callable
from transformers import AutoModelForCausalLM
from .operator import Operator
from interpretability.tokenizers import Tokenizer
from interpretability.attention_managers import AttentionManager
from abc import ABC, abstractmethod

class BaseMambaOperator(Operator, ABC):
    """
    Operator for Mamba models. This is an abstract class.
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
        self.n_heads = n_heads
        super().__init__(tokenizer, model, device, dtype, n_layers, n_layers * n_heads)
        
    def get_fv_remove_head_attn_hook(self):
        return None
    
    @abstractmethod
    def _get_add_mean_hook(self) -> Callable:
        pass
    
    def attention2kwargs(
        self,
        scan_outputs: AttentionManager | None,
        scan_intervention_fn: Callable = None,
        layers: list[int] = None,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            scan_outputs (AttentionManager | None): intervention values
            scan_intervention_fn (Callable): intervention function for scan, defaults to None for using add mean hook for Mamba models
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            **kwargs: additional kwargs for intervention function
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = self.ALL_LAYERS
        if scan_intervention_fn is None:
            scan_intervention_fn = self._get_add_mean_hook()
        params = ()
        scan_outputs_ = scan_outputs.scan_outputs if scan_outputs else None
        for layer in self.ALL_LAYERS:
            scan = scan_outputs_[layer] if scan_outputs_ and layer in layers else None
            params += ((scan_intervention_fn, scan, kwargs),)
        return {"attention_overrides": params}
