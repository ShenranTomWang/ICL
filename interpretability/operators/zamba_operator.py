from typing import Callable
import shutup; shutup.please()
from interpretability.models.zamba2 import Zamba2ForCausalLM
import torch
from transformers import AutoTokenizer
from .hybrid_operator import HybridOperator
from interpretability.tokenizers import HybridTokenizer
from interpretability.hooks import fv_remove_head_generic, add_mean_hybrid
from interpretability.attention_managers import HybridMamba2AttentionManager

class ZambaOperator(HybridOperator):
    """Subclassing HymbaOperator to avoid redundant code
    """
    
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Zamba2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        attn_layers = [i for i in range(n_layers) if model.config.layers_block_type[i] == "hybrid"]
        scan_layers = [i for i in range(n_layers) if model.config.layers_block_type[i] == "mamba"]
        n_attn_heads = model.config.num_attention_heads
        n_scan_heads = model.config.n_mamba_heads
        super().__init__(HybridTokenizer(tokenizer), model, device, dtype, n_layers, attn_layers, scan_layers, n_attn_heads, n_scan_heads)

    def get_fv_remove_head_scan_hook(self) -> Callable:
        return fv_remove_head_generic
    
    def get_scan_add_mean_hook(self) -> Callable:
        return add_mean_hybrid
    
    def _get_attention_manager_class(self) -> type[HybridMamba2AttentionManager]:
        return HybridMamba2AttentionManager