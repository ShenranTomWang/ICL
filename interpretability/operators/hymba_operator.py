import shutup; shutup.please()
from interpretability.models.hymba import HymbaForCausalLM
from interpretability.tokenizers import HybridTokenizer
from transformers import AutoTokenizer
import torch
from .hybrid_operator import HybridOperator
from interpretability.hooks import fv_remove_head_mamba

class HymbaOperator(HybridOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = HymbaForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_attn_heads = model.config.num_attention_heads
        n_scan_heads = 1
        attn_layers = [i for i in range(n_layers) if model.config.layer_type[i] == "h" or model.config.layer_type[i] == "a"]
        scan_layers = [i for i in range(n_layers) if model.config.layer_type[i] == "h" or model.config.layer_type[i] == "m"]
        super().__init__(HybridTokenizer(tokenizer), model, device, dtype, n_layers, attn_layers, scan_layers, n_attn_heads, n_scan_heads)

    def get_fv_remove_head_scan_hook(self):
        return fv_remove_head_mamba