import shutup; shutup.please()
from interpretability.models.mamba2 import Mamba2ForCausalLM
from .base_mamba_operator import BaseMambaOperator
import torch
from transformers import AutoTokenizer
from interpretability.tokenizers import StandardTokenizer

class Mamba2Operator(BaseMambaOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = Mamba2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_heads
        super().__init__(model, StandardTokenizer(tokenizer), device, dtype, n_layers, n_heads)