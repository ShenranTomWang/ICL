from .transformer_operator import TransformerOperator
from interpretability.models.qwen2 import Qwen2ForCausalLM
from interpretability.tokenizers import StandardTokenizer
from transformers import AutoTokenizer
import torch

class Qwen2Operator(TransformerOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        super().__init__(model, StandardTokenizer(tokenizer), device, dtype, n_layers, n_heads)