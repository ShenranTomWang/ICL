from .transformer_operator import TransformerOperator
from interpretability.models.llama import LlamaForCausalLM
from interpretability.tokenizers import Llama3Tokenizer
from transformers import AutoTokenizer
import torch

class LlamaOperator(TransformerOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(path).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        super().__init__(model, Llama3Tokenizer(tokenizer), device, dtype, n_layers, n_heads)