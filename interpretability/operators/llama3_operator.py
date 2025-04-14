from .transformer_operator import TransformerOperator
from interpretability.tokenizers import Llama3Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

class Llama3Operator(TransformerOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        warnings.warn("Llama3Operator does not support interventions or non-standard hidden state extraction, please use with caution")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        super().__init__(model, Llama3Tokenizer(tokenizer), device, dtype, n_layers, n_heads)