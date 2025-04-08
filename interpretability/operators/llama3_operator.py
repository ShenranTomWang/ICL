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
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        super().__init__(model, Llama3Tokenizer(tokenizer), device, dtype)