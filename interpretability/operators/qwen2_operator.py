from .transformer_operator import TransformerOperator
from interpretability.models.qwen2 import Qwen2ForCausalLM
from interpretability.tokenizers import StandardTokenizer
from transformers import AutoTokenizer
import torch

class Qwen2Operator(TransformerOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        super().__init__(StandardTokenizer(tokenizer), model, device, dtype)