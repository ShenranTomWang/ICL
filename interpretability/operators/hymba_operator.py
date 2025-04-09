import shutup; shutup.please()
from interpretability.models.hymba import HymbaForCausalLM
from interpretability.tokenizers import HybridTokenizer
from transformers import AutoTokenizer
import torch
from .hybrid_operator import HybridOperator

class HymbaOperator(HybridOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = HymbaForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        self.KV_LAYERS = [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 19, 21, 23, 25, 27, 29, 31]
        super().__init__(HybridTokenizer(tokenizer), model, device, dtype)
