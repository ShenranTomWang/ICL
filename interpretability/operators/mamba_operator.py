import shutup; shutup.please()
from interpretability.models.mamba import MambaForCausalLM
from .base_mamba_operator import BaseMambaOperator
import torch
from transformers import AutoTokenizer
from interpretability.tokenizers import StandardTokenizer

class MambaOperator(BaseMambaOperator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = MambaForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.n_layer)]
        super().__init__(model, StandardTokenizer(tokenizer), device, dtype)