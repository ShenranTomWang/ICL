import shutup; shutup.please()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .operator import Operator
import warnings

class RWKVOperator(Operator):
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        warnings.warn("RWKVOperator is not fully implemented, please use with caution")
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        super().__init__(tokenizer, model, device, dtype)
