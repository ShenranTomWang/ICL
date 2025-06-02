from .transformer_operator import TransformerOperator
from interpretability.tokenizers import Llama3Tokenizer
from interpretability.fv_maps import FVMap
from interpretability.attention_managers import AttentionManager
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable

class ForwardWrapperTransformerOperator(TransformerOperator):
    """
    This is a wrapper class for general transformer models. It does not support any interpretability features.
    It is used to run the model in inference mode and get the output.
    """
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to(device).to(dtype)
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        super().__init__(model, Llama3Tokenizer(tokenizer), device, dtype, n_layers, n_heads)
        
    def generate_AIE_map(self, steer, inputs, label_ids) -> FVMap:
        raise NotImplementedError("ForwardWrapperTransformerOperator does not support AIE map generation")
    
    def get_attention_add_mean_hook(self) -> Callable:
        raise NotImplementedError("ForwardWrapperTransformerOperator does not support intervention features")
    
    def extract_attention_managers(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[AttentionManager]:
        raise NotImplementedError("ForwardWrapperTransformerOperator does not support intervention features")