import shutup; shutup.please()
from interpretability.models.zamba2 import Zamba2ForCausalLM
import torch
from transformers import AutoTokenizer
from .hybrid_operator import HybridOperator
from interpretability.tokenizers import HybridTokenizer

class ZambaOperator(HybridOperator):
    """Subclassing HymbaOperator to avoid redundant code
    """
    
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Zamba2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        self.HYBRID_LAYERS = model.config.hybrid_layer_ids
        super().__init__(HybridTokenizer(tokenizer), model, device, dtype)
