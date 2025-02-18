import torch
from torch import nn

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        self.config = config
        
        self.rope_type = config.rope_type
        
        self.factor = 2
        
        max_position_embeddings = self.config.max_position_embeddings

        if config.rope_type is None or config.rope_type == "default":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == 'ntk':
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings
            
            base = base * ((self.factor * max_position_embeddings / orig_max_position_embeddings) - (self.factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            
            self.max_seq_len_cached = orig_max_position_embeddings
            
        elif config.rope_type == 'dynamic_ntk':
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.original_inv_freq = inv_freq
            self.max_seq_len_cached = self.config.orig_max_position_embeddings
                
        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)