import torch.nn as nn
from .config import HymbaConfig
from transformers.activations import ACT2FN

class HymbaMLP(nn.Module):
    def __init__(self, config: HymbaConfig):
        super().__init__()
        # self.config = config
        self.act_fn_name = config.mlp_hidden_act
        self.act_fn = ACT2FN[self.act_fn_name]
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        if self.act_fn_name == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)


    def forward(self, x):
        if self.act_fn_name == "silu":
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        elif self.act_fn_name == "relu2":
            return self.down_proj(self.act_fn(self.up_proj(x)))
        else:
            raise NotImplementedError(f"No such hidden_act: {self.act_fn_name}")