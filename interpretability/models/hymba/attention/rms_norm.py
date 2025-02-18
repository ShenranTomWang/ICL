import torch
from torch import nn

class PerheadHymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, num_heads, eps=1e-6):
        """
        For per-head kq normalization
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_heads, 1, hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # assert 1==0, f"hiddens_states shape: {hidden_states.shape}" # [bsz, num_heads, seq_len, head_dim]
        assert hidden_states.shape[1] == self.weight.shape[1], f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        assert hidden_states.shape[3] == self.weight.shape[3], f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)