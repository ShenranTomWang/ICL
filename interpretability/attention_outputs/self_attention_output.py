import torch
from .attention_output import AttentionOutput, AttentionOutputItem

class SelfAttentionOutput(AttentionOutput):
    def __init__(self, all_attns: list[torch.Tensor] | None, attn_outputs: list[torch.Tensor] | None):
        self.all_attns = AttentionOutputItem(all_attns) if all_attns is not None else None
        self.attn_outputs = AttentionOutputItem(attn_outputs) if attn_outputs is not None else None
        
    def __add__(self, other: "SelfAttentionOutput") -> "SelfAttentionOutput":
        if other is None:
            return SelfAttentionOutput(None, self.attn_outputs)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs
        return SelfAttentionOutput(None, attn_outputs)
        
    def __sub__(self, other: "SelfAttentionOutput") -> "SelfAttentionOutput":
        if other is None:
            return SelfAttentionOutput(None, self.attn_outputs)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else [-1 * attn for attn in other.attn_outputs]
        return SelfAttentionOutput(None, attn_outputs)
    
    def __truediv__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs)
    
    def __mul__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs)
    
    def __rmul__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs)
    
    def __iter__(self):
        yield self.all_attns
        yield self.attn_outputs
        
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, f"{path}")
    
    def mean(self) -> "SelfAttentionOutput":
        all_attn, attn_output = self.all_attns, self.attn_outputs
        for i, attn_i in enumerate(attn_output):
            if attn_i is not None:
                attn_output[i] = attn_i.mean(dim=1).unsqueeze(1)   # (1, attn_channels)
        return SelfAttentionOutput(all_attn, attn_output)