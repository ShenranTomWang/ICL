import torch
from .attention_output import AttentionOutput, AttentionOutputItem

class SelfAttentionOutput(AttentionOutput):
    def __init__(self, all_attns: list[torch.Tensor] | None, attn_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.all_attns = AttentionOutputItem(all_attns).to(device) if all_attns is not None else None
        self.attn_outputs = AttentionOutputItem(attn_outputs).to(device) if attn_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "SelfAttentionOutput") -> "SelfAttentionOutput":
        if other is None:
            return SelfAttentionOutput(None, self.attn_outputs, self.device)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs
        return SelfAttentionOutput(None, attn_outputs, self.device)
        
    def __sub__(self, other: "SelfAttentionOutput") -> "SelfAttentionOutput":
        if other is None:
            return SelfAttentionOutput(None, self.attn_outputs, self.device)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else [-1 * attn for attn in other.attn_outputs]
        return SelfAttentionOutput(None, attn_outputs, self.device)
    
    def __truediv__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs, self.device)
    
    def __mul__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs, self.device)
    
    def __rmul__(self, other: int) -> "SelfAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs, self.device)
    
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
        return SelfAttentionOutput(all_attn, attn_output, self.device)
    
    def to(self, device: str) -> "SelfAttentionOutput":
        all_attns = self.all_attns.to(device) if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.to(device) if self.attn_outputs is not None else None
        return SelfAttentionOutput(all_attns, attn_outputs, self.device)