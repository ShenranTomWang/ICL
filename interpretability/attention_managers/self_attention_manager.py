import torch
from .attention_manager import AttentionManager
from .manager_item import AttentionManagerItem

class SelfAttentionManager(AttentionManager):
    def __init__(self, all_attns: list[torch.Tensor] | None, attn_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.all_attns = AttentionManagerItem(all_attns).to(device) if all_attns is not None else None
        self.attn_outputs = AttentionManagerItem(attn_outputs).to(device) if attn_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "SelfAttentionManager") -> "SelfAttentionManager":
        if other is None:
            return SelfAttentionManager(None, self.attn_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs
        return SelfAttentionManager(None, attn_outputs, self.device)
        
    def __sub__(self, other: "SelfAttentionManager") -> "SelfAttentionManager":
        if other is None:
            return SelfAttentionManager(None, self.attn_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else [-1 * attn for attn in other.attn_outputs]
        return SelfAttentionManager(None, attn_outputs, self.device)
    
    def __truediv__(self, other: int) -> "SelfAttentionManager":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        return SelfAttentionManager(all_attns, attn_outputs, self.device)
    
    def __mul__(self, other: int) -> "SelfAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionManager(all_attns, attn_outputs, self.device)
    
    def __rmul__(self, other: int) -> "SelfAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        return SelfAttentionManager(all_attns, attn_outputs, self.device)
    
    def __iter__(self):
        yield self.all_attns
        yield self.attn_outputs
        
    def get_last_token(self) -> "SelfAttentionManager":
        all_attns = self.all_attns.get_last_token() if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.get_last_token() if self.attn_outputs is not None else None
        return SelfAttentionManager(all_attns, attn_outputs, self.device)
    
    def mean(self) -> "SelfAttentionManager":
        all_attn, attn_output = self.all_attns.clone(), self.attn_outputs.clone()
        for i, attn_i in enumerate(attn_output):
            if attn_i is not None:
                attn_output[i] = attn_i.mean(dim=1).unsqueeze(1)   # (1, attn_channels)
        return SelfAttentionManager(all_attn, attn_output, self.device)
    
    def to(self, device: str) -> "SelfAttentionManager":
        all_attns = self.all_attns.to(device) if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.to(device) if self.attn_outputs is not None else None
        return SelfAttentionManager(all_attns, attn_outputs, device)