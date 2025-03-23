import torch
from .attention_output import AttentionOutput, AttentionOutputItem

class HybridAttentionOutput(AttentionOutput):
    def __init__(self, all_attns: list[torch.Tensor] | None, attn_outputs: list[torch.Tensor] | None, scan_outputs: list[torch.Tensor] | None):
        self.all_attns = AttentionOutputItem(all_attns) if all_attns is not None else None
        self.attn_outputs = AttentionOutputItem(attn_outputs) if attn_outputs is not None else None
        self.scan_outputs = AttentionOutputItem(scan_outputs) if scan_outputs is not None else None
        
    def __add__(self, other: "HybridAttentionOutput") -> "HybridAttentionOutput":
        if other is None:
            return HybridAttentionOutput(None, self.attn_outputs, self.scan_outputs)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return HybridAttentionOutput(None, attn_outputs, scan_outputs)
        
    def __sub__(self, other: "HybridAttentionOutput") -> "HybridAttentionOutput":
        if other is None:
            return HybridAttentionOutput(None, self.attn_outputs, self.scan_outputs)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else [-1 * attn for attn in other.attn_outputs]
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return HybridAttentionOutput(None, attn_outputs, scan_outputs)
    
    def __truediv__(self, other: int) -> "HybridAttentionOutput":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return HybridAttentionOutput(all_attns, attn_outputs, scan_outputs)
    
    def __mul__(self, other: int) -> "HybridAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return HybridAttentionOutput(all_attns, attn_outputs, scan_outputs)
    
    def __rmul__(self, other: int) -> "HybridAttentionOutput":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return HybridAttentionOutput(all_attns, attn_outputs, scan_outputs)
    
    def __iter__(self):
        yield self.all_attns
        yield self.attn_outputs
        yield self.scan_outputs
        
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, f"{path}")
    
    def mean(self) -> "HybridAttentionOutput":
        all_attn, attn_output, scan_output = self.all_attns, self.attn_outputs, self.scan_outputs
        for i, attn_i in enumerate(attn_output):
            if attn_i is not None:
                attn_output[i] = attn_i.mean(dim=1).unsqueeze(1)   # (1, attn_channels)
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=1).unsqueeze(1)   # (1, attn_channels)
        return HybridAttentionOutput(all_attn, attn_output, scan_output)