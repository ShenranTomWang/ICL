import torch
from .attention_output import AttentionOutput, AttentionOutputItem

class ScanOutput(AttentionOutput):
    def __init__(self, scan_outputs: list[torch.Tensor] | None):
        self.scan_outputs = AttentionOutputItem(scan_outputs) if scan_outputs is not None else None
        
    def __add__(self, other: "ScanOutput") -> "ScanOutput":
        if other is None:
            return ScanOutput(self.scan_outputs)
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return ScanOutput(scan_outputs)
        
    def __sub__(self, other: "ScanOutput") -> "ScanOutput":
        if other is None:
            return ScanOutput(self.scan_outputs)
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return ScanOutput(scan_outputs)
    
    def __truediv__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs)
    
    def __mul__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs)
    
    def __rmul__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs)
    
    def __iter__(self):
        yield self.scan_outputs
        
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, f"{path}")
    
    def mean(self) -> "ScanOutput":
        scan_output = self.scan_outputs
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=2).unsqueeze(2)
        return ScanOutput(scan_output)
    
    def to(self, device: str) -> "ScanOutput":
        scan_outputs = self.scan_outputs.to(device)
        return ScanOutput(scan_outputs)