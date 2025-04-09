import torch
from .attention_output import AttentionOutput, AttentionOutputItem

class ScanOutput(AttentionOutput):
    def __init__(self, scan_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.scan_outputs = AttentionOutputItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "ScanOutput") -> "ScanOutput":
        if other is None:
            return ScanOutput(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return ScanOutput(scan_outputs, self.device)
        
    def __sub__(self, other: "ScanOutput") -> "ScanOutput":
        if other is None:
            return ScanOutput(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return ScanOutput(scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "ScanOutput":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs, self.device)
    
    def __iter__(self):
        yield self.scan_outputs
        
    def get_last_token(self) -> "ScanOutput":
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return ScanOutput(scan_outputs, self.device)
        
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, f"{path}")
    
    def mean(self) -> "ScanOutput":
        scan_output = self.scan_outputs
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=2).unsqueeze(2)
        return ScanOutput(scan_output, self.device)
    
    def to(self, device: str) -> "ScanOutput":
        scan_outputs = self.scan_outputs.to(device)
        return ScanOutput(scan_outputs, self.device)