import torch
from .attention_manager import AttentionManager, AttentionManagerItem

class ScanManager(AttentionManager):
    def __init__(self, scan_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.scan_outputs = AttentionManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "ScanManager") -> "ScanManager":
        if other is None:
            return ScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return ScanManager(scan_outputs, self.device)
        
    def __sub__(self, other: "ScanManager") -> "ScanManager":
        if other is None:
            return ScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return ScanManager(scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "ScanManager":
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return ScanManager(scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "ScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanManager(scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "ScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return ScanManager(scan_outputs, self.device)
    
    def __iter__(self):
        yield self.scan_outputs
        
    def get_last_token(self) -> "ScanManager":
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return ScanManager(scan_outputs, self.device)
    
    def mean(self) -> "ScanManager":
        scan_output = self.scan_outputs
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=2).unsqueeze(2)
        return ScanManager(scan_output, self.device)
    
    def to(self, device: str) -> "ScanManager":
        scan_outputs = self.scan_outputs.to(device)
        return ScanManager(scan_outputs, self.device)