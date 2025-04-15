import torch
from .attention_manager import AttentionManager
from .manager_item import MambaScanManagerItem, AttentionManagerItem

class MambaScanManager(AttentionManager):
    def __init__(self, scan_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.scan_outputs = MambaScanManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "MambaScanManager") -> "MambaScanManager":
        if other is None:
            return MambaScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return MambaScanManager(scan_outputs, self.device)
        
    def __sub__(self, other: "MambaScanManager") -> "MambaScanManager":
        if other is None:
            return MambaScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return MambaScanManager(scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "MambaScanManager":
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return MambaScanManager(scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "MambaScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return MambaScanManager(scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "MambaScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return MambaScanManager(scan_outputs, self.device)
    
    def __iter__(self):
        yield self.scan_outputs
        
    def get_last_token(self) -> "MambaScanManager":
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return MambaScanManager(scan_outputs, self.device)
    
    def mean(self) -> "MambaScanManager":
        scan_output = self.scan_outputs
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=2).unsqueeze(2)
        return MambaScanManager(scan_output, self.device)
    
    def to(self, device: str) -> "MambaScanManager":
        scan_outputs = self.scan_outputs.to(device)
        return MambaScanManager(scan_outputs, device)
    
class Mamba2ScanManager(AttentionManager):
    def __init__(self, scan_outputs: list[torch.Tensor] | None, device: torch.DeviceObjType = "cpu"):
        self.scan_outputs = AttentionManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "Mamba2ScanManager") -> "Mamba2ScanManager":
        if other is None:
            return Mamba2ScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs
        return Mamba2ScanManager(scan_outputs, self.device)
        
    def __sub__(self, other: "Mamba2ScanManager") -> "Mamba2ScanManager":
        if other is None:
            return Mamba2ScanManager(self.scan_outputs, self.device)
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else [-1 * attn for attn in other.scan_outputs]
        return Mamba2ScanManager(scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "Mamba2ScanManager":
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return Mamba2ScanManager(scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "Mamba2ScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return Mamba2ScanManager(scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "Mamba2ScanManager":
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return Mamba2ScanManager(scan_outputs, self.device)
    
    def __iter__(self):
        yield self.scan_outputs
        
    def get_last_token(self) -> "Mamba2ScanManager":
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return Mamba2ScanManager(scan_outputs, self.device)
    
    def mean(self) -> "Mamba2ScanManager":
        scan_output = self.scan_outputs
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=2).unsqueeze(2)
        return Mamba2ScanManager(scan_output, self.device)
    
    def to(self, device: str) -> "Mamba2ScanManager":
        scan_outputs = self.scan_outputs.to(device)
        return Mamba2ScanManager(scan_outputs, device)
