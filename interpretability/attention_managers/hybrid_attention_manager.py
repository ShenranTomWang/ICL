import torch
from .attention_manager import AttentionManager
from .manager_item import AttentionManagerItem

class HybridAttentionManager(AttentionManager):
    def __init__(
        self,
        all_attns: list[torch.Tensor] | None,
        attn_outputs: list[torch.Tensor] | None,
        scan_outputs: list[torch.Tensor] | None,
        device: str = "cpu"
    ):
        self.all_attns = AttentionManagerItem(all_attns).to(device) if all_attns is not None else None
        self.attn_outputs = AttentionManagerItem(attn_outputs).to(device) if attn_outputs is not None else None
        self.scan_outputs = AttentionManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "HybridAttentionManager") -> "HybridAttentionManager":
        if other is None:
            return HybridAttentionManager(None, self.attn_outputs.clone(), self.scan_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs.clone()
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs.clone()
        return HybridAttentionManager(None, attn_outputs, scan_outputs, self.device)
        
    def __sub__(self, other: "HybridAttentionManager") -> "HybridAttentionManager":
        if other is None:
            return HybridAttentionManager(None, self.attn_outputs.clone(), self.scan_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else -1 * other.attn_outputs
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else -1 * other.scan_outputs
        return HybridAttentionManager(None, attn_outputs, scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return HybridAttentionManager(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return HybridAttentionManager(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return HybridAttentionManager(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __iter__(self):
        yield self.all_attns
        yield self.attn_outputs
        yield self.scan_outputs
        
    def get_last_token(self) -> "HybridAttentionManager":
        all_attns = self.all_attns.get_last_token() if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.get_last_token() if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return HybridAttentionManager(all_attns, attn_outputs, scan_outputs, self.device)
    
    def mean(self) -> "HybridAttentionManager":
        all_attn, attn_output, scan_output = self.all_attns.clone(), self.attn_outputs.clone(), self.scan_outputs.clone()
        for i, attn_i in enumerate(attn_output):
            if attn_i is not None:
                attn_output[i] = attn_i.mean(dim=1)
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=1)
        return HybridAttentionManager(all_attn, attn_output, scan_output, self.device)
    
    def to(self, device: str | torch.DeviceObjType) -> "HybridAttentionManager":
        all_attns = self.all_attns.to(device) if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.to(device) if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs.to(device) if self.scan_outputs is not None else None
        return HybridAttentionManager(all_attns, attn_outputs, scan_outputs, device)