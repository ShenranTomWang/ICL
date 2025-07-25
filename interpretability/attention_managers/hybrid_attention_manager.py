import torch
from .attention_manager import AttentionManager
from .manager_item import GenericManagerItem, MambaScanManagerItem, ManagerItem

class HybridAttentionManager(AttentionManager):
    """
    This is the manager class for hybrid models, where we have both self-attention and Mamba streams
    """
    def __init__(
        self,
        all_attns: ManagerItem | None,
        attn_outputs: ManagerItem | None,
        scan_outputs: ManagerItem | None,
        device: str = "cpu"
    ):
        self.all_attns = all_attns.to(device) if all_attns is not None else None
        self.attn_outputs = attn_outputs.to(device) if attn_outputs is not None else None
        self.scan_outputs = scan_outputs.to(device) if scan_outputs is not None else None
        super().__init__(device)
        
    def __add__(self, other: "HybridAttentionManager") -> "HybridAttentionManager":
        if other is None:
            return self.__class__(None, self.attn_outputs.clone(), self.scan_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs + other.attn_outputs if self.attn_outputs is not None else other.attn_outputs.clone()
        scan_outputs = self.scan_outputs + other.scan_outputs if self.scan_outputs is not None else other.scan_outputs.clone()
        return self.__class__(None, attn_outputs, scan_outputs, self.device)
        
    def __sub__(self, other: "HybridAttentionManager") -> "HybridAttentionManager":
        if other is None:
            return self.__class__(None, self.attn_outputs.clone(), self.scan_outputs.clone(), self.device)
        attn_outputs = self.attn_outputs - other.attn_outputs if self.attn_outputs is not None else -1 * other.attn_outputs
        scan_outputs = self.scan_outputs - other.scan_outputs if self.scan_outputs is not None else -1 * other.scan_outputs
        return self.__class__(None, attn_outputs, scan_outputs, self.device)
    
    def __truediv__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns / other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs / other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs / other if self.scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __mul__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __rmul__(self, other: int) -> "HybridAttentionManager":
        all_attns = self.all_attns * other if self.all_attns is not None else None
        attn_outputs = self.attn_outputs * other if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs * other if self.scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, self.device)
    
    def __iter__(self):
        yield self.all_attns
        yield self.attn_outputs
        yield self.scan_outputs
        
    def get_last_token(self) -> "HybridAttentionManager":
        all_attns = self.all_attns.get_last_token() if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.get_last_token() if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs.get_last_token() if self.scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, self.device)
    
    def get_stream(self, stream: str) -> torch.Tensor:
        if stream == "attn":
            return self.attn_outputs.clone() if self.attn_outputs is not None else None
        elif stream == "scan":
            return self.scan_outputs.clone() if self.scan_outputs is not None else None
        else:
            raise ValueError(f"Unknown stream: {stream}")
    
    def mean(self) -> "HybridAttentionManager":
        all_attn, attn_output, scan_output = self.all_attns.clone(), self.attn_outputs.clone(), self.scan_outputs.clone()
        for i, attn_i in enumerate(attn_output):
            if attn_i is not None:
                attn_output[i] = attn_i.mean(dim=1)
        for i, scan_i in enumerate(scan_output):
            if scan_i is not None:
                scan_output[i] = scan_i.mean(dim=1)
        return self.__class__(all_attn, attn_output, scan_output, self.device)
    
    def to(self, device: str | torch.DeviceObjType) -> "HybridAttentionManager":
        all_attns = self.all_attns.to(device) if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.to(device) if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs.to(device) if self.scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, device)
    
    @classmethod
    def zeros_like(cls, other: "HybridAttentionManager") -> "HybridAttentionManager":
        all_attns = GenericManagerItem.zeros_like(other.all_attns) if other.all_attns is not None else None
        attn_outputs = GenericManagerItem.zeros_like(other.attn_outputs) if other.attn_outputs is not None else None
        scan_outputs = GenericManagerItem.zeros_like(other.scan_outputs) if other.scan_outputs is not None else None
        return other.__class__(all_attns, attn_outputs, scan_outputs, other.device)
    
    def set_head_values(self, head_values: "HybridAttentionManager", head_indices: dict) -> "HybridAttentionManager":
        all_attns = self.all_attns.clone() if self.all_attns is not None else None
        attn_outputs = self.attn_outputs.clone() if self.attn_outputs is not None else None
        scan_outputs = self.scan_outputs.clone() if self.scan_outputs is not None else None
        attn_outputs = attn_outputs.set_head_values(head_values.attn_outputs, head_indices, "attn") if attn_outputs is not None else None
        scan_outputs = scan_outputs.set_head_values(head_values.scan_outputs, head_indices, "scan") if scan_outputs is not None else None
        return self.__class__(all_attns, attn_outputs, scan_outputs, self.device)
        
class HybridMambaAttentionManager(HybridAttentionManager):
    """
    This is the manager class for hybrid models, where we have both self-attention and Mamba streams
    """
    def __init__(
        self,
        all_attns: list[torch.Tensor] | None,
        attn_outputs: list[torch.Tensor] | None,
        scan_outputs: list[torch.Tensor] | None,
        device: str = "cpu"
    ):
        all_attns = GenericManagerItem(all_attns).to(device) if all_attns is not None else None
        attn_outputs = GenericManagerItem(attn_outputs).to(device) if attn_outputs is not None else None
        scan_outputs = MambaScanManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(all_attns=all_attns, attn_outputs=attn_outputs, scan_outputs=scan_outputs, device=device)
        
class HybridMamba2AttentionManager(HybridAttentionManager):
    """
    This is the manager class for hybrid models, where we have both self-attention and Mamba2 streams
    """
    def __init__(
        self,
        all_attns: list[torch.Tensor] | None,
        attn_outputs: list[torch.Tensor] | None,
        scan_outputs: list[torch.Tensor] | None,
        device: str = "cpu"
    ):
        all_attns = GenericManagerItem(all_attns).to(device) if all_attns is not None else None
        attn_outputs = GenericManagerItem(attn_outputs).to(device) if attn_outputs is not None else None
        scan_outputs = GenericManagerItem(scan_outputs).to(device) if scan_outputs is not None else None
        super().__init__(all_attns=all_attns, attn_outputs=attn_outputs, scan_outputs=scan_outputs, device=device)