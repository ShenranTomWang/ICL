from abc import ABC, abstractmethod
import torch
from interpretability.fv_maps import FVMap

class AttentionManager(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    @staticmethod
    def mean_of(outputs: list["AttentionManager"]) -> "AttentionManager":
        if len(outputs) == 0:
            return None
        sum = outputs[0]
        for output in outputs[1:]:
            sum += output
        return sum / len(outputs)
    
    @abstractmethod
    def __add__(self, other: "AttentionManager") -> "AttentionManager":
        pass
    
    @abstractmethod
    def __truediv__(self, other: int) -> "AttentionManager":
        pass
    
    @abstractmethod
    def __sub__(self, other: "AttentionManager") -> "AttentionManager":
        pass
    
    @abstractmethod
    def __mul__(self, other: int) -> "AttentionManager":
        pass
    
    @abstractmethod
    def __rmul__(self, other: int) -> "AttentionManager":
        pass
    
    @abstractmethod
    def __iter__(self) -> tuple:
        pass
    
    @abstractmethod
    def get_last_token(self) -> "AttentionManager":
        pass
    
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, path)
    
    @abstractmethod
    def mean(self) -> "AttentionManager":
        pass
    
    @abstractmethod
    def to(self, device: str) -> "AttentionManager":
        pass
    
    @abstractmethod
    def top_p_heads(self, p: float, aie_map: FVMap) -> "AttentionManager":
        """
        Keep only top p of heads based on ranking in FVMap

        Args:
            p (float): percentage [0, 1]
            aie_map (FVMap): provides ranking of heads

        Returns:
            AttentionManager: AttentionManager object with only top p heads, others being None
        """
        pass
