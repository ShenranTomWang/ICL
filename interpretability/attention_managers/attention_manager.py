from abc import ABC, abstractmethod
import torch

class AttentionManager(ABC):
    """
    AttentionManager manages the attention values of a model (e.g. attention map, attention output). 
    It supports basic arithmetics and device transfer of tensors, as well as some other operations required for interpretability.
    """
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
        """Returns the last token of the attention manager
        Returns:
            AttentionManager: last token
        """
        pass
    
    @abstractmethod
    def get_stream(self, stream: str) -> torch.Tensor:
        """Returns the stream of the attention manager
        Args:
            stream (str): name of the stream ("attn" or "scan")
        Returns:
            torch.Tensor: stream of the attention manager
        """
        pass
    
    def save(self, path: str) -> None:
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self, path)
    
    @abstractmethod
    def mean(self) -> "AttentionManager":
        """Computes mean along sequence dimension

        Returns:
            AttentionManager: mean
        """
        pass
    
    @abstractmethod
    def to(self, device: str) -> "AttentionManager":
        pass
