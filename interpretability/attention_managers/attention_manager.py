from abc import ABC, abstractmethod
import torch

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

class AttentionManagerItem(list):
    def __add__(self, other: "AttentionManagerItem") -> "AttentionManagerItem":
        if other is None:
            return self
        if not isinstance(other, AttentionManagerItem):
            raise TypeError("Addition is only supported between AttentionManagerItem objects")
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(other[i])
            elif other[i] is None:
                mylist.append(item)
            else:
                mylist.append(item + other[i])
        return AttentionManagerItem(mylist)
    
    def __sub__(self, other: "AttentionManagerItem") -> "AttentionManagerItem":
        if other is None:
            import pdb; pdb.set_trace()
            return self
        if not isinstance(other, AttentionManagerItem):
            raise TypeError("Subtraction is only supported between AttentionManagerItem objects")
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(-1 * other[i])
            elif other[i] is None:
                mylist.append(item)
            else:
                mylist.append(item - other[i])
        return AttentionManagerItem(mylist)
    
    def __truediv__(self, other: int) -> "AttentionManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item / other)
        return AttentionManagerItem(mylist)
    
    def __mul__(self, other: int) -> "AttentionManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item * other)
        return AttentionManagerItem(mylist)
    
    def __rmul__(self, other: int) -> "AttentionManagerItem":
        return self.__mul__(other)
    
    def get_last_token(self) -> "AttentionManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item[..., -1, :])
        return AttentionManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "AttentionManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return AttentionManagerItem(mylist)