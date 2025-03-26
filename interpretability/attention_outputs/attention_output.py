from abc import ABC, abstractmethod
import torch

class AttentionOutput(ABC):
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    @staticmethod
    def mean_of(outputs: list["AttentionOutput"]) -> "AttentionOutput":
        if len(outputs) == 0:
            return None
        sum = outputs[0]
        for output in outputs[1:]:
            sum += output
        return sum / len(outputs)
    
    @abstractmethod
    def __add__(self, other: "AttentionOutput") -> "AttentionOutput":
        pass
    
    @abstractmethod
    def __truediv__(self, other: int) -> "AttentionOutput":
        pass
    
    @abstractmethod
    def __sub__(self, other: "AttentionOutput") -> "AttentionOutput":
        pass
    
    @abstractmethod
    def __mul__(self, other: int) -> "AttentionOutput":
        pass
    
    @abstractmethod
    def __rmul__(self, other: int) -> "AttentionOutput":
        pass
    
    @abstractmethod
    def __iter__(self):
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def mean(self) -> "AttentionOutput":
        pass
    
    @abstractmethod
    def to(self, device: str) -> "AttentionOutput":
        pass

class AttentionOutputItem(list):
    def __add__(self, other: "AttentionOutputItem") -> "AttentionOutputItem":
        if other is None:
            return self
        if not isinstance(other, AttentionOutputItem):
            raise TypeError("Addition is only supported between AttentionOutputItem objects")
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
        return AttentionOutputItem(mylist)
    
    def __sub__(self, other: "AttentionOutputItem") -> "AttentionOutputItem":
        if other is None:
            import pdb; pdb.set_trace()
            return self
        if not isinstance(other, AttentionOutputItem):
            raise TypeError("Subtraction is only supported between AttentionOutputItem objects")
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
        return AttentionOutputItem(mylist)
    
    def __truediv__(self, other: int) -> "AttentionOutputItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item / other)
        return AttentionOutputItem(mylist)
    
    def __mul__(self, other: int) -> "AttentionOutputItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item * other)
        return AttentionOutputItem(mylist)
    
    def __rmul__(self, other: int) -> "AttentionOutputItem":
        return self.__mul__(other)
    
    def to(self, device: str | torch.DeviceObjType) -> "AttentionOutputItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return AttentionOutputItem(mylist)