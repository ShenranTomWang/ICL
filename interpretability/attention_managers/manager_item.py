from abc import ABC, abstractmethod
import torch

class ManagerItem(list, ABC):
    """
    Abstract base class for attention manager items. This is a list of tensors. It supports basic arithmetic operations
    and device transfer for interpretability tasks.
    """
    @abstractmethod
    def __add__(self, other: "ManagerItem") -> "ManagerItem":
        pass
    
    @abstractmethod
    def __sub__(self, other: "ManagerItem") -> "ManagerItem":
        pass
    
    @abstractmethod
    def __truediv__(self, other: int) -> "ManagerItem":
        pass
    
    @abstractmethod
    def __mul__(self, other: int) -> "ManagerItem":
        pass
    
    @abstractmethod
    def __rmul__(self, other: int) -> "ManagerItem":
        pass
    
    @abstractmethod
    def get_last_token(self) -> "ManagerItem":
        pass
    
    @abstractmethod
    def to(self, device: str) -> "ManagerItem":
        pass
    
    def clone(self) -> "ManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.clone())
        return ManagerItem(mylist)

class GenericManagerItem(ManagerItem):
    """
    For generic use, this class is designed for attention outputs that are if shape (batch_size, seqlen, stream_dims...)
    """
    def __add__(self, other: "GenericManagerItem") -> "GenericManagerItem":
        if other is None:
            return self.clone()
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(other[i].clone())
            elif other[i] is None:
                mylist.append(item.clone())
            else:
                mylist.append(item + other[i])
        return GenericManagerItem(mylist)
    
    def __sub__(self, other: "GenericManagerItem") -> "GenericManagerItem":
        if other is None:
            return self.clone()
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(-1 * other[i])
            elif other[i] is None:
                mylist.append(item.clone())
            else:
                mylist.append(item - other[i])
        return GenericManagerItem(mylist)
    
    def __truediv__(self, other: int) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item / other)
        return GenericManagerItem(mylist)
    
    def __mul__(self, other: int) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item * other)
        return GenericManagerItem(mylist)
    
    def __rmul__(self, other: int) -> "GenericManagerItem":
        return self.__mul__(other)
    
    def get_last_token(self) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item[:, -1, ...].clone())
        return GenericManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return GenericManagerItem(mylist)
    
class MambaScanManagerItem(ManagerItem):
    """
    Since Mamba has a different attention output shape, we need to override the methods in the base class.
    This class is designed for attention outputs that are of shape (batch_size, stream_dims..., seqlen)
    """
    def __add__(self, other: "MambaScanManagerItem") -> "MambaScanManagerItem":
        if other is None:
            return self.clone()
        if not isinstance(other, MambaScanManagerItem):
            raise TypeError("Addition is only supported between MambaScanManagerItem objects")
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(other[i].clone())
            elif other[i] is None:
                mylist.append(item.clone())
            else:
                mylist.append(item + other[i])
        return MambaScanManagerItem(mylist)
    
    def __sub__(self, other: "MambaScanManagerItem") -> "MambaScanManagerItem":
        if other is None:
            return self.clone()
        if not isinstance(other, MambaScanManagerItem):
            raise TypeError("Subtraction is only supported between MambaScanManagerItem objects")
        mylist = []
        for i, item in enumerate(self):
            if item is None and other[i] is None:
                mylist.append(None)
            elif item is None:
                mylist.append(-1 * other[i])
            elif other[i] is None:
                mylist.append(item.clone())
            else:
                mylist.append(item - other[i])
        return MambaScanManagerItem(mylist)
    
    def __truediv__(self, other: int) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item / other)
        return MambaScanManagerItem(mylist)
    
    def __mul__(self, other: int) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item * other)
        return MambaScanManagerItem(mylist)
    
    def __rmul__(self, other: int) -> "MambaScanManagerItem":
        return self.__mul__(other)
    
    def get_last_token(self) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item[..., -1].clone())
        return MambaScanManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return MambaScanManagerItem(mylist)