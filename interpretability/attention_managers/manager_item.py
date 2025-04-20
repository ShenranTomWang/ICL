from abc import ABC, abstractmethod
import torch

class ManagerItem(list, ABC):
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

class AttentionManagerItem(ManagerItem):
    def __add__(self, other: "AttentionManagerItem") -> "AttentionManagerItem":
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
        return AttentionManagerItem(mylist)
    
    def __sub__(self, other: "AttentionManagerItem") -> "AttentionManagerItem":
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
                mylist.append(item[:, -1, ...].clone())
        return AttentionManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "AttentionManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return AttentionManagerItem(mylist)
    
class MambaScanManagerItem(ManagerItem):
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