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
        return self.__class__(mylist)

    @classmethod
    @abstractmethod
    def zeros_like(cls, other: "ManagerItem") -> "ManagerItem":
        """
        Create a new ManagerItem with the same shape as the other, filled with zeros.
        Args:
            other (ManagerItem): The item to match the shape of.
        Returns:
            ManagerItem: A new item filled with zeros.
        """
        pass
    
    @abstractmethod
    def set_head_values(self, head_values: "ManagerItem", head_indices: dict, stream: str) -> "ManagerItem":
        """
        Set the head values in the manager item.
        Args:
            head_values (ManagerItem): The head values to set.
            head_indices (dict): The indices of the heads to set.
            stream (str): The stream to set the head values for.
        Returns:
            ManagerItem: A new item with the head values set.
        """
        pass

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
    
    def get_token(self, index: int) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item[:, index, ...].clone())
        return GenericManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "GenericManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return GenericManagerItem(mylist)
    
    def set_head_values(self, head_values: "GenericManagerItem", head_indices: dict, stream: str) -> "GenericManagerItem":
        mylist = self.clone()
        for layer in range(len(mylist)):
            if layer in head_indices:
                heads = head_indices[layer]
                for head in heads:
                    head_idx =  head["head"]
                    _stream = head["stream"]
                    if _stream == stream:
                        item = head_values[layer].clone()
                        mylist[layer][:, head_idx, :] = item[:, head_idx, :]
        return GenericManagerItem(mylist)
    
    @classmethod
    def zeros_like(cls, other: "GenericManagerItem") -> "GenericManagerItem":
        """
        Create a new GenericManagerItem with the same shape as the other, filled with zeros.
        Args:
            other (GenericManagerItem): The item to match the shape of.
        Returns:
            GenericManagerItem: A new item filled with zeros.
        """
        mylist = []
        for item in other:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(torch.zeros_like(item))
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
    
    def get_token(self, index: int) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item[..., index].clone())
        return MambaScanManagerItem(mylist)
    
    def to(self, device: str | torch.DeviceObjType) -> "MambaScanManagerItem":
        mylist = []
        for item in self:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(item.to(device))
        return MambaScanManagerItem(mylist)
    
    @classmethod
    def zeros_like(cls, other: "MambaScanManagerItem") -> "MambaScanManagerItem":
        """
        Create a new MambaScanManagerItem with the same shape as the other, filled with zeros.
        Args:
            other (MambaScanManagerItem): The item to match the shape of.
        Returns:
            MambaScanManagerItem: A new item filled with zeros.
        """
        mylist = []
        for item in other:
            if item is None:
                mylist.append(None)
            else:
                mylist.append(torch.zeros_like(item))
        return MambaScanManagerItem(mylist)
    
    def set_head_values(self, head_values: "MambaScanManagerItem", head_indices: dict, stream: str) -> "MambaScanManagerItem":
        mylist = self.clone()
        for layer in range(len(mylist)):
            if layer in head_indices:
                heads = head_indices[layer]
                for head in heads:
                    _stream = head["stream"]
                    if _stream == stream:
                        item = head_values[layer].clone()
                        mylist[layer] = item
        return MambaScanManagerItem(mylist)