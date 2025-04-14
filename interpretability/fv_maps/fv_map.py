from abc import ABC, abstractmethod
from matplotlib.figure import Figure

class FVMap(ABC):
    @staticmethod
    def mean_of(maps: list["FVMap"]) -> "FVMap":
        """
        Compute the mean of a list of FVMap objects.
        
        Args:
            maps (list[FVMap]): List of FVMap objects.
        
        Returns:
            FVMap: A new FVMap object that is the mean of the input maps.
        """
        return sum(maps) / len(maps)
    
    @abstractmethod
    def __add__(self, other: "FVMap") -> "FVMap":
        """
        Add two FVMap objects together.
        
        Args:
            other (FVMap): The FVMap object to add.
        
        Returns:
            FVMap: A new FVMap object that is the sum of the two.
        """
        pass
    
    @abstractmethod
    def __truediv__(self, other: int | float) -> "FVMap":
        """
        Divide self by a number
        
        Args:
            other (number): The number to divide by.
        
        Returns:
            FVMap: A new FVMap object that is the result of the division.
        """
        pass
    
    @abstractmethod
    def visualize(self, save_path: str = None) -> Figure:
        """
        Visualize the FVMap

        Args:
            save_path (str, optional): path to save figure to. Defaults to None for not saving.
            
        Returns:
            Figure: The matplotlib figure object.
        """
        pass