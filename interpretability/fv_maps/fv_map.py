from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

class FVMap(ABC):
    """
    This is an abstract base class for AIE heatmaps for the function vector experiments.
    It defines the interface for FVMap objects, which are used to visualize and analyze the function vector maps.
    """
    def __init__(self, total_heads: int, dtype: torch.dtype):
        self.total_heads = total_heads
        self.dtype = dtype

    @staticmethod
    def mean_of(maps: list["FVMap"]) -> "FVMap":
        """
        Compute the mean of a list of FVMap objects.
        
        Args:
            maps (list[FVMap]): List of FVMap objects.
        
        Returns:
            FVMap: A new FVMap object that is the mean of the input maps.
        """
        mean_map = maps[0]
        for map_ in maps[1:]:
            mean_map += map_
        return mean_map / len(maps)
    
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
    
    @abstractmethod
    def visualize_on_spec(self, spec: gridspec.SubplotSpec) -> None:
        """
        Visualize the FVMap on a given global axis.
        
        Args:
            spec (gridspec.SubplotSpec): The axis to plot on.
        """
        pass
    
    @abstractmethod
    def top_p_heads(self, k: int, **kwargs) -> dict:
        """
        Get the top p heads for the FVMap.
        
        Args:
            p (int): The proportion of heads to retrieve.
            kwargs: Additional arguments for the method.
        
        Returns:
            dict[int: {head: int, stream: str}]: at each layer, the head and stream, stream is either "attn" or "scan".
        """
        pass
    
    @abstractmethod
    def exclusion_ablation_heads(self, top_p: float, ablation_p: float, **kwargs) -> dict:
        """
        Get the heads for exclusion ablation.

        Args:
            top_p (float): The proportion of heads to exclude.
            ablation_p (float): The proportion of heads to select for ablation.
            kwargs: Additional arguments for the method.

        Returns:
            dict[int: {head: int, stream: str}]: at each layer, the head and stream, stream is either "attn" or "scan".
        """
        pass
    
    @staticmethod
    def visualize_all(maps: list["FVMap"], titles: list[str] = None, save_path: str = None) -> Figure:
        """
        Visualize all FVMaps in a list in a single, lossless figure.
        
        Args:
            maps (list[FVMap]): List of FVMap objects.
            titles (list[str], optional): List of titles for each map. Defaults to Map i.
            save_path (str, optional): Path to save the figure.
        
        Returns:
            Figure: The composite matplotlib figure.
        """
        width = int(np.sqrt(len(maps)))
        height = len(maps) // width + (len(maps) % width > 0)
        fig = plt.figure(figsize=(width * 14, height * 12))
        gs = gridspec.GridSpec(height, width, figure=fig)

        if titles is None:
            titles = [f"Map {i}" for i in range(n)]
        
        for i, map_ in enumerate(maps):
            row, col = divmod(i, width)
            spec = gs[row, col]
            map_.visualize_on_spec(spec)
            fig.text(
                x = (col + 0.5) / width,          # Horizontal center of the cell
                y = 1 - (row / height) + 0.001,    # Slightly above the top of the cell
                s=titles[i],
                ha="center", va="bottom", fontsize=24, weight="bold"
            )
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")

        return fig