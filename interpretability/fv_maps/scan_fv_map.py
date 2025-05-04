from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib import gridspec
import os

class ScanFVMap(FVMap):
    """
    ScanFVMap is a class that represents a function vector AIE map for the Mamba (or Mamba2) model.
    """
    def __init__(self, scan_map: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Args:
            scan_map (torch.Tensor): (n_layers, n_heads)
            dtype (torch.dtype, optional): device. Defaults to torch.float32.
        """
        self.scan_map = scan_map.to("cpu").to(dtype)
        self.dtype = dtype
        
    def __add__(self, other: "ScanFVMap") -> "ScanFVMap":
        return ScanFVMap(self.scan_map + other.scan_map, self.dtype)
    
    def __truediv__(self, other: int | float) -> "ScanFVMap":
        return ScanFVMap(self.scan_map / other, self.dtype)
    
    def visualize_on_axis(self, ax: plt.Axes) -> None:
        sns.heatmap(self.scan_map.to(torch.float32).numpy(), ax=ax, cmap="viridis")
        ax.set_title("Mamba Stream Function Vectors")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")
    
    def visualize_on_spec(self, spec: gridspec.SubplotSpec) -> None:
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=spec, wspace=0.5)
        ax = plt.subplot(gs_inner[0])
        self.visualize_on_axis(ax)
        
    def top_k_heads(self, k: int, **kwargs) -> list[tuple[int, int, str]]:
        top_k_indices = torch.topk(self.scan_map.flatten(), k).indices
        top_k_heads = {}
        for i in top_k_indices:
            if i in top_k_heads:
                top_k_heads[i].append({"head": i % self.scan_map.shape[1], "stream": "scan"})
            else:
                top_k_heads[i] = [{"head": i % self.scan_map.shape[1], "stream": "scan"}]
        return top_k_heads
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, ax = plt.subplots()
        self.visualize_on_axis(ax)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig