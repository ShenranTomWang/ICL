from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

class ScanFVMap(FVMap):
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
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, ax = plt.subplots()
        sns.heatmap(self.scan_map.to(torch.float32).numpy(), ax=ax, cmap="viridis")
        ax.set_title("Mamba Stream Function Vectors")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")
        if save_path is not None:
            plt.savefig(save_path)
        return fig