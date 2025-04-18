from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
import os

class HybridFVMap(FVMap):
    def __init__(
        self,
        attn_map: torch.Tensor,
        scan_map: torch.Tensor,
        attn_layers: list[int] = None,
        scan_layers: list[int] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            attn_map (torch.Tensor): (n_layers, n_heads)
            scan_map (torch.Tensor): (n_layers, n_heads)
            attn_layers (list[int], optional): list of layers that has attention. Defaults to None to use same indexing as attn_map.
            scan_layers (list[int], optional): list of layers that has scan. Defaults to None to use same indexing as scan_map.
            dtype (torch.dtype, optional): device. Defaults to torch.float32.
        """
        self.attn_map = attn_map.to("cpu").to(dtype)
        self.scan_map = scan_map.to("cpu").to(dtype)
        self.attn_layers = attn_layers if attn_layers is not None else list(range(attn_map.shape[0]))
        self.scan_layers = scan_layers if scan_layers is not None else list(range(scan_map.shape[0]))
        self.dtype = dtype
        
    def __add__(self, other: "HybridFVMap") -> "HybridFVMap":
        attn_map = self.attn_map + other.attn_map
        scan_map = self.scan_map + other.scan_map
        return HybridFVMap(attn_map, scan_map, self.attn_layers, self.scan_layers, self.dtype)
    
    def __truediv__(self, other: int | float) -> "HybridFVMap":
        attn_map = self.attn_map / other
        scan_map = self.scan_map / other
        return HybridFVMap(attn_map, scan_map, self.attn_layers, self.scan_layers, self.dtype)
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(self.attn_map.to(torch.float32).numpy(), ax=ax1, cmap="viridis", yticklabels=self.attn_layers)
        ax1.set_title("Attention Stream Function Vectors")
        ax1.set_xlabel("Heads")
        ax1.set_ylabel("Layers")
        sns.heatmap(self.scan_map.to(torch.float32).numpy(), ax=ax2, cmap="viridis", yticklabels=self.scan_layers)
        ax2.set_title("Mamba Stream Function Vectors")
        ax2.set_xlabel("Heads")
        ax2.set_ylabel("Layers")
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig