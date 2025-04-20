from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib import gridspec
import os

class TransformerFVMap(FVMap):
    def __init__(self, attn_map: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Args:
            attn_map (torch.Tensor): (n_layers, n_heads)
            dtype (torch.dtype, optional): device. Defaults to torch.float32.
        """
        self.attn_map = attn_map.to("cpu").to(dtype)
        self.dtype = dtype
        
    def __add__(self, other: "TransformerFVMap") -> "TransformerFVMap":
        return TransformerFVMap(self.attn_map + other.attn_map, self.dtype)
    
    def __truediv__(self, other: int | float) -> "TransformerFVMap":
        return TransformerFVMap(self.attn_map / other, self.dtype)
    
    def visualize_on_axis(self, ax: plt.Axes) -> None:
        sns.heatmap(self.attn_map.to(torch.float32).numpy(), ax=ax, cmap="viridis")
        ax.set_title("Attention Stream Function Vectors")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")
    
    def visualize_on_spec(self, spec: gridspec.SubplotSpec) -> None:
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=spec, wspace=0.1)
        ax = plt.subplot(gs_inner[0])
        self.visualize_on_axis(ax)
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, ax = plt.subplots()
        self.visualize_on_axis(ax)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig