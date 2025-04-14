from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

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
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, ax = plt.subplots()
        sns.heatmap(self.attn_map.numpy(), ax=ax, cmap="viridis")
        ax.set_title("Attention Stream Function Vectors")
        ax.set_xlabel("Heads")
        ax.set_ylabel("Layers")
        if save_path:
            plt.savefig(save_path)
        return fig