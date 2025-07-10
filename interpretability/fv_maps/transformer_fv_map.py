from .fv_map import FVMap
import torch
import matplotlib as mpl; mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib import gridspec
import os, random

class TransformerFVMap(FVMap):
    """
    TransformerFVMap is a class that represents a function vector AIE map for the Transformer model.
    """
    def __init__(self, attn_map: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Args:
            attn_map (torch.Tensor): (n_layers, n_heads)
            dtype (torch.dtype, optional): device. Defaults to torch.float32.
        """
        self.attn_map = attn_map.to("cpu").to(dtype)
        super().__init__(total_heads=attn_map.numel(), dtype=dtype)

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
        
    def top_p_heads(self, p: int, **kwargs) -> dict:
        k = int(self.attn_map.numel() * p)
        top_k_indices = torch.topk(self.attn_map.flatten(), k).indices
        top_k_heads = {}
        for i in top_k_indices:
            layer = (i // self.attn_map.shape[1]).item()
            head = (i % self.attn_map.shape[1]).item()
            if i in top_k_heads:
                top_k_heads[layer].append({"head": head, "stream": "attn"})
            else:
                top_k_heads[layer] = [{"head": head, "stream": "attn"}]
        return top_k_heads
    
    def exclusion_ablation_heads(self, top_p: float, ablation_p: float, **kwargs) -> dict:
        k = int(self.attn_map.numel() * top_p)
        target_k = int(self.attn_map.numel() * ablation_p)
        all_indices = set(range(self.attn_map.numel()))
        top_k_indices = set(torch.topk(self.attn_map.flatten(), k).indices.tolist())
        available_indices = list(all_indices - top_k_indices)
        if target_k > len(available_indices):
            raise ValueError("ablation_p is too large, not enough available indices outside top_k.")
        indices = random.sample(available_indices, target_k)
        heads = {}
        for i in indices:
            layer = (i // self.attn_map.shape[1])
            head = (i % self.attn_map.shape[1])
            if layer in heads:
                heads[layer].append({"head": head, "stream": "attn"})
            else:
                heads[layer] = [{"head": head, "stream": "attn"}]
        return heads
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, ax = plt.subplots()
        self.visualize_on_axis(ax)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig