from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import gridspec
from matplotlib.gridspec import SubplotSpec
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
    
    def visualize_on_axis(self, ax1: Axes, ax2: Axes) -> None:
        sns.heatmap(self.attn_map.to(torch.float32).numpy(), ax=ax1, cmap="viridis", yticklabels=self.attn_layers)
        ax1.set_title("Attention Stream Function Vectors")
        ax1.set_xlabel("Heads")
        ax1.set_ylabel("Layers")
        sns.heatmap(self.scan_map.to(torch.float32).numpy(), ax=ax2, cmap="viridis", yticklabels=self.scan_layers)
        ax2.set_title("Mamba Stream Function Vectors")
        ax2.set_xlabel("Heads")
    
    def visualize_on_spec(self, spec: SubplotSpec) -> None:
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.15)
        ax1 = plt.subplot(gs_inner[0])
        ax2 = plt.subplot(gs_inner[1])
        self.visualize_on_axis(ax1, ax2)

    def top_k_heads(self, k: int, stream: str = None, **kwargs) -> list[tuple[int, int, str]]:
        """
        Get the top k heads overall in the attention map.
        
        Args:
            k (int): number of heads to return.
            stream (str, optional): stream to use, one of attn or scan. Defaults to None to use both streams.
            **kwargs: not used.
        
        Returns:
            list[tuple[int, int, str]]: list of tuples of (layer, head, stream) for the top k heads.
        """
        if stream is not None:
            if stream == "attn":
                map_ = self.attn_map.flatten()
            elif stream == "scan":
                map_ = self.scan_map.flatten()
            else:
                raise ValueError(f"Invalid stream: {stream}. Must be one of attn or scan.")
            top_k = torch.topk(map_, k).indices
            return [(i // self.attn_map.shape[1], i % self.attn_map.shape[1], stream) for i in top_k]
        else:
            map_ = torch.cat((self.attn_map.flatten(), self.scan_map.flatten()))
            stream_cutoff = self.attn_map.numel()
            top_k = torch.topk(map_, k).indices
            top_k_heads = []
            for i in top_k:
                if i < stream_cutoff:
                    layer = i // self.attn_map.shape[1]
                    head = i % self.attn_map.shape[1]
                    top_k_heads.append((layer, head, "attn"))
                else:
                    layer = (i - stream_cutoff) // self.scan_map.shape[1]
                    head = (i - stream_cutoff) % self.scan_map.shape[1]
                    top_k_heads.append((layer, head, "scan"))
            return top_k_heads
    
    def visualize(self, save_path: str = None) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.visualize_on_axis(ax1, ax2)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig