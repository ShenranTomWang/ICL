from .fv_map import FVMap
import torch
import matplotlib as mpl; mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib import gridspec
import os, random

class ScanFVMap(FVMap):
    """
    ScanFVMap is a class that represents a function vector AIE map for the Mamba (or Mamba2) model.
    """
    def __init__(self, scan_map: torch.Tensor, dtype: torch.dtype = torch.float32, figsize: tuple[int, int] = (10, 10)):
        """
        Args:
            scan_map (torch.Tensor): (n_layers, n_heads)
            dtype (torch.dtype, optional): device. Defaults to torch.float32.
            figsize (tuple[int, int], optional): figure size for visualization. Defaults to (10, 10).
        """
        self.scan_map = scan_map.to("cpu").to(dtype)
        super().__init__(total_heads=scan_map.numel(), dtype=dtype, figsize=figsize)

    def __add__(self, other: "ScanFVMap") -> "ScanFVMap":
        if not hasattr(self, "figsize"):
            self.figsize = (10, 10)
        return ScanFVMap(self.scan_map + other.scan_map, self.dtype, self.figsize)
    
    def __truediv__(self, other: int | float) -> "ScanFVMap":
        if not hasattr(self, "figsize"):
            self.figsize = (10, 10)
        return ScanFVMap(self.scan_map / other, self.dtype, self.figsize)

    def visualize_on_axis(self, ax: plt.Axes) -> None:
        hm = sns.heatmap(self.scan_map.to(torch.float32).numpy(), ax=ax, cmap="viridis")
        hm.set_yticklabels(hm.get_yticklabels(), fontsize=12)
        hm.set_xticklabels(hm.get_xticklabels(), fontsize=12)
        hm.collections[0].colorbar.ax.tick_params(labelsize=16)
        ax.set_title("Mamba Stream", fontsize=28)
        ax.set_xlabel("Heads", fontsize=22)
        ax.set_ylabel("Layers", fontsize=22)

    def visualize_on_spec(self, spec: gridspec.SubplotSpec) -> None:
        gs_inner = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=spec, wspace=0.5)
        ax = plt.subplot(gs_inner[0])
        self.visualize_on_axis(ax)
        
    def top_p_heads(self, p: int, **kwargs) -> dict:
        k = int(self.scan_map.numel() * p)
        top_k_indices = torch.topk(self.scan_map.flatten(), k).indices
        top_k_heads = {}
        for i in top_k_indices:
            layer = (i // self.scan_map.shape[1]).item()
            head = (i % self.scan_map.shape[1]).item()
            if layer in top_k_heads:
                top_k_heads[layer].append({"head": head, "stream": "scan"})
            else:
                top_k_heads[layer] = [{"head": head, "stream": "scan"}]
        return top_k_heads

    def exclusion_ablation_heads(self, top_p: float, ablation_p: float, **kwargs) -> dict:
        k = int(self.scan_map.numel() * top_p)
        target_k = int(self.scan_map.numel() * ablation_p)
        all_indices = set(range(self.scan_map.numel()))
        top_k_indices = set(torch.topk(self.scan_map.flatten(), k).indices.tolist())
        available_indices = list(all_indices - top_k_indices)
        if target_k > len(available_indices):
            raise ValueError("ablation_p is too large, not enough available indices outside top_k.")
        indices = random.sample(available_indices, target_k)
        heads = {}
        for i in indices:
            layer = (i // self.scan_map.shape[1])
            head = (i % self.scan_map.shape[1])
            if layer in heads:
                heads[layer].append({"head": head, "stream": "scan"})
            else:
                heads[layer] = [{"head": head, "stream": "scan"}]
        return heads

    def visualize(self, title: str = None, save_path: str = None) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        self.visualize_on_axis(ax)
        if title is not None:
            fig.suptitle(title, fontsize=48)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig