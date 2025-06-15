from .fv_map import FVMap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import gridspec
from matplotlib.gridspec import SubplotSpec
import os
import random

class HybridFVMap(FVMap):
    """
    HybridFVMap is a class that represents a hybrid attention and Mamba (or Mamba2) function vector AIE map.
    """
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

    def top_p_heads(self, p: float, stream: str = None, **kwargs) -> dict:
        if stream is not None:
            if stream == "attn":
                map_ = self.attn_map
                layers = self.attn_layers
                k = int(self.attn_map.numel() * p)
            elif stream == "scan":
                map_ = self.scan_map
                layers = self.scan_layers
                k = int(self.scan_map.numel() * p)
            else:
                raise ValueError(f"Invalid stream: {stream}. Must be one of attn or scan.")
            top_k = torch.topk(map_.flatten(), k).indices
            top_k_heads = {}
            for i in top_k:
                layer = (i // map_.shape[1]).item()
                layer = layers[layer] if layers is not None else layer
                head = (i % map_.shape[1]).item()
                if layer in top_k_heads:
                    top_k_heads[layer].append({"head": head, "stream": stream})
                else:
                    top_k_heads[layer] = [{"head": head, "stream": stream}]
            return top_k_heads
        else:
            map_ = torch.cat((self.attn_map.flatten(), self.scan_map.flatten()))
            stream_cutoff = self.attn_map.numel()
            k = int(map_.numel() * p)
            top_k = torch.topk(map_, k).indices
            top_k_heads = {}
            for i in top_k:
                if i < stream_cutoff:
                    layer = (i // self.attn_map.shape[1]).item()
                    layer = self.attn_layers[layer] if self.attn_layers is not None else layer
                    head = (i % self.attn_map.shape[1]).item()
                    stream = "attn"
                else:
                    layer = ((i - stream_cutoff) // self.scan_map.shape[1]).item()
                    layer = self.scan_layers[layer] if self.scan_layers is not None else layer
                    head = ((i - stream_cutoff) % self.scan_map.shape[1]).item()
                    stream = "scan"
                if layer in top_k_heads:
                    top_k_heads[layer].append({"head": head, "stream": stream})
                else:
                    top_k_heads[layer] = [{"head": head, "stream": stream}]
            return top_k_heads

    def exclusion_ablation_heads(self, top_p: float, ablation_p: float, stream: str = None, **kwargs) -> dict:
        if stream is not None:
            if stream == "attn":
                map_ = self.attn_map
                layers = self.attn_layers
                k = int(self.attn_map.numel() * top_p)
                target_k = int(self.attn_map.numel() * ablation_p)
                all_indices = set(range(self.attn_map.numel()))
            elif stream == "scan":
                map_ = self.scan_map
                layers = self.scan_layers
                k = int(self.scan_map.numel() * top_p)
                target_k = int(self.scan_map.numel() * ablation_p)
                all_indices = set(range(self.scan_map.numel()))
            else:
                raise ValueError(f"Invalid stream: {stream}. Must be one of attn or scan.")
            top_k = set(torch.topk(map_.flatten(), k).indices.tolist())
            available_indices = list(all_indices - top_k)
            if target_k > len(available_indices):
                raise ValueError("ablation_p is too large, not enough available indices outside top_k.")
            indices = random.sample(available_indices, target_k)
            top_k_heads = {}
            for i in indices:
                layer = (i // map_.shape[1])
                layer = layers[layer] if layers is not None else layer
                head = (i % map_.shape[1])
                if layer in top_k_heads:
                    top_k_heads[layer].append({"head": head, "stream": stream})
                else:
                    top_k_heads[layer] = [{"head": head, "stream": stream}]
            return top_k_heads
        else:
            map_ = torch.cat((self.attn_map.flatten(), self.scan_map.flatten()))
            stream_cutoff = self.attn_map.numel()
            k = int(map_.numel() * top_p)
            target_k = int(map_.numel() * ablation_p)
            top_k = set(torch.topk(map_, k).indices.tolist())
            all_indices = set(range(map_.numel()))
            available_indices = list(all_indices - top_k)
            if target_k > len(available_indices):
                raise ValueError("ablation_p is too large, not enough available indices outside top_k.")
            indices = random.sample(available_indices, target_k)
            heads = {}
            for i in indices:
                if i < stream_cutoff:
                    layer = (i // self.attn_map.shape[1])
                    layer = self.attn_layers[layer] if self.attn_layers is not None else layer
                    head = (i % self.attn_map.shape[1])
                    stream_i = "attn"
                else:
                    layer = ((i - stream_cutoff) // self.scan_map.shape[1])
                    layer = self.scan_layers[layer] if self.scan_layers is not None else layer
                    head = ((i - stream_cutoff) % self.scan_map.shape[1])
                    stream_i = "scan"
                if layer in heads:
                    heads[layer].append({"head": head, "stream": stream_i})
                else:
                    heads[layer] = [{"head": head, "stream": stream_i}]
            return heads

    def visualize(self, save_path: str = None) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.visualize_on_axis(ax1, ax2)
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        return fig