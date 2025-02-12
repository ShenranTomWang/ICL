import torch
import matplotlib.pyplot as plt
import seaborn as sns

def draw_heatmap(tensor: torch.Tensor, title: str, axis=None):
    """
    Draws a heatmap of the tensor.
    args:
        tensor: The tensor to draw the heatmap of. Must have dimensionality of 2
        title: The title of the heatmap.
        axis: The axis to draw the heatmap on (for side-by-side plots), default to None if not needed
    """
    data = tensor.to(torch.float32).cpu().numpy()
    sns.heatmap(data, cmap='viridis', cbar=True, xticklabels=False, yticklabels=True, ax=axis)

    ax = axis if axis is not None else plt.gca()
    ax.set_xlabel('Features')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    
def draw_attention(attention_weights: torch.Tensor,  title: str, axis=None, cmap='viridis', tokens_q=False, tokens_k=False, center=None):
    """
    Draws a heatmap of the self-attention weights with token descriptions.
    args:
        attention_weights: The (seqlen_q, seqlen_k) matrix of self-attention weights.
        tokens_q: The list of token descriptions for query. Default to False if not needed.
        tokens_k: The list of token descriptions for key. Default to False if not needed.
        title: The title of the heatmap.
        axis: The axis to draw the heatmap on (for side-by-side plots), default to None if not needed.
        cmap: The colormap to use for the heatmap, default to 'viridis'.
        center: The center value of the colormap, default to None.
    """
    data = attention_weights.transpose(0, 1).cpu().numpy()
    sns.heatmap(data, cmap=cmap, cbar=True, xticklabels=tokens_q, yticklabels=tokens_k, ax=axis, center=center)

    ax = axis if axis is not None else plt.gca()
    ax.set_xlabel('Query Tokens')
    ax.set_ylabel('Key Tokens')
    ax.set_title(title)
