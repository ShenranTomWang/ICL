"""
A collection of hooks used for intervention. Each hook is a function of (stream, intervention, layer, curr_stream, **kwargs)
"""

import torch
from interpretability.attention_managers.attention_manager import AttentionManager

def fv_replace_head_generic(
    stream: torch.Tensor,
    intervention: torch.Tensor | None,
    head: int = None,
    **kwargs
) -> torch.Tensor:
    """
    Hook for attributing AIE of heads on generic (and potentially multihead) models. This will add intervention to the last token of stream
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, seqlen, stream_dims...)
        intervention (torch.Tensor | None): intervention tensor (extracted mean of last token), shape (batch_size, stream_dims...) or None if user does not perform intervention on stream
        head (int, optional): head index to add to. Defaults to None for single head models.
        **kwargs: additional kwargs, not used

    Returns:
        torch.Tensor: intervened results
    """
    if intervention is None:
        return stream
    stream[:, -1, head, :] = intervention[:, head, :]
    return stream

def fv_replace_head_mamba(
    stream: torch.Tensor,
    intervention: torch.Tensor | None,
    **kwargs
) -> torch.Tensor:
    """
    Hook for attributing AIE of heads on Mamba. This will add intervention to the last token of stream
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, n_channels, seqlen)
        intervention (torch.Tensor | None): intervention tensor (extracted mean of last token), shape (batch_size, n_channels) or None if user does not perform intervention on stream
        **kwargs: additional kwargs, not used

    Returns:
        torch.Tensor: intervened results
    """
    if intervention is None:
        return stream
    stream[:, :, -1] = intervention
    return stream

def fv_remove_head_generic(
    stream: torch.Tensor,
    intervention: torch.Tensor | None,
    heads: map,
    layer: int,
    curr_stream: str,
    ablation_type: str = "zero_ablation",
    ablation_value: AttentionManager | None = None,
    ablate_token: int = None,
    **kwargs
) -> torch.Tensor:
    """
    Hook for removing head during forward pass for generic (and potentially multihead) models. This will set the output of the specific head
    to zero or a specific value depending on ablation_type
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, seqlen, stream_dims...)
        intervention (torch.Tensor | None): intervention tensor, not used
        heads (map): map of heads to remove, obtained from operator.top_p_heads
        layer (int): current layer index
        curr_stream (str): current stream, one of "attn" or "scan"
        ablation_type (str, optional): type of ablation to perform. Defaults to "zero_ablation", which means set to zero. If "mean_ablation", set to a ablation_value
        ablation_value (AttentionManager | None): value to set the head to if ablation_type is "mean_ablation". Defaults to None, which means no value is set
        ablate_token (int, optional): token index to ablate. If None, ablate all tokens. Defaults to None
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor: intervened results
    """
    if layer in heads:
        heads_to_ablate = heads[layer]
        for item in heads_to_ablate:
            stream_to_ablate = item["stream"]
            head_to_ablate = item["head"]
            if stream_to_ablate == curr_stream:
                if ablation_type == "zero_ablation":
                    if ablate_token is not None:
                        stream[:, ablate_token, head_to_ablate, :] = 0
                    else:
                        stream[:, :, head_to_ablate, :] = 0
                elif ablation_type == "mean_ablation":
                    ablation_value_stream = ablation_value.get_stream(curr_stream)
                    ablation_value_stream = ablation_value_stream[layer]
                    if ablate_token is not None:
                        ablation_value_stream[:, ablate_token, head_to_ablate, :] = ablation_value_stream[:, head_to_ablate, :]
                    else:
                        ablation_value_stream = ablation_value_stream.unsqueeze(1).expand(stream.shape)
                        stream[:, :, head_to_ablate, :] = ablation_value_stream[:, :, head_to_ablate, :]
                else:
                    raise ValueError(f"Invalid ablation_type: {ablation_type}. Must be one of zero_ablation or mean_ablation.")
    return stream

def fv_remove_head_mamba(
    stream: torch.Tensor,
    intervention: torch.Tensor | None,
    heads: map,
    layer: int,
    curr_stream: str,
    ablation_type: str = "zero_ablation",
    ablation_value: AttentionManager | None = None,
    ablate_token: int = None,
    **kwargs
) -> torch.Tensor:
    """
    Hook for removing head during forward pass for mamba models. This will set the output of the specific head
    to zero or a specific value depending on ablation_type. Note that Mamba models are not multihead, so random ablation is not supported.
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, n_channels, seqlen)
        intervention (torch.Tensor | None): intervention tensor, not used
        heads (map): map of heads to remove, obtained from operator.top_p_heads
        layer (int): current layer index
        curr_stream (str): current stream, one of "attn" or "scan"
        ablation_type (str, optional): type of ablation to perform. Defaults to "zero_ablation", which means set to zero. "mean_ablation" will set head to ablation_value
        ablation_value (AttentionManager | None): value to set the head to if ablation_type is "mean_ablation". Defaults to None, which means no value is set
        ablate_token (int, optional): token index to ablate. If None, ablate all tokens. Defaults to None
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor: intervened results
    """
    if layer in heads:
        for head in heads[layer]:
            stream_to_ablate = head["stream"]
            if stream_to_ablate == curr_stream:
                if ablation_type == "zero_ablation":
                    if ablate_token is not None:
                        stream[..., ablate_token] = 0
                    else:
                        stream[:, :, :] = 0
                elif ablation_type == "mean_ablation":
                    ablation_value_stream = ablation_value.get_stream(curr_stream)
                    ablation_value_stream = ablation_value_stream[layer]
                    if ablate_token is not None:
                        ablation_value_stream[..., ablate_token] = ablation_value_stream[..., ablate_token]
                    else:
                        ablation_value_stream = ablation_value_stream.unsqueeze(-1).expand(stream.shape)
                        stream[:, :, :] = ablation_value_stream
                else:
                    raise ValueError(f"Invalid ablation_type: {ablation_type} for Mamba.")
    return stream

def add_mean_scan_mamba(
    stream: torch.Tensor,
    intervention_mean: torch.Tensor | None,
    first_k: int = -1,
    last_k: int = -1,
    **kwargs
) -> torch.Tensor:
    """
    Add intervention_mean to stream for Mamba. Only one of first_k and last_k can be set to a positive value.
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, stream_dims..., seqlen)
        intervention_mean (torch.Tensor | None): intervention tensor, mean of previous sequence of shape (batch_size, stream_dims...) or None if user does not perform intervention on stream
        first_k (int, optional): only add intervention_mean to first k elements of stream. Defaults to -1, which means all elements.
        last_k (int, optional): only add intervention_mean to last k elements of stream. Defaults to -1, which means all elements.
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor
    """
    assert not (first_k > 0 and last_k > 0), "Only one of first_k and last_k can be set to a positive value."
    if intervention_mean is None:
        return stream
    _, _, seqlen = stream.shape
    intervention_mean = intervention_mean.unsqueeze(-1)
    if first_k > 0:
        intervention_mean = intervention_mean.repeat(1, 1, seqlen)
        intervention_mean[:, :, first_k:] = 0
    elif last_k > 0:
        intervention_mean = intervention_mean.repeat(1, 1, seqlen)
        intervention_mean[:, :, :-last_k] = 0
    else:
        intervention_mean = intervention_mean.expand(-1, -1, seqlen)
    stream += intervention_mean
    return stream

def add_mean_hybrid(
    stream: torch.Tensor,
    intervention_mean: torch.Tensor | None,
    first_k: int = -1,
    last_k: int = -1,
    heads: list[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Add intervention_mean to stream. Only one of first_k and last_k can be set to a positive value.
    Args:
        stream (torch.Tensor | None): stream tensor of shape (batch_size, seqlen, stream_dims...)
        intervention_mean (torch.Tensor): intervention tensor, mean of some previous sequence of shape (batch_size, stream_dims...) or None if user does not perform intervention on stream
        first_k (int, optional): only add intervention_mean to first k elements of stream. Defaults to -1, which means all elements.
        last_k (int, optional): only add intervention_mean to last k elements of stream. Defaults to -1, which means all elements.
        heads (list[int], optional): list of heads to add intervention_mean to. Defaults to None, which means all heads.
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor: intervention tensor with mean of original tensor added
    """
    assert not (first_k > 0 and last_k > 0), "Only one of first_k and last_k can be set to a positive value."
    if intervention_mean is None:
        return stream
    seqlen = stream.shape[1]
    intervention_mean = intervention_mean.unsqueeze(1)
    intervention_mean = intervention_mean.repeat(1, seqlen, 1, 1)
    if first_k > 0:
        intervention_mean[:, first_k:, ...] = 0
    elif last_k > 0:
        intervention_mean[:, :-last_k, ...] = 0
    if heads is not None and intervention_mean.dim() > 3:
        for head in range(intervention_mean.shape[-2]):
            if head not in heads:
                intervention_mean[..., head, :] = 0
    stream += intervention_mean
    return stream