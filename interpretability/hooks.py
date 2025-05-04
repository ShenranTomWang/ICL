"""
A collection of hooks used for intervention. Each hook is a function of (stream, intervention, layer, **kwargs)
"""

import torch

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
    intervention_mean = intervention_mean.repeat(1, seqlen, *[1] * (intervention_mean.dim() - 2))
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
    if head is not None and stream.dim() > 3:
        stream[:, -1, head, :] = intervention[:, head, :]
    else:
        stream[:, -1, :] = intervention
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