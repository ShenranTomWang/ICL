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

def add_mean_scan(
    stream: torch.Tensor,
    intervention_mean: torch.Tensor | None,
    first_k: int = -1,
    last_k: int = -1,
    **kwargs
) -> torch.Tensor:
    """
    TODO: debug this with Mamba2
    Add intervention_mean to stream. Only one of first_k and last_k can be set to a positive value.
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