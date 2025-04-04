import torch

def add_mean_hybrid(stream: torch.Tensor, intervention_mean: torch.Tensor | None, first_k: int = -1, **kwargs) -> torch.Tensor:
    """
    Add intervention_mean to stream
    Args:
        stream (torch.Tensor | None): stream tensor of shape (batch_size, seqlen, stream_dims)
        intervention_mean (torch.Tensor): intervention tensor, mean of some previous sequence of shape (batch_size, 1, stream_dims) or None if user does not perform intervention on stream
        first_k (int, optional): only add intervention_mean to first k elements of stream. Defaults to -1, which means all elements.
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor: intervention tensor with mean of original tensor added
    """
    if intervention_mean is None:
        return stream
    intervention_mean = intervention_mean.expand(stream.shape)
    if first_k > 0:
        intervention_mean[:, first_k:, :] = 0
    stream += intervention_mean
    return stream

def add_mean_scan(stream: torch.Tensor, intervention_mean: torch.Tensor | None, first_k: int = -1, **kwargs) -> torch.Tensor:
    """
    Add intervention_mean to stream
    Args:
        stream (torch.Tensor): stream tensor of shape (batch_size, stream_dims, seqlen)
        intervention_mean (torch.Tensor | None): intervention tensor, mean of previous sequence of shape (batch_size, stream_dims, 1) or None if user does not perform intervention on stream
        first_k (int, optional): only add intervention_mean to first k elements of stream. Defaults to -1, which means all elements.
        **kwargs: additional kwargs, not used
    Returns:
        torch.Tensor
    """
    if intervention_mean is None:
        return stream
    intervention_mean = intervention_mean.expand(stream.shape)
    if first_k > 0:
        intervention_mean[:, first_k:, :] = 0
    stream += intervention_mean
    return stream