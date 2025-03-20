import torch

def add_mean(stream: torch.Tensor, intervention_mean: torch.Tensor | None) -> torch.Tensor:
    """
    Add intervention_mean to stream
    Args:
        stream (torch.Tensor | None): stream tensor of shape (batch_size, seqlen, stream_dims) or None if user does not perform intervention on stream
        intervention_mean (torch.Tensor): intervention tensor, mean of some previous sequence of shape (batch_size, 1, stream_dims)
    Returns:
        torch.Tensor: intervention tensor with mean of original tensor added
    """
    if intervention_mean is None:
        return stream
    for i in range(stream.size(1)):
        stream[:, i, :] += intervention_mean[:, 0, :]
    return stream