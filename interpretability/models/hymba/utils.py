import torch
from typing import Tuple, Optional
from torch.nn import functional as F

def shift_zeros_to_front(attention_mask, hidden_states, position_ids):
    """
    Move all zero entries in 'attention_mask' to the front of the sequence
    and reorder 'hidden_states' accordingly, preserving the order of zeros
    and the order of ones.

    Args:
      attention_mask: (batch_size, seq_len), values in {0, 1}.
      hidden_states:  (batch_size, seq_len, dim).

    Returns:
      shifted_mask:   (batch_size, seq_len) with zeros at the front.
      shifted_states: (batch_size, seq_len, dim) reordered accordingly.
    """
    B, L = attention_mask.shape
    D = hidden_states.shape[-1]

    shifted_mask = torch.empty_like(attention_mask)
    shifted_states = torch.empty_like(hidden_states)
    shifted_position_ids = torch.empty_like(position_ids)

    # Process each batch row independently
    for b in range(B):
        row_mask = attention_mask[b]       # (seq_len,)
        row_states = hidden_states[b]      # (seq_len, dim)
        row_pos = position_ids[b]       # (seq_len,)

        # Find positions of zeros and ones
        zero_indices = torch.where(row_mask == 0)[0]
        one_indices  = torch.where(row_mask == 1)[0]

        # Concatenate zero indices (in order) then one indices
        new_order = torch.cat([zero_indices, one_indices], dim=0)

        # Reorder mask and states
        shifted_mask[b] = row_mask[new_order]
        shifted_states[b] = row_states[new_order]
        shifted_position_ids[b] = row_pos[new_order]

    return shifted_mask, shifted_states, shifted_position_ids

def pad_at_dim(t, pad: Tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)
  
def load_balancing_loss_func(
        gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits if layer_gate.shape[1] > 1], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts