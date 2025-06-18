from interpretability.attention_managers import AttentionManager
import torch

def exclusion_ablation_sanity_check(
    top_p_heads: dict,
    exclusion_ablation_heads: dict,
    stream: str = "attn"
) -> None:
    """
    Sanity check for exclusion ablation heads.
    Args:
        top_p_heads (dict): top p heads from fv_map
        exclusion_ablation_heads (dict): exclusion ablation heads from fv_map
        stream (str): stream to check, either "attn" or "scan"
    Raises:
        AssertionError: if the exclusion ablation heads are not a subset of top p heads or vice versa, or if either set is empty
    """
    exclusion_ablation_heads_set = set()
    for layer, heads in exclusion_ablation_heads.items():
        for head in heads:
            head_stream = head["stream"]
            head_idx = head["head"]
            assert head_stream == stream or stream is None, f"Exclusion ablation head stream {head_stream} does not match expected stream {stream}"
            exclusion_ablation_heads_set.add((layer, head_idx, head_stream))
    top_p_heads_set = set()
    for layer, heads in top_p_heads.items():
        for head in heads:
            head_stream = head["stream"]
            head_idx = head["head"]
            assert stream == head_stream or stream is None, f"Top p head stream {head_stream} does not match expected stream {stream}"
            top_p_heads_set.add((layer, head_idx, head_stream))
    assert top_p_heads_set, "Top p heads set is empty"
    assert not exclusion_ablation_heads_set & top_p_heads_set, f"Exclusion ablation heads should not be disjoint with top p heads, but seeing {exclusion_ablation_heads_set & top_p_heads_set}"
    
def ablation_steer_sanity_check(
    fv_steer: AttentionManager,
    top_p_heads: dict
) -> None:
    """
    Sanity check for ablation steer.
    Args:
        fv_steer (AttentionManager): attention manager for steer
        top_p_heads (dict): top p heads from fv_map
    Raises:
        AssertionError: if the steer is all zero for any head in top p heads
    """
    for layer, heads in top_p_heads.items():
        for head in heads:
            head_stream = head["stream"]
            head_idx = head["head"]
            if head_stream == "attn":
                steer_value = fv_steer.attn_outputs[layer][..., head_idx, :]
            elif head_stream == "scan":
                steer_value = fv_steer.scan_outputs[layer][..., head_idx, :]
            else:
                raise ValueError(f"Invalid stream: {head_stream}. Must be one of attn or scan.")
            assert not torch.all(steer_value == 0), f"Steer is all zero for head {head_idx} in layer {layer} of stream {head_stream}"