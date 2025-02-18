from .config import HymbaConfig
import torch
import warnings
import torch.nn as nn
from typing import Optional, Tuple
from .hymba_block import HymbaBlock
from .rms_norm import HymbaRMSNorm
from .sparse_moe_block import HymbaSparseMoeBlock
from .attention.cache import HybridMambaAttentionDynamicCache

class HymbaDecoderLayer(nn.Module):
    def __init__(self, config: HymbaConfig, num_experts: int, layer_idx: int, reuse_kv: bool = False):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.reuse_kv = reuse_kv
        
        self.mamba = HymbaBlock(config=config, layer_idx=layer_idx, reuse_kv=reuse_kv)
        
        self.input_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.intermediate_size = config.intermediate_size
        if self.intermediate_size > 0:
            num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1

            self.moe = HymbaSparseMoeBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)

            self.pre_moe_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attention_mask_raw: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            kv_last_layer = None,
            use_swa=False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, attn_key_value, present_key_value = self.mamba(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_last_layer=kv_last_layer,
            use_cache=use_cache,
            use_swa=use_swa
        )

        bs, seqlen, _ = hidden_states.shape
        past_seqlen = self._get_past_seqlen(past_key_value, seqlen)
        num_attention_heads = self.mamba.config.num_attention_heads
        self_attn_weights = torch.empty(bs, num_attention_heads, seqlen, past_seqlen, device="meta")

        # residual connection after mamba
        hidden_states = residual + hidden_states

        if self.intermediate_size > 0:
            residual = hidden_states
            hidden_states = self.pre_moe_layernorm(hidden_states)
            hidden_states, router_logits = self.moe(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
        
        outputs += (attn_key_value,)

        return outputs

    def _get_past_seqlen(self, past_key_value, seqlen):
        if past_key_value is None:
            return seqlen
        past_seqlen = past_key_value.get_seq_length()

        if past_seqlen == 0:
            return seqlen

        return past_seqlen