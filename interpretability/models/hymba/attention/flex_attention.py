import torch
from .flash_attention import HymbaFlashAttention2
import warnings
from typing import Optional
from transformers.cache_utils import Cache
from .utils import apply_rotary_pos_emb, repeat_kv
import logging, inspect
from flash_attn import flash_attn_func

logger = logging.getLogger(__name__)
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

class HymbaFlexAttention(HymbaFlashAttention2):
    """
    Hymba flash attention module. This module inherits from `HymbaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.config.num_memory_tokens > 0
        # assert self.config.sliding_window is not None

        from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks, or_masks
        from functools import partial

        self.create_block_mask = create_block_mask

        def sliding_window(b, h, q_idx, kv_idx):
            return q_idx - kv_idx <= self.config.sliding_window

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
                
        if self.config.sliding_window is not None and self.config.global_attn_idx is not None and self.layer_idx not in self.config.global_attn_idx:
            attn_mask = and_masks(causal_mask, sliding_window)
        else:
            attn_mask = causal_mask
        
        if self.config.memory_tokens_interspersed_every > 0:
            # !If see errors, note that deprecated n_ctx, using seq_length or max_position_embeddings instead
            num_memory_band = self.config.seq_length // self.config.memory_tokens_interspersed_every
            qk_length = self.config.seq_length + num_memory_band * self.config.num_memory_tokens
            num_tokens_per_band = qk_length // num_memory_band
            
            for i in range(num_memory_band):
                left_mask = lambda b, h, q_idx, kv_idx, i=i: kv_idx > i * num_tokens_per_band
                right_mask = lambda b, h, q_idx, kv_idx, i=i: kv_idx < i * num_tokens_per_band + self.config.num_memory_tokens

                band_mask = and_masks(left_mask, right_mask)

                if i == 0:
                    prefix_mask_interspersed = band_mask
                else:
                    prefix_mask_interspersed = or_masks(prefix_mask_interspersed, band_mask)

            register_mask = and_masks(causal_mask, prefix_mask_interspersed)
        else:
            def prefix_mask(b, h, q_idx, kv_idx):
                return kv_idx < self.config.num_memory_tokens
        
            register_mask = and_masks(causal_mask, prefix_mask)
            qk_length = self.config.seq_length + self.config.num_memory_tokens

        self.attn_mask = or_masks(attn_mask, register_mask)

        self.block_mask = create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=qk_length, KV_LEN=qk_length)

        self.flex_attention = torch.compile(flex_attention)

    
    def recompile_flexattn(self):
        from torch.nn.attention.flex_attention import flex_attention
        self.flex_attention = torch.compile(flex_attention)


    def forward(
            self,
            hidden_states: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_last_layer=None,
            # kv_proj_last_layer = None,
            use_swa=False,
            query_states = None,
            key_states=None,
            value_states=None,
            **kwargs,
    ):  
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            attention_mask = kwargs.pop("padding_mask")

        if self.attn_only_wo_proj:
            assert query_states is not None
            bsz, q_len, _ = query_states.size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if not self.attn_only_wo_proj:
            query_states = self.q_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            
        if self.config.rope:
            if self.attn_only_wo_proj:
                cos, sin = self.rotary_emb(query_states, position_ids)
            else:
                cos, sin = self.rotary_emb(hidden_states, position_ids)
            query_states, _ = apply_rotary_pos_emb(query_states, None, cos, sin)
        
        if self.reuse_kv:
            assert kv_last_layer is not None
            key_states, value_states = kv_last_layer  # (batch, num_heads, slen, head_dim)

        else:
            if not self.attn_only_wo_proj:
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.kq_head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)

            if self.k_norm is not None:
                key_states = self.k_norm(key_states)
            
            if self.config.rope:
                # cos, sin = self.rotary_emb(hidden_states, position_ids)
                _, key_states = apply_rotary_pos_emb(None, key_states, cos, sin)
        
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and not self.reuse_kv:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        use_sliding_windows = (
                _flash_supports_window_size
                and getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > (self.config.sliding_window + self.config.num_memory_tokens if self.config.num_memory_tokens > 0 else self.config.sliding_window)
                and use_swa
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )
                
        swa_processed_flag = False
        if past_key_value is not None and use_cache and not self.reuse_kv:
            kv_layer_idx = self.layer_idx

            cache_has_contents = past_key_value.get_seq_length(kv_layer_idx) > 0
            
            if (
                    getattr(self.config, "sliding_window", None) is not None
                    and kv_seq_len > (self.config.sliding_window + self.config.num_memory_tokens if self.config.num_memory_tokens > 0 else self.config.sliding_window)
                    and cache_has_contents
                    and use_swa
            ):              
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[kv_layer_idx][0]
                past_value = past_key_value[kv_layer_idx][1]
                
                if self.config.num_memory_tokens > 0:
                    # num_fetched_memory_tokens = min(kv_seq_len - self.config.sliding_window, self.config.num_memory_tokens)
                    num_fetched_memory_tokens = self.config.num_memory_tokens

                    past_key = torch.cat([past_key[:, :, :num_fetched_memory_tokens, :], past_key[:, :, slicing_tokens:, :]], dim=-2).contiguous()
                    past_value = torch.cat([past_value[:, :, :num_fetched_memory_tokens, :], past_value[:, :, slicing_tokens:, :]], dim=-2).contiguous()
                    
                else:
                    past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                    past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                ### only keep sliding_window tokens in kv cache: Removed as this will impact the kv_seq_len calculation, resulting in errors for all swa cases
                past_key_value.key_cache[kv_layer_idx] = past_key
                past_key_value.value_cache[kv_layer_idx] = past_value
                                                        
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
                
                swa_processed_flag = True
                            
            key_states, value_states = past_key_value.update(key_states, value_states, kv_layer_idx)
            
            # print(key_states.shape, value_states.shape)
        else:
            cache_has_contents = False


        # repeat k/v heads if n_kv_heads < n_heads
        key_states_no_repeat = key_states
        value_states_no_repeat = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        
        if past_key_value is not None and use_cache and (not use_swa or query_states.shape[-2] <= self.config.sliding_window):            
            query_states = query_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
            key_states = key_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
            value_states = value_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
            
            attn_output = self._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                use_sliding_windows=use_sliding_windows and not swa_processed_flag,
            )

            v_dim = value_states.shape[-2] * value_states.shape[-1]
            attn_output = attn_output.reshape(bsz, q_len, v_dim).contiguous()
        
        else:
            if key_states.shape[-2] <= self.block_mask.shape[-2] - 128 or key_states.shape[-2] > self.block_mask.shape[-2]:
                block_mask = self.create_block_mask(self.attn_mask, B=None, H=None, Q_LEN=key_states.shape[-2], KV_LEN=key_states.shape[-2]) # , _compile=True)
            else:
                block_mask = self.block_mask
                
            if value_states.shape[-1] == query_states.shape[-1] * 2:
                attn_output1 = self.flex_attention(query_states, key_states, value_states[...,:query_states.shape[-1]], block_mask=block_mask)
                attn_output2 = self.flex_attention(query_states, key_states, value_states[...,query_states.shape[-1]:], block_mask=block_mask)

                attn_output = torch.cat([attn_output1, attn_output2], dim=-1)
            else:
                attn_output = self.flex_attention(query_states, key_states, value_states, block_mask=block_mask)

            attn_output = attn_output.transpose(1, 2).contiguous() ## [batch_size, seq_length, num_head, v_head_dim]
            
            if hasattr(self, 'head_mask') and self.head_mask is not None:
                head_mask = self.head_mask.to(attn_output)
                head_mask = head_mask.view(1, 1, -1, 1)
                attn_output = attn_output * head_mask
            
            attn_output = attn_output.reshape(bsz, q_len, self.v_head_dim * self.num_heads)
  
        if self.attn_only_wo_proj:
            return attn_output, (key_states_no_repeat, value_states_no_repeat)
        
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, (key_states_no_repeat, value_states_no_repeat)

    def set_head_mask(self, mask):
        self.head_mask = mask

