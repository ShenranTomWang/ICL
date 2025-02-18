import torch
import torch.nn.functional as F
import inspect
from typing import Optional
from .attention import HymbaAttention
from transformers.cache_utils import Cache
import logging
import warnings
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from .utils import _get_unpad_data, apply_rotary_pos_emb, repeat_kv

logger = logging.getLogger(__name__)
_flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

class HymbaFlashAttention2(HymbaAttention):
    """
    Hymba flash attention module. This module inherits from `HymbaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

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

            # overwrite attention_mask with padding_mask
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

                past_key_value.key_cache[kv_layer_idx] = past_key
                past_key_value.value_cache[kv_layer_idx] = past_value
                                                        
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
                
                swa_processed_flag = True
                            
            key_states, value_states = past_key_value.update(key_states, value_states, kv_layer_idx)
            
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

        # Reashape to the expected shape for Flash Attention
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

        if self.attn_only_wo_proj:
            return attn_output, (key_states_no_repeat, value_states_no_repeat)
        
        attn_output = self.o_proj(attn_output)
            
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, (key_states_no_repeat, value_states_no_repeat)

    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
            use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if value_states.shape[-1] == query_states.shape[-1] * 2:
                value_states1 = value_states[...,:query_states.shape[-1]]

                batch_size = query_states.shape[0]
                
                query_states1, key_states1, value_states1, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states1, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad1 = flash_attn_varlen_func(
                        query_states1,
                        key_states1,
                        value_states1,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad1 = flash_attn_varlen_func(
                        query_states1,
                        key_states1,
                        value_states1,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

                attn_output1 = pad_input(attn_output_unpad1, indices_q, batch_size, query_length)

                value_states2 = value_states[...,query_states.shape[-1]:]

                query_states2, key_states2, value_states2, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states2, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad2 = flash_attn_varlen_func(
                        query_states2,
                        key_states2,
                        value_states2,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad2 = flash_attn_varlen_func(
                        query_states2,
                        key_states2,
                        value_states2,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

                attn_output2 = pad_input(attn_output_unpad2, indices_q, batch_size, query_length)

                attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

            else:
                batch_size = query_states.shape[0]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                if not use_sliding_windows:
                    attn_output_unpad = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output_unpad = flash_attn_varlen_func(
                        query_states,
                        key_states,
                        value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

                attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if value_states.shape[-1] == query_states.shape[-1] * 2:
                if not use_sliding_windows:
                    attn_output1 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[...,:query_states.shape[-1]],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )

                    attn_output2 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[...,query_states.shape[-1]:],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )

                    attn_output = torch.cat([attn_output1, attn_output2], dim=-1)
                    
                else:
                    attn_output1 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[...,:query_states.shape[-1]],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

                    attn_output2 = flash_attn_func(
                        query_states,
                        key_states,
                        value_states[...,query_states.shape[-1]:],
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

                    attn_output = torch.cat([attn_output1, attn_output2], dim=-1)

            else:
                if not use_sliding_windows:
                    attn_output = flash_attn_func(
                        query_states,
                        key_states,
                        value_states,
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )
                else:
                    attn_output = flash_attn_func(
                        query_states,
                        key_states,
                        value_states,
                        dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=(self.config.sliding_window, self.config.sliding_window),
                    )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        
        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
