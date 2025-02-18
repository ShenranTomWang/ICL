from transformers.modeling_utils import PreTrainedModel
from typing import Optional, Tuple, Union, List
from .config import HymbaConfig
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
import math
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from .decoder import HymbaDecoderLayer
from .rms_norm import HymbaRMSNorm
from .attention.cache import HybridMambaAttentionDynamicCache
import logging
from .utils import shift_zeros_to_front, pad_at_dim, load_balancing_loss_func
from einops import rearrange, repeat, pack, unpack
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)

HYMBA_INPUTS_DOCSTRING = r"""
    Args: To be added later. Please refer to the forward function.
"""
_CONFIG_FOR_DOC = "HymbaConfig"

class HymbaPreTrainedModel(PreTrainedModel):
    config_class = HymbaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HymbaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def _convert_to_standard_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. have the seqlen as the third dim
        also for mamba layers
        """
        attn_layer_index = [k.shape == v.shape for k, v in past_key_value].index(True)
        seqlen = past_key_value[attn_layer_index][0].shape[2]
        standard_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                # expand doesn't use more memory, so it's fine to do it here
                standard_past_key_value += ((k.expand(-1, -1, seqlen, -1), v.expand(-1, -1, seqlen, -1)),)
            else:
                standard_past_key_value += ((k, v),)
        return standard_past_key_value

    @staticmethod
    def _convert_to_hymba_cache(
            past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Hymba, i.e. dummy seqlen dimesion with size 1 for mamba layers
        """
        hymba_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                hymba_past_key_value += ((k[:, :, :1, :], v[:, :, :1, :]),)
            else:
                hymba_past_key_value += ((k, v),)
        return hymba_past_key_value

class HymbaModel(HymbaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`HymbaDecoderLayer`]

    Args:
        config: HymbaConfig
    """

    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        config.attn_implementation = config.attn_implementation_new
        config._attn_implementation = config.attn_implementation_new

        self.config = config
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.inter_layer_kv_reuse = config.kv_reuse_every_i_layer > 0 or config.kv_reuse_group is not None
        self.kv_reuse_group = config.kv_reuse_group
        self.kv_reuse_every_i_layer = config.kv_reuse_every_i_layer

        decoder_layers = []
        
        if self.kv_reuse_group is not None:
            self.kv_reuse_group = [{'producer': group[0], 'consumer': group[1:]} for group in self.kv_reuse_group]

        layer_type = []
        for i in range(config.num_hidden_layers):
            if self.inter_layer_kv_reuse:
                if self.kv_reuse_group is not None:
                    reuse_kv = False
                    for group_id, item in enumerate(self.kv_reuse_group):
                        if i in item['consumer']:
                            reuse_kv = True

                else:
                    if i % config.kv_reuse_every_i_layer == 0:
                        reuse_kv = False
                    else:
                        reuse_kv = True
            else:
                reuse_kv = False
            
            layer_type.append('h')
            decoder_layer = HymbaDecoderLayer(config, num_experts=1, layer_idx=i, reuse_kv=reuse_kv)

            decoder_layers.append(decoder_layer)
            
        config.layer_type = layer_type
        
        if config.sliding_window is not None:
            self.sliding_window = config.sliding_window
            self.global_attn_idx = config.global_attn_idx
        else:
            self.sliding_window = None
            self.global_attn_idx = None

        self._attn_layer_index = []
        self._hymba_layer_index = [isinstance(layer, HymbaDecoderLayer) for layer in decoder_layers].index(True)

        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config.attn_implementation
        self.final_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.config.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(self.config.num_memory_tokens, self.config.hidden_size))
        self.gradient_checkpointing = False

        self.post_init()

    # Ignore copy
    @add_start_docstrings_to_model_forward(HYMBA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[List[torch.FloatTensor], HybridMambaAttentionDynamicCache]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # TODO: modify this method to return cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = False
            # past_key_values_length = past_key_values.get_usable_length(seq_length, self._attn_layer_index)
            if past_key_values is not None:
                past_key_values_length = past_key_values.get_usable_length(seq_length, 0)
            else:
                use_cache = False

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if self.config.num_memory_tokens > 0 and past_key_values is not None and past_key_values.get_seq_length() == 0:
                position_ids = position_ids.view(-1, seq_length + self.config.num_memory_tokens).long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.config.num_memory_tokens > 0 and (past_key_values is None or past_key_values.get_seq_length() == 0):
            ori_b, ori_n = inputs_embeds.shape[0], inputs_embeds.shape[1]
            
            if self.config.memory_tokens_interspersed_every > 0:
                mem_every = self.config.memory_tokens_interspersed_every
                next_seq_len = math.ceil(ori_n / mem_every) * mem_every

                # print(f"before padding: {inputs_embeds.shape}")
                inputs_embeds = pad_at_dim(inputs_embeds, (0, next_seq_len - ori_n), dim = -2, value = 0.)
                # print(f"after padding: {inputs_embeds.shape}")
                inputs_embeds = rearrange(inputs_embeds, 'b (n m) d -> (b n) m d', m = mem_every) # m is the segment length

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = inputs_embeds.shape[0]) # prepend the memory to every segment of m by repeating the memory tokens
            inputs_embeds, mem_packed_shape = pack((mem, inputs_embeds), 'b * d')      

            if self.config.memory_tokens_interspersed_every > 0:
                inputs_embeds = rearrange(inputs_embeds, '(b n) m d -> b (n m) d', b = ori_b)
            
            if position_ids is not None and position_ids.shape[1] != inputs_embeds.shape[1]:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

            ## Handle paddings: Shift all padding tokens to the beginning of the sequence
            if inputs_embeds.shape[1] > 1 and attention_mask is not None and (attention_mask == 0).any():
                attention_mask, inputs_embeds, position_ids = shift_zeros_to_front(attention_mask, inputs_embeds, position_ids)

        attention_mask_raw = attention_mask

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Hymba. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
                
        if self._attn_implementation == "flash_attention_2" or self._attn_implementation == "flex":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask_swa = attention_mask
            
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask_input = attention_mask

            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

            if self.sliding_window is not None:
                attention_mask_swa = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask_input,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.sliding_window
                )

        else:

            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
            

            if self.sliding_window is not None:
                attention_mask_swa = _prepare_4d_causal_attention_mask(
                    attention_mask_input,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.sliding_window
                )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        kv_last_layer = None

        shared_kv_cache_dict = {}

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.inter_layer_kv_reuse and self.kv_reuse_group is not None:
                no_reuse_flag = True
                for group_id, item in enumerate(self.kv_reuse_group):
                    if i in item['consumer']:
                        kv_last_layer = shared_kv_cache_dict[group_id]
                        no_reuse_flag = False
                        # print(f'[Layer-{i}]: Reuse KV cache from Layer-{self.kv_reuse_group[group_id]["producer"]}')
                        break
                
                if no_reuse_flag:
                    kv_last_layer = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask if (self.sliding_window is None or i in self.global_attn_idx) else attention_mask_swa,
                    attention_mask_raw,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    kv_last_layer,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask if (self.sliding_window is None or i in self.global_attn_idx) else attention_mask_swa,
                    attention_mask_raw=attention_mask_raw,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    kv_last_layer=kv_last_layer if self.inter_layer_kv_reuse else None,
                    use_swa=self.sliding_window is not None and i not in self.global_attn_idx,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[3],)

            if self.inter_layer_kv_reuse:
                kv_last_layer = layer_outputs[-1]

                if self.kv_reuse_group is not None:
                    for group_id, item in enumerate(self.kv_reuse_group):
                        if i == item['producer']:
                            shared_kv_cache_dict[group_id] = kv_last_layer
                            break
        
        del shared_kv_cache_dict

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.config.num_memory_tokens > 0 and (past_key_values is None or past_key_values.get_seq_length() == 0):
            if self.config.memory_tokens_interspersed_every > 0:
                hidden_states = rearrange(hidden_states, 'b (n m) d -> (b n) m d', m = (self.config.num_memory_tokens + self.config.memory_tokens_interspersed_every))

            mem, hidden_states = unpack(hidden_states, mem_packed_shape, 'b * d')

            if self.config.memory_tokens_interspersed_every > 0:
                hidden_states = rearrange(hidden_states, '(b n) m d -> b (n m) d', b = ori_b)

            hidden_states = hidden_states[:, :ori_n, :]

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

class HymbaForCausalLM(HymbaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        self.config = config
        self.model = HymbaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    @add_start_docstrings_to_model_forward(HYMBA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            calc_logits_for_entire_prompt: Optional[bool] = True,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
                
        r"""
        TODO: modify this method to return cache
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            calc_logits_for_entire_prompt (`bool`, *optional*):
                Whether or not to calculate the logits for the entire prompt, or just the last token. Only last token
                logits are needed for generation, and calculating them only for that token can save memory,
                which becomes pretty significant for long sequences.

        Returns:
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -1:, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        # print("hidden_states.shape:", hidden_states.shape, "input_ids.shape:", input_ids.shape, "logits.shape:", logits.shape)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            output_router_logits=False,
            **kwargs,
    ):         
        if self.config.num_memory_tokens > 0:
            attention_mask = torch.cat([torch.ones(input_ids.shape[0], self.config.num_memory_tokens, device=attention_mask.device), attention_mask], dim=1)

        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            if isinstance(past_key_values, Tuple):
                if past_key_values[self.model._hymba_layer_index][0].shape[2] > 1:
                    past_key_values = self._convert_to_hymba_cache(past_key_values)

            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()

                past_length = cache_length

            else:
                cache_length = past_length = past_key_values[self.model._attn_layer_index][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif self.config.num_memory_tokens <= 0 and past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                
            elif self.config.num_memory_tokens > 0 and past_length < input_ids.shape[1] + self.config.num_memory_tokens:
                new_query_id = past_length - self.config.num_memory_tokens
                input_ids = input_ids[:, new_query_id:]

            if self.config.sliding_window is not None and (self.config.global_attn_idx is None or len(self.config.global_attn_idx) == 0):
                input_ids = input_ids[:, -1:]
                    
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                    max_cache_length is not None
                    and attention_mask is not None
                    and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device, layer_type=self.config.layer_type
            )

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values.get_seq_length() > 0:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "calc_logits_for_entire_prompt": self.config.calc_logits_for_entire_prompt,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past