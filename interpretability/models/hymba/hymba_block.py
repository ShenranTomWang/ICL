import torch
from torch import nn
from .config import HymbaConfig
from .rms_norm import HymbaRMSNorm
from .attention.cache import HybridMambaAttentionDynamicCache
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from typing import Optional, Tuple
from .attention.flex_attention import HymbaFlexAttention
from transformers.activations import ACT2FN

class HymbaBlock(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: HymbaConfig, layer_idx, reuse_kv=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv

        self.intermediate_size = int(config.mamba_expand * config.hidden_size)

        self.reuse_kv = reuse_kv

        self.attn_hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        config.v_head_dim = self.intermediate_size // self.num_attention_heads

        self.k_hidden_size = int(self.num_key_value_heads/self.num_attention_heads * self.attn_hidden_size)
        self.v_hidden_size = int(self.num_key_value_heads/self.num_attention_heads * self.attn_hidden_size * config.mamba_expand)

        self.self_attn = HymbaFlexAttention(config, layer_idx, attn_only_wo_proj=True, reuse_kv=reuse_kv)

        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        self.use_fast_kernels = True # config.use_mamba_kernels

        if self.reuse_kv:
            self.latent_dim = self.intermediate_size + self.attn_hidden_size  ## mamba plus q
        else:
            self.latent_dim = self.intermediate_size + self.attn_hidden_size + self.k_hidden_size + self.v_hidden_size  ## mamba plus qkv

        self.pre_avg_layernorm1 = HymbaRMSNorm(self.intermediate_size, eps=config.rms_norm_eps)
        self.pre_avg_layernorm2 = HymbaRMSNorm(self.intermediate_size, eps=config.rms_norm_eps)

        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim + self.intermediate_size, bias=self.use_bias)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        num_ssm_param = 1

        if not hasattr(config, 'conv_dim'):
            config.conv_dim = {str(i):0 for i in range(config.num_hidden_layers)}

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1
            )

        config.conv_dim[str(self.layer_idx)] = self.intermediate_size

        self.x_proj = nn.ModuleList([nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False) for _ in range(num_ssm_param)])
        self.dt_proj = nn.ModuleList([nn.Linear(self.time_step_rank, self.intermediate_size, bias=True) for _ in range(num_ssm_param)])

        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.ParameterList([nn.Parameter(torch.log(A)) for _ in range(num_ssm_param)])

        self.D = nn.ParameterList([nn.Parameter(torch.ones(self.intermediate_size)) for _ in range(num_ssm_param)])

        if self.apply_inner_layernorms:
            self.dt_layernorm = HymbaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
            self.B_layernorm = HymbaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
            self.C_layernorm = HymbaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def set_attn_mamba_mask(self, attn_branch_mask, mamba_branch_mask):
        self.attn_branch_mask = attn_branch_mask
        self.mamba_branch_mask = mamba_branch_mask
        
        
    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: HybridMambaAttentionDynamicCache = None, attention_mask=None, position_ids=None, kv_last_layer=None, use_cache=False, use_swa=False):
        projected_states = self.in_proj(hidden_states).transpose(1, 2)  ## (bs, latent_dim, seq_len) 

        ## Handle padding for Mamba: Set padding tokens to 0
        if projected_states.shape[-1] > 1 and attention_mask is not None and (attention_mask == 0).any():
            projected_states = projected_states * attention_mask.unsqueeze(1).to(projected_states)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and use_cache
        )

        hidden_states, gate = projected_states.tensor_split((self.latent_dim,), dim=1)

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

        if self.reuse_kv:
            query_states, hidden_states = hidden_states.tensor_split((self.attn_hidden_size,), dim=1)
            query_states = query_states.transpose(1,2)
        else:
            query_states, key_states, value_states, hidden_states = hidden_states.tensor_split((self.attn_hidden_size, self.attn_hidden_size + self.k_hidden_size, self.attn_hidden_size + self.k_hidden_size + self.v_hidden_size), dim=1)

            query_states = query_states.transpose(1,2)
            key_states = key_states.transpose(1,2)
            value_states = value_states.transpose(1,2)

        if use_precomputed_states:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)

            cache_params.mamba_past_length[self.layer_idx] += seq_len
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )

                cache_params.conv_states[self.layer_idx].copy_(conv_states)

                cache_params.mamba_past_length[self.layer_idx] += seq_len
            
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        ## Handle padding for Mamba: Set padding tokens to 0
        if seq_len > 1 and attention_mask is not None and (attention_mask == 0).any():
            hidden_states = hidden_states * attention_mask.unsqueeze(1).to(hidden_states)
            
        if self.reuse_kv:
            assert kv_last_layer is not None
            attn_outputs, attn_key_value = self.self_attn(attention_mask=attention_mask, position_ids=position_ids, query_states=query_states, kv_last_layer=kv_last_layer, use_swa=use_swa, use_cache=use_cache, past_key_value=cache_params)
        else:
            attn_outputs, attn_key_value = self.self_attn(attention_mask=attention_mask, position_ids=position_ids, query_states=query_states, key_states=key_states, value_states=value_states, use_swa=use_swa, use_cache=use_cache, past_key_value=cache_params)

        ## Mamba head
        index = 0
        ssm_parameters = self.x_proj[index](hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)

        if hasattr(self.dt_proj[index], "base_layer"):
            time_proj_bias = self.dt_proj[index].base_layer.bias
            self.dt_proj[index].base_layer.bias = None
        else:
            time_proj_bias = self.dt_proj[index].bias
            self.dt_proj[index].bias = None
        discrete_time_step = self.dt_proj[index](time_step).transpose(1, 2)  # [batch, intermediate_size, seq_len]

        if hasattr(self.dt_proj[index], "base_layer"):
            self.dt_proj[index].base_layer.bias = time_proj_bias
        else:
            self.dt_proj[index].bias = time_proj_bias

        A = -torch.exp(self.A_log[index].float())

        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D[index],
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            outputs = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D[index].float(),
                z=gate,
                delta_bias=time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            
            if len(outputs) == 3:
                scan_outputs, ssm_state, _ = outputs
            else:
                scan_outputs, ssm_state = outputs

            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)
                
        scan_outputs = scan_outputs.transpose(1, 2)

        hidden_states = (self.pre_avg_layernorm1(attn_outputs) + self.pre_avg_layernorm2(scan_outputs)) / 2
        contextualized_states = self.out_proj(hidden_states)

        return contextualized_states, attn_key_value


    def mixer_forward(self, hidden_states, cache_params: HybridMambaAttentionDynamicCache = None, attention_mask=None, position_ids=None, kv_last_layer=None, use_cache=False, use_swa=False):
        if self.use_fast_kernels:
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask=attention_mask, position_ids=position_ids, kv_last_layer=kv_last_layer, use_cache=use_cache, use_swa=use_swa)
        else:
            raise ValueError("Support Mamba kernel only")


    def forward(
            self,
            hidden_states: torch.Tensor,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:

        res, attn_key_value = self.mixer_forward(hidden_states, cache_params=past_key_value, attention_mask=kwargs['attention_mask'], kv_last_layer=kwargs['kv_last_layer'], position_ids=kwargs['position_ids'], use_cache=kwargs['use_cache'], use_swa=kwargs['use_swa'])

        return res, attn_key_value, past_key_value
    