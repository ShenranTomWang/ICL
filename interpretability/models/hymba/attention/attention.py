import torch
from torch import nn
from ..config import HymbaConfig
from ..rms_norm import HymbaRMSNorm
from typing import Optional, Tuple
from .rms_norm import PerheadHymbaRMSNorm
from transformers.cache_utils import Cache
from .rotary_embedding import LlamaRotaryEmbedding
from abc import ABC, abstractmethod

class HymbaAttention(nn.Module, ABC):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: HymbaConfig, layer_idx: Optional[int] = None, reuse_kv=False, output_hidden_size=None, attn_only_wo_proj=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # self.hidden_size = config.hidden_size
        self.hidden_size = config.attn_hidden_size if config.attn_hidden_size > 0 else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.attn_only_wo_proj = attn_only_wo_proj

        self.kq_head_dim = config.kq_head_dim if config.kq_head_dim > 0 else self.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim > 0 else self.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        if not self.attn_only_wo_proj:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.kq_head_dim, bias=False)

        self.reuse_kv = reuse_kv

        if not self.attn_only_wo_proj and not self.reuse_kv:
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kq_head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False)

        if output_hidden_size is None:
            output_hidden_size = self.hidden_size

        if not self.attn_only_wo_proj:
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, output_hidden_size, bias=False)

        if self.config.kq_norm == "rms":
            self.k_norm = HymbaRMSNorm(self.kq_head_dim)
            self.q_norm = HymbaRMSNorm(self.kq_head_dim)
        elif self.config.kq_norm == "perhead-rms":
            self.k_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_key_value_heads)
            self.q_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_heads)
        elif self.config.kq_norm == "none":
            self.k_norm = None
            self.q_norm = None
        else:
            raise NotImplementedError(f"Unknown kq_norm: {self.config.kq_norm}")

        if self.config.rope:
            self._init_rope()


    def set_rope(self, rope_type, orig_max_position_embeddings, max_position_embeddings):
        self.config.rope_type = rope_type
        self.config.orig_max_position_embeddings = orig_max_position_embeddings
        self.config.max_position_embeddings = max_position_embeddings
        
        self._init_rope()
            
            
    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            dim=self.kq_head_dim,
            base=self.rope_theta,
            device=torch.device("cuda"),
            ) 

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @abstractmethod
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_last_layer = None,
            # kv_proj_last_layer = None,
            use_swa=False,
            query_states = None,
            key_states=None,
            value_states=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError("HymbaAttention is an abstract class. Use one of the subclasses.")