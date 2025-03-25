import shutup; shutup.please()
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Callable
from .operator import Operator
from interpretability.attention_outputs import HybridAttentionOutput
from interpretability.hooks import add_mean_hybrid
from abc import ABC
import logging, os

class HybridOperator(Operator, ABC):
    
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.DeviceObjType, dtype: torch.dtype):
        super().__init__(tokenizer, model, device, dtype)
        
    def get_attention_add_mean_hook(self):
        return add_mean_hybrid
    
    @torch.inference_mode()
    def extract_attention_outputs(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[HybridAttentionOutput]:
        """
        Extract attentions from the model
        Args:
            inputs (_type_): list of inputs (string)
            activation_callback (Callable, optional): callback function to process attentions. Defaults to ....

        Returns:
            list[HybridAttentionOutput]: attentions (self-attention and scan)
        """
        outputs = []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            output = self.model(**tokenized, output_attentions=True)
            # n_layers * (batch_size, n_heads, pad_len + seqlen, pad_len + seqlen), (batch_size, pad_len + seqlen, attn_channels), (batch_size, pad_len + seqlen, attn_channels)
            all_attn, attn_output, scan_output = output.attentions
            all_attn, attn_output, scan_output = list(all_attn), list(attn_output), list(scan_output)
            hybrid_output = HybridAttentionOutput(all_attn, attn_output, scan_output).to("cpu")
            hybrid_output = activation_callback(hybrid_output)
            outputs.append(hybrid_output)
        return outputs

    def attention2kwargs(
        self,
        attention: HybridAttentionOutput,
        attention_intervention_fn: Callable = add_mean_hybrid,
        scan_intervention_fn: Callable = add_mean_hybrid,
        layers: list[int] = None,
        keep_scan: bool = True,
        keep_attention: bool = True,
        **kwargs
    ) -> dict:
        """
        Convert attention outputs to kwargs for intervention
        Args:
            attention (HybridAttentionOutput)
            attention_intervention_fn (Callable, optional): intervention function for attention, defaults to add_mean_hybrid
            scan_intervention_fn (Callable, optional): intervention function for scan, defaults to add_mean_hybrid
            layers (list[int], optional): list of layers to use attention, if None, use all layers. Defaults to None.
            keep_scan (bool, optional): whether to keep scan outputs. Defaults to True.
            keep_attention (bool, optional): whether to keep attention outputs. Defaults to True.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        if layers is None:
            layers = set(self.ALL_LAYERS)
        _, attn_outputs, scan_outputs = attention
        params = ()
        for layer in self.ALL_LAYERS:
            attn = attn_outputs[layer] if keep_attention and layer in layers else None
            scan = scan_outputs[layer] if keep_scan and layer in layers else None
            params += ((attention_intervention_fn, attn, scan_intervention_fn, scan),)
        return {"attention_overrides": params}
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        TODO: should return cache object
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for k, v, ssm_state, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of k (n_layers, n_heads, seqlen, k_head_dim), list of v (n_layers, n_heads, seqlen, v_head_dim),
                list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size), list of conv_states (n_layers, conv_intermediate_size, conv_state_size)
        """
        cache = self.get_cache_instance()
        ks, vs, ssm_states, conv_states = [], [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            _ = self.model(**tokenized, use_cache=True, past_key_values=cache)
            k = cache.key_cache
            v = cache.value_cache
            ssm_state = cache.ssm_states
            conv_state = cache.conv_states
            k = [k[layer] for layer in len(k) if len(k[layer].shape) == 4]
            v = [v[layer] for layer in len(v) if len(v[layer].shape) == 4]
            k = torch.stack(k, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, k_head_dim)
            v = torch.stack(v, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, v_head_dim)
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            conv_state = torch.stack(conv_state, dim=0).squeeze(1)  # (n_layers, conv_intermediate_size, conv_state_size)
            cache = (k, v, ssm_state, conv_state)
            k, v, ssm_state, conv_state = activation_callback(cache)
            ks.append(k)
            vs.append(v)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)
        
        return ks, vs, ssm_states, conv_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str, fname: str = "") -> None:
        """
        TODO: should take a cache object then save
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]])
            path (str)
            fname (str, optional): filename. Defaults to "".
        """
        logger = logging.getLogger(__name__)
        ks, vs, ssm_states, conv_states = cache
        if path.endswith(".pt"):
            path = path[:-3]
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        for i, (k, v, ssm_state, conv_state) in enumerate(zip(ks, vs, ssm_states, conv_states)):
            torch.save(k, f"{path}_cache_{fname}_k_{i}.pt")
            torch.save(v, f"{path}_cache_{fname}_v_{i}.pt")
            torch.save(ssm_state, f"{path}_cache_{fname}_ssm_state_{i}.pt")
            torch.save(conv_state, f"{path}_cache_{fname}_conv_state_{i}.pt")
        logger.info(f"Stored cache to {path}")
        
    def load_cache(self, dir: str, split: str, index: int, fname: str = "") -> tuple[torch.Tensor]:
        """
        TODO: should load into a cache object
        Load cache from specified directory
        Args:
            dir (str)
            split (str): one of demo, test and train
            index (int)
            fname (str, optional): filename. Defaults to "".

        Returns:
            tuple[torch.Tensor]: k, v, ssm_state, conv_state
        """
        k_path = os.path.join(dir, f"{split}_cache_{fname}_k_{index}.pt")
        v_path = os.path.join(dir, f"{split}_cache_{fname}_v_{index}.pt")
        ssm_state_path = os.path.join(dir, f"{split}_cache_{fname}_ssm_state_{index}.pt")
        conv_state_path = os.path.join(dir, f"{split}_cache_{fname}_conv_state_{index}.pt")
        k = torch.load(k_path, map_location=self.device).to(self.dtype)
        v = torch.load(v_path, map_location=self.device).to(self.dtype)
        ssm_state = torch.load(ssm_state_path, map_location=self.device).to(self.dtype)
        conv_state = torch.load(conv_state_path, map_location=self.device).to(self.dtype)
        return k, v, ssm_state, conv_state
    
    def cache2kwargs(
        self,
        cache: tuple[torch.Tensor],
        kv_layers: list[int] = None,
        layers: list[int] = None,
        keep_kv: bool = True,
        keep_ssm: bool = True,
        keep_conv: bool = True,
        **kwargs
    ) -> dict:
        """
        TODO: debug this function for proper cache conversion, should take a cache object
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            kv_layers (list[int]): list of layers kv cache maps to
            layers (list[int], optional): list of layers to use cache, if None, use all layers. Defaults to None.
            keep_kv (bool, optional): whether to keep kv cache. Defaults to True.
            keep_ssm (bool, optional): whether to keep ssm cache. Defaults to True.
            keep_conv (bool, optional): whether to keep conv cache. Defaults to True.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        if kv_layers is None:
            kv_layers = self.KV_LAYERS
        if layers is None:
            layers = self.ALL_LAYERS
        k, v, ssm_state, conv_state = cache
        cache_instance = self.get_cache_instance()
        k = [k[i: i + 1, ...] for i in range(k.shape[0])]
        v = [v[i: i + 1, ...] for i in range(v.shape[0])]
        ssm_state = [ssm_state[i: i + 1, ...] for i in range(ssm_state.shape[0])]
        conv_state = [conv_state[i: i + 1, ...] for i in range(conv_state.shape[0])]
        k_list, v_list = [], []
        if keep_kv:
            i = 0
            for layer in range(self.model.config.num_hidden_layers):
                if layer in kv_layers and layer in layers:
                    k_list.append(k[i])
                    v_list.append(v[i])
                    i += 1
                else:
                    k_list.append(torch.zeros((1, 0)))
                    v_list.append(torch.zeros((1, 0)))
            cache_instance.key_cache = k_list
            cache_instance.value_cache = v_list
        if keep_ssm:
            cache_instance.ssm_states = [ssm_state[layer] for layer in layers]
        if keep_conv:
            cache_instance.conv_states = [conv_state[layer] for layer in layers]
        kwargs = {"use_cache": True, "past_key_values": cache_instance}
        return kwargs
        