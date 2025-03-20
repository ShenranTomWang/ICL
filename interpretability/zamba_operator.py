import shutup; shutup.please()
from transformers import AutoTokenizer
from interpretability.models.zamba2 import ZambaHybridDynamicCache, Zamba2ForCausalLM
import torch
from typing import Callable
from .operator import Operator
import logging, os

class ZambaOperator(Operator):
    
    def __init__(self, path: str, device: torch.DeviceObjType, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Zamba2ForCausalLM.from_pretrained(path).to(device).to(dtype)
        self.ALL_LAYERS = [i for i in range(model.config.num_hidden_layers)]
        self.HYBRID_LAYERS = model.config.hybrid_layer_ids
        super().__init__(tokenizer, model, device, dtype)
    
    def get_cache_instance(self):
        cache = ZambaHybridDynamicCache(self.model.config, 1, dtype=self.dtype, device=self.device)
        return cache
    
    def get_attention_mean(self, attn: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        all_attn, attn_output, scan_output = attn
        attn_output = attn_output.mean(dim=1).unsqueeze(1)   # (n_layers, n_heads, attn_channels)
        scan_output = scan_output.mean(dim=1).unsqueeze(1)   # (n_layers, n_heads, attn_channels)
        return all_attn, attn_output, scan_output
    
    @torch.inference_mode()
    def extract_attention_outputs(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> tuple[torch.Tensor]:
        """
        Extract attentions from the model
        Args:
            inputs (string): list of inputs
            activation_callback (Callable, optional): callback function to process attentions. Defaults to lambda x: x.

        Returns:
            tuple[torch.Tensor]: all_attn, attn_output, scan_output
        """
        all_attns, attn_outputs, scan_outputs = [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True)
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            output = self.model(**tokenized, output_attentions=True)
            all_attn, attn_output, scan_output = output.attentions
            all_attn = [attn_map for attn_map in all_attn if attn_map is not None]
            attn_output = [attn for attn in attn_output if attn is not None]
            all_attn = torch.stack(all_attn, dim=0).squeeze(1)          # (n_layers, n_heads, seqlen, seqlen)
            attn_output = torch.stack(attn_output, dim=0).squeeze(1)    # (n_layers, seqlen, attn_channels)
            scan_output = torch.stack(scan_output, dim=0).squeeze(1)    # (n_layers, seqlen, scan_channels)
            all_attn, attn_output, scan_output = activation_callback((all_attn, attn_output, scan_output))
            all_attns.append(all_attn)
            attn_outputs.append(attn_output)
            scan_outputs.append(scan_output)
        return all_attns, attn_outputs, scan_outputs
    
    def store_attention_outputs(self, attention_outputs: tuple[list[torch.Tensor]], path: str, fname: str = "") -> None:
        """
        Store attention outputs to path
        Args:
            attention_outputs (tuple[list[torch.Tensor]]): attentions (attention map and self-attention)
            path (str): path to store
            fname (str, optional): special filename. Defaults to "".
        """
        logger = logging.getLogger(__name__)
        all_attns, attn_outputs, scan_outputs = attention_outputs
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".pt"):
            path = path[:-3]
        if fname != "":
            fname = f"{fname}_"
        for i, (all_attn, attn_output, scan_output) in enumerate(zip(all_attns, attn_outputs, scan_outputs)):
            torch.save(all_attn, f"{path}_attn_{fname}all_attn_{i}.pt")
            torch.save(attn_output, f"{path}_attn_{fname}attn_output_{i}.pt")
            torch.save(scan_output, f"{path}_attn_{fname}scan_output_{i}.pt")
        logger.info(f"Stored attention outputs to {path}")
        
    def load_attention_outputs(self, dir: str, split: str, index: int, fname: str = "") -> tuple[torch.Tensor]:
        """
        Load attention outputs from specified directory
        Args:
            dir (str): directory
            split (str): one of demo, test and train
            index (int): index
            fname (str, optional): special filename. Defaults to "".
        
        Returns:
            tuple[torch.Tensor]: all_attn, attn_output, scan_output
        """
        if fname != "":
            fname = f"{fname}_"
        all_attn_path = os.path.join(dir, f"{split}_attn_{fname}all_attn_{index}.pt")
        attn_output_path = os.path.join(dir, f"{split}_attn_{fname}attn_output_{index}.pt")
        scan_output_path = os.path.join(dir, f"{split}_attn_{fname}scan_output_{index}.pt")
        all_attn = torch.load(all_attn_path, map_location=self.device).to(self.dtype)
        attn_output = torch.load(attn_output_path, map_location=self.device).to(self.dtype)
        scan_output = torch.load(scan_output_path, map_location=self.device).to(self.dtype)
        all_attn = [all_attn[layer: layer + 1] for layer in range(all_attn.shape[0])]
        attn_output = [attn_output[layer: layer + 1] for layer in range(attn_output.shape[0])]
        scan_output = [scan_output[layer: layer + 1] for layer in range(scan_output.shape[0])]
        return all_attn, attn_output, scan_output
    
    @torch.inference_mode()
    def extract_cache(self, inputs: list, layers: list, activation_callback: Callable = lambda x: x) -> tuple[list[torch.Tensor]]:
        """
        TODO: should return a cache object
        Extract internal representations at specified layers of cache
        Args:
            inputs (list): list of inputs
            layers (list): list of layer indices
            activation_callback_k (function(tuple[torch.Tensor]) -> tuple[torch.Tensor]): callback function for k, v, ssm_state, conv_state, applied to all cache from all layers
        Returns:
            tuple[list[torch.Tensor]]: list of k (n_layers, n_heads, seqlen, k_head_dim), list of v (n_layers, n_heads, seqlen, v_head_dim),
            list of ssm_states (n_layers, ssm_intermediate_size, ssm_state_size), list of conv_states (n_layers, conv_intermediate_size, conv_state_size)
        """
        ks, vs, ssm_states, conv_states = [], [], [], []
        for input in inputs:
            tokenized = self.tokenizer(input, return_tensors="pt", truncation=True).to(self.device)
            cache = self.model(**tokenized, use_cache=True).past_key_values
            k = cache.key_cache
            v = cache.value_cache
            ssm_state = cache.ssm_states
            conv_state = cache.conv_states
            k = [k[layer] for layer in layers if len(k[layer].shape) == 4]
            v = [v[layer] for layer in layers if len(v[layer].shape) == 4]
            ssm_state = [ssm_state[layer] for layer in layers]
            conv_state = [conv_state[layer] for layer in layers]
            k = torch.stack(k, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, k_head_dim)
            v = torch.stack(v, dim=0).squeeze(1)                    # (n_layers, n_heads, seqlen, v_head_dim)
            ssm_state = torch.stack(ssm_state, dim=0).squeeze(1)    # (n_layers, ssm_intermediate_size, ssm_state_size)
            conv_state = torch.stack(conv_state, dim=0).squeeze(1)  # (n_layers, conv_intermediate_size, conv_state_size)
            cache = (k, v, ssm_state, conv_state)
            ssm_state = activation_callback(cache)
            ks.append(k)
            vs.append(v)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)
        
        return ks, vs, ssm_states, conv_states
    
    def store_cache(self, cache: tuple[list[torch.Tensor]], path: str, fname: str = "") -> None:
        """
        TODO: modify this method to take a cache object and save
        Store cache to path
        Args:
            cache (tuple[list[torch.Tensor]])
            path (str)
            fname (str, optional): special filename. Defaults to "".
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
            fname (str, optional): special filename. Defaults to "".

        Returns:
            tuple[torch.Tensor]: ssm_state
        """
        k_path = os.path.join(dir, f"{split}_cache_{fname}_k_{index}.pt")
        v_path = os.path.join(dir, f"{split}_cache_{fname}_v_{index}.pt")
        ssm_state_path = os.path.join(dir, f"{split}_cache_{fname}_ssm_state_{index}.pt")
        conv_state_path = os.path.join(dir, f"{split}_cache_{fname}_conv_state_{index}.pt")
        k = torch.load(k_path)
        v = torch.load(v_path)
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
        TODO: should take a cache object and return kwargs
        Convert cache to kwargs
        Args:
            cache (tuple[torch.Tensor])
            layers (list[int], optional): list of layers to use cache, if None, use all layers. Defaults to None.
            keep_ssm (bool, optional): whether to keep ssm cache. Defaults to True.
            keep_conv (bool, optional): whether to keep conv cache. Defaults to True.
            **kwargs: additional kwargs, not used
        Returns:
            dict: kwargs
        """
        if kv_layers is None:
            kv_layers = self.HYBRID_LAYERS
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
        