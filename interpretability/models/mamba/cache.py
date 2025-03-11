import torch
from typing import Optional, Union, List
from transformers.configuration_utils import PretrainedConfig

class MambaCache:
    """
    Cache for mamba model which does not have attention mechanism and key value states.

    Arguments:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float16`):
            The default `dtype` to use when initializing the layer.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. Should be the same as the layer.
            The recommended way however is not not indicate any `device`, in that case cache will be initialized on `meta`
            device by default, and then moved to input device when updating.

    Attributes:
        dtype: (`torch.dtype`):
            The default `dtype` used to initializing the cache.
        device (`torch.device`):
            The default device on which the cache was initialized.
        intermediate_size: (`int`):
            Model's intermediate_size taken from config.
        ssm_state_size: (`int`):
            Model's state_size taken from config.
        conv_kernel_size: (`int`):
            Model's convolution kernel size taken from config
        conv_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, conv_kernel_size]` that holds convolutional states.
        ssm_states: (`torch.Tensor`):
            A tensor of shape `[layer_idx, batch_size, intermediate_size, ssm_state_size]` that holds ssm states

    Example:

        ```python
        >>> from transformers import AutoTokenizer, MambaForCausalLM, MambaCache

        >>> model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

        >>> inputs = tokenizer(text="My name is Mamba", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = MambaCache(config=model.config, batch_size=1, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values
        MambaCache()
        ```
    """

    is_compileable = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[torch.device, str]] = None,
        max_batch_size: Optional[int] = None,
        ssm_states: Optional[List[torch.Tensor]] = None,
        conv_states: Optional[List[torch.Tensor]] = None,
    ):
        self.dtype = dtype
        self.max_batch_size = batch_size or max_batch_size
        self.intermediate_size = config.intermediate_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.device = torch.device(device) if device is not None else torch.device("meta")

        self.conv_states: List[torch.Tensor] = []
        self.ssm_states: List[torch.Tensor] = []
        
        if ssm_states is None:
            for _ in range(config.num_hidden_layers):
                ssm_state: torch.Tensor = torch.zeros(
                    self.max_batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    device=self.device,
                    dtype=dtype,
                )
                torch._dynamo.mark_static_address(ssm_state)
                self.ssm_states.append(ssm_state)
        else:
            self.ssm_states = ssm_states
            
        if conv_states is None:
            for _ in range(config.num_hidden_layers):
                conv_state: torch.Tensor = torch.zeros(
                    self.max_batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    device=self.device,
                    dtype=dtype,
                )
                torch._dynamo.mark_static_address(conv_state)
                self.conv_states.append(conv_state)
        else:
            self.conv_states = conv_states


    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        if self.conv_states[layer_idx].device.type == "meta":
            self.conv_states[layer_idx] = torch.zeros_like(
                self.conv_states[layer_idx],
                device=new_conv_state.device,
            )

        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(device=conv_state.device, dtype=conv_state.dtype)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor):
        self.ssm_states[layer_idx] = new_ssm_state.to(self.ssm_states[layer_idx].device)
        return self.ssm_states[layer_idx]

    def reset(self):
        for layer_idx in range(len(self.conv_states)):
            if self.conv_states[layer_idx].device.type != "meta":
                # In-place ops prevent breaking the static address
                self.conv_states[layer_idx].zero_()
                self.ssm_states[layer_idx].zero_()

    @property
    def batch_size(self):
        return self.max_batch_size