import shutup; shutup.please()
from transformers import AutoModelForCausalLM
from interpretability.tokenizers import Tokenizer
import torch
from typing import Callable
from interpretability.attention_outputs import AttentionOutput
from abc import ABC, abstractmethod
import os, logging

def layer_callback(resid: torch.Tensor) -> torch.Tensor:
    return resid[0, -1, :]

class Operator(ABC):
    def __init__(self, tokenizer: Tokenizer, model: AutoModelForCausalLM, device: torch.DeviceObjType, dtype: torch.dtype):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        return self.forward(text, **kwargs)
    
    def forward(self, text: str, **kwargs) -> torch.Tensor:
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model(**tokenized, **kwargs)
        return output
    
    @abstractmethod
    def get_attention_add_mean_hook(self) -> Callable:
        """
        Get hook function to add mean attention values along sequence dimension

        Returns:
            Callable: hook function
        """
        pass
    
    def get_attention_mean(self, attn: AttentionOutput) -> AttentionOutput:
        """
        Get mean attention values along sequence dimension
        Args:
            attn (AttentionOutput): attention values
        Returns:
            AttentionOutput: mean attention values
        """
        return attn.mean()
    
    @abstractmethod
    def extract_attention_outputs(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[AttentionOutput]:
        """
        Extract attentions at specified layers of attention stream
        Args:
            inputs (list): list of inputs
            activation_callback (function(torch.Tensor)): callback function to extracted activations, applied to activation values at each layer
        Returns:
            list[AttentionOutput]: attention outputs
        """
        pass
    
    def store_attention_outputs(self, attention_outputs: list[AttentionOutput], dir: str, fnames: list[str] = None) -> None:
        """
        Store attention outputs to specified path
        Args:
            attention_outputs (list[AttentionOutput]): list of attention outputs
            dir (str): directory to save to
            fnames (list[str]): list of filenames to override default naming of indexing
        """
        logger = logging.getLogger(__name__)
        if not os.path.exists(dir):
            os.makedirs(os.path.dirname(dir), exist_ok=True)
        for i, attention_output in enumerate(attention_outputs):
            if attention_output is not None:
                if fnames != None:
                    attention_output.save(f"{dir}/{fnames[i]}.pth")
                else:
                    attention_output.save(f"{dir}/{i}.pth")
        logger.info(f"Stored attention outputs to {dir}")
    
    def load_attention_output(self, fname: str = "") -> AttentionOutput:
        """
        Load attention outputs
        Args:
            dir (str): directory to load attention outputs
            fname (str): special filename
        Returns:
            AttentionOutput: attention outputs
        """
        hybrid_output = torch.load(fname).to(self.device)
        return hybrid_output