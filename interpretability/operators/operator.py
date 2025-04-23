import shutup; shutup.please()
from transformers import AutoModelForCausalLM
from interpretability.tokenizers import Tokenizer
from interpretability.fv_maps import FVMap
import torch
from typing import Callable
from interpretability.attention_managers import AttentionManager
from abc import ABC, abstractmethod
import os, logging

class Operator(ABC):
    """
    Operator class is a base class that operates an LLM. It contains all the methods necessary for interpretability experiments.
    """
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
    
    def get_attention_mean(self, attn: AttentionManager) -> AttentionManager:
        """
        Get mean attention values along sequence dimension
        Args:
            attn (AttentionManager): attention values
        Returns:
            AttentionManager: mean attention values
        """
        return attn.mean()
    
    def get_attention_last_token(self, attn: AttentionManager) -> AttentionManager:
        """
        Get attention values of last token
        Args:
            attn (AttentionManager): attention values
        Returns:
            AttentionManager: last token attention values at each layer
        """
        return attn.get_last_token()
    
    @abstractmethod
    def extract_attention_managers(self, inputs: list[str], activation_callback: Callable = lambda x: x) -> list[AttentionManager]:
        """
        Extract attentions at specified layers of attention stream
        Args:
            inputs (list): list of inputs
            activation_callback (function(AttentionManager)): callback function to extracted activations, applied to activation values at each layer
        Returns:
            list[AttentionManager]: attention outputs
        """
        pass
    
    def store_attention_managers(self, attention_outputs: list[AttentionManager], dir: str, fnames: list[str] = None) -> None:
        """
        Store attention managers to specified path
        Args:
            attention_outputs (list[AttentionManager]): list of attention outputs
            dir (str): directory to save to
            fnames (list[str]): list of filenames to override default naming of indexing
        """
        logger = logging.getLogger(__name__)
        if not dir.endswith("/"):
            dir += "/"
        if not os.path.exists(dir):
            os.makedirs(os.path.dirname(dir), exist_ok=True)
        for i, attention_output in enumerate(attention_outputs):
            if attention_output is not None:
                if fnames != None:
                    attention_output.save(f"{dir}{fnames[i]}.pth")
                else:
                    attention_output.save(f"{dir}{i}.pth")
        logger.info(f"Stored attention outputs to {dir}")
    
    def load_attention_manager(self, fname: str = "") -> AttentionManager:
        """
        Load attention manager object
        Args:
            dir (str): directory to load attention outputs
            fname (str): special filename
        Returns:
            AttentionManager: attention outputs
        """
        manager = torch.load(fname).to(self.device)
        return manager
    
    @abstractmethod
    def attention2kwargs(self, attn: AttentionManager, **kwargs) -> dict:
        """
        Convert attention manager to kwargs for model override
        Args:
            attn (AttentionManager): attention manager
            **kwargs: additional arguments
        Returns:
            dict: kwargs for model
        """
        pass
    
    @abstractmethod
    def generate_AIE_map(self, steer: list[AttentionManager], inputs: list[list[str]], label_ids: list[torch.Tensor]) -> FVMap:
        """
        Generate AIE map for each attention head at each layer using inputs and steer for a list of tasks

        Args:
            steer (list[AttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
        Returns:
            FVMap: FVMap object
        """
        pass

    @staticmethod
    def compute_CIE(intervened_logits: torch.Tensor, original_logits: torch.Tensor, label_ids: torch.Tensor) -> float:
        """
        Compute CIE (conditional indirect effect) for batch

        Args:
            intervened_logits (torch.Tensor): last logit after intervention, (batch_size, vocab_size)
            original_logits (torch.Tensor): last logit without intervention, (batch_size, vocab_size)
            label_ids (torch.Tensor): ids of labels, (batch_size,)

        Returns:
            float: CIE value
        """
        intervened_logits = torch.softmax(intervened_logits, dim=-1)
        original_logits = torch.softmax(original_logits, dim=-1)
        label_ids = label_ids.unsqueeze(-1)
        intervened_logits = intervened_logits.gather(1, label_ids)
        original_logits = original_logits.gather(1, label_ids)
        intervened_logits = intervened_logits.squeeze(-1)
        original_logits = original_logits.squeeze(-1)
        cie = intervened_logits - original_logits
        return cie.mean().item()

    @staticmethod
    def compute_AIE(intervened_tasks: list[torch.Tensor], original_tasks: list[torch.Tensor], label_ids: list[torch.Tensor]) -> float:
        """
        Compute AIE (average indirect effect) for batch

        Args:
            intervened_tasks (list[torch.Tensor]): last logits [(batch_size, vocab_size)] * n_tasks
            original_tasks (list[torch.Tensor]): last logits [(n_tasks, batch_size, vocab_size)] * n_tasks
            label_ids (list[torch.Tensor]): [(n_tasks, batch_size)] * n_tasks

        Returns:
            float: AIE value
        """
        assert len(intervened_tasks) == len(original_tasks) == len(label_ids), "intervened_tasks, original_tasks, and label_ids must have the same length"
        aie = 0
        for i in range(len(intervened_tasks)):
            intervened_batch = intervened_tasks[i]
            original_batch = original_tasks[i]
            label_batch = label_ids[i]
            cie = Operator.compute_CIE(intervened_batch, original_batch, label_batch)
            aie += cie
        aie /= len(intervened_tasks)
        return aie