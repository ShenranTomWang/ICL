import shutup; shutup.please()
from transformers import AutoModelForCausalLM
from interpretability.tokenizers import Tokenizer
from interpretability.fv_maps import FVMap
import torch
from typing import Callable
import numpy as np
from collections import defaultdict
from interpretability.attention_managers import AttentionManager
from abc import ABC, abstractmethod
import os, logging

def evaluate(predictions: list, groundtruths: list) -> tuple[float]:
    """Evaluate the predictions against the groundtruths.
    Args:
        predictions (list): list of predictions
        groundtruths (list): list of groundtruths
    Returns:
        (float, float): F1, accuracy
    """
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        if prediction is None:
            continue
        prediction = prediction
        is_correct = prediction in groundtruth if type(groundtruth) == list else prediction == groundtruth
        accs.append(is_correct)
        recalls[groundtruth].append(is_correct)
        precisions[prediction].append(is_correct)

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision + recall == 0:
            f1s.append(0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))

    return np.mean(f1s), np.mean(accs)

class Operator(ABC):
    """
    Operator class is a base class that operates an LLM. It contains all the methods necessary for interpretability experiments.
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: AutoModelForCausalLM,
        device: torch.DeviceObjType,
        dtype: torch.dtype,
        n_layers: int,
        total_n_heads: int
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        self.ALL_LAYERS = [i for i in range(n_layers)]
        self.total_n_heads = total_n_heads
        
    def __call__(self, text: str, **kwargs) -> torch.Tensor:
        return self.forward(text, **kwargs)
    
    def forward(self, text: str, **kwargs) -> torch.Tensor:
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model(**tokenized, **kwargs)
        return output
    
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
    
    def top_p_heads(self, fv_map: FVMap, top_p: float, **kwargs) -> map:
        """
        Get top p heads from fv_map
        Args:
            fv_map (FVMap): fv_map object
            top_p (float): top p value in [0, 1]
            kwargs: additional arguments passed to fv_map for top p function usage
        Returns:
            map[int: list[{head: int, stream: str}]]: map of top p heads in corresponding streams at specific layers.
                This is to be passed to hooks as a kwarg, stream is one of attn or scan
        """
        top_p_heads = fv_map.top_p_heads(top_p, **kwargs)
        return top_p_heads
    
    def exclusion_ablation_heads(self, fv_map: FVMap, top_p: float, ablation_p: float, **kwargs) -> map:
        """
        Get exclusion ablation heads from fv_map. This will select random heads not in top p function heads.
        Args:
            fv_map (FVMap): fv_map object
            top_p (float): top p value in [0, 1] for top p function heads to exclude
            ablation_p (float): ablation p value in [0, 1] for number of heads to select for ablation
            kwargs: additional arguments passed to fv_map for ablation usage
        Returns:
            map[int: list[{head: int, stream: str}]]: map of exclusion ablation heads in corresponding streams at specific layers.
                This is to be passed to hooks as a kwarg, stream is one of attn or scan
        """
        exclusion_ablation_heads = fv_map.exclusion_ablation_heads(top_p, ablation_p, **kwargs)
        return exclusion_ablation_heads

    @abstractmethod
    def get_fv_remove_head_attn_hook(self) -> Callable:
        """
        Get hook function to remove head from attention stream, returns None if not supported
        Returns:
            Callable: hook function
        """
        pass
    
    @abstractmethod
    def get_fv_remove_head_scan_hook(self) -> Callable:
        """
        Get hook function to remove head from scan stream, returns None if not supported
        Returns:
            Callable: hook function
        """
        pass
    
    @abstractmethod
    def attention2kwargs(self, attn: AttentionManager | None, layers: list[int] = None, **kwargs) -> dict:
        """
        Convert attention manager to kwargs for model override
        Args:
            attn (AttentionManager | None): attention manager, set to None for a dummy value
            layers (list[int], optional): layers to perform intervention, defaults to None to perform on all layers
            **kwargs: additional arguments
        Returns:
            dict: kwargs for model
        """
        pass
    
    @abstractmethod
    def generate_AIE_map(
        self,
        steer: list[AttentionManager],
        inputs: list[list[str]],
        label_ids: list[torch.Tensor],
        return_F1: bool = False,
        **kwargs
    ) -> FVMap:
        """
        Generate AIE map for each attention head at each layer using inputs and steer for a list of tasks

        Args:
            steer (list[AttentionManager]): steer values for each task
            inputs (list[list[str]]): list of inputs for each task
            label_ids (list[torch.Tensor]): list of label ids for each task
            return_F1 (bool, optional): whether to return F1 score, defaults to False to return AIE
            **kwargs: additional arguments, potentially including intervention functions for attention/scan
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
            original_tasks (list[torch.Tensor]): last logits [(batch_size, vocab_size)] * n_tasks
            label_ids (list[torch.Tensor]): [(batch_size,)] * n_tasks

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
    
    @staticmethod
    def compute_F1(intervened_tasks: list[torch.Tensor], original_tasks: list[torch.Tensor], label_ids: list[torch.Tensor]) -> float:
        """
        Compute F1 score for batch

        Args:
            intervened_tasks (list[torch.Tensor]): last logits [(batch_size, vocab_size)] * n_tasks
            original_tasks (list[torch.Tensor]): last logits [(batch_size, vocab_size)] * n_tasks
            label_ids (list[torch.Tensor]): [(batch_size,)] * n_tasks

        Returns:
            float: F1 score
        """
        assert len(intervened_tasks) == len(original_tasks) == len(label_ids), "intervened_tasks, original_tasks, and label_ids must have the same length"
        all_preds = []
        all_labels = []
        for i in range(len(intervened_tasks)):
            intervened_batch = intervened_tasks[i]
            label_batch = label_ids[i]
            preds = torch.argmax(intervened_batch, dim=-1).cpu().numpy()
            labels = label_batch.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
        f1, _ = evaluate(all_preds, all_labels)
        return f1