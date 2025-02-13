import logging, os
from collections import Counter
from utils.dataset import Dataset
from interpretability.transformer_operator import TransformerOperator
import torch

def demo_handler(train_counter: Counter, train_data: list, test_data: list, operator: TransformerOperator, args, seed: int) -> None:
    """
    Activation extraction handler for demo split
    
    Args:
        train_counter (Counter)
        train_data (list)
        test_data (list)
        operator (TransformerOperator)
        args (NameSpace)
        seed (int)
    """
    logger = logging.getLogger(__name__)
    for train_task in train_counter:
        logger.info(f"Processing {train_task} (demo)")
        curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
        
        dataset = Dataset(curr_train_data, [], add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.prepare_demo()
        activation = operator.extract_tf([dataset.demo], layers=args.layers, activation_callback=lambda x: x[..., -1, :])      # extract last token
        activation = torch.stack(activation, dim=0)       # (n, n_layers, 1, hidden_size)
        
        out_path = f"{args.out_dir}/{train_task}/{seed}/post_resid.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(activation, out_path)
        logger.info(f"Saved activations to {out_path}")
        
def dev_handler(test_counter: Counter, train_data: list, test_data: list, operator: TransformerOperator, args, seed: int) -> None:
    """
    Activation extraction handler for dev split
    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (TransformerOperator)
        args (NameSpace)
        seed (int)
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed)
    
def test_handler(test_counter: Counter, train_data: list, test_data: list, operator: TransformerOperator, args, seed: int) -> None:
    """
    Activation extraction handler for test split
    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (TransformerOperator)
        args (NameSpace)
        seed (int)
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed)

def basic_handler(test_counter: Counter, train_data: list, test_data: list, operator: TransformerOperator, args, seed: int) -> None:
    """
    Activation extraction handler for generic split

    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (TransformerOperator)
        args (NameSpace)
        seed (int)
    """
    logger = logging.getLogger(__name__)
    for test_task in test_counter:
        logger.info(f"Processing {test_task}")
        curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
        curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
        assert len(curr_test_data) > 0
        
        dataset = Dataset(curr_train_data, curr_test_data, add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.preprocess()
        activation = operator.extract_tf(dataset.inputs, layers=args.layers, activation_callback=lambda x: x[..., -1, :])
        activation = torch.stack(activation, dim=0)       # (n, n_layers, 1, hidden_size)
        
        out_path = f"{args.out_dir}/{test_task}/{seed}/post_resid.pt"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(activation, out_path)
        logger.info(f"Saved activations to {out_path}")