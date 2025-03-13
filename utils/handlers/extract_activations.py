import logging
from collections import Counter
from utils.dataset import Dataset
from interpretability import Operator

def demo_handler(
    train_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int, fname: str = None
) -> None:
    """
    Activation extraction handler for demo split
    
    Args:
        train_counter (Counter)
        train_data (list)
        test_data (list)
        operator (Operator)
        args (NameSpace)
        seed (int)
        fname (str): filename
    """
    logger = logging.getLogger(__name__)
    for train_task in train_counter:
        logger.info(f"Processing {train_task} (demo)")
        curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
        
        dataset = Dataset(curr_train_data, [], add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.prepare_demo()
        run_operator(operator, args, seed, fname, train_task, [dataset.demo])
        
def dev_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int, fname: str = None
) -> None:
    """
    Activation extraction handler for dev split on residual stream
    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (Operator)
        args (NameSpace)
        seed (int)
        fname (str): filename
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed, fname)
    
def test_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int, fname: str = None
) -> None:
    """
    Activation extraction handler for test split on residual stream
    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (Operator)
        args (NameSpace)
        seed (int)
        fname (str): filename
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed, fname)

def basic_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int, fname: str
) -> None:
    """
    Activation extraction handler for generic split on residual stream

    Args:
        test_counter (Counter)
        train_data (list)
        test_data (list)
        operator (Operator)
        args (NameSpace)
        seed (int)
        fname (str): filename
    """
    logger = logging.getLogger(__name__)
    for test_task in test_counter:
        logger.info(f"Processing {test_task}")
        curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
        curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
        assert len(curr_test_data) > 0
        
        dataset = Dataset(curr_train_data, curr_test_data, add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.preprocess()
        run_operator(operator, args, seed, fname, test_task, dataset.inputs)

def run_operator(operator: Operator, args, seed: int, fname: str, test_task: str, inputs: list[str]) -> None:
    """
    Run operator to extract activations from dataset
    Args:
        operator (Operator)
        args (NameSpace)
        seed (int)
        fname (str)
        test_task (str)
        inputs (list[str])
    """
    if args.stream == "resid":
        if fname is None:
            fname = f"{args.split}_resid.pt"
        activation = operator.extract_resid(inputs, layers=args.layers)
        operator.store_resid(activation, f"{args.out_dir}/{test_task}/{seed}/{fname}")
    elif args.stream == "cache":
        if fname is None:
            fname = f"{args.split}_cache.pt"
        cache = operator.extract_cache(inputs)
        operator.store_cache(cache, f"{args.out_dir}/{test_task}/{seed}/{fname}")