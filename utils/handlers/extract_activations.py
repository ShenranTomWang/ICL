import logging, os
from collections import Counter
from utils.dataset import Dataset
from interpretability import Operator
from interpretability import AttentionOutput

def train_handler(
    train_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
) -> None:
    """
    Activation extraction handler for train split

    Args:
        train_counter (Counter)
        train_data (list)
        test_data (list)
        operator (Operator)
        args (Namespace)
        seed (int)
    """
    logger = logging.getLogger(__name__)
    for train_task in train_counter:
        logger.info(f"Processing {train_task} (demo)")
        curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
        
        dataset = Dataset(curr_train_data, [], add_newlines=args.add_newlines, verbose=args.verbose)
        assert len(dataset.options_raw) == 2, "Steer stream only works for binary classification"
        inputs_1 = [dp["input"] for dp in dataset.train if dp["output"] == dataset.options_raw[0]]
        inputs_0 = [dp["input"] for dp in dataset.train if dp["output"] == dataset.options_raw[1]]
        dir = f"{args.out_dir}/{train_task}/{seed}/{args.split}_steer_{dataset.options_raw[1]}-{dataset.options_raw[0]}_k={int(args.k)}/"
        run_operator(operator, args, seed, train_task, (inputs_1, inputs_0), dir)

def demo_handler(
    train_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    """
    logger = logging.getLogger(__name__)
    for train_task in train_counter:
        logger.info(f"Processing {train_task} (demo)")
        curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
        
        dataset = Dataset(curr_train_data, [], add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.prepare_demo()
        run_operator(operator, args, seed, train_task, [dataset.demo], f"{args.out_dir}/{train_task}/{seed}/{args.split}_{args.stream}")
        
def dev_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed)
    
def test_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    """
    basic_handler(test_counter, train_data, test_data, operator, args, seed)

def basic_handler(
    test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    """
    logger = logging.getLogger(__name__)
    for test_task in test_counter:
        logger.info(f"Processing {test_task}")
        curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
        curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
        assert len(curr_test_data) > 0
        
        dataset = Dataset(curr_train_data, curr_test_data, add_newlines=args.add_newlines, verbose=args.verbose)
        dataset.preprocess()
        run_operator(operator, args, seed, test_task, dataset.inputs, f"{args.out_dir}/{test_task}/{seed}/{args.split}_{args.stream}/")

def run_operator(operator: Operator, args, seed: int, test_task: str, inputs: list | tuple, dir: str = "") -> None:
    """
    Run operator to extract activations from dataset
    Args:
        operator (Operator)
        args (NameSpace)
        seed (int)
        dir (str, optional): special filename, Defaults to "".
        test_task (str)
        inputs (list): tuple of list of inputs if stream is steer (need to take difference), otherwise list of inputs
    """
    if args.stream == "resid":
        activation = operator.extract_resid(inputs, layers=args.layers)
        operator.store_resid(activation, f"{args.out_dir}/{test_task}/{seed}/{args.split}", dir)
    elif args.stream == "cache":
        cache = operator.extract_cache(inputs)
        operator.store_cache(cache, f"{args.out_dir}/{test_task}/{seed}/{args.split}", dir)
    elif args.stream == "attn":
        attn = operator.extract_attention_outputs(inputs)
        operator.store_attention_outputs(attn, f"{args.out_dir}/{test_task}/{seed}/{args.split}", dir)
    elif args.stream == "attn_mean":
        attn = operator.extract_attention_outputs(inputs, operator.get_attention_mean)
        operator.store_attention_outputs(attn, f"{args.out_dir}/{test_task}/{seed}/{args.split}_{args.stream}")
    elif args.stream == "steer":
        inputs1, inputs0 = inputs
        steer1 = [output.mean() for output in operator.extract_attention_outputs(inputs1, operator.get_attention_mean)]
        steer0 = [output.mean() for output in operator.extract_attention_outputs(inputs0, operator.get_attention_mean)]
        steer1 = AttentionOutput.mean_of(steer1)
        steer0 = AttentionOutput.mean_of(steer0)
        if steer1 != None:
            steer = steer1 - steer0
        else:
            steer = -1 * steer0
        operator.store_attention_outputs([steer0, steer1, steer], dir, fnames=["steer0", "steer1", "steer"])
    else:
        raise ValueError(f"Invalid stream: {args.stream}")