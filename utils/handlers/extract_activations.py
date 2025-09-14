import logging
from collections import Counter
from utils.dataset import Dataset
from interpretability.operators import Operator
from interpretability import AttentionManager

def train_handler(
    train_counter: Counter, test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
        logger.info(f"Processing {train_task} (train)")
        train = [dp for dp in train_data if dp["task"] == train_task]
        options = train[0]["options"]
        assert len(options) == 2, "Steer stream only works for binary classification"
        inputs_0 = [dp["input"] for dp in train if dp["output"] == options[0]]
        inputs_1 = [dp["input"] for dp in train if dp["output"] == options[1]]
        inputs = [dp["input"] for dp in train]
        dir = f"{args.out_dir}/{train_task}/{seed}/{args.split}_{args.stream}"
        if args.stream == "steer":
            dir += f"_k={int(args.k)}/"
            run_operator_steer(
                operator,
                args.stream,
                (inputs_0, inputs_1),
                dir,
                [f"steer_{options[0]}", f"steer_{options[1]}", f"steer_{options[0]}->{options[1]}"]
            )
        else:
            run_operator_generic(operator, args, inputs, f"{dir}/")

def demo_handler(
    train_counter: Counter, test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
        
        dataset = Dataset(curr_train_data, [], verbose=args.verbose, template=args.use_template)
        dataset.prepare_demo()
        run_operator_generic(operator, args, [dataset.demo], f"{args.out_dir}/{train_task}/{seed}/{args.split}_{args.stream}")
        
def dev_handler(
    train_counter: Counter, test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    if args.stream == "fv_steer":
        logger = logging.getLogger(__name__)
        for train_task in train_counter:
            logger.info(f"Processing {train_task} (dev)")
            curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
            curr_test_data = [dp for dp in test_data if dp["task"] == train_task]
            
            dataset = Dataset(curr_train_data, curr_test_data, verbose=args.verbose, template=args.use_template)
            run_operator_fv_steer(args, operator, dataset, f"{args.out_dir}/{train_task}/{seed}/", seed)
    elif args.stream == "attn_mean" and args.choice != None:
        logger = logging.getLogger(__name__)
        for train_task in train_counter:
            logger.info(f"Processing {train_task} (dev)")
            curr_train_data = [dp for dp in train_data if dp["task"] == train_task]
            curr_test_data = [dp for dp in test_data if dp["task"] == train_task]
            
            dataset = Dataset(curr_train_data, curr_test_data, verbose=args.verbose, template=args.use_template)
            dataset.choose(args.choice, seed)
            dataset.preprocess()
            run_operator_generic(operator, args, dataset.inputs, f"{args.out_dir}/{train_task}/{seed}/{args.split}_{args.stream}_choice={args.choice}/")
    else:
        basic_handler(test_counter, train_data, test_data, operator, args, seed)
    
def test_handler(
    train_counter: Counter, test_counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    counter: Counter, train_data: list, test_data: list, operator: Operator, args, seed: int
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
    for task in counter:
        logger.info(f"Processing {task}")
        curr_test_data = [dp for dp in test_data if dp["task"] == task]
        curr_train_data = [dp for dp in train_data if dp["task"] == task]
        assert len(curr_test_data) > 0
        
        dataset = Dataset(curr_train_data, curr_test_data, verbose=args.verbose, template=args.use_template)
        dataset.preprocess()
        run_operator_generic(operator, args, dataset.inputs, f"{args.out_dir}/{task}/{seed}/{args.split}_{args.stream}/")

def run_operator_generic(operator: Operator, args, inputs: list, dir: str = "") -> None:
    """
    Run operator to extract activations from dataset
    Args:
        operator (Operator)
        args (NameSpace)
        seed (int)
        dir (str, optional): special filename, Defaults to "".
        test_task (str)
        inputs (list)
    """
    if args.stream == "attn":
        attn = operator.extract_attention_managers(inputs)
        operator.store_attention_managers(attn, dir)
    elif args.stream == "attn_mean":
        attn = operator.extract_attention_managers(inputs, operator.get_attention_last_token)
        attn = AttentionManager.mean_of(attn)
        operator.store_attention_managers([attn], dir, fnames=["attn_mean"])
    else:
        raise ValueError(f"Invalid stream split combination: {args.stream} and {args.split}")
    
def run_operator_steer(operator: Operator, stream: str, inputs: list, dir: str, fnames: list) -> None:
    """
    Run operator to extract activations

    Args:
        operator (Operator)
        stream (str)
        inputs (list)
        dir (str)
        fnames (list): list of filenames to override default naming of indexing, should be length 3 for steer0, steer1 and steer1 - steer0
    """
    if stream == "steer":
        inputs0, inputs1 = inputs
        steer1 = [output.mean() for output in operator.extract_attention_managers(inputs1, operator.get_attention_mean)]
        steer0 = [output.mean() for output in operator.extract_attention_managers(inputs0, operator.get_attention_mean)]
        steer1 = AttentionManager.mean_of(steer1)
        steer0 = AttentionManager.mean_of(steer0)
        if steer1 != None:
            steer = steer1 - steer0
        else:
            steer = -1 * steer0
        operator.store_attention_managers([steer0, steer1, steer], dir, fnames=fnames)
    else:
        raise ValueError(f"Invalid stream: {stream}")
    
def run_operator_fv_steer(args, operator: Operator, dataset: Dataset, dir: str, seed: int) -> None:
    """
    Run operator to extract activations

    Args:
        args (Namespace)
        operator (Operator)
        dataset (Dataset)
        dir (str): directory to store outputs
        seed (int): seed
    """
    dataset.choose(args.choice, seed)
    dataset.preprocess()
    attn = operator.extract_attention_managers(dataset.inputs, operator.get_attention_last_token)
    attn = AttentionManager.mean_of(attn)
    operator.store_attention_managers([attn], dir, fnames=["fv_steer"])