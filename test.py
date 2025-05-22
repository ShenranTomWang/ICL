import logging, argparse, os
from utils.utils import log_counters, init_counters
from utils.data import load_data
import numpy as np
import torch
from utils.dataset import Dataset
from interpretability.operators import Operator
import interpretability
from utils.inference import do_inference, evaluate
from constants import ALL_OPERATORS, ALL_DTYPES
            
def run(
    args, dataset, operator, seed, kwargs: dict = {}
) -> float:
    """Run testing with dataset, return performance
    
    Args:
        args (dict): arguments
        dataset (Dataset): dataset
        operator (Operator): operator
        seed (str): seed
        is_classification (bool): whether the task is classification
        kwargs (dict, optional): kwargs for model fwd pass, default is {}.
    
    Returns:
        (float): performance, macro-F1 for classification tasks, accuracy for non-classification tasks
    """
    prediction_path = os.path.join(
        args.out_dir,
        dataset.task,
        seed,
        "{}-{}-{}-{}.txt".format(
            dataset.task,
            args.split,
            "k={}".format(args.k),
            "n={}".format(args.n),
            )
        )
    logger.info(prediction_path)
    if not os.path.exists(prediction_path):
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    predictions = do_inference(operator, dataset, kwargs, args.device, args.verbose)

    groundtruths = dataset.outputs
    f1, acc = evaluate(predictions, groundtruths)
    logger.info(f"F1 = {f1}")
    logger.info(f"Acc = {acc}")

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            if prediction is None:
                f.write("None")
            else:
                f.write(prediction)
                f.write("\n")

    return f1, acc

def main(args):
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    logger.info("Model loaded")
    
    use_demonstrations = args.k != 0

    seeds = args.seed.split(",")
    errors, f1s, accs = [], [], []
    for seed in seeds:
        train_data, test_data = load_data(args.task, args.dataset, args.split, args.k, args.n, seed)

        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            assert len(curr_test_data) > 0
            assert not use_demonstrations or len(curr_train_data) == args.k, (use_demonstrations, len(curr_train_data), args.k)
            
            dataset = Dataset(curr_train_data, curr_test_data, verbose=args.verbose, template=args.use_template)
            dataset.tensorize(operator.tokenizer)
            if args.ablate_top_p_heads > 0:
                fv_map = torch.load(f"{args.fv_map_load_dir}/{test_task}_random/100/function_vectors.pth")
                top_p_heads = operator.top_p_heads(fv_map, args.ablate_top_p_heads)
                kwargs = operator.attention2kwargs(
                    None,
                    attention_intervention_fn=operator.get_fv_remove_head_attn_hook(),
                    scan_intervention_fn=operator.get_fv_remove_head_scan_hook(),
                    heads=top_p_heads
                )
            else:
                kwargs = {}
            f1, acc = run(args, dataset, operator, seed, kwargs=kwargs)

            if f1 is None or acc is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                f1s.append(f1)
                accs.append(acc)

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(f1s) // len(seeds), 100 * np.mean(f1s)))
    logger.info("Accuracy of %s over %d target tasks: %.1f" % (args.task, len(accs) // len(seeds), 100 * np.mean(accs)))

    if len(errors) > 0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--use_template", default=False, action="store_true")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--operator", type=str, required=True, choices=ALL_OPERATORS)
    parser.add_argument("--ablate_top_p_heads", type=float, default=0.0, help="Ablate top p heads")
    parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map from this directory (only needed when ablate_top_p_heads > 0), will use out_dir if not specified")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "out/" + "/".join(args.model.split("/")[-1:])
    if args.fv_map_load_dir is None:
        args.fv_map_load_dir = args.out_dir
    
    assert args.dataset is not None or args.task is not None, "Either dataset or task must be provided"
        
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.operator = getattr(interpretability.operators, args.operator)

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(args)