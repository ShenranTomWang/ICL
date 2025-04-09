import argparse, logging
import torch
from utils.data import load_data
import utils.utils as utils
import utils.handlers.extract_activations as handlers
import interpretability
from constants import ALL_DTYPES, ALL_OPERATORS

def main(args: object, logger: logging.Logger) -> None:
    operator: interpretability.Operator = args.operator(args.model, args.device, args.dtype)
    
    if args.layers == "-1":
        args.layers = [i for i in range(operator.model.config.num_hidden_layers)]
    else:
        args.layers = [int(layer) for layer in args.layers.split(",")]
    logger.info(f"Layers: {args.layers}")
    
    args.seed = [int(seed) for seed in args.seed.split(",")]
    for seed in args.seed:
        train_data, test_data = load_data(args.task, args.dataset, args.split, args.k, args.n, seed)
        train_counter, test_counter = utils.init_counters(train_data, test_data)
        utils.log_counters(train_counter, test_counter)
        args.handler(train_counter, test_counter, train_data, test_data, operator, args, seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="demo", choices=["demo", "test", "dev", "train"])
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--n", type=int, default=-1, help="number of test points to use")
    parser.add_argument("--k", type=int, default=4, help="number of examples")
    
    parser.add_argument("--layers", type=str, default="-1", help="comma separated list of layer indices, or -1 for all layers")
    parser.add_argument("--stream", type=str, default="resid", choices=["attn", "attn_mean", "attn_last", "steer"])
    parser.add_argument("--operator", type=str, default="TransformerOperator", choices=ALL_OPERATORS)
    parser.add_argument("--verbose", default=False, action="store_true")
    
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="out/activations")
    args = parser.parse_args()
    
    if args.stream == "steer":
        assert args.split == "train" or args.split == "demo", "Can only extract steer stream for train or demo split"
    args.device = torch.device(args.device)
    args.handler = getattr(handlers, f"{args.split}_handler")
    args.operator = getattr(interpretability, args.operator)
    args.dtype = getattr(torch, args.dtype)
    assert args.task is not None or args.dataset is not None, "Either task or dataset must be provided"
    
    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)
    main(args, logger)
    