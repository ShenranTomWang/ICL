import argparse, logging
import interpretability
from constants import ALL_OPERATORS, ALL_DTYPES
import utils.handlers.function_vectors as handlers
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate function vectors. This script requires you to run extract_activations.py with fv_steer first.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--operator", type=str, choices=ALL_OPERATORS, required=True, help="Operator to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES, help="Data type to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87", help="Comma separated list of seeds")
    parser.add_argument("--k", type=int, default=10, help="Number of ICL examples to use")
    parser.add_argument("--n", type=int, default=25, help="Number of inputs to use for each task")
    parser.add_argument("--fv_load_dir", type=str, required=True, help="Directory to load function vectors from")
    parser.add_argument("--use_template", default=False, action="store_true", help="Use template for ICL")
    subparser = parser.add_subparsers(dest="operation", required=True)
    aie_subparser = subparser.add_parser("AIE", help="Generate AIE function vectors")
    aie_subparser.add_argument("--split", type=str, default="dev", choices=["test", "dev"], help="Split to use for AIE")
    neg_aie_subparser = subparser.add_parser("neg_AIE", help="Generate negative AIE function vectors")
    neg_aie_subparser.add_argument("--split", type=str, default="dev", choices=["test", "dev"], help="Split to use for negative AIE")
    
    args = parser.parse_args()
    args.operator = getattr(interpretability.operators, args.operator)
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    seed = args.seed.split(",")
    args.seed = [int(s) for s in seed]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)

    handler = getattr(handlers, f"{args.operation}_handler")
    handler(args)