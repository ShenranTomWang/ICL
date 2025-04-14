import argparse
import interpretability
from constants import ALL_OPERATORS, ALL_DTYPES
from utils.handlers.function_vectors import function_vectors_handler
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate function vectors")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--operator", type=str, choices=ALL_OPERATORS, required=True, help="Operator to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES, help="Data type to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87", help="Comma separated list of seeds")
    parser.add_argument("--k", type=int, default=10, help="Number of ICL examples to use")
    parser.add_argument("--fv_load_dir", type=str, required=True, help="Directory to load function vectors from")
    
    args = parser.parse_args()
    args.operator = getattr(interpretability.operators, args.operator)
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    seed = args.seed.split(",")
    args.seed = [int(s) for s in seed]
    function_vectors_handler(args)