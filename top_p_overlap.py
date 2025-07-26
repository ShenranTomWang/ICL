import os, argparse, json
import interpretability
import torch
from constants import ALL_DTYPES
from interpretability.fv_maps.fv_map import FVMap
import logging

def top_p2set(top_p: map) -> set:
    myset = set()
    for layer, heads in top_p.items():
        for head in heads:
            myset.add((layer, head['head'], head['stream']))
    return myset

def main(args):
    operator: interpretability.operators.Operator = args.operator(args.model, args.dtype, args.device)
    with open(f"config/{args.task1}.json", "r") as f:
        task1 = json.load(f)
    with open(f"config/{args.task2}.json", "r") as f:
        task2 = json.load(f)
    task1_fv_maps, task2_fv_maps = [], []
    for task in task1:
        fv_map = torch.load(os.path.join(args.fv_map_load_dir, task, "100", "function_vectors.pth"))
        task1_fv_maps.append(fv_map)
    for task in task2:
        fv_map = torch.load(os.path.join(args.fv_map_load_dir, task, "100", "function_vectors.pth"))
        task2_fv_maps.append(fv_map)
    task1_fv_map = FVMap.mean_of(task1_fv_maps)
    task2_fv_map = FVMap.mean_of(task2_fv_maps)
    top_p_heads1 = top_p2set(operator.top_p_heads(task1_fv_map, top_p=args.p))
    top_p_heads2 = top_p2set(operator.top_p_heads(task2_fv_map, top_p=args.p))
    overlap = top_p_heads1.intersection(top_p_heads2)
    if len(top_p_heads1) == 0:
        logger.warning(f"No heads found for task {args.task1} with top p {args.p}.")
        return
    overlap_percentage = len(overlap) / len(top_p_heads1) * 100
    logger.info(f"Overlap percentage between top {args.p} heads: {overlap_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run top_p_overlap.py script")
    parser.add_argument("--operator", type=str, required=True, help="Operator to use")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--task1", type=str, required=True, help="Path to the first task")
    parser.add_argument("--task2", type=str, required=True, help="Path to the second task")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES, help="Data type to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--log_file", type=str, default="out/log.log", help="Log file path")
    parser.add_argument("--fv_map_load_dir", type=str, default="out", help="Directory to load FV maps from, defaults to 'out'")
    parser.add_argument("--p", type=float, required=True, help="Top p value for overlap calculation")
    args = parser.parse_args()
    
    args.operator = getattr(interpretability.operators, args.operator)
    args.device = torch.device(args.device)
    args.dtype = getattr(torch, args.dtype)
    
    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    main(args)
