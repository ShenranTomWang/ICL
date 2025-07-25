import argparse, json
import torch
from interpretability.fv_maps import FVMap
from utils.data import to_base

def main(args):
    with open(f"config/{args.task}.json", "r") as f:
        datasets = json.load(f)
    all_maps = []
    titles = [to_base(dataset) for dataset in datasets]
    for dataset in datasets:
        task_maps = []
        for seed in args.seed.split(","):
            load_dir = f"{args.load_dir}/{dataset}/{seed}"
            map_ = torch.load(f"{load_dir}/function_vectors.pth")
            map_.figsize = args.figsize
            torch.save(map_, f"{load_dir}/function_vectors.pth")
            task_maps.append(map_)
        task_map = FVMap.mean_of(task_maps)
        all_maps.append(task_map)
    FVMap.visualize_all(all_maps, titles, save_path=f"{args.out_dir}/{args.task}_function_vectors.pdf", figsize=task_map.figsize)
    mean_map = FVMap.mean_of(all_maps)
    mean_map.visualize(f"{args.out_dir}/{args.task}_mean_function_vector.pdf")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate function vectors")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87", help="Comma separated list of seeds")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory to load function vector maps from")
    parser.add_argument("--figsize", type=int, nargs=2, default=(12, 8), help="Figure size for visualization")
    
    args = parser.parse_args()
    main(args)