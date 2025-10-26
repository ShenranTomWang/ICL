import argparse, json
import torch
from interpretability.fv_maps import FVMap

def main(args):
    with open(f"config/{args.task}.json", "r") as f:
        datasets = json.load(f)
    all_maps = []
    titles = [dataset for dataset in datasets]
    for dataset in datasets:
        task_maps = []
        for seed in args.seed.split(","):
            load_dir = f"{args.load_dir}/{dataset}/{seed}"
            load = f"{load_dir}/{args.fname}.pth" if not args.neg_AIE else f"{load_dir}/neg_{args.fname}{'_F1' if args.use_F1 else ''}.pth"
            map_ = torch.load(load)
            map_.figsize = args.figsize
            torch.save(map_, load)
            task_maps.append(map_)
        task_map = FVMap.mean_of(task_maps)
        all_maps.append(task_map)
    save_path = f"{args.out_dir}/{args.task}_{args.fname}.pdf" if not args.neg_AIE else f"{args.out_dir}/{args.task}_neg_{args.fname}{'_F1' if args.use_F1 else ''}.pdf"
    FVMap.visualize_all(all_maps, titles, save_path=save_path, figsize=task_map.figsize)
    mean_map = FVMap.mean_of(all_maps)
    save_path = f"{args.out_dir}/{args.task}_mean_{args.fname}.pdf" if not args.neg_AIE else f"{args.out_dir}/{args.task}_mean_neg_{args.fname}{'_F1' if args.use_F1 else ''}.pdf"
    mean_map.visualize(title=args.title, save_path=save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate function vectors")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory to save")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--title", type=str, default=None, help="Title for the mean visualization")
    parser.add_argument("--seed", type=str, default="100,13,21,42,87", help="Comma separated list of seeds")
    parser.add_argument("--load_dir", type=str, required=True, help="Directory to load function vector maps from")
    parser.add_argument("--figsize", type=int, nargs=2, default=(12, 10), help="Figure size for visualization")
    parser.add_argument("--neg_AIE", action="store_true", help="Whether to visualize negative AIE maps")
    parser.add_argument("--use_F1", action="store_true", help="Whether the visualization is a F1 score maps")
    parser.add_argument("--fname", type=str, default="function_vectors", help="fv steer file name, only used if neg_AIE is False")
    
    args = parser.parse_args()
    main(args)