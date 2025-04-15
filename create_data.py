import argparse
import utils.handlers.create_data as handlers
import os, json
from constants import ALL_VARIANTS

def main(args: dict) -> None:
    if not args.variant.endswith("correct"):
        handler = getattr(handlers, f"{args.variant}_handler")
    else:
        handler = getattr(handlers, "percent_" + args.variant + "_handler")
    handler(args)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--variant", type=str, required=True, choices=ALL_VARIANTS)
    parser.add_argument("--method", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--config_dir", type=str, default="config")

    args = parser.parse_args()
    
    assert args.datasets is not None or args.task is not None, "Either datasets or task should be provided"
    if args.datasets is None:
        with open(os.path.join("config", args.task + ".json"), "r") as f:
            args.datasets = json.load(f)

    main(args)