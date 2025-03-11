import argparse
import utils.handlers as handlers
import os, json

def main(args: dict) -> None:
    handler = getattr(handlers, args.variant + "_handler")
    handler(args)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--method", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--corpus_path", type=str, default=None)

    args = parser.parse_args()
    
    assert args.datasets is not None or args.task is not None, "Either datasets or task should be provided"
    if args.datasets is None:
        with open(os.path.join("config", args.task + ".json"), "r") as f:
            config = json.load(f)
        args.datasets = config["train"] + config["test"] + config["unseen_domain_test"]

    main(args)