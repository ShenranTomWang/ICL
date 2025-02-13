import os, json
import numpy as np
from utils.data import load_jsonl, save_jsonl, save_config

def random_option(data: list, seed: int) -> list:
    """assign random option to each data point

    Args:
        data (list)
        seed (int)

    Returns:
        list: randomized
    """
    np.random.seed(seed)
    for dp in data:
        random_output = np.random.randint(0, len(dp["options"]))
        dp["output"] = dp["options"][random_output]
    return data

def random_handler(args) -> None:
    """handler for create data with random output

    Args:
        args (NameSpace): should contain the following keys:
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
            - corpus_path (str): the path to the corpus file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    for dataset in args.datasets:
        data_dir = os.path.join(args.data_dir, dataset)
        if os.path.exists(data_dir):
            for seed in seeds:
                try:
                    train_data_path = os.path.join(data_dir, f"{dataset}_{args.k}_{seed}_train.jsonl")
                    train_data = load_jsonl(train_data_path)
                    train_data = random_option(train_data, seed)
                    
                    dev_data_path = os.path.join(data_dir, f"{dataset}_{args.k}_{seed}_dev.jsonl")
                    dev_data = load_jsonl(dev_data_path)
                    
                    test_data_path = os.path.join(data_dir, f"{dataset}_{args.k}_{seed}_test.jsonl")
                    test_data = load_jsonl(test_data_path)
                    
                    new_train_path = os.path.join(args.data_dir, dataset + "_random", f"{dataset}_random_{args.k}_{seed}_train.jsonl")
                    new_dev_path = os.path.join(args.data_dir, dataset + "_random", f"{dataset}_random_{args.k}_{seed}_dev.jsonl")
                    new_test_path = os.path.join(args.data_dir, dataset + "_random", f"{dataset}_random_{args.k}_{seed}_test.jsonl")
                    
                    save_jsonl(train_data, new_train_path)
                    save_jsonl(dev_data, new_dev_path)
                    save_jsonl(test_data, new_test_path)
                    
                    config_file = os.path.join(args.config_dir, "tasks", dataset)
                    with open(config_file + ".json", "r") as f:
                        config = json.load(f)
                    save_config(config, os.path.join(args.config_dir, "tasks", f"{dataset}_random.json"))
                    
                    print(f"Completed for seed {seed} of dataset {dataset}")
                except Exception as e:
                    print(f"Failed for seed {seed} of dataset {dataset}")
                    print(e)
        else:
            print(f"Data directory for {dataset} does not exist")