# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import numpy as np
import logging

def load_jsonl(path: os.PathLike) -> list:
    data = []
    with open(path, "r") as f:
        for line in f:
            dp = json.loads(line)
            dp["task"] = "{}_random".format(dp["task"])
            data.append(dp)
    return data


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


def save_jsonl(data: list, path: os.PathLike) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")


def save_config(config: dict, path: os.PathLike) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def random_handler(args: dict) -> None:
    """handler for create data with random output

    Args:
        args (dict): should contain the following keys:
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


def load_data_by_task(task, split, k, seed=0, config_split=None, is_null=False):
    if config_split is None:
        config_split = split

    with open(os.path.join("config", task + ".json"), "r") as f:
        config = json.load(f)
    datasets = config[config_split]

    data = load_data_by_datasets(datasets=datasets, k=k, seed=seed, split=split, is_null=is_null)
    return data

def load_data_by_datasets(datasets, k, split, seed=0, is_null=False):
    logger = logging.getLogger(__name__)
    assert k <= 16
    data = []
    for dataset in datasets:
        try:
            data_path = os.path.join("data", dataset, "{}_{}_{}_{}.jsonl".format(dataset, "16", seed, split))
            with open(data_path, "r") as f:
                for i, line in enumerate(f):
                    if k != -1 and i >= k:
                        break
                    dp = json.loads(line)
                    if is_null:
                        dp["input"] = "N/A"
                    data.append(dp)
        except Exception as e:
            logger.error(f"Error loading data for {dataset}")
            logger.error(e)
    return data
