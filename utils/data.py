# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging

def load_jsonl(path: os.PathLike) -> list:
    data = []
    with open(path, "r") as f:
        for line in f:
            dp = json.loads(line)
            dp["task"] = "{}_random".format(dp["task"])
            data.append(dp)
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
        
def load_config(test_task: str) -> dict:
    config_file = "config/tasks/{}.json".format(test_task)
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def load_data(args, seed) -> tuple:
    """Load train and test data from args using seed, handles both loading by task and dataset cases, empty test_data if split is "demo"

    Returns:
        tuple<list<{task: str, input: str, output: str, options: list<str>}>>: train_data, test_data
    """
    logger = logging.getLogger(__name__)
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
    if args.split != "demo":
        if args.task != None:
            train_data = load_data_by_task(args.task, "train", args.k, seed=seed, config_split=config_split)
            test_data = load_data_by_task(args.task, args.split, args.n, seed=seed, config_split=config_split, is_null=args.is_null)
        else:
            assert args.dataset is not None
            train_data = load_data_by_datasets(args.dataset.split(","), args.k, "train", seed=seed)
            test_data = load_data_by_datasets(args.dataset.split(","), args.n, args.split, seed=seed, is_null=args.is_null)
    else:
        if args.task != None:
            train_data = load_data_by_task(args.task, "train", args.k, seed=seed, config_split=config_split)
        else:
            assert args.dataset is not None
            train_data = load_data_by_datasets(args.dataset.split(","), args.k, "train", seed=seed)
        test_data = []
    logger.info("Loaded data for seed %s" % seed)
    return train_data, test_data

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
