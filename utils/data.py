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
            data.append(dp)
    return data

def update_task(data: list, variant: str) -> list:
    """Update task in data with variant

    Args:
        data (list): data to update
        variant (str): variant to update to

    Returns:
        list: updated data
    """
    for dp in data:
        dp["task"] = f"{dp['task']}_{variant}"
    return data

def save_jsonl(data: list, path: os.PathLike) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for dp in data:
            f.write(json.dumps(dp) + "\n")

def save_json(config: dict, path: os.PathLike) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
        
def load_config(test_task: str) -> dict:
    config_file = "config/tasks/{}.json".format(test_task)
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

def load_data(task: str | None, dataset: str | None, split: str, k: int, n: int, seed: int) -> tuple:
    """Load train and test data from args using seed, handles both loading by task and dataset cases, empty test_data if split is "demo"
    
    Args:
        task (str | None): task name, if None, dataset must be provided
        dataset (str | None): dataset name, if None, task must be provided
        split (str): split name
        k (int): number of training samples
        n (int): number of test samples to load, -1 to load all
        seed (int): seed

    Returns:
        tuple<list<{task: str, input: str, output: str, options: list<str>}>>: train_data, test_data
    """
    logger = logging.getLogger(__name__)
    if split != "demo":
        if task != None:
            train_data = load_data_by_task(task, "train", k, seed=seed)
            test_data = load_data_by_task(task, split, n, seed=seed)
        else:
            assert dataset is not None
            train_data = load_data_by_datasets(dataset.split(","), k, "train", seed=seed)
            test_data = load_data_by_datasets(dataset.split(","), n, split, seed=seed)
    else:
        if task != None:
            train_data = load_data_by_task(task, "train", k, seed=seed)
        else:
            assert dataset is not None
            train_data = load_data_by_datasets(dataset.split(","), k, "train", seed=seed)
        test_data = []
    logger.info("Loaded data for seed %s" % seed)
    return train_data, test_data

def load_data_by_task(task, split, k, seed=0):
    with open(os.path.join("config", task + ".json"), "r") as f:
        datasets = json.load(f)

    data = load_data_by_datasets(datasets=datasets, k=k, seed=seed, split=split)
    return data

def load_data_by_datasets(datasets, k, split, seed=0):
    logger = logging.getLogger(__name__)
    data = []
    for dataset in datasets:
        try:
            data_path = os.path.join("data", dataset, "{}_{}_{}_{}.jsonl".format(dataset, "16", seed, split))
            with open(data_path, "r") as f:
                for i, line in enumerate(f):
                    if k != -1 and i >= k:
                        break
                    dp = json.loads(line)
                    data.append(dp)
        except Exception as e:
            logger.error(f"Error loading data for {dataset}")
            logger.error(e)
    return data
