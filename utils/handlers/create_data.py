import os, json
import numpy as np
from utils.data import load_jsonl, save_jsonl, save_json
from typing import Callable

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

def percent_option(data: list, seed: int, percent: float) -> list:
    """assign option to each data point such that we have <percent> of correct answers

    Args:
        data (list)
        seed (int)
        percent (float): percentage of correct answers

    Returns:
        list: randomized
    """
    np.random.seed(seed)
    correct_indeces = np.random.choice(len(data), int(len(data) * percent), replace=False)
    
    for i, dp in enumerate(data):
        if i not in correct_indeces:
            options = dp["options"]
            correct_option = options.index(dp["output"])
            incorrect_options = options[:correct_option] + options[correct_option + 1:]
            option = np.random.choice(incorrect_options)
            dp["output"] = option
    return data

def percent_0_correct_handler(args) -> None:
    """handler for create data with 0_correct output

    Args:
        args (NameSpace): should contain the following keys:
            - task (str): the task name
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    generic_handler(
        datasets=args.datasets,
        seeds=seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        task=args.task,
        variant=args.variant,
        k=args.k,
        option_fn=percent_option,
        percent=0.0,
    )

def percent_25_correct_handler(args) -> None:
    """handler for create data with 25_correct output

    Args:
        args (NameSpace): should contain the following keys:
            - task (str): the task name
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    generic_handler(
        datasets=args.datasets,
        seeds=seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        task=args.task,
        variant=args.variant,
        k=args.k,
        option_fn=percent_option,
        percent=0.25,
    )
    
def percent_50_correct_handler(args) -> None:
    """handler for create data with 50_correct output

    Args:
        args (NameSpace): should contain the following keys:
            - task (str): the task name
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    generic_handler(
        datasets=args.datasets,
        seeds=seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        task=args.task,
        variant=args.variant,
        k=args.k,
        option_fn=percent_option,
        percent=0.5,
    )
    
def percent_75_correct_handler(args) -> None:
    """handler for create data with 75_correct output

    Args:
        args (NameSpace): should contain the following keys:
            - task (str): the task name
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    generic_handler(
        datasets=args.datasets,
        seeds=seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        task=args.task,
        variant=args.variant,
        k=args.k,
        option_fn=percent_option,
        percent=0.75,
    )

def random_handler(args) -> None:
    """handler for create data with random output

    Args:
        args (NameSpace): should contain the following keys:
            - task (str): the task name
            - dataset (str): the dataset name
            - k (int): the number of distractors
            - seed (str): the seed for random number generator
            - data_dir (str): the directory to save the data
            - variant (str): the variant of the handler
            - method (str): the method to use for creating the data (direct or channel)
            - config_dir (str): the directory to save the config file
    """
    seeds = [int(seed) for seed in args.seed.split(",")]
    generic_handler(
        datasets=args.datasets,
        seeds=seeds,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        task=args.task,
        variant=args.variant,
        k=args.k,
        option_fn=random_option
    )
        
def generic_handler(
    datasets: list[str],
    seeds: list[int],
    data_dir: str,
    config_dir: str,
    task: str,
    variant: str,
    k: int,
    option_fn: Callable,
    **kwargs
) -> None:
    """generic handler

    Args:
        datasets (list[str]): list of datasets
        seeds (list[int]): list of seeds
        data_dir (str): directory to save the data
        config_dir (str): directory to save the config file
        task (str): task name
        variant (str): variant of the handler
        k (int): number of distractors
        option_fn (Callable): function to shuffle options
        **kwargs: additional arguments for option_fn
    """
    for dataset in datasets:
        curr_data_dir = os.path.join(data_dir, dataset)
        if os.path.exists(curr_data_dir):
            for seed in seeds:
                try:
                    train_data_path = os.path.join(curr_data_dir, f"{dataset}_{k}_{seed}_train.jsonl")
                    train_data = load_jsonl(train_data_path)
                    train_data = option_fn(train_data, seed, **kwargs)
                    
                    dev_data_path = os.path.join(curr_data_dir, f"{dataset}_{k}_{seed}_dev.jsonl")
                    dev_data = load_jsonl(dev_data_path)
                    
                    test_data_path = os.path.join(curr_data_dir, f"{dataset}_{k}_{seed}_test.jsonl")
                    test_data = load_jsonl(test_data_path)
                    
                    new_train_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_train.jsonl")
                    new_dev_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_dev.jsonl")
                    new_test_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_test.jsonl")
                    
                    save_jsonl(train_data, new_train_path)
                    save_jsonl(dev_data, new_dev_path)
                    save_jsonl(test_data, new_test_path)
                    
                    config_file = os.path.join(config_dir, "tasks", dataset)
                    with open(config_file + ".json", "r") as f:
                        config = json.load(f)
                    save_json(config, os.path.join(config_dir, "tasks", f"{dataset}_{variant}.json"))
                    
                    print(f"Completed for seed {seed} of dataset {dataset}")
                except Exception as e:
                    print(f"Failed for seed {seed} of dataset {dataset}")
                    print(e)
        else:
            print(f"Data directory for {dataset} does not exist")
    if task != None:
        save_config(config_dir, datasets, task, variant)

def save_config(config_dir: str, datasets: list, task: str, variant: str) -> None:
    config = [f"{dataset}_{variant}" for dataset in datasets]
    config_file = os.path.join(config_dir, f"{task}_{variant}.json")
    save_json(config, config_file)