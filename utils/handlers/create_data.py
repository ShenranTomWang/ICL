import os, json
import numpy as np
from utils.data import load_jsonl, save_jsonl, save_json, update_task
from typing import Callable
from english_words import english_words_set
import copy

def change_options_and_output(data: list, mapping: dict, **kwargs) -> list:
    """Change the options and output of the data according to the mapping

    Args:
        data (list)
        mapping (dict): mapping from option to random english words
        **kwargs: not used

    Returns:
        list
    """
    for dp in data:
        dp["options"] = [mapping[option] for option in dp["options"]]
        dp["output"] = mapping[dp["output"]]
    return data

def random_option(data: list, seed: int, **kwargs) -> list:
    """assign random option to each data point

    Args:
        data (list)
        seed (int)
        **kwargs: not used

    Returns:
        list: randomized
    """
    np.random.seed(seed)
    for dp in data:
        random_output = np.random.randint(0, len(dp["options"]))
        dp["output"] = dp["options"][random_output]
    return data

def percent_option(data: list, seed: int, percent: float, **kwargs) -> list:
    """assign option to each data point such that we have <percent> of correct answers

    Args:
        data (list)
        seed (int)
        percent (float): percentage of correct answers
        **kwargs: not used

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

def random_english_words_handler(args) -> None:
    """handler for create data with random english words output
    
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
    for dataset in args.datasets:
        curr_data_dir = os.path.join(args.data_dir, dataset)
        if os.path.exists(curr_data_dir):
            for seed in seeds:
                try:
                    np.random.seed(seed)
                    train_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_train.jsonl")
                    train_data = load_jsonl(train_data_path)
                    train_data = update_task(train_data, args.variant)
                    true_options = train_data[0]["options"]
                    mapping = {option: np.random.choice(sorted(english_words_set)) for option in true_options}
                    train_data = change_options_and_output(train_data, mapping)
                    
                    dev_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_dev.jsonl")
                    dev_data = load_jsonl(dev_data_path)
                    dev_data = update_task(dev_data, args.variant)
                    dev_data = change_options_and_output(dev_data, mapping)
                    
                    test_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_test.jsonl")
                    test_data = load_jsonl(test_data_path)
                    test_data = update_task(test_data, args.variant)
                    test_data = change_options_and_output(test_data, mapping)
                    
                    new_train_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_train.jsonl")
                    new_dev_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_dev.jsonl")
                    new_test_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_test.jsonl")
                    
                    save_jsonl(train_data, new_train_path)
                    save_jsonl(dev_data, new_dev_path)
                    save_jsonl(test_data, new_test_path)
                    
                    config_file = os.path.join(args.config_dir, "tasks", dataset)
                    with open(config_file + ".json", "r") as f:
                        config = json.load(f)
                    save_json(config, os.path.join(args.config_dir, "tasks", f"{dataset}_{args.variant}.json"))
                    
                    print(f"Completed for seed {seed} of dataset {dataset}")
                except Exception as e:
                    print(f"Failed for seed {seed} of dataset {dataset}")
                    print(e)
        else:
            print(f"Data directory for {dataset} does not exist")
    if args.task != None:
        save_config(args.config_dir, args.datasets, args.task, args.variant)
        
def incorrect_mapping_handler(args) -> None:
    """handler for create data with incorrect outputs. Each output maps to an incorrect option
    
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
    for dataset in args.datasets:
        curr_data_dir = os.path.join(args.data_dir, dataset)
        if os.path.exists(curr_data_dir):
            for seed in seeds:
                try:
                    np.random.seed(seed)
                    train_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_train.jsonl")
                    train_data = load_jsonl(train_data_path)
                    train_data = update_task(train_data, args.variant)
                    true_options: list = train_data[0]["options"]
                    mapping = {}
                    for option in true_options:
                        curr_options = copy.deepcopy(true_options)
                        curr_options.remove(option)
                        mapping[option] = np.random.choice(curr_options)
                    train_data = change_options_and_output(train_data, mapping)
                    
                    dev_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_dev.jsonl")
                    dev_data = load_jsonl(dev_data_path)
                    dev_data = update_task(dev_data, args.variant)
                    dev_data = change_options_and_output(dev_data, mapping)
                    
                    test_data_path = os.path.join(curr_data_dir, f"{dataset}_{args.k}_{seed}_test.jsonl")
                    test_data = load_jsonl(test_data_path)
                    test_data = update_task(test_data, args.variant)
                    test_data = change_options_and_output(test_data, mapping)
                    
                    new_train_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_train.jsonl")
                    new_dev_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_dev.jsonl")
                    new_test_path = os.path.join(args.data_dir, f"{dataset}_{args.variant}", f"{dataset}_{args.variant}_{args.k}_{seed}_test.jsonl")
                    
                    save_jsonl(train_data, new_train_path)
                    save_jsonl(dev_data, new_dev_path)
                    save_jsonl(test_data, new_test_path)
                    
                    config_file = os.path.join(args.config_dir, "tasks", dataset)
                    with open(config_file + ".json", "r") as f:
                        config = json.load(f)
                    save_json(config, os.path.join(args.config_dir, "tasks", f"{dataset}_{args.variant}.json"))
                    
                    print(f"Completed for seed {seed} of dataset {dataset}")
                except Exception as e:
                    print(f"Failed for seed {seed} of dataset {dataset}")
                    print(e)
        else:
            print(f"Data directory for {dataset} does not exist")
    if args.task != None:
        save_config(args.config_dir, args.datasets, args.task, args.variant)
        
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
                    train_data = update_task(train_data, variant)
                    train_data = option_fn(train_data, seed, **kwargs)
                    
                    dev_data_path = os.path.join(curr_data_dir, f"{dataset}_{k}_{seed}_dev.jsonl")
                    dev_data = load_jsonl(dev_data_path)
                    dev_data = update_task(dev_data, variant)
                    
                    test_data_path = os.path.join(curr_data_dir, f"{dataset}_{k}_{seed}_test.jsonl")
                    test_data = load_jsonl(test_data_path)
                    test_data = update_task(test_data, variant)
                    
                    new_train_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_train.jsonl")
                    new_dev_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_dev.jsonl")
                    new_test_path = os.path.join(data_dir, f"{dataset}_{variant}", f"{dataset}_{variant}_{k}_{seed}_test.jsonl")
                    
                    save_jsonl(train_data, new_train_path)
                    save_jsonl(dev_data, new_dev_path)
                    save_jsonl(test_data, new_test_path)
                    
                    try:
                        config_file = os.path.join(config_dir, "tasks", dataset)
                        with open(config_file + ".json", "r") as f:
                            config = json.load(f)
                        save_json(config, os.path.join(config_dir, "tasks", f"{dataset}_{variant}.json"))
                    except FileNotFoundError:
                        pass
                    
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