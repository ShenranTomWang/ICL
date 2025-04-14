from interpretability.operators import Operator
from utils.data import load_data
from utils.dataset import Dataset
from utils.utils import init_counters, log_counters
import torch
from interpretability.fv_maps import FVMap

def to_base(dataset: str) -> str:
    if dataset.endswith("_random"):
        return dataset[:-7]
    elif dataset.endswith("_0_correct"):
        return dataset[:-11]
    elif dataset.endswith("_25_correct") or dataset.endswith("_50_correct") or dataset.endswith("_75_correct"):
        return dataset[:-13]
    else:
        return dataset

def function_vectors_handler(args):
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    all_fv_maps = []
    for seed in args.seed:
        train_data, test_data = load_data(args.task, None, "dev", -1, -1, seed)
        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        datasets, steers = [], []
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            dataset = Dataset(curr_train_data, curr_test_data)
            dataset.choose(args.k, seed)
            dataset.preprocess()
            dataset.tensorize(operator.tokenizer)
            datasets.append(dataset)
            test_task_base = to_base(test_task)
            steer = operator.load_attention_manager(f"{args.out_dir}/{test_task_base}/{seed}/fv_steer.pth")
            steers.append(steer)
        inputs = [dataset.inputs for dataset in datasets]
        label_ids = [torch.tensor(dataset.output_ids) for dataset in datasets]
        fv_map = operator.generate_AIE_map(steers, inputs, label_ids)
        all_fv_maps.append(fv_map)
        fv_map.visualize(f"{args.output_dir}/{seed}/function_vectors.png")
    mean_fv_map = FVMap.mean_of(all_fv_maps)
    mean_fv_map.visualize(f"{args.output_dir}/mean_function_vectors.png")