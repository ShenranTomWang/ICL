from collections import Counter
from utils.inference import do_inference, evaluate
from utils.dataset import Dataset
from interpretability import Operator
import torch

def baseline_handler(operator: Operator, dataset: Dataset, device: torch.DeviceObjType, verbose: bool) -> dict:
    """
    handler for baseline steering

    Args:
        operator (Operator)
        dataset (Dataset)
        device (torch.DeviceObjType)
        verbose (bool)
    Returns:
        dict: results, containing the count of each option, f1, and outputs for raw outputs
    """
    results_baseline = do_inference(operator, dataset, 1, {}, device, verbose)
    f1 = evaluate(results_baseline, dataset.outputs, is_classification=True)
    results_task = Counter(results_baseline)
    results_task = {k: v / len(results_baseline) for k, v in results_task.items()}
    results_task["f1"] = f1
    results_task["outputs"] = results_baseline
    return results_task

def intervene_direct_handler(
    operator: Operator,
    dataset: Dataset,
    device: torch.DeviceObjType,
    verbose: bool,
    strength: float,
    layers: list,
    keep_scan: bool,
    keep_attention: bool,
    test_task: str,
    seed: int,
    load_dir: str,
    k: int,
) -> tuple[dict, dict]:
    """
    handler for direct intervention steering

    Args:
        operator (Operator)
        dataset (Dataset)
        device (torch.DeviceObjType)
        verbose (bool)
        strength (float)
        layers (list)
        keep_scan (bool)
        keep_attention (bool)
        test_task (str)
        seed (int)
        load_dir (str)
        k (int)
    Returns:
        tuple[dict, dict]: results_task_steer0, results_task_steer1, each containing the count of each option, and outputs for raw outputs
    """
    steer0 = operator.load_attention_output(f"{load_dir}/{test_task}/{seed}/train_steer_k={k}/steer_{dataset.options[0]}.pth")
    steer1 = operator.load_attention_output(f"{load_dir}/{test_task}/{seed}/train_steer_k={k}/steer_{dataset.options[1]}.pth")
    steer0 = strength * steer0
    steer1 = strength * steer1
    steer0 = operator.attention2kwargs(steer0, layers=layers, keep_attention=keep_attention, keep_scan=keep_scan)
    steer1 = operator.attention2kwargs(steer1, layers=layers, keep_attention=keep_attention, keep_scan=keep_scan)
    
    steer0_results = do_inference(operator, dataset, 1, steer0, device, verbose)
    steer1_results = do_inference(operator, dataset, 1, steer1, device, verbose)
    
    results_task_steer0 = Counter(steer0_results)
    results_task_steer0 = {k: v / len(steer0_results) for k, v in results_task_steer0.items()}
    results_task_steer0["outputs"] = steer0_results
    results_task_steer1 = Counter(steer1_results)
    results_task_steer1 = {k: v / len(steer1_results) for k, v in results_task_steer1.items()}
    results_task_steer1["outputs"] = steer1_results
    
    return results_task_steer0, results_task_steer1

def intervene_diff_handler(
    operator: Operator,
    dataset: Dataset,
    strength: float,
    layers: list,
    keep_scan: bool,
    keep_attention: bool,
    test_task: str,
    seed: int,
    load_dir: str,
    k: int,
    device: torch.DeviceObjType,
    verbose: bool
) -> tuple[dict, dict]:
    """
    handler for diff intervention steering

    Args:
        operator (Operator)
        dataset (Dataset)
        strength (float)
        layers (list)
        keep_scan (bool)
        keep_attention (bool)
        test_task (str)
        seed (int)
        load_dir (str)
        k (int)
        device (torch.DeviceObjType)
        verbose (bool)

    Returns:
        tuple[dict, dict]: results_task_positive, results_task_negative, each containing the count of each option, and outputs for raw outputs
    """
    steer_positive = operator.load_attention_output(f"{load_dir}/{test_task}/{seed}/train_steer_k={k}/steer_{dataset.options[0]}->{dataset.options[1]}.pth")
    steer_positive = strength * steer_positive
    steer_negative = -1 * steer_positive
    
    steer_positive = operator.attention2kwargs(steer_positive, layers=layers, keep_scan=keep_scan, keep_attention=keep_attention)
    steer_negative = operator.attention2kwargs(steer_negative, layers=layers)
    
    steer_results_positive = do_inference(operator, dataset, 1, steer_positive, device, verbose)
    steer_results_negative = do_inference(operator, dataset, 1, steer_negative, device, verbose)
    
    results_task_positive = Counter(steer_results_positive)
    results_task_negative = Counter(steer_results_negative)
    results_task_positive = {k: v / len(steer_results_positive) for k, v in results_task_positive.items()}
    results_task_negative = {k: v / len(steer_results_negative) for k, v in results_task_negative.items()}
    results_task_positive["outputs"] = steer_results_positive
    results_task_negative["outputs"] = steer_results_negative
    return results_task_positive, results_task_negative