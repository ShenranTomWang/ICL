import torch, logging
from utils.dataset import Dataset
from interpretability import Operator
import numpy as np
from collections import defaultdict

def count_option_changes(baseline: list[str], intervened: list[str], option1: str, option2: str) -> dict:
    """
    Count how many times option1 changed to option2 and vice versa.

    Parameters:
    - baseline (list[str]): Original outputs
    - intervened (list[str]): Outputs after intervention
    - option1 (str): First option to track
    - option2 (str): Second option to track

    Returns:
        dict: Counts of option1->option2 and option2->option1
    """

    if len(baseline) != len(intervened):
        raise ValueError("Baseline and intervened lists must be the same length.")

    option1_to_option2 = 0
    option2_to_option1 = 0

    for i in range(len(baseline)):
        base = baseline[i]
        inter = intervened[i]

        if base == option1 and inter == option2:
            option1_to_option2 += 1
        elif base == option2 and inter == option1:
            option2_to_option1 += 1

    result = {
        f"{option1}->{option2}": option1_to_option2,
        f"{option2}->{option1}": option2_to_option1
    }

    return result

def evaluate(predictions: list, groundtruths: list, is_classification: bool) -> float:
    """Evaluate the predictions against the groundtruths. Return accuracy for non-classification tasks, and macro-F1 for classification tasks.
    """
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        if prediction is None:
            continue
        prediction = prediction.strip()
        groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth) == list else groundtruth.strip()
        is_correct = prediction in groundtruth if type(groundtruth) == list else prediction == groundtruth
        accs.append(is_correct)
        if is_classification:
            recalls[groundtruth].append(is_correct)
            precisions[prediction].append(is_correct)

    if not is_classification:
        return np.mean(accs)

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision + recall == 0:
            f1s.append(0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))

    return np.mean(f1s)

@torch.inference_mode()
def do_inference(operator: Operator, dataset: Dataset, kwargs: dict, device: torch.DeviceObjType, verbose: bool = False) -> list:
    """Perform inference on dataset in batch with batch_size
    Args:
        operator (Operator)
        dataset (Dataset): dataset
        kwargs (dict): kwargs for model fwd pass
        device (torch.DeviceObjType): device
        verbose (bool): whether to print exception

    Returns:
        list<str>: predictions
    """
    logger = logging.getLogger(__name__)
    outputs = []
    inputs = dataset.inputs
    indices = dataset.indices
    option_ids = torch.tensor(dataset.option_ids, device=device, dtype=torch.long)
    options = dataset.options
    for i in range(0, len(inputs)):
        input = inputs[i]
        index = torch.tensor(indices[i]).to(device)
        try:
            output = operator(input, **kwargs)
            logit = output.logits
            output_logits = logit[..., index, :].squeeze(-2)        # (batch_size, vocab_size)
            output_logits = output_logits[..., option_ids]          # (batch_size, num_options)
            output = torch.argmax(output_logits, dim=-1)            # (batch_size)
            outputs.append(output.cpu())
        except Exception as e:
            if verbose:
                logger.exception(e)
            else:
                logger.error(e)
            output = torch.full((1,), -1, device="cpu", dtype=torch.long)
            outputs.append(output)
    
    outputs = torch.stack(outputs)
    outputs = outputs.cpu().tolist()
    outputs = [options[i] if i != -1 else None for i in outputs]
    return outputs