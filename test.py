import logging, argparse, os
from utils.utils import log_counters, init_counters
from utils.data import load_data
import numpy as np
import torch
from utils.dataset import Dataset
from interpretability.operators import Operator
import interpretability
from utils.inference import do_inference, evaluate
from constants import ALL_OPERATORS, ALL_DTYPES
from interpretability.attention_managers import AttentionManager
from interpretability.fv_maps import FVMap
import interpretability.attention_managers as attention_managers
from utils.test import exclusion_ablation_sanity_check, ablation_steer_sanity_check

def run(
    args, dataset, operator, seed, kwargs: dict = {}
) -> float:
    """Run testing with dataset, return performance
    
    Args:
        args (dict): arguments
        dataset (Dataset): dataset
        operator (Operator): operator
        seed (str): seed
        is_classification (bool): whether the task is classification
        kwargs (dict, optional): kwargs for model fwd pass, default is {}.
    
    Returns:
        (float): performance, macro-F1 for classification tasks, accuracy for non-classification tasks
    """
    prediction_path = os.path.join(
        args.out_dir,
        dataset.task,
        seed,
        "{}-{}-{}-{}.txt".format(
            dataset.task,
            args.split,
            "k={}".format(args.k),
            "n={}".format(args.n),
            )
        )
    logger.info(prediction_path)
    if not os.path.exists(prediction_path):
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    predictions = do_inference(operator, dataset, kwargs, args.device, args.verbose)

    groundtruths = dataset.outputs
    f1, acc = evaluate(predictions, groundtruths)
    logger.info(f"F1 = {f1}")
    logger.info(f"Acc = {acc}")

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            if prediction is None:
                f.write("None")
            else:
                f.write(prediction)
                f.write("\n")

    return f1, acc

def main(args):
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    logger.info("Model loaded")
    
    use_demonstrations = args.k != 0

    seeds = args.seed.split(",")
    errors, f1s, accs = [], [], []
    for seed in seeds:
        train_data, test_data = load_data(args.task, args.dataset, args.split, args.k, args.n, seed)

        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        if args.ablation_type == "mean_ablation" or args.ablation_type == "exclusion_mean_ablation":
            embeddings = []
            for test_task in test_counter:
                embedding = torch.load(f"{args.mean_load_dir}/{test_task}/{seed}/dev_attn_mean/attn_mean.pth")
                embedding = embedding.to(args.device)
                embeddings.append(embedding)
            mean_embedding = AttentionManager.mean_of(embeddings)
        
        if args.mean_pool:
            fv_maps = []
            for test_task in test_counter:
                fv_map = torch.load(f"{args.fv_map_load_dir}/{test_task}{args.target}/100/function_vectors.pth")
                fv_maps.append(fv_map)
            fv_map = FVMap.mean_of(fv_maps)
        
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            assert len(curr_test_data) > 0
            assert not use_demonstrations or len(curr_train_data) == args.k, (use_demonstrations, len(curr_train_data), args.k)
            
            dataset = Dataset(curr_train_data, curr_test_data, verbose=args.verbose, template=args.use_template)
            dataset.tensorize(operator.tokenizer)
            if args.p > 0:
                if args.ablation_type == "mean_ablation" or args.ablation_type == "zero_ablation":
                    if not args.mean_pool:
                        fv_map = torch.load(f"{args.fv_map_load_dir}/{test_task}{args.target}/100/function_vectors.pth")
                    top_p_heads = operator.top_p_heads(fv_map, args.p, stream=args.stream)
                    kwargs = operator.attention2kwargs(
                        None,
                        attention_intervention_fn=operator.get_fv_remove_head_attn_hook(),
                        scan_intervention_fn=operator.get_fv_remove_head_scan_hook(),
                        heads=top_p_heads,
                        ablation_type=args.ablation_type,
                        ablation_value=mean_embedding if args.ablation_type == "mean_ablation" else None,
                        ablate_token=-1
                    )
                elif args.ablation_type == "exclusion_mean_ablation" or args.ablation_type == "exclusion_zero_ablation":
                    ablation_type = "mean_ablation" if args.ablation_type == "exclusion_mean_ablation" else "zero_ablation"
                    if not args.mean_pool:
                        fv_map = torch.load(f"{args.fv_map_load_dir}/{test_task}{args.target}/100/function_vectors.pth")
                    heads_to_ablate = operator.exclusion_ablation_heads(fv_map=fv_map, top_p=args.exclude_p, ablation_p=args.p, stream=args.stream)
                    kwargs = operator.attention2kwargs(
                        None,
                        attention_intervention_fn=operator.get_fv_remove_head_attn_hook(),
                        scan_intervention_fn=operator.get_fv_remove_head_scan_hook(),
                        heads=heads_to_ablate,
                        ablation_type=ablation_type,
                        ablation_value=mean_embedding if ablation_type == "mean_ablation" else None,
                        ablate_token=-1
                    )
                    top_p_heads = operator.top_p_heads(fv_map, args.exclude_p, stream=args.stream)
                    exclusion_ablation_sanity_check(top_p_heads, heads_to_ablate, stream=args.stream)
                elif args.ablation_type == "steer":
                    if not args.mean_pool:
                        fv_map = torch.load(f"{args.fv_map_load_dir}/{test_task}{args.target}/100/function_vectors.pth")
                    fv_steer = operator.load_attention_manager(f"{args.fv_map_load_dir}/{test_task}/fv_steer.pth")
                    zeros = attention_managers.zeros_like(fv_steer)
                    top_p_heads = operator.top_p_heads(fv_map=fv_map, top_p=args.p, stream=args.stream)
                    fv_steer = zeros.set_head_values(head_values=fv_steer, head_indices=top_p_heads)
                    fv_steer = fv_steer * args.alpha
                    ablation_steer_sanity_check(fv_steer, top_p_heads)
                    kwargs = operator.attention2kwargs(fv_steer, last_k=1)
            else:
                kwargs = {}
            f1, acc = run(args, dataset, operator, seed, kwargs=kwargs)

            if f1 is None or acc is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                f1s.append(f1)
                accs.append(acc)

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(f1s) // len(seeds), 100 * np.mean(f1s)))
    logger.info("Accuracy of %s over %d target tasks: %.1f" % (args.task, len(accs) // len(seeds), 100 * np.mean(accs)))

    if len(errors) > 0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--use_template", default=False, action="store_true")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=ALL_DTYPES)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--operator", type=str, required=True, choices=ALL_OPERATORS)
    ablation_subparser = parser.add_subparsers(dest="ablation_type", required=False)
    
    zero_parser = ablation_subparser.add_parser("zero_ablation", help="Zero out heads. This will require have ran function_vectors.py first to obtain AIE maps.")
    zero_parser.add_argument("--p", type=float, default=0.0, help="Ablate top p heads")
    zero_parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map from this directory (only needed when p > 0), will use out_dir if not specified")
    zero_parser.add_argument("--stream", type=str, default=None, choices=["attn", "scan"], help="Stream to ablate, either attn or scan, defaults to None to ablate both streams")
    zero_parser.add_argument("--mean_pool", default=False, action="store_true", help="Whether to mean pool the fv_map, defaults to False")

    mean_parser = ablation_subparser.add_parser("mean_ablation", help="Mean out heads. This will require have ran function_vectors.py first to obtain AIE maps.")
    mean_parser.add_argument("--p", type=float, default=0.0, help="Ablate top p heads")
    mean_parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map from this directory (only needed when p > 0), will use out_dir if not specified")
    mean_parser.add_argument("--mean_load_dir", type=str, default=None, help="Load mean from this directory (only needed when p > 0), will use out_dir if not specified")
    mean_parser.add_argument("--stream", type=str, default=None, choices=["attn", "scan"], help="Stream to ablate, either attn or scan, defaults to None to ablate both streams")
    mean_parser.add_argument("--mean_pool", default=False, action="store_true", help="Whether to mean pool the fv_map, defaults to False")
    
    exclusion_zero_parser = ablation_subparser.add_parser("exclusion_zero_ablation", help="Randomly ablate heads that are not function heads to zero. This will require have ran function_vectors.py first to obtain AIE maps.")
    exclusion_zero_parser.add_argument("--p", type=float, default=0.0, help="Percentage of heads to ablate, defaults to 0.0")
    exclusion_zero_parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map (AIE score map) from this directory (only needed when p > 0), will use out_dir if not specified")
    exclusion_zero_parser.add_argument("--stream", type=str, default=None, choices=["attn", "scan"], help="Stream to ablate, either attn or scan, defaults to None to ablate both streams")
    exclusion_zero_parser.add_argument("--exclude_p", type=float, default=0.0, help="Percentage of function heads to exclude from ablation, defaults to 0.0")
    exclusion_zero_parser.add_argument("--mean_pool", default=False, action="store_true", help="Whether to mean pool the fv_map, defaults to False")
    
    exclusion_mean_parser = ablation_subparser.add_parser("exclusion_mean_ablation", help="Randomly ablate heads that are not function heads to the mean. This will require have ran function_vectors.py first to obtain AIE maps.")
    exclusion_mean_parser.add_argument("--p", type=float, default=0.0, help="Percentage of heads to ablate, defaults to 0.0")
    exclusion_mean_parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map (AIE score map) from this directory (only needed when p > 0), will use out_dir if not specified")
    exclusion_mean_parser.add_argument("--stream", type=str, default=None, choices=["attn", "scan"], help="Stream to ablate, either attn or scan, defaults to None to ablate both streams")
    exclusion_mean_parser.add_argument("--exclude_p", type=float, default=0.0, help="Percentage of function heads to exclude from ablation, defaults to 0.0")
    exclusion_mean_parser.add_argument("--mean_load_dir", type=str, default=None, help="Load mean from this directory (only needed when p > 0), will use out_dir if not specified")
    exclusion_mean_parser.add_argument("--mean_pool", default=False, action="store_true", help="Whether to mean pool the fv_map, defaults to False")
    
    steering_parser = ablation_subparser.add_parser("steer", help="Steer selected function heads. This will require have ran extract_activations.py with fv_steer first.")
    steering_parser.add_argument("--p", type=float, default=0.0, help="Percentage of heads to steer, defaults to 0.0")
    steering_parser.add_argument("--fv_map_load_dir", type=str, default=None, help="Load fv_map from this directory (only needed when p > 0), will use out_dir if not specified")
    steering_parser.add_argument("--stream", type=str, default=None, choices=["attn", "scan"], help="Stream to steer, either attn or scan, defaults to None to steer both streams")
    steering_parser.add_argument("--target", type=str, default="random", choices=["incorrect_mapping", "random", "0_correct", "25_correct", "50_correct", "75_correct"], help="Target task to steer towards, defaults to random for steering with base FVs")
    steering_parser.add_argument("--alpha", type=float, default=1.0, help="Steering strength, defaults to 1.0")
    steering_parser.add_argument("--mean_pool", default=False, action="store_true", help="Whether to mean pool the fv_steer, defaults to False")
    
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "out/" + "/".join(args.model.split("/")[-1:])
    if hasattr(args, "fv_map_load_dir") and args.fv_map_load_dir is None:
        args.fv_map_load_dir = args.out_dir
    if hasattr(args, "mean_load_dir") and args.mean_load_dir is None:
        args.mean_load_dir = args.out_dir
    if not hasattr(args, "p"):
        args.p = 0.0

    args.target = ("_" + args.target) if hasattr(args, "target") and args.target is not None else "_random"
    assert (args.dataset is not None) ^ (args.task is not None), "Either dataset or task must be provided, but not both"

    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.operator = getattr(interpretability.operators, args.operator)

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(args)