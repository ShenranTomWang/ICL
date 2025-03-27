import torch
from utils.inference import count_option_changes
from utils.dataset import do_inference
from utils.data import load_data
from utils.utils import init_counters, log_counters
from interpretability import Operator
import argparse, os, logging, json
import interpretability
from collections import Counter
from utils.handlers.steer import baseline_handler, intervene_direct_handler, intervene_diff_handler

def main(args):
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    logger.info("Model loaded")

    seeds = args.seed.split(",")
    results = {}
    layers = "baseline"
    if args.mode != "baseline":
        layers = ",".join(args.layers) if args.layers is not None else "all"
    for seed in seeds:
        train_data, test_data = load_data(args.task, args.dataset, args.split, args.k, -1, seed)
        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        results_seed = {}
        for test_task in test_counter:
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            curr_train_output = [dp["output"] for dp in curr_train_data]
            label_counter = Counter(curr_train_output)
            logger.info(f"Label distribution for {test_task} seed {seed}: {label_counter}")
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            assert len(curr_test_data) > 0
            dataset = do_inference(curr_train_data, curr_test_data)
            dataset.tensorize(operator.tokenizer)
            assert len(dataset.options_raw) == 2, f"Expected 2 options, got {len(dataset.options_raw)}"
            
            resuls_task_baseline = baseline_handler(operator, dataset, args.device, args.verbose)
            results_seed[test_task] = {"baseline": resuls_task_baseline}
            logger.info(resuls_task_baseline)
            with open(f"{args.out_dir}/{test_task}/{seed}/steer_results_baseline.json", "w") as f:
                json.dump(resuls_task_baseline, f, indent=4)
            if args.mode == "intervene_direct":
                results_task_steer0, results_task_steer1 = intervene_direct_handler(
                    operator=operator, dataset=dataset, device=args.device, verbose=args.verbose, strength=args.strength, layers=args.layers,
                    keep_scan=args.keep_scan, keep_attention=args.keep_attention, test_task=test_task, seed=seed, load_dir=args.load_dir, k=args.k
                )
                change0 = count_option_changes(resuls_task_baseline["outputs"], results_task_steer0["outputs"], dataset.options[0], dataset.options[1])
                change1 = count_option_changes(resuls_task_baseline["outputs"], results_task_steer1["outputs"], dataset.options[1], dataset.options[0])
                results_task_steer0.update({"change": change0})
                results_task_steer1.update({"change": change1})
                results_seed[test_task].update({dataset.options_raw[0]: results_task_steer0, dataset.options_raw[1]: results_task_steer1})
                logger.info(results_seed[test_task])
                
                with open(f"{args.out_dir}/{test_task}/{seed}/steer_results_{dataset.options_raw[0]}_{layers}.json", "w") as f:
                    json.dump(results_task_steer0, f, indent=4)
                with open(f"{args.out_dir}/{test_task}/{seed}/steer_results_{dataset.options_raw[1]}_{layers}.json", "w") as f:
                    json.dump(results_task_steer1, f, indent=4)
            elif args.mode == "intervene_diff":
                results_task_positive, results_task_negative = intervene_diff_handler(
                    operator=operator, dataset=dataset, device=args.device, verbose=args.verbose, strength=args.strength, layers=args.layers,
                    keep_scan=args.keep_scan, keep_attention=args.keep_attention, test_task=test_task, seed=seed, load_dir=args.load_dir, k=args.k
                )
                change_positive = count_option_changes(resuls_task_baseline["outputs"], results_task_positive["outputs"], dataset.options[0], dataset.options[1])
                change_negative = count_option_changes(resuls_task_baseline["outputs"], results_task_negative["outputs"], dataset.options[1], dataset.options[0])
                results_task_positive.update({"change": change_positive})
                results_task_negative.update({"change": change_negative})
                results_seed[test_task].update({f"{dataset.options_raw[0]}->{dataset.options_raw[1]}": results_task_positive, f"{dataset.options_raw[1]}->{dataset.options_raw[0]}": results_task_negative})
                logger.info(results_seed[test_task])
                with open(f"{args.out_dir}/{test_task}/{seed}/steer_results_diff_{layers}.json", "w") as f:
                    json.dump(results_seed[test_task], f, indent=4)
            else:
                raise ValueError(f"Unknown mode {args.mode}")
            
            with open("{}/{}/results{}.json".format(args.out_dir, test_task, f"_{layers}"), "w") as f:
                json.dump(results_seed, f, indent=4)
        results[seed] = results_seed
    logger.info(results)

def add_baseline_args(baseline_parser: argparse.ArgumentParser) -> None:
    pass

def add_intervene_args(intervene_parser: argparse.ArgumentParser) -> None:
    intervene_parser.add_argument("--layers", default=None, type=str, help="comma separated list of layers, defaults to all layers")
    intervene_parser.add_argument("--strength", default=1.0, type=float, help="strength of intervention")
    intervene_parser.add_argument("--keep_scan", default=False, action="store_true", help="keep ssm stream")
    intervene_parser.add_argument("--keep_attention", default=False, action="store_true", help="keep attention stream")
        
def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    mode_subparser = parser.add_subparsers(dest="mode", required=True)
    parser_baseline = mode_subparser.add_parser("baseline")
    add_baseline_args(parser_baseline)
    parser_intervene = mode_subparser.add_parser("intervene_direct")
    add_intervene_args(parser_intervene)
    parser_intervene = mode_subparser.add_parser("intervene_diff")
    add_intervene_args(parser_intervene)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--task", default=None, type=str)
    parser.add_argument("--seed", default="13,21,42,87,100", type=str, help="comma separated list of seeds")
    parser.add_argument("--split", default="test", type=str, choices=["test", "dev"])
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--load_dir", required=True, type=str)
    parser.add_argument("--operator", required=True, type=str, choices=["TransformerOperator", "HymbaOperator", "RWKVOperator", "MambaOperator", "ZambaOperator"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--n_skips", default=0, type=int, help="number of tokens to skip in the output, assumes having <bos> token, set to -1 if tokenizer has no <bos> token")
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--k", default=16, type=int)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    
    args = parser.parse_args()
    assert args.dataset is not None or args.task is not None, "Either dataset or task must be provided"

    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.operator = getattr(interpretability, args.operator)
    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(args)
    
    main(args)