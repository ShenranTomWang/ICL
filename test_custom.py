import logging, argparse, os, json
from utils.utils import log_counters, init_counters
from utils.data import load_data
import numpy as np
import torch
from utils.dataset import Dataset
from interpretability import Operator
import interpretability
from utils.inference import do_inference, evaluate
# from english_words import english_words_set

# english_words_set = sorted(english_words_set)
            
def run(
    args, dataset, operator, seed, is_classification, kwargs
) -> float:
    """Run testing with dataset, return performance
    
    Args:
        args (dict): arguments
        dataset (Dataset): dataset
        operator (Operator): operator
        seed (str): seed
        is_classification (bool): whether the task is classification
        kwargs (dict): kwargs for model fwd pass
    
    Returns:
        (float): performance, macro-F1 for classification tasks, accuracy for non-classification tasks, none if args.is_null
    """
    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(
            args.out_dir,
            dataset.task,
            seed,
            "{}-{}-{}{}{}{}{}{}.pkl".format(
                dataset.task,
                split_name,
                args.method,
                "-k={}".format(args.k),
                "-s={}".format(seed) if args.use_random_english_words else "",
                "-n={}".format(args.n),
                "" if args.add_newlines else "-no-newlines",
                "-randomEnglish" if args.use_random_english_words else ""
            )
        )
    else:
        cache_path = os.path.join(
            args.out_dir,
            dataset.task,
            seed,
            "{}-{}-{}{}{}{}{}.pkl".format(
                dataset.task,
                args.split,
                args.method,
                "-k={}".format(args.k),
                "-s={}".format(seed) if args.use_random_english_words else "",
                "-n={}".format(args.n),
                "-randomEnglish" if args.use_random_english_words else ""
                )
            )
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if not os.path.exists(prediction_path):
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

    predictions = do_inference(operator, dataset, kwargs, args.device, args.verbose)

    groundtruths = dataset.outputs
    perf = evaluate(predictions, groundtruths, is_classification)
    logger.info(f"{'F1' if is_classification else 'Accuracy'} = {perf}")

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            if prediction is None:
                f.write("None")
            else:
                f.write(prediction)
                f.write("\n")

    return perf

def main(args):
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    logger.info("Model loaded")
    
    use_demonstrations = args.k != 0

    seeds = args.seed.split(",")
    errors, results = [], []
    for seed in seeds:
        train_data, test_data = load_data(args.task, args.dataset, args.split, args.k, args.n, seed, args.is_null)

        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            assert len(curr_test_data) > 0
            assert not use_demonstrations or len(curr_train_data) == args.k, (use_demonstrations, len(curr_train_data), args.k)
            
            dataset = Dataset([] if args.use_demo_cache else curr_train_data, curr_test_data, verbose=args.verbose)
            dataset.tensorize(operator.tokenizer)
            
            if args.use_demo_cache:
                if args.demo_cache_dir is not None:
                    demo_cache_dir = os.path.join(args.demo_cache_dir, test_task, seed)
                    cache = operator.load_cache(demo_cache_dir, "demo", 0)
                else:
                    dataset.prepare_demo()
                    cache = operator.extract_cache([dataset.demo])
                    if args.save_demo_cache:
                        operator.store_cache(cache, os.path.join(args.out_dir, test_task, seed, "demo"))
                    cache = cache[0]
                demo_length = dataset.demo_tokenized["input_ids"].shape[1]
                cache_kwargs = operator.cache2kwargs(cache, demo_length=demo_length, **args.cache2kwargs_kwargs)
            else:
                cache_kwargs = None

            # if args.use_random_english_words:
            #     # create a mapping
            #     options = curr_test_data[0]["options"]
            #     mapping = {option: np.random.choice(english_words_set) for option in options}
            #     new_options = list(mapping.values())
            #     for dp_idx, dp in enumerate(curr_train_data):
            #         assert dp["output"] in options, (dp, options)
            #         curr_train_data[dp_idx]["output"] = mapping[dp["output"]]
            #         curr_train_data[dp_idx]["options"] = new_options
            #     for dp_idx, dp in enumerate(curr_test_data):
            #         assert dp["output"] in options, (dp, options)
            #         curr_test_data[dp_idx]["output"] = mapping[dp["output"]]
            #         curr_test_data[dp_idx]["options"] = new_options
            result = run(args, dataset, operator, seed, True, cache_kwargs)

            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                results.append(result)

    if args.is_null:
        return

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100 * np.mean(results)))

    if len(errors) > 0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--add_newlines", default=False, action="store_true")
    parser.add_argument("--do_zeroshot", default=False, action="store_true")        # TODO: this is probably not properly implemented

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--n_skips", type=int, default=0, help="number of tokens to skip in the output, assumes having <bos> token, set to -1 if tokenizer has no <bos> token")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    
    parser.add_argument("--demo_cache_dir", default=None, help="out dir of demo cache previously extracted, should contain all children dirs of tasks. If not provided, will extract cache from scratch")
    parser.add_argument("--use_demo_cache", default=False, action="store_true", help="use cache for demo of each task")
    parser.add_argument("--save_demo_cache", default=False, action="store_true", help="save cache for demo of each task")
    parser.add_argument("--cache2kwargs_kwargs", default="{}", help="kwargs for operator.cache2kwargs")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--verbose", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--operator", type=str, required=True, choices=["Qwen2Operator", "HymbaOperator", "RWKVOperator", "MambaOperator", "Mamba2Operator", "ZambaOperator"])

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "out/" + "/".join(args.model.split("/")[-1:])
    
    assert args.dataset is not None or args.task is not None, "Either dataset or task must be provided"
        
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    args.operator = getattr(interpretability, args.operator)
    args.cache2kwargs_kwargs = json.loads(args.cache2kwargs_kwargs)

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