import logging, argparse, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData
from utils.data import load_data_by_datasets, load_data_by_task
import pickle as pkl
import numpy as np
import torch
from utils.dataset import Dataset
from collections import Counter, defaultdict

torch.float

def evaluate(predictions: list, groundtruths: list, is_classification: bool) -> float:
    """Evaluate the predictions against the groundtruths. Return accuracy for non-classification tasks, and macro-F1 for classification tasks.
    """
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = prediction.strip()
        groundtruth = [gt.strip() for gt in groundtruth] if type(groundtruth)==list else groundtruth.strip()
        is_correct = prediction in groundtruth if type(groundtruth)==list else prediction==groundtruth
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
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return np.mean(f1s)

def load_model(args: dict) -> any:
    """load model from args
    """
    if args.meta_icl:
        model = MetaICLModel(logger, args.out_dir, fp16=args.dtype == "float16")
        model.load(model_name=args.model)
        model.to_device()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    return model

def load_data(args, seed) -> tuple:
    """Load train and test data from args using seed

    Returns:
        tuple<list<{task: str, input: str, output: str, options: list<str>}>>: train_data, test_data
    """
    if args.task != None:
        config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
        train_data = load_data_by_task(args.task, "train", args.k, seed=seed, config_split=config_split)
        test_data = load_data_by_task(args.task, args.split, args.n, seed=seed, config_split=config_split, is_null=args.is_null)
    else:
        assert args.dataset is not None
        train_data = load_data_by_datasets(args.dataset.split(","), args.k, "train", seed=seed)
        test_data = load_data_by_datasets(args.dataset.split(","), args.n, args.split, seed=seed, is_null=args.is_null)
    logger.info("Loaded data for seed %s" % seed)
    return train_data, test_data

def do_inference_meta_icl(
    model: MetaICLModel, dataset: MetaICLData, cache_path: str, checkpoint: str, test_data: list
) -> list:
    """Do inference with MetaICL model

    Returns:
        list<str>: predictions
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        if model.is_none():
            model.load(checkpoint, model_name=args.model)
            model.cuda()
            model.eval()

        losses = model.do_inference(dataset, args.test_batch_size)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)
    if args.is_null:
        return None

    # if args.use_calibration:
    #     assert args.do_zeroshot
    #     bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
    #     assert os.path.exists(bias_path), bias_path
    #     with open(bias_path, "rb") as f:
    #         bias_losses = pkl.load(f)

    #     losses = np.array(losses)
    #     bias_losses = np.array(bias_losses)
    #     assert losses.shape == bias_losses.shape
    #     losses -= bias_losses

    predictions = model.do_predict(test_data, losses=losses)
    return predictions

def do_inference_hf(model: AutoModelForCausalLM, dataset: Dataset, batch_size: int) -> list:
    """Perform inference on dataset in batch with batch_size

    Returns:
        list<str>: predictions
    """
    outputs = []
    inputs = dataset.inputs
    indices = dataset.indices
    option_ids = dataset.option_ids
    options = torch.tensor(dataset.options)
    for i in range(0, len(inputs), min(batch_size, len(inputs) - i)):
        batch = inputs[i:i+batch_size]
        index = indices[i:i+batch_size]
        output = model(batch)
        logit = output.logits
        output_logits = logit.gather(-2, index).squeeze(-2)        # (batch_size, vocab_size)
        output_logits = output_logits.gather(-1, option_ids)        # (batch_size, num_options)
        output = torch.argmax(output_logits, dim=-1).squeeze(-1)    # (batch_size)
        outputs.append(output)
    
    outputs = torch.cat(outputs, dim=0)
    outputs = options[outputs]
    return outputs.tolist()

def main(logger, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args)
    logger.info("Model loaded")
    
    # TODO: the following two lines may need to be changed on demand
    add_newlines = True
    checkpoint = None
    use_demonstrations = args.k != 0
    
    max_length_per_example = 256
    max_length = 256
    if use_demonstrations:
        max_length = min(max_length * args.k, 1024)
    logger.info(
        "batch_size={}\tmax_length={}\tmax_length_per_example={}".format(
            args.test_batch_size, max_length, max_length_per_example
        )
    )

    seeds = args.seed.split(",")
    errors, results = [], []
    for seed in seeds:
        train_data, test_data = load_data(args, seed)

        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(logger, train_counter, test_counter)
        
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            assert len(curr_test_data) > 0
            assert not use_demonstrations or len(curr_train_data) == args.k, (use_demonstrations, len(curr_train_data), args.k)

            config_file = "config/tasks/{}.json".format(test_task)
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"]=="classification"

            # TODO: this should be done in create_data_custom.py
            # if args.use_random_english_words:
            #     # create a mapping
            #     options = curr_dev_data[0]["options"]
            #     mapping = {option: np.random.choice(english_words_set) for option in options}
            #     new_options = list(mapping.values())
            #     for dp_idx, dp in enumerate(curr_train_data):
            #         assert dp["output"] in options, (dp, options)
            #         curr_train_data[dp_idx]["output"] = mapping[dp["output"]]
            #         curr_train_data[dp_idx]["options"] = new_options
            #     for dp_idx, dp in enumerate(curr_dev_data):
            #         assert dp["output"] in options, (dp, options)
            #         curr_dev_data[dp_idx]["output"] = mapping[dp["output"]]
            #         curr_dev_data[dp_idx]["options"] = new_options

            if args.meta_icl:
                tokenized_dataset = MetaICLData(logger, tokenizer, args.method, use_demonstrations, args.k, max_length, max_length_per_example)
                tokenized_dataset.tensorize(train_data, test_data, add_newlines=add_newlines)
                tokenized_dataset.print_tensorized_example()
            else:
                tokenized_dataset = Dataset(curr_train_data, curr_test_data, add_newlines=add_newlines)
                tokenized_dataset.preprocess()
                tokenized_dataset.tensorize(tokenizer)
            result = run(
                args, logger, test_task, tokenized_dataset, model, curr_test_data, seed, checkpoint, is_classification, add_newlines
            )

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


def log_counters(logger, train_counter, test_counter):
    for k, v in train_counter.items():
        logger.info("[Train] %s\t%d" % (k, v))
    for k, v in test_counter.items():
        logger.info("[Test] %s\t%d" % (k, v))

    logger.info("method %s on %s (%d train, %d test)" % (args.method, args.dataset, len(train_counter), len(test_counter)))


def init_counters(train_data, test_data):
    train_counter = Counter()
    test_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    for dp in test_data:
        test_counter[dp["task"]] += 1
    return train_counter, test_counter

            
def run(
    args, logger, task, dataset, model, test_data, seed, checkpoint, is_classification, add_newlines
):
    """run testing with test_data
    """

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(
            args.out_dir,
            "{}-{}-{}{}{}{}{}{}.pkl".format(
                task,
                split_name,
                args.method,
                "-k={}".format(args.k),
                "-s={}".format(seed) if args.use_random_english_words else "",
                "-n={}".format(args.n),
                "" if add_newlines else "-no-newlines",
                "-randomEnglish" if args.use_random_english_words else ""
            )
        )
    else:
        assert add_newlines
        cache_path = os.path.join(
            args.out_dir,
            "{}-{}-{}{}{}{}{}.pkl".format(
                task,
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
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

    if args.meta_icl:
        predictions = do_inference_meta_icl(args, model, dataset, cache_path, checkpoint, test_data, is_classification)
    else:
        predictions = do_inference_hf(model, dataset, args.test_batch_size)

    groundtruths = [dp["output"] for dp in test_data]
    perf = evaluate(predictions, groundtruths, is_classification)
    logger.info(f"{"F1" if is_classification else "Accuracy"} = {perf}")

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    # parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--meta_icl", default=False, action="store_true")
    parser.add_argument("--test_batch_size", type=int, default=4)
    # parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")       # TODO: this should be done in create_data_custom.py

    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--is_null", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--model", type=str, default="/scratch/st-jzhu71-1/shenranw/models/openai-community/gpt2")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "out/" + "/".join(args.model.split("/")[-1:])

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)