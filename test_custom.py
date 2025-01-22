import logging, argparse, os, json
from transformers import AutoTokenizer
from metaicl.model import MetaICLModel
from metaicl.data import MetaICLData
from utils.data import load_data_by_datasets, load_data_by_task
import pickle as pkl
import numpy as np
from collections import Counter

def main(logger, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MetaICLModel(logger, args.out_dir)
    model.load(model_name=args.model)
    model.to_device()
    logger.info("Model loaded")
    
    # TODO: the following two lines may need to be changed on demand
    add_newlines = True
    checkpoint = None
    
    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        max_length = min(max_length * args.k, 1024)
    logger.info(
        "batch_size={}\tmax_length={}\tmax_length_per_example={}".format(
            args.test_batch_size, max_length, max_length_per_example
        )
    )
    metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations, args.k, max_length, max_length_per_example)
    
    seeds = args.seed.split(",")
    errors, results = [], []
    for seed in seeds:
        if args.task != None:
            config_split = "unseen_domain_test" if args.unseen_domain_only else "test"
            train_data = load_data_by_task(args.task, "train", args.k, seed=seed, config_split=config_split)
            test_data = load_data_by_task(args.task, args.split, args.k, seed=seed, config_split=config_split, is_null=args.is_null)
        else:
            assert args.dataset is not None
            train_data = load_data_by_datasets(args.dataset.split(","), args.k, "train", seed=seed)
            test_data = load_data_by_datasets(args.dataset.split(","), args.k, args.split, seed=seed, is_null=args.is_null)
        logger.info("Loaded data for seed %s" % seed)

        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(logger, train_counter, test_counter)
        
        for test_task in test_counter:
            curr_test_data = [dp for dp in test_data if dp["task"]==test_task]
            curr_train_data = [dp for dp in train_data if dp["task"]==test_task]
            assert len(curr_test_data) > 0
            assert not args.use_demonstrations or len(curr_train_data) == args.k, (args.use_demonstrations, len(curr_train_data), args.k)

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

            result = run(logger, test_task, metaicl_data, model,
                         curr_train_data, curr_test_data, seed, checkpoint, is_classification, add_newlines)

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

            
def run(logger, task, metaicl_data, metaicl_model, train_data, dev_data, seed,
        checkpoint, is_classification, add_newlines):
    """run testing with dev_data
    """

    if args.do_zeroshot:
        split_name = args.split
        if args.is_null:
            split_name += "-null"
        cache_path = os.path.join(
            args.out_dir,
            "{}-{}-{}{}{}{}{}.pkl".format(
                task,
                split_name,
                metaicl_data.method,
                "-k={}".format(args.k) if args.use_demonstrations else "",
                "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                "" if add_newlines else "-no-newlines",
                "-randomEnglish" if args.use_random_english_words else ""
            )
        )
    else:
        assert add_newlines
        cache_path = os.path.join(
            args.out_dir,
            "{}-{}-{}{}{}{}.pkl".format(
                task,
                args.split,
                metaicl_data.method,
                "-k={}".format(args.k) if args.use_demonstrations else "",
                "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                "-randomEnglish" if args.use_random_english_words else ""
                )
            )

    metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")
    if args.use_calibration:
        prediction_path = prediction_path.replace(".txt", "-calibrated.txt")

    if os.path.exists(prediction_path):
        return 0

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        if metaicl_model.is_none():
            metaicl_model.load(checkpoint, model_name=args.model)
            metaicl_model.cuda()
            metaicl_model.eval()

        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)

    assert len(losses)==len(metaicl_data)

    if args.is_null:
        return None

    if args.use_calibration:
        assert args.do_zeroshot
        bias_path = cache_path.replace("/"+task+"-"+args.split, "/"+task+"-"+args.split+"-null")
        assert os.path.exists(bias_path), bias_path
        with open(bias_path, "rb") as f:
            bias_losses = pkl.load(f)

        losses = np.array(losses)
        bias_losses = np.array(bias_losses)
        assert losses.shape == bias_losses.shape
        losses -= bias_losses

    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in dev_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_zeroshot", default=False, action="store_true")
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--use_calibration", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")

    parser.add_argument("--log_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

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