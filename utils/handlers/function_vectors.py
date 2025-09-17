from interpretability.operators import Operator
from utils.data import load_data
from utils.dataset import Dataset
from utils.utils import init_counters, log_counters
import torch
import os, logging
from utils.data import to_base
from interpretability import AttentionManager

def AIE_handler(args):
    logger = logging.getLogger(__name__)
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    for seed in args.seed:
        train_data, test_data = load_data(args.task, None, args.split, -1, args.n, seed)
        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        for test_task in test_counter:
            logging.info(f"Processing task {test_task} with seed {seed}")
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            dataset = Dataset(curr_train_data, curr_test_data, template=args.use_template)
            dataset.choose(args.k, seed)
            dataset.preprocess()
            dataset.tensorize(operator.tokenizer)
            test_task_base = to_base(test_task)
            steer = operator.load_attention_manager(f"{args.fv_load_dir}/{test_task_base}/{seed}/fv_steer.pth")
            inputs = dataset.inputs
            label_id = torch.tensor(dataset.output_ids)
            fv_map = operator.generate_AIE_map([steer], [inputs], [label_id])
            out_dir = f"{args.out_dir}/{test_task}/{seed}"
            os.makedirs(out_dir, exist_ok=True)
            torch.save(fv_map, f"{out_dir}/function_vectors.pth")
            fv_map.visualize(f"{out_dir}/function_vectors.png")
            logger.info(f"Function vectors saved to {out_dir}/{args.task}_function_vectors.pth")
            logger.info(f"Function vectors visualization saved to {out_dir}/{args.task}_function_vectors.png")

def neg_AIE_handler(args):
    logger = logging.getLogger(__name__)
    operator: Operator = args.operator(args.model, args.device, args.dtype)
    for seed in args.seed:
        train_data, test_data = load_data(args.task, None, args.split, -1, args.n, seed)
        train_counter, test_counter = init_counters(train_data, test_data)
        log_counters(train_counter, test_counter)
        
        for test_task in test_counter:
            logging.info(f"Processing task {test_task} with seed {seed}")
            curr_test_data = [dp for dp in test_data if dp["task"] == test_task]
            curr_train_data = [dp for dp in train_data if dp["task"] == test_task]
            dataset = Dataset(curr_train_data, curr_test_data, template=args.use_template)
            dataset.choose(args.k, seed)
            dataset.preprocess()
            dataset.tensorize(operator.tokenizer)
            test_task_base = to_base(test_task)
            steer = []
            for seed in args.seed:
                am = operator.load_attention_manager(f"{args.fv_load_dir}/{test_task_base}/{seed}/{args.split}_attn_mean/attn_mean.pth")
                am = am.to(args.device)
                steer.append(am)
            steer = AttentionManager.mean_of(steer)
            inputs = dataset.inputs
            label_id = torch.tensor(dataset.output_ids)
            fv_map = operator.generate_AIE_map([steer], [inputs], [label_id])
            out_dir = f"{args.out_dir}/{test_task}/{seed}"
            os.makedirs(out_dir, exist_ok=True)
            torch.save(fv_map, f"{out_dir}/neg_function_vectors.pth")
            fv_map.visualize(f"{out_dir}/neg_function_vectors.png")
            logger.info(f"Negative function vectors saved to {out_dir}/{args.task}_neg_function_vectors.pth")
            logger.info(f"Negative function vectors visualization saved to {out_dir}/{args.task}_neg_function_vectors.png")