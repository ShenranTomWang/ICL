import argparse, os
import torch
from utils.plotting import draw_attention
import matplotlib.pyplot as plt
from utils.data import load_data
from utils.utils import init_counters, log_counters
from utils.dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    train_data, test_data = load_data(None, args.dataset, args.split, -1, 1, args.seed)
    train_counter, test_counter = init_counters(train_data, test_data)
    log_counters(train_counter, test_counter)
    dataset = Dataset(train_data, test_data, template=True)
    dataset.choose(args.k, args.seed)
    dataset.preprocess()
    
    tokenized = tokenizer(dataset.inputs, return_tensors="pt", truncation=True).to(args.device)
    all_attns = model(**tokenized, output_attentions=True, return_dict=True).attentions
    selected_attn = all_attns[args.layer][0, args.head].cpu().detach()
    tokenized = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
    tokenized = [t.replace("Ġ", "") for t in tokenized]
    tokenized = [t.replace("Ċ", "\n") for t in tokenized]
    
    _, axis = plt.subplots(1, 1, figsize=(10, 8))
    draw_attention(selected_attn, title=f"Layer {args.layer} Head {args.head} Attention Map", axis=axis, tokens_q=tokenized, tokens_k=tokenized)
    plt.savefig(args.out_dir)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attn map for a specific input (in a dataset)")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev"], help="Dataset split to use")
    parser.add_argument("--k", type=int, default=10, help="Number of demos to use")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Data type to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--layer", type=int, required=True, help="Layer of head to visualize")
    parser.add_argument("--head", type=int, required=True, help="Index of Head to visualize")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to visualize")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory to save visualization, defaults to ./out/{model}/attn_map.pdf")
    
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.device = torch.device(args.device)
    if args.out_dir is None:
        args.out_dir = os.path.join("out", args.model.split("/")[-1], "attn_map.pdf")
    os.makedirs(os.path.dirname(args.out_dir), exist_ok=True)
    main(args)