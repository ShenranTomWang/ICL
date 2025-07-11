import interpretability.operators as operators
import subprocess
import torch

def main(args):
    operator = args.operator_cls(args.model, args.device, args.dtype)
    all_layers = operator.ALL_LAYERS
    out_dir = f"logs/{args.model.split('/')[-1]}"
    for layer in all_layers:
        subprocess.run(
            [
                "python", "test.py",
                "--model", args.model, 
                "--out_dir", out_dir,
                "--operator", args.operator,
                "--task", args.task,
                "--k", "0",
                "--device", args.device,
                "--log_file", f"{out_dir}/16/{args.task}/layer_steer/{layer}/log_no_demo_mean.log",
                "steer_layer",
                "--layers", str(layer),
                "--mean_pool" if args.mean_pool else "",
            ]
        )
        if type(operator) == operators.HybridOperator:
            subprocess.run(
                [
                    "python", "test.py",
                    "--model", args.model, 
                    "--out_dir", out_dir,
                    "--operator", args.operator,
                    "--task", args.task,
                    "--k", "0",
                    "--device", args.device,
                    "--log_file", f"{out_dir}/16/{args.task}/layer_steer/{layer}/log_no_demo_mean_attn.log",
                    "steer_layer",
                    "--layers", str(layer),
                    "--mean_pool" if args.mean_pool else "",
                    "--stream", "attn"
                ]
            )
            subprocess.run(
                [
                    "python", "test.py",
                    "--model", args.model, 
                    "--out_dir", out_dir,
                    "--operator", args.operator,
                    "--task", args.task,
                    "--k", "0",
                    "--device", args.device,
                    "--log_file", f"{out_dir}/16/{args.task}/layer_steer/{layer}/log_no_demo_mean_scan.log",
                    "steer_layer",
                    "--layers", str(layer),
                    "--mean_pool" if args.mean_pool else "",
                    "--stream", "scan"
                ]
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run layer level steering ablation")
    parser.add_argument("--operator", type=str, required=True, help="Operator being used")
    parser.add_argument("--model", type=str, required=True, help="Model to use for the operator")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on, defaults to cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the model, defaults to bfloat16")
    parser.add_argument("--task", type=str, required=True, help="Task to run the operator on")
    parser.add_argument("--mean_pool", action="store_true", help="Whether to mean pool the attention values, defaults to False")
    
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    args.operator_cls = getattr(operators, args.operator)
    
    main(args)
    