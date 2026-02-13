export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator HymbaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Qwen2Operator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator LlamaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator MambaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator Mamba2Operator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE --fname trajectory  --out_fname trajectory_heatmap
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR --fname trajectory_heatmap

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator ZambaOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR

export MODEL="EleutherAI/pythia-1B"
export OUT_DIR="out/pythia-1B"
python function_vectors.py --model $MODEL --out_dir $OUT_DIR --operator GPTNeoXOperator --task function_vectors_original_random --device $DEVICE --fv_load_dir $OUT_DIR --seed 100 --use_template AIE
python ./visualize_fv.py --out_dir $OUT_DIR --seed 100 --task function_vectors_original_random --load_dir $OUT_DIR