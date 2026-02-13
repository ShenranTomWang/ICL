export DEVICE="cuda"
export TOKENIZERS_PARALLELISM=false

export MODEL="Qwen/Qwen2.5-1.5B"
export OUT_DIR="out/Qwen2.5-1.5B"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator Qwen2Operator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator Qwen2Operator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator Qwen2Operator --device $DEVICE fv_steer

export MODEL="nvidia/Hymba-1.5B-Base"
export OUT_DIR="out/Hymba-1.5B-Base"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator HymbaOperator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator HymbaOperator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator HymbaOperator --device $DEVICE fv_steer

export MODEL="state-spaces/mamba-1.4b-hf"
export OUT_DIR="out/mamba-1.4b-hf"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator MambaOperator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator MambaOperator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator MambaOperator --device $DEVICE fv_steer

export MODEL="AntonV/mamba2-1.3b-hf"
export OUT_DIR="out/mamba2-1.3b-hf"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator Mamba2Operator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator Mamba2Operator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator Mamba2Operator --device $DEVICE fv_steer

export MODEL="Zyphra/Zamba2-1.2B"
export OUT_DIR="out/Zamba2-1.2B"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator ZambaOperator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator ZambaOperator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator ZambaOperator --device $DEVICE fv_steer

export MODEL="meta-llama/Llama-3.2-1B"
export OUT_DIR="out/Llama-3.2-1B"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator LlamaOperator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator LlamaOperator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator LlamaOperator --device $DEVICE fv_steer

export MODEL="EleutherAI/pythia-1B"
export OUT_DIR="out/pythia-1B"
python extract_activations.py --model $MODEL --task classification --k -1 --split dev --out_dir $OUT_DIR --operator GPTNeoXOperator --device $DEVICE fv_steer
python extract_activations.py --model $MODEL --task function_vectors_original --k -1 --split dev --out_dir $OUT_DIR --operator GPTNeoXOperator --device $DEVICE --use_template fv_steer
python extract_activations.py --model $MODEL --task function_vectors_incorrect_mapping --k -1 --split dev --out_dir $OUT_DIR --operator GPTNeoXOperator --device $DEVICE fv_steer