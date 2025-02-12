import torch
from typing import Dict, Optional, Union
from pathlib import Path
import logging
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions import (
    convert_bert_weights,
    convert_bloom_weights,
    convert_coder_weights,
    convert_gemma_weights,
    convert_gpt2_weights,
    convert_gptj_weights,
    convert_llama_weights,
    convert_mistral_weights,
    convert_mixtral_weights,
    convert_neo_weights,
    convert_neox_weights,
    convert_opt_weights,
    convert_phi3_weights,
    convert_phi_weights,
    convert_qwen2_weights,
    convert_qwen_weights,
    convert_t5_weights
)

def load_tl_model(
    path_to_model: str,
    device: torch.device,
    default_padding_side: str = "right",
    fold_ln: bool = True,
    dtype: torch.dtype = torch.float32,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    refactor_factored_attn_matrices: bool = False,
    fold_value_biases: bool = True,
    verbose: bool = False
):
    """Loads a pretrained model given path to model (on local machine).

    Args:
        path_to_model (str): path to local machine model repo.
        device (torch.device): device to load model to.
        default_padding_side (str): default padding side, defaults to "right".
        fold_ln (bool): whether tp fold in the LayerNorm weights to subsequent linear layer, defaults to True
        dtype (torch.dtype): data type of model, defaults to torch.float32.
        center_writing_weights: Whether to center weights
            writing to the residual stream (ie set mean to be zero). Due to LayerNorm this
            doesn't change the computation.

            A related idea to folding layernorm (``fold_ln``) - *every* component reading an
            input from the residual stream is preceded by a LayerNorm, which means that the mean
            of a residual stream vector (ie the component in the direction of all ones) never
            matters. This means we can remove the all ones component of weights and biases whose
            output *writes* to the residual stream. Mathematically, ``W_writing -=
            W_writing.mean(dim=1, keepdim=True)``.
        center_unembed: Whether to center W_U (ie set mean
            to be zero). Softmax is translation invariant so this doesn't affect log probs or
            loss, but does change logits.

            The logits are fed into a softmax. Softmax is translation invariant (eg, adding 1 to
            every logit doesn't change the output), so we can simplify things by setting the
            mean of the logits to be zero. This is equivalent to setting the mean of every
            output vector of ``W_U`` to zero. In code, ``W_U -= W_U.mean(dim=-1,
            keepdim=True)``.
        refactor_factored_attn_matrices: Whether to convert the factored
            matrices (W_Q & W_K, and W_O & W_V) to be "even". Defaults to False.
        fold_value_biases: Each attention head has a value bias. Values are averaged to create
            mixed values (``z``), weighted by the attention pattern, but as the bias is
            constant, its contribution to ``z`` is exactly the same. The output of a head is ``z
            @ W_O``, and so the value bias just linearly adds to the output of the head. This
            means that the value bias of a head has nothing to do with the head, and is just a
            constant added to the attention layer outputs. We can take the sum across these and
            b_O to get an "effective bias" for the layer. In code, we set ``b_V=0``. and ``b_O =
            (b_V @ W_O).sum(dim=0) + b_O``.

            The technical derivation of this is as follows. ``v = residual @ W_V[h] +
            broadcast_b_V[h]`` for each head ``h`` (where ``b_V`` is broadcast up from shape
            ``d_head`` to shape ``[position, d_head]``). And ``z = pattern[h] @ v = pattern[h] @
            residual @ W_V[h] + pattern[h] @ broadcast_b_V[h]``. Because ``pattern[h]`` is
            ``[destination_position, source_position]`` and ``broadcast_b_V`` is constant along
            the ``(source_)position`` dimension, we're basically just multiplying it by the sum
            of the pattern across the ``source_position`` dimension, which is just ``1``. So it
            remains exactly the same, and so is just broadcast across the destination positions.
        verbose: whether to print debugging information, defaults to False.
    """
    assert Path(path_to_model).exists()
    cfg = get_pretrained_model_config(path_to_model, device=device)
    hf_model = AutoModelForCausalLM.from_pretrained(path_to_model, torch_dtype=dtype, use_safetensors=True, device_map=device)
    if verbose:
        print(hf_model)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"After loading model allocated {allocated_memory} GB cuda memory")
    state_dict = get_pretrained_state_dict(cfg, hf_model)
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(device)
    if verbose:
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        print(f"After loading state_dict allocated {allocated_memory} GB cuda memory")
    del hf_model
    
    tokenizer = AutoTokenizer.from_pretrained(path_to_model, torch_dtype=dtype, use_safetensors=True, device_map=device)
    if verbose:
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        print(f"Before loading HookedTransformer allocated {allocated_memory} GB cuda memory")
    hooked_model = HookedTransformer(cfg, tokenizer, move_to_device=True, default_padding_side=default_padding_side, )
    if verbose:
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
        print(f"After loading HookedTransformer allocated {allocated_memory} GB cuda memory")
    
    hooked_model.load_and_process_state_dict(
        state_dict,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
    )
    return hooked_model

def get_pretrained_state_dict(
    cfg: HookedTransformerConfig,
    hf_model
) -> Dict[str, torch.Tensor]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes.

    Args:
        cfg: HookedTransformerConfig object.
        hf_model: a HuggingFace model object.
    """
    for param in hf_model.parameters():
        param.requires_grad = False

    if cfg.original_architecture == "GPT2LMHeadModel":
        state_dict = convert_gpt2_weights(hf_model, cfg)
    elif cfg.original_architecture == "GPTNeoForCausalLM":
        state_dict = convert_neo_weights(hf_model, cfg)
    elif cfg.original_architecture == "OPTForCausalLM":
        state_dict = convert_opt_weights(hf_model, cfg)
    elif cfg.original_architecture == "GPTJForCausalLM":
        state_dict = convert_gptj_weights(hf_model, cfg)
    elif cfg.original_architecture == "GPTNeoXForCausalLM":
        state_dict = convert_neox_weights(hf_model, cfg)
    elif cfg.original_architecture == "LlamaForCausalLM":
        state_dict = convert_llama_weights(hf_model, cfg)
    elif cfg.original_architecture == "BertForMaskedLM":
        state_dict = convert_bert_weights(hf_model, cfg)
    elif cfg.original_architecture == "T5ForConditionalGeneration":
        state_dict = convert_t5_weights(hf_model, cfg)
    elif cfg.original_architecture == "MistralForCausalLM":
        state_dict = convert_mistral_weights(hf_model, cfg)
    elif cfg.original_architecture == "MixtralForCausalLM":
        state_dict = convert_mixtral_weights(hf_model, cfg)
    elif cfg.original_architecture == "BloomForCausalLM":
        state_dict = convert_bloom_weights(hf_model, cfg)
    elif cfg.original_architecture == "GPT2LMHeadCustomModel":
        state_dict = convert_coder_weights(hf_model, cfg)
    elif cfg.original_architecture == "QWenLMHeadModel":
        state_dict = convert_qwen_weights(hf_model, cfg)
    elif cfg.original_architecture == "Qwen2ForCausalLM":
        state_dict = convert_qwen2_weights(hf_model, cfg)
    elif cfg.original_architecture == "PhiForCausalLM":
        state_dict = convert_phi_weights(hf_model, cfg)
    elif cfg.original_architecture == "Phi3ForCausalLM":
        state_dict = convert_phi3_weights(hf_model, cfg)
    elif cfg.original_architecture == "GemmaForCausalLM":
        state_dict = convert_gemma_weights(hf_model, cfg)
    elif cfg.original_architecture == "Gemma2ForCausalLM":
        state_dict = convert_gemma_weights(hf_model, cfg)
    else:
        raise ValueError(
            f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
        )

    return state_dict

def get_pretrained_model_config(
    model_path: str,
    hf_cfg: Optional[dict] = None,
    fold_ln: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    n_devices: int = 1,
    default_prepend_bos: bool = True,
    dtype: torch.dtype = torch.float32,
    first_n_layers: Optional[int] = None,
    **kwargs,
):
    """Returns the pretrained model config as an HookedTransformerConfig object.

    Args:
        model_path: The path to the model.
        hf_cfg (dict, optional): Config of a loaded pretrained HF model,
            converted to a dictionary.
        fold_ln (bool, optional): Whether to fold the layer norm into the
            subsequent linear layers (see HookedTransformer.fold_layer_norm for
            details). Defaults to False.
        device (str, optional): The device to load the model onto. By
            default will load to CUDA if available, else CPU.
        n_devices (int, optional): The number of devices to split the model across. Defaults to 1.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        dtype (torch.dtype, optional): The dtype to load the TransformerLens model in.
        kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
            Also given to other HuggingFace functions when compatible.

    """
    assert Path(model_path).exists()
    # If the model_name is a path, it's a local model
    cfg_dict = convert_hf_model_config(model_path, **kwargs)
    official_model_name = model_path
    # Processing common to both model types
    # Remove any prefix, saying the organization who made a model.
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    # Don't need to initialize weights, we're loading from pretrained
    cfg_dict["init_weights"] = False

    if (
        "positional_embedding_type" in cfg_dict
        and cfg_dict["positional_embedding_type"] == "shortformer"
        and fold_ln
    ):
        logging.warning(
            "You tried to specify fold_ln=True for a shortformer model, but this can't be done! Setting fold_ln=False instead."
        )
        fold_ln = False

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict["dtype"] = dtype

    if fold_ln:
        if cfg_dict["normalization_type"] in ["LN", "LNPre"]:
            cfg_dict["normalization_type"] = "LNPre"
        elif cfg_dict["normalization_type"] in ["RMS", "RMSPre"]:
            cfg_dict["normalization_type"] = "RMSPre"
        else:
            logging.warning("Cannot fold in layer norm, normalization_type is not LN.")

    cfg_dict["from_checkpoint"] = False

    cfg_dict["device"] = device
    cfg_dict["n_devices"] = n_devices
    cfg_dict["default_prepend_bos"] = default_prepend_bos
    if hf_cfg is not None:
        cfg_dict["load_in_4bit"] = hf_cfg.get("quantization_config", {}).get("load_in_4bit", False)
    if first_n_layers is not None:
        cfg_dict["n_layers"] = first_n_layers

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg

def convert_hf_model_config(model_name: str, **kwargs):
    """
    Returns the model config for a HuggingFace model in local machine, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the path to model as an input.
    """
    # In case the user passed in an alias
    assert (Path(model_name) / "config.json").exists()
    with open(Path(model_name) / "config.json", "r") as f:
        hf_config = json.load(f)
    logging.info("Loading model config from local directory")
    official_model_name = "/".join(model_name.split("/")[2:])

    # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
    elif "gemma-2" in official_model_name.lower():
        architecture = "Gemma2ForCausalLM"
    elif "gemma" in official_model_name.lower():
        architecture = "GemmaForCausalLM"

    if official_model_name.startswith(
        ("llama-7b", "meta-llama/Llama-2-7b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("codellama"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Llama-3.2-1B" in official_model_name:
        cfg_dict = {
            "d_model": 2048,
            "d_head": 64,
            "n_heads": 32,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 64,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
        }
    elif "Llama-3.2-3B" in official_model_name:
        cfg_dict = {
            "d_model": 3072,
            "d_head": 128,
            "n_heads": 24,
            "d_mlp": 8192,
            "n_layers": 28,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
        }
    elif "Llama-3.1-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
        }
    elif "Llama-3.1-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "rotary_adjacent_pairs": True,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "BertForMaskedLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu",
            "attention_dir": "bidirectional",
        }
    elif architecture == "MistralForCausalLM":
        use_local_attn = True if hf_config.sliding_window else False
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.head_dim
            if hasattr(hf_config, "head_dim") and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "window_size": hf_config.sliding_window,  # None if no sliding window was used
            "attn_types": ["local"] * hf_config.num_hidden_layers if use_local_attn else None,
            "eps": hf_config.rms_norm_eps,
            "rotary_base": hf_config.rope_theta,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "use_local_attn": use_local_attn,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif architecture == "MixtralForCausalLM":
        cfg_dict = {
            "dtype": torch.bfloat16,
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "window_size": hf_config.sliding_window,  # This is None, as no sliding window was used
            "attn_types": ["global"] * 32,
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "gated_mlp": True,
            "use_local_attn": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "num_experts": hf_config.num_local_experts,
            "experts_per_token": hf_config.num_experts_per_tok,
        }
    elif architecture == "BloomForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": 2048,  # Capped due to HF Tokenizer Constraints
            "d_vocab": hf_config.vocab_size,
            "act_fn": "gelu_fast",
            "eps": hf_config.layer_norm_epsilon,
            "normalization_type": "LN",
            "post_embedding_ln": True,
            "positional_embedding_type": "alibi",
        }
    elif architecture == "GPT2LMHeadCustomModel":
        # santacoder
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "trust_remote_code": "santacoder"
            in official_model_name,  # Only santacoder needs trust_remote_code
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "tokenizer_prepends_bos": True,
            "trust_remote_code": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "tokenizer_prepends_bos": True,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "PhiForCausalLM":
        # Architecture for microsoft/phi models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
        partial_rotary_factor = hf_config.partial_rotary_factor
        cfg_dict["rotary_dim"] = round(partial_rotary_factor * cfg_dict["d_head"])
    elif architecture == "Phi3ForCausalLM":
        # Architecture for microsoft/phi3 models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "trust_remote_code": True,
            "rotary_base": hf_config.rope_theta,
            "use_attn_scale": True,
            "gated_mlp": True,
            "parallel_attn_mlp": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
        }

    elif official_model_name.startswith("google/gemma-2b"):
        # Architecture for Gemma 2b and Gemma 2b Instruct models
        cfg_dict = {
            "d_model": 2048,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 16384,
            "n_layers": 18,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 1,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-7b"):
        # Architecture for Gemma 7b and Gemma 7b Instruct models
        cfg_dict = {
            "d_model": 3072,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 24576,
            "n_layers": 28,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_new",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "rotary_dim": 256,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 16,
            "gated_mlp": True,
            "final_rms": True,
        }
    elif official_model_name.startswith("google/gemma-2-2b"):
        # Architecture for Gemma-2 2b and Gemma-2 2b Instruct models
        cfg_dict = {
            "d_model": 2304,
            "d_head": 256,
            "n_heads": 8,
            "d_mlp": 9216,
            "n_layers": 26,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 4,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-9b"):
        # Architecture for Gemma-2 9b and Gemma-2 9b Instruct models
        cfg_dict = {
            "d_model": 3584,
            "d_head": 256,
            "n_heads": 16,
            "d_mlp": 14336,
            "n_layers": 42,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "n_key_value_heads": 8,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif official_model_name.startswith("google/gemma-2-27b"):
        # Architecture for Gemma-2 27b and Gemma-2 27b Instruct models
        cfg_dict = {
            "d_model": 4608,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 36864,
            "n_layers": 46,
            "n_ctx": 8192,
            "eps": 1e-06,
            "d_vocab": 256000,
            "act_fn": "gelu_pytorch_tanh",
            "initializer_range": 0.02,
            "normalization_type": "RMS",
            "rotary_base": 10000.0,
            "positional_embedding_type": "rotary",
            "use_attn_scale": True,
            "attn_scale": 12.0,
            "n_key_value_heads": 16,
            "window_size": 4096,
            "use_local_attn": True,
            "attn_types": ["global", "local"] * 23,  # Alternate global and local attn
            "attn_scores_soft_cap": 50.0,
            "output_logits_soft_cap": 30.0,
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": True,
        }
    elif architecture == "T5ForConditionalGeneration":
        cfg_dict = {
            "d_model": hf_config.d_model,
            "d_head": hf_config.d_kv,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.d_ff,
            "d_vocab": hf_config.vocab_size,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_length,
            "eps": hf_config.layer_norm_epsilon,
            "act_fn": hf_config.feed_forward_proj,
            "positional_embedding_type": "relative_positional_bias",
            "relative_attention_max_distance": hf_config.relative_attention_max_distance,
            "relative_attention_num_buckets": hf_config.relative_attention_num_buckets,
            "decoder_start_token_id": hf_config.decoder_start_token_id,
            "attention_dir": "bidirectional",
            "use_attn_scale": False,
            "tie_word_embeddings": hf_config.tie_word_embeddings,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict
