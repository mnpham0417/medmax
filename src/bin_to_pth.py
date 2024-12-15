import os
import re
import argparse
import shutil
from typing import Dict, Optional
from pathlib import Path
from transformers import ChameleonForConditionalGeneration

import torch


files_to_copy = [
    "models/7b/checklist.chk",
    # "models/7b/config.json",
    "models/7b/consolidate_params.json",
    "models/7b/params.json",
    "tokenizer/checklist.chk",
    "tokenizer/text_tokenizer.json",
    "tokenizer/vqgan.ckpt",
    "tokenizer/vqgan.yaml",
]


_FROM_BIN = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_normalization.weight",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_normalization.weight",
    "model.layers.{}.self_attn.q_norm.bias": "layers.{}.attention.q_normalization.bias",
    "model.layers.{}.self_attn.k_norm.bias": "layers.{}.attention.k_normalization.bias",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}

def revert_permute(w_permuted, n_heads=32, dim1=4096, dim2=4096):
    # Reverse reshape and transpose to get the original w
    return w_permuted.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def revert_transform(x_transformed, dims_per_head=128, n_heads=32):
    x_single = x_transformed[0].view(2, dims_per_head // 2)
    x_original_shape = x_single.t()
    x_reverted = x_original_shape.reshape(dims_per_head)    
    return x_reverted

def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if "layers" in key:
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)  # type: ignore
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def bin_to_pth(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    dim: int = 4096,
    qk_norm: bool = True,
    head_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    if not qk_norm:
        _FROM_BIN.pop("model.layers.{}.self_attn.q_norm.weight")
        _FROM_BIN.pop("model.layers.{}.self_attn.k_norm.weight")
        _FROM_BIN.pop("model.layers.{}.self_attn.q_norm.bias")
        _FROM_BIN.pop("model.layers.{}.self_attn.k_norm.bias")

    for key, value in state_dict.items():
        if "rope.freqs" in key:
            print(key)
        if "model.vqmodel" not in key:
            new_key = get_mapped_key(key, _FROM_BIN)
            if 'q_norm.weight' in key or 'q_norm.bias' in key or 'k_norm.weight' in key or 'k_norm.bias' in key:
                value = revert_transform(value)
                # print('transformed')
            if 'self_attn.q_proj.weight' in key or 'self_attn.k_proj.weight' in key:
                value = revert_permute(value)
                # print('reverted')
            converted_state_dict[new_key] = value

    for key, value in converted_state_dict.items():
        converted_state_dict[key] = value.bfloat16()
    converted_state_dict["rope.freqs"] = None
    return converted_state_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_ckpt', default='', type=str, help="trained checkpoint")
    parser.add_argument('--original_ckpt', default='', type=str, help="original anole checkpoint that can be loaded w. chlm code")
    parser.add_argument('--new_ckpt', default='', type=str, help="folder for the converted checkpoint")
    args = parser.parse_args()

    ANOLE_PATH_HF_TRAINED = Path(args.trained_ckpt)
    ANOLE_PATH_TORCH = Path(args.original_ckpt)

    ANOLE_PATH_TORCH_NEW = Path(args.new_ckpt)

    # Create directories if they do not exist
    os.makedirs(ANOLE_PATH_TORCH_NEW / "models/7b", exist_ok=True)
    os.makedirs(ANOLE_PATH_TORCH_NEW / "tokenizer", exist_ok=True)

    model = ChameleonForConditionalGeneration.from_pretrained(ANOLE_PATH_HF_TRAINED)
    bin_state_dict = model.state_dict()

    print(f"loaded anole-bin weights from {ANOLE_PATH_HF_TRAINED}")

    # TODO: the following setting only for 7b model
    pth_state_dict = bin_to_pth(
        bin_state_dict,
        num_heads=32,
        dim=4096,
        qk_norm=True,
    )

    torch.save(pth_state_dict, ANOLE_PATH_TORCH_NEW / "models/7b/consolidated.pth")
    print(f"saved anole-pth weights to {ANOLE_PATH_TORCH_NEW / 'models/7b/consolidated.pth'}")

    for filename in files_to_copy:
        shutil.copy(ANOLE_PATH_TORCH / filename, ANOLE_PATH_TORCH_NEW / filename)
