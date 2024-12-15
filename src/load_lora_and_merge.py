import os
import torch
import argparse
from peft import LoraConfig, get_peft_model

def main(args):

    path = args.ckpt_path

    from transformers import ChameleonForConditionalGeneration
    trained_model = ChameleonForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")

    trained_sd = trained_model.state_dict()
    trained_sd = {"base_model.model." + k: v for k, v in trained_sd.items()}

    base_path = args.base_path
    base_model = ChameleonForConditionalGeneration.from_pretrained(base_path)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj","gate_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)

    model.load_state_dict(trained_sd)

    model = model.merge_and_unload()    
    os.makedirs(args.output_dir, exist_ok = True)

    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help = 'path to trained lora checkpoint')
    parser.add_argument("--base_path", help = 'path to base path')
    parser.add_argument("--output_dir", help = 'path to output dir')
    args = parser.parse_args()
    main(args)