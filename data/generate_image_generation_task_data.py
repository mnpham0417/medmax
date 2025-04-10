import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
import os
import json
import argparse

#set seed
torch.manual_seed(42)

def load_prompts(prompt_file):
    #load json file
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    return prompts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="/scratch/mp5847/hugging_face_models/stable-diffusion-2-1-base")
    parser.add_argument("--save_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metadata = []
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    
    prompts = load_prompts(args.prompt_file)

    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt
        ).images[0]
        image.save(os.path.join(args.output_dir, "train", f"img_{args.save_name}_{i}.png"))    
        # metadata.append({"img_path": os.path.join(args.output_dir, "train", f"img_{args.save_name}_{i}.png"), "text": prompt + "<reserved08706><image>"})
        metadata.append({"img_path": os.path.join(args.output_dir, "train", f"img_{args.save_name}_{i}.png"), "text": prompt + "<image>"})

    with open(os.path.join(args.output_dir, 'metadata.jsonl'), 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")


