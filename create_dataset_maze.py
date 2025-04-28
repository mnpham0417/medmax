from src.image_tokenization import VQVAEImageProcessor, ImageDataset
import argparse 
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
from chameleon.inference.chameleon import TokenManager
import os
import json
from PIL import Image
import PIL
import torch
from pathlib import Path
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import accelerate
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
from tokenizers import Tokenizer
from src.tokenization import offset_image_tokens
import random
import json 

# import json
# import argparse
# from tqdm import tqdm
from tokenizers import Tokenizer

def write_list_of_dicts_to_jsonl(data, filename):
    """Writes a list of dictionaries to a JSON Lines file.

    Args:
        data: A list of dictionaries.
        filename: The name of the output file.
    """
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')




class MazeDataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self,path=''):
        super().__init__()

        self.path = path
        self.maze_dirs = os.listdir(path)
        
        #shuffle the maze_dirs
        random.shuffle(self.maze_dirs)
        # start_maze_paths = glob.glob(image_paths+ "/*/step_00.png")
        # all_maze_paths = glob.glob(image_paths + )
        # solution_text = glob.glob
        
        self.processor = VQVAEImageProcessor()
        
    def __getitem__(self, index):
        # item = self.data[index]
        image_path = self.path+ '/'+self.maze_dirs[index] + "/maze_0.png"
        text_path = self.path+ '/'+self.maze_dirs[index] + "/direction.json"

        with open(text_path, 'r') as file:
            content = json.load(file)

        try:
            image = Image.open(image_path).convert('RGB')
            assert image.height >= 1 and image.width >= 1
        except:
            image = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))

        
        image = self.processor(image)
        return dict(
            image_path=image_path,
            image=image,
            text_path=text_path, 
            text=content["final_destimation_point"],
            ground_truth_path=content["direction"]
        )
        
    def __len__(self):
        return len(self.maze_dirs)
        # return 1000



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,default='', help='input csv with image column')
    parser.add_argument('--output', type=str, default='', help='output folder where the vqgan image tokens should be stored')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--ckpt',type=str,default='', help='path to medmax folder')
    parser.add_argument('--tokenizer_file',type=str,default='', help='path to medmax folder')
    
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    os.makedirs(args.output, exist_ok=True)
    
    # dist.init_process_group()
    dataset = MazeDataset(args.input)
    dataloader = DataLoader(dataset,  batch_size=args.bs) #sampler=sampler,
    output_path = Path(args.output)

    vqgan_cfg_path = (ckpt_path / "tokenizer" / "vqgan.ckpt").as_posix()
    token_manager = TokenManager(
        tokenizer_path=  (ckpt_path / "tokenizer" / "text_tokenizer.json").as_posix(),
        vqgan_cfg_path= (ckpt_path / "tokenizer" / "vqgan.yaml").as_posix(),
        vqgan_ckpt_path=vqgan_cfg_path,
        device="cuda",
    )
    img_path_list, tokens_list = [], []
    partition_count = 0
    batch_size = args.bs

    tokenizer = Tokenizer.from_file(args.tokenizer_file)
    tokenizer.bos = 0
    tokenizer.pad = 1
    tokenizer.eos = 2
    tokenizer.image_id = 8711

    output_list = []
    

    for batch in tqdm(dataloader):
        bsz = len(batch['image'])
        with torch.no_grad():
            image_token_from_tensor = token_manager.image_tokenizer.image_token_from_tensor(batch['image'].cuda())
            image_token_from_tensor = image_token_from_tensor.cpu().view(bsz,-1)

        
        for b in range(bsz):
            entry = {}
            entry['image_path'] = batch['image_path'][b]
            entry['text_path'] = batch['text_path'][b]
            entry["image_tokens"] = image_token_from_tensor[b].cpu().numpy().tolist()
            image_tokens = offset_image_tokens(image_token_from_tensor[b].cpu().numpy().tolist())
            image_tokens = [8197] + image_tokens + [8196]
            
            # text = f"Task: Maze Navigation Simulation Determine the final destination (A, B, C or D) from the starting point (red point) following the action sequence. The definitions of the actions are as below. * Go up/left/down/right: move one grid space in the absolute up/left/down/right direction. Full Action Sequence: {batch['ground_truth_path'][b]}. Initial maze: <image> Response: <reserved08706> The answer is " + batch["text"][b]
            text = f"Task: Maze Navigation Simulation\nDetermine the shortest path that leads to the final destination (A, B, C or D) from the starting point (red point) following the action sequence. The definitions of the actions are as below.\n* Go up/left/down/right: move one grid space in the absolute up/left/down/right direction.\nInitial maze: <image> Final destination: {batch['text'][b]}.\nResponse: <reserved08706> The answer is {batch['ground_truth_path'][b]}"
            print(text)
            # assert False
            entry["text"] = text
            text_tokenized = [tokenizer.bos] + tokenizer.encode(text).ids + [tokenizer.eos]

            new_tokens = []
            for token in text_tokenized:
                if token == tokenizer.image_id:
                    new_tokens = new_tokens + image_tokens
                else:
                    new_tokens = new_tokens + [token]

            entry['tokens'] = new_tokens

            # print(entry)
            output_list.append(entry)
            
    
    write_list_of_dicts_to_jsonl(output_list, args.output + "/train_metadata_pathfinding_all.jsonl")     



    
    

    
    