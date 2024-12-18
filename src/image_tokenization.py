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
import argparse

class VQVAEImageProcessor:
    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> torch.Tensor:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        tensor_img = (
            torch.from_numpy(np_img).permute(2, 0, 1).float()
        )  # (Channels, Height, Width) format.

        # Add batch dimension.
        return tensor_img#.unsqueeze(0)

    def __call__(self, image):
        image = self._whiten_transparency(image)
        vqgan_input = self._vqgan_input_from(image)#.to(self._device).to(self._dtype)
        return vqgan_input
    



class ImageDataset(torch.utils.data.dataset.Dataset):
    
    def __init__(self,annotation_path='',image_key=None,limit=-1):
        super().__init__()
        if os.path.isfile(annotation_path):
            df = pd.read_csv(annotation_path)
        else:
            df = pd.read_csv(os.path.join(annotation_path,'metadata.csv'))
            df['image_path'] = [os.path.join(annotation_path,x) for x in df.file_name]
        
        if image_key is not None:
            df['image_path'] = df[image_key]
        else:
            keys = df.columns
            if 'img_path' in keys:
                    df['image_path'] = df['img_path'] 
        self.data = df.to_dict(orient='records')
        if limit > 0:
            self.data = self.data[:limit]
        self.processor = VQVAEImageProcessor()
        
    def __getitem__(self, index):
        item = self.data[index]
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            assert image.height >= 1 and image.width >= 1
        except:
            image = Image.fromarray(np.zeros((224,224,3),dtype=np.uint8))
        image = self.processor(image)
        return dict(
            image_path=image_path,
            image=image,
        )
        
    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    '''
    torchrun --nproc-per-node=1 --standalone --master-port 1111 dist_tokenization.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,default='', help='input csv with image column')
    parser.add_argument('--output', type=str, default='', help='output folder where the vqgan image tokens should be stored')
    parser.add_argument('--partition_size', type=int, default=5_000, help='partition size of the dataset')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--n_limit', type=int, default=-1)
    parser.add_argument('--ckpt',type=str,default='', help='path to medmax folder')

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    os.makedirs(args.output, exist_ok=True)
    
    dist.init_process_group()
    dataset = ImageDataset(args.input, limit=args.n_limit)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.bs)
    local_rank = os.environ['LOCAL_RANK']
    local_rank = int(local_rank)
    print(f"RANK: {local_rank}, init")
    torch.cuda.set_device(local_rank)
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
    batch_size = args.partition_size

    for batch in tqdm(dataloader):
        out = output_path/f'partition_{partition_count}_rank_{local_rank}.parquet'
        if os.path.exists(out):
            partition_count += 1
            continue
        bsz = len(batch['image'])
        with torch.no_grad():
            image_token_from_tensor = token_manager.image_tokenizer.image_token_from_tensor(batch['image'].cuda())
            image_token_from_tensor = image_token_from_tensor.cpu().view(bsz,-1)
        
        img_path_list.extend(batch['image_path'])
        tokens_list.extend(image_token_from_tensor.tolist())
        
        # Check if batch size is reached
        if len(img_path_list) >= batch_size:
            # Create a DataFrame from accumulated data
            df = pd.DataFrame({
                'img_path': img_path_list,
                'img_tokens': tokens_list
            })
            # Convert DataFrame to Apache Arrow Table for parquet saving
            # Save the partition as Parquet
            df.to_parquet(out)
            # Increment partition counter and reset lists
            partition_count += 1
            img_path_list, tokens_list = [], []
    # Save any remaining data not saved in the last batch
    out = output_path/f'partition_{partition_count}_rank_{local_rank}.parquet'
    if not os.path.exists(out):
        df = pd.DataFrame({
            'img_path': img_path_list,
            'img_tokens': tokens_list
        })
        df.to_parquet(out)