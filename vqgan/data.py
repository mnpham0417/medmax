import pandas as pd
from torch.utils.data import Dataset
from omegaconf import OmegaConf
import os
import glob
from taming.data.base import ImagePaths
from taming.util import retrieve
class MedicalImageDataset(Dataset):
    def __init__(self, root=None,random_crop=False, fmt='*.[pjJ][npNP][gG]',config=None,mode='train',val_images=1000):
        self.config = config or OmegaConf.create()
        self.random_crop =random_crop
        assert root is not None
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        if isinstance(root,str):
            root = [root]
        all_images = []
        for root_path in root:
            if root_path.endswith('.csv'):
                images = pd.read_csv(root_path)['image_path'].tolist()
            else:
                images = glob.glob(os.path.join(root_path,'**',fmt))
                if len(images) == 0:
                    images = glob.glob(os.path.join(root_path,fmt))
                images = sorted(images)
            all_images.extend(images)
        images = all_images
        n = len(images)
        n_train = n - val_images
        assert n_train > 0
        if mode == 'train':
            images = images[val_images:]
        else:
            images = images[:val_images]
        print(f"INIT: {len(images)} Images Found")
        self.data = ImagePaths(images,
                               labels=None,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop
                               )
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)