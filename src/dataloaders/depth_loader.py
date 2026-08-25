import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional
from pathlib import Path

import src.dataloaders

class DepthNYUDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, split):

        assert split in ("train", "eval"), f"split param must be 'train' or 'eval', got {split}"

        self.data_dir = Path(data_dir)
        self.samples = []
        image_files = sorted(self.data_dir.glob('*_rgb.png'))

        for rgb_path in image_files:

            depth_name = rgb_path.name.replace('_rgb.png', '_depth.npy')
            depth_path = rgb_path.with_name(depth_name)

            if depth_path.exists():
                self.samples.append((rgb_path, depth_path))
            else:
                print(f"skip {rgb_path}, file {depth_path} not found")

        if split == "train":
            self.transform = src.dataloaders.built_transform_train_imagenet()
        else:
            self.transform = src.dataloaders.built_transform_eval_imagenet()



    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        image = Image.open(rgb_path).convert("RGB")
        rgb_tensor = torchvision.transforms.functional.to_tensor(image) # (3, H, W), [0.0, 1.0] dtype=torch.float32

        depth_arr = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth_arr).float() # (3, H, W), [0.0, 10.0] dtype=torch.float32
        
        if depth_tensor.dim() == 2:
            depth_tensor = depth_tensor.unsqueeze(0)

        rgb_tensor, depth_tensor = self.transform(rgb_tensor, depth_tensor)
            
        return rgb_tensor, depth_tensor # assert ( rgb(3, 480, 680) , depth(1, 480, 680) ), dtype=torch.float32, rgb[0.0, 1.0] depth[0.0, 10.0]
