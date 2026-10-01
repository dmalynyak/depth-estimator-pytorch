import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional
from pathlib import Path
import cv2
import glob

import src.dataloaders

def load_K(calib_dir):
    values = {}
    with open(f"{calib_dir}/calib_cam_to_cam.txt") as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            try:
                values[key.strip()] = np.array([float(x) for x in rest.split()])
            except ValueError:
                continue

    P = values["P_rect_02"].reshape(3, 4)
    return P[:3, :3].astype(np.float64)

class DepthKITTIDataset(torch.utils.data.Dataset):

    def __init__(self, root, drives, height=192, width=640, train=True):

        self.root = root
        self.height, self.width = height, width
        self.sizes = (self.height, self.width)
        self.samples = []
        self.calibs = {}
        self.train = train

        for date, drive in drives:
            img_dir = f"{root}/{date}/{drive}/image_02/data"
            n = len(glob.glob(f"{img_dir}/*.png"))

            for i in range(1, n - 1):
                self.samples.append((date, drive, i))

            if date not in self.calibs:
                self.calibs[date] = load_K(f"{root}/{date}/{date}_calib")

        self.jitter = src.dataloaders.KittiJitter()
        self.normalize = src.dataloaders.NormalizeKittiImageNet()

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        date, drive, i = self.samples[idx]
        d = f"{self.root}/{date}/{drive}/image_02/data"

        pil = [Image.open(f"{d}/{i+o:010d}.png").convert("RGB") for o in (-1, 0, 1)]
        orig_w, orig_h = pil[0].size

        pil_resized = [p.resize((self.width, self.height), Image.LANCZOS) for p in pil]
        imgs_resized = torch.stack([torchvision.transforms.functional.to_tensor(p) for p in pil_resized]) # (3,3,H,W)

        if self.train:
            imgs_augmentated = self.jitter(imgs_resized)
        else:
            imgs_augmentated = imgs_resized

        imgs_augmentated = self.normalize(imgs_augmentated) 

        K = self.calibs[date].copy()
        K[0] *= self.width  / orig_w
        K[1] *= self.height / orig_h

        out = {
            "imgs": imgs_resized,
            "imgs_augmented": imgs_augmentated,
            "K": torch.from_numpy(K).float(),
            "inv_K": torch.from_numpy(np.linalg.inv(K)).float(),
        }

        if not self.train:
            gt = np.load(f"{self.root}/{date}/{drive}/gt_depth/{i:010d}.npy")
            out["gt_depth"] = torch.from_numpy(gt)

        return out

def get_inference_tensor_kitti(path, heigh=192, width=640, device="cpu"):
    img_bgr = cv2.imread(path) # returns brg image 
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # BGR (cv2) -> RGB
    pil = Image.fromarray(rgb) # numpy -> PIL
    transform = src.dataloaders.built_one_img_transform(heigh, width)
    return transform(pil).unsqueeze(0).to(device)

def get_inference_frame_kitti(frame_bgr, transform, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return transform(pil).unsqueeze(0).to(device)