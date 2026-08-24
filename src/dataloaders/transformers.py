import torch, torchvision
import random

# inputs are (rgb, depth) already (3, 480, 680), (1, 480, 680), 
# torch.float32, rgb[0.0, 1.0] depth[0.0, 10.0]
# input as 2 parameters, not tuple

# mean and std values for normalization (used in pretraining of ImageNet model so we use the same)
# as an outpur every rgb pixel is roughly [-2.1, +2.6]. So its like BatchNorm before first layer 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# makes every photo 4x smaller 
class ResizeImageNet:

    def __init__(self, size=(240, 320)):
        self.size = size

    def __call__(self, rgb, depth):
        rgb = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return rgb, depth


class HorizontalFlip:

    def __init__(self):
        self.p = 0.5

    def __call__(self, rgb, depth):
        if random.random() > self.p:
            rgb = torch.flip(rgb, dims=[-1])
            depth = torch.flip(depth, dims=[-1])
        return rgb, depth


class ColorJitter:

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)


    def __call__(self, rgb, depth):
        rgb = self.jitter(rgb)
        return rgb, depth


class NormalizeImageNet:

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.normalize = torchvision.transforms.Normalize(mean = mean, std = std)

    def __call__(self, rgb, depth):
        rgb = self.normalize(rgb)
        return rgb, depth


class Compose:

    def __init__(self, transformers_chain):
        self.transformers_chain = transformers_chain

    def __call__(self, rgb, depth):
        for transform in self.transformers_chain:
            rgb, depth = transform(rgb, depth)
        return rgb, depth


def denormalize_image_net(rgb, depth):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    return (rgb.cpu() * std + mean).clamp(0, 1), depth


def built_transform_train_imagenet():

    return Compose([
        ResizeImageNet(),
        NormalizeImageNet(),
        HorizontalFlip(),
        ColorJitter()
    ])


def built_transform_eval_imagenet():

    return Compose([
        ResizeImageNet(),
        NormalizeImageNet()
    ])

