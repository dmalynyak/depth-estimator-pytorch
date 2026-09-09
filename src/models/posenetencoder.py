import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights


# for KITTI input (B, 3, 192, 640) and (B, 3, 192, 640) t,t+1 augmented frames
# two tensors are stacked
# output (B, 512, 6, 20)
class PosenetResnet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=w)

        old = m.conv1 # ImageNet pretraining has input channels = 3, so we create new one with 6 channels
        new = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        new.weight.data = torch.cat([old.weight.data] * 2, dim=1) / 2 # makes a copy of pretraied weights and cats them
        self.stem = nn.Sequential(new, m.bn1, m.relu)

        self.pool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(self.pool(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x # (B, 512, 6, 20)  