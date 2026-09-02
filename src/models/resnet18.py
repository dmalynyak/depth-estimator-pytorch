# from torchinfo import summary          # pip install torchinfo
# m = resnet18()
# summary(m, input_size=(1, 3, 480, 640))  # your KITTI input size
# print(m)

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class Resnet18Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=w)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
        self.pool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.num_ch = [64, 64, 128, 256, 512]

    def forward(self, x):
        feats = [] # for NYU (B, 3, 480, 640) it has [ [B, 64, 240, 320] , 
        #                                              [B, 64, 120, 160] , 
        #                                              [B, 128, 60, 80] , 
        #                                              [B, 256, 30, 40] , 
        #                                              [B, 512, 15, 20] ]
        # last avgpool and fc are discarder

        x = self.stem(x)
        feats.append(x) # 1/2, 64ch

        x = self.layer1(self.pool(x))
        feats.append(x) # 1/4, 64ch

        x = self.layer2(x)
        feats.append(x) # 1/8, 128ch

        x = self.layer3(x)
        feats.append(x) # 1/16, 256ch

        x = self.layer4(x)
        feats.append(x) # 1/32, 512ch

        return feats