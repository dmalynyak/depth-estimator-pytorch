from src.models.kittidecoder import KITTIdecoder
from src.models.resnet18 import Resnet18Encoder

import torch.nn as nn

class KITTIdepthNET(nn.Module):
    def __init__(self):
        super().__init__()

        # input: (B, 3, 192, 640)
        self.encoder = Resnet18Encoder()
        self.decoder = KITTIdecoder()
        # output: [B, 16, 192, 640][B, 32, 96, 320][B, 64, 48, 160][B, 128, 24, 80]

    def forward(self, img):
        feats = self.encoder(img)
        preds = self.decoder(feats)
        
        return preds