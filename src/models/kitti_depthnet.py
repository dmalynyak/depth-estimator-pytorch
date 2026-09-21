from src.models.kittidecoder import KITTIdecoder
from src.models.resnet18 import Resnet18Encoder

import torch.nn as nn

class KITTIdepthNET(nn.Module):
    def __init__(self):
        super().__init__()

        # input: (B, 3, 192, 640)
        self.encoder = Resnet18Encoder()
        self.decoder = KITTIdecoder()
        # output: [B, 1, 192, 640][B, 1, 96, 320][B, 1, 48, 160][B, 1, 24, 80]
        # output is not depth but normilized disparity = (disp − min_disp) / (max_disp − min_disp), where disp = 1/depth
        # conversion to depth: 
        #   scaled = min_disp + (max_disp − min_disp) · sigmoid_output(normilized disparity)
        #   depth  = 1 / scaled

    def forward(self, img):
        feats = self.encoder(img)
        preds = self.decoder(feats)
        
        return preds