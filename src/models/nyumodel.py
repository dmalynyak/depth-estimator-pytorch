from src.models.nyudecoder import NYUdecoder
from src.models.resnet18 import Resnet18Encoder

import torch.nn as nn

class NYUmodel(nn.Module):
    def __init__(self):
        super().__init__()

        # input: (B, 3, 240, 320)
        self.encoder = Resnet18Encoder()
        self.decoder = NYUdecoder()
        # output: (B, 1, 240, 320)

    def forward(self, rgbs):
        feats = self.encoder(rgbs)
        preds = self.decoder(feats)
        
        return preds