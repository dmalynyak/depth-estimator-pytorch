from src.models.nyudecoder import NYUdecoder
from src.models.resnet18 import Resnet18Encoder

import torch.nn as nn

class NYUmodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Resnet18Encoder()
        self.decoder = NYUdecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x