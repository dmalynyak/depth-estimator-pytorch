from src.models.posenetdecoder import PosenetDecoder
from src.models.posenetencoder import PosenetResnet18Encoder

import torch.nn as nn
import torch

class KITTIposeNET(nn.Module):
    def __init__(self):
        super().__init__()

        # input: (B, 6, 192, 640) t,t+1 frames stacked
        self.encoder = PosenetResnet18Encoder()
        self.decoder = PosenetDecoder()
        # output: (B, 6) (pith, yaw, roll, left/right, down/up, backward/forward)

    def forward(self, frame_a, frame_b):
        imgs_stacked = torch.cat([frame_a, frame_b], dim=1)
        feats = self.encoder(imgs_stacked)
        pred = self.decoder(feats)
        
        return pred