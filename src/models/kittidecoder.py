# encoder returns feats:
# [ [B, 64, 96, 320] ,
#   [B, 64, 48, 160] ,
#   [B, 128, 24, 80] ,
#   [B, 256, 12, 40] ,
#   [B, 512, 6, 20] ]

import torch
import torch.nn as nn
import torch.nn.functional

class Block(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, in_channels2, out_channels2, kernel_size2, stride2, padding2):
        super().__init__()

        self.conv1_h = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        )
        self.conv2_h = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels2, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        )
        self.activation = nn.ELU()

    def forward(self, x, passthrow=None):

        x = self.activation(self.conv1_h(x))
        
        if passthrow is not None:
            x = torch.nn.functional.interpolate(x, size=passthrow.shape[-2:], mode='bilinear', align_corners=False)
            # print(f"x: {x.shape} passthrow: {passthrow.shape}")
            x = torch.cat([passthrow, x], dim=1)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.activation(self.conv2_h(x))
        return x



class KITTIdecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]

        enc, dec = self.num_ch_enc, self.num_ch_dec
        self.block1 = Block(enc[4], dec[4], 3, 1, 0, dec[4] + enc[3], dec[4], 3, 1, 0)
        self.block2 = Block(dec[4], dec[3], 3, 1, 0, dec[3] + enc[2], dec[3], 3, 1, 0)
        self.block3 = Block(dec[3], dec[2], 3, 1, 0, dec[2] + enc[1], dec[2], 3, 1, 0)
        self.block4 = Block(dec[2], dec[1], 3, 1, 0, dec[1] + enc[0], dec[1], 3, 1, 0)
        self.block5 = Block(dec[1], dec[0], 3, 1, 0, dec[0],          dec[0], 3, 1, 0)

        self.conv0 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(16, 1, 3, padding=0))
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(32, 1, 3, padding=0))
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(64, 1, 3, padding=0))
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(128, 1, 3, padding=0))
        

    def forward(self, feats):
        x = feats[4] #  [B, 512, 6, 20]
        # print(f"test {x.shape}")
        x = self.block1(x, feats[3]) # [B, 256, 12, 40]
        x = self.block2(x, feats[2]) # [B, 128, 24, 80]
        d3 = x
        x = self.block3(x, feats[1]) # [B, 64, 48, 160]
        d2 = x
        x = self.block4(x, feats[0]) # [B, 32, 96, 320]
        d1 = x
        x = self.block5(x) # [B, 16, 192, 640]
        d0 = x
        
        return {
            0: torch.sigmoid(self.conv0(d0)),
            1: torch.sigmoid(self.conv1(d1)),
            2: torch.sigmoid(self.conv2(d2)),
            3: torch.sigmoid(self.conv3(d3)),
        }
