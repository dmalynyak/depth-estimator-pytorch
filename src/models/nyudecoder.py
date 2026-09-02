# encoder returns feats:
#  
# [B, 64, 240, 320]
#  [B, 64, 120, 160] 
#  [B, 128, 60, 80] 
#  [B, 256, 30, 40]
#  [B, 512, 15, 20]

import torch.nn as nn
import torch.nn.functional

class Block(nn.Module):
    def __init__(self, in_channels1, out_channels1, kernel_size1, stride1, padding1, in_channels2, out_channels2, kernel_size2, stride2, padding2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels1, out_channels=out_channels1, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.conv2 = nn.Conv2d(in_channels=in_channels2, out_channels=out_channels2, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.activation = nn.ELU()

    def forward(self, x, passthrow=None):

        x = self.activation(self.conv1(x))
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        if passthrow is not None:
            x = torch.cat([passthrow, x], dim=1)
        x = self.activation(self.conv2(x))
        return x



class NYUdecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.block1 = Block(self.num_ch_enc[4], self.num_ch_dec[4], 3, 1, 1, self.num_ch_dec[4]*2, self.num_ch_dec[4], 3, 1, 1)
        self.block2 = Block(self.num_ch_enc[3], self.num_ch_dec[3], 3, 1, 1, self.num_ch_dec[3]*2, self.num_ch_dec[3], 3, 1, 1)
        self.block3 = Block(self.num_ch_enc[2], self.num_ch_dec[2], 3, 1, 1, self.num_ch_dec[2]*2, self.num_ch_dec[2], 3, 1, 1)
        self.block4 = Block(64, 64, 3, 1, 1, 128, 32, 3, 1, 1)
        self.block5 = Block(32, 16, 3, 1, 1, 16, 16, 3, 1, 1)

        self.head = nn.Conv2d(in_channels=16, out_channels=1, stride=1, kernel_size=3, padding=1)
        

    def forward(self, feats):
        x = feats[4] #  [B, 512, 15, 20]
        x = self.block1(x, feats[3]) # [B, 256, 30, 40]
        x = self.block2(x, feats[2]) # [B, 128, 60, 80]
        x = self.block3(x, feats[1]) # [B, 64, 120, 160]
        x = self.block4(x, feats[0]) # [B, 32, 240, 320]
        x = self.block5(x) # [B, 16, 480, 640]

        x = self.head(x)
        return torch.sigmoid(x)




        # x = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1) #  [B, 256, 15, 20]
        # x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest') #  [B, 256, 30, 40]
        # x = torch.cat([fetures[2], x], dim=1) #  [B, 512, 30, 40]
        # x = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1) #  [B, 128, 30, 40]
        # x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest') #  [B, 128, 60, 80]
        # x = torch.cat([fetures[1], x], dim=1) #  [B, 256, 60, 80]