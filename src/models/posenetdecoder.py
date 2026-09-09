import torch.nn as nn

# PoseNet encoder outputs (B, 512, 6, 20)
# PoseNet decoder outputs (B, 6) (pith, yaw, roll, left/right, down/up, backward/forward)
class PosenetDecoder(nn.Module):
    def __init__(self, num_ch_enc=512):
        super().__init__()
        self.conv1 = nn.Conv2d(num_ch_enc, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 6, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x) # (B, 6, 6, 20)

        x = x.mean(dim=(2, 3)) * 0.01 # (B, 6)

        return x[:, :3], x[:, 3:] # rotation(x, y, z), translation(x, y, z)