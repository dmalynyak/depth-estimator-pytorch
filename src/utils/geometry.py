import torch
import torch.nn.functional as F

def upsample_x2(tensor, mode="bilinear"):

    is_3d = tensor.dim() == 3
    
    # interpolate needs (B, C, H, W)
    if is_3d:
        tensor = tensor.unsqueeze(0)
        
    upsampled = F.interpolate(tensor, scale_factor=2.0, mode=mode, align_corners=False)
    
    if is_3d:
        upsampled = upsampled.squeeze(0)
        
    return upsampled