import numpy as np
from PIL import Image
import torch

'''
K is saved in calib file. line P_rect_02: 12 values (3, 4) last col is unused
K: 
fx 0  cx
0  fy cy 
0  0  1
all in pixel lenghts, so we need to adjust values for image resolution resizing
'''

# 3d are (3, N) (X Y Z) - real point 
# pixs are (3, N) (x, y, 1) - pixel on img
# all tensors must be torch for gradient flow
def warp_3d_to_pixs(points, K):
    cam = K @ points
    return cam[:2] / cam[2] # (x, y)

def warp_pixs_to_3d(pixels, depth_pred, inv_K):
    rays = inv_K @ pixels
    return rays * depth_pred


def _get_pix_coords(H, W):
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    pix_coords = torch.stack([u.flatten(), v.flatten(), torch.ones(H*W)])
    pix_coords = pix_coords.unsqueeze(0).repeat(B, 1, 1).float()   # (B, 3, H*W)

"""
K, inv_K   : (B, 3, 3)        already this shape
T          : (B, 4, 4)        needs reshaping
depth      : (B, 1, H, W)     pred from DepthNet
rgb_t1     : (B, 3, H, W)     from dataloader, [0, 1]
pix_coords : (B, 3, H*W)        grid coord, made before
returns    : (B, 3, H, W)     frame t with t+1 pixels
"""
def get_warped_t_from_t1(K, inv_K, T, depth, rgb_t1):

    B, _, H, W = rgb_t1.shape
    N = H * W

    pix_coords = _get_pix_coords(H, W)

    d = depth.flatten(start_dim=2) # (B, 1, H*W) depth in metres

    rays = inv_K @ pix_coords # (B, 3, H*W) rays 3D directions, z = 1

    pts = rays * d # (B, 3, H*W) rays 3D directions with depth, frame t
    ones = torch.ones(B, 1, N, device=d.device)
    pts = torch.cat([pts, ones], dim=1) # (B, 4, N)
    pts = T @ pts # (B, 4, N) 3D points, frame t+1

    cam = K @ pts[:, :3] # (B, 3, N) (u*z, v*z, z) cam pix coords
    uv = cam[:, :2] / (cam[:, 2:3] + 1e-7) # (B, 2, N) cam pix coords t+1

    uv = uv.view(B, 2, H, W) # (B, 2, H, W)
    uv = uv.permute(0, 2, 3, 1) # (B, H, W, 2)

    # (B, H, W, 2) normalized [-1, 1]
    uv = torch.stack([
        uv[..., 0] / (W - 1) * 2 - 1,
        uv[..., 1] / (H - 1) * 2 - 1,
    ], dim=-1)

    # (B, 3, H, W)
    return F.grid_sample(rgb_t1, uv, mode="bilinear", padding_mode="border", align_corners=True)