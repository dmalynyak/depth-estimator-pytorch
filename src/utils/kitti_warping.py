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
def warp_3d_to_pixs(self, points, K):
    cam = K @ points
    return cam[:2] / cam[2] # (x, y)

def warp_pixs_to_3d(self, pixels, depth_pred, inv_K):
    rays = inv_K @ pixels
    return rays * depth_pred