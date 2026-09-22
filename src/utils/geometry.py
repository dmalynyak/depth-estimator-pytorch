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

def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled

    return scaled, depth

def _rot_from_axisangle(vec):
    # (B, 3) axis-angle to (B, 3, 3) rotation matrix
    angle = vec.norm(dim=1, keepdim=True) # (B, 1)
    axis = vec / (angle + 1e-7) # (B, 3) unit axis

    ca = torch.cos(angle) # (B, 1)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[:, 0:1] # (B, 1) each
    y = axis[:, 1:2]
    z = axis[:, 2:3]

    R = torch.cat([
        x*x*C + ca,    x*y*C - z*sa,  x*z*C + y*sa,
        y*x*C + z*sa,  y*y*C + ca,    y*z*C - x*sa,
        z*x*C - y*sa,  z*y*C + x*sa,  z*z*C + ca,
    ], dim=1) #(B, 9)

    return R.view(-1, 3, 3)


def pose_to_matrix(rot, trans, invert=False):
    # rot, trans: (B, 3) -> T: (B, 4, 4)
    R = _rot_from_axisangle(rot) # (B, 3, 3)
    t = trans.unsqueeze(-1) # (B, 3, 1)

    if invert:
        R = R.transpose(1, 2)
        t = -R @ t

    T = torch.eye(4, device=rot.device).repeat(rot.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3:4] = t
    return T