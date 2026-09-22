'''
As input loss function gets:
    1. DepthNet output for frame t: [B, 1, 192, 640][B, 1, 96, 320][B, 1, 48, 160][B, 1, 24, 80]
    2. PoseNet outputs: T for t-1 and T for t+1
    2. rgb t-1, t, t+1; K, inv_K from dataloader

Loss function does: 
    for step (dataloader: t-1, t, t+1  model: pred depths):
        1.1 map_prev = error(t-1, t) - photometric error output: (B,1,H,W)
        1.2 map_next = error(t+1, t) - photometric error output: (B,1,H,W)
        1.3 map = min(map_prev, map_next)

        for scale s in model prediction (s = 1, 2, 4, 8)
            2.0 upsample depth(disp) pred of t [B, 1, 192/s, 640/s] to [B, 1, 192, 640], convert disp to depth (comment in kitti_depthnet.py)
            2.1 warp depth of t [B, 1, 192, 640] and rgb of t-1 into rgb pred of t = warped t from t-1
            2.2 warp depth of t [B, 1, 192, 640] and rgb of t+1 into rgb pred of t = warped t from t+1
            2.3 warp_prev = error(warped t from t-1, t) - photometric error output: (B,1,H,W)
            2.4 warp_next = error(warped t from t+1, t) - photometric error output: (B,1,H,W)
            2.5 warp = min(warp_prev, warp_next) - output: (B,1,H,W)

            3.1 mask = warp < map 
            3.2 error_s = mean(warp * mask) (+ smth with smoothenest) - one float value
            
        loss = mean over every scale s
        loss.backward()
'''

'''
As input error function gets:
    1. (B, 3, H, W) warped frame for prediction or raw neighbour for identity mask
    2. target (B, 3, H, W) raw frame t
    all pictures are [0, 1]

Error function does:
    1. L1 part
        1.1 l1 = |pred - target| (B, 3, H, W)
        1.2 l1 = mean over 3 colours channels (B, 1, H, W)

    2. SSIM part, for one 3x3 windows
        2.1 mu = mean over 9 pixels one color channel(mu_pred, mu_target) - avg_pool in vectorized implementation
        2.2 var(sigma^2) = variance E(X^2) - E^2(X) (var_pred, var_target) - few avg_pools in vectorizes implementation
        2.3 cov = covariance beetwen pred and target E(XY) - E(x)E(Y) - few avg_pools in vectorized implementation
        
        3.1 SSIM(for one 3x3 one color) = (2 * mu_p * mu_t + C1)(2 * cov_pt + C2)
                                        -------------------------------------------     One value
                                          (mu_p^2 + mu_t^2 + C1)(var_p + var_t + C2) 

        3.2 SSIM_loss_color = clamp((1 - SSIM) / 2, 0, 1) - one value for every 3x3 value for every color channel
        3.3 SSIM_loss = mean over colors for every window 

        3.4 in vectorized implementation at this point we will have (B, 1, H, W) tensor

    3. error = 0.85 · ssim_loss + 0.15 · l1       (B, 1, H, W)
'''


import torch
import torch.nn as nn
import src.utils
import torch.nn.functional


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

        self.avpool   = nn.AvgPool2d(kernel_size = 3, stride=1)

        self.refl = nn.ReflectionPad2d(padding=1) # so we can calculate loss of border pixels

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):

        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.avpool(x)
        mu_y = self.avpool(y)

        var_x  = self.avpool(x ** 2) - mu_x ** 2
        var_y  = self.avpool(y ** 2) - mu_y ** 2
        covar_xy = self.avpool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * covar_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (var_x + var_y + self.C2)

        SSIM = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return torch.mean(SSIM, dim=1, keepdim=True)

class L1error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        l1 = torch.abs(x - y)
        l1 = torch.mean(l1, dim=1, keepdim=True)

        return l1

class PhotometricError(nn.Module):
    def __init__(self, alpha = 0.85):
        super().__init__()

        self.alpha = alpha
        self.ssim = SSIM()
        self.l1 = L1error()

    def forward(self, x, y):

        ssim = self.ssim(x, y)
        l1 = self.l1(x, y)

        return ( self.alpha * ssim + (1-self.alpha) * l1 )

# disparity (DepthNet output) is tensor with gradient tree, so gradient flows throw it and traines during trainig
# while image is clean tensor from dataload without gradient tree so here its just a parameter
def smoothness(disp, img):
    mean = disp.mean(dim=(2,3), keepdim=True) + 1e-7
    disp = disp / mean

    d_dx = (disp[..., :, :-1] - disp[..., :, 1:]).abs()
    d_dy = (disp[..., :-1, :] - disp[..., 1:, :]).abs()

    i_dx = (img[..., :, :-1] - img[..., :, 1:]).abs().mean(1, keepdim=True)
    i_dy = (img[..., :-1, :] - img[..., 1:, :]).abs().mean(1, keepdim=True)

    return (d_dx * torch.exp(-i_dx)).mean() + (d_dy * torch.exp(-i_dy)).mean()

class PhotometricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Error = PhotometricError()

    def forward(self, dataloader_out, depth_out, T_prev, T_next, device):
        rgbs_clean = dataloader_out["imgs"]
        K = dataloader_out["K"]
        inv_K = dataloader_out["inv_K"]
        clean_prev = rgbs_clean[:, 0]
        clean_t = rgbs_clean[:, 1]
        clean_next = rgbs_clean[:, 2]
        H, W = clean_t.shape[-2:]

        map_prev = self.Error(clean_prev, clean_t)
        map_next = self.Error(clean_next, clean_t)
        identity_map = torch.min(map_prev, map_next)
        identity_map = identity_map + torch.randn_like(identity_map) * 1e-5

        loss = 0.0
        for s in range(4):
            disp = torch.nn.functional.interpolate(depth_out[s], size=(H, W), mode="bilinear", align_corners=False)
            disp_scaled, depth = src.utils.disp_to_depth(disp) ### ???????????????????????????????????

            error_prev = self.Error(src.utils.get_warped_t_from_t1(clean_prev, depth, T_prev, K, inv_K, device), clean_t)
            error_next = self.Error(src.utils.get_warped_t_from_t1(clean_next, depth, T_next, K, inv_K, device), clean_t)
            error = torch.min(error_prev, error_next)

            mask = (error < identity_map).float()
            loss = loss + (error * mask).mean()
            loss = loss + 1e-3 * smoothness(disp_scaled, clean_t) / (2 ** s) # should contribute a little

        return loss / 4
            