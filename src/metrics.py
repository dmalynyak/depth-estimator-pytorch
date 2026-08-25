import torch
import src.utils

def get_RMSE(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    return torch.sqrt(torch.mean( (pred - gt) ** 2) )

def get_RMSE_log(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    return torch.sqrt(torch.mean( (torch.log(pred) - torch.log(gt))** 2) )

def get_AbsRel(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    return torch.mean(torch.abs(pred - gt) / gt)

def get_del1(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    mask = (torch.max(pred/gt, gt/pred) < 1.25).float()
    return mask.mean()

def get_del2(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    mask = (torch.max(pred/gt, gt/pred) < 1.25**2).float()
    return mask.mean()

def get_del3(pred, gt, valid_mask):
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    mask = (torch.max(pred/gt, gt/pred) < 1.25**3).float()
    return mask.mean()

# applyies eigen crop, condition 0<gt<=10, condition gt is not Nan/Inf
def get_valid_mask(gt):
    
    eigen_mask = torch.zeros_like(gt)
    eigen_mask[..., 45:471, 41:601] = 1 # eigen mask
    
    mask = (gt > 0.0) & (gt <= 10)
    mask = mask & ~torch.isnan(gt) & ~torch.isinf(gt)

    return eigen_mask.bool() & mask

# must be upscaled
def get_metrics(pred, gt):

    assert pred.shape[-1] in (320, 640), f"width of tensor must be 320 or 640, got {pred.shape[-1]}"

    if pred.shape[-1] == 320:
        pred = src.utils.upsample_x2(pred)
        print(f"shape after upscaling {pred.shape}")
    if gt.shape[-1] == 320:
        gt = src.utils.upsample_x2(gt)
        print(f"shape after upscaling {pred.shape}")

    pred = pred.clamp(1e-3, 10.0)
    valid_mask = get_valid_mask(gt)
    print(f"shape gt {gt.shape}")

    return {
        "parameters": "Eigen crop, gt range(0,10], natural log, prediction upsample, median scsaling = False",
        "rmse": get_RMSE(pred, gt, valid_mask),
        "rmse_log": get_RMSE_log(pred, gt, valid_mask),
        "abs_rel": get_AbsRel(pred, gt, valid_mask),
        "d1": get_del1(pred, gt, valid_mask),
        "d2": get_del2(pred, gt, valid_mask),
        "d3": get_del3(pred, gt, valid_mask)
        }

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


# give final eval metrics 
# for rgb, gt in test_loader:

#     pred = model(rgb)
#     batch_metrics = src.metrics.get_metrics(pred, gt)
    
#     batch_size = gt.size(0) 

#     for key, value in batch_metrics.items():
#         if isinstance(value, torch.Tensor):
#             meters[key].update(value.item(), batch_size)

# print(f"Parameters: {batch_metrics['parameters']}")
# for key, meter in meters.items():
#     print(f"{key}: {meter.avg:.4f}")