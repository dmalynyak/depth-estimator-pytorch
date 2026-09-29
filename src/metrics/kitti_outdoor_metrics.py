import numpy as np
import src.utils

def get_RMSE(pred, gt):
    return np.sqrt(np.mean( (pred - gt) ** 2) ).item()

def get_RMSE_log(pred, gt):
    return np.sqrt(np.mean( (np.log(pred) - np.log(gt))** 2) ).item()

def get_AbsRel(pred, gt):
    return np.mean(np.abs(pred - gt) / gt).item()

def get_del1(pred, gt):
    max_ratio = np.maximum(pred / gt, gt / pred)
    mask = (max_ratio < 1.25).astype(float)
    return mask.mean().item()

def get_del2(pred, gt):
    max_ratio = np.maximum(pred / gt, gt / pred)
    mask = (max_ratio < 1.25**2).astype(float)
    return mask.mean().item()

def get_del3(pred, gt):
    max_ratio = np.maximum(pred / gt, gt / pred)
    mask = (max_ratio < 1.25**3).astype(float)
    return mask.mean().item()

# def get_scale_ratio(pred, gt):
#     return np.median(gt) / np.median(pred)

def kitti_get_metrics(pred, gt):

    return {
        "rmse": get_RMSE(pred, gt),
        "rmse_log": get_RMSE_log(pred, gt),
        "abs_rel": get_AbsRel(pred, gt),
        "d1": get_del1(pred, gt),
        "d2": get_del2(pred, gt),
        "d3": get_del3(pred, gt)
        # "scale": get_scale_ratio(pred, gt)
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
