# intorpolation is made at the end of the model, so we presumably get 480x640 gt and pred

import torch
import torch.nn as nn

def _get_valid_mask(gt):
     
    mask = (gt > 0.0) & (gt <= 10)
    mask = mask & ~torch.isnan(gt) & ~torch.isinf(gt)
    el_num = mask.numel()
    val_el_num = mask.float().sum().item()

    return mask, el_num, val_el_num

def _get_L1_lin_loss(gt, pred, device):

    val_mask, el_num, val_el_num = _get_valid_mask(gt)

    # we need to keep gradient tree
    if val_el_num == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = torch.abs(gt[val_mask] - pred[val_mask]).sum() / val_el_num
    return loss # this returns as (1, ) tensor. It keeps gradient tree so .backward() can built gradient descent

class NYULoss(nn.Module):

    def __init__(self, l=0.1):
        super().__init__()
        self.l = l

    def forward(self, pred, gt, device):

        l1_lin = _get_L1_lin_loss(gt, pred, device=device)

        return self.l * l1_lin

