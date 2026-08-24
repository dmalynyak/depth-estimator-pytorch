import torch
import pytest
import src.utils
import src.metrics
import math

from src.metrics import (
    get_RMSE, get_RMSE_log, get_AbsRel,
    get_del1, get_del2, get_del3,
    get_valid_mask,
)
 
 
def full(shape, value):
    return torch.full(shape, float(value), dtype=torch.float64)


def test_perfect_prediction():

    gt = torch.rand(32, 32, dtype=torch.float64) * 9 + 0.7
    all_valid_mask = torch.ones_like(gt, dtype=torch.bool)
    assert src.metrics.get_RMSE(gt.clone(), gt, all_valid_mask) == pytest.approx(0.0)
    assert src.metrics.get_RMSE_log(gt.clone(), gt, all_valid_mask) == pytest.approx(0.0)
    assert src.metrics.get_AbsRel(gt.clone(), gt, all_valid_mask) == pytest.approx(0.0)
    assert src.metrics.get_del1(gt.clone(), gt, all_valid_mask) == pytest.approx(1.0)

def test_rmse_constant_offset():
    gt = torch.rand(32, 32, dtype=torch.float64) * 8 + 1
    assert get_RMSE(gt + 0.5, gt, torch.ones_like(gt, dtype=torch.bool)) == pytest.approx(0.5)
 
 
def test_abs_rel_divides_by_gt_not_pred():
    gt = full((16, 16), 4.0)
    all_valid_mask = torch.ones_like(gt, dtype=torch.bool)
    assert get_AbsRel(gt * 2, gt, all_valid_mask) == pytest.approx(1.0)
    assert get_AbsRel(gt / 2, gt, all_valid_mask) == pytest.approx(0.5)


def test_deltas():
    gt = full((100,), 4.0)
    pred = gt.clone()
    pred[:30] *= 1.1
    pred[30:] *= 1.9
    m = torch.ones_like(gt, dtype=torch.bool)
    assert get_del1(pred, gt, m) == pytest.approx(0.30)
    assert get_del2(pred, gt, m) == pytest.approx(0.30)
    assert get_del3(pred, gt, m) == pytest.approx(1.00)
 


def test_valid_mask_shape_and_region():
    gt = full((480, 640), 5.0)
    m = get_valid_mask(gt)
 
    assert m.shape == gt.shape, f"expected {tuple(gt.shape)}, got {tuple(m.shape)}"
    assert m.dtype == torch.bool
 
    assert not m[:45].any()
    assert not m[471:].any()
    assert not m[:, :41].any()
    assert not m[:, 601:].any()
    assert m[45:471, 41:601].all()
 
 
def test_valid_mask_rejects_out_of_range_gt():
    gt = full((480, 640), 5.0)
    gt[100, 100] = 0.0
    gt[100, 101] = -1.0
    gt[100, 102] = 11.0
    gt[100, 103] = float("nan")
    gt[100, 104] = float("inf")
 
    m = get_valid_mask(gt)
    assert not m[100, 100:105].any(), "problem here"
    assert m[200, 200]
 