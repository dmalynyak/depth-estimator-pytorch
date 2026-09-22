import torch
import pytest
import math
from src.losses import _get_valid_mask, _get_L1_lin_loss, _get_grad_loss, NYULoss
import src

@pytest.fixture
def device():
    return torch.device("cpu")

def test_valid_mask():
    gt = torch.tensor([
        [0.0, 5.0, 10.5],
        [float('nan'), float('inf'), 10.0]
    ])
    
    mask, el_num, val_el_num = _get_valid_mask(gt)
    
    assert el_num == 6
    assert val_el_num == 2 
    
    expected_mask = torch.tensor([
        [False, True, False],
        [False, False, True]
    ])
    assert torch.equal(mask, expected_mask)

def test_l1_lin_loss(device):
    gt = torch.tensor([
        [0.0, 5.0], 
        [float('nan'), 2.0] ], device=device)
    
    pred = torch.tensor([
        [99.0, 4.0],
        [99.0, 4.0] ], device=device, requires_grad=True)
    

    loss = _get_L1_lin_loss(gt, pred, device=device)
    
    assert math.isclose(loss.item(), 1.5)
    assert loss.requires_grad == True

def test_loss_cls(device):

    gt = torch.tensor([
        [0.0, 5.0], 
        [float('nan'), 2.0] ], device=device)
    
    pred = torch.tensor([
        [99.0, 4.0],
        [99.0, 4.0] ], device=device, requires_grad=True)

    loss = NYULoss(1)

    tensor_value = loss(pred, gt, device=device)
    grad = _get_grad_loss(gt, pred, device=device)

    assert math.isclose((tensor_value - grad).item(), 1.5)
    assert tensor_value.requires_grad == True


def test_grad_loss_is_zero_on_perfect_pred(device):
    gt = torch.tensor([
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 0.0],
        [float('nan'), 0.5, 1.0]
    ], device=device)
    pred = gt.clone().detach().requires_grad_(True)

    loss = _get_grad_loss(gt, pred, device=device)
    assert math.isclose(loss.item(), 0.0)    

def test_photometric_loss():
    B, H, W = 2, 192, 640

    batch = {
        "imgs": torch.rand(B, 3, 3, H, W),
        "K": torch.tensor([[371., 0, 320], [0, 371., 96], [0, 0, 1]]).repeat(B, 1, 1),
    }
    batch["inv_K"] = torch.inverse(batch["K"])

    disps = {s: torch.rand(B, 1, H >> s, W >> s, requires_grad=True) for s in range(4)}
    T = torch.eye(4).repeat(B, 1, 1)

    loss = src.PhotometricLoss()(batch, disps, T, T, "cpu")

    assert loss.shape == (), f"must be scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"got {loss}"
    assert loss.requires_grad

    loss.backward()
    assert disps[0].grad is not None and disps[0].grad.abs().sum() > 0