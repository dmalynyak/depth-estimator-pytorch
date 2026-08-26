import torch
import pytest
import math
from src.losses import _get_valid_mask, _get_L1_lin_loss, NYULoss

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

    assert math.isclose(tensor_value.item(), 1.5)
    assert tensor_value.requires_grad == True
    
