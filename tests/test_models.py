import torch
import pytest
# Замените на ваши реальные импорты
from src.models.resnet18 import Resnet18Encoder
from src.models.nyudecoder import NYUdecoder
from src import NYUmodel 

@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 480, 640)

def test_encoder_output(dummy_input):
    encoder = Resnet18Encoder(pretrained=False)
    feats = encoder(dummy_input)
    
    assert len(feats) == 5
    assert feats[1].shape == (2, 64, 120, 160)
    assert feats[4].shape == (2, 512, 15, 20)

def test_full_model_forward(dummy_input):
    model = NYUmodel()
    out = model(dummy_input)

    assert out.shape == (2, 1, 480, 640), f"wrong shape: {out.shape}"

    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)