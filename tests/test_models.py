import torch
import pytest
from src.models.resnet18 import Resnet18Encoder
from src.models.nyudecoder import NYUdecoder
from src.models.kittidecoder import KITTIdecoder
from src.models.posenetdecoder import PosenetDecoder
from src.models.posenetencoder import PosenetResnet18Encoder
from src import NYUmodel
import src

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

    assert torch.all(out >= 0.0) and torch.all(out <= 10.0)


def test_kitti_depthbet():
    enc = Resnet18Encoder(pretrained=False)
    dec = KITTIdecoder()
    mod = src.KITTIdepthNET()

    out = dec(enc(torch.randn(2, 3, 192, 640)))
    mod_out = mod(torch.randn(2, 3, 192, 640))

    for s in range(4):
        d = out[s]
        assert d.shape == (2, 1, 192 >> s, 640 >> s), f"got {d.shape}"
        assert 0.0 <= d.min() and d.max() <= 1.0, f"not in [0,1]"
    for s in range(4):
        d1 = mod_out[s]
        assert d1.shape == (2, 1, 192 >> s, 640 >> s), f"got {d1.shape}"
        assert 0.0 <= d1.min() and d1.max() <= 1.0, f"not in [0,1]"

def test_kitti_posenet():
    enc = PosenetResnet18Encoder(pretrained=False)
    dec = PosenetDecoder()
    mod = src.KITTIposeNET()

    x = torch.randn(2, 6, 192, 640)
    y = torch.randn(2, 3, 192, 640)
    z = torch.randn(2, 3, 192, 640)

    rot, trans = dec(enc(x))
    mod_rot, mod_trans = mod(y, z)

    for r, t in [(rot, trans), (mod_rot, mod_trans)]:
        assert r.shape == (2, 3), f"got {r.shape}"
        assert t.shape == (2, 3), f"got {t.shape}"
        assert r.abs().max() < 0.1, f"{r.abs().max()}"