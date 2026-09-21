from src.utils import get_warped_t_from_t1
from src.utils.kitti_warping import _get_pix_coords
import torch, pytest

@pytest.fixture
def device():
    return torch.device("cpu")

def test_warp_identity(device):
    B, H, W = 2, 192, 640

    K = torch.tensor([[371., 0, 320], [0, 371., 96], [0, 0, 1]]).repeat(B, 1, 1)
    inv_K = torch.inverse(K)
    T = torch.eye(4).repeat(B, 1, 1)

    depth = torch.rand(B, 1, H, W) * 50 + 1
    rgb = torch.rand(B, 3, H, W)

    out = get_warped_t_from_t1(K, inv_K, T, depth, rgb, device=device)

    assert out.shape == (B, 3, H, W), f"got {out.shape}"
    assert torch.allclose(out, rgb, atol=1e-3), f"not the same"