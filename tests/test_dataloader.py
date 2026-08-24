import torch
import src


def test_depth_dataset_loader():
    dataset = src.DepthNYUDataset(data_dir="tests/test_data", split="train")
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for rgb, depth in test_loader:

        # shape tests
        assert rgb.shape == (1, 3, 240, 320), f"must be rgb(1, 3, 240, 320), got {rgb.shape}"
        assert depth.shape == (1, 1, 240, 320), f"must be depth(1, 1, 240, 320), got {depth.shape}"

        # dtype tests
        assert rgb.dtype == torch.float32, f"must be rgb(torch.float32), got {rgb.dtype}"
        assert depth.dtype == torch.float32, f"must be depth(torch.float32), got {depth.dtype}"

        # value tests
        assert not torch.isnan(depth).any(), "must be depth values not NaN"
        assert not torch.isinf(depth).any(), "must be depth values not Inf"

        assert torch.min(rgb) >= -3, f"must be rgb values in [-3, 3], got min {torch.min(rgb)}"
        assert torch.max(rgb) <= 3, f"must be rgb values in [-3, 3]. got max {torch.max(rgb)}"
        assert torch.min(depth) >= 0.0, f"must be depth >= 0.0, got {torch.min(depth)}"
        assert torch.max(depth) <= 10.0, f"must be depth <= 10.0, got {torch.max(depth)}"
