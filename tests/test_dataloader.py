import torch
import src
import numpy as np

import src
from src.dataloaders import KittiJitter



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


import torch
import numpy as np

def test_kitti_dataset_loader_shapes_dtypes():
    drives = [("2011_09_26", "0001")]
    dataset = src.DepthKITTIDataset(root="tests/test_data/test_data_kitti", drives=drives, height=192, width=640)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    for batch in test_loader:
        assert isinstance(batch, dict), f"got {type(batch)}"
        expected_keys = {"imgs", "imgs_augmentated", "K", "inv_K"}

        assert batch["imgs"].shape == (2, 3, 3, 192, 640), f"imgs shape: {batch['imgs'].shape}"
        assert batch["imgs_augmentated"].shape == (2, 3, 3, 192, 640), f"got: {batch['imgs_augmentated'].shape}"
        assert batch["K"].shape == (2, 3, 3), f"got: {batch['K'].shape}"
        assert batch["inv_K"].shape == (2, 3, 3), f"got: {batch['inv_K'].shape}"

        assert batch["imgs"].dtype == torch.float32, f"float32 got: {batch['imgs'].dtype}"
        assert batch["imgs_augmentated"].dtype == torch.float32, f"float32 got: {batch['imgs_augmentated'].dtype}"
        assert batch["K"].dtype == torch.float32, f"float32 got: {batch['K'].dtype}"
        assert batch["inv_K"].dtype == torch.float32, f"float32 got: {batch['inv_K'].dtype}"
        break