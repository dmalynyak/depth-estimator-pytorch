import torch
import matplotlib.pyplot as plt

import src.dataloaders

def visualize_1chw(rgb, depth, depth_pred=None):

    assert rgb.shape == (1, 3, 240, 320), f"must be rgb(1, 3, 240, 320), got {rgb.shape}"
    assert depth.shape == (1, 1, 240, 320), f"must be depth(1, 1, 240, 320), got {depth.shape}"
    assert rgb.dtype == torch.float32, f"must be rgb(torch.float32), got {rgb.dtype}"
    assert depth.dtype == torch.float32, f"must be depth(torch.float32), got {depth.dtype}"
    assert not torch.isnan(depth).any(), "must be depth values not NaN"
    assert not torch.isinf(depth).any(), "must be depth values not Inf"

    rgb, depth = src.dataloaders.denormalize_image_net(rgb, depth)

    assert torch.min(rgb) >= 0.0, "must be rgb values in [0.0, 1.0]"
    assert torch.max(rgb) <= 1.0, "must be rgb values in [0.0, 1.0]"
    assert torch.min(depth) >= 0.0, "must be depth >= 0.0"
    assert torch.max(depth) <= 10.0, "must be depth <= 10.0"

    rgb = rgb.squeeze(0)
    depth = depth.squeeze(0)

    if depth_pred is None:
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))

        ax[0].imshow(rgb.permute(1, 2, 0))
        ax[0].set_title(f"RGB:")
        ax[0].axis("off")

        im = ax[1].imshow(depth.permute(1, 2, 0), cmap="plasma", vmin=0, vmax=10)
        ax[1].set_title("Depth GT")
        ax[1].axis("off")
        plt.colorbar(im, ax=ax[1], fraction=0.046)

    elif depth_pred is not None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        ax[0].imshow(rgb.permute(1, 2, 0))
        ax[0].set_title(f"RGB:")
        ax[0].axis("off")

        im = ax[1].imshow(depth.permute(1, 2, 0), cmap="plasma", vmin=0, vmax=10)
        ax[1].set_title("Depth GT")
        ax[1].axis("off")
        plt.colorbar(im, ax=ax[1], fraction=0.046)

        im = ax[2].imshow(depth.permute(1, 2, 0), cmap="plasma", vmin=0, vmax=10)
        ax[2].set_title("Depth prediction")
        ax[2].axis("off")
        plt.colorbar(im, ax=ax[2], fraction=0.046)

    plt.show()

def draw_prediction(rgb, depth_pred, save_path):

    rgb, depth_pred = src.dataloaders.denormalize_image_net(rgb, depth_pred)
    rgb = rgb.squeeze(0)
    #depth_pred = depth_pred.squeeze(0)
    print(f"rgb: {rgb.shape}")
    print(f"depth: {depth_pred.shape}")

    rgb_plot = rgb.permute(1, 2, 0).detach().cpu().numpy()
    depth_plot = depth_pred.permute(1, 2, 0).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].imshow(rgb_plot)
    ax[0].set_title("RGB")
    ax[0].axis("off")

    im = ax[1].imshow(depth_plot, cmap="plasma", vmin=0, vmax=10)
    ax[1].set_title("Depth Prediction")
    ax[1].axis("off")
    plt.colorbar(im, ax=ax[1], fraction=0.046)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close(fig)