# Written by AI (the only file in the project)
import numpy as np
import glob
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt



def read_calib_file(path):
    values = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            try:
                values[key.strip()] = np.array([float(x) for x in rest.split()])
            except ValueError:
                continue
    return values


def load_K(calib_dir):
    values = read_calib_file(f"{calib_dir}/calib_cam_to_cam.txt")
    P = values["P_rect_02"].reshape(3, 4)
    return P[:3, :3].astype(np.float64)


def load_P_rect(calib_dir):
    values = read_calib_file(f"{calib_dir}/calib_cam_to_cam.txt")
    return values["P_rect_02"].reshape(3, 4)                    # (3, 4)


def load_R_rect(calib_dir):
    values = read_calib_file(f"{calib_dir}/calib_cam_to_cam.txt")
    R_rect = np.eye(4)
    R_rect[:3, :3] = values["R_rect_00"].reshape(3, 3)
    return R_rect                                               # (4, 4)


def load_velo_to_cam(calib_dir):
    values = read_calib_file(f"{calib_dir}/calib_velo_to_cam.txt")
    velo_to_cam = np.eye(4)
    velo_to_cam[:3, :3] = values["R"].reshape(3, 3)
    velo_to_cam[:3, 3] = values["T"]
    return velo_to_cam                                          # (4, 4)


def load_velo_to_img(calib_dir):
    return load_P_rect(calib_dir) @ load_R_rect(calib_dir) @ load_velo_to_cam(calib_dir)  # (3, 4)



def lidar_to_depth(bin_path, velo_to_img, height, width):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    pts = pts[pts[:, 0] > 0]
    pts[:, 3] = 1.0

    proj = pts @ velo_to_img.T
    d = proj[:, 2]
    u = np.round(proj[:, 0] / d).astype(int)
    v = np.round(proj[:, 1] / d).astype(int)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (d > 0)
    u, v, d = u[inside], v[inside], d[inside]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    np.minimum.at(depth, (v, u), d)
    depth[np.isinf(depth)] = 0
    return depth


def generate_gt(root, drives):
    for date, drive in drives:
        velo_to_img = load_velo_to_img(f"{root}/{date}/{date}_calib")
        img_dir = f"{root}/{date}/{drive}/image_02/data"
        velo_dir = f"{root}/{date}/{drive}/velodyne_points/data"
        out_dir = Path(f"{root}/{date}/{drive}/gt_depth")
        out_dir.mkdir(exist_ok=True)

        n = len(glob.glob(f"{img_dir}/*.png"))
        for i in range(1, n - 1):
            bin_path = f"{velo_dir}/{i:010d}.bin"
            if not Path(bin_path).exists():
                print(f"missing {bin_path}, skipped")
                continue
            width, height = Image.open(f"{img_dir}/{i:010d}.png").size
            depth = lidar_to_depth(bin_path, velo_to_img, height, width)
            np.save(out_dir / f"{i:010d}.npy", depth)

        print(f"{drive}: saved depth maps to {out_dir}")


def save_overlay(root, date, drive, i, out_path="gt_overlay.png"):
    rgb = Image.open(f"{root}/{date}/{drive}/image_02/data/{i:010d}.png").convert("RGB")
    depth = np.load(f"{root}/{date}/{drive}/gt_depth/{i:010d}.npy")
    v, u = np.nonzero(depth)

    print(f"shape {depth.shape}, valid pixels {len(v) / depth.size:.1%}, "
          f"depth {depth[v, u].min():.1f} - {depth[v, u].max():.1f} m")

    plt.figure(figsize=(15, 5))
    plt.imshow(rgb)
    plt.scatter(u, v, c=depth[v, u], cmap="turbo", s=1)
    plt.colorbar(label="depth (m)")
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import src.dataloaders

    root = "data/kitti"
    train, val, test = src.dataloaders.make_splits(root)
    generate_gt(root, val + test)
    save_overlay(root, *val[0], i=5)