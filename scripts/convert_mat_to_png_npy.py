import json
import os
import h5py
import numpy as np
import scipy.io
from PIL import Image

MAT_PATH = "data/nyu/nyu_depth_v2_labeled.mat"
SPLITS_PATH = "data/nyu/splits.mat"
OUT_ROOT = "data/nyu/"


def read_matlab_string(f, ref):
    return "".join(chr(int(c)) for c in f[ref][:].flatten())


def main():
    os.makedirs(f"{OUT_ROOT}/train", exist_ok=True)
    os.makedirs(f"{OUT_ROOT}/test", exist_ok=True)


    splits = scipy.io.loadmat(SPLITS_PATH)
    train_idx = set((splits["trainNdxs"].flatten() - 1).tolist()) # 1-based -> 0-based
    test_idx = set((splits["testNdxs"].flatten() - 1).tolist())


    f = h5py.File(MAT_PATH, "r")

    n = f["images"].shape[0]

    train_sum = 0.0
    train_px = 0

    scene_types = []
    counts = {"train": 0, "test": 0}

    for i in range(n):
        if i in train_idx:
            split = "train"
        elif i in test_idx:
            split = "test"
        else:
            raise RuntimeError(f"index {i} in neither split")

        rgb_path = f"{OUT_ROOT}/{split}/{i:05d}_rgb.png"
        depth_path = f"{OUT_ROOT}/{split}/{i:05d}_depth.npy"

        if os.path.exists(rgb_path) and os.path.exists(depth_path):
            counts[split] += 1
            if split == "train":
                d = np.load(depth_path)
                train_sum += float(d.sum())
                train_px += d.size
            scene_types.append(read_matlab_string(f, f["sceneTypes"][0][i]))
            continue

        rgb = f["images"][i].transpose(2, 1, 0)          # (480, 640, 3) uint8

        # depths was (480, 640, 1449) = (H, W, N) -> (N, W, H),
        depth = f["depths"][i].T.astype(np.float32) # (480, 640) meters

        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)

        counts[split] += 1
        if split == "train":
            train_sum += float(depth.sum())
            train_px += depth.size

        scene_types.append(read_matlab_string(f, f["sceneTypes"][0][i]))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")

    f.close()

    train_mean_depth = train_sum / train_px

    meta = {
        "n_train": counts["train"],
        "n_test": counts["test"],
        "height": 480,
        "width": 640,
        "depth_units": "meters",
        "depth_source": "depths (inpainted, no holes)",
        "train_mean_depth": round(train_mean_depth, 6),
        "scene_types": scene_types,
    }
    with open(f"{OUT_ROOT}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    n_train_files = len([x for x in os.listdir(f"{OUT_ROOT}/train") if x.endswith("_rgb.png")])
    n_test_files = len([x for x in os.listdir(f"{OUT_ROOT}/test") if x.endswith("_rgb.png")])

    print("dataset conversion from .mat to .png/.npy complete")


if __name__ == "__main__":
    main()