import random
from pathlib import Path

# Eigen test split
EIGEN_TEST_DRIVES = {
    "2011_09_26_drive_0002_sync", "2011_09_26_drive_0009_sync",
    "2011_09_26_drive_0013_sync", "2011_09_26_drive_0020_sync",
    "2011_09_26_drive_0023_sync", "2011_09_26_drive_0027_sync",
    "2011_09_26_drive_0029_sync", "2011_09_26_drive_0036_sync",
    "2011_09_26_drive_0046_sync", "2011_09_26_drive_0048_sync",
    "2011_09_26_drive_0052_sync", "2011_09_26_drive_0056_sync",
    "2011_09_26_drive_0059_sync", "2011_09_26_drive_0064_sync",
    "2011_09_26_drive_0084_sync", "2011_09_26_drive_0086_sync",
    "2011_09_26_drive_0093_sync", "2011_09_26_drive_0096_sync",
    "2011_09_26_drive_0101_sync", "2011_09_26_drive_0106_sync",
    "2011_09_26_drive_0117_sync", "2011_09_28_drive_0002_sync",
    "2011_09_29_drive_0071_sync", "2011_09_30_drive_0016_sync",
    "2011_09_30_drive_0018_sync", "2011_09_30_drive_0027_sync",
    "2011_10_03_drive_0027_sync", "2011_10_03_drive_0047_sync",
}

VAL_DRIVES = {
    "2011_09_26_drive_0005_sync",
    "2011_09_26_drive_0091_sync",
}



def discover_drives(root):
    # return [(date, drive), ...]
    root = Path(root)
    drives = []
    for date_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for drive_dir in sorted(date_dir.glob("*_drive_*_sync")):
            if (drive_dir / "image_02" / "data").is_dir():
                drives.append((date_dir.name, drive_dir.name))
    return drives



def make_splits(root):
    all_drives = discover_drives(root)
    test  = [d for d in all_drives if d[1] in EIGEN_TEST_DRIVES]
    val   = [d for d in all_drives if d[1] in VAL_DRIVES]
    train = [d for d in all_drives if d[1] not in EIGEN_TEST_DRIVES | VAL_DRIVES]

    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)} drives")
    return train, val, test