import torch
import numpy as np
from tqdm import tqdm
import os, csv
import src
 

def log_val_metrics_add_test(new_metrics, saved_metrics=None):

    if saved_metrics is None:
        return new_metrics
    
    for key in saved_metrics:
        if key in ["parameters", "saved"]:
            continue
        saved_metrics[key] += new_metrics[key]

    return saved_metrics

def log_val_metrics_devide_batches_test(metrics, loader_len):

    assert metrics is not None, f"got empty metrics dictionary"

    for key in metrics:
        if key in ["parameters", "saved"]:
            continue
        metrics[key] = metrics[key] / loader_len

    return metrics
 
@torch.no_grad
def test(model_path):
    device = torch.device("cuda")
    model = src.KITTIdepthNET().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['depth_net'])

    model.eval()

    train_drives, val_drives, test_drives = src.dataloaders.make_splits("data/kitti")
    test_dataset = src.DepthKITTIDataset("data/kitti", test_drives, train=False)
    train_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True)

    metrics_sum = None

    pbar = tqdm(train_loader, leave=False, disable=True)
    for i, load in enumerate(pbar):
        imgs = load["imgs_augmented"].to(device) # (B, 3, 3, H, W)
        gts = load["gt_depth"][0].numpy()

        img_clear_t = imgs[:, 1]
        disp_pred = model(img_clear_t)[0]
        disp_pred = torch.nn.functional.interpolate(disp_pred, size=gts.shape, mode="bilinear", align_corners=False)
        _, depth = src.utils.disp_to_depth(disp_pred)
        pred = depth[0, 0].cpu().numpy()

        H, W = gts.shape[-2], gts.shape[-1]
        mask = (gts > 1e-3) & (gts < 80)
        crop = np.zeros_like(mask)
        # Garg crop to match Lidar res and rgb red
        crop[int(0.40810811 * H):int(0.99189189 * H), int(0.03594771 * W):int(0.96405229 * W)] = True
        mask = mask & crop

        gt_m = gts[mask]
        pred_m = pred[mask]
        ratio = np.median(gt_m) / np.median(pred_m)
        pred_m *= ratio
        pred_m = np.clip(pred_m, 1e-3, 80)

        batch_metrics = src.kitti_get_metrics(pred_m, gt_m)
        batch_metrics.update({"scale" : np.abs(ratio)})
        metrics_sum = log_val_metrics_add_test(batch_metrics, metrics_sum)

    metrics = log_val_metrics_devide_batches_test(metrics_sum, len(train_loader))

    return metrics

def log_test_metrics_write(metrics, log_path):

    file_exists = os.path.exists(f"{log_path}")

    metrics_only = {k: v for k, v in metrics.items() if k != "parameters"}
    ordered_fieldnames = metrics_only

    with open(f"{log_path}", mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        
        if not file_exists:
            if "parameters" in metrics:
                f.write(f"# Parameters: {metrics['parameters']}\n")
            
            writer.writeheader()


        writer.writerow(metrics_only)



def main():
    model_path = 'excess/kitti/best5.pt'
    metrics = test(model_path)

    log_test_metrics_write(metrics, "excess/kitti/test_metrics.csv")


if __name__ == "__main__":
    main()

