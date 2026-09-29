# T must be reshaped before warping (geometry.py)

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import src

class TrainerOutdoor:
    def __init__(self, DepthNet, PoseNet, train_loader, eval_loader, criterion, optimizer, scheduler, logger, device):
        self.DepthNet = DepthNet.to(device)
        self.PoseNet = PoseNet.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device

    def train_epoch(self, epoch):
        self.DepthNet.train()
        self.PoseNet.train()
        loss_value = 0.0

        pbar = tqdm(self.train_loader, desc= f"epoch {epoch}", leave=False, disable=False)
        for i, load in enumerate(pbar):
            imgs = load["imgs"].to(self.device) # (B, 3, 3, H, W)
            imgs_aug = load["imgs_augmented"].to(self.device) # (B, 3, 3, H, W)
            K = load["K"].to(self.device) # (B, 3, 3)
            inv_K = load["inv_K"].to(self.device)

            self.optimizer.zero_grad()

            img_aug_t = imgs_aug[:, 1, ...]
            img_aug_prev = imgs_aug[:, 0, ...]
            img_aug_next = imgs_aug[:, 2, ...]

            depth_pred = self.DepthNet(img_aug_t)
            rot_prev, tran_prev = self.PoseNet(img_aug_prev, img_aug_t)
            rot_next, tran_next = self.PoseNet(img_aug_t, img_aug_next)
            T_prev_pred = src.geometry.pose_to_matrix(rot_prev, tran_prev)
            T_next_pred = src.geometry.pose_to_matrix(rot_next, tran_next)

            loss = self.criterion(load, depth_pred, T_prev_pred, T_next_pred, self.device)
            loss_value += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

            loss.backward()

            self.optimizer.step()

        return loss_value / len(self.train_loader)


    @torch.no_grad
    def validate(self, epoch):
        self.DepthNet.eval()
        self.PoseNet.eval()
        metrics_sum = None

        pbar = tqdm(self.eval_loader, desc= f"epoch {epoch}", leave=False, disable=True)
        for i, load in enumerate(pbar):
            imgs = load["imgs_augmented"].to(self.device) # (B, 3, 3, H, W)
            gts = load["gt_depth"][0].numpy()

            img_clear_t = imgs[:, 1]
            disp_pred = self.DepthNet(img_clear_t)[0]
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
            metrics_sum = self.logger.log_val_metrics_add(batch_metrics, metrics_sum)

        metrics = self.logger.log_val_metrics_devide_batches(metrics_sum, len(self.eval_loader))
        metrics.update({"epoch": epoch})

        return metrics


    def fit(self, epochs):
        abs_rel_best = float('inf')
        # tb_writer = SummaryWriter(log_dir=self.logger.log_path.replace(".csv", "_tb"))

        for epoch in range(epochs):    
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            metrics = self.validate(epoch)
            metrics.update({
                "train_loss": train_loss,
                "saved": "True" if metrics["abs_rel"] < abs_rel_best else "-"
                })

            if metrics["abs_rel"] < abs_rel_best:
                abs_rel_best = metrics["abs_rel"]
                print(f"epoch: {epoch}, checkpoint saved")
                self.logger.log_save_weights(metrics)

            self.logger.log_val_metrics_write(metrics)

            # tb_writer.add_scalars('train loss', train_loss, epoch)
            # tb_writer.add_scalar('metrics abs_rel', metrics["abs_rel"], epoch)
            # tb_writer.add_scalar('metrics d1', metrics["d1"], epoch)