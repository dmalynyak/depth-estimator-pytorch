# T must be reshaped before warping (geometry.py)

from tqdm import tqdm
import torch
import src
import matplotlib.pyplot as plt

class TrainerOutdoor:
    def __init__(self, DepthNet, PoseNet, train_loader, eval_loader, criterion, optimizer, scheduler, device):
        self.DepthNet = DepthNet.to(device)
        self.PoseNet = PoseNet.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, epoch):
        self.DepthNet.train()
        self.PoseNet.train()
        loss_value = 0.0

        pbar = tqdm(self.train_loader, desc= f"epoch {epoch}", leave=False, disable=True)
        for i, load in enumerate(pbar):
            imgs = load["imgs"].to(self.device) # (B, 3, 3, H, W)
            imgs_aug = load["imgs_augmentated"].to(self.device) # (B, 3, 3, H, W)
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

            if epoch % 50 == 0:
                fig, ax = plt.subplots(2, 1, figsize=(10, 9))
                ax[0].imshow(imgs[0, 1].permute(1, 2, 0).cpu())                  # RGB frame t
                ax[1].imshow(depth_pred[0][0, 0].detach().cpu(), cmap="magma")            # predicted disparity
                # ax[2].imshow(per_pixel[0, 0].detach().cpu(), cmap="viridis")     # remaining error
                for a in ax: a.axis("off")
                plt.savefig(f"debug_epoch{epoch}.png", bbox_inches="tight")
                plt.close()


        return loss_value / len(self.train_loader)

    def fit(self, epochs):
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            self.scheduler.step()
            if epoch % 50 == 0:
                print(f"train_loss {train_loss}, lr {self.optimizer.param_groups[0]['lr']:.1e}")


    @torch.no_grad
    def validate(self, epoch):
        pass