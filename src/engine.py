import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import src

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

    def train_epoch(self, epoch):
        self.model.train()
        loss_value = 0.0

        pbar = tqdm(self.train_loader, desc= f"epoch {epoch}", leave=False)
        for i, (rgbs, depths) in enumerate(pbar):
            rgbs = rgbs.to(self.device)
            depths = depths.to(self.device)
            self.optimizer.zero_grad()

            predictions = self.model(rgbs)
            # print(f"pred: {predictions.shape} depth: {depths.shape}")
            loss = self.criterion(predictions, depths, self.device)
            loss_value += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()

        return loss_value / len(self.train_loader)

    @torch.no_grad
    def validate(self, epoch):
        self.model.eval()
        metrics_sum = None
        loss_value = 0.0

        pbar = tqdm(self.val_loader, desc= f"validation after epoch {epoch}", leave=False)
        for i, (rgbs, depths) in enumerate(pbar):
            rgbs = rgbs.to(self.device)
            depths = depths.to(self.device)

            predictions = self.model(rgbs)
            val_loss = self.criterion(predictions, depths, self.device)
            loss_value += val_loss.item()

            batch_metrics = src.get_metrics(predictions, depths)
            metrics_sum = self.logger.log_val_metrics_add(batch_metrics, metrics_sum)

        metrics = self.logger.log_val_metrics_devide_batches(metrics_sum, len(self.val_loader))
        metrics.update({"epoch": epoch})
       

        return metrics, loss_value / len(self.val_loader)




    def fit(self, epochs):
        abs_rel_best = 1.0
        tb_writer = SummaryWriter(log_dir=self.logger.log_path.replace(".csv", "_tb"))

        for epoch in range(epochs):    
            train_loss = self.train_epoch(epoch)
            metrics, val_loss = self.validate(epoch)
            metrics.update({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "saved": "True" if metrics["abs_rel"] < abs_rel_best else "-"
                })

            if metrics["abs_rel"] < abs_rel_best:
                abs_rel_best = metrics["abs_rel"]
                self.logger.log_save_weights(metrics)

            self.logger.log_val_metrics_write(metrics)

            tb_writer.add_scalars('losses', {'train': train_loss, 'val': val_loss}, epoch)
            tb_writer.add_scalar('metrics abs_rel', metrics["abs_rel"], epoch)
            tb_writer.add_scalar('metrics d1', metrics["d1"], epoch)

