from src.train_engines import TrainerIndoor
import src
import torch
import src.utils

from torch.utils.data import Subset, DataLoader

def main(args):

    train_drives, val_drives, test_drives = src.dataloaders.make_splits("data/kitti")
    # print(train_drives)
    train_dataset = src.DepthKITTIDataset("data/kitti", train_drives)
    eval_dataset = src.DepthKITTIDataset("data/kitti", val_drives, train=False)
    # tiny = Subset(train_dataset, range(4))
    # loader = DataLoader(tiny, batch_size=4, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4, persistent_workers=True)

    device = src.utils.parse_device(args.device)
    log_path = args.log_path
    chkpt_path = args.chkpt_path
    DepthNet = src.KITTIdepthNET()
    PoseNet = src.KITTIposeNET()
    params = list(DepthNet.parameters()) + list(PoseNet.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = src.PhotometricLoss()
    logger = src.utils.Logger(DepthNet, log_path, chkpt_path, PoseNet, optimizer, scheduler)

    trainer = src.TrainerOutdoor(DepthNet, PoseNet, train_loader, eval_loader, criterion, optimizer, scheduler, logger, device)

    trainer.fit(epochs=100)

if __name__ == "__main__":
    args = src.utils.parse_args()
    print(args)
    main(args)