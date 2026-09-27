from src.train_engines import TrainerIndoor
import src
import torch
import src.utils

def main():

    train_drives, val_drives, test_drives = src.dataloaders.make_splits("data/kitti")
    train_dataset = src.DepthKITTIDataset("data/kitti", train_drives)

    tiny = torch.utils.data.Subset(train_dataset, range(4))     
    print(len(tiny))
    eval_dataset = src.DepthKITTIDataset("data/kitti", val_drives)
    train_loader = torch.utils.data.DataLoader(tiny, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)
    #eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)

    # single_batch = next(iter(train_loader))
    # fake_loader = [single_batch]

    device = src.utils.parse_device("cuda")
    #log_path = args.log_path
    #chkpt_path = args.chkpt_path
    DepthNet = src.KITTIdepthNET()
    PoseNet = src.KITTIposeNET()
    params = list(DepthNet.parameters()) + list(PoseNet.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=1)
    criterion = src.PhotometricLoss()
    # logger = src.utils.Logger(model, save_log_path=log_path, save_checkpoint_path=chkpt_path)

    trainer = src.TrainerOutdoor(DepthNet, PoseNet, train_loader, train_loader, criterion, optimizer, scheduler, device)

    trainer.fit(epochs=1000)

if __name__ == "__main__":
    main()
#     args = src.utils.parse_args()
#     print(args)
#     main(args)