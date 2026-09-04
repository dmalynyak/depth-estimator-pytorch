from src.engine import Trainer
import src
import torch
import src.utils

def main(args):

    train_path = "data/nyu/train"
    val_path = "data/nyu/test"
    train_dataset = src.DepthNYUDataset(train_path, split='train')
    eval_dataset = src.DepthNYUDataset(val_path, split='eval')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True)

    single_batch = next(iter(train_loader))
    fake_loader = [single_batch]

    device = src.utils.parse_device(args.device)
    logs_save_path = args.logs_save_path
    weights_save_path = args.weights_save_path
    model = src.NYUmodel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = src.NYULoss()
    logger = src.utils.Logger(logs_save_path)

    trainer = src.Trainer(model, fake_loader, fake_loader, criterion, optimizer, device, logger)

    trainer.fit(epochs=300)

if __name__ == "__main__":
    args = src.utils.parse_args()
    print(args)
    main(args)