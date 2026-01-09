import os
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from datasets.color_dataset import ColorizationImageDataset
from models.unet import TinyUNet
from utils.color import lab_to_rgb_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple grayscale-to-color model")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training images (RGB)")
    parser.add_argument("--val_dir", type=str, default=None, help="Optional validation images path")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint: {path}")


def validate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    l1 = nn.L1Loss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            L = batch["L"].to(device)
            gt_ab = batch["ab"].to(device)
            pred_ab = model(L)
            loss = l1(pred_ab, gt_ab)
            bs = L.size(0)
            total += loss.item() * bs
            count += bs
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()

    device = torch.device(args.device)
    train_ds = ColorizationImageDataset(args.train_dir, image_size=args.image_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader: Optional[DataLoader] = None
    if args.val_dir is not None and os.path.isdir(args.val_dir):
        val_ds = ColorizationImageDataset(args.val_dir, image_size=args.image_size, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = TinyUNet(in_channels=1, out_channels=2, base_c=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        for step, batch in enumerate(train_loader, start=1):
            L = batch["L"].to(device)
            gt_ab = batch["ab"].to(device)

            pred_ab = model(L)
            loss = l1(pred_ab, gt_ab)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = L.size(0)
            running += loss.item() * bs
            seen += bs

            if step % 50 == 0:
                avg = running / max(1, seen)
                print(f"Epoch {epoch} Step {step}/{len(train_loader)} - Train L1: {avg:.4f}")

        train_avg = running / max(1, seen)
        print(f"Epoch {epoch} - Train L1: {train_avg:.4f}")

        if val_loader is not None:
            val_avg = validate(model, val_loader, device)
            print(f"Epoch {epoch} - Val L1: {val_avg:.4f}")

        save_checkpoint(model, optimizer, epoch, args.save_dir)


if __name__ == "__main__":
    main()


