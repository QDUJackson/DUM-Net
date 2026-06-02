import argparse
import os
import time

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from MIDataSet import MultimodalRegistrationDataset
from losses import NCCLoss, dice_coefficient, gradient_loss
from module.UTN_More_Layers_Init_Lambda import UTNet


def make_dataset(root, split):
    split_dir = os.path.join(root, split)
    transform = transforms.Compose([transforms.ToTensor()])
    return MultimodalRegistrationDataset(
        os.path.join(split_dir, "t1_warp"),
        os.path.join(split_dir, "t2"),
        os.path.join(split_dir, "seg"),
        os.path.join(split_dir, "seg_warp"),
        os.path.join(split_dir, "t1"),
        transform=transform,
    )


def make_model(device, batch_size, num_layers):
    model = UTNet(
        beta=0.01,
        enc_nf=[16, 32, 32, 32],
        dec_nf=[32, 32, 32, 32, 32, 16, 16],
        size=[256, 256],
        device=device,
        size_tensor=(batch_size, 1, 256, 256),
        num_layers=num_layers,
        shold_values=0.16,
    )
    return model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, epoch, model, optimizer, best_dice_loss, history):
    torch.save(
        {
            "epoch": epoch,
            "best_dice_loss": best_dice_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_dice_loss"], checkpoint.get("history", [])


def plot_history(history, out_dir):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Total loss")
    axes[1].plot(epochs, [h["train_dice"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_dice"] for h in history], label="val")
    axes[1].set_title("Dice loss (negative)")
    axes[2].plot(epochs, [h["train_ncc"] for h in history], label="train")
    axes[2].plot(epochs, [h["val_ncc"] for h in history], label="val")
    axes[2].set_title("NCC loss")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close(fig)


def run_epoch(model, loader, device, optimizer, ncc_loss_fn, miu1, train):
    model.train(train)
    total_loss = 0.0
    dice_loss = 0.0
    ncc_loss = 0.0

    for data in loader:
        model1_img = data["model1"].to(device)
        model2_img = data["model2"].to(device)
        label = data["label"].to(device) * 255.0
        seg = data["seg"].to(device)
        model1_normal_img = data["model_normal"].to(device)

        if train:
            optimizer.zero_grad()

        model1_hat, fai, seg_wrapped = model(model1_img, model2_img, seg)
        reg_loss = gradient_loss(fai * 255.0, penalty="l2")
        image_loss = ncc_loss_fn(model1_hat, model1_normal_img)
        loss = reg_loss + miu1 * image_loss

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        dice_loss += dice_coefficient(seg_wrapped, label).item()
        ncc_loss += image_loss.item()

    denom = len(loader)
    return total_loss / denom, dice_loss / denom, ncc_loss / denom


def main():
    parser = argparse.ArgumentParser(description="Train or finetune the 26_525 T=7 elasticity model.")
    parser.add_argument("--data-root", default="data8k10")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "train_26_525"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--miu1", type=float, default=40.0)
    parser.add_argument("--resume", default="", help="Optional checkpoint path, e.g. outputs/train_26_525/latest.pth.")
    parser.add_argument("--init-weights", default="", help="Optional model state_dict path for finetuning.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = make_dataset(args.data_root, "train")
    val_set = make_dataset(args.data_root, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(device, args.batch_size, args.num_layers)
    if args.init_weights:
        model.load_state_dict(torch.load(args.init_weights, map_location=device), strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ncc_loss_fn = NCCLoss()
    start_epoch = 0
    best_dice_loss = float("inf")
    history = []

    if args.resume:
        start_epoch, best_dice_loss, history = load_checkpoint(args.resume, model, optimizer, device)
        print(f"resumed from {args.resume}, last finished epoch={start_epoch}")

    print(f"device: {device}")
    print(f"train samples: {len(train_set)}")
    print(f"val samples: {len(val_set)}")
    print(f"trainable params: {count_parameters(model):,}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice, train_ncc = run_epoch(
            model, train_loader, device, optimizer, ncc_loss_fn, args.miu1, train=True
        )
        with torch.no_grad():
            val_loss, val_dice, val_ncc = run_epoch(
                model, val_loader, device, optimizer, ncc_loss_fn, args.miu1, train=False
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_ncc": train_ncc,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_ncc": val_ncc,
        }
        history.append(row)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.5f} train_dice={train_dice:.5f} "
            f"val_loss={val_loss:.5f} val_dice={val_dice:.5f} "
            f"time={time.time() - t0:.1f}s"
        )

        with open(os.path.join(args.out_dir, "metrics.tsv"), "a", encoding="utf-8") as f:
            f.write(
                f"{epoch}\t{train_loss:.8f}\t{train_dice:.8f}\t{train_ncc:.8f}\t"
                f"{val_loss:.8f}\t{val_dice:.8f}\t{val_ncc:.8f}\n"
            )

        if val_dice < best_dice_loss:
            best_dice_loss = val_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_dice_model_swin.pth"))

        save_checkpoint(os.path.join(args.out_dir, "latest.pth"), epoch, model, optimizer, best_dice_loss, history)
        plot_history(history, args.out_dir)


if __name__ == "__main__":
    main()
