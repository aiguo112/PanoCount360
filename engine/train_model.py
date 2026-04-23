import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import time
import json
import argparse
import random

import numpy as np
import torch

# Speed optimizations for RTX GPUs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from data.panocount_dataset import PanoCountDataset
from engine.losses import DensityCountLoss
from engine.metrics import compute_count_errors
from models.model_factory import build_model


def _apply_color_jitter(images, brightness=0.1, contrast=0.1):
    """Apply random brightness and contrast to a batch [B, C, H, W]. In-place friendly."""
    b, c, h, w = images.shape
    brightness_factor = 1.0 + (torch.rand(1, device=images.device).item() * 2 - 1) * brightness
    contrast_factor = 1.0 + (torch.rand(1, device=images.device).item() * 2 - 1) * contrast
    images = images * brightness_factor
    mean = images.view(b, c, -1).mean(dim=(1, 2), keepdim=True).view(b, 1, 1, 1)
    images = (images - mean) * contrast_factor + mean
    return images


def train_one_epoch(model, loader, criterion, optimizer, device, augment_color=False):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)

        if augment_color:
            images = _apply_color_jitter(images, brightness=0.1, contrast=0.1)

        pred_density = model(images)
        out = criterion(pred_density, gt_density)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Train Loss {loss.item():.5f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    mae = 0.0
    rmse = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)

        pred_density = model(images)
        abs_err, sq_err, _, _ = compute_count_errors(pred_density, gt_density)

        mae += abs_err.sum().item()
        rmse += sq_err.sum().item()

    mae /= len(loader.dataset)
    rmse = (rmse / len(loader.dataset)) ** 0.5
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["csrnet", "mcnn", "can", "panocsrnet", "csrnet_context", "csrnet_pano", "csrnet_pano_circular_only", "csrnet_pano_latprior_only"])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs before cosine decay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 penalty for Adam (default 1e-4)")
    parser.add_argument("--augment-color", action="store_true", help="Apply light brightness/contrast jitter during training")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=1024)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "data", "dataset")
    train_split = os.path.join(dataset_root, "splits", "train.txt")
    val_split = os.path.join(dataset_root, "splits", "val.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", torch.cuda.is_available(), torch.cuda.device_count())

    train_dataset = PanoCountDataset(
        dataset_root=dataset_root,
        split_file=train_split,
        image_size=(args.image_height, args.image_width),
        training=True,
        use_horizontal_roll=True,
        roll_p=0.5,
    )

    val_dataset = PanoCountDataset(
        dataset_root=dataset_root,
        split_file=val_split,
        image_size=(args.image_height, args.image_width),
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model).to(device)
    criterion = DensityCountLoss(density_weight=1.0, count_weight=0.0)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.2, total_iters=args.warmup_epochs)
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    save_dir = os.path.join(project_root, "checkpoints", args.model)
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")

    # CSRNetPano final run: backup path (timestamped, never overwritten by future runs) and CSV log
    final_backup_path = None
    final_backup_path_timestamped = None
    csv_log_path = None
    if args.model == "csrnet_pano":
        ts = time.strftime("%Y%m%d_%H%M%S")
        final_backup_path = os.path.join(save_dir, "best_model_FINAL_seed42.pth")
        final_backup_path_timestamped = os.path.join(save_dir, f"best_model_FINAL_seed42_{ts}.pth")
        csv_log_path = os.path.join(save_dir, "training_log_seed42.csv")
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,val_mae,is_best\n")
            f.flush()
    elif args.model == "csrnet_pano_circular_only":
        csv_log_path = os.path.join(save_dir, "training_log_seed42.csv")
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,val_mae,is_best\n")
            f.flush()
    elif args.model == "csrnet_pano_latprior_only":
        csv_log_path = os.path.join(save_dir, "training_log_seed42.csv")
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write("epoch,val_mae,is_best\n")
            f.flush()

    best_mae = float("inf")
    best_epoch = -1
    no_improve = 0
    total_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            augment_color=args.augment_color,
        )
        val_mae, val_rmse = validate(model, val_loader, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - total_start
        epochs_left = args.epochs - epoch - 1
        eta = epoch_time * epochs_left
        current_lr = optimizer.param_groups[0]["lr"]

        log = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"LR {current_lr:.2e} | "
            f"Train Loss {train_loss:.6f} | "
            f"Val MAE {val_mae:.2f} | "
            f"Val RMSE {val_rmse:.2f} | "
            f"Epoch Time {epoch_time/60:.2f} min | "
            f"ETA {eta/3600:.2f} hr"
        )
        print(log)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log + "\n")

        # CSRNetPano: append val MAE per epoch to CSV (flush so it survives interrupt)
        if csv_log_path is not None:
            is_best = val_mae < best_mae
            with open(csv_log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{val_mae:.6f},{is_best}\n")
                f.flush()

        latest_ckpt = {
            "epoch": epoch + 1,
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_mae": best_mae,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        }
        torch.save(latest_ckpt, os.path.join(save_dir, "latest_checkpoint.pth"))

        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            no_improve = 0

            best_ckpt = {
                "epoch": epoch + 1,
                "model_name": args.model,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_mae": best_mae,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
            torch.save(best_ckpt, os.path.join(save_dir, "best_model.pth"))
            # CSRNetPano final run: also save to FINAL backup (fixed name for eval + timestamped copy)
            if final_backup_path is not None:
                torch.save(best_ckpt, final_backup_path)
                torch.save(best_ckpt, final_backup_path_timestamped)
                print(f"Saved new BEST model for {args.model} at epoch {epoch+1} (and FINAL backup).")
            else:
                print(f"Saved new BEST model for {args.model} at epoch {epoch+1}.")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping triggered for {args.model}. Best epoch: {best_epoch}, best MAE: {best_mae:.2f}")
            if epoch + 1 < 50:
                print("ERROR: Training stopped before epoch 50 (early stopping fired too early). Aborting.")
                sys.exit(1)
            break

    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_mae": best_mae,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()