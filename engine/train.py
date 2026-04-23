import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from data.panocount_dataset import PanoCountDataset
from models.csrnet import CSRNet
from engine.losses import DensityCountLoss
from engine.metrics import compute_count_errors


def main():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")

    TRAIN_SPLIT = os.path.join(DATASET_ROOT, "splits", "train.txt")
    VAL_SPLIT = os.path.join(DATASET_ROOT, "splits", "val.txt")

    SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_FILE = os.path.join(PROJECT_ROOT, "training_log.txt")

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = PanoCountDataset(
        dataset_root=DATASET_ROOT,
        split_file=TRAIN_SPLIT,
        image_size=(512, 1024),
        training=True,
        use_horizontal_roll=True,
    )

    val_dataset = PanoCountDataset(
        dataset_root=DATASET_ROOT,
        split_file=VAL_SPLIT,
        image_size=(512, 1024),
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    model = CSRNet(load_pretrained_vgg=True).to(device)
    criterion = DensityCountLoss(density_weight=1.0, count_weight=0.0)
    optimizer = Adam(model.parameters(), lr=1e-5)

    epochs = 5
    best_mae = float("inf")
    start_time_total = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader)
        for batch in pbar:
            images = batch["image"].to(device)
            gt_density = batch["density"].to(device)

            pred_density = model(images)
            out = criterion(pred_density, gt_density)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Train Loss {loss.item():.5f}")

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        mae = 0.0
        rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                gt_density = batch["density"].to(device)
                pred_density = model(images)

                abs_err, sq_err, _, _ = compute_count_errors(pred_density, gt_density)
                mae += abs_err.sum().item()
                rmse += sq_err.sum().item()

        mae /= len(val_dataset)
        rmse = (rmse / len(val_dataset)) ** 0.5

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch - 1)

        log = (
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {avg_train_loss:.6f} | "
            f"Val MAE {mae:.2f} | "
            f"Val RMSE {rmse:.2f} | "
            f"Epoch Time {epoch_time/60:.2f} min | "
            f"ETA {eta/3600:.2f} hr"
        )
        print("\n" + log)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log + "\n")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "mae": mae,
            "rmse": rmse
        }, os.path.join(SAVE_DIR, "latest_checkpoint.pth"))

        if mae < best_mae:
            best_mae = mae
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "mae": mae,
                "rmse": rmse
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print("Saved new BEST model.")


if __name__ == "__main__":
    main()