import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.panocount_dataset import PanoCountDataset
from engine.metrics import compute_count_errors
from models.model_factory import build_model

# Same bin boundaries as panocount_make_split.py
BIN_ORDER = ["0-49", "50-99", "100-199", "200-299", "300-499", "500+"]
BIN_LABELS = {
    "0-49": "Sparse",
    "50-99": "Medium-low",
    "100-199": "Medium",
    "200-299": "Medium-high",
    "300-499": "Dense",
    "500+": "Ultra-dense",
}


def get_bin(c):
    if c < 50:
        return "0-49"
    elif c < 100:
        return "50-99"
    elif c < 200:
        return "100-199"
    elif c < 300:
        return "200-299"
    elif c < 500:
        return "300-499"
    else:
        return "500+"


def compute_per_bin_mae(results):
    """Group per-image results by GT count bin and compute MAE and RMSE per bin."""
    by_bin = defaultdict(list)
    for r in results:
        gt = r["gt_count"]
        err = r["abs_error"]
        sq_err = (r["pred_count"] - gt) ** 2
        by_bin[get_bin(gt)].append((err, sq_err))
    per_bin = {}
    for bin_name in BIN_ORDER:
        if bin_name not in by_bin:
            per_bin[bin_name] = {"mae": None, "rmse": None, "n": 0}
            continue
        errs = by_bin[bin_name]
        n = len(errs)
        mae = sum(e[0] for e in errs) / n
        rmse = (sum(e[1] for e in errs) / n) ** 0.5
        per_bin[bin_name] = {"mae": mae, "rmse": rmse, "n": n}
    return per_bin


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    mae = 0.0
    rmse = 0.0
    results = []

    for batch in tqdm(loader):
        images = batch["image"].to(device, non_blocking=True)
        gt_density = batch["density"].to(device, non_blocking=True)
        ids = batch["id"]

        pred_density = model(images)
        abs_err, sq_err, pred_count, gt_count = compute_count_errors(pred_density, gt_density)

        mae += abs_err.sum().item()
        rmse += sq_err.sum().item()

        for i in range(len(ids)):
            results.append({
                "id": ids[i],
                "pred_count": float(pred_count[i].item()),
                "gt_count": float(gt_count[i].item()),
                "abs_error": float(abs_err[i].item())
            })

    mae /= len(loader.dataset)
    rmse = (rmse / len(loader.dataset)) ** 0.5
    return mae, rmse, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["csrnet", "mcnn", "can", "panocsrnet", "csrnet_context", "csrnet_pano", "csrnet_pano_circular_only", "csrnet_pano_latprior_only"], required=True)
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Evaluate on val or test split")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None, help="Write results JSON to this path instead of checkpoints/<model>/<split>_results.json")
    parser.add_argument("--save-predictions", action="store_true", help="Include per_image_results in JSON (default: always saved)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, "data", "dataset")
    split_file = os.path.join(dataset_root, "splits", f"{args.split}.txt")
    ckpt_dir = os.path.join(project_root, "checkpoints", args.model)

    # Resolve checkpoint path: explicit .pth path (absolute or relative to project root), "best", or "latest"
    candidate = os.path.join(project_root, args.checkpoint) if not os.path.isabs(args.checkpoint) else args.checkpoint
    if args.checkpoint not in ("best", "latest") and candidate.endswith(".pth") and os.path.isfile(candidate):
        ckpt_path = os.path.abspath(candidate)
    elif args.checkpoint == "best":
        ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    else:
        ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PanoCountDataset(
        dataset_root=dataset_root,
        split_file=split_file,
        image_size=(args.image_height, args.image_width),
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args.model).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    mae, rmse, results = evaluate(model, loader, device)

    split_label = args.split.capitalize()
    print(f"{split_label} MAE: {mae:.2f}")
    print(f"{split_label} RMSE: {rmse:.2f}")

    per_bin = compute_per_bin_mae(results)
    print(f"\nPer-density-bin MAE ({args.split}):")
    for bin_name in BIN_ORDER:
        info = per_bin[bin_name]
        if info["n"] == 0:
            continue
        label = BIN_LABELS.get(bin_name, bin_name)
        print(f"  {bin_name} ({label}): n={info['n']}, MAE={info['mae']:.2f}, RMSE={info['rmse']:.2f}")

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(ckpt_dir, f"{args.split}_results.json")
    else:
        out_path = os.path.join(project_root, out_path) if not os.path.isabs(out_path) else out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "checkpoint": ckpt_path,
            "split": args.split,
            f"{args.split}_mae": mae,
            f"{args.split}_rmse": rmse,
            "per_bin_mae": {b: per_bin[b] for b in BIN_ORDER},
            "per_image_results": results
        }, f, indent=2)


if __name__ == "__main__":
    main()