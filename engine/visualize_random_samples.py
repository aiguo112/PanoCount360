
import os
import sys
import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.model_factory import build_model


MODEL_LIST = ["csrnet", "mcnn", "can", "csrnet_pano"]


def find_image_path(images_dir, image_id):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = os.path.join(images_dir, image_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing image for id={image_id}")


def read_split_ids(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_points(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pts = []
    for key in ["points", "ann_points", "gt_points", "annotations"]:
        if key in data and isinstance(data[key], list):
            raw = data[key]
            for p in raw:
                if isinstance(p, dict):
                    x = p.get("x", p.get("X"))
                    y = p.get("y", p.get("Y"))
                    if x is not None and y is not None:
                        pts.append([float(x), float(y)])
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append([float(p[0]), float(p[1])])
            break
    return np.array(pts, dtype=np.float32) if len(pts) else np.zeros((0, 2), dtype=np.float32)


def resize_density_for_display(density, out_h, out_w):
    if density.shape == (out_h, out_w):
        return density
    return cv2.resize(density, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def preprocess_image(img_rgb, out_h, out_w):
    img = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).float()


def get_density_group(gt_count):
    if gt_count < 50:
        return "Sparse"
    elif gt_count < 200:
        return "Medium"
    elif gt_count < 500:
        return "Dense"
    else:
        return "Ultra-dense"


@torch.no_grad()
def predict_all(models, img_tensor, device):
    results = {}
    for name, model in models.items():
        pred = model(img_tensor.to(device))
        pred = pred.squeeze().detach().cpu().numpy()
        results[name] = {
            "density": pred,
            "count": float(pred.sum())
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    project_root = Path(PROJECT_ROOT)
    dataset_root = project_root / "data" / "dataset"
    split_file = dataset_root / "splits" / f"{args.split}.txt"
    images_dir = dataset_root / "images"
    json_dir = dataset_root / "json"
    density_dir = dataset_root / "density_maps"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "analysis_outputs" / args.split / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = read_split_ids(str(split_file))
    chosen = random.sample(ids, min(args.num_samples, len(ids)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {}
    for model_name in MODEL_LIST:
        ckpt_path = project_root / "checkpoints" / model_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint: {ckpt_path}")
            continue
        model = build_model(model_name).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[model_name] = model

    manifest = []

    for image_id in chosen:
        image_path = find_image_path(str(images_dir), image_id)
        json_path = json_dir / f"{image_id}.json"
        density_path = density_dir / f"{image_id}.npy"

        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_density = np.load(str(density_path)).astype(np.float32)
        gt_count = float(gt_density.sum())
        gt_points = load_points(str(json_path))

        disp_img = cv2.resize(img_rgb, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
        scale_x = args.image_width / img_rgb.shape[1]
        scale_y = args.image_height / img_rgb.shape[0]
        disp_points = gt_points.copy()
        if len(disp_points):
            disp_points[:, 0] *= scale_x
            disp_points[:, 1] *= scale_y

        img_tensor = preprocess_image(img_rgb, args.image_height, args.image_width)
        preds = predict_all(models, img_tensor, device)

        gt_density_disp = resize_density_for_display(gt_density, args.image_height, args.image_width)

        n_models = len(MODEL_LIST)
        n_cols = max(3, n_models + 3)
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 9))
        axes = axes.ravel()

        # 1. raw image
        axes[0].imshow(disp_img)
        axes[0].set_title(f"Image ID: {image_id}\nGT Count: {gt_count:.1f} | Group: {get_density_group(gt_count)}")
        axes[0].axis("off")

        # 2. image + red dots
        axes[1].imshow(disp_img)
        if len(disp_points):
            axes[1].scatter(disp_points[:, 0], disp_points[:, 1], s=8, c="red", alpha=0.8)
        axes[1].set_title(f"Head Annotations (red dots)\nNum points: {len(gt_points)}")
        axes[1].axis("off")

        # 3. GT heatmap
        axes[2].imshow(disp_img, alpha=0.55)
        axes[2].imshow(gt_density_disp, cmap="jet", alpha=0.55)
        axes[2].set_title(f"GT Density Heatmap\nSum={gt_count:.1f}")
        axes[2].axis("off")

        # 4+ model predictions
        for ax_idx, model_name in enumerate(MODEL_LIST, start=3):
            ax = axes[ax_idx]
            ax.imshow(disp_img, alpha=0.55)
            if model_name in preds:
                pred_disp = resize_density_for_display(preds[model_name]["density"], args.image_height, args.image_width)
                ax.imshow(pred_disp, cmap="jet", alpha=0.55)
                label = model_name.upper().replace("_", " ")
                ax.set_title(f"{label} Prediction\nPred={preds[model_name]['count']:.1f} | Err={preds[model_name]['count'] - gt_count:+.1f}")
            else:
                ax.set_title(f"{model_name.upper()} missing checkpoint")
            ax.axis("off")

        # hide unused axes
        for ax_idx in range(3 + n_models, len(axes)):
            axes[ax_idx].set_visible(False)

        plt.tight_layout()
        save_path = output_dir / f"{image_id}_comparison.png"
        plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
        plt.close(fig)

        manifest.append({
            "id": image_id,
            "gt_count": gt_count,
            "density_group": get_density_group(gt_count),
            "num_points": int(len(gt_points)),
            **{f"{m}_pred": preds.get(m, {}).get("count") for m in MODEL_LIST},
            "figure_path": str(save_path),
        })
        print(f"Saved figure: {save_path}")

    with open(output_dir / "figure_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} figures to: {output_dir}")


if __name__ == "__main__":
    main()
