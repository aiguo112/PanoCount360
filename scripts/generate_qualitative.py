"""Generate Chapter 5 qualitative outputs for all test images."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from inference_wrappers.base_model import MODEL_CONFIGS, build_and_load, run_forward
from utils.image_utils import TARGET_H, TARGET_W, prepare_model_input_tensor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IDS_JSON = os.path.join(PROJECT_ROOT, "results", "chapter5_final", "csrnetlp_correct_predictions.json")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "dataset")
JSON_ANN_DIR = os.path.join(DATASET_ROOT, "json")

OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "qualitative")
OUT_DIRS = {
    "originals": os.path.join(OUT_ROOT, "00_originals"),
    "gt_overlay": os.path.join(OUT_ROOT, "01_ground_truth"),
    "mcnn_overlay": os.path.join(OUT_ROOT, "02_mcnn"),
    "can_overlay": os.path.join(OUT_ROOT, "03_can"),
    "csrnet_overlay": os.path.join(OUT_ROOT, "04_csrnet"),
    "csrnetlp_overlay": os.path.join(OUT_ROOT, "05_csrnetlp"),
    "strips": os.path.join(OUT_ROOT, "06_comparison_strips"),
    "density_only": os.path.join(OUT_ROOT, "07_density_maps_only"),
    "summary": os.path.join(OUT_ROOT, "08_summary"),
}

MODEL_ORDER = ["MCNN", "CAN", "CSRNet", "CSRNet-LP"]
COLORS = {
    "original": (255, 255, 255),
    "gt": (255, 215, 0),
    "mcnn": (220, 145, 68),
    "can": (215, 194, 96),
    "csrnet": (52, 152, 219),
    "csrnetlp": (39, 174, 96),
    "best": (0, 255, 136),
    "navy_bg": (10, 14, 26),
    "label_bg": (13, 17, 23),
}

LABEL_COL_W = 350
IMAGE_COL_W = 1050
ROW_H = 525
ROW_GAP = 10
VERT_ROWS = 6
VERT_STRIP_W = LABEL_COL_W + IMAGE_COL_W
VERT_STRIP_H = (VERT_ROWS * ROW_H) + ((VERT_ROWS - 1) * ROW_GAP)
THUMB_W = 416
THUMB_H = 951


@dataclass
class ImageResult:
    image_id: str
    gt_count: float
    pred_counts: Dict[str, float]
    abs_errors: Dict[str, float]
    pct_errors: Dict[str, float]
    best_model: str
    mean_abs_error: float
    bucket: str


def to_ascii_text(text: str) -> str:
    replacements = {
        "★": "[*]",
        "↑": "(+)",
        "↓": "(-)",
        "✓": "OK",
        "✗": "FAIL",
        "→": "->",
        "←": "<-",
        "°": "deg",
        "±": "+/-",
    }
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def get_font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Load a font that supports all characters needed for strip labels on Windows/Linux.
    Try fonts in order until one works.
    """
    font_candidates = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/timesi.ttf",
        "C:/Windows/Fonts/timesbi.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    ]
    if bold:
        bold_candidates = [
            "C:/Windows/Fonts/timesbd.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        ]
        font_candidates = bold_candidates + font_candidates

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative Chapter 5 outputs.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit of images.")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--strips-only",
        action="store_true",
        help="Regenerate strips only using saved predictions, no inference",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="When used with --strips-only, regenerate one strip only",
    )
    return parser.parse_args()


def select_device(choice: str) -> str:
    if choice == "cuda":
        return "cuda"
    if choice == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs() -> None:
    for p in OUT_DIRS.values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIRS["strips"], "thumbnails"), exist_ok=True)


def load_test_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids: List[str] = []
    for row in data.get("per_image_results", []):
        raw = str(row.get("id", "")).strip()
        if not raw:
            continue
        ids.append(os.path.splitext(raw)[0])
    return ids


def candidate_image_paths(image_id: str) -> List[str]:
    stem = os.path.splitext(image_id)[0]
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG"]
    folders = [
        os.path.join(DATASET_ROOT, "images"),
        os.path.join(DATASET_ROOT, "image"),
        os.path.join(DATASET_ROOT, "imgs"),
        os.path.join(DATASET_ROOT, "img"),
        os.path.join(PROJECT_ROOT, "data", "images"),
    ]
    out = []
    for d in folders:
        for ext in exts:
            out.append(os.path.join(d, stem + ext))
    return out


def find_image_path(image_id: str) -> str:
    for p in candidate_image_paths(image_id):
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No image file found for id={image_id}")


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_density_preserve_count(density: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = density.shape
    if in_h == out_h and in_w == out_w:
        return density.astype(np.float32)
    resized = cv2.resize(density.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    scale = (in_h * in_w) / float(out_h * out_w)
    return (resized * scale).astype(np.float32)


def density_to_viridis_bgr(density: np.ndarray) -> np.ndarray:
    d = np.clip(density.astype(np.float32), 0, None)
    dmax = float(d.max()) if d.size else 0.0
    if dmax <= 1e-9:
        norm_u8 = np.zeros_like(d, dtype=np.uint8)
    else:
        norm_u8 = np.clip(d / dmax * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm_u8, cv2.COLORMAP_VIRIDIS)


def make_overlay(rgb: np.ndarray, density: np.ndarray, alpha_heat: float = 0.55) -> np.ndarray:
    h, w = rgb.shape[:2]
    den = resize_density_preserve_count(density, h, w)
    heat_bgr = density_to_viridis_bgr(den)
    base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(base_bgr, 1.0 - alpha_heat, heat_bgr, alpha_heat, 0.0)
    return out


def load_annotation_points(image_id: str) -> Tuple[List[Tuple[float, float]], int]:
    ann_path = os.path.join(JSON_ANN_DIR, f"{os.path.splitext(image_id)[0]}.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing annotation json: {ann_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    pts = ann.get("points", [])
    out_pts = []
    for p in pts:
        x = float(p.get("x", -1))
        y = float(p.get("y", -1))
        out_pts.append((x, y))
    human_num = int(ann.get("human_num", len(out_pts)))
    return out_pts, human_num


def build_gt_density(image_shape_hw: Tuple[int, int], points: List[Tuple[float, float]], sigma: float = 4.0) -> np.ndarray:
    h, w = image_shape_hw
    impulse = np.zeros((h, w), dtype=np.float32)
    valid = 0
    for x, y in points:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            impulse[yi, xi] += 1.0
            valid += 1
    if valid == 0:
        return impulse
    k = max(3, int(sigma * 4) * 2 + 1)
    k |= 1
    den = cv2.GaussianBlur(impulse, (k, k), sigmaX=sigma, sigmaY=sigma)
    s = float(den.sum())
    if s > 1e-9:
        den = den * (float(valid) / s)
    return den.astype(np.float32)


def load_models(device: str) -> Dict[str, torch.nn.Module]:
    models: Dict[str, torch.nn.Module] = {}
    for cfg in MODEL_CONFIGS:
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", cfg.default_ckpt_subdir, "best_model.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint missing for {cfg.gui_id}: {ckpt}")
        model, demo = build_and_load(cfg.factory_name, ckpt, device)
        if demo:
            raise RuntimeError(f"Failed to load weights for {cfg.gui_id} (entered demo mode): {ckpt}")
        models[cfg.gui_id] = model
    return models


def run_all_models(models: Dict[str, torch.nn.Module], rgb: np.ndarray, device: str) -> Dict[str, np.ndarray]:
    resized_rgb, img_t = prepare_model_input_tensor(rgb, device)
    _ = resized_rgb
    out: Dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        den = run_forward(models[model_name], img_t)
        out[model_name] = den.astype(np.float32)
    return out


def overlay_for_panel(original_bgr: np.ndarray, density_map: np.ndarray) -> np.ndarray:
    h, w = ROW_H, IMAGE_COL_W
    den = resize_density_preserve_count(density_map, h, w)
    heat_bgr = density_to_viridis_bgr(den)
    base_bgr = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return cv2.addWeighted(base_bgr, 0.6, heat_bgr, 0.4, 0.0)


def draw_latitude_guides(panel_bgr: np.ndarray) -> np.ndarray:
    out = panel_bgr.copy()
    overlay = out.copy()
    step = max(1, IMAGE_COL_W // 80)
    guide_rows = [0, 66, 131, 197, 262, 328, 393, 459, 524]
    for y in guide_rows:
        for x in range(0, IMAGE_COL_W, step * 2):
            x2 = min(IMAGE_COL_W - 1, x + step)
            cv2.line(overlay, (x, y), (x2, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0, out)
    return out


def rank_models(gt_count: float, pred_counts: Dict[str, float]) -> List[str]:
    return sorted(MODEL_ORDER, key=lambda m: abs(pred_counts[m] - gt_count))


def build_vertical_row_defs(
    image_w: int,
    image_h: int,
    gt_count: float,
    pred_counts: Dict[str, float],
) -> List[Dict[str, object]]:
    ranked = rank_models(gt_count, pred_counts)
    best = ranked[0] if ranked else ""
    row_defs: List[Dict[str, object]] = [
        {
            "key": "ORIGINAL",
            "title": "Input Image",
            "sub": f"Resolution: {image_w}x{image_h}",
            "title_color": COLORS["original"],
            "sub_color": COLORS["original"],
        },
        {
            "key": "GT",
            "title": "Ground Truth",
            "sub": f"Count: {int(round(gt_count))}",
            "title_color": COLORS["gt"],
            "sub_color": COLORS["gt"],
        },
    ]
    for m in MODEL_ORDER:
        pred = float(pred_counts[m])
        err_pct = ((pred - gt_count) / max(gt_count, 1.0)) * 100.0
        sign = "+" if err_pct >= 0 else ""
        is_best = m == best
        title = "CSRNet-LP Proposed" if m == "CSRNet-LP" else m
        sub = f"Pred: {int(round(pred))}   Error: {sign}{err_pct:.1f}%"
        if m == "CSRNet-LP" and is_best:
            sub += "  BEST"
        c = COLORS["csrnetlp" if m == "CSRNet-LP" else m.lower()]
        row_defs.append(
            {
                "key": m,
                "title": title,
                "sub": sub,
                "title_color": c,
                "sub_color": c,
                "is_best": is_best,
            }
        )
    return row_defs


def draw_left_label_panel(
    draw: ImageDraw.ImageDraw,
    row_def: Dict[str, object],
    y_start: int
) -> None:
    left_pad = 20
    right_pad = 20
    row_bg = COLORS["label_bg"]
    if row_def["key"] == "GT":
        row_bg = (26, 26, 46)
    if row_def["key"] == "CSRNet-LP" and row_def.get("is_best", False):
        row_bg = (10, 42, 10)
    draw.rectangle([0, y_start, LABEL_COL_W - 1, y_start + ROW_H - 1], fill=row_bg)

    if row_def["key"] == "GT":
        draw.line([(0, y_start + ROW_H - 2), (LABEL_COL_W - 1, y_start + ROW_H - 2)], fill=COLORS["gt"], width=2)
    if row_def["key"] == "CSRNet-LP" and row_def.get("is_best", False):
        draw.line([(0, y_start + 1), (LABEL_COL_W - 1, y_start + 1)], fill=COLORS["best"], width=3)
        draw.line([(0, y_start + ROW_H - 2), (LABEL_COL_W - 1, y_start + ROW_H - 2)], fill=COLORS["best"], width=3)

    title = to_ascii_text(str(row_def["title"]))
    sub = to_ascii_text(str(row_def["sub"]))
    title_font = get_font(size=22, bold=True)
    right_font = get_font(size=20, bold=False)

    def text_width(txt: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> int:
        bb = draw.textbbox((0, 0), txt, font=font)
        return max(0, bb[2] - bb[0])

    def truncate_to_width(txt: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_w: int) -> str:
        if max_w <= 0:
            return ""
        if text_width(txt, font) <= max_w:
            return txt
        ellipsis = "..."
        if text_width(ellipsis, font) > max_w:
            return ""
        keep = txt
        while keep and text_width(keep + ellipsis, font) > max_w:
            keep = keep[:-1]
        return (keep + ellipsis) if keep else ellipsis

    max_w = LABEL_COL_W - left_pad - right_pad
    if title == "CSRNet-LP Proposed" and text_width(title, title_font) > max_w:
        title = "CSRNet-LP"
    title = truncate_to_width(title, title_font, max_w)
    sub = truncate_to_width(sub, right_font, max_w)
    draw.text((left_pad, y_start + 188), title, font=title_font, fill=row_def["title_color"])
    draw.text((left_pad, y_start + 232), sub, font=right_font, fill=row_def["sub_color"])


def save_strip_and_thumbnail(strip_bgr: np.ndarray, image_id: str, output_dir: str) -> str:
    strip_path = os.path.join(output_dir, f"{image_id}_strip.png")
    cv2.imwrite(strip_path, strip_bgr)
    thumb_dir = os.path.join(output_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    thumb = cv2.resize(strip_bgr, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        os.path.join(thumb_dir, f"{image_id}_thumb.jpg"),
        thumb,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    return strip_path


def generate_vertical_strip(
    image: np.ndarray,
    gt_density: np.ndarray,
    predictions: Dict[str, Dict[str, object]],
    gt_count: float,
    image_id: str,
    output_dir: str,
) -> str:
    row_defs = build_vertical_row_defs(image.shape[1], image.shape[0], gt_count, {m: float(predictions[m]["count"]) for m in MODEL_ORDER})
    strip = np.zeros((VERT_STRIP_H, VERT_STRIP_W, 3), dtype=np.uint8)

    original_panel = cv2.resize(image, (IMAGE_COL_W, ROW_H), interpolation=cv2.INTER_LANCZOS4)
    original_panel = draw_latitude_guides(original_panel)
    if gt_density.ndim == 3:
        gt_panel = cv2.resize(gt_density, (IMAGE_COL_W, ROW_H), interpolation=cv2.INTER_LANCZOS4)
    else:
        gt_panel = overlay_for_panel(image, gt_density)
    model_panels: Dict[str, np.ndarray] = {}
    for m in MODEL_ORDER:
        if "panel_bgr" in predictions[m]:
            model_panels[m] = cv2.resize(
                np.asarray(predictions[m]["panel_bgr"]),
                (IMAGE_COL_W, ROW_H),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            model_panels[m] = overlay_for_panel(image, np.asarray(predictions[m]["density_map"]))
    row_panels = [original_panel, gt_panel, model_panels["MCNN"], model_panels["CAN"], model_panels["CSRNet"], model_panels["CSRNet-LP"]]

    pil = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    y = 0
    for idx, row_def in enumerate(row_defs):
        draw_left_label_panel(draw, row_def, y)
        panel_rgb = cv2.cvtColor(row_panels[idx], cv2.COLOR_BGR2RGB)
        panel_pil = Image.fromarray(panel_rgb)
        pil.paste(panel_pil, (LABEL_COL_W, y))
        y += ROW_H
        if idx < (VERT_ROWS - 1):
            y += ROW_GAP
    strip_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return save_strip_and_thumbnail(strip_bgr, image_id, output_dir)

def bucket_name(gt_count: float) -> str:
    if gt_count < 50:
        return "sparse"
    if gt_count < 100:
        return "med_low"
    if gt_count < 200:
        return "medium"
    if gt_count < 300:
        return "med_high"
    if gt_count < 500:
        return "dense"
    return "ultra_dense"


def save_density_only(image_id: str, gt_den: np.ndarray, pred_densities: Dict[str, np.ndarray], original_shape: Tuple[int, int]) -> None:
    h, w = original_shape
    gt_resized = resize_density_preserve_count(gt_den, h, w)
    cv2.imwrite(os.path.join(OUT_DIRS["density_only"], f"{image_id}_gt.png"), density_to_viridis_bgr(gt_resized))
    key_map = [("MCNN", "mcnn"), ("CAN", "can"), ("CSRNet", "csrnet"), ("CSRNet-LP", "csrnetlp")]
    for model_key, tag in key_map:
        den = resize_density_preserve_count(pred_densities[model_key], h, w)
        cv2.imwrite(os.path.join(OUT_DIRS["density_only"], f"{image_id}_{tag}.png"), density_to_viridis_bgr(den))


def write_summary_files(rows: List[ImageResult]) -> None:
    summary_dir = OUT_DIRS["summary"]
    meta_path = os.path.join(summary_dir, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "gt_count",
                "mcnn_pred",
                "can_pred",
                "csrnet_pred",
                "csrnetlp_pred",
                "mcnn_abs_err",
                "can_abs_err",
                "csrnet_abs_err",
                "csrnetlp_abs_err",
                "mcnn_err_pct",
                "can_err_pct",
                "csrnet_err_pct",
                "csrnetlp_err_pct",
                "best_model",
                "mean_abs_error",
                "bucket",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    to_ascii_text(r.image_id),
                    round(r.gt_count, 3),
                    round(r.pred_counts["MCNN"], 3),
                    round(r.pred_counts["CAN"], 3),
                    round(r.pred_counts["CSRNet"], 3),
                    round(r.pred_counts["CSRNet-LP"], 3),
                    round(r.abs_errors["MCNN"], 3),
                    round(r.abs_errors["CAN"], 3),
                    round(r.abs_errors["CSRNet"], 3),
                    round(r.abs_errors["CSRNet-LP"], 3),
                    round(r.pct_errors["MCNN"], 3),
                    round(r.pct_errors["CAN"], 3),
                    round(r.pct_errors["CSRNet"], 3),
                    round(r.pct_errors["CSRNet-LP"], 3),
                    to_ascii_text(r.best_model),
                    round(r.mean_abs_error, 3),
                    to_ascii_text(r.bucket),
                ]
            )

    worst = sorted(rows, key=lambda x: x.mean_abs_error, reverse=True)
    lp_best = [r for r in rows if r.best_model == "CSRNet-LP"]
    sparse_lp_best = [r for r in lp_best if r.bucket in {"sparse", "med_low"}]
    dense_lp_best = [r for r in lp_best if r.bucket in {"dense", "ultra_dense"}]

    def write_case_file(path: str, title: str, items: List[ImageResult], top_k: int = 30) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(title + "\n")
            f.write("=" * len(title) + "\n\n")
            for r in items[:top_k]:
                f.write(
                    f"{to_ascii_text(r.image_id)} | GT={r.gt_count:.0f} | "
                    f"MCNN={r.pred_counts['MCNN']:.0f} (AE={r.abs_errors['MCNN']:.1f}) | "
                    f"CAN={r.pred_counts['CAN']:.0f} (AE={r.abs_errors['CAN']:.1f}) | "
                    f"CSRNet={r.pred_counts['CSRNet']:.0f} (AE={r.abs_errors['CSRNet']:.1f}) | "
                    f"CSRNet-LP={r.pred_counts['CSRNet-LP']:.0f} (AE={r.abs_errors['CSRNet-LP']:.1f}) | "
                    f"best={to_ascii_text(r.best_model)} | bucket={to_ascii_text(r.bucket)}\n"
                )

    write_case_file(os.path.join(summary_dir, "worst_cases.txt"), "Worst cases by mean absolute error", worst)
    write_case_file(
        os.path.join(summary_dir, "best_cases.txt"),
        "Best cases where CSRNet-LP has minimum absolute error",
        sorted(lp_best, key=lambda x: x.abs_errors["CSRNet-LP"]),
    )
    write_case_file(
        os.path.join(summary_dir, "sparse_best.txt"),
        "Sparse/med-low cases where CSRNet-LP is best",
        sorted(sparse_lp_best, key=lambda x: x.abs_errors["CSRNet-LP"]),
    )
    write_case_file(
        os.path.join(summary_dir, "dense_best.txt"),
        "Dense/ultra-dense cases where CSRNet-LP is best",
        sorted(dense_lp_best, key=lambda x: x.abs_errors["CSRNet-LP"]),
    )


def _find_existing_image(path_without_ext: str) -> str:
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".PNG", ".JPEG"]
    for ext in exts:
        p = f"{path_without_ext}{ext}"
        if os.path.isfile(p):
            return p
    return ""


def _load_panel_from_saved(path_no_ext: str) -> np.ndarray:
    p = _find_existing_image(path_no_ext)
    if p:
        panel = cv2.imread(p, cv2.IMREAD_COLOR)
        if panel is not None:
            return cv2.resize(panel, (IMAGE_COL_W, ROW_H), interpolation=cv2.INTER_LANCZOS4)
    return np.zeros((ROW_H, IMAGE_COL_W, 3), dtype=np.uint8)


def _build_grid(image_ids: List[str], out_path: str) -> None:
    thumb_dir = os.path.join(OUT_DIRS["strips"], "thumbnails")
    thumbs: List[np.ndarray] = []
    for image_id in image_ids:
        p = os.path.join(thumb_dir, f"{image_id}_thumb.jpg")
        if os.path.isfile(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                thumbs.append(cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    while len(thumbs) < 6:
        thumbs.append(np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
    grid = np.concatenate(thumbs[:6], axis=1)
    cv2.imwrite(out_path, grid)


def _choose_best_grid_ids(metadata: pd.DataFrame) -> List[str]:
    if metadata.empty:
        return []
    df = metadata.copy()
    df["image_id"] = df["image_id"].astype(str)
    for col in ["mcnn_abs_err", "can_abs_err", "csrnet_abs_err", "csrnetlp_abs_err", "gt_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    ranked: List[str] = []
    for _, row in df.iterrows():
        errs = {
            "MCNN": float(row["mcnn_abs_err"]),
            "CAN": float(row["can_abs_err"]),
            "CSRNet": float(row["csrnet_abs_err"]),
            "CSRNet-LP": float(row["csrnetlp_abs_err"]),
        }
        order = sorted(errs.keys(), key=lambda m: errs[m])
        if order.index("CSRNet-LP") <= 1:
            ranked.append(str(row["image_id"]))
    gt_map = {str(r["image_id"]): float(r["gt_count"]) for _, r in df.iterrows()}
    sparse = [i for i in ranked if gt_map.get(i, 0.0) < 100]
    medium = [i for i in ranked if 100 <= gt_map.get(i, 0.0) < 500]
    dense = [i for i in ranked if gt_map.get(i, 0.0) > 500]
    selected: List[str] = []
    selected.extend(sparse[:2])
    selected.extend([x for x in medium if x not in selected][:2])
    selected.extend([x for x in dense if x not in selected][:1])
    for x in ranked:
        if x not in selected:
            selected.append(x)
        if len(selected) == 6:
            break
    return selected[:6]


def _choose_worst_grid_ids(metadata: pd.DataFrame) -> List[str]:
    if metadata.empty:
        return []
    df = metadata.copy()
    df["image_id"] = df["image_id"].astype(str)
    if "mean_abs_error" not in df.columns:
        df["mean_abs_error"] = (
            pd.to_numeric(df["mcnn_abs_err"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["can_abs_err"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["csrnet_abs_err"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["csrnetlp_abs_err"], errors="coerce").fillna(0.0)
        ) / 4.0
    df = df.sort_values("mean_abs_error", ascending=False)
    return [str(x) for x in df["image_id"].head(6).tolist()]


def build_summary_grids_from_metadata(metadata: pd.DataFrame) -> None:
    best_ids = _choose_best_grid_ids(metadata)
    worst_ids = _choose_worst_grid_ids(metadata)
    _build_grid(best_ids, os.path.join(OUT_DIRS["summary"], "best_grid.png"))
    _build_grid(worst_ids, os.path.join(OUT_DIRS["summary"], "worst_grid.png"))


def regenerate_strips_only(test_one: bool = False) -> None:
    """
    Load saved predictions from metadata.csv and regenerate all strips with fixed fonts.
    Much faster than re-running inference.
    """
    ensure_dirs()
    meta_path = os.path.join(OUT_DIRS["summary"], "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    metadata = pd.read_csv(meta_path)
    if metadata.empty:
        print("metadata.csv is empty; nothing to regenerate.")
        return
    if test_one:
        metadata = metadata.head(1)

    first_output: Optional[str] = None
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Regenerating strips"):
        img_id = str(row["image_id"]).strip()
        if not img_id:
            continue
        original_path = _find_existing_image(os.path.join(OUT_DIRS["originals"], img_id))
        if not original_path:
            continue
        original_bgr = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if original_bgr is None:
            continue
        gt_overlay = _load_panel_from_saved(os.path.join(OUT_DIRS["gt_overlay"], img_id))
        mcnn_overlay = _load_panel_from_saved(os.path.join(OUT_DIRS["mcnn_overlay"], img_id))
        can_overlay = _load_panel_from_saved(os.path.join(OUT_DIRS["can_overlay"], img_id))
        csrnet_overlay = _load_panel_from_saved(os.path.join(OUT_DIRS["csrnet_overlay"], img_id))
        csrnetlp_overlay = _load_panel_from_saved(os.path.join(OUT_DIRS["csrnetlp_overlay"], img_id))
        gt_density = gt_overlay
        predictions = {
            "MCNN": {"count": float(row["mcnn_pred"]), "density_map": np.zeros((1, 1), dtype=np.float32), "panel_bgr": mcnn_overlay},
            "CAN": {"count": float(row["can_pred"]), "density_map": np.zeros((1, 1), dtype=np.float32), "panel_bgr": can_overlay},
            "CSRNet": {"count": float(row["csrnet_pred"]), "density_map": np.zeros((1, 1), dtype=np.float32), "panel_bgr": csrnet_overlay},
            "CSRNet-LP": {"count": float(row["csrnetlp_pred"]), "density_map": np.zeros((1, 1), dtype=np.float32), "panel_bgr": csrnetlp_overlay},
        }
        gt_count = float(row["gt_count"])
        strip_path = generate_vertical_strip(
            image=original_bgr,
            gt_density=gt_density,
            predictions=predictions,
            gt_count=gt_count,
            image_id=img_id,
            output_dir=OUT_DIRS["strips"],
        )
        if first_output is None:
            first_output = strip_path

    build_summary_grids_from_metadata(metadata)
    if test_one and first_output and hasattr(os, "startfile"):
        try:
            os.startfile(first_output)  # type: ignore[attr-defined]
        except Exception:
            pass

    print("Done. Check results/qualitative/06_comparison_strips/")


def print_regeneration_instructions() -> None:
    print(
        "Updated to left-label thesis layout.\n\n"
        "Test:\n"
        "python scripts/generate_qualitative.py --strips-only --test-one\n\n"
        "All:\n"
        "python scripts/generate_qualitative.py --strips-only"
    )


def main(args: argparse.Namespace) -> None:
    ensure_dirs()
    device = select_device(args.device)

    test_ids = load_test_ids(TEST_IDS_JSON)
    if args.limit > 0:
        test_ids = test_ids[: args.limit]
    if not test_ids:
        raise RuntimeError("No test IDs found.")

    print(f"Loaded {len(test_ids)} image IDs from {TEST_IDS_JSON}")
    print(f"Using device: {device}")
    models = load_models(device)
    print("Loaded all 4 model checkpoints successfully.")

    model_overlay_folder = {
        "MCNN": OUT_DIRS["mcnn_overlay"],
        "CAN": OUT_DIRS["can_overlay"],
        "CSRNet": OUT_DIRS["csrnet_overlay"],
        "CSRNet-LP": OUT_DIRS["csrnetlp_overlay"],
    }
    rows: List[ImageResult] = []
    skipped: List[Tuple[str, str]] = []
    per_model_abs_errs = {m: [] for m in MODEL_ORDER}

    start = time.perf_counter()
    n = len(test_ids)
    for idx, image_id in enumerate(test_ids, start=1):
        t0 = time.perf_counter()
        try:
            img_path = find_image_path(image_id)
            rgb = load_rgb(img_path)
            h, w = rgb.shape[:2]

            pts, _human_num = load_annotation_points(image_id)
            gt_den = build_gt_density((h, w), pts, sigma=4.0)
            gt_count = float(gt_den.sum())

            pred_densities = run_all_models(models, rgb, device)
            pred_counts = {m: float(pred_densities[m].sum()) for m in MODEL_ORDER}

            out_id = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(OUT_DIRS["originals"], f"{out_id}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUT_DIRS["gt_overlay"], f"{out_id}.png"), make_overlay(rgb, gt_den))

            for model_name in MODEL_ORDER:
                overlay = make_overlay(rgb, pred_densities[model_name])
                cv2.imwrite(os.path.join(model_overlay_folder[model_name], f"{out_id}.png"), overlay)

            save_density_only(out_id, gt_den, pred_densities, (h, w))

            original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pred_struct = {m: {"count": pred_counts[m], "density_map": pred_densities[m]} for m in MODEL_ORDER}
            generate_vertical_strip(
                image=original_bgr,
                gt_density=gt_den,
                predictions=pred_struct,
                gt_count=gt_count,
                image_id=out_id,
                output_dir=OUT_DIRS["strips"],
            )

            abs_errors = {m: abs(pred_counts[m] - gt_count) for m in MODEL_ORDER}
            pct_errors = {m: (abs_errors[m] / gt_count * 100.0 if gt_count > 1e-9 else 0.0) for m in MODEL_ORDER}
            best_model = min(MODEL_ORDER, key=lambda m: abs_errors[m])
            mean_abs = float(np.mean([abs_errors[m] for m in MODEL_ORDER]))
            bucket = bucket_name(gt_count)
            rows.append(
                ImageResult(
                    image_id=out_id,
                    gt_count=gt_count,
                    pred_counts=pred_counts,
                    abs_errors=abs_errors,
                    pct_errors=pct_errors,
                    best_model=best_model,
                    mean_abs_error=mean_abs,
                    bucket=bucket,
                )
            )
            for m in MODEL_ORDER:
                per_model_abs_errs[m].append(abs_errors[m])

        except Exception as e:
            skipped.append((image_id, str(e)))

        elapsed = time.perf_counter() - t0
        if idx == 1 or idx % max(1, args.progress_every) == 0 or idx == n:
            done_time = time.perf_counter() - start
            rate = done_time / max(idx, 1)
            eta = rate * (n - idx)
            print(f"[{idx}/{n}] processed id={image_id} in {elapsed:.2f}s | ETA {eta:.1f}s")

    write_summary_files(rows)
    try:
        meta_path = os.path.join(OUT_DIRS["summary"], "metadata.csv")
        if os.path.isfile(meta_path):
            build_summary_grids_from_metadata(pd.read_csv(meta_path))
    except Exception:
        pass
    if skipped:
        with open(os.path.join(OUT_DIRS["summary"], "skipped_ids.txt"), "w", encoding="utf-8") as f:
            for sid, err in skipped:
                f.write(f"{sid}: {err}\n")

    print("\n=== Qualitative generation complete ===")
    print(f"Processed successfully: {len(rows)}/{len(test_ids)}")
    print(f"Skipped: {len(skipped)}")
    for m in MODEL_ORDER:
        vals = per_model_abs_errs[m]
        mae = float(np.mean(vals)) if vals else float("nan")
        print(f"{m} MAE on processed images: {mae:.3f}")
    lp_wins = sum(1 for r in rows if r.best_model == "CSRNet-LP")
    print(f"CSRNet-LP wins (lowest AE): {lp_wins}/{len(rows)}")
    print(f"Output root: {OUT_ROOT}")
    print_regeneration_instructions()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.strips_only:
        regenerate_strips_only(test_one=cli_args.test_one)
        print_regeneration_instructions()
    else:
        main(cli_args)
