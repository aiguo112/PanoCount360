"""Export results to JSON / images / clipboard text."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def build_summary_dict(
    image_path: str,
    model_counts: Dict[str, float],
    settings: Dict[str, Any],
    demo_mode: bool,
    gpu_name: Optional[str],
    inference_seconds: float,
) -> Dict[str, Any]:
    return {
        "app": "PanoCount Analyzer",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": image_path,
        "demo_mode": demo_mode,
        "device": settings.get("device", ""),
        "gpu_name": gpu_name,
        "inference_seconds": inference_seconds,
        "density_sigma": settings.get("sigma", 4.0),
        "predicted_counts": {k: float(v) for k, v in model_counts.items()},
        "settings": settings,
    }


def save_json(path: str, data: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_density_maps_png(
    out_dir: str,
    densities: Dict[str, np.ndarray],
    prefix: str = "density",
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for name, d in densities.items():
        d2 = np.asarray(d, dtype=np.float32)
        d2 = np.clip(d2 / (d2.max() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
        # false color
        heat = cv2.applyColorMap(d2, cv2.COLORMAP_VIRIDIS)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        fp = os.path.join(out_dir, f"{prefix}_{safe}.png")
        cv2.imwrite(fp, heat)
        saved.append(fp)
    return saved


def format_clipboard_summary(
    model_counts: Dict[str, float],
    image_path: str,
    demo_mode: bool,
) -> str:
    lines = [
        "PanoCount Analyzer — Summary",
        f"Image: {image_path}",
        f"Mode: {'DEMO' if demo_mode else 'Checkpoint'}",
        "",
    ]
    for k in sorted(model_counts.keys()):
        lines.append(f"  {k}: {model_counts[k]:.1f}")
    return "\n".join(lines)
