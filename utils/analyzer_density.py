"""Density post-processing, ERP distortion visualization, latitude bands."""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# phi = 90 - 180 * y / (H-1), y from top (north pole) to bottom (south pole)


def y_to_phi_deg(y: np.ndarray | float, img_h: int) -> np.ndarray | float:
    return 90.0 - 180.0 * y / max(img_h - 1, 1)


def phi_to_y(phi_deg: float, img_h: int) -> float:
    return (90.0 - phi_deg) / 180.0 * (img_h - 1)


def gaussian_smooth_density(density: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth 2D density; preserves approximate integral if sigma small."""
    if sigma <= 0.01:
        return density.astype(np.float32)
    k = max(3, int(sigma * 4) * 2 + 1)
    k = k | 1
    return cv2.GaussianBlur(density.astype(np.float32), (k, k), sigmaX=sigma, sigmaY=sigma)


def erp_distortion_row_weights(img_h: int) -> np.ndarray:
    """
    Relative pixel area stretch vs equator for equirectangular sampling (proportional to 1/cos(phi)).
    Normalized to [0,1] for colormap display (equator=low, poles=high).
    """
    ys = np.arange(img_h, dtype=np.float64)
    phi_rad = np.deg2rad(y_to_phi_deg(ys, img_h))
    cos_phi = np.clip(np.cos(phi_rad), 1e-3, 1.0)
    w = 1.0 / cos_phi
    w = (w - w.min()) / max(w.max() - w.min(), 1e-9)
    return w.astype(np.float32)


def erp_distortion_map(img_h: int, img_w: int) -> np.ndarray:
    """2D map (H,W) same distortion weight per row."""
    col = erp_distortion_row_weights(img_h)
    return np.tile(col[:, np.newaxis], (1, img_w))


def latitude_band_fractions(density: np.ndarray) -> Dict[str, float]:
    """
    Sum density in three latitude bands (|phi| zones).
    Returns fraction of total count per band (keys polar, mid, equatorial).
    """
    h, w = density.shape
    total = float(density.sum()) + 1e-9
    ys = np.arange(h, dtype=np.float64)
    phi = np.abs(y_to_phi_deg(ys, h))
    polar_mask = phi > 60.0
    mid_mask = (phi > 30.0) & (phi <= 60.0)
    eq_mask = phi <= 30.0

    polar = float(density[polar_mask, :].sum()) / total
    mid = float(density[mid_mask, :].sum()) / total
    eq = float(density[eq_mask, :].sum()) / total
    return {"polar": polar, "mid": mid, "equatorial": eq}


def latitude_band_absolutes(density: np.ndarray) -> Dict[str, float]:
    """Absolute predicted count per latitude band (same regions as latitude_band_fractions)."""
    fr = latitude_band_fractions(density)
    total = float(density.sum())
    return {k: fr[k] * total for k in fr}


def overlay_density_on_rgb(
    rgb: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.45,
    cmap_name: str = "viridis",
) -> np.ndarray:
    """Blend viridis-colored density over RGB image."""
    import matplotlib

    d = np.clip(density, 0, None)
    dmax = float(d.max()) if d.size else 1.0
    if dmax < 1e-9:
        dmax = 1.0
    norm = d / dmax
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except Exception:
        import matplotlib.cm as cm

        cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)[:, :, :3]
    rgb_f = rgb.astype(np.float32) / 255.0
    out = (1.0 - alpha) * rgb_f + alpha * rgba.astype(np.float32)
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8)


def draw_latitude_lines(
    rgb: np.ndarray,
    phis: Tuple[float, ...] = (90, 60, 30, 0, -30, -60, -90),
    color: Tuple[int, int, int] = (255, 255, 255),
    dash_period: int = 10,
) -> np.ndarray:
    """Draw dashed horizontal latitude lines on a copy of rgb."""
    out = rgb.copy()
    h, w = out.shape[:2]
    for i, phi in enumerate(phis):
        y = int(round(phi_to_y(float(phi), h)))
        y = max(0, min(h - 1, y))
        for x in range(0, w, dash_period * 2):
            x1 = min(x + dash_period, w - 1)
            cv2.line(out, (x, y), (x1, y), color, 1, cv2.LINE_AA)
        # small label could be added by caller
    return out


def synthetic_demo_densities(
    img_h: int,
    img_w: int,
    target_counts: Dict[str, float],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Gaussian blobs scaled to match target total count per model key."""
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for name, tgt in target_counts.items():
        d = np.zeros((img_h, img_w), dtype=np.float32)
        n_blobs = int(rng.integers(4, 10))
        for _ in range(n_blobs):
            cx = rng.uniform(img_w * 0.1, img_w * 0.9)
            cy = rng.uniform(img_h * 0.1, img_h * 0.9)
            sx = rng.uniform(15, 45)
            sy = rng.uniform(15, 45)
            amp = rng.uniform(0.5, 1.5)
            xs = np.arange(img_w, dtype=np.float32)
            ys = np.arange(img_h, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            d += amp * np.exp(-(((gx - cx) ** 2) / (2 * sx * sx) + ((gy - cy) ** 2) / (2 * sy * sy)))
        s = float(d.sum())
        if s > 1e-9:
            d *= float(tgt) / s
        out[name] = d
    return out
