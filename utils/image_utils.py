"""ERP / panoramic image helpers for the analyzer GUI."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

# Training default: height x width
TARGET_H, TARGET_W = 512, 1024

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class LoadedImage:
    """Original path, RGB uint8, and optional resized tensor inputs."""

    path: str
    rgb: np.ndarray  # H, W, 3 uint8 original size
    resized_rgb: np.ndarray  # TARGET_H, TARGET_W, 3 uint8
    width: int
    height: int
    file_size_bytes: int
    aspect_ratio: float
    is_valid_erp_ratio: bool


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def prepare_model_input_tensor(
    rgb_uint8: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, object]:
    """Resize to (512,1024), ImageNet-normalize, return raw resized RGB and tensor NCHW."""
    import torch

    if rgb_uint8.shape[0] != TARGET_H or rgb_uint8.shape[1] != TARGET_W:
        rgb_uint8 = cv2.resize(rgb_uint8, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    img = rgb_uint8.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(chw).float().unsqueeze(0).to(device)
    return rgb_uint8, t


def load_erp_image(path: str) -> LoadedImage:
    bgr = load_image_bgr(path)
    rgb = bgr_to_rgb(bgr)
    h, w = rgb.shape[:2]
    file_size = os.path.getsize(path)
    ar = w / float(h) if h else 0.0
    # Standard full ERP often 2:1 (width = 2*height)
    is_2_1 = abs(ar - 2.0) < 0.02
    resized = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return LoadedImage(
        path=path,
        rgb=rgb,
        resized_rgb=resized,
        width=w,
        height=h,
        file_size_bytes=file_size,
        aspect_ratio=ar,
        is_valid_erp_ratio=is_2_1,
    )


def format_aspect_label(ratio: float) -> str:
    if ratio <= 0:
        return "—"
    # Snap to common ratios
    for num, den in [(2, 1), (16, 9), (4, 3)]:
        t = num / den
        if abs(ratio - t) < 0.02:
            return f"{num}:{den}"
    return f"{ratio:.2f}:1"
