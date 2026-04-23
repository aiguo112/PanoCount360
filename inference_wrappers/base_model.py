"""Load checkpoints and run forward pass for PanoCount Analyzer."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.model_factory import build_model


@dataclass(frozen=True)
class ModelGuiConfig:
    """GUI-facing model id -> training checkpoint folder name under `checkpoints/`."""

    gui_id: str
    factory_name: str
    default_ckpt_subdir: str


# Order matches user UI: MCNN, CAN, CSRNet, CSRNet-LP
MODEL_CONFIGS: List[ModelGuiConfig] = [
    ModelGuiConfig("MCNN", "mcnn", "mcnn"),
    ModelGuiConfig("CAN", "can", "can"),
    ModelGuiConfig("CSRNet", "csrnet", "csrnet"),
    ModelGuiConfig("CSRNet-LP", "csrnet_pano_latprior_only", "csrnet_pano_latprior_only"),
]


def _default_ckpt_path(project_root: str, subdir: str) -> str:
    return os.path.join(project_root, "checkpoints", subdir, "best_model.pth")


def load_state_into_model(model: nn.Module, ckpt_path: str, device: str) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model_state" in state:
            state_dict = state["model_state"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=False)
    model.eval()


def build_and_load(
    factory_name: str,
    ckpt_path: Optional[str],
    device: str,
) -> Tuple[nn.Module, bool]:
    """
    Build model and load weights if checkpoint exists and is readable.
    Returns (model, demo_mode) where demo_mode True means weights not loaded.
    """
    model = build_model(factory_name).to(device)
    demo = True
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            load_state_into_model(model, ckpt_path, device)
            demo = False
        except Exception:
            demo = True
    return model, demo


def run_forward(model: nn.Module, img_t: torch.Tensor) -> np.ndarray:
    """Returns 2D float density map (H, W)."""
    with torch.no_grad():
        pred = model(img_t)
    x = pred.cpu().numpy()
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D density, got shape {pred.shape}")
    return x.astype(np.float32)
