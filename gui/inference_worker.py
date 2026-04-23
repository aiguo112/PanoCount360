"""Background inference for PanoCount Analyzer (PySide6 QThread)."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from PySide6.QtCore import QObject, QThread, Signal

from inference_wrappers.base_model import MODEL_CONFIGS, build_and_load, run_forward
from utils import analyzer_density
from utils.image_utils import TARGET_H, TARGET_W, prepare_model_input_tensor

# Project root on sys.path (run from main.py at root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelRunSpec:
    gui_id: str
    factory_name: str
    ckpt_path: str
    selected: bool


def default_checkpoint_path(factory_subdir: str) -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints", factory_subdir, "best_model.pth")


class InferenceWorker(QThread):
    log_line = Signal(str)
    progress = Signal(int, str)
    """Emitted after each model finishes: gui_id, predicted count, display density (numpy)."""
    partial_result = Signal(str, float, object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._specs: List[ModelRunSpec] = []
        self._rgb_resized: Optional[np.ndarray] = None
        self._device_str: str = "cpu"
        self._sigma: float = 4.0

    def configure(
        self,
        rgb_resized: np.ndarray,
        specs: List[ModelRunSpec],
        device_str: str,
        sigma: float,
    ) -> None:
        self._rgb_resized = rgb_resized.copy()
        self._specs = specs
        self._device_str = device_str
        self._sigma = sigma

    def run(self) -> None:
        if self._rgb_resized is None:
            self.failed.emit("No image loaded.")
            return
        t0 = time.perf_counter()
        try:
            device = torch.device(self._device_str)
            _, img_t = prepare_model_input_tensor(self._rgb_resized, str(device))

            demo_targets = {
                "MCNN": 110.0 + np.random.randn() * 5,
                "CAN": 125.0 + np.random.randn() * 5,
                "CSRNet": 82.0 + np.random.randn() * 5,
                "CSRNet-LP": 68.0 + np.random.randn() * 5,
            }

            densities_raw: Dict[str, np.ndarray] = {}
            densities_display: Dict[str, np.ndarray] = {}
            counts: Dict[str, float] = {}
            demo_flags: Dict[str, bool] = {}
            n_sel = max(1, sum(1 for s in self._specs if s.selected))
            done = 0

            for spec in self._specs:
                if not spec.selected:
                    continue
                self.progress.emit(
                    int(100 * done / max(n_sel, 1)),
                    f"Running {spec.gui_id} inference... ({done + 1}/{n_sel} models)",
                )
                model, is_demo = build_and_load(spec.factory_name, spec.ckpt_path or None, str(device))
                demo_flags[spec.gui_id] = is_demo

                if is_demo:
                    self.log_line.emit(f"[DEMO] {spec.gui_id}: synthetic density (no valid checkpoint)")
                    syn = analyzer_density.synthetic_demo_densities(
                        TARGET_H,
                        TARGET_W,
                        {spec.gui_id: max(10.0, demo_targets.get(spec.gui_id, 80.0))},
                        seed=int(time.time() * 1000) % (2**31),
                    )
                    d_raw = syn[spec.gui_id]
                else:
                    d_raw = run_forward(model, img_t)

                counts[spec.gui_id] = float(d_raw.sum())
                d_disp = analyzer_density.gaussian_smooth_density(d_raw, self._sigma)
                densities_raw[spec.gui_id] = d_raw
                densities_display[spec.gui_id] = d_disp
                done += 1
                self.progress.emit(
                    int(100 * done / n_sel),
                    f"Finished {spec.gui_id} ({done}/{n_sel} models)",
                )
                self.partial_result.emit(spec.gui_id, counts[spec.gui_id], d_disp.copy())
                self.log_line.emit(
                    f"{spec.gui_id} inference complete: {counts[spec.gui_id]:.1f} persons"
                )

            self.progress.emit(100, "Done")
            payload = {
                "densities_raw": densities_raw,
                "densities_display": densities_display,
                "counts": counts,
                "demo_flags": demo_flags,
                "demo_mode_any": any(demo_flags.values()),
                "elapsed_sec": time.perf_counter() - t0,
                "device": str(device),
            }
            self.finished_ok.emit(payload)
        except Exception as e:
            import traceback

            self.failed.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def resolve_device(user_choice: str) -> str:
    if user_choice.lower().startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def gpu_display_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA"
    return ""
