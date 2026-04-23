"""Main window: three-column layout + header + status."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui.center_panel import CenterPanel
from gui.header_bar import HeaderBar
from gui.inference_worker import InferenceWorker, ModelRunSpec, gpu_display_name, resolve_device
from gui.left_panel import LeftPanel
from gui.right_panel import RightPanel
from gui.status_bar import StatusDock
from gui.styles import APP_STYLESHEET, HEADER_STYLE
from inference_wrappers.base_model import MODEL_CONFIGS
from utils.export_utils import build_summary_dict, format_clipboard_summary, save_density_maps_png, save_json
from utils.image_utils import LoadedImage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PanoCount Analyzer — 360° Crowd Counting System")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 950)

        self._settings = QSettings("PanoCrowd", "Analyzer")
        self._worker: Optional[InferenceWorker] = None
        self._last_payload: Optional[dict] = None
        self._loaded: Optional[LoadedImage] = None
        self._running_counts: dict = {}
        self._running_densities: dict = {}

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._header = HeaderBar()
        self._header.setStyleSheet(HEADER_STYLE)

        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._left = LeftPanel()
        self._center = CenterPanel()
        self._right = RightPanel()

        self._split.addWidget(self._left)
        self._split.addWidget(self._center)
        self._split.addWidget(self._right)
        self._split.setStretchFactor(0, 25)
        self._split.setStretchFactor(1, 50)
        self._split.setStretchFactor(2, 25)

        self._status = StatusDock()

        layout.addWidget(self._header)
        layout.addWidget(self._split, stretch=1)
        layout.addWidget(self._status)

        self.setStyleSheet(APP_STYLESHEET)

        self._left.imageLoaded.connect(self._on_image_loaded)
        self._left.runRequested.connect(self._on_run)
        self._left.optionsChanged.connect(self._on_options_changed)
        self._right.export_json_clicked.connect(self._on_export_json)
        self._right.export_images_clicked.connect(self._on_export_png)
        self._right.copy_clipboard_clicked.connect(self._on_copy)

        self._refresh_header_device()

    def _on_options_changed(self) -> None:
        self._center.set_display_options(self._left.display_options())
        if self._last_payload:
            self._center.set_results(
                self._last_payload["densities_display"],
                self._last_payload["counts"],
            )
            self._right.set_results(
                self._last_payload["counts"],
                self._last_payload["densities_raw"],
                gt=self._left.ground_truth_count(),
                demo_flags=self._last_payload.get("demo_flags", {}),
            )

    def _refresh_header_device(self) -> None:
        cuda = torch.cuda.is_available()
        self._header.set_device_badge(cuda, gpu_display_name())
        ver = torch.__version__.split("+")[0]
        self._header.set_pytorch_label(f"PyTorch {ver.split('.')[0]}.{ver.split('.')[1]}.x")

    def _on_image_loaded(self, loaded: LoadedImage) -> None:
        self._loaded = loaded
        self._center.set_image(loaded.resized_rgb)
        self._status.append_log(f"Loading image: {os.path.basename(loaded.path)}")
        self._status.append_log(
            f"Resolution: {loaded.width}×{loaded.height} — {'Valid ERP ✓' if loaded.is_valid_erp_ratio else 'Non-standard ratio'}"
        )

    def _on_run(self) -> None:
        if self._loaded is None:
            QMessageBox.warning(self, "No image", "Please load an ERP image first.")
            return
        sel = self._left.selected_models()
        if not sel:
            QMessageBox.warning(self, "No models", "Select at least one model.")
            return

        dev = resolve_device(self._left.device_choice())
        sigma = self._left.sigma_value()
        ckpts = self._left.checkpoint_paths()

        specs: list[ModelRunSpec] = []
        for cfg in MODEL_CONFIGS:
            if cfg.gui_id not in sel:
                continue
            specs.append(
                ModelRunSpec(
                    gui_id=cfg.gui_id,
                    factory_name=cfg.factory_name,
                    ckpt_path=ckpts.get(cfg.gui_id, ""),
                    selected=True,
                )
            )

        self._status.set_state("processing")
        self._status.set_progress(0, "Starting…")
        self._status.append_log("Starting inference…")
        self._running_counts = {}
        self._running_densities = {}

        self._worker = InferenceWorker(self)
        self._worker.configure(
            self._loaded.resized_rgb.copy(),
            specs,
            dev,
            sigma,
        )
        self._worker.log_line.connect(self._status.append_log)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.partial_result.connect(self._on_worker_partial)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_fail)
        self._left.set_run_enabled(False)
        self._worker.finished.connect(lambda: self._left.set_run_enabled(True))
        self._worker.start()

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self._status.set_progress(pct, msg)

    def _on_worker_partial(self, gui_id: str, count: float, dens: object) -> None:
        if not isinstance(dens, np.ndarray):
            return
        self._running_counts[gui_id] = float(count)
        self._running_densities[gui_id] = dens
        gt = self._left.ground_truth_count()
        self._center.set_results(self._running_densities, self._running_counts)
        self._right.set_results(
            self._running_counts,
            self._running_densities,
            gt=gt,
            demo_flags={},
        )

    def _on_worker_done(self, payload: object) -> None:
        assert isinstance(payload, dict)
        self._last_payload = payload
        demo_any = payload.get("demo_mode_any", False)
        self._header.set_demo_mode(demo_any)
        if demo_any:
            self._status.append_log("[DEMO] Using synthetic density maps where checkpoints missing")

        counts = payload["counts"]
        dens = payload["densities_display"]
        gt = self._left.ground_truth_count()
        self._center.set_display_options(self._left.display_options())
        self._center.set_results(dens, counts)
        self._right.set_results(
            counts,
            payload["densities_raw"],
            gt=gt,
            demo_flags=payload.get("demo_flags", {}),
        )

        elapsed = payload.get("elapsed_sec", 0.0)
        dev = payload.get("device", "cpu")
        gpu = gpu_display_name() if dev == "cuda" else "—"
        self._status.set_timing_line(f"Total inference: {elapsed:.2f}s | GPU: {gpu}")
        self._status.set_state("complete")
        self._status.set_progress(100, "Complete")
        self._status.append_log(f"Analysis complete in {elapsed:.2f}s")

    def _on_worker_fail(self, msg: str) -> None:
        self._status.set_state("error")
        self._status.append_log("ERROR: " + msg[:500])
        QMessageBox.critical(self, "Inference failed", msg[:2000])

    def _on_export_json(self) -> None:
        if not self._last_payload:
            QMessageBox.information(self, "Nothing to export", "Run analysis first.")
            return
        start = self._settings.value("last_export_dir", PROJECT_ROOT)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", str(start), "JSON (*.json)"
        )
        if not path:
            return
        self._settings.setValue("last_export_dir", os.path.dirname(path))
        counts = self._last_payload["counts"]
        data = build_summary_dict(
            self._loaded.path if self._loaded else "",
            counts,
            {"sigma": self._left.sigma_value(), "device": self._left.device_choice()},
            self._last_payload.get("demo_mode_any", False),
            gpu_display_name() if torch.cuda.is_available() else None,
            float(self._last_payload.get("elapsed_sec", 0.0)),
        )
        save_json(path, data)
        self._status.append_log(f"Exported JSON: {path}")

    def _on_export_png(self) -> None:
        if not self._last_payload:
            QMessageBox.information(self, "Nothing to export", "Run analysis first.")
            return
        start = self._settings.value("last_export_dir", PROJECT_ROOT)
        d = QFileDialog.getExistingDirectory(self, "Output folder", str(start))
        if not d:
            return
        self._settings.setValue("last_export_dir", d)
        dens = self._last_payload["densities_raw"]
        paths = save_density_maps_png(d, dens, prefix="density")
        self._status.append_log(f"Saved {len(paths)} density map(s) to {d}")

    def _on_copy(self) -> None:
        if not self._last_payload:
            return
        from PySide6.QtWidgets import QApplication

        txt = format_clipboard_summary(
            self._last_payload["counts"],
            self._loaded.path if self._loaded else "",
            self._last_payload.get("demo_mode_any", False),
        )
        QApplication.clipboard().setText(txt)
        self._status.append_log("Summary copied to clipboard.")
