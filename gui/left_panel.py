"""Left control panel: upload, models, options, checkpoints."""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from inference_wrappers.base_model import MODEL_CONFIGS, ModelGuiConfig
from inference_wrappers import can_wrapper, csrnet_wrapper, csrnetlp_wrapper, mcnn_wrapper
from utils.image_utils import LoadedImage, format_aspect_label, load_erp_image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_ckpt(cfg: ModelGuiConfig) -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints", cfg.default_ckpt_subdir, "best_model.pth")


class DropZone(QFrame):
    file_chosen = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 200)
        self.setStyleSheet(
            """
            DropZone {
                border: 2px dashed rgba(0, 212, 255, 0.35);
                border-radius: 8px;
                background: rgba(26, 31, 58, 0.45);
            }
            """
        )
        self._lbl = QLabel("Drop ERP Image Here\nor click to browse\nSupported: JPG, PNG, TIFF")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("color: #9AA0A6; font-size: 12px;")
        lay = QVBoxLayout(self)
        lay.addWidget(self._lbl)

    def mousePressEvent(self, event) -> None:
        self.browse()

    def browse(self) -> None:
        settings = QSettings("PanoCrowd", "Analyzer")
        start = settings.value("last_image_dir", PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open panoramic image",
            str(start),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All (*.*)",
        )
        if path:
            settings.setValue("last_image_dir", os.path.dirname(path))
            self.file_chosen.emit(path)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                self.styleSheet()
                + "DropZone { border: 2px solid #00D4FF; background: rgba(0, 212, 255, 0.08); }"
            )

    def dragLeaveEvent(self, event) -> None:
        self._reset_style()

    def dropEvent(self, event: QDropEvent) -> None:
        self._reset_style()
        urls = event.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if p:
                self.file_chosen.emit(p)

    def _reset_style(self) -> None:
        self.setStyleSheet(
            """
            DropZone {
                border: 2px dashed rgba(0, 212, 255, 0.35);
                border-radius: 8px;
                background: rgba(26, 31, 58, 0.45);
            }
            """
        )

    def set_preview(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self._lbl.setText("Drop ERP Image Here\nor click to browse\nSupported: JPG, PNG, TIFF")
            return
        scaled = pixmap.scaled(280, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._lbl.setPixmap(scaled)
        self._lbl.setText("")


class LeftPanel(QScrollArea):
    runRequested = Signal()
    imageLoaded = Signal(object)
    optionsChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._loaded: LoadedImage | None = None
        self._settings = QSettings("PanoCrowd", "Analyzer")

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setSpacing(12)

        title = QLabel("Control Panel")
        title.setStyleSheet("color: #00D4FF; font-size: 16px; font-weight: bold;")
        root.addWidget(title)

        # Upload
        gb_up = QGroupBox("Image Upload")
        v = QVBoxLayout()
        self._drop = DropZone()
        self._drop.file_chosen.connect(self._on_file)
        v.addWidget(self._drop)
        self._btn_load = QPushButton("Load Image")
        self._btn_load.clicked.connect(self._drop.browse)
        v.addWidget(self._btn_load)
        self._info = QLabel("No image loaded.")
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #9AA0A6; font-size: 11px;")
        v.addWidget(self._info)
        gb_up.setLayout(v)
        root.addWidget(gb_up)

        # Models
        gb_m = QGroupBox("Select Models to Compare")
        self._model_checks: Dict[str, QCheckBox] = {}
        colors = {"MCNN": "#FF4444", "CAN": "#FF8800", "CSRNet": "#4488FF", "CSRNet-LP": "#00FF88"}
        tooltips = {
            "MCNN": mcnn_wrapper.TOOLTIP,
            "CAN": can_wrapper.TOOLTIP,
            "CSRNet": csrnet_wrapper.TOOLTIP,
            "CSRNet-LP": csrnetlp_wrapper.TOOLTIP,
        }
        gl = QGridLayout()
        for i, cfg in enumerate(MODEL_CONFIGS):
            cb = QCheckBox(cfg.gui_id)
            cb.setChecked(True)
            cb.setToolTip(tooltips.get(cfg.gui_id, ""))
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {colors.get(cfg.gui_id, '#fff')};")
            gl.addWidget(dot, i, 0)
            gl.addWidget(cb, i, 1)
            self._model_checks[cfg.gui_id] = cb
        gb_m.setLayout(gl)
        root.addWidget(gb_m)

        # Display options
        gb_d = QGroupBox("Display Options")
        fl = QVBoxLayout()
        self._opt_density = QCheckBox("Show Density Maps")
        self._opt_density.setChecked(True)
        self._opt_density.setProperty("class", "toggle")
        self._opt_count = QCheckBox("Show Count Annotations")
        self._opt_count.setChecked(True)
        self._opt_grid = QCheckBox("Show Latitude Grid Overlay")
        self._opt_grid.setChecked(True)
        self._opt_dist = QCheckBox("Show ERP Distortion Heatmap")
        self._opt_dist.setChecked(True)
        self._opt_side = QCheckBox("Side-by-side comparison mode")
        self._opt_side.setChecked(False)
        for w in (self._opt_density, self._opt_count, self._opt_grid, self._opt_dist, self._opt_side):
            w.stateChanged.connect(lambda *_: self.optionsChanged.emit())
            fl.addWidget(w)
        gb_d.setLayout(fl)
        root.addWidget(gb_d)

        # Inference
        gb_i = QGroupBox("Inference Settings")
        form = QFormLayout()
        self._sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self._sigma_slider.setRange(1, 100)
        self._sigma_slider.setValue(40)
        self._sigma_label = QLabel("Density Map Sigma: 4.0")
        self._sigma_slider.valueChanged.connect(self._on_sigma)
        form.addRow(self._sigma_label, self._sigma_slider)

        self._device = QComboBox()
        self._device.addItems(["CUDA (GPU)", "CPU"])
        form.addRow("Device", self._device)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setProperty("class", "primary")
        self._run_btn.clicked.connect(self.runRequested.emit)
        form.addRow(self._run_btn)
        gb_i.setLayout(form)
        root.addWidget(gb_i)

        # Checkpoints
        gb_c = QGroupBox("Model Checkpoints")
        self._ckpt_edits: Dict[str, QLineEdit] = {}
        self._ckpt_status: Dict[str, QLabel] = {}
        cv = QVBoxLayout()
        for cfg in MODEL_CONFIGS:
            row = QHBoxLayout()
            ed = QLineEdit(_default_ckpt(cfg))
            ed.setReadOnly(False)
            btn = QPushButton("…")
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _=False, c=cfg, e=ed: self._browse_ckpt(c, e))
            st = QLabel("—")
            st.setFixedWidth(22)
            ed.textChanged.connect(lambda _t, e=ed, s=st: self._update_ckpt_status(e, s))
            row.addWidget(ed)
            row.addWidget(btn)
            self._ckpt_edits[cfg.gui_id] = ed
            self._ckpt_status[cfg.gui_id] = st
            lab = QLabel(cfg.gui_id)
            lab.setMinimumWidth(90)
            hl = QHBoxLayout()
            hl.addWidget(lab)
            hl.addLayout(row)
            hl.addWidget(st)
            self._update_ckpt_status(ed, st)
            cv.addLayout(hl)
        gb_c.setLayout(cv)
        root.addWidget(gb_c)

        gb_gt = QGroupBox("Ground Truth (Optional)")
        gt_row = QHBoxLayout()
        gt_row.addWidget(QLabel("GT Count:"))
        self._gt_edit = QLineEdit()
        self._gt_edit.setPlaceholderText("Enter actual count if known")
        self._gt_edit.setValidator(QIntValidator(0, 999999999, self))
        gt_row.addWidget(self._gt_edit)
        gb_gt.setLayout(gt_row)
        root.addWidget(gb_gt)

        root.addStretch()
        self.setWidget(inner)

    def _on_sigma(self, v: int) -> None:
        s = v / 10.0
        self._sigma_label.setText(f"Density Map Sigma: {s:.1f}")

    def sigma_value(self) -> float:
        return self._sigma_slider.value() / 10.0

    def _update_ckpt_status(self, edit: QLineEdit, st: QLabel) -> None:
        p = edit.text().strip()
        if p and os.path.isfile(p):
            st.setText("✓")
            st.setStyleSheet("color: #00FF88;")
        else:
            st.setText("✗")
            st.setStyleSheet("color: #FF6B35;")

    def _browse_ckpt(self, cfg: ModelGuiConfig, edit: QLineEdit) -> None:
        start = self._settings.value("last_ckpt_dir", PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Checkpoint for {cfg.gui_id}",
            str(start),
            "PyTorch (*.pth *.pt);;All (*.*)",
        )
        if path:
            self._settings.setValue("last_ckpt_dir", os.path.dirname(path))
            edit.setText(path)

    def _on_file(self, path: str) -> None:
        try:
            loaded = load_erp_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Image load error", str(e))
            return
        self._loaded = loaded
        self.imageLoaded.emit(loaded)
        # preview
        from PySide6.QtGui import QImage

        rgb = np.ascontiguousarray(loaded.rgb)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._drop.set_preview(QPixmap.fromImage(qimg))
        # info
        ar = format_aspect_label(loaded.aspect_ratio)
        warn = ""
        if not loaded.is_valid_erp_ratio:
            warn = "\nNot standard ERP ratio (expected ~2:1)"
        mb = loaded.file_size_bytes / (1024 * 1024)
        self._info.setText(
            f"Resolution: {loaded.width} × {loaded.height}\n"
            f"Aspect Ratio: {ar} {'(Valid ERP)' if loaded.is_valid_erp_ratio else ''}\n"
            f"File Size: {mb:.2f} MB{warn}"
        )
        if not loaded.is_valid_erp_ratio:
            QMessageBox.warning(
                self,
                "Aspect ratio",
                "Image is not ~2:1 equirectangular. Results may be unreliable.",
            )

    def loaded_image(self) -> LoadedImage | None:
        return self._loaded

    def selected_models(self) -> List[str]:
        return [k for k, cb in self._model_checks.items() if cb.isChecked()]

    def checkpoint_paths(self) -> Dict[str, str]:
        return {k: ed.text().strip() for k, ed in self._ckpt_edits.items()}

    def device_choice(self) -> str:
        return self._device.currentText()

    def set_run_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(enabled)

    def display_options(self) -> Dict[str, bool]:
        return {
            "density": self._opt_density.isChecked(),
            "counts": self._opt_count.isChecked(),
            "grid": self._opt_grid.isChecked(),
            "distortion": self._opt_dist.isChecked(),
            "side_by_side": self._opt_side.isChecked(),
        }

    def ground_truth_count(self) -> int | None:
        t = self._gt_edit.text().strip()
        if not t:
            return None
        try:
            return int(t)
        except ValueError:
            return None
