"""Center: tabbed image viewer with matplotlib overlays."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from utils import analyzer_density

TAB_LABELS = ["Original", "MCNN", "CAN", "CSRNet", "CSRNet-LP"]
MODEL_KEYS = ["MCNN", "CAN", "CSRNet", "CSRNet-LP"]
CHART_LAT_PHIS = (60.0, 30.0, 0.0, -30.0, -60.0)


class ClickableThumb(QLabel):
    """Thumbnail that emits clicked(index) and supports hover/selection border."""

    clicked = Signal(int)

    def __init__(self, index: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self._hover = False
        self.setFixedSize(140, 78)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_style()

    def set_selected(self, on: bool) -> None:
        self._selected = on
        self._apply_style()

    def enterEvent(self, event) -> None:
        self._hover = True
        self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self._apply_style()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mousePressEvent(event)

    def _apply_style(self) -> None:
        if self._selected:
            border = "2px solid #00D4FF"
        elif self._hover:
            border = "2px solid rgba(0, 212, 255, 0.75)"
        else:
            border = "1px solid rgba(0, 212, 255, 0.3)"
        self.setStyleSheet(
            f"QLabel {{ border: {border}; border-radius: 4px; background: rgba(26, 31, 58, 0.45); }}"
        )


class CenterPanel(QFrame):
    thumbnailActivated = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("class", "glow-panel")
        self._rgb: Optional[np.ndarray] = None
        self._counts: Dict[str, float] = {}
        self._densities: Dict[str, np.ndarray] = {}
        self._opts: Dict[str, bool] = {
            "density": True,
            "counts": True,
            "grid": True,
            "distortion": True,
        }
        self._overlay_alpha = 0.6
        self._cmap_name = "viridis"
        self._show_lat_chart_lines = False
        self._scroll_cid: Optional[int] = None

        title = QLabel("Image Analysis View")
        title.setStyleSheet("color: #00D4FF; font-size: 16px; font-weight: bold;")

        self._tab_bar = QTabBar()
        for name in TAB_LABELS:
            self._tab_bar.addTab(name)
        self._tab_bar.currentChanged.connect(self._on_tab_changed)

        self._fig = Figure(figsize=(9, 5.5), layout="tight")
        self._fig.patch.set_facecolor("#0A0E1A")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._btn_fit = QPushButton("Fit")
        self._btn_fit.setFixedWidth(64)
        self._btn_fit.clicked.connect(self._on_fit)

        self._btn_home = QPushButton("Home")
        self._btn_home.setToolTip("Reset view (zoom/pan)")
        self._btn_home.clicked.connect(self._toolbar.home)

        self._btn_back = QPushButton("Back")
        self._btn_back.setToolTip("Previous tab")
        self._btn_back.clicked.connect(self._tab_prev)

        self._btn_fwd = QPushButton("Forward")
        self._btn_fwd.setToolTip("Next tab")
        self._btn_fwd.clicked.connect(self._tab_next)

        self._btn_pan = QPushButton("Pan")
        self._btn_pan.setToolTip("Pan mode (drag)")
        self._btn_pan.clicked.connect(self._toolbar.pan)

        self._btn_zoom = QPushButton("Zoom")
        self._btn_zoom.setToolTip("Zoom mode (drag box); mouse wheel also zooms")
        self._btn_zoom.clicked.connect(self._toolbar.zoom)

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.setToolTip("Overlay opacity and colormap")
        self._btn_settings.clicked.connect(self._open_settings)

        self._btn_chart = QPushButton("Chart lines")
        self._btn_chart.setToolTip("Toggle latitude reference lines (0°, ±30°, ±60°)")
        self._btn_chart.setCheckable(True)
        self._btn_chart.toggled.connect(self._on_chart_lines_toggled)

        self._btn_save = QPushButton("Save PNG")
        self._btn_save.setToolTip("Save current figure view")
        self._btn_save.clicked.connect(self._save_png)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("View:"))
        zoom_row.addWidget(self._toolbar)
        zoom_row.addWidget(self._btn_home)
        zoom_row.addWidget(self._btn_back)
        zoom_row.addWidget(self._btn_fwd)
        zoom_row.addWidget(self._btn_pan)
        zoom_row.addWidget(self._btn_zoom)
        zoom_row.addWidget(self._btn_settings)
        zoom_row.addWidget(self._btn_chart)
        zoom_row.addWidget(self._btn_save)
        zoom_row.addWidget(self._btn_fit)
        zoom_row.addStretch()

        self._thumb_row = QHBoxLayout()
        self._thumbs: list[ClickableThumb] = []
        for i in range(4):
            lab = ClickableThumb(i)
            lab.clicked.connect(self._on_thumb_clicked)
            self._thumbs.append(lab)
            self._thumb_row.addWidget(lab)

        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(self._tab_bar)
        root.addWidget(self._canvas, stretch=1)
        root.addLayout(zoom_row)
        root.addWidget(QLabel("Model density previews (click)"))
        thumb_w = QWidget()
        thumb_w.setLayout(self._thumb_row)
        root.addWidget(thumb_w)

        self._scroll_cid = self._canvas.mpl_connect("scroll_event", self._on_fig_scroll)

    def _on_tab_changed(self, index: int) -> None:
        self._sync_thumb_selection()
        self._redraw()

    def _sync_thumb_selection(self) -> None:
        idx = self._tab_bar.currentIndex()
        for i, lab in enumerate(self._thumbs):
            lab.set_selected(idx == i + 1)

    def _on_thumb_clicked(self, thumb_index: int) -> None:
        self._tab_bar.setCurrentIndex(thumb_index + 1)
        self.thumbnailActivated.emit(thumb_index)

    def _tab_prev(self) -> None:
        i = self._tab_bar.currentIndex()
        if i > 0:
            self._tab_bar.setCurrentIndex(i - 1)

    def _tab_next(self) -> None:
        i = self._tab_bar.currentIndex()
        if i < self._tab_bar.count() - 1:
            self._tab_bar.setCurrentIndex(i + 1)

    def _on_fit(self) -> None:
        self._toolbar.home()

    def _on_chart_lines_toggled(self, on: bool) -> None:
        self._show_lat_chart_lines = on
        self._redraw()

    def _open_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("View settings")
        dlg.setMinimumWidth(320)
        lay = QVBoxLayout(dlg)
        form = QFormLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(round(self._overlay_alpha * 100)))
        combo = QComboBox()
        for name in ("viridis", "plasma", "hot", "jet"):
            combo.addItem(name)
        idx = combo.findText(self._cmap_name)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        form.addRow("Overlay opacity (%):", slider)
        form.addRow("Colormap:", combo)
        lay.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._overlay_alpha = max(0.05, min(1.0, slider.value() / 100.0))
        self._cmap_name = combo.currentText()
        self._redraw()

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save view as PNG",
            "",
            "PNG (*.png);;All (*.*)",
        )
        if not path:
            return
        self._fig.savefig(path, facecolor="#0A0E1A", bbox_inches="tight", dpi=150)

    def _on_fig_scroll(self, event) -> None:
        if self._rgb is None or not self._fig.axes:
            return
        ax = self._fig.axes[0]
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        step = getattr(event, "step", None)
        if step is not None:
            base_scale = 1.15 if step < 0 else 1.0 / 1.15
        else:
            base_scale = 1.15 if event.button == "down" else 1.0 / 1.15
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        w = (cur_xlim[1] - cur_xlim[0]) * base_scale
        h = (cur_ylim[1] - cur_ylim[0]) * base_scale
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - w * (1 - relx), xdata + w * relx])
        ax.set_ylim([ydata - h * (1 - rely), ydata + h * rely])
        self._canvas.draw_idle()

    def current_tab_index(self) -> int:
        return self._tab_bar.currentIndex()

    def set_display_options(self, opts: Dict[str, bool]) -> None:
        self._opts = opts
        self._redraw()

    def set_image(self, rgb: np.ndarray) -> None:
        self._rgb = rgb
        self._redraw()

    def set_results(
        self,
        densities: Dict[str, np.ndarray],
        counts: Dict[str, float],
    ) -> None:
        self._densities = densities
        self._counts = counts
        self._update_thumbnails()
        self._sync_thumb_selection()
        self._redraw()

    def _update_thumbnails(self) -> None:
        for i, k in enumerate(MODEL_KEYS):
            if k not in self._densities:
                self._thumbs[i].clear()
                continue
            d = self._densities[k]
            d_norm = (d / (d.max() + 1e-9) * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(d_norm, cv2.COLORMAP_VIRIDIS)
            heat = np.ascontiguousarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
            h, w, _ = heat.shape
            qimg = QImage(heat.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self._thumbs[i].setPixmap(
                pix.scaled(
                    self._thumbs[i].size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _prepare_base(self) -> np.ndarray:
        assert self._rgb is not None
        base = self._rgb.copy()
        if self._opts.get("grid", True):
            base = analyzer_density.draw_latitude_lines(base)
        if self._show_lat_chart_lines:
            base = analyzer_density.draw_latitude_lines(base, phis=CHART_LAT_PHIS)
        return base

    def _redraw(self) -> None:
        self._fig.clf()
        if self._rgb is None:
            self._canvas.draw_idle()
            return
        idx = self._tab_bar.currentIndex()
        h, w = self._rgb.shape[:2]

        gs = GridSpec(1, 3, width_ratios=[1, 0.04, 0.04], wspace=0.2)
        ax = self._fig.add_subplot(gs[0, 0])
        cax = self._fig.add_subplot(gs[0, 1])
        dax = self._fig.add_subplot(gs[0, 2])

        base = self._prepare_base()

        show_den = self._opts.get("density", True)
        show_dist = self._opts.get("distortion", True)

        if idx == 0:
            ax.imshow(base)
            ax.set_axis_off()
            cax.axis("off")
            dax.axis("off")
            self._fig.tight_layout()
            self._canvas.draw_idle()
            return

        key = TAB_LABELS[idx]
        den = self._densities.get(key)
        if den is None or not show_den:
            ax.imshow(base)
            cax.axis("off")
        else:
            blend = analyzer_density.overlay_density_on_rgb(
                base, den, alpha=self._overlay_alpha, cmap_name=self._cmap_name
            )
            ax.imshow(blend)
            norm = mpl.colors.Normalize(vmin=0, vmax=float(den.max() + 1e-9))
            sm = mpl.cm.ScalarMappable(cmap=self._cmap_name, norm=norm)
            sm.set_array([])
            self._fig.colorbar(sm, cax=cax, orientation="vertical")
            cax.set_ylabel("Density", color="#9AA0A6", fontsize=8)
            cax.tick_params(colors="#9AA0A6")

        if show_dist:
            dist = analyzer_density.erp_distortion_map(h, w)
            dax.imshow(dist, aspect="auto", cmap="RdYlGn_r")
            dax.set_xticks([])
            dax.set_yticks([])
            dax.set_ylabel("Lat. dist.", color="#9AA0A6", fontsize=8)

        ax.set_axis_off()

        if self._opts.get("counts", True) and key in self._counts:
            ax.text(
                0.98,
                0.98,
                f"Predicted: {self._counts[key]:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#00D4FF",
                fontsize=14,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="#0A0E1A",
                    edgecolor="#00D4FF",
                    alpha=0.88,
                ),
            )

        self._fig.tight_layout()
        self._canvas.draw_idle()
