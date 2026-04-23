"""Right panel: table, matplotlib charts, export."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.matplotlib_theme import apply_scientific_dark_theme
from utils import analyzer_density

MODEL_ORDER = ["MCNN", "CAN", "CSRNet", "CSRNet-LP"]
COLORS = {
    "MCNN": "#E74C3C",
    "CAN": "#E67E22",
    "CSRNet": "#3498DB",
    "CSRNet-LP": "#27AE60",
}
HATCHES = ["///", "xxx", "...", ""]


class RightPanel(QFrame):
    export_json_clicked = Signal()
    export_images_clicked = Signal()
    copy_clipboard_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("class", "glow-panel")
        apply_scientific_dark_theme()

        self._gt: Optional[int] = None
        self._demo_flags: Dict[str, bool] = {}
        self._has_gt = False

        title = QLabel("Results Dashboard")
        title.setStyleSheet("color: #00D4FF; font-size: 16px; font-weight: bold;")

        self._table = QTableWidget(4, 3)
        self._table.setHorizontalHeaderLabels(["Model", "Count", "Status"])
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)

        self._fig_bar = Figure(figsize=(4, 3.2), facecolor="#0A0E1A")
        self._canvas_bar = FigureCanvasQTAgg(self._fig_bar)
        gb1 = QGroupBox("Predicted Count Comparison")
        v1 = QVBoxLayout()
        v1.addWidget(self._canvas_bar)
        gb1.setLayout(v1)

        self._conf_row = QHBoxLayout()
        self._conf_bars: List[QProgressBar] = []
        self._conf_labels: List[QLabel] = []
        gb_conf = QGroupBox("Confidence (relative) — share of max predicted count")
        for i, name in enumerate(MODEL_ORDER):
            col = QVBoxLayout()
            pb = QProgressBar()
            pb.setRange(0, 100)
            pb.setTextVisible(True)
            pb.setFormat("%v%")
            c = COLORS[name]
            pb.setStyleSheet(
                f"QProgressBar {{ border: 2px solid {c}; border-radius: 6px; height: 18px; }}"
                f"QProgressBar::chunk {{ background: {c}; }}"
            )
            lab = QLabel(name)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet("color:#9AA0A6;font-size:10px;")
            col.addWidget(pb)
            col.addWidget(lab)
            self._conf_bars.append(pb)
            self._conf_labels.append(lab)
            self._conf_row.addLayout(col)
        gb_conf.setLayout(self._conf_row)

        self._fig_lat = Figure(figsize=(4, 2.8), facecolor="#0A0E1A")
        self._canvas_lat = FigureCanvasQTAgg(self._fig_lat)
        gb2 = QGroupBox("Count by Latitude Band")
        v2 = QVBoxLayout()
        v2.addWidget(self._canvas_lat)
        gb2.setLayout(v2)

        self._btn_json = QPushButton("Export Results (JSON)")
        self._btn_json.setProperty("class", "export-json")
        self._btn_png = QPushButton("Save Density Maps")
        self._btn_png.setProperty("class", "export-img")
        self._btn_clip = QPushButton("Copy Summary to Clipboard")
        self._btn_clip.setProperty("class", "export-clip")
        self._btn_json.clicked.connect(self.export_json_clicked.emit)
        self._btn_png.clicked.connect(self.export_images_clicked.emit)
        self._btn_clip.clicked.connect(self.copy_clipboard_clicked.emit)

        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(self._table)
        root.addWidget(gb1)
        root.addWidget(gb_conf)
        root.addWidget(gb2)
        root.addWidget(self._btn_json)
        root.addWidget(self._btn_png)
        root.addWidget(self._btn_clip)

        self._counts_anim: Dict[str, float] = {}
        self._timer: Optional[QTimer] = None
        self._last_densities: Dict[str, np.ndarray] = {}

    def _set_table_mode(self, has_gt: bool) -> None:
        self._has_gt = has_gt
        if has_gt:
            self._table.setColumnCount(5)
            self._table.setHorizontalHeaderLabels(["Model", "Count", "Error", "Error%", "Status"])
        else:
            self._table.setColumnCount(3)
            self._table.setHorizontalHeaderLabels(["Model", "Count", "Status"])

    def set_results(
        self,
        counts: Dict[str, float],
        densities: Dict[str, np.ndarray],
        gt: Optional[int] = None,
        demo_flags: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._target_counts = counts.copy()
        self._last_densities = densities
        self._gt = gt
        self._demo_flags = demo_flags or {}
        self._set_table_mode(gt is not None)

        z = {k: 0.0 for k in MODEL_ORDER}
        self._fill_table(z)
        self._plot_bars(counts)
        self._plot_confidence(counts)
        self._plot_latitude(densities)
        self._animate_counts(counts)

    def _model_label(self, name: str) -> str:
        return f"{name} ★" if name == "CSRNet-LP" else name

    def _fill_table(self, counts: Dict[str, float]) -> None:
        from PySide6.QtGui import QBrush, QColor

        if self._has_gt and self._gt is not None and self._gt > 0:
            errors = {k: abs(counts.get(k, 0.0) - float(self._gt)) for k in MODEL_ORDER}
            best_key = min(errors, key=lambda k: errors[k]) if errors else ""
        else:
            present = {k: counts[k] for k in MODEL_ORDER if k in counts and counts[k] > 0}
            best_key = min(present, key=lambda k: present[k]) if present else ""

        hi = QBrush(QColor(0, 255, 136, 40))

        for i, name in enumerate(MODEL_ORDER):

            c = counts.get(name, 0.0)

            self._table.setItem(i, 0, QTableWidgetItem(self._model_label(name)))

            it = QTableWidgetItem(f"{c:.0f}")

            it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self._table.setItem(i, 1, it)



            if self._has_gt and self._gt is not None:

                err = abs(c - float(self._gt))

                pct = (err / float(self._gt) * 100.0) if self._gt > 0 else 0.0

                e_it = QTableWidgetItem(f"{err:.0f}")

                e_it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                p_it = QTableWidgetItem(f"{pct:.1f}%")

                p_it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

                self._table.setItem(i, 2, e_it)

                self._table.setItem(i, 3, p_it)

                st_txt = "✓" if name in counts else "—"

                st = QTableWidgetItem(st_txt)

                self._table.setItem(i, 4, st)

                ncol = 5

            else:

                ok = name in counts and counts[name] >= 0

                st_txt = "✓" if ok else "—"

                st = QTableWidgetItem(st_txt)

                self._table.setItem(i, 2, st)

                ncol = 3



            for j in range(ncol):

                item = self._table.item(i, j)

                if item and name == best_key:

                    item.setBackground(hi)



    def _plot_bars(self, counts: Dict[str, float]) -> None:

        self._fig_bar.clf()

        ax = self._fig_bar.add_subplot(111)

        ax.set_facecolor("#12172A")

        vals = [counts.get(m, 0.0) for m in MODEL_ORDER]

        y = np.arange(len(MODEL_ORDER))

        for i, m in enumerate(MODEL_ORDER):

            v = vals[i]

            c = COLORS[m]

            h = HATCHES[i]

            ax.barh(

                i,

                v,

                color=c,

                alpha=0.85,

                hatch=h,

                edgecolor="#E8EAED",

                linewidth=0.5,

            )

            ax.text(v, i, f"  {v:.0f}", va="center", color="#E8EAED", fontsize=9)

        ax.set_yticks(y)

        ax.set_yticklabels(MODEL_ORDER, color="#E8EAED")

        ax.set_xlabel("Predicted count", color="#9AA0A6")

        ax.set_title("Predicted Count Comparison", color="#E8EAED", fontsize=11)

        ax.tick_params(colors="#9AA0A6")

        ax.set_facecolor("#12172A")

        if self._gt is not None and self._gt > 0:

            ax.axvline(float(self._gt), color="#FFFFFF", linestyle="--", linewidth=1.2, alpha=0.9)

            ax.text(

                float(self._gt),

                len(MODEL_ORDER) - 0.4,

                f"  GT: {self._gt:.0f}",

                color="#FFFFFF",

                fontsize=8,

                va="bottom",

            )

        self._fig_bar.tight_layout()

        self._canvas_bar.draw_idle()



    def _plot_confidence(self, counts: Dict[str, float]) -> None:

        mx = max(counts.values()) if counts else 1.0

        mx = max(mx, 1e-6)

        for i, name in enumerate(MODEL_ORDER):

            v = int(round(100.0 * counts.get(name, 0.0) / mx))

            self._conf_bars[i].setValue(v)

            self._conf_labels[i].setText(f"{name}\n{counts.get(name, 0):.0f}")



    def _plot_latitude(self, densities: Dict[str, np.ndarray]) -> None:

        self._fig_lat.clf()

        ax = self._fig_lat.add_subplot(111)

        models = [m for m in MODEL_ORDER if m in densities]

        if not models:

            ax.text(0.5, 0.5, "No density", ha="center", va="center", color="#9AA0A6", transform=ax.transAxes)

            ax.set_facecolor("#12172A")

            self._fig_lat.tight_layout()

            self._canvas_lat.draw_idle()

            return



        band_labels = ["Polar\n|φ|>60°", "Mid\n30°<|φ|≤60°", "Equatorial\n|φ|≤30°"]

        band_keys = ["polar", "mid", "equatorial"]

        x = np.arange(3)

        n = len(models)

        width = min(0.22, 0.8 / max(n, 1))

        for i, m in enumerate(models):

            ab = analyzer_density.latitude_band_absolutes(densities[m])

            vals = [ab[k] for k in band_keys]

            offset = (i - (n - 1) / 2.0) * width

            ax.bar(

                x + offset,

                vals,

                width,

                label=m,

                color=COLORS[m],

                edgecolor="#E8EAED",

                linewidth=0.4,

            )

        ax.set_xticks(x)

        ax.set_xticklabels(band_labels, fontsize=7, color="#9AA0A6")

        ax.set_ylabel("Predicted count in band", color="#9AA0A6")

        ax.set_facecolor("#12172A")

        ax.tick_params(colors="#9AA0A6")

        ax.legend(loc="upper right", fontsize=7, facecolor="#12172A", edgecolor="#9AA0A6", labelcolor="#E8EAED")

        self._fig_lat.tight_layout()

        self._canvas_lat.draw_idle()



    def _animate_counts(self, target: Dict[str, float]) -> None:

        self._counts_anim = {k: 0.0 for k in target}

        self._target_counts = target.copy()

        steps = 20

        self._anim_step = 0

        if self._timer:

            self._timer.stop()

        self._timer = QTimer(self)

        self._timer.timeout.connect(lambda: self._anim_tick(steps))

        self._timer.start(50)



    def _anim_tick(self, total_steps: int) -> None:

        self._anim_step += 1

        t = min(1.0, self._anim_step / float(total_steps))

        t = 1 - (1 - t) ** 3

        cur = {}

        for k, tgt in self._target_counts.items():

            cur[k] = tgt * t

        for i, name in enumerate(MODEL_ORDER):

            it = self._table.item(i, 1)

            if it:

                it.setText(f"{cur.get(name, 0):.0f}")

        if self._anim_step >= total_steps:

            self._timer.stop()

            self._fill_table(self._target_counts)

            self._plot_bars(self._target_counts)

            self._plot_confidence(self._target_counts)

            self._plot_latitude(self._last_densities)

