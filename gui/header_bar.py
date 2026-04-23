"""Top header bar: logo, title, badges."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QWidget


class LogoWidget(QWidget):
    """Simple circular panoramic-style icon."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(44, 44)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0, QColor("#00D4FF"))
        grad.setColorAt(1, QColor("#7B2FFF"))
        p.setBrush(grad)
        p.setPen(QPen(QColor("#00D4FF"), 2))
        r = min(self.width(), self.height()) - 4
        p.drawEllipse(2, 2, r, r)
        p.setPen(QPen(QColor("#0A0E1A"), 1))
        for i in range(5):
            x0 = 10 + i * 5
            p.drawArc(x0, 14, 24, 20, 30 * 16, 120 * 16)


class HeaderBar(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("headerBar")
        self._demo_badge = QLabel("")
        self._demo_badge.setProperty("class", "badge-orange")
        self._demo_badge.setVisible(False)
        self._setup_ui()

    def _setup_ui(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 8, 16, 8)

        left = QHBoxLayout()
        left.addWidget(LogoWidget())
        titles = QVBoxLayout()
        t1 = QLabel("PanoCount Analyzer")
        t1.setStyleSheet("color: #00D4FF; font-size: 20px; font-weight: bold;")
        t2 = QLabel("360° Panoramic Crowd Counting Analysis System")
        t2.setStyleSheet("color: #9AA0A6; font-size: 11px;")
        titles.addWidget(t1)
        titles.addWidget(t2)
        left.addLayout(titles)
        left.addWidget(self._demo_badge)
        lay.addLayout(left)
        lay.addStretch()

        self._badge_torch = QLabel("PyTorch 2.x")
        self._badge_torch.setProperty("class", "badge")
        self._badge_device = QLabel("CPU Mode")
        self._badge_device.setProperty("class", "badge-orange")
        self._badge_ver = QLabel("v1.0")
        self._badge_ver.setProperty("class", "badge-grey")

        right = QHBoxLayout()
        right.setSpacing(8)
        right.addWidget(self._badge_torch)
        right.addWidget(self._badge_device)
        right.addWidget(self._badge_ver)
        lay.addLayout(right)

    def set_device_badge(self, cuda: bool, gpu_name: str = "") -> None:
        if cuda:
            self._badge_device.setText("CUDA Enabled" if not gpu_name else gpu_name[:24])
            self._badge_device.setProperty("class", "badge-green")
        else:
            self._badge_device.setText("CPU Mode")
            self._badge_device.setProperty("class", "badge-orange")
        self._badge_device.style().unpolish(self._badge_device)
        self._badge_device.style().polish(self._badge_device)

    def set_demo_mode(self, on: bool) -> None:
        self._demo_badge.setVisible(on)
        if on:
            self._demo_badge.setText("DEMO MODE")
            self._demo_badge.setProperty("class", "badge-orange")

    def set_pytorch_label(self, text: str) -> None:
        self._badge_torch.setText(text)
